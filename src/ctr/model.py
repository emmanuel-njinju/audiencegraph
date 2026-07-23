"""
Module 4 - CTR / Conversion Optimizer (Regression at scale). Lotame "Optimizer".

Predict click probability for every ad impression - the model that ranks and
prices inventory in a performance-optimization engine. Trained on the full
impression log (~1e6+ rows here, the same code path handles 1e9 on EMR) with
the hashing trick so high-cardinality categoricals (campaign x creative x
publisher x category) never require a fitted vocabulary and the pipeline stays
online-trainable.

Why hashed logistic regression
------------------------------
At ad-serving scale and latency, a hashed linear model is the workhorse: O(1)
feature mapping, no vocab to store or skew on, cheap to score in the auction,
and trivially warm-startable. It is the honest first thing to ship before
reaching for anything heavier - and the JD asks precisely for that "select the
approach and say why" judgment.

Run:  PYTHONPATH=. .venv/bin/python src/ctr/model.py --data data/synthetic
"""
from __future__ import annotations

import argparse
import json
import math
import os

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import FeatureHasher
from pyspark.sql import functions as F

from src.common.spark import get_spark
from src.common.consent import allowed_person_ids

NUM_HASH = 1 << 18   # 262,144-dim hashed feature space


def run(data: str) -> dict:
    spark = get_spark("ctr")
    persons = spark.read.parquet(f"{data}/persons.parquet")
    allowed = allowed_person_ids(persons)                    # consent gate
    devices = spark.read.parquet(f"{data}/devices.parquet").join(allowed, "person_id")
    events = (spark.read.parquet(f"{data}/events.parquet")
                   .join(devices.select("device_id", "device_type"), "device_id"))

    cols = ["content_cat", "campaign", "creative", "device_type", "position", "bid"]
    hasher = FeatureHasher(inputCols=cols, outputCol="features", numFeatures=NUM_HASH,
                           categoricalCols=["content_cat", "campaign", "creative", "device_type"])
    df = hasher.transform(events).select("features", F.col("click").cast("double").alias("label")).cache()
    train, test = df.randomSplit([0.7, 0.3], seed=42)

    model = LogisticRegression(featuresCol="features", labelCol="label",
                               maxIter=50, regParam=1e-4, elasticNetParam=0.0).fit(train)
    pred = model.transform(test).cache()

    auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(pred)
    logloss = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                probabilityCol="probability", metricName="logLoss").evaluate(pred)

    # Baseline: predict the constant base rate -> its logloss is the label entropy.
    base = float(train.agg(F.avg("label")).first()[0])
    baseline_logloss = -(base * math.log(base) + (1 - base) * math.log(1 - base))

    metrics = {
        "n_impressions": int(df.count()),
        "hashed_dim": NUM_HASH,
        "base_ctr": round(base, 4),
        "auc": round(float(auc), 4),
        "logloss": round(float(logloss), 4),
        "baseline_logloss": round(baseline_logloss, 4),
        "logloss_reduction_vs_baseline": round((baseline_logloss - float(logloss)) / baseline_logloss, 4),
    }
    spark.stop()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="reports/ctr_metrics.json")
    args = ap.parse_args()
    m = run(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(m, open(args.out, "w"), indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
