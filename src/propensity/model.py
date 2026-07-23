"""
Module 3 - Behavioral Propensity (Classification).

Predict whether a consented person will convert, from observed content affinity
plus first-party attributes. Two models are compared head-to-head so the choice
is argued from evidence:

  - Logistic Regression: linear, calibrated, interpretable - the deployable
    baseline whose coefficients a stakeholder can read.
  - Gradient-Boosted Trees: captures the interaction + nonlinearity in the true
    propensity surface; expected to win on AUC, at the cost of interpretability.

The gap between them is the honest value of the nonlinear model - reported, not
assumed.

Run:  PYTHONPATH=. .venv/bin/python src/propensity/model.py --data data/synthetic
"""
from __future__ import annotations

import argparse
import json
import os

from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F

from src.common.spark import get_spark
from src.common.consent import allowed_person_ids

CATS = [f"cat_{i:02d}" for i in range(12)]


def run(data: str) -> dict:
    spark = get_spark("propensity")
    persons = spark.read.parquet(f"{data}/persons.parquet")
    allowed = allowed_person_ids(persons)                    # consent gate
    devices = spark.read.parquet(f"{data}/devices.parquet").join(allowed, "person_id")
    events = spark.read.parquet(f"{data}/events.parquet")

    # Observed content affinity (behavioral features), per person.
    ev = events.join(devices.select("device_id", "person_id"), "device_id")
    aff = (ev.groupBy("person_id", "content_cat")
             .agg((3 * F.sum("click") + F.count("*")).cast("double").alias("w"))
             .groupBy("person_id").pivot("content_cat", CATS).agg(F.first("w")).na.fill(0.0))

    # First-party attributes + label.
    attrs = persons.select("person_id", "age", "income_z", "recency_days", "converted_true") \
                   .join(allowed, "person_id")
    df = aff.join(attrs, "person_id")

    feat_cols = CATS + ["age", "income_z", "recency_days"]
    data_v = VectorAssembler(inputCols=feat_cols, outputCol="features").transform(df)
    train, test = data_v.randomSplit([0.7, 0.3], seed=42)
    train.cache(); test.cache()

    auc = BinaryClassificationEvaluator(labelCol="converted_true", metricName="areaUnderROC")

    lr = LogisticRegression(featuresCol="features", labelCol="converted_true", maxIter=100).fit(train)
    gbt = GBTClassifier(featuresCol="features", labelCol="converted_true", maxIter=60, maxDepth=5, seed=42).fit(train)
    auc_lr = float(auc.evaluate(lr.transform(test)))
    auc_gbt = float(auc.evaluate(gbt.transform(test)))

    base_rate = float(df.agg(F.avg("converted_true")).first()[0])
    metrics = {
        "n_people": int(df.count()),
        "base_conversion_rate": round(base_rate, 4),
        "auc_logistic_regression": round(auc_lr, 4),
        "auc_gbt": round(auc_gbt, 4),
        "gbt_auc_gain_over_lr": round(auc_gbt - auc_lr, 4),
        "winner": "gbt" if auc_gbt >= auc_lr else "logistic_regression",
        "n_features": len(feat_cols),
    }
    spark.stop()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="reports/propensity_metrics.json")
    args = ap.parse_args()
    m = run(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(m, open(args.out, "w"), indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
