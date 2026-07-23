"""
Module 2 - Audience Segmentation (Clustering).

Group consented people into behavioral audience segments from their OBSERVED
content-affinity (built from the ad-impression log, not the latent truth), then
name each segment by its dominant content and score the clustering against the
hidden ground-truth segment with Adjusted Rand Index.

Why these algorithms
--------------------
- K-means: fast, scales linearly, the right default for large spherical-ish
  audience clusters; k chosen by silhouette.
- Gaussian Mixture: a soft, elliptical alternative that models overlapping
  audiences (a person is rarely 100% one segment); compared head-to-head so the
  choice is justified, not assumed - which is what the JD asks for.

Run:  PYTHONPATH=. .venv/bin/python src/segmentation/segment.py --data data/synthetic
"""
from __future__ import annotations

import argparse
import json
import os

from pyspark.ml.clustering import GaussianMixture, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, functions as F
from sklearn.metrics import adjusted_rand_score

from src.common.spark import get_spark
from src.common.consent import allowed_person_ids

CATS = [f"cat_{i:02d}" for i in range(12)]


def person_affinity(events: DataFrame, devices: DataFrame) -> DataFrame:
    """Person x content-category affinity from observed impressions (consented)."""
    ev = events.join(devices.select("device_id", "person_id"), "device_id")
    counts = (ev.groupBy("person_id", "content_cat")
                .agg((3 * F.sum("click") + F.count("*")).cast("double").alias("w")))
    wide = (counts.groupBy("person_id")
                  .pivot("content_cat", CATS)
                  .agg(F.first("w")).na.fill(0.0))
    return wide


def run(data: str, k_grid=range(4, 11)) -> dict:
    spark = get_spark("segmentation")
    persons = spark.read.parquet(f"{data}/persons.parquet")
    devices = spark.read.parquet(f"{data}/devices.parquet").join(
        allowed_person_ids(persons), "person_id")           # consent gate
    events = spark.read.parquet(f"{data}/events.parquet")

    feats = person_affinity(events, devices).cache()
    assembled = VectorAssembler(inputCols=CATS, outputCol="raw").transform(feats)
    scaled = StandardScaler(inputCol="raw", outputCol="features", withStd=True, withMean=True) \
        .fit(assembled).transform(assembled).cache()

    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")

    # Pick k for K-means by silhouette.
    km_scores = {}
    for k in k_grid:
        model = KMeans(k=k, seed=42, featuresCol="features").fit(scaled)
        km_scores[k] = float(evaluator.evaluate(model.transform(scaled)))
    best_k = max(km_scores, key=km_scores.get)

    km = KMeans(k=best_k, seed=42, featuresCol="features").fit(scaled)
    km_pred = km.transform(scaled).select("person_id", F.col("prediction").alias("kmeans"))
    gm = GaussianMixture(k=best_k, seed=42, featuresCol="features").fit(scaled)
    gm_pred = gm.transform(scaled).select("person_id", F.col("prediction").alias("gmm"))

    # External validation vs the hidden true segment (ARI).
    truth = persons.select("person_id", "true_segment")
    joined = (km_pred.join(gm_pred, "person_id").join(truth, "person_id")).toPandas()
    ari_km = adjusted_rand_score(joined["true_segment"], joined["kmeans"])
    ari_gm = adjusted_rand_score(joined["true_segment"], joined["gmm"])

    # Name each K-means segment by its top content categories (audience labels).
    centers = km.clusterCenters()   # in standardized space; rank dims by center value
    named = {}
    for cid, c in enumerate(centers):
        top = sorted(range(len(CATS)), key=lambda i: -c[i])[:2]
        named[cid] = " + ".join(CATS[i] for i in top) + " affinity"

    sizes = (km_pred.groupBy("kmeans").count()
                    .orderBy("kmeans").rdd.map(lambda r: (int(r["kmeans"]), int(r["count"]))).collect())

    metrics = {
        "n_people": int(joined.shape[0]),
        "chosen_k": int(best_k),
        "silhouette_by_k": {str(k): round(v, 4) for k, v in km_scores.items()},
        "ari_kmeans": round(float(ari_km), 4),
        "ari_gmm": round(float(ari_gm), 4),
        "winner": "kmeans" if ari_km >= ari_gm else "gmm",
        "named_segments": {str(cid): {"label": named[cid], "size": dict(sizes).get(cid, 0)}
                           for cid in named},
    }
    spark.stop()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="reports/segmentation_metrics.json")
    args = ap.parse_args()
    m = run(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(m, open(args.out, "w"), indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
