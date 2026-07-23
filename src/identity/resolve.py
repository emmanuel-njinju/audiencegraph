"""
Release 1 - Identity Resolution Engine (the centerpiece; Lotame "Cross Device").

Given only device-level signals (cookies, MAIDs, hashed emails, IPs) plus the
behavioral event log, reconstruct which devices belong to the same person - the
core of cross-device identity resolution and the input to every downstream
audience product.

Pipeline
--------
0. CONSENT GATE          drop non-consented / opted-out people first (privacy-safe).
1. DETERMINISTIC MATCH   devices sharing a hashed_email or MAID are the same person
                         (high precision, low recall - misses the logged-out long tail).
2. PROBABILISTIC MATCH   for device pairs that only share a household IP, a supervised
                         pairwise model (Spark MLlib logistic regression on behavioral
                         similarity) decides same-person vs same-household, recovering
                         links deterministic matching cannot - WITHOUT merging a whole
                         household into one person.
3. GRAPH RESOLUTION      union the edges, run connected components -> one id per person.
4. EVALUATION            pairwise precision / recall / F1 against the hidden ground-truth
                         person id, reported as an ABLATION (deterministic-only vs
                         deterministic+probabilistic) so the lift from step 2 is measurable.

The ground-truth person id is used ONLY in evaluation and to label training pairs;
it is never an input to the resolver.

Run:  PYTHONPATH=. .venv/bin/python src/identity/resolve.py --data data/synthetic
"""
from __future__ import annotations

import argparse
import json
import os

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F

from src.common.spark import get_spark
from src.common.consent import consent_report, gate_devices
from src.common.connected_components import connected_components

MAX_BLOCK = 50          # skip IP blocks bigger than this (public/transient IPs)
PRECISION_TARGET = 0.90  # operate the probabilistic matcher at high precision


# ------------------------------- features ---------------------------------- #

def device_unit_vectors(events: DataFrame) -> DataFrame:
    """Per-device L2-normalized content-affinity vector (behavioral fingerprint)."""
    vec = (events.groupBy("device_id", "content_cat")
                 .agg((3 * F.sum("click") + F.count("*")).cast("double").alias("w")))
    norm = vec.groupBy("device_id").agg(F.sqrt(F.sum(F.col("w") * F.col("w"))).alias("nrm"))
    return (vec.join(norm, "device_id")
               .select("device_id", "content_cat", (F.col("w") / F.col("nrm")).alias("u")))


def _self_pairs(signals: DataFrame, types: list[str], max_block: int) -> DataFrame:
    """Undirected device pairs that share a signal value of the given type(s)."""
    s = (signals.where(F.col("signal_type").isin(types))
                .select("device_id", "signal_value").distinct())
    small = s.groupBy("signal_value").count().where(F.col("count") <= max_block).select("signal_value")
    s = s.join(small, "signal_value")
    a, b = s.alias("a"), s.alias("b")
    return (a.join(b, (F.col("a.signal_value") == F.col("b.signal_value")) &
                      (F.col("a.device_id") < F.col("b.device_id")))
             .select(F.col("a.device_id").alias("src"), F.col("b.device_id").alias("dst"))
             .distinct())


def pair_features(cands: DataFrame, uvec: DataFrame, devices: DataFrame) -> DataFrame:
    """Behavioral-similarity features for candidate pairs, plus the ground-truth label."""
    usrc = uvec.select(F.col("device_id").alias("src"), "content_cat", F.col("u").alias("us"))
    udst = uvec.select(F.col("device_id").alias("dst"), "content_cat", F.col("u").alias("ud"))
    cos = (cands.join(usrc, "src").join(udst, ["dst", "content_cat"])
                .groupBy("src", "dst").agg(F.sum(F.col("us") * F.col("ud")).alias("cosine")))
    feat = cands.join(cos, ["src", "dst"], "left").fillna({"cosine": 0.0})

    dt = devices.select("device_id", "device_type", "person_id")
    feat = (feat.join(dt.select(F.col("device_id").alias("src"),
                                F.col("device_type").alias("ts"),
                                F.col("person_id").alias("psrc")), "src")
                .join(dt.select(F.col("device_id").alias("dst"),
                                F.col("device_type").alias("td"),
                                F.col("person_id").alias("pdst")), "dst"))
    return feat.select(
        "src", "dst", "cosine",
        (F.col("ts") == F.col("td")).cast("double").alias("same_type"),
        (F.col("psrc") == F.col("pdst")).cast("double").alias("label"),  # eval/training only
    )


# ------------------------ probabilistic matcher ---------------------------- #

def train_probabilistic_matcher(feat: DataFrame) -> tuple[DataFrame, dict]:
    """Spark MLlib logistic regression over pairwise features. Returns scored pairs + report."""
    va = VectorAssembler(inputCols=["cosine", "same_type"], outputCol="features")
    data = va.transform(feat)
    train, test = data.randomSplit([0.7, 0.3], seed=42)
    model = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50).fit(train)

    auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC") \
        .evaluate(model.transform(test))

    # Choose the operating threshold that hits PRECISION_TARGET (avoid over-merge),
    # from the held-out test set.
    test_scored = (model.transform(test)
                        .withColumn("p", vector_to_array("probability")[1])
                        .select("p", "label")).collect()
    threshold = _threshold_for_precision([(r["p"], r["label"]) for r in test_scored], PRECISION_TARGET)

    scored = model.transform(data).withColumn("p", vector_to_array("probability")[1])
    report = {"auc": round(float(auc), 4), "operating_threshold": round(threshold, 3),
              "coef": [round(float(c), 3) for c in model.coefficients], "n_pairs": data.count()}
    return scored, report


def _threshold_for_precision(pairs: list[tuple[float, float]], target: float) -> float:
    """Smallest threshold whose precision >= target (max recall at that precision)."""
    pairs = sorted(pairs, key=lambda x: -x[0])
    tp = fp = 0
    best = 0.5
    for p, label in pairs:
        if label == 1.0:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        if precision >= target:
            best = p                       # keep lowering the threshold while precision holds
    return best


# ------------------------------ evaluation --------------------------------- #

def _pairs(df: DataFrame, group: list[str]) -> float:
    g = df.groupBy(*group).count()
    v = g.select(F.sum(F.col("count") * (F.col("count") - 1) / 2.0).alias("p")).first()["p"]
    return float(v or 0.0)


def pairwise_prf(assignment: DataFrame, truth: DataFrame) -> dict:
    """Pairwise precision/recall/F1 of a clustering vs ground-truth persons."""
    j = assignment.join(truth, "id")                       # id, component, person
    pred = _pairs(j, ["component"])
    real = _pairs(truth, ["person"])
    tp = _pairs(j, ["component", "person"])
    precision = tp / pred if pred else 0.0
    recall = tp / real if real else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# -------------------------------- pipeline --------------------------------- #

def run(data: str) -> dict:
    spark = get_spark("identity-resolution")
    persons = spark.read.parquet(f"{data}/persons.parquet")
    devices_all = spark.read.parquet(f"{data}/devices.parquet")
    signals_all = spark.read.parquet(f"{data}/identity_signals.parquet")
    events_all = spark.read.parquet(f"{data}/events.parquet")

    # 0. Consent gate - everything below sees only the consented population.
    creport = consent_report(persons)
    devices = gate_devices(devices_all, persons).cache()
    keep = devices.select("device_id")
    signals = signals_all.join(keep, "device_id")
    events = events_all.join(keep, "device_id")

    nodes = devices.select(F.col("device_id").alias("id"))
    truth = devices.select(F.col("device_id").alias("id"), F.col("person_id").alias("person"))

    # 1. Deterministic edges.
    det = _self_pairs(signals, ["hashed_email", "maid"], MAX_BLOCK).cache()

    # 2. Probabilistic edges (IP-blocked candidates, minus the ones already linked).
    ip_cands = _self_pairs(signals, ["ip"], MAX_BLOCK).join(det, ["src", "dst"], "left_anti")
    uvec = device_unit_vectors(events)
    feat = pair_features(ip_cands, uvec, devices)
    scored, matcher = train_probabilistic_matcher(feat)
    prob = scored.where(F.col("p") >= matcher["operating_threshold"]).select("src", "dst")

    # 3. Graph resolution - ablation: deterministic-only vs deterministic+probabilistic.
    det_labels = connected_components(nodes, det)
    all_edges = det.select("src", "dst").union(prob).distinct()
    full_labels = connected_components(nodes, all_edges)

    # 4. Evaluate.
    metrics = {
        "consent": creport.as_dict(),
        "probabilistic_matcher": matcher,
        "resolution": {
            "deterministic_only": pairwise_prf(det_labels, truth),
            "deterministic_plus_probabilistic": pairwise_prf(full_labels, truth),
        },
        "counts": {
            "devices_in_graph": nodes.count(),
            "true_persons": truth.select("person").distinct().count(),
            "resolved_entities_full": full_labels.select("component").distinct().count(),
            "deterministic_edges": det.count(),
            "probabilistic_edges": prob.count(),
        },
    }
    d0 = metrics["resolution"]["deterministic_only"]["f1"]
    d1 = metrics["resolution"]["deterministic_plus_probabilistic"]["f1"]
    metrics["resolution"]["f1_lift_from_probabilistic"] = round(d1 - d0, 4)
    spark.stop()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="reports/identity_metrics.json")
    args = ap.parse_args()
    metrics = run(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
