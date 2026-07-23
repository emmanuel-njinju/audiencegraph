"""
Module 5 - Lookalike Expansion (Collaborative Filtering).

Learn latent audience factors from the person x content interaction matrix with
ALS (implicit feedback), then expand a seed audience (converters) to the
non-seed people whose latent profile is most similar - the "seed-to-scale"
lookalike product. Success = the expanded audience converts at a materially
higher rate than the base population (measured lift against ground truth).

Why ALS collaborative filtering
-------------------------------
Content-based rules ("show cat_03 lovers more cat_03") miss cross-category
affinities. ALS discovers latent factors that place behaviorally-similar people
near each other even when they share no single category outright, and Spark's
ALS scales to hundreds of millions of interactions - the reason it is the
standard for audience/rec factorization.

Run:  PYTHONPATH=. .venv/bin/python src/lookalike/expand.py --data data/synthetic
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F

from src.common.spark import get_spark
from src.common.consent import allowed_person_ids

CATS = [f"cat_{i:02d}" for i in range(12)]
RANK = 8
EXPAND_FRACTION = 0.20   # size of the lookalike audience, as a share of non-seed people
CHAR_ITEMS = 2           # how many seed-characteristic items drive the expansion score


def run(data: str) -> dict:
    spark = get_spark("lookalike")
    persons = spark.read.parquet(f"{data}/persons.parquet")
    allowed = allowed_person_ids(persons)                    # consent gate
    inter = spark.read.parquet(f"{data}/interactions.parquet").join(allowed, "person_id")

    # Map content category -> integer item id for ALS.
    cat_idx = {c: i for i, c in enumerate(CATS)}
    idx_expr = F.create_map([x for c in CATS for x in (F.lit(c), F.lit(cat_idx[c]))])
    ratings = inter.select(F.col("person_id").cast("int").alias("user"),
                           idx_expr[F.col("content_cat")].alias("item"),
                           F.col("weight").cast("float").alias("rating"))

    als = ALS(rank=RANK, maxIter=12, implicitPrefs=True, alpha=20.0,
              userCol="user", itemCol="item", ratingCol="rating",
              coldStartStrategy="drop", seed=42, nonnegative=False)
    model = als.fit(ratings)

    # Pull user AND item factors to the driver (n ~ 1e4, 12 items - tiny).
    uf = model.userFactors.toPandas()
    U = np.vstack(uf["features"].to_numpy())
    uid = uf["id"].to_numpy()
    vf = model.itemFactors.toPandas()
    V = np.zeros((len(CATS), U.shape[1]))
    for _, r in vf.iterrows():
        V[int(r["id"])] = r["features"]
    pref = U @ V.T                                            # ALS-predicted user x item preference

    labels = (persons.select("person_id", "converted_true")
                     .join(allowed, "person_id").toPandas()
                     .set_index("person_id")["converted_true"].to_dict())
    converted = np.array([labels.get(int(i), 0) for i in uid])

    # Proper lookalike evaluation: HOLD OUT half the converters. Build the model
    # from the other half, expand over a pool that mixes the held-out converters
    # with all non-converters, and measure whether the expanded audience surfaces
    # the held-out converters at a higher-than-base rate. This answers the real
    # product question ("does seed-to-scale find NEW converters?") without the
    # circularity of scoring the seed against itself.
    positions = np.arange(len(uid))
    conv_pos = positions[converted == 1]
    if len(conv_pos) < 20:
        raise RuntimeError("too few converters to seed + hold out")
    np.random.default_rng(0).shuffle(conv_pos)
    half = len(conv_pos) // 2
    seed_mask = np.zeros(len(uid), bool); seed_mask[conv_pos[:half]] = True
    target = np.zeros(len(uid), bool); target[conv_pos[half:]] = True     # held-out converters

    # Seed-characteristic items: the items the seed over-indexes on relative to
    # the whole population, in ALS-predicted-preference space (data-driven - no
    # ground-truth item labels). Score every person by their ALS-predicted
    # preference for those items. This uses ALS to GENERALIZE (it scores people
    # with sparse direct history via the latent factors), which is the point of
    # collaborative filtering for expansion.
    char = np.argsort(-(pref[seed_mask].mean(0) - pref.mean(0)))[:CHAR_ITEMS]
    score = pref[:, char].sum(axis=1)

    pool_mask = ~seed_mask
    pool_target = target[pool_mask]
    order = np.argsort(-score[pool_mask])
    k = max(1, int(EXPAND_FRACTION * pool_mask.sum()))

    base_rate = float(pool_target.mean())
    look_rate = float(pool_target[order[:k]].mean())
    metrics = {
        "n_users": int(len(uid)),
        "n_seed_converters": int(seed_mask.sum()),
        "n_heldout_converters": int(target.sum()),
        "als_rank": RANK,
        "characteristic_items": [CATS[int(i)] for i in char],
        "expand_top_k": int(k),
        "base_heldout_rate": round(base_rate, 4),
        "lookalike_heldout_rate": round(look_rate, 4),
        "lift": round(look_rate / base_rate, 3) if base_rate else None,
    }
    spark.stop()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="reports/lookalike_metrics.json")
    args = ap.parse_args()
    m = run(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(m, open(args.out, "w"), indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
