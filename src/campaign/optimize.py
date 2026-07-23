"""
Module 6 - Campaign Optimizer (Bayesian Data Analysis). Lotame "Optimizer".

Two Bayesian decisions on creative allocation:

1. Bayesian A/B test - from a uniform-allocation reward stream, form a
   Beta-Binomial posterior per creative, report each arm's posterior mean, 95%
   credible interval, and P(arm is best) by Monte Carlo. This answers "which
   creative wins, and how sure are we?" - not a p-value, an actual probability.

2. Thompson-sampling bandit - allocate impressions adaptively by sampling each
   arm's posterior and serving the argmax, so budget flows to winners while the
   test is still running. Measured by cumulative regret vs the (unknown) best
   arm, against a uniform-A/B baseline.

Bayesian decision-making under uncertainty is the core of my research background
(large-scale Bayesian inversion); here it is the same machinery pointed at
creative optimization.

Run:  PYTHONPATH=. .venv/bin/python src/campaign/optimize.py --data data/synthetic
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

SEED = 42
MC_SAMPLES = 200_000
BANDIT_ROUNDS = 40_000


def bayesian_ab(stream: pd.DataFrame, arms: list[str], rng: np.random.Generator) -> dict:
    """Beta(1,1)-prior posteriors from the uniform-allocation stream."""
    post = {}
    for a in arms:
        r = stream.loc[stream["creative"] == a, "reward"]
        succ, fail = int(r.sum()), int((r == 0).sum())
        post[a] = (1 + succ, 1 + fail)
    # Monte-Carlo P(best) and credible intervals.
    samples = np.vstack([rng.beta(post[a][0], post[a][1], MC_SAMPLES) for a in arms])
    p_best = np.bincount(samples.argmax(axis=0), minlength=len(arms)) / MC_SAMPLES
    out = {}
    for i, a in enumerate(arms):
        alpha, beta = post[a]
        lo, hi = np.percentile(samples[i], [2.5, 97.5])
        out[a] = {"posterior_mean": round(alpha / (alpha + beta), 4),
                  "ci95": [round(float(lo), 4), round(float(hi), 4)],
                  "p_best": round(float(p_best[i]), 4)}
    return out


def thompson_regret(true_cr: np.ndarray, rng: np.random.Generator, rounds: int) -> tuple[float, np.ndarray]:
    """Adaptive allocation; returns cumulative regret and per-arm pull counts."""
    n = len(true_cr)
    a = np.ones(n); b = np.ones(n)          # Beta(1,1) priors
    best = true_cr.max()
    pulls = np.zeros(n, dtype=int)
    regret = 0.0
    for _ in range(rounds):
        theta = rng.beta(a, b)
        arm = int(theta.argmax())
        reward = 1 if rng.random() < true_cr[arm] else 0
        a[arm] += reward; b[arm] += 1 - reward
        pulls[arm] += 1
        regret += best - true_cr[arm]
    return regret, pulls


def run(data: str) -> dict:
    truth = pd.read_parquet(f"{data}/campaign_truth.parquet")
    stream = pd.read_parquet(f"{data}/campaign_stream.parquet")
    arms = list(truth["creative"])
    true_cr = truth.set_index("creative")["true_conv_rate"].reindex(arms).to_numpy()
    rng = np.random.default_rng(SEED)

    ab = bayesian_ab(stream, arms, rng)
    true_best_i = int(true_cr.argmax())
    bayes_best = max(ab, key=lambda a: ab[a]["p_best"])

    # Thompson vs uniform-A/B regret over the same horizon.
    t_regret, pulls = thompson_regret(true_cr, rng, BANDIT_ROUNDS)
    uniform_regret = BANDIT_ROUNDS * (true_cr.max() - true_cr.mean())
    best_share = float(pulls[true_best_i] / pulls.sum())

    metrics = {
        "arms": len(arms),
        "true_best_creative": arms[true_best_i],
        "true_best_rate": round(float(true_cr[true_best_i]), 4),
        "bayesian_identified_best": bayes_best,
        "identified_best_correct": bool(bayes_best == arms[true_best_i]),
        "p_best_of_winner": ab[bayes_best]["p_best"],
        "posteriors": ab,
        "thompson_regret": round(float(t_regret), 1),
        "uniform_ab_regret": round(float(uniform_regret), 1),
        "regret_reduction_vs_uniform": round(1 - t_regret / uniform_regret, 4),
        "thompson_share_to_best_arm": round(best_share, 4),
    }
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--out", default="reports/campaign_metrics.json")
    args = ap.parse_args()
    m = run(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(m, open(args.out, "w"), indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
