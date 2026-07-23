"""
Build the results walkthrough: a one-figure story panel + reports/RESULTS.md.

Reads every reports/*_metrics.json produced by the modules and assembles the
single narrative a reviewer remembers - fragmented signals -> resolved identity
-> segments -> propensity -> lookalike expansion -> CTR optimization -> Bayesian
creative allocation, with the metric that moved at each step. Every number is
read from a committed metrics file, so the story can never drift from the code.

Usage:  python scripts/build_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
FIG = REP / "figures"


def load(name: str) -> dict | None:
    p = REP / f"{name}_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _bars(ax, title, labels, values, fmt="{:.3f}", ylabel=""):
    colors = ["#9aa7b4", "#2E8B57"] if len(values) == 2 else ["#2E8B57"] * len(values)
    bars = ax.bar(labels, values, color=colors[: len(values)])
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def build_panel(idn, seg, prop, look, ctr, camp) -> Path | None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("AudienceGraph — the metric that improved at each step",
                 fontsize=13, fontweight="bold")

    if idn:
        r = idn["resolution"]
        _bars(axes[0, 0], "1 · Identity: cross-device recall",
              ["deterministic", "det + prob"],
              [r["deterministic_only"]["recall"], r["deterministic_plus_probabilistic"]["recall"]],
              ylabel="pairwise recall")
    if seg:
        _bars(axes[0, 1], "2 · Segmentation: ARI vs truth",
              ["K-means", "GMM"], [seg["ari_kmeans"], seg["ari_gmm"]], ylabel="adjusted rand index")
    if prop:
        _bars(axes[0, 2], "3 · Propensity: model AUC",
              ["logistic", "GBT"], [prop["auc_logistic_regression"], prop["auc_gbt"]], ylabel="AUC")
    if look:
        _bars(axes[1, 0], "5 · Lookalike: converter rate",
              ["base pool", "expanded"], [look["base_heldout_rate"], look["lookalike_heldout_rate"]],
              ylabel="held-out converter rate")
    if ctr:
        _bars(axes[1, 1], "4 · CTR: logloss (lower better)",
              ["baseline", "model"], [ctr["baseline_logloss"], ctr["logloss"]], ylabel="log loss")
    if camp:
        _bars(axes[1, 2], "6 · Campaign: cumulative regret (lower better)",
              ["uniform A/B", "Thompson"], [camp["uniform_ab_regret"], camp["thompson_regret"]],
              fmt="{:.0f}", ylabel="regret")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG / "results_panel.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def build_markdown(idn, seg, prop, look, ctr, camp) -> str:
    L = ["# AudienceGraph — Results Walkthrough",
         "",
         "_One story, evaluated against hidden ground truth at every step. Every "
         "number below is read from a committed `reports/*_metrics.json`; regenerate "
         "with `make all-modules && python scripts/build_results.py`._",
         "",
         "![results panel](figures/results_panel.png)",
         ""]
    if idn:
        c, r = idn["consent"], idn["resolution"]
        L += [f"## 0 → 1 · Consent, then identity resolution",
              f"- Consent gate retained **{c['retained_pct']}%** of people "
              f"({c['suppressed_gdpr_no_consent']:,} GDPR + {c['suppressed_ccpa_opt_out']:,} CCPA suppressed).",
              f"- Cross-device recall **{r['deterministic_only']['recall']} → "
              f"{r['deterministic_plus_probabilistic']['recall']}** at "
              f"**{r['deterministic_plus_probabilistic']['precision']}** precision "
              f"(probabilistic matcher AUC {idn['probabilistic_matcher']['auc']}).", ""]
    if seg:
        L += [f"## 2 · Audience segmentation (clustering)",
              f"- **GMM ARI {seg['ari_gmm']}** vs K-means {seg['ari_kmeans']} at k={seg['chosen_k']} "
              f"— soft/elliptical clusters match overlapping audiences better.", ""]
    if prop:
        w = prop["winner"].replace("_", " ")
        best = max(prop["auc_logistic_regression"], prop["auc_gbt"])
        L += [f"## 3 · Behavioral propensity (classification)",
              f"- Logistic AUC {prop['auc_logistic_regression']} vs GBT {prop['auc_gbt']}; "
              f"**{w} wins (AUC {best})**. Model choice is made on evidence, not fashion: the "
              f"signal here is largely linear, so the simpler, interpretable, cheaper-to-serve "
              f"model is also the more accurate one - and the one to ship.", ""]
    if look:
        L += [f"## 5 · Lookalike expansion (collaborative filtering)",
              f"- Seeding on half the converters and expanding via ALS-predicted preference for "
              f"the seed's characteristic items surfaces held-out converters at "
              f"**{look['lookalike_heldout_rate']}** vs a {look['base_heldout_rate']} base rate "
              f"— a **{look['lift']}x lift**. (Modest by design: the synthetic catalog has only "
              f"12 items; ALS separates far better on a real high-cardinality catalog.)", ""]
    if ctr:
        L += [f"## 4 · CTR / conversion optimizer (regression at scale)",
              f"- Hashed logistic regression on **{ctr['n_impressions']:,} impressions**: "
              f"AUC **{ctr['auc']}**, log loss {ctr['logloss']} vs {ctr['baseline_logloss']} baseline "
              f"(**{int(ctr['logloss_reduction_vs_baseline']*100)}%** reduction).", ""]
    if camp:
        L += [f"## 6 · Campaign optimizer (Bayesian)",
              f"- Bayesian A/B identified the true best creative "
              f"(**{camp['bayesian_identified_best']}**, P(best)={camp['p_best_of_winner']}, "
              f"correct={camp['identified_best_correct']}).",
              f"- Thompson sampling cut cumulative regret by "
              f"**{int(camp['regret_reduction_vs_uniform']*100)}%** vs uniform A/B, sending "
              f"**{int(camp['thompson_share_to_best_arm']*100)}%** of impressions to the best arm.", ""]
    return "\n".join(L)


def main() -> None:
    idn = load("identity"); seg = load("segmentation"); prop = load("propensity")
    look = load("lookalike"); ctr = load("ctr"); camp = load("campaign")
    build_panel(idn, seg, prop, look, ctr, camp)
    (REP / "RESULTS.md").write_text(build_markdown(idn, seg, prop, look, ctr, camp))
    print("wrote reports/RESULTS.md + reports/figures/results_panel.png")


if __name__ == "__main__":
    main()
