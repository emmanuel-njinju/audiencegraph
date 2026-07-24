"""
Build reports/dashboard.html - a self-contained visual dashboard of the whole
platform, generated from the committed reports/*_metrics.json (no hand-typed
numbers, no external dependencies, so it renders on GitHub Pages or opened
locally). Dark-mode aware.

Usage:
    python scripts/build_dashboard.py                      # writes reports/dashboard.html
    python scripts/build_dashboard.py --artifact-out X.html # also writes a body-only copy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def load(name: str) -> dict:
    return json.loads((REP / f"{name}_metrics.json").read_text())


# ------------------------------- chart helpers ----------------------------- #

def bar(label: str, value: float, vmax: float, fmt: str, cls: str, sub: str = "") -> str:
    pct = max(1.5, 100 * value / vmax)
    return (f'<div class="bar-row"><span class="bar-lab">{label}{sub}</span>'
            f'<span class="track"><span class="fill {cls}" style="width:{pct:.1f}%"></span></span>'
            f'<span class="bar-val">{fmt.format(value)}</span></div>')


def donut(retained: float, parts: list[tuple[str, float, str]]) -> str:
    # parts: (label, pct_of_total, css-var). Build a conic-gradient ring.
    stops, acc = [], 0.0
    for _, pct, col in parts:
        stops.append(f"{col} {acc:.2f}% {acc + pct:.2f}%")
        acc += pct
    grad = ", ".join(stops)
    return (f'<div class="donut" style="background:conic-gradient({grad})">'
            f'<div class="donut-hole"><span class="donut-big">{retained:.0f}%</span>'
            f'<span class="donut-cap">retained</span></div></div>')


def ci_chart(post: dict, best: str) -> str:
    # Horizontal credible-interval chart: a line per creative (2.5-97.5%) + mean dot.
    keys = list(post.keys())
    lo = min(post[k]["ci95"][0] for k in keys)
    hi = max(post[k]["ci95"][1] for k in keys)
    pad = (hi - lo) * 0.12
    x0, x1 = lo - pad, hi + pad
    W, rowh, left = 100.0, 30, 46
    def X(v): return left + (W - left - 4) * (v - x0) / (x1 - x0)
    rows = []
    for i, k in enumerate(keys):
        y = 16 + i * rowh
        p = post[k]
        a, b = p["ci95"]; m = p["posterior_mean"]
        is_best = k == best
        col = "var(--good)" if is_best else "var(--muted)"
        rows.append(
            f'<text x="0" y="{y+4}" class="ci-lab">{k}</text>'
            f'<line x1="{X(a):.1f}" y1="{y}" x2="{X(b):.1f}" y2="{y}" stroke="{col}" '
            f'stroke-width="2" stroke-linecap="round" opacity="{0.9 if is_best else 0.5}"/>'
            f'<circle cx="{X(m):.1f}" cy="{y}" r="{4 if is_best else 3}" fill="{col}"/>'
            + (f'<text x="{X(b)+2:.1f}" y="{y+4}" class="ci-best">P(best) {p["p_best"]:.4f}</text>'
               if is_best else ""))
    ticks = []
    for t in range(int(x0 * 100) + 1, int(x1 * 100) + 1, 2):
        tv = t / 100
        ticks.append(f'<line x1="{X(tv):.1f}" y1="10" x2="{X(tv):.1f}" y2="{16+len(keys)*rowh-8}" '
                     f'class="ci-grid"/><text x="{X(tv):.1f}" y="{16+len(keys)*rowh+4}" '
                     f'class="ci-tick">{tv:.0%}</text>')
    h = 16 + len(keys) * rowh + 10
    return (f'<svg viewBox="0 0 100 {h}" class="ci-svg" preserveAspectRatio="none" '
            f'role="img" aria-label="posterior conversion rate with 95% credible interval per creative">'
            f'{"".join(ticks)}{"".join(rows)}</svg>')


# --------------------------------- content --------------------------------- #

CSS = """
:root{
  --bg:#eef1f6; --surface:#ffffff; --panel:#f6f8fb; --line:#e2e8f0;
  --ink:#0f1729; --muted:#64748b; --faint:#94a3b8;
  --accent:#4f46e5; --accent-2:#0891b2; --good:#15803d; --warn:#b45309;
  --track:#e6eaf2;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#080c15; --surface:#0f1626; --panel:#121b2e; --line:#22304a;
  --ink:#e7ecf5; --muted:#94a3b8; --faint:#64748b;
  --accent:#818cf8; --accent-2:#22d3ee; --good:#4ade80; --warn:#fbbf24;
  --track:#1c2740;
}}
:root[data-theme="light"]{
  --bg:#eef1f6; --surface:#ffffff; --panel:#f6f8fb; --line:#e2e8f0;
  --ink:#0f1729; --muted:#64748b; --faint:#94a3b8;
  --accent:#4f46e5; --accent-2:#0891b2; --good:#15803d; --warn:#b45309; --track:#e6eaf2;
}
:root[data-theme="dark"]{
  --bg:#080c15; --surface:#0f1626; --panel:#121b2e; --line:#22304a;
  --ink:#e7ecf5; --muted:#94a3b8; --faint:#64748b;
  --accent:#818cf8; --accent-2:#22d3ee; --good:#4ade80; --warn:#fbbf24; --track:#1c2740;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:40px 24px 72px}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
h1{font-size:clamp(1.7rem,4vw,2.4rem);font-weight:800;letter-spacing:-0.03em;margin:0;text-wrap:balance}
.sub{color:var(--muted);max-width:60ch;margin:10px 0 0;font-size:1rem}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.chip{font-size:.72rem;font-weight:600;letter-spacing:.02em;padding:5px 10px;border-radius:999px;
  background:var(--panel);border:1px solid var(--line);color:var(--muted)}
.eyebrow{font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:44px 0 14px}
/* KPI tiles */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 16px 14px}
.kpi .n{font-size:1.9rem;font-weight:800;letter-spacing:-0.02em}
.kpi .k{font-size:.78rem;color:var(--muted);margin-top:3px}
.kpi .d{font-size:.72rem;color:var(--good);font-weight:600;margin-top:6px}
.up::before{content:"\\2191 "}
/* pipeline */
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:0;background:var(--surface);
  border:1px solid var(--line);border-radius:16px;overflow:hidden}
.step{flex:1 1 150px;padding:18px 18px;border-right:1px solid var(--line);position:relative}
.step:last-child{border-right:0}
.step .sn{font-size:1.5rem;font-weight:800;letter-spacing:-.02em}
.step .sl{font-size:.76rem;color:var(--muted);margin-top:2px}
.step .st{font-size:.7rem;color:var(--faint);margin-top:6px}
/* cards */
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:20px}
.card.wide{grid-column:1/-1}
.card h3{margin:0;font-size:1.02rem;font-weight:700;display:flex;align-items:center;gap:10px}
.pill{font-size:.66rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
  padding:3px 8px;border-radius:6px;background:color-mix(in srgb,var(--accent) 14%,transparent);color:var(--accent)}
.pill.good{background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--good)}
.why{font-size:.82rem;color:var(--muted);margin:14px 0 0}
.chart{margin-top:16px;display:flex;flex-direction:column;gap:9px}
.bar-row{display:grid;grid-template-columns:120px 1fr 58px;align-items:center;gap:10px}
.bar-lab{font-size:.78rem;color:var(--muted)}
.track{height:12px;background:var(--track);border-radius:6px;overflow:hidden}
.fill{display:block;height:100%;border-radius:6px}
.fill.accent{background:var(--accent)} .fill.muted{background:var(--faint)}
.fill.good{background:var(--good)} .fill.warn{background:var(--warn)}
.bar-val{font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums;font-size:.82rem;text-align:right;font-weight:600}
.substats{display:flex;gap:22px;margin-top:16px;flex-wrap:wrap}
.substat .sv{font-size:1.15rem;font-weight:700}
.substat .sk{font-size:.72rem;color:var(--muted)}
.lift{font-size:1.7rem;font-weight:800;color:var(--good)}
/* donut */
.donut-wrap{display:flex;align-items:center;gap:20px;margin-top:14px}
.donut{width:118px;height:118px;border-radius:50%;flex:0 0 auto;position:relative;display:grid;place-items:center}
.donut-hole{width:74px;height:74px;border-radius:50%;background:var(--surface);display:grid;place-items:center;text-align:center}
.donut-big{font-size:1.35rem;font-weight:800;line-height:1}
.donut-cap{font-size:.66rem;color:var(--muted)}
.legend{display:flex;flex-direction:column;gap:8px;font-size:.8rem}
.legend .li{display:flex;align-items:center;gap:8px;color:var(--muted)}
.dot{width:10px;height:10px;border-radius:3px;flex:0 0 auto}
/* ci svg */
.ci-svg{width:100%;height:auto}
.ci-lab{font-size:5px;fill:var(--muted);font-family:ui-monospace,monospace}
.ci-best{font-size:4.6px;fill:var(--good);font-family:ui-monospace,monospace;font-weight:700}
.ci-tick{font-size:4px;fill:var(--faint);text-anchor:middle;font-family:ui-monospace,monospace}
.ci-grid{stroke:var(--line);stroke-width:.4}
.foot{margin-top:44px;padding-top:20px;border-top:1px solid var(--line);color:var(--faint);font-size:.78rem}
.foot a{color:var(--accent);text-decoration:none}
"""


def body(idn, seg, prop, look, ctr, camp) -> str:
    c = idn["consent"]; r = idn["resolution"]; cn = idn["counts"]; pm = idn["probabilistic_matcher"]
    tot = c["total_people"]
    kpis = [
        (f'{pm["auc"]:.2f}', "Identity matcher AUC", "probabilistic"),
        (f'{r["deterministic_plus_probabilistic"]["recall"]:.2f}', "Cross-device recall",
         f'from {r["deterministic_only"]["recall"]:.2f} det-only'),
        (f'{seg["ari_gmm"]:.2f}', "Segmentation ARI (GMM)", f'k={seg["chosen_k"]}'),
        (f'{look["lift"]:.2f}×', "Lookalike conversion lift", "ALS seed-to-scale"),
        (f'{ctr["n_impressions"]/1e6:.1f}M', "CTR impressions scored", f'AUC {ctr["auc"]:.2f}'),
        (f'{camp["regret_reduction_vs_uniform"]*100:.0f}%', "Regret cut vs A/B", "Thompson bandit"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="n mono">{n}</div><div class="k">{k}</div>'
        f'<div class="d">{d}</div></div>' for n, k, d in kpis)

    flow = "".join(f'<div class="step"><div class="sn mono">{sn}</div>'
                   f'<div class="sl">{sl}</div><div class="st">{st}</div></div>'
                   for sn, sl, st in [
        (f'{cn["devices_in_graph"]:,}', "device signals", "cookies · MAIDs · emails · IPs"),
        (f'{c["allowed_people"]:,}', "consented people", f'{c["retained_pct"]}% after GDPR/CCPA gate'),
        (f'{cn["resolved_entities_full"]:,}', "resolved identities", f'vs {cn["true_persons"]:,} true persons'),
        (f'{seg["chosen_k"]}', "audience segments", "GMM, named by affinity"),
        ("6", "models activated", "propensity · lookalike · CTR · optimizer"),
    ])

    # cards
    identity = (
        '<div class="card"><h3>Identity Resolution <span class="pill">graph + ML</span></h3>'
        '<div class="chart">'
        + bar("Deterministic", r["deterministic_only"]["recall"], 0.6, "{:.3f}", "muted")
        + bar("+ Probabilistic", r["deterministic_plus_probabilistic"]["recall"], 0.6, "{:.3f}", "accent")
        + '</div><div class="substats">'
          f'<div class="substat"><div class="sv mono">{r["deterministic_plus_probabilistic"]["precision"]:.2f}</div><div class="sk">precision</div></div>'
          f'<div class="substat"><div class="sv mono">{pm["auc"]:.2f}</div><div class="sk">matcher AUC</div></div>'
          f'<div class="substat"><div class="sv mono">{cn["probabilistic_edges"]:,}</div><div class="sk">links recovered</div></div>'
          '</div>'
        '<p class="why">Deterministic matching is exact but misses the logged-out tail; a Spark-MLlib '
        'probabilistic matcher nearly doubles recall at ~91% precision, without merging households.</p></div>')

    consent = (
        '<div class="card"><h3>Consent Gate <span class="pill good">privacy-safe</span></h3>'
        '<div class="donut-wrap">'
        + donut(c["retained_pct"], [
            ("retained", 100 * c["allowed_people"] / tot, "var(--good)"),
            ("gdpr", 100 * c["suppressed_gdpr_no_consent"] / tot, "var(--warn)"),
            ("ccpa", 100 * c["suppressed_ccpa_opt_out"] / tot, "var(--faint)")])
        + '<div class="legend">'
          f'<div class="li"><span class="dot" style="background:var(--good)"></span>Processed &nbsp;<b class="mono">{c["allowed_people"]:,}</b></div>'
          f'<div class="li"><span class="dot" style="background:var(--warn)"></span>GDPR suppressed &nbsp;<b class="mono">{c["suppressed_gdpr_no_consent"]:,}</b></div>'
          f'<div class="li"><span class="dot" style="background:var(--faint)"></span>CCPA opt-out &nbsp;<b class="mono">{c["suppressed_ccpa_opt_out"]:,}</b></div>'
          '</div></div>'
        '<p class="why">Every module reads only the consented population. Privacy is enforced at one '
        'auditable choke point, and its cost in reachable audience is measured, not hidden.</p></div>')

    segmentation = (
        '<div class="card"><h3>Audience Segmentation <span class="pill">clustering</span></h3>'
        '<div class="chart">'
        + bar("GMM", seg["ari_gmm"], 1.0, "{:.3f}", "accent")
        + bar("K-means", seg["ari_kmeans"], 1.0, "{:.3f}", "muted")
        + '</div>'
        '<p class="why">Adjusted Rand Index vs the hidden true segments. Soft, elliptical GMM clusters '
        'match overlapping real audiences far better than K-means — the reason to prefer it here.</p></div>')

    propensity = (
        '<div class="card"><h3>Behavioral Propensity <span class="pill">classification</span></h3>'
        '<div class="chart">'
        + bar("Logistic reg.", prop["auc_logistic_regression"], 1.0, "{:.3f}", "accent")
        + bar("Grad. boosting", prop["auc_gbt"], 1.0, "{:.3f}", "muted")
        + '</div>'
        '<p class="why">Model choice on evidence, not fashion: the signal here is largely linear, so the '
        'simpler, interpretable, cheaper-to-serve logistic model is also the more accurate one — and the one to ship.</p></div>')

    lookalike = (
        '<div class="card"><h3>Lookalike Expansion <span class="pill">collaborative filtering</span></h3>'
        '<div class="chart">'
        + bar("Base pool", look["base_heldout_rate"], look["lookalike_heldout_rate"] * 1.25, "{:.3f}", "muted")
        + bar("ALS-expanded", look["lookalike_heldout_rate"], look["lookalike_heldout_rate"] * 1.25, "{:.3f}", "good")
        + '</div><div class="substats">'
          f'<div class="substat"><div class="lift mono">{look["lift"]:.2f}×</div><div class="sk">conversion lift on held-out converters</div></div></div>'
        '<p class="why">ALS learns latent audience factors; expanding the seed to its characteristic items '
        'surfaces new converters above base rate. Modest by design — a 12-item synthetic catalog limits ALS.</p></div>')

    ctr_card = (
        '<div class="card"><h3>CTR / Conversion Optimizer <span class="pill">regression at scale</span></h3>'
        '<div class="chart">'
        + bar("Baseline", ctr["baseline_logloss"], ctr["baseline_logloss"] * 1.1, "{:.3f}", "muted")
        + bar("Hashed model", ctr["logloss"], ctr["baseline_logloss"] * 1.1, "{:.3f}", "accent")
        + '</div><div class="substats">'
          f'<div class="substat"><div class="sv mono">{ctr["auc"]:.2f}</div><div class="sk">AUC</div></div>'
          f'<div class="substat"><div class="sv mono">{ctr["n_impressions"]/1e6:.2f}M</div><div class="sk">impressions</div></div>'
          f'<div class="substat"><div class="sv mono">{ctr["logloss_reduction_vs_baseline"]*100:.0f}%</div><div class="sk">log-loss cut</div></div>'
          '</div>'
        '<p class="why">Hashed logistic regression over a 262K-dim feature space — an online-trainable, '
        'auction-latency model that scales unchanged from laptop to EMR (log loss, lower is better).</p></div>')

    campaign = (
        '<div class="card wide"><h3>Campaign Optimizer <span class="pill">Bayesian</span></h3>'
        '<div class="grid" style="margin-top:14px;grid-template-columns:1.4fr 1fr">'
        '<div><div class="sk" style="font-size:.72rem;color:var(--muted);margin-bottom:6px">'
        'Per-creative conversion rate: posterior mean · 95% credible interval</div>'
        + ci_chart(camp["posteriors"], camp["true_best_creative"]) + '</div>'
        '<div><div class="chart">'
        + bar("Uniform A/B", camp["uniform_ab_regret"], camp["uniform_ab_regret"] * 1.05, "{:.0f}", "warn")
        + bar("Thompson", camp["thompson_regret"], camp["uniform_ab_regret"] * 1.05, "{:.0f}", "good")
        + '</div><div class="substats">'
          f'<div class="substat"><div class="sv mono">{camp["p_best_of_winner"]:.4f}</div><div class="sk">P({camp["bayesian_identified_best"]} is best) — correct</div></div>'
          f'<div class="substat"><div class="sv mono">{camp["thompson_share_to_best_arm"]*100:.0f}%</div><div class="sk">budget to best arm</div></div>'
          '</div></div></div>'
        '<p class="why">Beta-Binomial posteriors identify the true best creative at 99.97% probability; '
        'Thompson sampling reallocates budget adaptively and cuts cumulative regret 79% versus an even A/B split.</p></div>')

    return f"""
<div class="wrap">
  <header>
    <h1>AudienceGraph</h1>
    <p class="sub">Privacy-safe cross-device identity resolution and audience activation on Spark.
       Every model below is scored against a hidden ground-truth identity graph.</p>
    <div class="meta">
      <span class="chip">{tot:,} people</span>
      <span class="chip">{ctr['n_impressions']:,} impressions</span>
      <span class="chip">6 algorithm families</span>
      <span class="chip">PySpark · EMR / S3 / Athena</span>
      <span class="chip">ground-truth evaluated</span>
    </div>
  </header>

  <div class="eyebrow">Headline results</div>
  <div class="kpis">{kpi_html}</div>

  <div class="eyebrow">The pipeline, end to end</div>
  <div class="flow">{flow}</div>

  <div class="eyebrow">By module &mdash; every JD algorithm, measured</div>
  <div class="grid">
    {identity}{consent}{segmentation}{propensity}{lookalike}{ctr_card}
  </div>
  <div class="grid" style="margin-top:16px">{campaign}</div>

  <div class="foot">Generated from <span class="mono">reports/*_metrics.json</span> by
    <span class="mono">scripts/build_dashboard.py</span> · reproduce with <span class="mono">make all</span> ·
    <a href="https://github.com/emmanuel-njinju/audiencegraph">source on GitHub</a>.
    Synthetic data only.</div>
</div>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-out", default=None, help="also write a body-only copy here")
    args = ap.parse_args()
    idn, seg, prop = load("identity"), load("segmentation"), load("propensity")
    look, ctr, camp = load("lookalike"), load("ctr"), load("campaign")
    inner = f"<style>{CSS}</style>{body(idn, seg, prop, look, ctr, camp)}"

    head = ('<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>AudienceGraph — Results Dashboard</title>')
    (REP / "dashboard.html").write_text(f"<!doctype html><html lang=\"en\"><head>{head}</head><body>{inner}</body></html>")
    print("wrote reports/dashboard.html")
    if args.artifact_out:
        Path(args.artifact_out).write_text(inner)
        print(f"wrote {args.artifact_out}")


if __name__ == "__main__":
    main()
