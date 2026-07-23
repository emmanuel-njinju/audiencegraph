"""Inject reports/identity_metrics.json into README.md and the model card.

Keeps the documented numbers honest and reproducible: run the pipeline, then run
this, and the docs reflect the exact metrics of the committed run - no hand-typed
figures that can drift from the code.

Usage:  python scripts/update_docs_metrics.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "identity_metrics.json"


def _replace(path: Path, tag: str, block: str) -> None:
    text = path.read_text()
    new = re.sub(
        rf"<!-- {tag}:START -->.*?<!-- {tag}:END -->",
        f"<!-- {tag}:START -->\n{block}\n<!-- {tag}:END -->",
        text,
        flags=re.DOTALL,
    )
    path.write_text(new)
    print(f"updated {path.relative_to(ROOT)} [{tag}]")


def readme_table(m: dict) -> str:
    c, pm = m["consent"], m["probabilistic_matcher"]
    d0 = m["resolution"]["deterministic_only"]
    d1 = m["resolution"]["deterministic_plus_probabilistic"]
    cnt = m["counts"]
    return "\n".join([
        f"**Scale:** {cnt['true_persons']:,} consented persons · "
        f"{cnt['devices_in_graph']:,} devices in graph "
        f"(from {c['total_people']:,} people; consent gate retained {c['retained_pct']}%).",
        "",
        "| Stage | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
        f"| Deterministic only | {d0['precision']} | {d0['recall']} | {d0['f1']} |",
        f"| **Deterministic + probabilistic** | **{d1['precision']}** | **{d1['recall']}** | **{d1['f1']}** |",
        "",
        f"- **Probabilistic matcher:** AUC **{pm['auc']}**, operated at threshold "
        f"{pm['operating_threshold']} (chosen for >=90% precision) over {pm['n_pairs']:,} candidate pairs.",
        f"- **F1 lift from the probabilistic step:** +{m['resolution']['f1_lift_from_probabilistic']} "
        f"at {d1['precision']} precision (recovers true links without over-merging households).",
        f"- **Consent cost:** {c['suppressed_gdpr_no_consent']:,} suppressed for GDPR (no consent), "
        f"{c['suppressed_ccpa_opt_out']:,} for CCPA (opt-out).",
    ])


def card_table(m: dict) -> str:
    pm = m["probabilistic_matcher"]
    d0 = m["resolution"]["deterministic_only"]
    d1 = m["resolution"]["deterministic_plus_probabilistic"]
    return "\n".join([
        f"- **Matcher AUC:** {pm['auc']} (held-out).",
        f"- **Operating threshold:** {pm['operating_threshold']} (precision-floor 0.90).",
        f"- **Graph, deterministic only:** P {d0['precision']} / R {d0['recall']} / F1 {d0['f1']}.",
        f"- **Graph, deterministic + probabilistic:** P {d1['precision']} / R {d1['recall']} / F1 {d1['f1']}.",
    ])


def main() -> None:
    m = json.loads(METRICS.read_text())
    _replace(ROOT / "README.md", "METRICS", readme_table(m))
    _replace(ROOT / "reports" / "identity_model_card.md", "CARD_METRICS", card_table(m))


if __name__ == "__main__":
    main()
