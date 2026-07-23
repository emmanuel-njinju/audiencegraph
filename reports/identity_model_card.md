# Model Card — Probabilistic Identity Matcher (Release 1)

A supervised pairwise classifier that decides whether two devices sharing a
household IP belong to the **same person** or merely the **same household**. It
supplies the probabilistic edges in the identity-resolution graph; deterministic
(shared hashed-email / MAID) edges do not go through this model.

## Intended use
- **In scope:** cross-device identity resolution on consented ad-tech signals,
  operated at a high-precision threshold so household members are not fused.
- **Out of scope:** any decision about a person who has not consented — the
  consent gate removes them before this model runs.

## Training data
- Candidate device pairs blocked on a shared IP (public/transient IP blocks
  larger than `MAX_BLOCK` are dropped), from the synthetic consumer universe.
- Label (same person) comes from the hidden ground-truth graph and is used
  **only** for training/evaluation — never as a model input.

## Features
- `cosine` — cosine similarity of the two devices' L2-normalized content-affinity
  vectors (the behavioral fingerprint). Primary signal.
- `same_type` — whether the two devices are the same device type.

## Model
- Spark MLlib `LogisticRegression`, 70/30 train/test split, `maxIter=50`.
- Operating threshold selected on held-out data for **precision ≥ 0.90**
  (recall is maximized subject to that precision floor).

## Metrics (canonical run — see `identity_metrics.json`)
<!-- CARD_METRICS:START -->
- **Matcher AUC:** 0.9623 (held-out).
- **Operating threshold:** 0.885 (precision-floor 0.90).
- **Graph, deterministic only:** P 1.0 / R 0.2524 / F1 0.4031.
- **Graph, deterministic + probabilistic:** P 0.9144 / R 0.4976 / F1 0.6445.
<!-- CARD_METRICS:END -->

## Ethical & privacy considerations
- Consent is enforced upstream and unconditionally (GDPR opt-in, CCPA opt-out).
- The high-precision operating point is a deliberate ethical choice: an
  over-merge attaches one person's behavior to another (a real privacy harm), so
  the model is tuned to avoid false merges even at the cost of some recall.

## Limitations
- Behavioral fingerprints are weak for low-activity devices; those true links
  are conservatively left unmerged (they fall to deterministic matching or stay
  separate).
- Evaluated on synthetic data with a known generative process; real signals are
  noisier and would warrant recalibration and additional features (temporal
  overlap, geo, publisher co-visitation).
