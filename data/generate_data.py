"""
Synthetic consumer-universe generator for the Audience Intelligence Platform.

One coherent generative model feeds every module in the platform, and because
the model is generative we keep the GROUND TRUTH for each task. That lets every
module report a real, honest metric (F1 for identity resolution, silhouette /
ARI for segmentation, AUC for propensity and CTR, recall@k for lookalike,
regret for the bandit) instead of an unfalsifiable "looks reasonable".

The universe, in one paragraph:
    A population of PEOPLE each carry a latent interest vector (a Dirichlet mix
    over K topics) and latent demographics. Interests place each person in a
    ground-truth SEGMENT (module 2) and drive a true CONVERSION PROPENSITY
    (module 3). Each person owns 1-4 DEVICES; devices emit shared SIGNALS
    (a person-level hashed email, a household-level IP, a device-level cookie)
    from which the identity graph must reconstruct the person (module 1). Every
    ad IMPRESSION's click/convert probability is a logistic function of
    interest-content match, creative quality and slot position, giving a true
    CTR surface (module 4). Person x content engagement forms the interaction
    matrix for lookalike expansion (module 5). Campaign creatives carry true
    conversion lifts that the Bayesian optimizer must recover from a reward
    stream (module 6).

Everything is seeded and deterministic. Scale with --people; the defaults run
on a laptop in Spark local mode, and the same code scales on EMR by pointing
--out at an s3:// prefix and raising --people.

Usage:
    python data/generate_data.py --people 20000 --out data/synthetic --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

# ----------------------------- configuration ------------------------------- #

N_TOPICS = 12          # latent interest topics (content categories share this space)
N_SEGMENTS = 8         # ground-truth audience segments
CONTENT_CATS = [f"cat_{i:02d}" for i in range(N_TOPICS)]
DEVICE_TYPES = ["mobile", "desktop", "tablet", "ctv"]
CAMPAIGNS = [f"cmp_{i}" for i in range(6)]
CREATIVES_PER_CAMPAIGN = 4
SITES = [f"pub_{i:02d}" for i in range(25)]


def _sha(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


# ------------------------------- people ------------------------------------ #

def make_people(rng: np.random.Generator, n_people: int) -> tuple[pd.DataFrame, np.ndarray]:
    """People with latent interests, a ground-truth segment, and true propensity."""
    # Segment "prototypes": each segment is a concentration over topics. People
    # are drawn near their segment prototype so segments are recoverable but
    # overlapping (realistic, not linearly separable).
    proto = rng.dirichlet(np.ones(N_TOPICS) * 0.6, size=N_SEGMENTS)
    seg = rng.integers(0, N_SEGMENTS, size=n_people)
    # interest = prototype nudged by per-person noise, renormalised.
    interest = proto[seg] * rng.gamma(shape=6.0, scale=1.0, size=(n_people, N_TOPICS))
    interest = interest / interest.sum(axis=1, keepdims=True)

    # Latent demographics used by the propensity model.
    age = np.clip(rng.normal(38, 12, n_people), 18, 85)
    income = np.clip(rng.normal(0, 1, n_people), -3, 3)          # standardized
    recency = rng.exponential(1.0, n_people)                     # days since last seen

    # True conversion propensity. Driven mainly by affinity to two high-intent
    # categories (cat_00, cat_01) - i.e. by INTEREST, which the impression log
    # reflects - so behavioral propensity (module 3) and ALS lookalike (module 5)
    # have real signal to learn. A nonlinear AND-interaction (both interests
    # high) is added so a linear model captures the main effect but gradient
    # boosting wins on the interaction - the story module 3 tells. A first-party
    # attribute (income) still helps, so attributes and behavior both matter.
    hv0, hv1 = interest[:, 0], interest[:, 1]
    both_high = ((hv0 > 0.15) & (hv1 > 0.15)).astype(float)   # nonlinear interaction
    z = (
        -2.4
        + 6.0 * (hv0 + hv1)         # strong interest signal (visible in behavior)
        + 1.6 * both_high           # nonlinearity -> GBT > logistic
        + 0.4 * income              # a first-party attribute still contributes
        - 0.15 * recency
    )
    p_convert = 1.0 / (1.0 + np.exp(-z))
    converted = (rng.random(n_people) < p_convert).astype(int)

    # Consent / privacy state. Lotame's platform is privacy-safe by design, so
    # the universe carries the regulatory signals a real DMP must honor:
    #   region        - drives which regime applies (GDPR in EU, CCPA in US).
    #   gdpr_consent  - EU users default to NO consent until given (opt-in).
    #   ccpa_opt_out  - US users default opted-in but may opt OUT (opt-out).
    # processing_allowed is the single gate every downstream module respects:
    # opted-out / non-consented people are dropped BEFORE any identity or model
    # work happens (see src/common/consent.py). This makes "privacy-safe" a
    # demonstrated behavior with a measurable cost (population retained), not a
    # marketing line.
    region = rng.choice(["EU", "US", "OTHER"], size=n_people, p=[0.35, 0.5, 0.15])
    gdpr_consent = np.where(region == "EU", rng.random(n_people) < 0.72, True)
    ccpa_opt_out = np.where(region == "US", rng.random(n_people) < 0.12, False)
    processing_allowed = gdpr_consent & (~ccpa_opt_out)

    people = pd.DataFrame({
        "person_id": np.arange(n_people),
        "true_segment": seg,
        "age": age.round(1),
        "income_z": income.round(3),
        "recency_days": recency.round(3),
        "propensity_true": p_convert.round(4),
        "converted_true": converted,
        "region": region,
        "gdpr_consent": gdpr_consent,
        "ccpa_opt_out": ccpa_opt_out,
        "processing_allowed": processing_allowed,
    })
    for k in range(N_TOPICS):
        people[f"interest_{k:02d}"] = interest[:, k].round(4)
    return people, interest


# ------------------------------ devices ------------------------------------ #

def make_devices(rng: np.random.Generator, people: pd.DataFrame) -> pd.DataFrame:
    """1-4 devices per person; devices are the pre-identity unit of observation."""
    rows = []
    for pid in people["person_id"].to_numpy():
        n_dev = rng.integers(1, 5)
        for _ in range(n_dev):
            rows.append((f"dev_{_sha(pid, rng.integers(1<<30))}", int(pid),
                         DEVICE_TYPES[rng.integers(0, len(DEVICE_TYPES))]))
    return pd.DataFrame(rows, columns=["device_id", "person_id", "device_type"])


def make_identity_signals(rng: np.random.Generator, devices: pd.DataFrame,
                          people: pd.DataFrame) -> pd.DataFrame:
    """
    Emit the shared signals the identity graph will stitch on.

      - hashed_email : PERSON-level, strong (present ~65% of devices). Two devices
                       sharing a hashed_email are almost surely one person.
      - ip           : HOUSEHOLD-level, noisy. Devices of the same person share an
                       IP, but so do different people in the same household -> this
                       is the over-merge trap the graph must handle with edge
                       weighting / confidence.
      - cookie       : DEVICE-level, unique (useful as a node key, not for merging).
    """
    n_people = len(people)
    person_email = {pid: _sha("email", pid) for pid in people["person_id"]}
    # Assign people to households (~1.8 people/household) to create shared IPs.
    household = rng.integers(0, max(1, int(n_people / 1.8)), size=n_people)
    household_ip = {h: f"ip_{_sha('ip', h)}" for h in np.unique(household)}
    person_household = dict(zip(people["person_id"], household))

    dev_type = dict(zip(devices["device_id"], devices["device_type"]))
    rows = []
    for did, pid in zip(devices["device_id"], devices["person_id"]):
        rows.append((did, "cookie", _sha("ck", did)))
        # Only ~50% of devices ever expose a hashed email (the logged-out reality
        # of the open web). Deterministic matching therefore leaves a large tail
        # that probabilistic matching must recover - which is the whole point of
        # having both.
        if rng.random() < 0.50:
            rows.append((did, "hashed_email", person_email[pid]))
        # MAID (mobile advertising id): device-level, present on non-desktop
        # devices, and resettable - ~10% present as a freshly reset id that
        # links to nothing (the churn a real graph must tolerate).
        if dev_type[did] != "desktop":
            maid = _sha("maid", did) if rng.random() < 0.90 else _sha("maid-reset", did, rng.integers(1<<30))
            rows.append((did, "maid", maid))
        # ~85% of devices expose the household IP; a small share use a transient
        # mobile IP that appears on no other device (dangling, realistic).
        if rng.random() < 0.85:
            rows.append((did, "ip", household_ip[person_household[pid]]))
        else:
            rows.append((did, "ip", f"ip_{_sha('transient', did)}"))
    return pd.DataFrame(rows, columns=["device_id", "signal_type", "signal_value"])


# ------------------------------- events ------------------------------------ #

def make_events(rng: np.random.Generator, people: pd.DataFrame, devices: pd.DataFrame,
                interest: np.ndarray, avg_events: int = 40) -> pd.DataFrame:
    """
    Ad impression log with a true click/convert surface.

    click prob = sigma( interest-content match + creative quality - position cost )
    convert prob = sigma( .. | click ) scaled by the person's latent propensity.
    """
    creative_quality = {(c, k): rng.normal(0, 0.6)
                        for c in CAMPAIGNS for k in range(CREATIVES_PER_CAMPAIGN)}
    dev_person = dict(zip(devices["device_id"], devices["person_id"]))
    dev_ids = devices["device_id"].to_numpy()
    prop = people.set_index("person_id")["propensity_true"].to_dict()

    n_events = int(len(devices) * avg_events)
    pick_dev = dev_ids[rng.integers(0, len(dev_ids), size=n_events)]
    rows = []
    for did in pick_dev:
        pid = dev_person[did]
        # Targeted delivery: 70% of impressions are drawn from the person's
        # interest distribution, 30% exploration (uniform). This is what makes a
        # device's impression mix a behavioral fingerprint of its owner - the
        # signal the probabilistic identity matcher (module 1) learns from, and
        # the interest evidence segmentation (module 2) clusters on.
        if rng.random() < 0.70:
            cat_idx = int(rng.choice(N_TOPICS, p=interest[pid]))
        else:
            cat_idx = int(rng.integers(0, N_TOPICS))
        campaign = CAMPAIGNS[rng.integers(0, len(CAMPAIGNS))]
        creative = int(rng.integers(0, CREATIVES_PER_CAMPAIGN))
        position = int(rng.integers(0, 5))              # 0 = top slot
        bid = float(np.round(rng.gamma(2.0, 0.6), 3))
        match = interest[pid, cat_idx]                  # how much they care
        zc = -2.2 + 5.0 * match + creative_quality[(campaign, creative)] - 0.35 * position
        p_click = 1.0 / (1.0 + np.exp(-zc))
        click = int(rng.random() < p_click)
        convert = int(click and (rng.random() < 0.12 * prop[pid] * (1 + 2 * match)))
        rows.append((did, CONTENT_CATS[cat_idx], campaign, creative, position,
                     bid, click, convert))
    ev = pd.DataFrame(rows, columns=["device_id", "content_cat", "campaign",
                                     "creative", "position", "bid", "click", "convert"])
    ev.insert(0, "event_id", np.arange(len(ev)))
    return ev


def make_interactions(events: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    """Person x content engagement matrix for collaborative filtering (module 5)."""
    ev = events.merge(devices[["device_id", "person_id"]], on="device_id")
    inter = (ev.groupby(["person_id", "content_cat"])
               .agg(weight=("click", "sum"), impressions=("event_id", "count"))
               .reset_index())
    # implicit-feedback weight: clicks matter more than mere impressions.
    inter["weight"] = inter["weight"] * 3 + inter["impressions"]
    return inter[inter["weight"] > 0]


def make_campaign_stream(rng: np.random.Generator, n_rounds: int = 40000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    A single campaign's creatives with TRUE conversion rates plus a reward
    stream, for the Bayesian A/B test and Thompson-sampling bandit (module 6).
    """
    n_arms = 6
    true_cr = np.round(rng.uniform(0.02, 0.09, n_arms), 4)   # ground truth
    truth = pd.DataFrame({"creative": [f"cr_{i}" for i in range(n_arms)],
                          "true_conv_rate": true_cr})
    # Uniform exploration stream (an A/B test allocates equally); the bandit
    # module re-simulates its own adaptive allocation from these Bernoulli means.
    arms = rng.integers(0, n_arms, size=n_rounds)
    reward = (rng.random(n_rounds) < true_cr[arms]).astype(int)
    stream = pd.DataFrame({"round": np.arange(n_rounds),
                           "creative": [f"cr_{a}" for a in arms],
                           "reward": reward})
    return truth, stream


# --------------------------------- main ------------------------------------ #

def _write(df: pd.DataFrame, out: str, name: str) -> None:
    path = f"{out}/{name}.parquet"
    if out.startswith("s3://"):
        df.to_parquet(path, index=False)               # pandas -> s3fs on EMR
    else:
        os.makedirs(out, exist_ok=True)
        df.to_parquet(path, index=False)
    print(f"  wrote {name:20s} {len(df):>10,} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--people", type=int, default=20000,
                    help="population size (drives every downstream table)")
    ap.add_argument("--avg-events", type=int, default=70, help="avg impressions per device")
    ap.add_argument("--out", type=str, default="data/synthetic", help="output prefix (local or s3://)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Generating consumer universe: people={args.people:,} seed={args.seed}")

    people, interest = make_people(rng, args.people)
    devices = make_devices(rng, people)
    signals = make_identity_signals(rng, devices, people)
    events = make_events(rng, people, devices, interest, args.avg_events)
    interactions = make_interactions(events, devices)
    camp_truth, camp_stream = make_campaign_stream(rng)

    for df, name in [(people, "persons"), (devices, "devices"),
                     (signals, "identity_signals"), (events, "events"),
                     (interactions, "interactions"),
                     (camp_truth, "campaign_truth"), (camp_stream, "campaign_stream")]:
        _write(df, args.out, name)

    # A tiny manifest so downstream jobs (and reviewers) know the ground-truth
    # columns without re-reading the generator.
    manifest = pd.DataFrame([
        ("persons", "person_id", "true_segment / converted_true / propensity_true / interest_*"),
        ("devices", "device_id", "person_id  <- identity ground truth"),
        ("identity_signals", "device_id", "signal_type in {cookie,hashed_email,ip}"),
        ("events", "event_id", "click / convert  <- CTR ground truth"),
        ("interactions", "person_id", "content_cat x weight  <- CF input"),
        ("campaign_truth", "creative", "true_conv_rate  <- bandit ground truth"),
        ("campaign_stream", "round", "creative x reward  <- reward stream"),
    ], columns=["table", "key", "notes"])
    _write(manifest, args.out, "_manifest")
    print("Done.")


if __name__ == "__main__":
    main()
