"""The consent gate - the first step of every pipeline in this repo.

Privacy-safe is not a slogan here: no device signal, profile, or model feature
derived from a person who has not consented (GDPR) or who has opted out (CCPA)
is allowed downstream. This module is the single choke point that enforces it,
so every module inherits the guarantee by construction rather than by remembering
to filter.

The gate is deliberately auditable: `apply_consent` returns both the allowed
population and a small suppression report (how many people/devices were dropped
and why), which the pipelines log and the README quotes. Privacy has a real
cost in reachable audience, and a serious platform measures it.
"""
from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame, functions as F


@dataclass
class ConsentReport:
    total_people: int
    allowed_people: int
    suppressed_gdpr: int      # EU, consent not given
    suppressed_ccpa: int      # US, opted out

    @property
    def retained_pct(self) -> float:
        return 100.0 * self.allowed_people / max(1, self.total_people)

    def as_dict(self) -> dict:
        return {
            "total_people": self.total_people,
            "allowed_people": self.allowed_people,
            "retained_pct": round(self.retained_pct, 2),
            "suppressed_gdpr_no_consent": self.suppressed_gdpr,
            "suppressed_ccpa_opt_out": self.suppressed_ccpa,
        }


def consent_report(persons: DataFrame) -> ConsentReport:
    agg = persons.agg(
        F.count("*").alias("total"),
        F.sum(F.col("processing_allowed").cast("int")).alias("allowed"),
        F.sum(((F.col("region") == "EU") & (~F.col("gdpr_consent"))).cast("int")).alias("gdpr"),
        F.sum(F.col("ccpa_opt_out").cast("int")).alias("ccpa"),
    ).first()
    return ConsentReport(int(agg["total"]), int(agg["allowed"]),
                         int(agg["gdpr"]), int(agg["ccpa"]))


def allowed_person_ids(persons: DataFrame) -> DataFrame:
    """The consented population - the only people any model is allowed to see."""
    return persons.filter(F.col("processing_allowed")).select("person_id")


def gate_devices(devices: DataFrame, persons: DataFrame) -> DataFrame:
    """Restrict a device table to devices owned by consented people."""
    return devices.join(allowed_person_ids(persons), on="person_id", how="inner")
