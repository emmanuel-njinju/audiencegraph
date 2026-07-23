"""Spark session factory - identical code path for laptop and AWS EMR.

Locally we run `local[*]`; on EMR the same job is `spark-submit`-ed and Spark
reads the master from the cluster, so the module code never changes between
a laptop test and a 100M-row cluster run. That "write once, scale out" property
is the whole point of building on Spark rather than pandas.
"""
from __future__ import annotations

import os

from pyspark.sql import SparkSession


def get_spark(app_name: str = "audiencegraph", shuffle_partitions: int | None = None) -> SparkSession:
    master = os.environ.get("AG_SPARK_MASTER", "local[*]")
    local = master.startswith("local")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        # Arrow speeds up the pandas <-> Spark hops the eval steps use.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.enabled", os.environ.get("AG_SPARK_UI", "false"))
    )
    if local:
        # Local mode runs a single driver JVM; give it real heap so the iterative
        # graph joins and MLlib training have room. (Set reliably via the env var
        # AG_DRIVER_MEM -> PYSPARK_SUBMIT_ARGS in run scripts; this config is a
        # backstop.) Broadcast joins stay ON - with adequate heap they are the
        # fast path for the small dimension tables here.
        builder = builder.config("spark.driver.memory", os.environ.get("AG_DRIVER_MEM", "4g"))
    # On a laptop, too many shuffle partitions is pure overhead; on EMR let the
    # cluster default (or an explicit override) stand.
    parts = shuffle_partitions or (8 if master.startswith("local") else None)
    if parts:
        builder = builder.config("spark.sql.shuffle.partitions", str(parts))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
