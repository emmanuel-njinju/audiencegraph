"""Connected components in native PySpark (label propagation to the minimum id).

Why hand-rolled instead of GraphFrames: (1) it removes a fragile external
package pin and runs on any Spark, (2) it demonstrates the graph algorithm
itself, which the JD explicitly asks for ("comprehension of graph algorithms"),
and (3) the identity-resolution components are shallow (a person's few devices
plus household), so label propagation converges in a handful of iterations -
the heavy machinery of GraphFrames buys nothing here.

Each node starts labelled with its own id. On every round a node adopts the
minimum label among itself and its neighbours; the process is monotone and
converges when no label changes. The final label is the component id, i.e. the
resolved-entity id.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, functions as F


def connected_components(nodes: DataFrame, edges: DataFrame, max_iter: int = 30) -> DataFrame:
    """
    nodes: DataFrame[id]                (string or numeric node key)
    edges: DataFrame[src, dst]          (undirected; direction is ignored)
    returns: DataFrame[id, component]   (component = min id in the component)
    """
    # Symmetrise edges once: propagation must flow both ways. localCheckpoint
    # materialises and TRUNCATES lineage - without it, an iterative Spark job
    # rebuilds the whole plan every round and blows up in time and memory.
    e = (edges.select(F.col("src"), F.col("dst"))
              .union(edges.select(F.col("dst").alias("src"), F.col("src").alias("dst")))
              .where(F.col("src") != F.col("dst"))
              .distinct()
              .localCheckpoint(eager=True))

    labels = nodes.select(F.col("id"), F.col("id").alias("component"))

    for _ in range(max_iter):
        # Each node hears the minimum component label of its neighbours.
        incoming = (e.join(labels, e.src == labels.id)
                     .groupBy(e.dst.alias("id"))
                     .agg(F.min("component").alias("neighbor_min")))
        updated = (labels.join(incoming, on="id", how="left")
                        .select(
                            "id",
                            F.least(F.col("component"),
                                    F.coalesce(F.col("neighbor_min"), F.col("component")))
                             .alias("component"))
                        .localCheckpoint(eager=True))     # truncate lineage each round
        # Converged when no node lowered its label this round.
        changed = (updated.join(labels.withColumnRenamed("component", "old"), on="id")
                          .where(F.col("component") != F.col("old"))
                          .limit(1).count())
        labels = updated
        if changed == 0:
            break

    return labels
