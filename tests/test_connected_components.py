from src.common.connected_components import connected_components


def test_two_components_and_isolated_node(spark):
    # a-b-c form one chain, d-e another, f is isolated.
    nodes = spark.createDataFrame([("a",), ("b",), ("c",), ("d",), ("e",), ("f",)], ["id"])
    edges = spark.createDataFrame([("a", "b"), ("b", "c"), ("d", "e")], ["src", "dst"])

    comp = {r["id"]: r["component"] for r in connected_components(nodes, edges).collect()}

    assert comp["a"] == comp["b"] == comp["c"]          # chain collapses to one component
    assert comp["d"] == comp["e"]
    assert comp["a"] != comp["d"]                        # distinct components stay distinct
    assert comp["f"] not in {comp["a"], comp["d"]}       # isolated node is its own component
    assert len({comp["a"], comp["d"], comp["f"]}) == 3   # exactly three components


def test_direction_is_ignored(spark):
    # Edges given one-way must still connect (algorithm is undirected).
    nodes = spark.createDataFrame([("x",), ("y",), ("z",)], ["id"])
    edges = spark.createDataFrame([("y", "x"), ("z", "y")], ["src", "dst"])
    comp = {r["id"]: r["component"] for r in connected_components(nodes, edges).collect()}
    assert comp["x"] == comp["y"] == comp["z"]
