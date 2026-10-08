# f04_features/feature_B_graph_tester_A.py
# Date reviewed:
#    1405/05/24-15:03 --> run rusul is OK for 28 tests

# Run:
#   pytest -v -s f04_features/feature_B_graph_tester_A.py

from __future__ import annotations
import pytest
from f04_features.feature_B_graph import FeatureGraph, FeatureNode

# ------------------------------------------------------------------- 1
def _specs() -> list[str]:
    return [
        'sma(column="close",period=10)@M1',
        'rsi(column="close",period=14)@M1',
        'macd(fast=12,slow=26,signal=9)@H1',
        'bollinger_bands(period=20,std=2.0)@H1',
    ]

# ------------------------------------------------------------------- 2
def test_graph_requires_specs() -> None:
    with pytest.raises(ValueError, match="specs is required"):
        FeatureGraph(None)

# ------------------------------------------------------------------- 3
def test_graph_builds_nodes_from_valid_specs() -> None:
    graph = FeatureGraph(_specs())
    assert len(graph) == 4
    assert all(isinstance(node, FeatureNode) for node in graph.all_nodes())
    assert [node.name for node in graph.all_nodes()] == [
        "sma", "rsi", "macd", "bollinger_bands"
    ]

# ------------------------------------------------------------------- 4
def test_graph_preserves_specification_order() -> None:
    graph = FeatureGraph(_specs())
    assert [node.raw for node in graph.all_nodes()] == _specs()

# ------------------------------------------------------------------- 5
def test_graph_node_identity_uses_canonical_spec() -> None:
    graph = FeatureGraph([
        "sma(column='close',period=10)@M1",
        'sma(column="close",period=10)@M1',
    ])
    assert len(graph) == 1
    node = graph.all_nodes()[0]
    assert node.id == node.canonical
    assert node.name == "sma"
    assert node.timeframe == "M1"

# ------------------------------------------------------------------- 6
def test_graph_rejects_invalid_spec() -> None:
    with pytest.raises(ValueError, match="Invalid feature specification"):
        FeatureGraph(["this is not a valid spec"])

# ------------------------------------------------------------------- 7
def test_graph_rejects_non_string_spec() -> None:
    with pytest.raises(TypeError, match="non-empty string"):
        FeatureGraph([123])

# ------------------------------------------------------------------- 8
def test_feature_node_kwargs_dict() -> None:
    node = FeatureGraph(
        ['sma(column="close",period=10)@M1']
    ).all_nodes()[0]
    assert node.kwargs_dict["column"] == "close"
    assert node.kwargs_dict["period"] == 10

# ------------------------------------------------------------------- 9
def test_feature_node_key_equals_id() -> None:
    node = FeatureGraph(
        ['sma(column="close",period=10)@M1']
    ).all_nodes()[0]
    assert node.key == node.id

# ------------------------------------------------------------------- 10
def test_get_by_timeframe() -> None:
    graph = FeatureGraph(_specs())
    assert [n.name for n in graph.get_by_timeframe("m1")] == ["sma", "rsi"]
    assert [n.name for n in graph.get_by_timeframe("H1")] == [
        "macd", "bollinger_bands"
    ]

# ------------------------------------------------------------------- 11
def test_nodes_for_timeframe_alias() -> None:
    graph = FeatureGraph(_specs())
    assert graph.nodes_for_timeframe("M1") == graph.get_by_timeframe("M1")

# ------------------------------------------------------------------- 12
def test_get_by_name() -> None:
    graph = FeatureGraph(_specs())
    nodes = graph.get_by_name("macd")
    assert len(nodes) == 1
    assert nodes[0].timeframe == "H1"

# ------------------------------------------------------------------- 13
def test_timeframes() -> None:
    assert FeatureGraph(_specs()).timeframes() == ["M1", "H1"]

# ------------------------------------------------------------------- 14
def test_names() -> None:
    assert FeatureGraph(_specs()).names() == [
        "sma", "rsi", "macd", "bollinger_bands"
    ]

# ------------------------------------------------------------------- 15
def test_initial_graph_has_no_inferred_edges() -> None:
    graph = FeatureGraph(_specs())
    assert graph.edges() == []
    for node in graph.all_nodes():
        assert graph.dependencies(node) == []
        assert graph.dependents(node) == []

# ------------------------------------------------------------------- 16
def test_add_dependency() -> None:
    graph = FeatureGraph([
        'ema(column="close",period=10)@M1',
        'rsi(column="close",period=14)@M1',
    ])
    ema = graph.get('ema(column="close",period=10)@M1')
    rsi = graph.get('rsi(column="close",period=14)@M1')

    graph.add_dependency(ema, rsi)

    assert graph.dependencies(rsi) == [ema]
    assert graph.dependents(ema) == [rsi]
    assert graph.edges() == [(ema, rsi)]

# ------------------------------------------------------------------- 17
def test_add_dependency_accepts_node_ids() -> None:
    graph = FeatureGraph([
        'ema(column="close",period=10)@M1',
        'rsi(column="close",period=14)@M1',
    ])
    ema_id = 'ema(column="close",period=10)@M1'
    rsi_id = 'rsi(column="close",period=14)@M1'

    graph.add_dependency(ema_id, rsi_id)

    assert graph.get(ema_id) in graph.dependencies(rsi_id)
    assert graph.get(rsi_id) in graph.dependents(ema_id)

# ------------------------------------------------------------------- 18
def test_self_dependency_is_rejected() -> None:
    graph = FeatureGraph(['sma(column="close",period=10)@M1'])
    node_id = 'sma(column="close",period=10)@M1'

    with pytest.raises(ValueError, match="cannot depend on itself"):
        graph.add_dependency(node_id, node_id)

# ------------------------------------------------------------------- 19
def test_unknown_dependency_node_is_rejected() -> None:
    graph = FeatureGraph(['sma(column="close",period=10)@M1'])

    with pytest.raises(KeyError, match="Unknown feature node"):
        graph.add_dependency(
            "missing",
            'sma(column="close",period=10)@M1',
        )

# ------------------------------------------------------------------- 20
def test_execution_order_without_dependencies_matches_spec_order() -> None:
    graph = FeatureGraph(_specs())
    assert graph.execution_order() == graph.all_nodes()

# ------------------------------------------------------------------- 21
def test_execution_order_respects_dependency() -> None:
    graph = FeatureGraph([
        'rsi(column="close",period=14)@M1',
        'ema(column="close",period=10)@M1',
    ])
    rsi = graph.get('rsi(column="close",period=14)@M1')
    ema = graph.get('ema(column="close",period=10)@M1')

    graph.add_dependency(ema, rsi)

    order = graph.execution_order()
    assert order.index(ema) < order.index(rsi)

# ------------------------------------------------------------------- 22
def test_cycle_is_rejected() -> None:
    graph = FeatureGraph([
        'ema(column="close",period=10)@M1',
        'rsi(column="close",period=14)@M1',
        'sma(column="close",period=20)@M1',
    ])

    ema = 'ema(column="close",period=10)@M1'
    rsi = 'rsi(column="close",period=14)@M1'
    sma = 'sma(column="close",period=20)@M1'

    graph.add_dependency(ema, rsi)
    graph.add_dependency(rsi, sma)

    with pytest.raises(ValueError, match="dependency cycle"):
        graph.add_dependency(sma, ema)

# ------------------------------------------------------------------- 23
def test_get_returns_node_by_canonical_id() -> None:
    graph = FeatureGraph(_specs())
    node = graph.get('sma(column="close",period=10)@M1')
    assert node.name == "sma"

# ------------------------------------------------------------------- 24
def test_get_unknown_node_raises() -> None:
    with pytest.raises(KeyError, match="Unknown feature node"):
        FeatureGraph(_specs()).get("missing")

# ------------------------------------------------------------------- 25
def test_roots_and_leaves_without_edges() -> None:
    graph = FeatureGraph(_specs())
    assert graph.roots() == graph.all_nodes()
    assert graph.leaves() == graph.all_nodes()

# ------------------------------------------------------------------- 26
def test_roots_and_leaves_with_dependency() -> None:
    graph = FeatureGraph([
        'ema(column="close",period=10)@M1',
        'rsi(column="close",period=14)@M1',
    ])
    ema = graph.get('ema(column="close",period=10)@M1')
    rsi = graph.get('rsi(column="close",period=14)@M1')

    graph.add_dependency(ema, rsi)

    assert graph.roots() == [ema]
    assert graph.leaves() == [rsi]

# ------------------------------------------------------------------- 27
def test_groups_by_timeframe_returns_copy() -> None:
    graph = FeatureGraph(_specs())
    groups = graph.groups_by_timeframe()

    assert set(groups.keys()) == {"M1", "H1"}
    assert len(groups["M1"]) == 2
    assert len(groups["H1"]) == 2

    groups["M1"].clear()
    assert len(graph.get_by_timeframe("M1")) == 2

# ------------------------------------------------------------------- 28
def test_info() -> None:
    graph = FeatureGraph(_specs())
    info = graph.info()

    assert info["node_count"] == 4
    assert info["edge_count"] == 0
    assert info["timeframes"] == ["M1", "H1"]
    assert info["indicator_names"] == [
        "sma", "rsi", "macd", "bollinger_bands"
    ]

# ------------------------------------------------------------------- 29
def test_graph_exposes_observation_builder_contract_data() -> None:
    graph = FeatureGraph([
        'macd(fast=12,slow=26,signal=9)@H1',
    ])
    node = graph.all_nodes()[0]

    assert node.name == "macd"
    assert node.timeframe == "H1"
    assert node.canonical == 'macd(fast=12,slow=26,signal=9)@H1'
    assert node.canonical in [n.canonical for n in graph.execution_order()]

# ------------------------------------------------------------------- END