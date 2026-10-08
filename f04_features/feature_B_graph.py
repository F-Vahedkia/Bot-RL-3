# f04_features/feature_B_graph.py
# Date reviewed:
#    1405/05/23-16:00 --> run result is OK for 28 tests of tester file ver. _A
"""
FeatureGraph
============

Structural graph of the feature specifications owned by FeaturePipeline.

This module does not calculate features, manage live state, persist datasets,
or build observations.  It only parses the supplied specifications and exposes
stable structural information for downstream consumers such as
ObservationBuilder.

Current contract
----------------
    FeaturePipeline
        -> FeatureGraph(specs)
        -> ObservationBuilder.build(feature_dataset, graph)

The graph is intentionally conservative: the current feature specification
language and Registry describe indicator inputs through OHLCV/source columns,
so there is no invented feature-to-feature dependency semantics here.  The
API nevertheless exposes explicit dependency/edge/topology hooks so that such
relationships can be added later without changing the public graph model.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from f02_utils.functions.parser import ParsedSpec, parse_spec


# =============================================================================
# Node
# =============================================================================
@dataclass(frozen=True, slots=True)
class FeatureNode:
    """One validated feature specification in the graph."""

    id: str
    raw: str
    canonical: str
    name: str
    timeframe: Optional[str]
    args: Tuple[object, ...] = ()
    kwargs: Tuple[Tuple[str, object], ...] = ()

    @property
    def key(self) -> str:
        """Stable node key used by graph maps."""
        return self.id

    @property
    def kwargs_dict(self) -> Dict[str, object]:
        """Return parsed keyword arguments as a normal dictionary."""
        return dict(self.kwargs)


# =============================================================================
# Feature Graph
# =============================================================================
class FeatureGraph:
    """
    Structural representation of the FeaturePipeline specification set.

    Responsibilities
    ----------------
    - Parse and validate the supplied feature specifications.
    - Create one FeatureNode per unique canonical specification.
    - Group nodes by timeframe and indicator name.
    - Maintain explicit directed edges when dependency information is added.
    - Expose a deterministic execution/topological order.

    Non-responsibilities
    --------------------
    - Feature calculation: FeatureEngine.
    - Cache/state: FeatureEngine / FeatureCache.
    - Dataset merge/persistence: FeatureStoreV2.
    - Observation construction: ObservationBuilder.
    - Configuration ownership: FeaturePipeline / higher-level orchestrator.
    """
    # ------------------------------------------------------------------------- 1
    def __init__(self, specs: Sequence[str]) -> None:
        if specs is None:
            raise ValueError("specs is required")

        self.specs: List[str] = list(specs)

        self.nodes: List[FeatureNode] = []
        self.by_tf: Dict[str, List[FeatureNode]] = defaultdict(list)
        self.by_name: Dict[str, List[FeatureNode]] = defaultdict(list)

        # Directed graph: dependency -> dependent node.
        self._edges: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_edges: Dict[str, Set[str]] = defaultdict(set)
        self._node_by_id: Dict[str, FeatureNode] = {}

        self._build()

    # ------------------------------------------------------------------------- 2
    def _build(self) -> None:
        """Parse specs and build deterministic structural indexes."""

        seen: Set[str] = set()

        for raw_spec in self.specs:
            if not isinstance(raw_spec, str) or not raw_spec.strip():
                raise TypeError(
                    f"Feature specification must be a non-empty string, "
                    f"got {raw_spec!r}"
                )

            try:
                parsed = parse_spec(raw_spec)
            except Exception as exc:
                raise ValueError(
                    f"Invalid feature specification {raw_spec!r}: {exc}"
                ) from exc

            node_id = parsed.canonical
            if node_id in seen:
                continue
            seen.add(node_id)

            node = self._make_node(parsed)
            self.nodes.append(node)
            self._node_by_id[node.id] = node

            tf = (node.timeframe or "GLOBAL").upper()
            self.by_tf[tf].append(node)
            self.by_name[node.name].append(node)

        # No feature-to-feature dependencies are inferred here.  The current
        # Parser/Registry contract exposes source columns, not graph edges.
        self._rebuild_empty_edge_maps()

    # ------------------------------------------------------------------------- 3
    @staticmethod
    def _make_node(parsed: ParsedSpec) -> FeatureNode:
        canonical = parsed.canonical
        kwargs_items = tuple(sorted(parsed.kwargs.items(), key=lambda item: item[0]))

        return FeatureNode(
            id=canonical,
            raw=parsed.raw,
            canonical=canonical,
            name=parsed.name,
            timeframe=parsed.timeframe,
            args=tuple(parsed.args),
            kwargs=kwargs_items,
        )

    # ------------------------------------------------------------------------- 4
    def _rebuild_empty_edge_maps(self) -> None:
        """Initialize edge maps for all known nodes."""
        for node in self.nodes:
            self._edges.setdefault(node.id, set())
            self._reverse_edges.setdefault(node.id, set())

    # ------------------------------------------------------------------------- 5
    def add_dependency(self, dependency: str | FeatureNode, dependent: str | FeatureNode) -> None:
        """
        Add an explicit directed dependency edge.

        The edge is ``dependency -> dependent``.
        This method is intentionally explicit; FeatureGraph does not guess
        dependencies from indicator parameter names.
        """

        dependency_id = self._resolve_node_id(dependency)
        dependent_id = self._resolve_node_id(dependent)

        if dependency_id == dependent_id:
            raise ValueError("A feature node cannot depend on itself.")

        self._edges[dependency_id].add(dependent_id)
        self._reverse_edges[dependent_id].add(dependency_id)

        # Validate immediately so a cycle cannot remain hidden.
        self.topological_order()

    # ------------------------------------------------------------------------- 6
    def _resolve_node_id(self, node: str | FeatureNode) -> str:
        if isinstance(node, FeatureNode):
            node_id = node.id
        elif isinstance(node, str):
            node_id = node
        else:
            raise TypeError(
                f"Expected FeatureNode or node id string, got {type(node).__name__}"
            )

        if node_id not in self._node_by_id:
            raise KeyError(f"Unknown feature node: {node_id!r}")

        return node_id

    # ------------------------------------------------------------------------- 7
    def dependencies(self, node: str | FeatureNode) -> List[FeatureNode]:
        """Return direct dependencies of a node in deterministic order."""

        node_id = self._resolve_node_id(node)
        return [
            self._node_by_id[node_id]
            for node_id in sorted(self._reverse_edges[node_id])
        ]

    # ------------------------------------------------------------------------- 8
    def dependents(self, node: str | FeatureNode) -> List[FeatureNode]:
        """Return direct dependents of a node in deterministic order."""

        node_id = self._resolve_node_id(node)
        return [
            self._node_by_id[node_id]
            for node_id in sorted(self._edges[node_id])
        ]

    # ------------------------------------------------------------------------- 9
    def edges(self) -> List[Tuple[FeatureNode, FeatureNode]]:
        """Return all directed edges as ``(dependency, dependent)`` tuples."""

        result: List[Tuple[FeatureNode, FeatureNode]] = []
        for source_id in sorted(self._edges):
            for target_id in sorted(self._edges[source_id]):
                result.append(
                    (self._node_by_id[source_id], self._node_by_id[target_id])
                )
        return result

    # ------------------------------------------------------------------------- 10
    def topological_order(self) -> List[FeatureNode]:
        """
        Return a deterministic topological ordering.

        With today's contracts this is normally the specification order,
        because no dependencies are inferred automatically.
        """

        indegree = {
            node.id: len(self._reverse_edges[node.id])
            for node in self.nodes
        }

        # Keep insertion order for nodes that become ready.  This gives a
        # stable order while still producing a valid topological ordering.
        queue = deque(node.id for node in self.nodes if indegree[node.id] == 0)
        ordered: List[FeatureNode] = []

        while queue:
            current = queue.popleft()
            ordered.append(self._node_by_id[current])

            for dependent in sorted(self._edges[current]):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)

        if len(ordered) != len(self.nodes):
            raise ValueError("FeatureGraph contains a dependency cycle.")

        return ordered

    # ------------------------------------------------------------------------- 11
    def execution_order(self) -> List[FeatureNode]:
        """Alias for the public deterministic topological execution order."""
        return self.topological_order()

    # ------------------------------------------------------------------------- 12
    def get_by_timeframe(self, tf: str) -> List[FeatureNode]:
        """Return nodes assigned to one timeframe."""
        if not isinstance(tf, str):
            raise TypeError("tf must be a string")
        return list(self.by_tf.get(tf.upper(), []))

    # ------------------------------------------------------------------------- 13
    def get_by_name(self, name: str) -> List[FeatureNode]:
        """Return nodes for one indicator name."""
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        return list(self.by_name.get(name, []))

    # ------------------------------------------------------------------------- 14
    def get(self, node_id: str) -> FeatureNode:
        """Return a node by its canonical specification id."""
        try:
            return self._node_by_id[node_id]
        except KeyError as exc:
            raise KeyError(f"Unknown feature node: {node_id!r}") from exc

    # ------------------------------------------------------------------------- 15
    def all_nodes(self) -> List[FeatureNode]:
        """Return all graph nodes in specification order."""
        return list(self.nodes)

    # ------------------------------------------------------------------------- 16
    def nodes_for_timeframe(self, tf: str) -> List[FeatureNode]:
        """Explicit alias used by downstream consumers."""
        return self.get_by_timeframe(tf)

    # ------------------------------------------------------------------------- 17
    def timeframes(self) -> List[str]:
        """Return all graph timeframes in deterministic order."""
        return list(self.by_tf.keys())

    # ------------------------------------------------------------------------- 18
    def names(self) -> List[str]:
        """Return all indicator names represented in the graph."""
        return list(self.by_name.keys())

    # ------------------------------------------------------------------------- 19
    def roots(self) -> List[FeatureNode]:
        """Return nodes with no explicit dependencies."""
        return [node for node in self.nodes if not self._reverse_edges[node.id]]

    # ------------------------------------------------------------------------- 20
    def leaves(self) -> List[FeatureNode]:
        """Return nodes with no explicit dependents."""
        return [node for node in self.nodes if not self._edges[node.id]]

    # ------------------------------------------------------------------------- 21
    def groups_by_timeframe(self) -> Mapping[str, List[FeatureNode]]:
        """Return a read-only-style mapping copy grouped by timeframe."""
        return {tf: list(nodes) for tf, nodes in self.by_tf.items()}

    # ------------------------------------------------------------------------- 22
    def info(self) -> Dict[str, object]:
        """Return concise structural information for diagnostics/tests."""
        return {
            "node_count": len(self.nodes),
            "edge_count": sum(len(targets) for targets in self._edges.values()),
            "timeframes": list(self.timeframes()),
            "indicator_names": list(self.names()),
        }

    # ------------------------------------------------------------------------- 23
    def __len__(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------------- 24
    def __iter__(self) -> Iterable[FeatureNode]:
        return iter(self.nodes)

# ============================================================================= END