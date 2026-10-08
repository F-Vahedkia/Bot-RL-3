# f04_features/observation_B_builder.py
# Date reviewed:
#    1405/05/23-16:00 --> run result for tester ver. _A is OK by 23 tests.

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from fnmatch import fnmatch
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from f03_data.mtf_dataset import MTFDataset
from f04_features.feature_B_graph import FeatureGraph, FeatureNode
from f04_features.feature_B_store import align_to_base
from f04_features.feature_C_registry_1 import get_indicator

# =============================================================================
# Observation Builder
# =============================================================================
class ObservationBuilder:
    """
    Build the final observation from the FeatureStore MTFDataset.

    Contract
    --------
    Input:
        - MTFDataset produced by FeaturePipeline after FeatureStoreV2.build().
        - FeatureGraph built from the same feature specifications.

    Output:
        - pandas.DataFrame indexed by the base timeframe timestamps, or
        - NumPy array through build_numpy().

    Responsibilities
    ----------------
    - Resolve graph nodes to the actual feature columns created by FeatureEngine.
    - Align higher-timeframe data onto the base timeframe without look-ahead.
    - Apply whitelist / blacklist selection.
    - Apply the configured feature shift.
    - Optionally remove only the leading warm-up rows.
    - Return a deterministic feature-column order.

    Non-responsibilities
    --------------------
    - Feature calculation: FeatureEngine.
    - Dataset merge / persistence: FeatureStoreV2.
    - Feature state / cache: FeatureEngine / FeatureCache.
    - Specification ownership: FeaturePipeline.
    - Dependency inference: FeatureGraph.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        if config is None:
            raise ValueError("config is required for ObservationBuilder")
        if not isinstance(config, dict):
            raise TypeError(
                f"config must be dict, got {type(config).__name__}"
            )

        # self.cfg: Dict[str, Any] = config

        features_cfg = config.get("features", {}) or {}
        observation_cfg = features_cfg.get("observation", {}) or {}
        # env_cfg = config.get("env", {}) or {}

        self.shift: int = int(observation_cfg.get("shift_features_by", 0) or 0)
        if self.shift < 0:
            raise ValueError("shift_features_by must be >= 0")

        self.drop_na_head: bool = bool(observation_cfg.get("drop_na_head", True))

        self.whitelist: tuple[str, ...] = tuple(
            str(x) for x in (observation_cfg.get("features_whitelist", []) or [])
        )
        self.blacklist: tuple[str, ...] = tuple(
            str(x) for x in (observation_cfg.get("features_blacklist", []) or [])
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_inputs(
        dataset: MTFDataset,
        graph: FeatureGraph,
    ) -> None:
        if dataset is None:
            raise ValueError("dataset is required")
        if graph is None:
            raise ValueError("graph is required")

        if not isinstance(dataset, MTFDataset):
            raise TypeError(
                f"Expected dataset to be MTFDataset, got {type(dataset).__name__}"
            )
        if not isinstance(graph, FeatureGraph):
            raise TypeError(
                f"Expected graph to be FeatureGraph, got {type(graph).__name__}"
            )
        if not dataset.frames:
            raise ValueError("MTFDataset contains no timeframe frames")

    # -------------------------------------------------------------------------
    @staticmethod
    def _output_columns(node: FeatureNode) -> List[str]:
        """
        Resolve the exact column names produced by FeatureEngine.

        Batch and Live use the same naming rule:
            one output  -> canonical
            many outputs -> output_name + canonical suffix after indicator name
        """
        spec = get_indicator(node.name, mode="train")
        if spec is None:
            raise ValueError(
                f"Indicator '{node.name}' from graph is not present in Registry"
            )

        output_names = list(spec.output_names or [])
        if not output_names:
            return [node.canonical]

        if len(output_names) == 1:
            return [node.canonical]

        suffix = node.canonical[len(node.name):]
        return [f"{name}{suffix}" for name in output_names]

    # -------------------------------------------------------------------------
    def _graph_columns(self, graph: FeatureGraph) -> List[str]:
        """Return feature columns in graph execution/specification order."""
        columns: List[str] = []
        seen: set[str] = set()

        for node in graph.execution_order():
            for column in self._output_columns(node):
                if column not in seen:
                    columns.append(column)
                    seen.add(column)

        return columns

    # -------------------------------------------------------------------------
    @staticmethod
    def _matches_pattern(column: str, pattern: str) -> bool:
        """Match exact names or glob patterns; non-glob patterns use substring."""
        if "*" in pattern or "?" in pattern or "[" in pattern:
            return fnmatch(column, pattern)
        return column == pattern or pattern in column

    # ------------------------------------------------------------------------- deleted & new
    def _apply_filters_deleted(self, columns: Sequence[str]) -> List[str]:
        selected = list(columns)

        if self.whitelist:
            selected = [
                column
                for column in selected
                if any(
                    self._matches_pattern(column, pattern)
                    for pattern in self.whitelist
                )
            ]

        if self.blacklist:
            selected = [
                column
                for column in selected
                if not any(
                    self._matches_pattern(column, pattern)
                    for pattern in self.blacklist
                )
            ]

        return selected


    def _filtered_graph_columns(self, graph: FeatureGraph) -> List[str]:              # new
        """
        Resolve graph nodes to their output columns and apply whitelist /
        blacklist at the Indicator-Node level.

        Whitelist semantics:
            1. If the Indicator Node matches the whitelist, all of its
               output columns are selected.
            2. Otherwise, individual output columns are tested against
               the whitelist, allowing selective output filtering.

        Blacklist semantics:
            1. If the Indicator Node matches the blacklist, all of its
               output columns are removed.
            2. Otherwise, individual output columns are tested against
               the blacklist.

        The original graph execution order is preserved.
        """
        columns: List[str] = []
        seen: set[str] = set()

        for node in graph.execution_order():
            output_columns = self._output_columns(node)

            # =============================================================
            # Whitelist
            # =============================================================
            if self.whitelist:
                node_whitelisted = (
                    any(
                        self._matches_pattern(node.name, pattern)
                        for pattern in self.whitelist
                    )
                    or any(
                        self._matches_pattern(node.canonical, pattern)
                        for pattern in self.whitelist
                    )
                )

                if node_whitelisted:
                    selected = list(output_columns)
                else:
                    selected = [
                        column
                        for column in output_columns
                        if any(
                            self._matches_pattern(column, pattern)
                            for pattern in self.whitelist
                        )
                    ]
            else:
                selected = list(output_columns)

            # =============================================================
            # Blacklist
            # =============================================================
            if self.blacklist:
                node_blacklisted = (
                    any(
                        self._matches_pattern(node.name, pattern)
                        for pattern in self.blacklist
                    )
                    or any(
                        self._matches_pattern(node.canonical, pattern)
                        for pattern in self.blacklist
                    )
                )

                if node_blacklisted:
                    selected = []
                else:
                    selected = [
                        column
                        for column in selected
                        if not any(
                            self._matches_pattern(column, pattern)
                            for pattern in self.blacklist
                        )
                    ]

            # =============================================================
            # Preserve graph order and remove duplicates
            # =============================================================
            for column in selected:
                if column not in seen:
                    columns.append(column)
                    seen.add(column)

        return columns
    
    # -------------------------------------------------------------------------
    def _aligned_dataset(self, dataset: MTFDataset) -> pd.DataFrame:
        """Create a base-timeframe view without mutating the MTFDataset."""
        return align_to_base(dataset)

    # -------------------------------------------------------------------------
    @staticmethod
    def _drop_leading_warmup(
        obs: pd.DataFrame,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        """
        Remove only leading rows required by feature warm-up.

        Interior NaNs are preserved. This is intentional: a temporary missing
        value in the middle of a dataset must not delete an otherwise valid
        observation row.
        """
        if obs.empty or not columns:
            return obs

        selected = obs[list(columns)]
        valid_rows = selected.notna().all(axis=1)
        if not valid_rows.any():
            return obs.iloc[0:0].copy()

        first_valid = valid_rows.idxmax()
        return obs.loc[first_valid:].copy()

    # -------------------------------------------------------------------------
    def build(
        self,
        dataset: MTFDataset,
        graph: FeatureGraph,
    ) -> pd.DataFrame:
        """
        Build the final pandas Observation.
        """
        self._validate_inputs(dataset, graph)

        aligned = self._aligned_dataset(dataset)
        feature_columns = self._filtered_graph_columns(graph)

        missing = [column for column in feature_columns if column not in aligned.columns]
        if missing:
            raise KeyError(
                "Feature columns required by FeatureGraph are missing from "
                f"the FeatureStore dataset: {missing}"
            )

        obs = aligned.loc[:, feature_columns].copy()

        if self.shift:
            obs = obs.shift(self.shift)

        if self.drop_na_head:
            obs = self._drop_leading_warmup(obs, feature_columns)

        return obs

    # -------------------------------------------------------------------------
    def build_numpy(
        self,
        dataset: MTFDataset,
        graph: FeatureGraph,
    ) -> np.ndarray:
        """Build the same observation as NumPy float64 values."""
        observation = self.build(dataset, graph)
        try:
            return observation.to_numpy(dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Observation contains values that cannot be converted to float."
            ) from exc

    # -------------------------------------------------------------------------
    def feature_columns(self, graph: FeatureGraph) -> List[str]:
        """Return graph-derived feature columns after whitelist/blacklist filters."""
        if graph is None:
            raise ValueError("graph is required")
        if not isinstance(graph, FeatureGraph):
            raise TypeError(
                f"Expected graph to be FeatureGraph, got {type(graph).__name__}"
            )
        return self._filtered_graph_columns(graph)
# ----------------------------------------------------------------------------- END