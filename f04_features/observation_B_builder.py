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
from f04_features.candle_window_builder import CandleWindowBuilder   # new add
from f02_utils.functions.constants import _TF_MINUTES                # new add

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
        # ----- new add start
        base_tf_cfg = observation_cfg.get("base_timeframe", {}) or {}
        if not isinstance(base_tf_cfg, dict):
            raise TypeError(
                "features.observation.base_timeframe must be a mapping"
            )

        self.observation_tf_mode = str(
            base_tf_cfg.get("mode", "union_min")
        ).strip().lower()

        if self.observation_tf_mode not in {
            "fixed",
            "feature_min",
            "candle_min",
            "union_min",
        }:
            raise ValueError(
                f"Unsupported base_timeframe.mode: "
                f"{self.observation_tf_mode!r}"
            )

        raw_tf = base_tf_cfg.get("timeframe")
        self.observation_tf: str | None = (
            str(raw_tf).strip().upper()
            if raw_tf is not None and str(raw_tf).strip()
            else None
        )

        if self.observation_tf_mode == "fixed" and not self.observation_tf:
            raise ValueError(
                "base_timeframe.timeframe is required when mode='fixed'"
            )

        self.candle_builder = CandleWindowBuilder(config)
        # ----- new add end

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
    # ----- new add-2 start
    def _resolve_observation_timeframe(
        self,
        dataset: MTFDataset,
        feature_columns: Sequence[str],
    ) -> str:
        feature_tfs = {
            tf.upper()
            for tf in dataset.timeframes
            if any(
                column in dataset.get(tf).columns
                for column in feature_columns
            )
        }

        candle_tfs = set(
            self.candle_builder.required_for_symbol(dataset.symbol)
            if self.candle_builder.names
            else {}
        )

        mode = self.observation_tf_mode

        if mode == "fixed":
            assert self.observation_tf is not None
            return self.observation_tf

        if mode == "feature_min":
            if not feature_tfs:
                raise ValueError(
                    "No Feature timeframe found for feature_min mode"
                )
            return min(feature_tfs, key=lambda tf: _TF_MINUTES[tf])

        if mode == "candle_min":
            if not candle_tfs:
                raise ValueError(
                    "No Candle timeframe found for candle_min mode"
                )
            return min(candle_tfs, key=lambda tf: _TF_MINUTES[tf])

        candidates = feature_tfs | candle_tfs

        if not candidates:
            raise ValueError(
                "No Feature or Candle timeframe available for union_min"
            )

        return min(candidates, key=lambda tf: _TF_MINUTES[tf])

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_observation_index(
        dataset: MTFDataset,
        timeframe: str,
    ) -> pd.DatetimeIndex:
        timeframe = timeframe.upper()

        if timeframe in dataset.frames:
            return dataset.get(timeframe).index.copy()

        non_empty = [
            df
            for df in dataset.frames.values()
            if df is not None and not df.empty
        ]

        if not non_empty:
            raise ValueError(
                "Cannot build observation_index from empty MTFDataset"
            )

        start = min(df.index.min() for df in non_empty)
        end = max(df.index.max() for df in non_empty)

        freq = f"{_TF_MINUTES[timeframe]}min"

        return pd.date_range(
            start=start.floor(freq),
            end=end.floor(freq),
            freq=freq,
            name=non_empty[0].index.name,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _aligned_dataset(
        dataset: MTFDataset,
        observation_index: pd.DatetimeIndex,
        observation_tf: str,
    ) -> pd.DataFrame:
        """
        Reuse the existing align_to_base() implementation while making
        observation_tf the temporary alignment base.

        dataset.base_tf is not modified.
        """
        temp = dataset.copy()
        temp.base_tf = observation_tf

        if observation_tf not in temp.frames:
            # align_to_base() returns immediately for a completely empty
            # base frame, so keep one dummy column.
            temp.add(
                observation_tf,
                pd.DataFrame(
                    {"__observation__": np.nan},
                    index=observation_index,
                ),
            )

        aligned = align_to_base(temp)

        if "__observation__" in aligned.columns:
            aligned = aligned.drop(columns="__observation__")

        return aligned.reindex(observation_index)    
    # ----- new add-2 end
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
    def build_old1(
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
    def build(
        self,
        dataset: MTFDataset,
        graph: FeatureGraph,
    ) -> pd.DataFrame:
        """
        Build the final pandas Observation.
        """

        self._validate_inputs(dataset, graph)

        feature_columns = self._filtered_graph_columns(graph)

        missing = [
            column
            for column in feature_columns
            if not any(
                column in dataset.get(tf).columns
                for tf in dataset.timeframes
            )
        ]

        if missing:
            raise KeyError(
                "Feature columns required by FeatureGraph are missing from "
                f"the FeatureStore dataset: {missing}"
            )

        # -------------------------------------------------------------
        # 1. Observation owns the Observation timeframe/index.
        # -------------------------------------------------------------

        observation_tf = self._resolve_observation_timeframe(
            dataset,
            feature_columns,
        )

        observation_index = self._build_observation_index(
            dataset,
            observation_tf,
        )

        # -------------------------------------------------------------
        # 2. Align Features onto observation_index.
        # -------------------------------------------------------------

        aligned = self._aligned_dataset(
            dataset,
            observation_index,
            observation_tf,
        )

        obs = aligned.loc[:, feature_columns].copy()

        if self.shift:
            obs = obs.shift(self.shift)

        if self.drop_na_head:
            obs = self._drop_leading_warmup(
                obs,
                feature_columns,
            )

        # -------------------------------------------------------------
        # 3. CandleWindowBuilder uses exactly the same final index.
        # -------------------------------------------------------------

        candles = self.candle_builder.build(
            dataset=dataset,
            observation_index=obs.index,
            symbol=dataset.symbol,
        )

        # -------------------------------------------------------------
        # 4. Final Observation
        # -------------------------------------------------------------

        overlap = set(obs.columns).intersection(candles.columns)
        if overlap:
            raise ValueError(
                "Observation column collision between features and candles: "
                f"{sorted(overlap)}"
            )

        return pd.concat(
            [obs, candles],
            axis=1,
        )

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