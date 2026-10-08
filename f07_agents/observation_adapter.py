# f07_agents/observation_adapter.py (11)
#
# Created: 1405/06/24

# Boundary Adapter:
#     f04_features.Observation
#                 |
#                 v
#     AgentObservation
#                 |
#                 v
#          f07_agents
#
# Responsibilities
# ----------------
# - Translate the final Observation DataFrame produced by f04_features
#   into the AgentObservation contract consumed by Symbol-Agent.
# - Preserve feature-column order exactly.
# - Preserve the observation timestamp from the DataFrame index.
# - Preserve symbol isolation.
# - Validate timezone-awareness, numeric values, and finite values.
#
# Non-responsibilities
# --------------------
# - Feature calculation.
# - Feature alignment.
# - Feature selection.
# - Feature shifting.
# - SymbolContext construction.
# - PortfolioContext construction.
# - RL inference.
# - Portfolio decision-making.
#
# Important architectural rule
# ----------------------------
# The Adapter consumes the FINAL observation produced by ObservationBuilder.
# It must not rebuild or reinterpret the Feature Layer output.

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from f07_agents.contracts import AgentObservation


class FeatureObservationAdapter:
    """Official boundary adapter between f04_features and f07_agents.

    Input:
        A final Observation DataFrame produced by ObservationBuilder /
        FeaturePipeline for exactly one symbol.

    Output:
        AgentObservation for exactly one Symbol-Agent.

    The adapter intentionally performs only semantic translation and
    boundary validation. It does not contain trading logic.
    """

    def __init__(self, *, symbol: str, base_tf: str) -> None:
        symbol = str(symbol).replace(" ", "")
        base_tf = str(base_tf).replace(" ", "").upper()

        if not symbol:
            raise ValueError("symbol is required")
        if not base_tf:
            raise ValueError("base_tf is required")

        self.symbol = symbol
        self.base_tf = base_tf

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_observation_frame(
        observation: pd.DataFrame,
    ) -> None:
        if observation is None:
            raise ValueError("observation is required")

        if not isinstance(observation, pd.DataFrame):
            raise TypeError("observation must be pandas.DataFrame")

        if len(observation.columns) == 0:
            raise ValueError(
                "observation must contain at least one feature column"
            )

        if observation.empty:
            raise ValueError("observation must not be empty")

        if not isinstance(observation.index, pd.DatetimeIndex):
            raise TypeError(
                "observation index must be pandas.DatetimeIndex"
            )

        if observation.index.tz is None:
            raise ValueError(
                "observation index must be timezone-aware"
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_timestamp(timestamp: pd.Timestamp) -> None:
        if timestamp.tzinfo is None:
            raise ValueError("observation timestamp must be timezone-aware")

    # -------------------------------------------------------------------------
    @staticmethod
    def _row_to_values(
        row: pd.Series,
        *,
        location: str,
    ) -> tuple[float, ...]:
        try:
            values = np.asarray(
                row.to_numpy(dtype=np.float64, copy=True),
                dtype=np.float64,
            ).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Observation feature values must be numeric at {location}"
            ) from exc

        if values.size == 0:
            raise ValueError(
                f"Observation row must not be empty at {location}"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Observation contains non-finite values at {location}"
            )

        return tuple(float(value) for value in values)

    # -------------------------------------------------------------------------
    @staticmethod
    def _feature_names(
        observation: pd.DataFrame,
    ) -> tuple[str, ...]:
        feature_names = tuple(str(column) for column in observation.columns)
        if not feature_names:
            raise ValueError("observation must contain feature columns")
        return feature_names

    # -------------------------------------------------------------------------
    def build(
        self,
        observation: pd.DataFrame,
    ) -> AgentObservation:
        """Convert the latest row into the official AgentObservation.

        This is the primary API for live/paper/shadow processing where the
        FeaturePipeline has produced the current final Observation frame.
        """
        self._validate_observation_frame(observation)

        timestamp = pd.Timestamp(observation.index[-1])
        self._validate_timestamp(timestamp)

        values = self._row_to_values(
            observation.iloc[-1],
            location=str(timestamp),
        )
        feature_names = self._feature_names(observation)

        return AgentObservation(
            symbol=self.symbol,
            base_tf=self.base_tf,
            timestamp=timestamp.to_pydatetime(),
            values=values,
            feature_names=feature_names,
        )

    # -------------------------------------------------------------------------
    def build_at(
        self,
        observation: pd.DataFrame,
        *,
        timestamp: pd.Timestamp | datetime,
    ) -> AgentObservation:
        """Convert one explicitly selected observation row.

        Intended for backtest/replay/evaluation workflows where the caller
        controls the decision timestamp.
        """
        self._validate_observation_frame(observation)

        timestamp = pd.Timestamp(timestamp)
        self._validate_timestamp(timestamp)

        if not observation.index.is_unique:
            raise ValueError("observation index must be unique")

        if timestamp not in observation.index:
            raise KeyError(
                f"Observation timestamp not found: {timestamp}"
            )

        row = observation.loc[timestamp]
        if isinstance(row, pd.DataFrame):
            raise ValueError("observation index must be unique")

        values = self._row_to_values(
            row,
            location=str(timestamp),
        )
        feature_names = self._feature_names(observation)

        return AgentObservation(
            symbol=self.symbol,
            base_tf=self.base_tf,
            timestamp=timestamp.to_pydatetime(),
            values=values,
            feature_names=feature_names,
        )

    # -------------------------------------------------------------------------
    def build_sequence(
        self,
        observation: pd.DataFrame,
    ) -> tuple[AgentObservation, ...]:
        """Convert every row into an AgentObservation.

        Intended for train/optimize/replay/evaluation workflows.
        Non-finite rows are rejected rather than silently dropped or changed.
        """
        self._validate_observation_frame(observation)

        if not observation.index.is_unique:
            raise ValueError("observation index must be unique")

        feature_names = self._feature_names(observation)
        result: list[AgentObservation] = []

        for timestamp, row in observation.iterrows():
            timestamp = pd.Timestamp(timestamp)
            self._validate_timestamp(timestamp)
            values = self._row_to_values(
                row,
                location=str(timestamp),
            )
            result.append(
                AgentObservation(
                    symbol=self.symbol,
                    base_tf=self.base_tf,
                    timestamp=timestamp.to_pydatetime(),
                    values=values,
                    feature_names=feature_names,
                )
            )

        return tuple(result)

    # -------------------------------------------------------------------------
    @classmethod
    def from_pipeline_output(
        cls,
        *,
        observation: pd.DataFrame,
        symbol: str,
        base_tf: str,
    ) -> AgentObservation:
        """Convenience entry point for FeaturePipeline output.

        Example:
            result = pipeline.run(dataset)
            agent_obs = FeatureObservationAdapter.from_pipeline_output(
                observation=result["observation"],
                symbol="XAUUSD",
                base_tf="M1",
            )
        """
        adapter = cls(symbol=symbol, base_tf=base_tf)
        return adapter.build(observation)


__all__ = [
    "FeatureObservationAdapter",
]
