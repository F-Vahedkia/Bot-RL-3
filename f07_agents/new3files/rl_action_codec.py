# f07_agents/rl_action_codec.py
#
# Backend-independent RL Action Encoding / Decoding
#
# Purpose
# -------
# Translate backend-native RL actions into the existing f07_agents contracts.
# The same decoder can be used with SB3 or RLlib.
#
# Important boundary rule
# -----------------------
# A backend produces a numeric/native RL action. This file assigns that action
# its trading semantics. The backend itself must not know about
# SymbolAgentOutput or PortfolioDecision.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from f07_agents.contracts import (
    AgentObservation,
    MetaPolicyOutput,
    PortfolioContext,
    PolicyOutput,
    SymbolAgentOutput,
)


@dataclass(frozen=True, slots=True)
class DiscreteSymbolActionCodecConfig:
    """Configuration for a discrete Symbol-Agent action decoder."""

    actions: tuple[str, ...] = ("short", "hold", "long")
    levels: tuple[int, ...] = (-1, 0, 1)
    exposure_by_signal: Mapping[int, float] = field(
        default_factory=lambda: {
            -1: 0.20,
            0: 0.0,
            1: 0.20,
        }
    )
    stop_price_source: str = "context_metadata"
    stop_price_key: str = "proposed_stop_price"

    def __post_init__(self) -> None:
        if len(self.actions) != len(self.levels):
            raise ValueError("actions and levels must have equal length.")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("levels must be unique.")
        if set(self.levels) != {-1, 0, 1}:
            raise ValueError("levels must contain -1, 0 and 1.")
        if self.stop_price_source not in {"context_metadata", "action_payload"}:
            raise ValueError(
                "stop_price_source must be 'context_metadata' or "
                "'action_payload'."
            )
        if not str(self.stop_price_key).strip():
            raise ValueError("stop_price_key must be non-empty.")
        for signal in (-1, 0, 1):
            if signal not in self.exposure_by_signal:
                raise ValueError(
                    f"exposure_by_signal is missing signal {signal}."
                )
            value = float(self.exposure_by_signal[signal])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"exposure_by_signal[{signal}] must be finite and in [0, 1]."
                )


class DiscreteSymbolActionCodec:
    """
    Decode a 3-level RL action into the existing PolicyOutput contract.

    Raw action forms accepted:
        - integer scalar: backend Discrete action index
        - one-element numeric array
        - mapping containing signal/desired_exposure/stop_price
        - existing PolicyOutput

    For a scalar discrete action, a non-flat stop must come from the configured
    SymbolContext.metadata key. This avoids inventing stop-loss semantics in
    the RL backend adapter.
    """

    def __init__(
        self,
        config: DiscreteSymbolActionCodecConfig | None = None,
    ) -> None:
        self.config = config or DiscreteSymbolActionCodecConfig()
        self._index_to_signal = {
            index: signal
            for index, signal in enumerate(self.config.levels)
        }

    def decode(
        self,
        raw_action: Any,
        observation: AgentObservation,
        context: Any,
    ) -> PolicyOutput:
        if isinstance(raw_action, PolicyOutput):
            return raw_action

        if isinstance(raw_action, Mapping):
            return self._decode_mapping(raw_action)

        index = self._decode_index(raw_action)
        signal = int(self._index_to_signal[index])
        exposure = float(self.config.exposure_by_signal[signal])

        if signal == 0:
            return PolicyOutput(
                signal=0,
                confidence=1.0,
                expected_return=0.0,
                risk_score=0.0,
                desired_exposure=0.0,
                stop_price=None,
            )

        stop_price = self._resolve_stop_price(context)
        return PolicyOutput(
            signal=signal,
            confidence=1.0,
            expected_return=0.0,
            risk_score=0.0,
            desired_exposure=exposure,
            stop_price=stop_price,
            metadata={
                "action_index": index,
                "action_label": self.config.actions[index],
                "decoder": "DiscreteSymbolActionCodec",
            },
        )

    def __call__(
        self,
        raw_action: Any,
        observation: AgentObservation,
        context: Any,
    ) -> PolicyOutput:
        return self.decode(raw_action, observation, context)

    def _decode_index(self, raw_action: Any) -> int:
        array = np.asarray(raw_action)
        if array.size != 1:
            raise ValueError(
                "Discrete Symbol action must contain exactly one value."
            )
        value = float(array.reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("Discrete Symbol action must be finite.")
        index = int(value)
        if float(index) != value:
            raise ValueError(
                "Discrete Symbol action index must be an integer."
            )
        if index not in self._index_to_signal:
            raise ValueError(
                f"Unknown discrete Symbol action index: {index}."
            )
        return index

    @staticmethod
    def _decode_mapping(payload: Mapping[str, Any]) -> PolicyOutput:
        required = {
            "signal",
            "confidence",
            "expected_return",
            "risk_score",
            "desired_exposure",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                "Symbol action payload is missing fields: "
                + ", ".join(missing)
            )

        return PolicyOutput(
            signal=int(payload["signal"]),
            confidence=float(payload["confidence"]),
            expected_return=float(payload["expected_return"]),
            risk_score=float(payload["risk_score"]),
            desired_exposure=float(payload["desired_exposure"]),
            stop_price=(
                None
                if payload.get("stop_price") is None
                else float(payload["stop_price"])
            ),
            metadata=dict(payload.get("metadata", {})),
        )

    def _resolve_stop_price(self, context: Any) -> float:
        if self.config.stop_price_source != "context_metadata":
            raise ValueError(
                "Scalar discrete Symbol actions require a context_metadata "
                "stop price source in this codec version."
            )

        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise ValueError(
                "SymbolContext.metadata is required to resolve stop_price."
            )
        if self.config.stop_price_key not in metadata:
            raise ValueError(
                "SymbolContext.metadata does not contain the configured "
                f"stop price key: {self.config.stop_price_key!r}"
            )
        value = float(metadata[self.config.stop_price_key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "Configured SymbolContext stop price must be finite and > 0."
            )
        return value


class MappingMetaActionCodec:
    """Decode a backend mapping into the existing MetaPolicyOutput contract."""

    def decode(
        self,
        raw_action: Any,
        signals: Mapping[str, SymbolAgentOutput],
        portfolio: PortfolioContext,
        context: Any = None,
    ) -> MetaPolicyOutput:
        if isinstance(raw_action, MetaPolicyOutput):
            return raw_action
        if not isinstance(raw_action, Mapping):
            raise TypeError(
                "Meta mapping decoder requires a mapping or MetaPolicyOutput."
            )

        required = {
            "capital_allocation",
            "target_signals",
            "target_exposure",
            "margin_allocation",
            "portfolio_risk",
        }
        missing = sorted(required - set(raw_action))
        if missing:
            raise ValueError(
                "Meta action payload is missing fields: "
                + ", ".join(missing)
            )

        return MetaPolicyOutput(
            capital_allocation=dict(raw_action["capital_allocation"]),
            target_signals={
                str(symbol): int(signal)
                for symbol, signal in raw_action["target_signals"].items()
            },
            target_exposure={
                str(symbol): float(value)
                for symbol, value in raw_action["target_exposure"].items()
            },
            margin_allocation=dict(raw_action["margin_allocation"]),
            portfolio_risk=float(raw_action["portfolio_risk"]),
            reason_codes=tuple(
                str(value)
                for value in raw_action.get("reason_codes", ())
            ),
            target_stop_price={
                str(symbol): (
                    None
                    if value is None
                    else float(value)
                )
                for symbol, value in raw_action.get(
                    "target_stop_price", {}
                ).items()
            },
        )

    def __call__(
        self,
        raw_action: Any,
        signals: Mapping[str, SymbolAgentOutput],
        portfolio: PortfolioContext,
        context: Any = None,
    ) -> MetaPolicyOutput:
        return self.decode(raw_action, signals, portfolio, context)


__all__ = [
    "DiscreteSymbolActionCodecConfig",
    "DiscreteSymbolActionCodec",
    "MappingMetaActionCodec",
]
