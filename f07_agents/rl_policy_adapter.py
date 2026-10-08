# f07_agents/rl_policy_adapter.py (7)
#
# Created at 1405/06/24

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from f07_agents.contracts import (
    AgentObservation,
    MetaPolicyOutput,
    PortfolioContext,
    PolicyOutput,
    SymbolAgentOutput,
)
from f07_agents.meta_agent import MetaPolicy
from f07_agents.symbol_agent import SymbolPolicy


class RLBackendProtocol(Protocol):
    def predict(self, observation: Any, *, deterministic: bool = True) -> Any:
        """Return the backend-native prediction."""


SymbolDecoder = Callable[
    [Any, AgentObservation, Any],
    PolicyOutput,
]
MetaDecoder = Callable[
    [Any, Mapping[str, SymbolAgentOutput], PortfolioContext, Any],
    MetaPolicyOutput,
]


def _finite_vector(value: Any, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric.") from exc
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _default_symbol_decoder(
    raw_action: Any,
    observation: AgentObservation,
    context: Any,
) -> PolicyOutput:
    if isinstance(raw_action, PolicyOutput):
        return raw_action
    raise TypeError(
        "No SymbolDecoder was supplied. The default decoder accepts only "
        "PolicyOutput; backend-native actions require an explicit decoder."
    )


def _default_meta_decoder(
    raw_action: Any,
    signals: Mapping[str, SymbolAgentOutput],
    portfolio: PortfolioContext,
    context: Any,
) -> MetaPolicyOutput:
    if isinstance(raw_action, MetaPolicyOutput):
        return raw_action
    raise TypeError(
        "No MetaDecoder was supplied. The default decoder accepts only "
        "MetaPolicyOutput; backend-native actions require an explicit decoder."
    )


@dataclass(frozen=True)
class RLPolicyAdapterConfig:
    backend_name: str
    algorithm_name: str
    policy_name: str
    deterministic: bool = True
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("backend_name", self.backend_name),
            ("algorithm_name", self.algorithm_name),
            ("policy_name", self.policy_name),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must be non-empty.")


class RLModelPolicyAdapter(SymbolPolicy):
    """Adapts an RL backend prediction to the existing SymbolPolicy contract."""

    def __init__(
        self,
        model: RLBackendProtocol,
        *,
        config: RLPolicyAdapterConfig,
        decoder: SymbolDecoder | None = None,
    ) -> None:
        if model is None:
            raise ValueError("model is required.")
        self.model = model
        self.config = config
        self.decoder = decoder or _default_symbol_decoder

    def predict(
        self,
        *,
        observation: AgentObservation,
        context: Any,
    ) -> PolicyOutput:
        if not isinstance(observation, AgentObservation):
            raise TypeError("observation must be AgentObservation.")

        raw_observation = _finite_vector(
            observation.values,
            "observation.values",
        )
        result = self.model.predict(
            raw_observation,
            deterministic=self.config.deterministic,
        )
        raw_action = result[0] if isinstance(result, tuple) else result
        output = self.decoder(raw_action, observation, context)

        if not isinstance(output, PolicyOutput):
            raise TypeError("SymbolDecoder must return PolicyOutput.")
        return output


class RLMetaPolicyAdapter(MetaPolicy):
    """Adapts an RL backend prediction to the existing MetaPolicy contract."""

    def __init__(
        self,
        model: RLBackendProtocol,
        *,
        config: RLPolicyAdapterConfig,
        observation_builder: Callable[
            [Mapping[str, SymbolAgentOutput], PortfolioContext],
            Sequence[float] | np.ndarray,
        ],
        decoder: MetaDecoder | None = None,
    ) -> None:
        if model is None:
            raise ValueError("model is required.")
        if observation_builder is None:
            raise ValueError("observation_builder is required.")
        self.model = model
        self.config = config
        self.observation_builder = observation_builder
        self.decoder = decoder or _default_meta_decoder

    def predict(
        self,
        *,
        signals: Mapping[str, SymbolAgentOutput],
        portfolio: PortfolioContext,
    ) -> MetaPolicyOutput:
        if not isinstance(portfolio, PortfolioContext):
            raise TypeError("portfolio must be PortfolioContext.")
        if not signals:
            raise ValueError("signals must not be empty.")

        raw_observation = _finite_vector(
            self.observation_builder(signals, portfolio),
            "meta observation",
        )
        result = self.model.predict(
            raw_observation,
            deterministic=self.config.deterministic,
        )
        raw_action = result[0] if isinstance(result, tuple) else result
        output = self.decoder(
            raw_action,
            signals,
            portfolio,
            self,
        )

        if not isinstance(output, MetaPolicyOutput):
            raise TypeError("MetaDecoder must return MetaPolicyOutput.")
        return output


__all__ = [
    "RLBackendProtocol",
    "RLPolicyAdapterConfig",
    "RLModelPolicyAdapter",
    "RLMetaPolicyAdapter",
    "SymbolDecoder",
    "MetaDecoder",
]
