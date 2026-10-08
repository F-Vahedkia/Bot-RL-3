# f07_agents/tests/agent_factory.py (9)
#
# Created: 1405/06/24

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from f07_agents.contracts import DecisionMode, ModelIdentity
from f07_agents.meta_agent import MetaAgent, MetaAgentConfig, MetaPolicy
from f07_agents.symbol_agent import SymbolAgent, SymbolAgentConfig, SymbolPolicy


def _normalize_symbol(value: str) -> str:
    return "".join(str(value).split()).upper()


@dataclass(frozen=True)
class AgentFactoryConfig:
    symbols: tuple[str, ...]
    mode: DecisionMode
    max_total_allocation: float = 1.0
    max_symbol_allocation: float = 0.5
    max_total_exposure: float = 1.0

    def __post_init__(self) -> None:
        normalized = tuple(_normalize_symbol(symbol) for symbol in self.symbols)
        if not normalized:
            raise ValueError("symbols must not be empty.")
        if any(not symbol for symbol in normalized):
            raise ValueError("symbols must contain non-empty names.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("symbols must be unique after normalization.")
        if not isinstance(self.mode, DecisionMode):
            raise TypeError("mode must be DecisionMode.")
        for name, value in (
            ("max_total_allocation", self.max_total_allocation),
            ("max_symbol_allocation", self.max_symbol_allocation),
            ("max_total_exposure", self.max_total_exposure),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        object.__setattr__(self, "symbols", normalized)


class AgentFactory:
    """Constructs Symbol-Agent(s) and the Meta-Agent from validated policies."""

    def __init__(self, *, config: AgentFactoryConfig) -> None:
        self.config = config

    def create_symbol_agents(
        self,
        *,
        policies: Mapping[str, SymbolPolicy],
        models: Mapping[str, ModelIdentity],
    ) -> dict[str, SymbolAgent]:
        if policies is None:
            raise ValueError("policies is required.")
        if models is None:
            raise ValueError("models is required.")

        normalized_policies = {
            _normalize_symbol(symbol): policy for symbol, policy in policies.items()
        }
        normalized_models = {
            _normalize_symbol(symbol): model for symbol, model in models.items()
        }

        expected = set(self.config.symbols)
        policy_keys = set(normalized_policies)
        model_keys = set(normalized_models)

        missing_policies = sorted(expected - policy_keys)
        extra_policies = sorted(policy_keys - expected)
        missing_models = sorted(expected - model_keys)
        extra_models = sorted(model_keys - expected)

        if missing_policies:
            raise ValueError(
                "Missing SymbolPolicy for symbols: " + ", ".join(missing_policies)
            )
        if extra_policies:
            raise ValueError(
                "Policies provided for unknown symbols: " + ", ".join(extra_policies)
            )
        if missing_models:
            raise ValueError(
                "Missing ModelIdentity for symbols: " + ", ".join(missing_models)
            )
        if extra_models:
            raise ValueError(
                "Models provided for unknown symbols: " + ", ".join(extra_models)
            )

        return {
            symbol: SymbolAgent(
                config=SymbolAgentConfig(
                    symbol=symbol,
                    mode=self.config.mode,
                    model=normalized_models[symbol],
                ),
                policy=normalized_policies[symbol],
            )
            for symbol in self.config.symbols
        }

    def create_meta_agent(
        self,
        *,
        policy: MetaPolicy,
        model: ModelIdentity,
    ) -> MetaAgent:
        if policy is None:
            raise ValueError("policy is required.")
        if model is None:
            raise ValueError("model is required.")

        return MetaAgent(
            config=MetaAgentConfig(
                mode=self.config.mode,
                model=model,
                max_total_allocation=self.config.max_total_allocation,
                max_symbol_allocation=self.config.max_symbol_allocation,
                max_total_exposure=self.config.max_total_exposure,
            ),
            policy=policy,
        )

    def create(
        self,
        *,
        symbol_policies: Mapping[str, SymbolPolicy],
        symbol_models: Mapping[str, ModelIdentity],
        meta_policy: MetaPolicy,
        meta_model: ModelIdentity,
    ) -> tuple[dict[str, SymbolAgent], MetaAgent]:
        symbol_agents = self.create_symbol_agents(
            policies=symbol_policies,
            models=symbol_models,
        )
        meta_agent = self.create_meta_agent(
            policy=meta_policy,
            model=meta_model,
        )
        return symbol_agents, meta_agent


__all__ = ["AgentFactoryConfig", "AgentFactory"]
