# f07_agents/policy_factory.py (8)
#
# Created at 1405/06/24

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from f07_agents.meta_agent import MetaPolicy
from f07_agents.rl_policy_adapter import (
    RLBackendProtocol,
    RLMetaPolicyAdapter,
    RLModelPolicyAdapter,
    RLPolicyAdapterConfig,
    MetaDecoder,
    SymbolDecoder,
)
from f07_agents.symbol_agent import SymbolPolicy


ModelProvider = Callable[[RLPolicyAdapterConfig], RLBackendProtocol]


@dataclass(frozen=True)
class PolicySpec:
    role: str
    backend: str
    algorithm: str
    policy: str
    deterministic: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in {"symbol", "meta"}:
            raise ValueError("role must be 'symbol' or 'meta'.")
        for name, value in (
            ("backend", self.backend),
            ("algorithm", self.algorithm),
            ("policy", self.policy),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must be non-empty.")
        object.__setattr__(self, "metadata", dict(self.metadata))


class PolicyFactory:
    """Creates policy adapters without hard-coding an RL backend or algorithm."""

    def __init__(self) -> None:
        self._providers: dict[str, ModelProvider] = {}

    def register_backend(
        self,
        backend_name: str,
        provider: ModelProvider,
        *,
        replace: bool = False,
    ) -> None:
        key = str(backend_name).strip().lower()
        if not key:
            raise ValueError("backend_name must be non-empty.")
        if provider is None:
            raise ValueError("provider is required.")
        if key in self._providers and not replace:
            raise ValueError(f"Backend already registered: {backend_name!r}")
        self._providers[key] = provider

    def unregister_backend(self, backend_name: str) -> None:
        self._providers.pop(str(backend_name).strip().lower(), None)

    def registered_backends(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def _build_model(
        self,
        spec: PolicySpec,
    ) -> tuple[RLPolicyAdapterConfig, RLBackendProtocol]:
        backend_key = str(spec.backend).strip().lower()
        provider = self._providers.get(backend_key)
        if provider is None:
            raise KeyError(f"No provider registered for backend {spec.backend!r}.")

        config = RLPolicyAdapterConfig(
            backend_name=backend_key,
            algorithm_name=str(spec.algorithm).strip(),
            policy_name=str(spec.policy).strip(),
            deterministic=spec.deterministic,
            metadata=spec.metadata,
        )
        model = provider(config)
        if model is None:
            raise RuntimeError(
                f"Provider returned no model for backend {spec.backend!r}."
            )
        return config, model

    def create_symbol_policy(
        self,
        spec: PolicySpec,
        *,
        decoder: SymbolDecoder | None = None,
    ) -> SymbolPolicy:
        if spec.role != "symbol":
            raise ValueError("create_symbol_policy requires role='symbol'.")
        config, model = self._build_model(spec)
        return RLModelPolicyAdapter(
            model,
            config=config,
            decoder=decoder,
        )

    def create_meta_policy(
        self,
        spec: PolicySpec,
        *,
        observation_builder: Callable,
        decoder: MetaDecoder | None = None,
    ) -> MetaPolicy:
        if spec.role != "meta":
            raise ValueError("create_meta_policy requires role='meta'.")
        config, model = self._build_model(spec)
        return RLMetaPolicyAdapter(
            model,
            config=config,
            observation_builder=observation_builder,
            decoder=decoder,
        )


__all__ = [
    "ModelProvider",
    "PolicySpec",
    "PolicyFactory",
]
