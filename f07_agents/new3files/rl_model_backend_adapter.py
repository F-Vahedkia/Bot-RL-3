# f07_agents/rl_model_backend_adapter.py
#
# RL Backend Boundary
#
# Purpose
# -------
# Keep f07_agents independent from a concrete RL framework.
#
# Supported backend in this file:
#     Stable-Baselines3 (SB3)
#
# RLlib has its own integration layer in:
#     f07_agents/rllib_integration.py
#
# Architecture:
#
#     f07_agents.rl_policy_adapter
#                 |
#                 v
#          RLBackendProtocol
#                 ^
#          -----------------
#          |               |
#       SB3 adapter    RLlib adapter
#                         |
#                 rllib_integration.py
#
# This file must not import Gymnasium spaces or a concrete SB3 algorithm
# at module import time. Optional dependencies are loaded lazily.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from f07_agents.rl_policy_adapter import RLBackendProtocol


class ModelPredictor(Protocol):
    """Minimal predict interface exposed by an RL inference model."""

    def predict(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
    ) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class SB3BackendConfig:
    """Configuration metadata for an SB3 inference backend."""

    algorithm: str
    policy: str = "MlpPolicy"
    device: str = "auto"

    def __post_init__(self) -> None:
        if not str(self.algorithm).strip():
            raise ValueError("algorithm must be non-empty.")
        if not str(self.policy).strip():
            raise ValueError("policy must be non-empty.")
        if not str(self.device).strip():
            raise ValueError("device must be non-empty.")


class SB3ModelBackendAdapter(RLBackendProtocol):
    """
    Adapt a Stable-Baselines3 model to f05's RLBackendProtocol.

    The adapter deliberately does not know about SymbolAgent, MetaAgent,
    PolicyOutput, or MetaPolicyOutput. Action semantics are handled by the
    backend-independent decoders in rl_action_codec.py.
    """

    def __init__(
        self,
        model: ModelPredictor,
        *,
        config: SB3BackendConfig,
    ) -> None:
        if model is None:
            raise ValueError("model is required.")
        if not callable(getattr(model, "predict", None)):
            raise TypeError(
                "model must expose a callable predict(observation, "
                "deterministic=...) method."
            )
        self.model = model
        self.config = config

    def predict(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
    ) -> Any:
        array = self._prepare_observation(observation)
        return self.model.predict(
            array,
            deterministic=bool(deterministic),
        )

    @staticmethod
    def _prepare_observation(observation: Any) -> np.ndarray:
        try:
            value = np.asarray(observation, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "SB3 observation must be numeric."
            ) from exc

        if value.size == 0:
            raise ValueError("SB3 observation must not be empty.")
        if not np.all(np.isfinite(value)):
            raise ValueError(
                "SB3 observation must contain only finite values."
            )
        return value


class SB3ModelProvider:
    """
    Provider helper compatible with PolicyFactory.ModelProvider.

    Example
    -------
    provider = SB3ModelProvider(
        loader=lambda cfg: PPO.load("...", device=cfg.device)
    )
    policy_factory.register_backend("sb3", provider)
    """

    def __init__(
        self,
        loader: Callable[[SB3BackendConfig], ModelPredictor],
    ) -> None:
        if loader is None:
            raise ValueError("loader is required.")
        self.loader = loader

    def __call__(self, config: Any) -> RLBackendProtocol:
        algorithm = str(getattr(config, "algorithm_name", "")).strip()
        policy = str(getattr(config, "policy_name", "MlpPolicy")).strip()
        metadata = getattr(config, "metadata", {}) or {}
        device = str(metadata.get("device", "auto"))

        backend_config = SB3BackendConfig(
            algorithm=algorithm,
            policy=policy,
            device=device,
        )
        model = self.loader(backend_config)
        return SB3ModelBackendAdapter(
            model,
            config=backend_config,
        )


__all__ = [
    "RLBackendProtocol",
    "ModelPredictor",
    "SB3BackendConfig",
    "SB3ModelBackendAdapter",
    "SB3ModelProvider",
]
