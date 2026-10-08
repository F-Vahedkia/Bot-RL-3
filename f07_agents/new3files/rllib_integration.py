# f07_agents/rllib_integration.py
#
# Ray RLlib Integration Layer
#
# Target RLlib API:
#     RLlib 2.58.x new API stack
#
# This module is deliberately outside the f07_agents core Decision contracts.
# It bridges RLlib's RLModule/Algorithm APIs to the existing
# RLBackendProtocol used by f07_agents.rl_policy_adapter.
#
# Architecture:
#
#     Gymnasium Environment
#             |
#             v
#        RLlib Algorithm
#             |
#        RLModule / Checkpoint
#             |
#             v
#     RLlibModelBackendAdapter
#             |
#             v
#       RLBackendProtocol
#             |
#             v
#     RLModelPolicyAdapter
#             |
#             v
#          SymbolAgent
#
# The module imports Ray lazily so that installing/using the SB3 backend does
# not require Ray to be installed.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from f07_agents.rl_policy_adapter import RLBackendProtocol


@dataclass(frozen=True, slots=True)
class RLlibIntegrationConfig:
    """Framework-neutral configuration needed to connect RLlib inference."""

    algorithm: str = "PPO"
    framework: str = "torch"
    deterministic: bool = True
    module_id: str | None = None
    checkpoint: str | None = None
    num_env_runners: int = 0
    algorithm_training: Mapping[str, Any] = field(default_factory=dict)
    model_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.algorithm).strip():
            raise ValueError("algorithm must be non-empty.")
        if str(self.framework).strip().lower() != "torch":
            raise ValueError(
                "This integration layer currently targets RLlib + PyTorch."
            )
        if self.num_env_runners < 0:
            raise ValueError("num_env_runners must be >= 0.")
        if self.checkpoint is not None and not str(self.checkpoint).strip():
            raise ValueError("checkpoint must be non-empty when supplied.")


class RLlibRLModuleBackendAdapter(RLBackendProtocol):
    """
    Adapt an RLlib RLModule to f05's RLBackendProtocol.

    RLlib's current RLModule API uses forward_inference()/forward_exploration()
    and action distributions rather than the SB3 model.predict() method.
    """

    def __init__(
        self,
        module: Any,
        *,
        deterministic: bool = True,
    ) -> None:
        if module is None:
            raise ValueError("module is required.")
        if not callable(getattr(module, "forward_inference", None)):
            raise TypeError(
                "module must expose RLlib RLModule forward_inference()."
            )
        if not callable(getattr(module, "forward_exploration", None)):
            raise TypeError(
                "module must expose RLlib RLModule forward_exploration()."
            )
        self.module = module
        self.deterministic = bool(deterministic)

    def predict(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
    ) -> Any:
        array = self._prepare_observation(observation)

        # Lazy imports keep Ray optional for SB3-only installations.
        try:
            import torch
            from ray.rllib.core.columns import Columns
        except ImportError as exc:
            raise RuntimeError(
                "RLlib inference requires ray[rllib] and PyTorch."
            ) from exc

        batch = {
            Columns.OBS: torch.as_tensor(
                array.reshape(1, -1),
                dtype=torch.float32,
            )
        }

        use_deterministic = bool(deterministic)
        forward = (
            self.module.forward_inference
            if use_deterministic
            else self.module.forward_exploration
        )
        outputs = forward(batch)

        if Columns.ACTIONS in outputs:
            return self._to_numpy_action(outputs[Columns.ACTIONS])

        if Columns.ACTION_DIST_INPUTS not in outputs:
            raise RuntimeError(
                "RLlib RLModule returned neither actions nor "
                "action_dist_inputs."
            )

        distribution_cls = (
            self.module.get_inference_action_dist_cls()
            if use_deterministic
            else self.module.get_exploration_action_dist_cls()
        )
        distribution = distribution_cls.from_logits(
            outputs[Columns.ACTION_DIST_INPUTS]
        )
        if use_deterministic and hasattr(
            distribution,
            "to_deterministic",
        ):
            distribution = distribution.to_deterministic()

        action = distribution.sample()
        return self._to_numpy_action(action)

    @staticmethod
    def _prepare_observation(observation: Any) -> np.ndarray:
        try:
            array = np.asarray(observation, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "RLlib observation must be numeric."
            ) from exc
        if array.size == 0:
            raise ValueError("RLlib observation must not be empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError(
                "RLlib observation must contain only finite values."
            )
        return array

    @staticmethod
    def _to_numpy_action(action: Any) -> Any:
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        else:
            action = np.asarray(action)
        if action.ndim > 0 and action.shape[0] == 1:
            action = action[0]
        return action


class RLlibAlgorithmBackendAdapter(RLBackendProtocol):
    """Resolve an RLModule from an RLlib Algorithm and delegate inference."""

    def __init__(
        self,
        algorithm: Any,
        *,
        module_id: str | None = None,
        deterministic: bool = True,
    ) -> None:
        if algorithm is None:
            raise ValueError("algorithm is required.")
        if not callable(getattr(algorithm, "get_module", None)):
            raise TypeError(
                "algorithm must expose RLlib Algorithm.get_module()."
            )
        self.algorithm = algorithm
        self.module_id = module_id
        self.module = algorithm.get_module(module_id)
        self._backend = RLlibRLModuleBackendAdapter(
            self.module,
            deterministic=deterministic,
        )

    def predict(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
    ) -> Any:
        return self._backend.predict(
            observation,
            deterministic=deterministic,
        )


class RLlibIntegrationLayer:
    """
    Build RLlib AlgorithmConfig/Algorithm objects and expose an inference
    backend compatible with PolicyFactory.

    The public training API stays in the training layer; this class only
    translates project configuration into RLlib's configuration API and
    provides the inference boundary needed by f07_agents.
    """

    _ALGORITHM_CONFIGS = {
        "ppo": ("ray.rllib.algorithms.ppo", "PPOConfig"),
        "dqn": ("ray.rllib.algorithms.dqn", "DQNConfig"),
        "sac": ("ray.rllib.algorithms.sac", "SACConfig"),
        "appo": ("ray.rllib.algorithms.appo", "APPOConfig"),
        "impala": ("ray.rllib.algorithms.impala", "IMPALAConfig"),
    }

    def __init__(self, config: RLlibIntegrationConfig) -> None:
        self.config = config

    def build_algorithm_config(
        self,
        *,
        env: Any | None = None,
    ) -> Any:
        """Build an RLlib AlgorithmConfig using RLlib's current builder API."""
        try:
            config_class = self._import_algorithm_config_class(
                self.config.algorithm
            )
        except ImportError as exc:
            raise RuntimeError(
                "RLlib is not installed or the requested RLlib algorithm "
                "is unavailable."
            ) from exc

        algorithm_config = config_class()

        # Current RLlib supports the new RLModule/Learner and
        # EnvRunner/ConnectorV2 APIs as a paired stack.
        if callable(getattr(algorithm_config, "api_stack", None)):
            algorithm_config = algorithm_config.api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )

        if callable(getattr(algorithm_config, "framework", None)):
            algorithm_config = algorithm_config.framework(
                self.config.framework
            )

        if env is not None:
            algorithm_config = algorithm_config.environment(env=env)

        if callable(getattr(algorithm_config, "env_runners", None)):
            algorithm_config = algorithm_config.env_runners(
                num_env_runners=self.config.num_env_runners,
            )

        if self.config.algorithm_training:
            algorithm_config = algorithm_config.training(
                **dict(self.config.algorithm_training)
            )

        if self.config.model_config and callable(
            getattr(algorithm_config, "rl_module", None)
        ):
            algorithm_config = algorithm_config.rl_module(
                model_config=dict(self.config.model_config)
            )

        return algorithm_config

    def build_algorithm(
        self,
        *,
        env: Any | None = None,
    ) -> Any:
        """Build and return an RLlib Algorithm."""
        config = self.build_algorithm_config(env=env)
        build = getattr(config, "build_algo", None)
        if not callable(build):
            build = getattr(config, "build", None)
        if not callable(build):
            raise RuntimeError(
                "RLlib AlgorithmConfig exposes neither build_algo() nor build()."
            )
        return build()

    def load_algorithm_from_checkpoint(self) -> Any:
        """Load an RLlib Algorithm from the configured checkpoint."""
        if not self.config.checkpoint:
            raise ValueError(
                "checkpoint is required to load an RLlib checkpoint."
            )
        try:
            algorithm_class = self._import_algorithm_class(
                self.config.algorithm
            )
        except ImportError as exc:
            raise RuntimeError(
                "RLlib is not installed or the requested RLlib algorithm "
                "is unavailable."
            ) from exc

        loader = getattr(algorithm_class, "from_checkpoint", None)
        if not callable(loader):
            raise RuntimeError(
                "Configured RLlib Algorithm does not expose "
                "from_checkpoint()."
            )
        return loader(self.config.checkpoint)

    def backend_from_algorithm(
        self,
        algorithm: Any,
    ) -> RLBackendProtocol:
        """Create the f05-compatible inference backend from an RLlib Algorithm."""
        return RLlibAlgorithmBackendAdapter(
            algorithm,
            module_id=self.config.module_id,
            deterministic=self.config.deterministic,
        )

    def backend_from_checkpoint(self) -> RLBackendProtocol:
        """Load a checkpoint and expose its module through RLBackendProtocol."""
        algorithm = self.load_algorithm_from_checkpoint()
        return self.backend_from_algorithm(algorithm)

    @classmethod
    def _import_algorithm_config_class(cls, algorithm: str) -> type:
        import importlib

        key = str(algorithm).strip().lower()
        entry = cls._ALGORITHM_CONFIGS.get(key)
        if entry is None:
            raise KeyError(
                f"Unsupported RLlib algorithm config: {algorithm!r}"
            )
        module_name, class_name = entry
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @classmethod
    def _import_algorithm_class(cls, algorithm: str) -> type:
        import importlib

        key = str(algorithm).strip().lower()
        module_name = {
            "ppo": "ray.rllib.algorithms.ppo",
            "dqn": "ray.rllib.algorithms.dqn",
            "sac": "ray.rllib.algorithms.sac",
            "appo": "ray.rllib.algorithms.appo",
            "impala": "ray.rllib.algorithms.impala",
        }.get(key)
        if module_name is None:
            raise KeyError(
                f"Unsupported RLlib algorithm: {algorithm!r}"
            )
        module = importlib.import_module(module_name)
        class_name = {
            "ppo": "PPO",
            "dqn": "DQN",
            "sac": "SAC",
            "appo": "APPO",
            "impala": "IMPALA",
        }[key]
        return getattr(module, class_name)


class RLlibModelProvider:
    """
    PolicyFactory provider helper.

    It can either receive an already-created Algorithm or an integration layer
    capable of loading/building one. PolicyFactory itself remains unchanged.
    """

    def __init__(
        self,
        integration: RLlibIntegrationLayer,
        *,
        algorithm: Any | None = None,
    ) -> None:
        if integration is None:
            raise ValueError("integration is required.")
        self.integration = integration
        self.algorithm = algorithm

    def __call__(self, config: Any) -> RLBackendProtocol:
        # If a live Algorithm was injected, reuse it. Otherwise use the
        # checkpoint defined by the integration config.
        if self.algorithm is not None:
            return self.integration.backend_from_algorithm(self.algorithm)
        return self.integration.backend_from_checkpoint()


__all__ = [
    "RLlibIntegrationConfig",
    "RLlibRLModuleBackendAdapter",
    "RLlibAlgorithmBackendAdapter",
    "RLlibIntegrationLayer",
    "RLlibModelProvider",
]
