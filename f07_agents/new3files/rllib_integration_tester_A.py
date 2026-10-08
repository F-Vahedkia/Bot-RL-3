# f07_agents/rllib_integration_tester_A.py
#
# Tests for the RLlib integration boundary.
# These tests are intentionally runnable without Ray installed.

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from f07_agents.rllib_integration import (
    RLlibAlgorithmBackendAdapter,
    RLlibIntegrationConfig,
    RLlibIntegrationLayer,
    RLlibRLModuleBackendAdapter,
    RLlibModelProvider,
)


# -----------------------------------------------------------------------------
# Helpers for validation-only tests
# -----------------------------------------------------------------------------
class FakeModule:
    def forward_inference(self, batch):
        return {"actions": np.array([[1]])}

    def forward_exploration(self, batch):
        return {"actions": np.array([[0]])}


class BadModule:
    def forward_inference(self, batch):
        return {}


class FakeAlgorithm:
    def __init__(self, module=None):
        self.module = module or FakeModule()
        self.requested_module_id = None

    def get_module(self, module_id=None):
        self.requested_module_id = module_id
        return self.module


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def test_config_defaults():
    cfg = RLlibIntegrationConfig()
    assert cfg.algorithm == "PPO"
    assert cfg.framework == "torch"
    assert cfg.deterministic is True
    assert cfg.num_env_runners == 0


def test_config_rejects_empty_algorithm():
    with pytest.raises(ValueError, match="algorithm must be non-empty"):
        RLlibIntegrationConfig(algorithm=" ")


def test_config_accepts_torch_only():
    with pytest.raises(ValueError, match=r"targets RLlib \+ PyTorch"):
        RLlibIntegrationConfig(framework="tf2")


def test_config_rejects_negative_env_runners():
    with pytest.raises(ValueError, match="num_env_runners must be >= 0"):
        RLlibIntegrationConfig(num_env_runners=-1)


def test_config_rejects_blank_checkpoint():
    with pytest.raises(ValueError, match="checkpoint must be non-empty"):
        RLlibIntegrationConfig(checkpoint=" ")


# -----------------------------------------------------------------------------
# RLModule adapter construction
# -----------------------------------------------------------------------------
def test_module_adapter_requires_module():
    with pytest.raises(ValueError, match="module is required"):
        RLlibRLModuleBackendAdapter(None)


def test_module_adapter_requires_forward_inference():
    with pytest.raises(TypeError, match="forward_inference"):
        RLlibRLModuleBackendAdapter(BadModule())


class MissingExploration:
    def forward_inference(self, batch):
        return {}


def test_module_adapter_requires_forward_exploration():
    with pytest.raises(TypeError, match="forward_exploration"):
        RLlibRLModuleBackendAdapter(MissingExploration())


# -----------------------------------------------------------------------------
# Static observation/action helpers
# -----------------------------------------------------------------------------
def test_prepare_observation_returns_float32_vector():
    out = RLlibRLModuleBackendAdapter._prepare_observation([[1, 2], [3, 4]])
    assert out.dtype == np.float32
    assert out.shape == (4,)


def test_prepare_observation_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        RLlibRLModuleBackendAdapter._prepare_observation([])


def test_prepare_observation_rejects_non_numeric():
    with pytest.raises(TypeError, match="must be numeric"):
        RLlibRLModuleBackendAdapter._prepare_observation(["a"])


def test_prepare_observation_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        RLlibRLModuleBackendAdapter._prepare_observation([1.0, np.nan])


def test_to_numpy_action_removes_single_batch_dimension():
    out = RLlibRLModuleBackendAdapter._to_numpy_action(np.array([[2]]))
    assert isinstance(out, np.ndarray)
    assert out.shape == ()
    assert out.item() == 2


def test_to_numpy_action_converts_tensor_like_object():
    class TensorLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([[3]])

    out = RLlibRLModuleBackendAdapter._to_numpy_action(TensorLike())
    assert isinstance(out, np.ndarray)
    assert out.item() == 3


# -----------------------------------------------------------------------------
# Algorithm adapter
# -----------------------------------------------------------------------------
def test_algorithm_adapter_requires_algorithm():
    with pytest.raises(ValueError, match="algorithm is required"):
        RLlibAlgorithmBackendAdapter(None)


def test_algorithm_adapter_requires_get_module():
    with pytest.raises(TypeError, match="get_module"):
        RLlibAlgorithmBackendAdapter(object())


def test_algorithm_adapter_selects_configured_module_id():
    algorithm = FakeAlgorithm()
    adapter = RLlibAlgorithmBackendAdapter(
        algorithm,
        module_id="symbol_policy",
    )
    assert algorithm.requested_module_id == "symbol_policy"
    assert isinstance(adapter.module, FakeModule)


# -----------------------------------------------------------------------------
# Integration layer import mapping -- no Ray required
# -----------------------------------------------------------------------------
def test_import_algorithm_config_rejects_unknown_algorithm():
    with pytest.raises(KeyError, match="Unsupported RLlib algorithm config"):
        RLlibIntegrationLayer._import_algorithm_config_class("unknown")


def test_import_algorithm_class_rejects_unknown_algorithm():
    with pytest.raises(KeyError, match="Unsupported RLlib algorithm"):
        RLlibIntegrationLayer._import_algorithm_class("unknown")


def test_build_algorithm_config_reports_missing_rllib_cleanly(monkeypatch):
    cfg = RLlibIntegrationConfig(algorithm="PPO")
    layer = RLlibIntegrationLayer(cfg)

    real_import = __import__("builtins").__dict__["__import__"]

    def fake_import(name, *args, **kwargs):
        if name.startswith("ray.rllib"):
            raise ImportError("ray intentionally absent in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="RLlib is not installed"):
        layer.build_algorithm_config()


def test_load_checkpoint_requires_checkpoint():
    layer = RLlibIntegrationLayer(RLlibIntegrationConfig())
    with pytest.raises(ValueError, match="checkpoint is required"):
        layer.load_algorithm_from_checkpoint()


def test_backend_from_algorithm_returns_compatible_backend():
    algorithm = FakeAlgorithm()
    layer = RLlibIntegrationLayer(RLlibIntegrationConfig())
    backend = layer.backend_from_algorithm(algorithm)
    assert isinstance(backend, RLlibAlgorithmBackendAdapter)


def test_backend_from_checkpoint_delegates_to_loader(monkeypatch):
    layer = RLlibIntegrationLayer(
        RLlibIntegrationConfig(checkpoint="checkpoint/path")
    )
    fake_algorithm = FakeAlgorithm()
    monkeypatch.setattr(layer, "load_algorithm_from_checkpoint", lambda: fake_algorithm)

    backend = layer.backend_from_checkpoint()
    assert isinstance(backend, RLlibAlgorithmBackendAdapter)


# -----------------------------------------------------------------------------
# Provider
# -----------------------------------------------------------------------------
def test_provider_requires_integration():
    with pytest.raises(ValueError, match="integration is required"):
        RLlibModelProvider(None)


def test_provider_uses_injected_algorithm():
    layer = RLlibIntegrationLayer(RLlibIntegrationConfig())
    algorithm = FakeAlgorithm()
    provider = RLlibModelProvider(layer, algorithm=algorithm)

    class FactoryConfig:
        pass

    backend = provider(FactoryConfig())
    assert isinstance(backend, RLlibAlgorithmBackendAdapter)


def test_provider_uses_checkpoint_when_no_algorithm_is_injected(monkeypatch):
    layer = RLlibIntegrationLayer(
        RLlibIntegrationConfig(checkpoint="checkpoint/path")
    )
    fake_algorithm = FakeAlgorithm()
    monkeypatch.setattr(layer, "backend_from_checkpoint", lambda: "backend")
    provider = RLlibModelProvider(layer)

    class FactoryConfig:
        pass

    assert provider(FactoryConfig()) == "backend"


# -----------------------------------------------------------------------------
# build_algorithm_config with fake RLlib modules
# -----------------------------------------------------------------------------
def test_build_algorithm_config_uses_current_builder_methods(monkeypatch):
    calls = []

    class FakeConfig:
        def api_stack(self, **kwargs):
            calls.append(("api_stack", kwargs))
            return self

        def framework(self, value):
            calls.append(("framework", value))
            return self

        def environment(self, **kwargs):
            calls.append(("environment", kwargs))
            return self

        def env_runners(self, **kwargs):
            calls.append(("env_runners", kwargs))
            return self

        def training(self, **kwargs):
            calls.append(("training", kwargs))
            return self

        def rl_module(self, **kwargs):
            calls.append(("rl_module", kwargs))
            return self

    fake_module = types.ModuleType("ray.rllib.algorithms.ppo")
    fake_module.PPOConfig = FakeConfig

    monkeypatch.setitem(sys.modules, "ray", types.ModuleType("ray"))
    monkeypatch.setitem(sys.modules, "ray.rllib", types.ModuleType("ray.rllib"))
    monkeypatch.setitem(sys.modules, "ray.rllib.algorithms", types.ModuleType("ray.rllib.algorithms"))
    monkeypatch.setitem(sys.modules, "ray.rllib.algorithms.ppo", fake_module)

    config = RLlibIntegrationConfig(
        algorithm="PPO",
        framework="torch",
        num_env_runners=2,
        algorithm_training={"gamma": 0.99},
        model_config={"fcnet_hiddens": [64, 64]},
    )
    layer = RLlibIntegrationLayer(config)
    env = object()

    result = layer.build_algorithm_config(env=env)

    assert isinstance(result, FakeConfig)
    assert any(name == "api_stack" for name, _ in calls)
    assert ("framework", "torch") in calls
    assert ("environment", {"env": env}) in calls
    assert ("env_runners", {"num_env_runners": 2}) in calls
    assert ("training", {"gamma": 0.99}) in calls
    assert ("rl_module", {"model_config": {"fcnet_hiddens": [64, 64]}}) in calls
