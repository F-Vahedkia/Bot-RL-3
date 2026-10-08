# f07_agents/rl_model_backend_adapter_tester_A.py
#
# Tests for the SB3 backend boundary adapter.
# These tests do not require Stable-Baselines3 to be installed.

from __future__ import annotations

import numpy as np
import pytest

from f07_agents.rl_model_backend_adapter import (
    SB3BackendConfig,
    SB3ModelBackendAdapter,
    SB3ModelProvider,
)


class FakeModel:
    def __init__(self, result="ok"):
        self.calls = []
        self.result = result

    def predict(self, observation, *, deterministic=True):
        self.calls.append((observation.copy(), deterministic))
        return self.result


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def test_config_accepts_valid_values():
    cfg = SB3BackendConfig(
        algorithm="PPO",
        policy="MlpPolicy",
        device="cpu",
    )
    assert cfg.algorithm == "PPO"
    assert cfg.policy == "MlpPolicy"
    assert cfg.device == "cpu"


def test_config_rejects_empty_algorithm():
    with pytest.raises(ValueError, match="algorithm must be non-empty"):
        SB3BackendConfig(algorithm=" ")


def test_config_rejects_empty_policy():
    with pytest.raises(ValueError, match="policy must be non-empty"):
        SB3BackendConfig(algorithm="PPO", policy=" ")


def test_config_rejects_empty_device():
    with pytest.raises(ValueError, match="device must be non-empty"):
        SB3BackendConfig(algorithm="PPO", device=" ")


# -----------------------------------------------------------------------------
# Adapter construction
# -----------------------------------------------------------------------------
def test_adapter_requires_model():
    with pytest.raises(ValueError, match="model is required"):
        SB3ModelBackendAdapter(
            None,
            config=SB3BackendConfig(algorithm="PPO"),
        )


def test_adapter_requires_predict_method():
    class BadModel:
        pass

    with pytest.raises(TypeError, match="callable predict"):
        SB3ModelBackendAdapter(
            BadModel(),
            config=SB3BackendConfig(algorithm="PPO"),
        )


def test_adapter_stores_model_and_config():
    model = FakeModel()
    cfg = SB3BackendConfig(algorithm="PPO")
    adapter = SB3ModelBackendAdapter(model, config=cfg)
    assert adapter.model is model
    assert adapter.config == cfg


# -----------------------------------------------------------------------------
# Observation preparation
# -----------------------------------------------------------------------------
def test_prepare_observation_returns_float32_flat_array():
    result = SB3ModelBackendAdapter._prepare_observation([[1, 2], [3, 4]])
    assert result.dtype == np.float32
    assert result.shape == (4,)
    np.testing.assert_allclose(result, [1, 2, 3, 4])


def test_prepare_observation_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        SB3ModelBackendAdapter._prepare_observation([])


def test_prepare_observation_rejects_non_numeric():
    with pytest.raises(TypeError, match="must be numeric"):
        SB3ModelBackendAdapter._prepare_observation(["a", "b"])


def test_prepare_observation_rejects_nan():
    with pytest.raises(ValueError, match="finite"):
        SB3ModelBackendAdapter._prepare_observation([1.0, np.nan])


def test_prepare_observation_rejects_inf():
    with pytest.raises(ValueError, match="finite"):
        SB3ModelBackendAdapter._prepare_observation([1.0, np.inf])


# -----------------------------------------------------------------------------
# Predict boundary
# -----------------------------------------------------------------------------
def test_predict_forwards_prepared_observation_and_deterministic():
    model = FakeModel(result=(np.array([2]), None))
    adapter = SB3ModelBackendAdapter(
        model,
        config=SB3BackendConfig(algorithm="PPO"),
    )

    result = adapter.predict([1.0, 2.0], deterministic=False)

    assert isinstance(result, tuple)
    assert model.calls[0][0].dtype == np.float32
    np.testing.assert_allclose(model.calls[0][0], [1.0, 2.0])
    assert model.calls[0][1] is False


def test_predict_converts_deterministic_to_bool():
    model = FakeModel()
    adapter = SB3ModelBackendAdapter(
        model,
        config=SB3BackendConfig(algorithm="PPO"),
    )
    adapter.predict([1.0], deterministic=0)
    assert model.calls[-1][1] is False


# -----------------------------------------------------------------------------
# Provider
# -----------------------------------------------------------------------------
def test_provider_requires_loader():
    with pytest.raises(ValueError, match="loader is required"):
        SB3ModelProvider(None)


def test_provider_builds_backend_adapter():
    received = {}
    model = FakeModel()

    def loader(config):
        received["config"] = config
        return model

    provider = SB3ModelProvider(loader)

    class FactoryConfig:
        algorithm_name = "PPO"
        policy_name = "MlpPolicy"
        metadata = {"device": "cpu"}

    backend = provider(FactoryConfig())

    assert isinstance(backend, SB3ModelBackendAdapter)
    assert backend.model is model
    assert backend.config.algorithm == "PPO"
    assert backend.config.policy == "MlpPolicy"
    assert backend.config.device == "cpu"
    assert received["config"].device == "cpu"


def test_provider_uses_default_device_when_metadata_has_no_device():
    def loader(config):
        assert config.device == "auto"
        return FakeModel()

    provider = SB3ModelProvider(loader)

    class FactoryConfig:
        algorithm_name = "PPO"
        policy_name = "MlpPolicy"
        metadata = {}

    backend = provider(FactoryConfig())
    assert backend.config.device == "auto"


def test_provider_accepts_missing_policy_name_with_default():
    def loader(config):
        assert config.policy == "MlpPolicy"
        return FakeModel()

    provider = SB3ModelProvider(loader)

    class FactoryConfig:
        algorithm_name = "PPO"
        metadata = {}

    backend = provider(FactoryConfig())
    assert backend.config.policy == "MlpPolicy"


def test_provider_rejects_provider_returning_none():
    provider = SB3ModelProvider(lambda config: None)
    with pytest.raises(ValueError, match="model is required"):
        provider(
            type(
                "FactoryConfig",
                (),
                {
                    "algorithm_name": "PPO",
                    "policy_name": "MlpPolicy",
                    "metadata": {},
                },
            )()
        )
