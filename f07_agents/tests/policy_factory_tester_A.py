# f07_agents/tests/policy_factory_tester_A.py (t7)
#
# Run: pytest -v -s f07_agents/tests/policy_factory_tester_A.py
#
# Purpose:
#     Contract validation for policy_factory.py
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from f07_agents.contracts import (
    AgentObservation,
    DecisionMode,
    MetaPolicyOutput,
    PolicyOutput,
    PortfolioContext,
    SymbolAgentOutput,
)
from f07_agents.policy_factory import PolicyFactory, PolicySpec
from f07_agents.rl_policy_adapter import RLPolicyAdapterConfig

TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class FakeBackend:
    def __init__(self, result=None):
        self.result = result if result is not None else PolicyOutput(
            signal=1,
            confidence=0.90,
            expected_return=0.02,
            risk_score=0.10,
            desired_exposure=0.20,
            stop_price=1900.0,
        )
        self.calls = []

    def predict(self, observation, *, deterministic=True):
        self.calls.append((observation, deterministic))
        return self.result


def symbol_decoder(raw_action, observation, context):
    return raw_action if isinstance(raw_action, PolicyOutput) else PolicyOutput(
        signal=1,
        confidence=0.90,
        expected_return=0.02,
        risk_score=0.10,
        desired_exposure=0.20,
        stop_price=1900.0,
    )


def meta_output():
    return MetaPolicyOutput(
        capital_allocation={"XAUUSD": 0.20},
        target_signals={"XAUUSD": 1},
        target_exposure={"XAUUSD": 0.20},
        margin_allocation={"XAUUSD": 2_000.0},
        portfolio_risk=0.20,
        reason_codes=("factory-test",),
    )


def make_meta_decoder(raw_action, signals, portfolio, adapter):
    return meta_output()


def test_policy_spec_validates_and_copies_metadata():
    metadata = {"experiment": "A"}
    spec = PolicySpec(
        role="symbol",
        backend="FakeBackend",
        algorithm="PPO",
        policy="symbol-policy",
        deterministic=False,
        metadata=metadata,
    )

    metadata["experiment"] = "B"

    assert spec.role == "symbol"
    assert spec.backend == "FakeBackend"
    assert spec.algorithm == "PPO"
    assert spec.policy == "symbol-policy"
    assert spec.deterministic is False
    assert spec.metadata == {"experiment": "A"}


def test_policy_spec_rejects_invalid_role_and_empty_names():
    with pytest.raises(ValueError, match="role"):
        PolicySpec(
            role="invalid",
            backend="backend",
            algorithm="algo",
            policy="policy",
        )

    with pytest.raises(ValueError, match="backend"):
        PolicySpec(
            role="symbol",
            backend=" ",
            algorithm="algo",
            policy="policy",
        )


def test_backend_registration_and_replacement():
    factory = PolicyFactory()
    provider_a = lambda config: FakeBackend()
    provider_b = lambda config: FakeBackend()

    factory.register_backend("TestBackend", provider_a)
    assert factory.registered_backends() == ("testbackend",)

    with pytest.raises(ValueError, match="already registered"):
        factory.register_backend("testbackend", provider_b)

    factory.register_backend("TESTBACKEND", provider_b, replace=True)
    assert factory.registered_backends() == ("testbackend",)

    factory.unregister_backend(" TESTBACKEND ")
    assert factory.registered_backends() == ()


def test_create_symbol_policy_builds_adapter_with_expected_config():
    factory = PolicyFactory()
    captured: list[RLPolicyAdapterConfig] = []

    def provider(config: RLPolicyAdapterConfig):
        captured.append(config)
        return FakeBackend()

    factory.register_backend("backend-a", provider)
    spec = PolicySpec(
        role="symbol",
        backend="BACKEND-A",
        algorithm="PPO",
        policy="symbol-policy",
        deterministic=False,
        metadata={"x": 1},
    )

    policy = factory.create_symbol_policy(spec)

    assert policy.config.backend_name == "backend-a"
    assert policy.config.algorithm_name == "PPO"
    assert policy.config.policy_name == "symbol-policy"
    assert policy.config.deterministic is False
    assert policy.config.metadata == {"x": 1}
    assert captured[0] == policy.config


def test_create_meta_policy_requires_meta_role_and_observation_builder():
    factory = PolicyFactory()
    factory.register_backend("backend-a", lambda config: FakeBackend(meta_output()))

    symbol_spec = PolicySpec(
        role="symbol",
        backend="backend-a",
        algorithm="PPO",
        policy="symbol-policy",
    )
    with pytest.raises(ValueError, match="role='meta'"):
        factory.create_meta_policy(
            symbol_spec,
            observation_builder=lambda signals, portfolio: (1.0,),
            decoder=make_meta_decoder,
        )

    meta_spec = PolicySpec(
        role="meta",
        backend="backend-a",
        algorithm="SAC",
        policy="meta-policy",
    )
    with pytest.raises(ValueError, match="observation_builder"):
        factory.create_meta_policy(
            meta_spec,
            observation_builder=None,
            decoder=make_meta_decoder,
        )
