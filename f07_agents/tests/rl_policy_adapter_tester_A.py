# f07_agents/tests/rl_policy_adapter_tester_A.py (t6)
#
# Run: pytest -v -s f07_agents/tests/rl_policy_adapter_tester_A.py
#
# Purpose:
#     Contract validation for rl_policy_adapter.py
#
#     Backend-native predictions are intentionally decoded through explicit
#     decoder functions. No RL framework or algorithm is hard-coded here.
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from f07_agents.contracts import (
    AgentObservation,
    DecisionMode,
    MetaPolicyOutput,
    PolicyOutput,
    PortfolioContext,
    SymbolAgentOutput,
    SymbolContext,
)
from f07_agents.rl_policy_adapter import (
    RLMetaPolicyAdapter,
    RLModelPolicyAdapter,
    RLPolicyAdapterConfig,
)

TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_observation(symbol: str = "XAUUSD") -> AgentObservation:
    return AgentObservation(
        symbol=symbol,
        base_tf="M1",
        timestamp=TIMESTAMP,
        values=(0.10, 0.20, 0.30),
        feature_names=("f1", "f2", "f3"),
    )


def make_context(symbol: str = "XAUUSD") -> SymbolContext:
    return SymbolContext(
        symbol=symbol,
        current_side=0,
        current_lots=0.0,
        exposure=0.0,
        drawdown=0.0,
        volatility=0.10,
        local_risk_score=0.10,
    )


def make_portfolio() -> PortfolioContext:
    return PortfolioContext(
        timestamp=TIMESTAMP,
        equity=10_000.0,
        balance=10_000.0,
        used_margin=1_000.0,
        free_margin=9_000.0,
        margin_level=10.0,
        drawdown=0.0,
        daily_drawdown=0.0,
        exposure={"XAUUSD": 0.10},
        concentration={},
        correlation={},
        risk_blocked=False,
        mode=DecisionMode.BACKTEST,
    )


class FakeBackend:
    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[object, bool]] = []

    def predict(self, observation, *, deterministic: bool = True):
        self.calls.append((observation, deterministic))
        return self.result


def make_symbol_output() -> PolicyOutput:
    return PolicyOutput(
        signal=1,
        confidence=0.90,
        expected_return=0.02,
        risk_score=0.10,
        desired_exposure=0.20,
        stop_price=1900.0,
    )


def make_symbol_decoder():
    def decoder(raw_action, observation, context):
        assert raw_action == "backend-action"
        assert observation.symbol == "XAUUSD"
        return make_symbol_output()

    return decoder


def make_symbol_agent_output(symbol: str, signal: int) -> SymbolAgentOutput:
    return SymbolAgentOutput(
        symbol=symbol,
        timestamp=TIMESTAMP,
        mode=DecisionMode.BACKTEST,
        signal=signal,
        confidence=0.90,
        expected_return=0.02 * signal,
        risk_score=0.10,
        desired_exposure=0.20,
        stop_price=1900.0 if signal > 0 else 2000.0,
        model=None,
        decision_id="symbol-001",
    )


def test_symbol_adapter_default_decoder_accepts_policy_output():
    expected = make_symbol_output()
    backend = FakeBackend((expected, {"unused": True}))
    adapter = RLModelPolicyAdapter(
        backend,
        config=RLPolicyAdapterConfig(
            backend_name="test-backend",
            algorithm_name="test-algorithm",
            policy_name="test-policy",
            deterministic=False,
        ),
    )

    result = adapter.predict(
        observation=make_observation(),
        context=make_context(),
    )

    assert result == expected
    raw_observation, deterministic = backend.calls[0]
    assert isinstance(raw_observation, np.ndarray)
    assert raw_observation.dtype == np.float64
    assert raw_observation.tolist() == [0.10, 0.20, 0.30]
    assert deterministic is False


def test_symbol_adapter_uses_explicit_decoder_for_backend_native_action():
    backend = FakeBackend("backend-action")
    adapter = RLModelPolicyAdapter(
        backend,
        config=RLPolicyAdapterConfig(
            backend_name="test-backend",
            algorithm_name="test-algorithm",
            policy_name="test-policy",
        ),
        decoder=make_symbol_decoder(),
    )

    result = adapter.predict(
        observation=make_observation(),
        context=make_context(),
    )

    assert result.signal == 1
    assert result.desired_exposure == pytest.approx(0.20)
    assert result.stop_price == pytest.approx(1900.0)


def test_symbol_adapter_rejects_backend_native_action_without_decoder():
    adapter = RLModelPolicyAdapter(
        FakeBackend(1),
        config=RLPolicyAdapterConfig(
            backend_name="test-backend",
            algorithm_name="test-algorithm",
            policy_name="test-policy",
        ),
    )

    with pytest.raises(TypeError, match="No SymbolDecoder"):
        adapter.predict(
            observation=make_observation(),
            context=make_context(),
        )


def test_agent_observation_rejects_non_finite_values_before_adapter():
    with pytest.raises(
        ValueError,
        match="observation contains non-finite values",
    ):
        AgentObservation(
            symbol="XAUUSD",
            base_tf="M1",
            timestamp=TIMESTAMP,
            values=(0.10, float("nan"), 0.30),
            feature_names=("f1", "f2", "f3"),
        )


def test_meta_adapter_builds_numeric_observation_and_decodes_output():
    signals = {
        "XAUUSD": make_symbol_agent_output("XAUUSD", 1),
        "EURUSD": make_symbol_agent_output("EURUSD", -1),
    }
    portfolio = make_portfolio()
    expected = MetaPolicyOutput(
        capital_allocation={"XAUUSD": 0.30, "EURUSD": 0.20},
        target_signals={"XAUUSD": 1, "EURUSD": -1},
        target_exposure={"XAUUSD": 0.25, "EURUSD": -0.20},
        margin_allocation={"XAUUSD": 3_000.0, "EURUSD": 2_000.0},
        portfolio_risk=0.20,
        reason_codes=("test",),
    )

    def builder(received_signals, received_portfolio):
        assert received_signals is signals
        assert received_portfolio is portfolio
        return (1.0, -0.5, 0.25)

    def decoder(raw_action, received_signals, received_portfolio, adapter):
        assert raw_action == "meta-action"
        assert received_signals is signals
        assert received_portfolio is portfolio
        assert adapter.config.algorithm_name == "test-algorithm"
        return expected

    backend = FakeBackend(("meta-action", {"unused": True}))
    adapter = RLMetaPolicyAdapter(
        backend,
        config=RLPolicyAdapterConfig(
            backend_name="test-backend",
            algorithm_name="test-algorithm",
            policy_name="meta-policy",
            deterministic=False,
        ),
        observation_builder=builder,
        decoder=decoder,
    )

    result = adapter.predict(signals=signals, portfolio=portfolio)

    assert result == expected
    raw_observation, deterministic = backend.calls[0]
    assert raw_observation.dtype == np.float64
    assert raw_observation.tolist() == [1.0, -0.5, 0.25]
    assert deterministic is False


def test_meta_adapter_requires_non_empty_signals():
    adapter = RLMetaPolicyAdapter(
        FakeBackend(None),
        config=RLPolicyAdapterConfig(
            backend_name="test-backend",
            algorithm_name="test-algorithm",
            policy_name="meta-policy",
        ),
        observation_builder=lambda signals, portfolio: (1.0,),
    )

    with pytest.raises(ValueError, match="signals must not be empty"):
        adapter.predict(signals={}, portfolio=make_portfolio())


def test_meta_adapter_rejects_non_numeric_observation_builder_result():
    adapter = RLMetaPolicyAdapter(
        FakeBackend(None),
        config=RLPolicyAdapterConfig(
            backend_name="test-backend",
            algorithm_name="test-algorithm",
            policy_name="meta-policy",
        ),
        observation_builder=lambda signals, portfolio: (1.0, "bad"),
    )

    signals = {"XAUUSD": make_symbol_agent_output("XAUUSD", 1)}

    with pytest.raises(TypeError, match="meta observation must be numeric"):
        adapter.predict(signals=signals, portfolio=make_portfolio())

