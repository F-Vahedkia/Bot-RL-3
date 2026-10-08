# f07_agents/tests/decision_engine_tester_A.py (t2)
#
# Run: pytest -v -s f07_agents/tests/decision_engine_tester_A.py

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from f07_agents.contracts import (
    AgentObservation,
    DecisionMode,
    MetaPolicyOutput,
    ModelIdentity,
    PortfolioContext,
    PolicyOutput,
    SymbolAgentOutput,
    SymbolContext,
)
from f07_agents.decision_engine import (
    MultiSymbolDecisionEngine,
)
from f07_agents.meta_agent import (
    MetaAgent,
    MetaAgentConfig,
    MetaPolicy,
)
from f07_agents.symbol_agent import (
    SymbolAgent,
    SymbolAgentConfig,
    SymbolPolicy,
)


TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

SYMBOL_MODEL = ModelIdentity(
    model_name="symbol_policy",
    model_version="1.0.0",
    policy_version="policy-1",
)

META_MODEL = ModelIdentity(
    model_name="meta_policy",
    model_version="1.0.0",
    policy_version="policy-1",
)

# =============================================================================
# Class-1
# =============================================================================

class FixedSymbolPolicy:
    """
    Policy قطعی فقط برای تست معماری.
    """

    def __init__(self, signal: int) -> None:
        self.signal = signal

    def predict(
        self,
        *,
        observation,
        context,
    ) -> PolicyOutput:
        stop_prices = {
            "XAUUSD": 2300.0,
            "EURUSD": 1.1700,
        }
        stop_price = stop_prices.get(
            observation.symbol
        )
        return PolicyOutput(
            signal=self.signal,
            confidence=0.9,
            expected_return=0.01,
            risk_score=0.10,
            desired_exposure=0.20,
            stop_price=stop_price,
        )


# =============================================================================
# Class-2
# =============================================================================

class FixedMetaPolicy:
    """
    Meta-Policy قطعی برای تست.
    """

    def predict(
        self,
        *,
        signals,
        portfolio,
    ) -> MetaPolicyOutput:

        target_signals = {
            symbol: signal.signal
            for symbol, signal in signals.items()
        }

        allocations = {
            symbol: 0.20
            for symbol in signals
        }

        exposures = {
            symbol: (
                0.20
                * signal.signal
            )
            for symbol, signal in signals.items()
        }

        margins = {
            symbol: 2_000.0
            for symbol in signals
        }

        return MetaPolicyOutput(
            capital_allocation=allocations,
            target_signals=target_signals,
            target_exposure=exposures,
            margin_allocation=margins,
            portfolio_risk=0.30,
            reason_codes=("fixed_test_policy",),
        )


# =============================================================================
# Functions
# =============================================================================

def make_observation(symbol: str) -> AgentObservation:
    return AgentObservation(
        symbol=symbol,
        base_tf="M1",
        timestamp=TIMESTAMP,
        values=(0.1, 0.2, 0.3),
        feature_names=("f1", "f2", "f3"),
    )


def make_context(symbol: str) -> SymbolContext:
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
        exposure={},
        concentration={},
        correlation={},
        risk_blocked=False,
        mode=DecisionMode.BACKTEST,
    )


def make_engine() -> MultiSymbolDecisionEngine:

    xau_agent = SymbolAgent(
        config=SymbolAgentConfig(
            symbol="XAUUSD",
            mode=DecisionMode.BACKTEST,
            model=SYMBOL_MODEL,
        ),
        policy=FixedSymbolPolicy(
            signal=1
        ),
    )

    eur_agent = SymbolAgent(
        config=SymbolAgentConfig(
            symbol="EURUSD",
            mode=DecisionMode.BACKTEST,
            model=SYMBOL_MODEL,
        ),
        policy=FixedSymbolPolicy(
            signal=-1
        ),
    )

    meta_agent = MetaAgent(
        config=MetaAgentConfig(
            mode=DecisionMode.BACKTEST,
            model=META_MODEL,
        ),
        policy=FixedMetaPolicy(),
    )

    return MultiSymbolDecisionEngine(
        symbol_agents={
            "XAUUSD": xau_agent,
            "EURUSD": eur_agent,
        },
        meta_agent=meta_agent,
    )


def test_multi_symbol_decision_flow():

    engine = make_engine()

    decision = engine.decide(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
        },
        portfolio=make_portfolio(),
        decision_id="decision-001",
    )

    assert decision.approved is True
    assert decision.target_signals["XAUUSD"] == 1
    assert decision.target_signals["EURUSD"] == -1
    assert set(
        decision.capital_allocation
    ) == {
        "XAUUSD",
        "EURUSD",
    }


def test_symbol_isolation():

    engine = make_engine()
    decision = engine.decide(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
        },
        portfolio=make_portfolio(),
        decision_id="decision-002",
    )

    xau_agent = engine.get_symbol_agent("XAUUSD")

    eur_agent = engine.get_symbol_agent("EURUSD")

    assert (xau_agent.state.last_signal == 1)
    assert (eur_agent.state.last_signal == -1)
    assert (
        decision.target_signals["XAUUSD"]
        !=
        decision.target_signals["EURUSD"]
    )


def test_missing_symbol_is_rejected():

    engine = make_engine()
    try:
        engine.decide(
            observations={
                "XAUUSD": make_observation("XAUUSD"),
            },
            contexts={
                "XAUUSD": make_context("XAUUSD"),
                "EURUSD": make_context("EURUSD"),
            },
            portfolio=make_portfolio(),
            decision_id="decision-003",
        )
    except ValueError as exc:
        assert ("Observation symbols" in str(exc))
    else:
        raise AssertionError("Missing symbol was not rejected")


def test_symbol_keys_remove_spaces_without_case_normalization():
    engine = make_engine()

    decision = engine.decide(
        observations={
            "XAU USD": make_observation("XAUUSD"),
            "EUR USD": make_observation("EURUSD"),
        },
        contexts={
            "XAU USD": make_context("XAUUSD"),
            "EUR USD": make_context("EURUSD"),
        },
        portfolio=make_portfolio(),
        decision_id="symbol-space-normalization-001",
    )

    assert decision.approved is True
    assert set(decision.target_signals) == {
        "XAUUSD",
        "EURUSD",
    }


def test_symbol_case_is_not_normalized():
    engine = make_engine()

    with pytest.raises(
        ValueError,
        match="Observation symbols",
    ):
        engine.decide(
            observations={
                "xauusd": make_observation("XAUUSD"),
                "EURUSD": make_observation("EURUSD"),
            },
            contexts={
                "xauusd": make_context("XAUUSD"),
                "EURUSD": make_context("EURUSD"),
            },
            portfolio=make_portfolio(),
            decision_id="symbol-case-sensitive-001",
        )


def test_engine_is_deterministic():

    engine_a = make_engine()
    engine_b = make_engine()

    kwargs = dict(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
        },
        portfolio=make_portfolio(),
        decision_id="decision-004",
    )
    first = engine_a.decide(**kwargs)
    second = engine_b.decide(**kwargs)
    assert first == second


def test_decision_cycle_requires_shared_timestamp():

    engine = make_engine()
    observations = {
        "XAUUSD": make_observation("XAUUSD"),
        "EURUSD": AgentObservation(
            symbol="EURUSD",
            base_tf="M1",
            timestamp=datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc),
            values=(0.1, 0.2, 0.3),
            feature_names=("f1", "f2", "f3"),
        ),
    }
    contexts = {
        "XAUUSD": make_context("XAUUSD"),
        "EURUSD": make_context("EURUSD"),
    }
    try:
        engine.decide(
            observations=observations,
            contexts=contexts,
            portfolio=make_portfolio(),
            decision_id="timestamp-001",
        )
    except ValueError as exc:
        assert "same timestamp" in str(exc)
    else:
        raise AssertionError("Mismatched timestamps were not rejected")


def test_decision_cycle_requires_portfolio_timestamp_match():

    engine = make_engine()

    portfolio = PortfolioContext(
        timestamp=datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc),
        equity=10_000.0,
        balance=10_000.0,
        used_margin=1_000.0,
        free_margin=9_000.0,
        margin_level=10.0,
        drawdown=0.0,
        daily_drawdown=0.0,
        exposure={},
        concentration={},
        correlation={},
        risk_blocked=False,
        mode=DecisionMode.BACKTEST,
    )

    try:
        engine.decide(
            observations={
                "XAUUSD": make_observation("XAUUSD"),
                "EURUSD": make_observation("EURUSD"),
            },
            contexts={
                "XAUUSD": make_context("XAUUSD"),
                "EURUSD": make_context("EURUSD"),
            },
            portfolio=portfolio,
            decision_id="timestamp-002",
        )
    except ValueError as exc:
        assert "PortfolioContext timestamp" in str(exc)
    else:
        raise AssertionError("Portfolio timestamp mismatch was not rejected")


def test_decision_outputs_are_kept_for_audit():

    engine = make_engine()
    decision = engine.decide(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
        },
        portfolio=make_portfolio(),
        decision_id="audit-001",
    )
    assert engine.decision_count == 1
    assert engine.last_decision == decision
    assert (
        engine.get_last_symbol_output("XAUUSD").decision_id
        == "audit-001"
    )
    assert (
        engine.get_last_symbol_output("EURUSD").decision_id
        == "audit-001"
    )


def test_engine_reset_clears_runtime_state():

    engine = make_engine()
    engine.decide(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
        },
        portfolio=make_portfolio(),
        decision_id="reset-001",
    )
    engine.reset()
    assert engine.decision_count == 0
    assert engine.last_decision is None
    assert engine.last_symbol_outputs == {}

# ============================================================================= END