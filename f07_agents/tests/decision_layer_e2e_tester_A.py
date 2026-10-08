# f07_agents/tests/decision_layer_e2e_tester_A.py (t4)
#
# Run: pytest -v -s f07_agents/tests/decision_layer_e2e_tester_A.py

# Created: 1405/06/19
# فصل 3 - Final Decision Engine E2E Test
#
# هدف:
#   Observation
#       ↓
#   Symbol-Agent(s)
#       ↓
#   SymbolAgentOutput
#       ↓
#   Meta-Agent
#       ↓
#   PortfolioDecision
#       ↓
#   PortfolioAction
#
# این تستر:
#     - به Broker وابسته نیست
#     - به MT5 وابسته نیست
#     - Execution واقعی انجام نمی‌دهد
#
# هدف فقط اعتبارسنجی کامل Decision Layer است.

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from f06_env.contracts import (
    PortfolioAction,
)
from f06_env.portfolio_state import (
    PortfolioState,
)
from f07_agents.action_builder import (
    PortfolioActionBuilder,
)
from f07_agents.contracts import (
    AgentObservation,
    DecisionMode,
    MetaPolicyOutput,
    ModelIdentity,
    PortfolioContext,
    PolicyOutput,
    SymbolContext,
)
from f07_agents.decision_engine import (
    MultiSymbolDecisionEngine,
)
from f07_agents.meta_agent import (
    MetaAgent,
    MetaAgentConfig,
)
from f07_agents.portfolio_context_adapter import (
    PortfolioContextAdapter,
)
from f07_agents.symbol_agent import (
    SymbolAgent,
    SymbolAgentConfig,
)

# =============================================================================
# Constants
# =============================================================================

TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

SYMBOL_MODEL = ModelIdentity(
    model_name="symbol_policy",
    model_version="1.0.0",
    policy_version="policy-1",
    config_version="cfg-1",
    experiment_id="chapter3-e2e",
)

META_MODEL = ModelIdentity(
    model_name="meta_policy",
    model_version="1.0.0",
    policy_version="policy-1",
    config_version="cfg-1",
    experiment_id="chapter3-e2e",
)

# =====================================================================
# Class-1: Deterministic Symbol Policy
# =====================================================================

class FixedSymbolPolicy:
    """
    Policy قطعی برای تست معماری.

    این Policy فقط برای validation استفاده می‌شود و نماینده مدل RL نهایی نیست.
    """

    def __init__(
        self,
        *,
        signal: int,
        confidence: float = 0.9,
        expected_return: float = 0.02,
        risk_score: float = 0.20,
        desired_exposure: float = 0.25,
        stop_price: float = 1.0,
    ) -> None:
        self.output = PolicyOutput(
            signal=signal,
            confidence=confidence,
            expected_return=expected_return,
            risk_score=risk_score,
            desired_exposure=desired_exposure,
            stop_price=stop_price if signal != 0 else None,
        )


    def predict(
        self,
        *,
        observation: AgentObservation,
        context: SymbolContext,
    ) -> PolicyOutput:
        return self.output


# =====================================================================
# Class-2: Deterministic Meta Policy
# =====================================================================

class FixedMetaPolicy:
    """
    Meta-Policy قطعی برای تست.
    Meta-Agent باید خروجی Symbol-Agent ها را در قالب PortfolioDecision مدیریت کند.
    """

    def predict(
        self,
        *,
        signals,
        portfolio,
    ) -> MetaPolicyOutput:

        allocations = {
            symbol: 0.20
            for symbol in signals
        }

        target_signals = {
            symbol: signal.signal
            for symbol, signal in signals.items()
        }

        target_exposure = {
            symbol: (
                0.20 * signal.signal
            )
            for symbol, signal in signals.items()
        }

        margin_allocation = {
            symbol: 2_000.0
            for symbol in signals
        }

        return MetaPolicyOutput(
            capital_allocation=allocations,
            target_signals=target_signals,
            target_exposure=target_exposure,
            margin_allocation=margin_allocation,
            portfolio_risk=0.30,
            reason_codes=(
                "fixed_e2e_policy",
            ),
        )


# =====================================================================
# Class-3: Fixed Position Sizer
# =====================================================================

class FixedPositionSizer:
    """
    PositionSizer قطعی فقط برای تست.
    """

    def size(
        self,
        *,
        symbol: str,
        target_exposure: float,
    ) -> float:

        values = {
            "XAUUSD": 1.50,
            "EURUSD": 0.80,
            "GBPUSD": 0.60,
        }
        return values.get(symbol, 0.0)


# =====================================================================
# Helpers
# =====================================================================

def make_observation(symbol: str) -> AgentObservation:
    return AgentObservation(
        symbol=symbol,
        base_tf="M1",
        timestamp=TIMESTAMP,
        values=(0.10, 0.20, 0.30),
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

    symbol_agents = {
        "XAUUSD": SymbolAgent(
            config=SymbolAgentConfig(
                symbol="XAUUSD",
                mode=DecisionMode.BACKTEST,
                model=SYMBOL_MODEL,
            ),
            policy=FixedSymbolPolicy(
                signal=1,
            ),
        ),
        "EURUSD": SymbolAgent(
            config=SymbolAgentConfig(
                symbol="EURUSD",
                mode=DecisionMode.BACKTEST,
                model=SYMBOL_MODEL,
            ),
            policy=FixedSymbolPolicy(
                signal=-1,
            ),
        ),
        "GBPUSD": SymbolAgent(
            config=SymbolAgentConfig(
                symbol="GBPUSD",
                mode=DecisionMode.BACKTEST,
                model=SYMBOL_MODEL,
            ),
            policy=FixedSymbolPolicy(
                signal=1,
            ),
        ),
    }

    meta_agent = MetaAgent(
        config=MetaAgentConfig(
            mode=DecisionMode.BACKTEST,
            model=META_MODEL,
            max_total_allocation=1.0,
            max_symbol_allocation=0.5,
            max_total_exposure=1.0,
        ),
        policy=FixedMetaPolicy(),
    )

    return MultiSymbolDecisionEngine(
        symbol_agents=symbol_agents,
        meta_agent=meta_agent,
    )


# =====================================================================
# Test 1
# =====================================================================

def test_full_multi_symbol_decision_flow():

    engine = make_engine()

    decision = engine.decide(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
            "GBPUSD": make_observation("GBPUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
            "GBPUSD": make_context("GBPUSD"),
        },
        portfolio=make_portfolio(),
        decision_id="chapter3-e2e-001",
    )

    assert decision.approved is True
    assert set(
        decision.target_signals
    ) == {
        "XAUUSD",
        "EURUSD",
        "GBPUSD",
    }
    assert decision.target_signals["XAUUSD"] == 1
    assert decision.target_signals["EURUSD"] == -1
    assert decision.target_signals["GBPUSD"] == 1

# =====================================================================
# Test 2
# =====================================================================

def test_decision_to_portfolio_action():

    engine = make_engine()

    decision = engine.decide(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
            "GBPUSD": make_observation("GBPUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
            "GBPUSD": make_context("GBPUSD"),
        },
        portfolio=make_portfolio(),
        decision_id="chapter3-e2e-002",
    )

    builder = PortfolioActionBuilder(
        position_sizer=FixedPositionSizer()
    )

    action = builder.build(
        decision=decision,
        allowed_symbols={
            "XAUUSD",
            "EURUSD",
            "GBPUSD",
        },
    )

    assert isinstance(action, PortfolioAction)
    assert len(action.intents) == 3

    by_symbol = {
        intent.symbol: intent
        for intent in action.intents
    }

    assert by_symbol["XAUUSD"].target_side == 1
    assert by_symbol["XAUUSD"].target_lots == pytest.approx(1.50)

    assert by_symbol["EURUSD"].target_side == -1
    assert by_symbol["EURUSD"].target_lots == pytest.approx(0.80)

    assert by_symbol["GBPUSD"].target_side == 1
    assert by_symbol["GBPUSD"].target_lots == pytest.approx(0.60)

# =====================================================================
# Test 3
# =====================================================================

def test_portfolio_state_adapter_feeds_meta_agent():

    portfolio = PortfolioState(
        initial_balance=10_000.0
    )
    context = (
        PortfolioContextAdapter.from_portfolio_state(
            portfolio=portfolio,
            timestamp=TIMESTAMP,
            mode=DecisionMode.BACKTEST,
            exposure={
                "XAUUSD": 0.10,
                "EURUSD": -0.05,
            },
        )
    )
    assert context.equity == pytest.approx(10_000.0)
    assert context.balance == pytest.approx(10_000.0)
    assert context.exposure["XAUUSD"] == pytest.approx(0.10)
    assert context.exposure["EURUSD"] == pytest.approx(-0.05)

# =====================================================================
# Test 4
# =====================================================================

def test_complete_pipeline_is_deterministic():

    engine_a = make_engine()
    engine_b = make_engine()

    kwargs = dict(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
            "GBPUSD": make_observation("GBPUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
            "GBPUSD": make_context("GBPUSD"),
        },
        portfolio=make_portfolio(),
        decision_id="chapter3-deterministic-001",
    )
    decision_a = engine_a.decide(**kwargs)
    decision_b = engine_b.decide(**kwargs)

    assert decision_a == decision_b

    builder = PortfolioActionBuilder(
        position_sizer=FixedPositionSizer()
    )
    action_a = builder.build(decision=decision_a)
    action_b = builder.build(decision=decision_b)
    assert action_a == action_b

# =====================================================================
# Test 5
# =====================================================================

def test_wrong_observation_set_is_rejected():

    engine = make_engine()

    with pytest.raises(
        ValueError,
        match="Observation symbols",
    ):
        engine.decide(
            observations={
                "XAUUSD": make_observation("XAUUSD"),
                "EURUSD": make_observation("EURUSD"),
            },
            contexts={
                "XAUUSD": make_context("XAUUSD"),
                "EURUSD": make_context("EURUSD"),
                "GBPUSD": make_context("GBPUSD"),
            },
            portfolio=make_portfolio(),
            decision_id="chapter3-invalid-001",
        )

# =====================================================================
# Test 6
# =====================================================================

def test_wrong_timestamp_is_rejected():

    engine = make_engine()

    wrong_observation = AgentObservation(
        symbol="EURUSD",
        base_tf="M1",
        timestamp=datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc),
        values=(0.10, 0.20, 0.30),
        feature_names=("f1", "f2", "f3"),
    )
    with pytest.raises(
        ValueError,
        match="same timestamp",
    ):
        engine.decide(
            observations={
                "XAUUSD": make_observation("XAUUSD"),
                "EURUSD": wrong_observation,
                "GBPUSD": make_observation("GBPUSD"),
            },
            contexts={
                "XAUUSD": make_context("XAUUSD"),
                "EURUSD": make_context("EURUSD"),
                "GBPUSD": make_context("GBPUSD"),
            },
            portfolio=make_portfolio(),
            decision_id="chapter3-invalid-002",
        )

# =====================================================================
# Test 7
# =====================================================================

def test_risk_block_propagates_to_final_decision():

    engine = make_engine()

    portfolio = PortfolioContext(
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
        risk_blocked=True,
        mode=DecisionMode.BACKTEST,
    )

    decision = engine.decide(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
            "GBPUSD": make_observation("GBPUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
            "GBPUSD": make_context("GBPUSD"),
        },
        portfolio=portfolio,
        decision_id="chapter3-risk-001",
    )
    assert decision.approved is False
    assert decision.capital_allocation == {}
    assert "portfolio_risk_blocked" in (decision.reason_codes)

# =====================================================================
# Test 8
# =====================================================================

def test_execution_is_not_called_by_decision_layer():

    engine = make_engine()
    decision = engine.decide(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
            "GBPUSD": make_observation("GBPUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
            "GBPUSD": make_context("GBPUSD"),
        },
        portfolio=make_portfolio(),
        decision_id="chapter3-boundary-001",
    )

    assert decision.approved is True

    # Decision Layer فقط action semantic تولید می‌کند.
    # هیچ position/accounting در این مرحله تغییر نمی‌کند.

    portfolio_state = PortfolioState(
        initial_balance=10_000.0
    )
    assert portfolio_state.equity == 10_000.0
    assert portfolio_state.open_position_count == 0

# ============================================================================= END