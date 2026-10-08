# f07_agents/tests/f07_agents_production_tester_A.py (t5)
#
# Run: pytest -v -s f07_agents/tests/f07_agents_production_tester_A.py

# Created: 1405/06/19
# فصل 3 - Final Production Validation
# هدف:
#     اعتبارسنجی نهایی Decision Layer برای:
#         - modeهای مختلف
#         - model identity / lineage
#         - multi-symbol decision
#         - decision -> action boundary
#         - replay determinism
#         - risk blocking
#         - state isolation
#
# این تستر:
#     - Broker ندارد
#     - MT5 ندارد
#     - Execution واقعی ندارد
#
# فصل 3 باید فقط Decision Layer را تحویل دهد.

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from f06_env.contracts import (
    PortfolioAction
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
from f07_agents.symbol_agent import (
    SymbolAgent,
    SymbolAgentConfig,
)

# =============================================================================
# Constants
# =============================================================================

TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

SYMBOL_MODEL = ModelIdentity(
    model_name="symbol-policy",
    model_version="1.0.0",
    policy_version="symbol-policy-1",
    config_version="cfg-1",
    experiment_id="prod-validation",
)

META_MODEL = ModelIdentity(
    model_name="meta-policy",
    model_version="1.0.0",
    policy_version="meta-policy-1",
    config_version="cfg-1",
    experiment_id="prod-validation",
)

# =====================================================================
# Classes-1: Fixed Test Policies
# =====================================================================

class FixedSymbolPolicy:
    """
    Policy قطعی فقط برای validation.
    """

    def __init__(
        self,
        *,
        signal: int,
        exposure: float,
    ) -> None:

        self.signal = signal
        self.exposure = exposure

    def predict(
        self,
        *,
        observation,
        context,
    ) -> PolicyOutput:

        return PolicyOutput(
            signal=self.signal,
            confidence=0.90,
            expected_return=0.02 * self.signal,
            risk_score=0.10,
            desired_exposure=self.exposure,
            stop_price=1.0 if self.signal != 0 else None,
        )


# =====================================================================
# Classes-2: Fixed Test Policies
# =====================================================================

class FixedMetaPolicy:
    """
    Meta-Policy قطعی برای validation.
    """

    def predict(
        self,
        *,
        signals,
        portfolio,
    ) -> MetaPolicyOutput:

        return MetaPolicyOutput(
            capital_allocation={
                symbol: 0.20
                for symbol in signals
            },
            target_signals={
                symbol: signal.signal
                for symbol, signal in signals.items()
            },
            target_exposure={
                symbol: (signal.signal * 0.20)
                for symbol, signal in signals.items()
            },
            margin_allocation={
                symbol: 2_000.0
                for symbol in signals
            },
            portfolio_risk=0.20,
            reason_codes=(
                "production_validation",
            ),
        )


# =====================================================================
# Classes-3: Fixed Test Policies
# =====================================================================

class FixedPositionSizer:
    """
    Sizing قطعی فقط برای تست مرز Decision -> Environment.
    """

    def size(
        self,
        *,
        symbol: str,
        target_exposure: float,
    ) -> float:

        return {
            "XAUUSD": 1.50,
            "EURUSD": 0.80,
            "GBPUSD": 0.60,
        }.get(symbol, 0.0)


# =====================================================================
# Helpers
# =====================================================================

def make_observation(
    symbol: str,
    timestamp: datetime = TIMESTAMP,
) -> AgentObservation:

    return AgentObservation(
        symbol=symbol,
        base_tf="M1",
        timestamp=timestamp,
        values=(0.10, 0.20, 0.30),
        feature_names=("f1", "f2", "f3"),
    )


def make_context(
    symbol: str,
) -> SymbolContext:

    return SymbolContext(
        symbol=symbol,
        current_side=0,
        current_lots=0.0,
        exposure=0.0,
        drawdown=0.0,
        volatility=0.10,
        local_risk_score=0.10,
    )


def make_portfolio(
    *,
    mode: DecisionMode = DecisionMode.BACKTEST,
    risk_blocked: bool = False,
) -> PortfolioContext:

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
        risk_blocked=risk_blocked,
        mode=mode,
    )


def make_engine(
    mode: DecisionMode,
) -> MultiSymbolDecisionEngine:

    xau_agent = SymbolAgent(
        config=SymbolAgentConfig(
            symbol="XAUUSD",
            mode=mode,
            model=SYMBOL_MODEL,
        ),
        policy=FixedSymbolPolicy(
            signal=1,
            exposure=0.20,
        ),
    )

    eur_agent = SymbolAgent(
        config=SymbolAgentConfig(
            symbol="EURUSD",
            mode=mode,
            model=SYMBOL_MODEL,
        ),
        policy=FixedSymbolPolicy(
            signal=-1,
            exposure=0.20,
        ),
    )

    meta_agent = MetaAgent(
        config=MetaAgentConfig(
            mode=mode,
            model=META_MODEL,
            max_total_allocation=1.0,
            max_symbol_allocation=0.50,
            max_total_exposure=1.0,
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


# =====================================================================
# Test 1 - All modes
# =====================================================================

@pytest.mark.parametrize(
    "mode",
    list(DecisionMode),
)
def test_all_supported_modes(mode):

    engine = make_engine(mode)

    decision = engine.decide(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
        },
        portfolio=make_portfolio(
            mode=mode
        ),
        decision_id=f"mode-{mode.value}",
    )
    assert decision.approved is True
    assert decision.mode == mode
    assert decision.model == META_MODEL
    assert decision.decision_id == (
        f"mode-{mode.value}"
    )


# =====================================================================
# Test 2 - Model lineage
# =====================================================================

def test_model_lineage_is_preserved():

    engine = make_engine(
        DecisionMode.BACKTEST
    )

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
        decision_id="lineage-001",
    )
    assert decision.model.model_name == ("meta-policy")
    assert decision.model.model_version == ("1.0.0")
    assert decision.model.policy_version == ("meta-policy-1")
    assert decision.model.config_version == ("cfg-1")
    assert decision.model.experiment_id == ("prod-validation")


# =====================================================================
# Test 3 - Final Decision -> PortfolioAction
# =====================================================================

def test_final_decision_to_environment_action():

    engine = make_engine(
        DecisionMode.LIVE
    )
    decision = engine.decide(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
        },
        portfolio=make_portfolio(
            mode=DecisionMode.LIVE
        ),
        decision_id="live-boundary-001",
    )

    builder = PortfolioActionBuilder(
        position_sizer=FixedPositionSizer()
    )

    action = builder.build(
        decision=decision,
        allowed_symbols={
            "XAUUSD",
            "EURUSD",
        },
    )
    assert isinstance(action, PortfolioAction)
    assert len(action.intents) == 2
    xau = next(
        intent
        for intent in action.intents
        if intent.symbol == "XAUUSD"
    )
    eur = next(
        intent
        for intent in action.intents
        if intent.symbol == "EURUSD"
    )
    assert xau.target_side == 1
    assert xau.target_lots == pytest.approx(1.50)
    assert eur.target_side == -1
    assert eur.target_lots == pytest.approx(0.80)


# =====================================================================
# Test 4 - Risk block
# =====================================================================

def test_risk_block_never_creates_trading_action():

    engine = make_engine(
        DecisionMode.LIVE
    )

    decision = engine.decide(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
        },
        portfolio=make_portfolio(
            mode=DecisionMode.LIVE,
            risk_blocked=True,
        ),
        decision_id="risk-block-001",
    )

    assert decision.approved is False

    builder = PortfolioActionBuilder(
        position_sizer=FixedPositionSizer()
    )

    action = builder.build(
        decision=decision
    )
    assert action.intents == ()


# =====================================================================
# Test 5 - Replay
# =====================================================================

def test_identical_decision_cycles_are_reproducible():

    engine_a = make_engine(
        DecisionMode.REPLAY
    )
    engine_b = make_engine(
        DecisionMode.REPLAY
    )
    kwargs = dict(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
        },
        portfolio=make_portfolio(
            mode=DecisionMode.REPLAY
        ),
        decision_id="replay-001",
    )
    decision_a = engine_a.decide(**kwargs)
    decision_b = engine_b.decide(**kwargs)
    assert decision_a == decision_b


# =====================================================================
# Test 6 - Decision history
# =====================================================================

def test_last_decision_cycle_is_auditable():

    engine = make_engine(
        DecisionMode.EVAL
    )
    decision = engine.decide(
        observations={
            "XAUUSD": make_observation("XAUUSD"),
            "EURUSD": make_observation("EURUSD"),
        },
        contexts={
            "XAUUSD": make_context("XAUUSD"),
            "EURUSD": make_context("EURUSD"),
        },
        portfolio=make_portfolio(
            mode=DecisionMode.EVAL
        ),
        decision_id="audit-001",
    )
    assert engine.decision_count == 1
    assert engine.last_decision == decision
    assert (
        engine.get_last_symbol_output(
            "XAUUSD"
        ).decision_id
        == "audit-001"
    )
    assert (
        engine.get_last_symbol_output(
            "EURUSD"
        ).decision_id
        == "audit-001"
    )


# =====================================================================
# Test 7 - State isolation
# =====================================================================

def test_symbol_states_remain_isolated():

    engine = make_engine(
        DecisionMode.BACKTEST
    )
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
        decision_id="isolation-001",
    )
    xau_state = (
        engine
        .get_symbol_agent("XAUUSD")
        .state
    )
    eur_state = (
        engine
        .get_symbol_agent("EURUSD")
        .state
    )
    assert xau_state.last_signal == 1
    assert eur_state.last_signal == -1
    assert xau_state.decision_count == 1
    assert eur_state.decision_count == 1


# =====================================================================
# Test 8 - Reset
# =====================================================================

def test_complete_engine_reset():

    engine = make_engine(
        DecisionMode.BACKTEST
    )
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
    assert (
        engine.get_symbol_agent("XAUUSD")
        .state.decision_count
        == 0
    )
    assert (
        engine.get_symbol_agent("EURUSD")
        .state.decision_count
        == 0
    )


# ============================================================================= END