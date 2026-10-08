# f07_agents/tests/action_builder_tester_A.py (between t2,t3)
#
# Run: pytest -v -s f07_agents/tests/action_builder_tester_A_deleted.py

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from datetime import datetime, timezone
import pytest

from f06_env.contracts import (
    PortfolioAction,
    PositionIntent,
)
from f07_agents.action_builder import (
    PortfolioActionBuilder,
)
from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PortfolioDecision,
)

# =============================================================================
# ای هوش مصنوعی chatGPT! نمیدانم این جا چی بنویسم؟
# =============================================================================

TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

MODEL = ModelIdentity(
    model_name="meta",
    model_version="1.0.0",
    policy_version="policy-1",
)

# =============================================================================
# Class-1
# =============================================================================

class FixedPositionSizer:
    """
    PositionSizer قطعی فقط برای تست.
    """

    def __init__(
        self,
        *,
        lots_by_symbol: dict[str, float],
    ) -> None:

        self.lots_by_symbol = {
            str(symbol).replace(" ", ""): float(lots)
            for symbol, lots
            in lots_by_symbol.items()
        }


    def size(
        self,
        *,
        symbol: str,
        target_exposure: float,
    ) -> float:

        return self.lots_by_symbol.get(
            symbol,
            0.0,
        )


# =============================================================================
# Functions
# =============================================================================
def make_decision_old1() -> PortfolioDecision:

    return PortfolioDecision(
        approved=True,
        timestamp=TIMESTAMP,
        mode=DecisionMode.BACKTEST,
        decision_id="decision-001",
        capital_allocation={
            "XAUUSD": 0.30,
            "EURUSD": 0.20,
        },
        margin_allocation={
            "XAUUSD": 3000.0,
            "EURUSD": 2000.0,
        },
        target_exposure={
            "XAUUSD": 0.30,
            "EURUSD": -0.20,
        },
        target_signals={
            "XAUUSD": 1,
            "EURUSD": -1,
        },
        portfolio_risk=0.30,
        reason_codes=(
            "approved",
        ),
        model=MODEL,
    )


def make_decision() -> PortfolioDecision:

    return PortfolioDecision(
        approved=True,
        timestamp=TIMESTAMP,
        mode=DecisionMode.BACKTEST,
        decision_id="decision-001",
        capital_allocation={
            "XAUUSD": 0.30,
            "EURUSD": 0.20,
        },
        margin_allocation={
            "XAUUSD": 3000.0,
            "EURUSD": 2000.0,
        },
        target_exposure={
            "XAUUSD": 0.30,
            "EURUSD": -0.20,
        },
        target_signals={
            "XAUUSD": 1,
            "EURUSD": -1,
        },
        target_stop_price={
            "XAUUSD": 2290.0,
            "EURUSD": 1.1800,
        },
        portfolio_risk=0.30,
        reason_codes=(
            "approved",
        ),
        model=MODEL,
    )


def test_build_multi_symbol_action():

    builder = PortfolioActionBuilder(
        position_sizer=FixedPositionSizer(
            lots_by_symbol={
                "XAUUSD": 1.5,
                "EURUSD": 0.8,
            }
        )
    )
    action = builder.build(
        decision=make_decision(),
        allowed_symbols={
            "XAUUSD",
            "EURUSD",
        },
    )
    assert isinstance(action, PortfolioAction)
    assert len(action.intents) == 2
    xau = next(
        x
        for x in action.intents
        if x.symbol == "XAUUSD"
    )
    eur = next(
        x
        for x in action.intents
        if x.symbol == "EURUSD"
    )
    assert xau.target_side == 1
    assert xau.target_lots == pytest.approx(1.5)
    assert xau.stop_price == pytest.approx(2290.0)

    assert eur.target_side == -1
    assert eur.target_lots == pytest.approx(0.8)
    assert eur.stop_price == pytest.approx(1.1800)


def test_rejected_decision_produces_empty_action():

    decision = make_decision()
    rejected = PortfolioDecision(
        approved=False,
        timestamp=decision.timestamp,
        mode=decision.mode,
        decision_id=decision.decision_id,
        capital_allocation={},
        margin_allocation={},
        target_exposure={},
        target_signals={},
        target_stop_price={},
        portfolio_risk=0.0,
        reason_codes=("rejected",),
        model=decision.model,
    )
    builder = PortfolioActionBuilder(
        position_sizer=FixedPositionSizer(
            lots_by_symbol={}
        )
    )
    action = builder.build(
        decision=rejected
    )
    assert action.intents == ()


def test_unknown_symbol_is_rejected():

    builder = PortfolioActionBuilder(
        position_sizer=FixedPositionSizer(
            lots_by_symbol={
                "XAUUSD": 1.0,
                "GBPUSD": 1.0,
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="not allowed",
    ):
        builder.build(
            decision=make_decision(),
            allowed_symbols={
                "XAUUSD",
            },
        )


def test_flat_signal_always_has_zero_lots():

    decision = make_decision()

    decision = PortfolioDecision(
        approved=True,
        timestamp=decision.timestamp,
        mode=decision.mode,
        decision_id=decision.decision_id,
        capital_allocation={"XAUUSD": 0.30},
        margin_allocation={"XAUUSD": 3000.0},
        target_exposure={"XAUUSD": 0.0},
        target_signals={"XAUUSD": 0},
        target_stop_price={"XAUUSD": None},
        portfolio_risk=0.10,
        reason_codes=("flat",),
        model=decision.model,
    )
    builder = PortfolioActionBuilder(
        position_sizer=FixedPositionSizer(
            lots_by_symbol={
                "XAUUSD": 99.0,
            }
        )
    )
    action = builder.build(
        decision=decision
    )
    assert action.intents[0].target_side == 0
    assert action.intents[0].target_lots == 0.0
    assert action.intents[0].stop_price is None


def test_symbol_order_is_deterministic():

    builder = PortfolioActionBuilder(
        position_sizer=FixedPositionSizer(
            lots_by_symbol={
                "XAUUSD": 1.0,
                "EURUSD": 1.0,
            }
        )
    )

    action_a = builder.build(
        decision=make_decision()
    )

    action_b = builder.build(
        decision=make_decision()
    )

    assert action_a == action_b
    assert [
        intent.symbol
        for intent in action_a.intents
    ] == [
        "EURUSD",
        "XAUUSD",
    ]


def test_symbol_spaces_are_removed_without_case_normalization():
    decision = PortfolioDecision(
        approved=True,
        timestamp=TIMESTAMP,
        mode=DecisionMode.BACKTEST,
        decision_id="decision-symbol-policy",
        capital_allocation={
            "XA UU SD": 0.30,
        },
        margin_allocation={
            "XA UU SD": 3000.0,
        },
        target_exposure={
            "XA UU SD": 0.30,
        },
        target_signals={
            "XA UU SD": 1,
        },
        target_stop_price={
            "XA UU SD": 2290.0,
        },
        portfolio_risk=0.10,
        reason_codes=(
            "symbol-policy",
        ),
        model=MODEL,
    )

    builder = PortfolioActionBuilder(
        position_sizer=FixedPositionSizer(
            lots_by_symbol={
                "X A UU   S D   ": 1.25,
            }
        )
    )

    action = builder.build(
        decision=decision,
        allowed_symbols={
            " XA U U   S D   ",
        },
    )

    assert len(action.intents) == 1

    intent = action.intents[0]

    assert intent.symbol == "X A U  USD   ".replace(" ","")
    assert intent.target_side == 1
    assert intent.target_lots == pytest.approx(1.25)
    assert intent.stop_price == pytest.approx(2290.0)
    
# ============================================================================= END