# f07_agents/tests/portfolio_context_adapter_tester_A.py (t3)
#
# Run: pytest -v -s f07_agents/tests/portfolio_context_adapter_tester_A.py

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from f06_env.portfolio_state import (
    PortfolioState,
    PositionState,
)
from f07_agents.contracts import (
    DecisionMode,
)
from f07_agents.portfolio_context_adapter import (
    PortfolioContextAdapter,
)

# =============================================================================
#
# =============================================================================

TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

# =============================================================================
# Functions
# =============================================================================

def test_portfolio_state_to_context():

    portfolio = PortfolioState(
        initial_balance=10_000.0
    )
    portfolio.update_mark_to_market(
        realized_delta=100.0,
        unrealized_total=50.0,
        used_margin=1_000.0,
    )
    context = (
        PortfolioContextAdapter.from_portfolio_state(
            portfolio=portfolio,
            timestamp=TIMESTAMP,
            mode=DecisionMode.BACKTEST,
            exposure={
                "XAUUSD": 0.20,
                "EURUSD": -0.10,
            },
        )
    )
    assert context.equity == pytest.approx(10_150.0)
    assert context.balance == pytest.approx(10_100.0)
    assert context.used_margin == pytest.approx(1_000.0)
    assert context.free_margin == pytest.approx(9_150.0)
    assert context.margin_level == pytest.approx(10.15)
    assert context.exposure["XAUUSD"] == pytest.approx(0.20)
    assert context.exposure["EURUSD"] == pytest.approx(-0.10)


def test_adapter_does_not_modify_portfolio():

    portfolio = PortfolioState(
        initial_balance=10_000.0
    )
    portfolio.set_position(
        PositionState(
            symbol="XAUUSD",
            side=1,
            lots=1.0,
        )
    )
    before = (
        portfolio.balance,
        portfolio.equity,
        portfolio.used_margin,
        portfolio.open_position_count,
    )
    PortfolioContextAdapter.from_portfolio_state(
        portfolio=portfolio,
        timestamp=TIMESTAMP,
        mode=DecisionMode.LIVE,
        exposure={
            "XAUUSD": 1.0,
        },
    )
    after = (
        portfolio.balance,
        portfolio.equity,
        portfolio.used_margin,
        portfolio.open_position_count,
    )
    assert before == after


def test_timezone_aware_timestamp_required():

    portfolio = PortfolioState(
        initial_balance=10_000.0
    )
    naive_timestamp = datetime(2026, 1, 1, 12, 0)

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        PortfolioContextAdapter.from_portfolio_state(
            portfolio=portfolio,
            timestamp=naive_timestamp,
            mode=DecisionMode.BACKTEST,
            exposure={},
        )


def test_exposure_symbols_are_normalized():

    portfolio = PortfolioState(
        initial_balance=10_000.0
    )
    context = (
        PortfolioContextAdapter.from_portfolio_state(
            portfolio=portfolio,
            timestamp=TIMESTAMP,
            mode=DecisionMode.BACKTEST,
            exposure={
                " XA U U SD ": 0.30,
                "E U RUS D  ": -0.20,
            },
        )
    )
    assert set(
        context.exposure.keys()
    ) == {
        "XAUUSD",
        "EURUSD",
    }


def test_correlation_and_concentration_are_preserved():

    portfolio = PortfolioState(
        initial_balance=10_000.0
    )
    context = (
        PortfolioContextAdapter.from_portfolio_state(
            portfolio=portfolio,
            timestamp=TIMESTAMP,
            mode=DecisionMode.BACKTEST,
            exposure={
                "XAUUSD": 0.20,
                "EURUSD": 0.20,
            },
            concentration={
                "XAUUSD": 0.20,
                "EURUSD": 0.20,
            },
            correlation={
                "XAUUSD": {"EURUSD": 0.92,},
                "EURUSD": {"XAUUSD": 0.92,},
            },
        )
    )
    assert context.concentration["XAUUSD"] == pytest.approx(0.20)
    assert context.correlation["XAUUSD"]["EURUSD"] == pytest.approx(0.92)


def test_risk_block_is_preserved():

    portfolio = PortfolioState(
        initial_balance=10_000.0
    )
    context = (
        PortfolioContextAdapter.from_portfolio_state(
            portfolio=portfolio,
            timestamp=TIMESTAMP,
            mode=DecisionMode.LIVE,
            exposure={},
            risk_blocked=True,
        )
    )
    assert context.risk_blocked is True
    assert context.mode == DecisionMode.LIVE


# ============================================================================= END