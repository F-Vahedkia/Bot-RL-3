# f06_env/portfolio_state_tester_A.py
# Run: pytest -v -s f06_env/portfolio_state_tester_A.py

import pytest

from f06_env.contracts import (
    EnvironmentConfig,
    PositionIntent,
    PortfolioAction,
)
from f06_env.portfolio_state import (
    PortfolioState,
    PositionState,
)


def test_environment_config():
    cfg = EnvironmentConfig(
        base_tf="M1",
        window_size=128,
        initial_balance=10_000,
    )

    assert cfg.base_tf == "M1"
    assert cfg.window_size == 128
    assert cfg.initial_balance == 10_000.0


def test_position_intent():
    x = PositionIntent(
        symbol="xauusd",
        target_side=1,
        target_lots=0.20,
    )

    assert x.symbol == "XAUUSD"
    assert x.target_side == 1
    assert x.target_lots == 0.20


def test_portfolio_action_rejects_duplicate_symbol():
    a = PositionIntent("XAUUSD", 1, 0.10)
    b = PositionIntent("xauusd", -1, 0.20)

    with pytest.raises(ValueError, match="Duplicate symbol"):
        PortfolioAction((a, b))


def test_portfolio_reset():
    p = PortfolioState(initial_balance=10_000)

    p.realized_pnl = 100
    p.unrealized_pnl = 50
    p.used_margin = 500
    p.step_count = 10

    p.reset()

    assert p.balance == 10_000
    assert p.equity == 10_000
    assert p.used_margin == 0
    assert p.free_margin == 10_000
    assert p.realized_pnl == 0
    assert p.unrealized_pnl == 0
    assert p.step_count == 0
    assert p.open_position_count == 0


def test_multi_symbol_positions():
    p = PortfolioState(initial_balance=10_000)

    p.set_position(
        PositionState(
            symbol="XAUUSD",
            side=1,
            lots=0.20,
        )
    )

    p.set_position(
        PositionState(
            symbol="EURUSD",
            side=-1,
            lots=0.10,
        )
    )

    assert p.open_position_count == 2
    assert set(p.open_position_symbols) == {"XAUUSD", "EURUSD"}


def test_accounting_and_drawdown():
    p = PortfolioState(initial_balance=10_000)

    p.update_mark_to_market(
        realized_delta=500,
        unrealized_total=200,
        used_margin=1_000,
    )

    assert p.balance == 10_500
    assert p.equity == 10_700
    assert p.free_margin == 9_700
    assert p.peak_equity == 10_700

    p.update_mark_to_market(
        unrealized_total=-700,
        used_margin=1_000,
    )

    assert p.equity == 9_800
    assert p.total_drawdown == pytest.approx(
        (10_700 - 9_800) / 10_700
    )

