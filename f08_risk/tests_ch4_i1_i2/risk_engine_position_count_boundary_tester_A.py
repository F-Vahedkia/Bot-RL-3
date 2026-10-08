# f08_risk/tests_ch4_i1_i2/risk_engine_position_count_boundary_tester_A.py (21)
#
# Run: pytest -v -s f08_risk/tests_ch4_i1_i2/risk_engine_position_count_boundary_tester_A.py

"""
Purpose:
    - Validate the portfolio-wide position-count hard boundary.

Contract:
    - RiskLimits.max_open_positions
    - RiskProjection.projected_position_count
    - RiskEngine -> MAX_OPEN_POSITIONS hard rejection

Important:
    This tester intentionally does NOT claim that max_positions_per_symbol can
    be enforced yet. The current RiskContext/SymbolRiskSnapshot contract exposes
    one aggregate net position per symbol (current_lots/current_side), not an
    explicit per-symbol position count. That separate contract gap must be solved
    before implementing a genuine per-symbol count guard.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PortfolioContext,
    PortfolioDecision,
)
from f08_risk.contracts import RiskDecisionStatus, RiskRequest
from f08_risk.limits import RiskLimits
from f08_risk.risk_context import (
    AccountRiskSnapshot,
    RiskContext,
    SymbolRiskSnapshot,
)
from f08_risk.risk_engine import RiskEngine


TS = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)

MODEL = ModelIdentity(
    model_name="test-model",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="exp-position-count",
)


def make_snapshot(symbol: str, exposure: float) -> SymbolRiskSnapshot:
    if exposure == 0.0:
        return SymbolRiskSnapshot(
            symbol=symbol,
            exposure=0.0,
            notional=0.0,
            used_margin=0.0,
            current_lots=0.0,
            current_side=0,
        )

    return SymbolRiskSnapshot(
        symbol=symbol,
        exposure=exposure,
        notional=1000.0,
        used_margin=250.0,
        current_lots=0.05,
        current_side=1 if exposure > 0.0 else -1,
    )


def make_context(
    *,
    symbols: dict[str, SymbolRiskSnapshot],
    open_position_count: int,
) -> RiskContext:
    equity = 10_000.0
    used_margin = 1_000.0

    correlation = {
        symbol: {
            other: (1.0 if symbol == other else 0.0)
            for other in symbols
        }
        for symbol in symbols
    }

    return RiskContext(
        timestamp=TS,
        account=AccountRiskSnapshot(
            balance=equity,
            equity=equity,
            used_margin=used_margin,
            free_margin=equity - used_margin,
            margin_level=1000.0,
            leverage=100.0,
            peak_equity=equity,
            day_start_equity=equity,
            open_position_count=open_position_count,
        ),
        symbols=symbols,
        correlation=correlation,
    )


def make_request(
    *,
    context: RiskContext,
    target_exposure: dict[str, float],
) -> RiskRequest:
    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="position-count-decision",
        capital_allocation={},
        margin_allocation={},
        target_exposure=target_exposure,
        target_signals={},
        portfolio_risk=0.10,
        reason_codes=(),
        model=MODEL,
    )

    portfolio = PortfolioContext(
        timestamp=TS,
        equity=context.account.equity,
        balance=context.account.balance,
        used_margin=context.account.used_margin,
        free_margin=context.account.free_margin,
        margin_level=context.account.margin_level,
        drawdown=0.0,
        daily_drawdown=0.0,
        exposure={
            symbol: snapshot.exposure
            for symbol, snapshot in context.symbols.items()
        },
        concentration={},
        correlation=context.correlation,
        risk_blocked=False,
        mode=DecisionMode.BACKTEST,
    )

    return RiskRequest(
        decision=decision,
        portfolio=portfolio,
        risk_context=context,
    )


def test_position_count_at_limit_can_be_maintained() -> None:
    context = make_context(
        symbols={"XAUUSD": make_snapshot("XAUUSD", 0.10)},
        open_position_count=1,
    )

    limits = RiskLimits(max_open_positions=1)

    result = RiskEngine(limits=limits).evaluate(
        make_request(
            context=context,
            target_exposure={"XAUUSD": 0.10},
        )
    )

    assert result.status == RiskDecisionStatus.APPROVED
    assert result.target_exposure["XAUUSD"] == pytest.approx(0.10)


def test_new_position_is_rejected_at_portfolio_position_limit() -> None:
    context = make_context(
        symbols={
            "XAUUSD": make_snapshot("XAUUSD", 0.10),
            "EURUSD": make_snapshot("EURUSD", 0.0),
        },
        open_position_count=1,
    )

    limits = RiskLimits(max_open_positions=1)

    result = RiskEngine(limits=limits).evaluate(
        make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.10,
                "EURUSD": 0.10,
            },
        )
    )

    assert result.status == RiskDecisionStatus.REJECTED
    assert any(
        violation.code == "MAX_OPEN_POSITIONS"
        for violation in result.violations
    )
    assert result.target_exposure == {}
    assert result.capital_allocation == {}
    assert result.margin_allocation == {}


def test_new_position_is_allowed_when_one_portfolio_slot_remains() -> None:
    context = make_context(
        symbols={
            "XAUUSD": make_snapshot("XAUUSD", 0.10),
            "EURUSD": make_snapshot("EURUSD", 0.0),
        },
        open_position_count=1,
    )

    limits = RiskLimits(max_open_positions=2)

    result = RiskEngine(limits=limits).evaluate(
        make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.10,
                "EURUSD": 0.10,
            },
        )
    )

    assert result.status != RiskDecisionStatus.REJECTED
    assert result.target_exposure["EURUSD"] == pytest.approx(0.10)


def test_closing_a_position_does_not_consume_a_position_slot() -> None:
    context = make_context(
        symbols={"XAUUSD": make_snapshot("XAUUSD", 0.10)},
        open_position_count=1,
    )

    limits = RiskLimits(max_open_positions=1)

    result = RiskEngine(limits=limits).evaluate(
        make_request(
            context=context,
            target_exposure={"XAUUSD": 0.0},
        )
    )

    assert result.status != RiskDecisionStatus.REJECTED
    assert result.target_exposure["XAUUSD"] == pytest.approx(0.0)


def test_position_limit_boundary_is_inclusive() -> None:
    context = make_context(
        symbols={
            "XAUUSD": make_snapshot("XAUUSD", 0.10),
            "EURUSD": make_snapshot("EURUSD", 0.0),
        },
        open_position_count=1,
    )

    limits = RiskLimits(max_open_positions=2)

    result = RiskEngine(limits=limits).evaluate(
        make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.10,
                "EURUSD": 0.10,
            },
        )
    )

    assert result.status != RiskDecisionStatus.REJECTED
    assert result.metadata["projected"]["projected_position_count"] <= 2
