# f08_risk/tests_ch4_i1_i2/risk_engine_symbol_position_count_boundary_tester_A.py (24)
#
# Run: pytest -v -s f08_risk/tests_ch4_i1_i2/risk_engine_symbol_position_count_boundary_tester_A.py

"""
# Purpose:
#   Validate the per-symbol open-position-count hard boundary in RiskEngine.
#
# Contract:
#   RiskLimits.max_positions_per_symbol
#   RiskProjection.projected per-symbol position count
#   RiskEngine -> MAX_POSITIONS_PER_SYMBOL hard rejection
#
# Important:
#   PortfolioDecision currently expresses target net exposure, not individual
#   order/ticket decomposition. Therefore this tester validates:
#
#       - current/projected count at the configured boundary
#       - flat -> one new position
#       - an already-over-limit symbol
#       - closing positions
#       - inclusive boundary
#
#   It intentionally does NOT infer creation of an additional ticket merely
#   from a target-exposure change.
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
from f08_risk.contracts import (
    RiskDecisionStatus,
    RiskRequest,
)
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
    experiment_id="exp-symbol-position-count",
)


def make_snapshot(
    symbol: str,
    *,
    exposure: float,
    position_count: int,
) -> SymbolRiskSnapshot:
    if exposure == 0.0:
        return SymbolRiskSnapshot(
            symbol=symbol,
            exposure=0.0,
            notional=0.0,
            used_margin=0.0,
            current_lots=0.0,
            current_side=0,
            position_count=position_count,
        )

    return SymbolRiskSnapshot(
        symbol=symbol,
        exposure=exposure,
        notional=1_000.0,
        used_margin=250.0,
        current_lots=0.20,
        current_side=1 if exposure > 0.0 else -1,
        position_count=position_count,
    )


def make_context(
    symbols: dict[str, SymbolRiskSnapshot],
) -> RiskContext:
    return RiskContext(
        timestamp=TS,
        account=AccountRiskSnapshot(
            balance=10_000.0,
            equity=10_000.0,
            used_margin=1_000.0,
            free_margin=9_000.0,
            margin_level=1_000.0,
            leverage=100.0,
            peak_equity=10_000.0,
            day_start_equity=10_000.0,
            open_position_count=sum(
                snapshot.position_count
                for snapshot in symbols.values()
            ),
        ),
        symbols=symbols,
        correlation={
            symbol: {
                other: (
                    1.0
                    if symbol == other
                    else 0.0
                )
                for other in symbols
            }
            for symbol in symbols
        },
    )


def make_portfolio_context(
    context: RiskContext,
) -> PortfolioContext:
    return PortfolioContext(
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


def make_request(
    *,
    context: RiskContext,
    target_exposure: dict[str, float],
) -> RiskRequest:
    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="symbol-position-count-decision",
        capital_allocation={},
        margin_allocation={},
        target_exposure=target_exposure,
        target_signals={},
        portfolio_risk=0.10,
        reason_codes=(),
        model=MODEL,
    )

    return RiskRequest(
        decision=decision,
        portfolio=make_portfolio_context(context),
        risk_context=context,
    )


def test_symbol_position_count_at_limit_can_be_maintained() -> None:
    context = make_context(
        {
            "XAUUSD": make_snapshot(
                "XAUUSD",
                exposure=0.10,
                position_count=1,
            )
        }
    )

    limits = RiskLimits(
        max_positions_per_symbol=1,
    )

    result = RiskEngine(
        limits=limits,
    ).evaluate(
        make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.10,
            },
        )
    )

    assert result.status == RiskDecisionStatus.APPROVED
    assert result.target_exposure["XAUUSD"] == pytest.approx(0.10)


def test_flat_symbol_can_open_one_position_at_limit() -> None:
    context = make_context(
        {
            "XAUUSD": make_snapshot(
                "XAUUSD",
                exposure=0.0,
                position_count=0,
            )
        }
    )

    limits = RiskLimits(
        max_positions_per_symbol=1,
    )

    result = RiskEngine(
        limits=limits,
    ).evaluate(
        make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.10,
            },
        )
    )

    assert result.status != RiskDecisionStatus.REJECTED
    assert result.metadata["projected"]["symbols"]["XAUUSD"][
        "projected_position_count"
    ] == 1


def test_symbol_already_above_limit_is_rejected() -> None:
    context = make_context(
        {
            "XAUUSD": make_snapshot(
                "XAUUSD",
                exposure=0.10,
                position_count=2,
            )
        }
    )

    limits = RiskLimits(
        max_positions_per_symbol=1,
    )

    result = RiskEngine(
        limits=limits,
    ).evaluate(
        make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.10,
            },
        )
    )

    assert result.status == RiskDecisionStatus.REJECTED

    violation_codes = {
        violation.code
        for violation in result.violations
    }

    assert "MAX_POSITIONS_PER_SYMBOL" in violation_codes


def test_closing_symbol_is_allowed_even_when_current_count_reaches_limit() -> None:
    context = make_context(
        {
            "XAUUSD": make_snapshot(
                "XAUUSD",
                exposure=0.10,
                position_count=1,
            )
        }
    )

    limits = RiskLimits(
        max_positions_per_symbol=1,
    )

    result = RiskEngine(
        limits=limits,
    ).evaluate(
        make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.0,
            },
        )
    )

    assert result.status != RiskDecisionStatus.REJECTED
    assert result.target_exposure["XAUUSD"] == pytest.approx(0.0)

    assert result.metadata["projected"]["symbols"]["XAUUSD"][
        "projected_position_count"
    ] == 0


def test_symbol_position_count_boundary_is_inclusive() -> None:
    context = make_context(
        {
            "XAUUSD": make_snapshot(
                "XAUUSD",
                exposure=0.10,
                position_count=3,
            )
        }
    )

    limits = RiskLimits(
        max_positions_per_symbol=3,
    )

    result = RiskEngine(
        limits=limits,
    ).evaluate(
        make_request(
            context=context,
            target_exposure={
                "XAUUSD": 0.10,
            },
        )
    )

    assert result.status == RiskDecisionStatus.APPROVED

    assert result.metadata["projected"]["symbols"]["XAUUSD"][
        "projected_position_count"
    ] == 3