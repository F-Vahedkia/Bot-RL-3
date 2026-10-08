# f08_risk/tests_others/risk_engine_risk_context_authority_boundary_tester_A.py
#
# Run: pytest -v -s f08_risk/tests_others/risk_engine_risk_context_authority_boundary_tester_A.py

# Chapter 4 - Risk Layer
# Item 2 - RiskContext authority for Risk Enforcement
#
# Purpose
# -------
# Lock the source-of-truth boundary for the projected RiskEngine path.
#
# Contract under test
# -------------------
# When RiskRequest.risk_context is present, RiskContext is authoritative for
# current risk state consumed by RiskEngine. PortfolioContext remains the
# decision-layer context owned by f07_agents.
#
# The tests deliberately create conflicting PortfolioContext and RiskContext
# values. A test must fail if RiskEngine silently falls back to the duplicate
# PortfolioContext state.

from __future__ import annotations

from datetime import datetime, timezone

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
from f08_risk.risk_context import (
    AccountRiskSnapshot,
    RiskContext,
    SymbolRiskSnapshot,
)
from f08_risk.risk_engine import RiskEngine


TS = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)


MODEL = ModelIdentity(
    model_name="risk-context-authority-test",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="exp-risk-context-authority",
)


# =============================================================================
# Shared test data
# =============================================================================


def make_account(
    *,
    equity: float = 10_000.0,
    used_margin: float = 1_000.0,
    free_margin: float | None = None,
    margin_level: float = 1_000.0,
    leverage: float = 100.0,
    peak_equity: float = 10_000.0,
    day_start_equity: float = 10_000.0,
) -> AccountRiskSnapshot:
    return AccountRiskSnapshot(
        balance=equity,
        equity=equity,
        used_margin=used_margin,
        free_margin=(
            equity - used_margin
            if free_margin is None
            else free_margin
        ),
        margin_level=margin_level,
        leverage=leverage,
        peak_equity=peak_equity,
        day_start_equity=day_start_equity,
        open_position_count=1,
    )


def make_context(
    *,
    account: AccountRiskSnapshot | None = None,
    risk_blocked: bool = False,
) -> RiskContext:
    actual_account = (
        account
        if account is not None
        else make_account()
    )

    symbols = {
        "EURUSD": SymbolRiskSnapshot(
            symbol="EURUSD",
            exposure=0.0,
            notional=0.0,
            used_margin=0.0,
            current_lots=0.0,
            current_side=0,
            position_count=0,
        )
    }

    correlation = {
        "EURUSD": {
            "EURUSD": 1.0,
        }
    }

    return RiskContext(
        timestamp=TS,
        account=actual_account,
        symbols=symbols,
        correlation=correlation,
        risk_blocked=risk_blocked,
    )


def make_portfolio(
    *,
    context: RiskContext,
    equity: float | None = None,
    used_margin: float | None = None,
    free_margin: float | None = None,
    margin_level: float | None = None,
    drawdown: float = 0.0,
    daily_drawdown: float = 0.0,
    risk_blocked: bool = False,
) -> PortfolioContext:
    account = context.account

    return PortfolioContext(
        timestamp=TS,
        equity=(
            account.equity
            if equity is None
            else equity
        ),
        balance=account.balance,
        used_margin=(
            account.used_margin
            if used_margin is None
            else used_margin
        ),
        free_margin=(
            account.free_margin
            if free_margin is None
            else free_margin
        ),
        margin_level=(
            account.margin_level
            if margin_level is None
            else margin_level
        ),
        drawdown=drawdown,
        daily_drawdown=daily_drawdown,
        exposure={
            "EURUSD": 0.0,
        },
        concentration={},
        correlation={
            "EURUSD": {
                "EURUSD": 1.0,
            }
        },
        risk_blocked=risk_blocked,
        mode=DecisionMode.BACKTEST,
    )


def make_request(
    *,
    context: RiskContext,
    portfolio: PortfolioContext,
) -> RiskRequest:
    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="risk-context-authority-decision",
        capital_allocation={},
        margin_allocation={},
        target_exposure={
            "EURUSD": 0.10,
        },
        target_signals={},
        portfolio_risk=0.10,
        reason_codes=(),
        model=MODEL,
    )

    return RiskRequest(
        decision=decision,
        portfolio=portfolio,
        risk_context=context,
    )


# =============================================================================
# Boundary tests
# =============================================================================


def test_projected_risk_block_authority_is_risk_context() -> None:
    """
    PortfolioContext says "not blocked" while RiskContext says "blocked".
    RiskEngine must reject from RiskContext.
    """
    context = make_context(
        risk_blocked=True,
    )
    portfolio = make_portfolio(
        context=context,
        risk_blocked=False,
    )

    result = RiskEngine().evaluate(
        make_request(
            context=context,
            portfolio=portfolio,
        )
    )

    assert result.status == RiskDecisionStatus.REJECTED
    assert tuple(v.code for v in result.violations) == (
        "RISK_BLOCKED",
    )


def test_projected_drawdown_authority_is_risk_context() -> None:
    """
    PortfolioContext says drawdown is safe while RiskContext.account.drawdown
    is above the configured maximum.
    """
    context = make_context(
        account=make_account(
            equity=7_500.0,
            peak_equity=10_000.0,  # drawdown = 25%
            day_start_equity=7_500.0,  # daily drawdown = 0%
        )
    )
    portfolio = make_portfolio(
        context=context,
        drawdown=0.0,
        daily_drawdown=0.0,
    )

    result = RiskEngine().evaluate(
        make_request(
            context=context,
            portfolio=portfolio,
        )
    )

    assert result.status == RiskDecisionStatus.REJECTED
    assert tuple(v.code for v in result.violations) == (
        "MAX_DRAWDOWN",
    )


def test_projected_daily_drawdown_authority_is_risk_context() -> None:
    """
    PortfolioContext says daily drawdown is safe while
    RiskContext.account.daily_drawdown is above the configured maximum.
    """
    context = make_context(
        account=make_account(
            equity=7_500.0,
            peak_equity=7_500.0,  # drawdown = 0%
            day_start_equity=10_000.0,  # daily drawdown = 25%
        )
    )
    portfolio = make_portfolio(
        context=context,
        drawdown=0.0,
        daily_drawdown=0.0,
    )

    result = RiskEngine().evaluate(
        make_request(
            context=context,
            portfolio=portfolio,
        )
    )

    assert result.status == RiskDecisionStatus.REJECTED
    assert tuple(v.code for v in result.violations) == (
        "MAX_DAILY_DRAWDOWN",
    )


def test_projected_account_margin_authority_is_risk_context() -> None:
    """
    PortfolioContext deliberately contains an invalid current margin level,
    while RiskContext.account is safe. Projected RiskEngine must use the
    RiskContext account snapshot and therefore must not reject
    INVALID_MARGIN_LEVEL.
    """
    context = make_context(
        account=make_account(
            equity=10_000.0,
            used_margin=1_000.0,
            free_margin=9_000.0,
            margin_level=1_000.0,
        )
    )
    portfolio = make_portfolio(
        context=context,
        equity=1.0,
        used_margin=1.0,
        free_margin=0.0,
        margin_level=0.0,
    )

    result = RiskEngine().evaluate(
        make_request(
            context=context,
            portfolio=portfolio,
        )
    )

    assert result.status != RiskDecisionStatus.REJECTED
    assert "INVALID_MARGIN_LEVEL" not in {
        violation.code
        for violation in result.violations
    }
