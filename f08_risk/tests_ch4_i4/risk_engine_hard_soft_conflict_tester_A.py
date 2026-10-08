# f08_risk/tests_ch4_i4/risk_engine_hard_soft_conflict_tester_A.py (29)
#
# Run: pytest -v -s f08_risk/tests_ch4_i4/risk_engine_hard_soft_conflict_tester_A.py

# Purpose:
#   Validate Hard-vs-Soft constraint precedence and deterministic
#   conflict resolution behavior of the projected RiskEngine path.
#
# Contract under test:
#
#   1. Hard constraints must override soft modifications.
#   2. A soft constraint alone must modify, not reject.
#   3. Multiple soft constraints must be applied deterministically.
#   4. A final hard constraint must reject after earlier soft modifications,
#      while preserving the earlier warning violations in the rejection result.
#
# Important:
#   This tester validates the CURRENT RiskEngine contract.
#   It does NOT introduce a new production policy.

from __future__ import annotations

from dataclasses import replace
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
    experiment_id="exp-hard-soft-conflict",
)


def make_snapshot(
    symbol: str,
    *,
    exposure: float = 0.0,
) -> SymbolRiskSnapshot:

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
        notional=1_000.0,
        used_margin=250.0,
        current_lots=0.05,
        current_side=1 if exposure > 0.0 else -1,
    )


def make_risk_context(
    *,
    open_position_count: int = 1,
    risk_blocked: bool = False,
    drawdown: float = 0.0,
    daily_drawdown: float = 0.0,
    leverage: float = 100.0,
) -> RiskContext:
    equity = 10_000.0

    used_margin = 1_000.0

    if drawdown > 0.0:
        peak_equity = equity / (1.0 - drawdown)
        peak_equity *= 1.0 + 1e-12
    else:
        peak_equity = equity

    if daily_drawdown > 0.0:
        day_start_equity = equity / (1.0 - daily_drawdown)
    else:
        day_start_equity = equity

    symbols = {
        "EURUSD": make_snapshot(
            "EURUSD",
            exposure=0.0,
        ),
        "XAUUSD": make_snapshot(
            "XAUUSD",
            exposure=0.0,
        ),
    }

    correlation = {
        "EURUSD": {
            "EURUSD": 1.0,
            "XAUUSD": 0.90,
        },
        "XAUUSD": {
            "EURUSD": 0.90,
            "XAUUSD": 1.0,
        },
    }

    return RiskContext(
        timestamp=TS,
        account=AccountRiskSnapshot(
            balance=equity,
            equity=equity,
            used_margin=used_margin,
            free_margin=equity - used_margin,
            margin_level=1_000.0,
            leverage=leverage,
            peak_equity=peak_equity,
            day_start_equity=day_start_equity,
            open_position_count=open_position_count,
        ),
        symbols=symbols,
        correlation=correlation,
        risk_blocked=risk_blocked,
    )


def make_portfolio_context(
    risk_context: RiskContext,
    *,
    risk_blocked: bool | None = None,
    drawdown: float | None = None,
    daily_drawdown: float | None = None,
) -> PortfolioContext:

    return PortfolioContext(
        timestamp=TS,
        equity=risk_context.account.equity,
        balance=risk_context.account.balance,
        used_margin=risk_context.account.used_margin,
        free_margin=risk_context.account.free_margin,
        margin_level=risk_context.account.margin_level,
        drawdown=(
            risk_context.account.equity * 0.0
            if drawdown is None
            else drawdown
        ),
        daily_drawdown=(
            risk_context.account.equity * 0.0
            if daily_drawdown is None
            else daily_drawdown
        ),
        exposure={
            symbol: snapshot.exposure
            for symbol, snapshot in risk_context.symbols.items()
        },
        concentration={},
        correlation=risk_context.correlation,
        risk_blocked=(
            False
            if risk_blocked is None
            else risk_blocked
        ),
        mode=DecisionMode.BACKTEST,
    )


def make_request(
    *,
    risk_context: RiskContext,
    target_exposure: dict[str, float],
    portfolio: PortfolioContext | None = None,
    portfolio_risk: float = 0.10,
) -> RiskRequest:

    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="hard-soft-decision",
        capital_allocation={},
        margin_allocation={},
        target_exposure=target_exposure,
        target_signals={},
        portfolio_risk=portfolio_risk,
        reason_codes=(),
        model=MODEL,
    )

    if portfolio is None:
        portfolio = make_portfolio_context(risk_context)

    return RiskRequest(
        decision=decision,
        portfolio=portfolio,
        risk_context=risk_context,
    )


def test_hard_constraint_overrides_soft_constraint() -> None:
    context = make_risk_context(
        risk_blocked=True,
    )

    request = make_request(
        risk_context=context,
        target_exposure={
            "XAUUSD": 0.80,
        },
        portfolio=make_portfolio_context(
            context,
            risk_blocked=True,
        ),
    )

    limits = RiskLimits(
        max_symbol_exposure=0.50,
    )

    result = RiskEngine(limits=limits).evaluate(request)

    assert result.status == RiskDecisionStatus.REJECTED

    assert tuple(
        violation.code
        for violation in result.violations
    ) == (
        "RISK_BLOCKED",
    )

    assert result.target_exposure == {}


def test_soft_constraint_modifies_without_rejecting() -> None:
    context = make_risk_context()

    request = make_request(
        risk_context=context,
        target_exposure={
            "XAUUSD": 0.80,
        },
    )

    limits = RiskLimits(
        max_symbol_exposure=0.50,
        max_total_exposure=1.00,
        max_correlated_exposure=1.00,
    )

    result = RiskEngine(limits=limits).evaluate(request)

    assert result.status == RiskDecisionStatus.MODIFIED

    assert result.target_exposure["XAUUSD"] == pytest.approx(
        0.50,
    )

    assert tuple(
        violation.code
        for violation in result.violations
    ) == (
        "MAX_SYMBOL_EXPOSURE",
    )

    assert all(
        violation.severity == "warning"
        for violation in result.violations
    )


def test_multiple_soft_constraints_are_applied_deterministically() -> None:
    context = make_risk_context()

    request = make_request(
        risk_context=context,
        target_exposure={
            "EURUSD": 0.80,
            "XAUUSD": 0.80,
        },
    )

    limits = RiskLimits(
        max_symbol_exposure=0.50,
        max_total_exposure=1.00,
        high_correlation_threshold=0.85,
        max_correlated_exposure=0.60,
    )

    result = RiskEngine(limits=limits).evaluate(request)

    assert result.status == RiskDecisionStatus.MODIFIED

    assert result.target_exposure["EURUSD"] == pytest.approx(
        0.30,
    )

    assert result.target_exposure["XAUUSD"] == pytest.approx(
        0.30,
    )

    assert tuple(
        violation.code
        for violation in result.violations
    ) == (
        "MAX_SYMBOL_EXPOSURE",
        "MAX_SYMBOL_EXPOSURE",
        "MAX_CORRELATED_EXPOSURE",
    )

    assert all(
        violation.severity == "warning"
        for violation in result.violations
    )


def test_hard_drawdown_reject_prevents_soft_exposure_modification() -> None:
    context = make_risk_context(
        drawdown=0.20,
    )
    portfolio = make_portfolio_context(
        context,
        drawdown=0.20,
    )

    request = make_request(
        risk_context=context,
        target_exposure={
            "XAUUSD": 0.80,
        },
        portfolio=portfolio,
    )

    limits = RiskLimits(
        max_drawdown=0.20,
        max_symbol_exposure=0.50,
    )

    result = RiskEngine(limits=limits).evaluate(request)

    assert result.status == RiskDecisionStatus.REJECTED

    assert tuple(
        violation.code
        for violation in result.violations
    ) == (
        "MAX_DRAWDOWN",
    )

    assert result.target_exposure == {}


def test_hard_portfolio_risk_prevents_soft_modification() -> None:
    context = make_risk_context()

    request = make_request(
        risk_context=context,
        target_exposure={
            "XAUUSD": 0.80,
        },
        portfolio_risk=0.60,
    )

    limits = RiskLimits(
        max_portfolio_risk=0.50,
        max_symbol_exposure=0.50,
    )

    result = RiskEngine(limits=limits).evaluate(request)

    assert result.status == RiskDecisionStatus.REJECTED

    assert tuple(
        violation.code
        for violation in result.violations
    ) == (
        "MAX_PORTFOLIO_RISK",
    )

    assert result.target_exposure == {}

