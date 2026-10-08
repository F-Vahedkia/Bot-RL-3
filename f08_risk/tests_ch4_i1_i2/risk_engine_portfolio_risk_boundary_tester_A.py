# f08_risk/tests_ch4_i1_i2/risk_engine_portfolio_risk_boundary_tester_A.py (20)
#
# Run: pytest -v -s f08_risk/tests_ch4_i1_i2/risk_engine_portfolio_risk_boundary_tester_A.py

# Purpose:
#   Explicitly validate the Portfolio-Risk hard boundary of RiskEngine.

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


TS = datetime(
    2026,
    1,
    5,
    12,
    0,
    tzinfo=timezone.utc,
)

MODEL = ModelIdentity(
    model_name="test-model",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="exp-test",
)


def risk_context() -> RiskContext:
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
            open_position_count=1,
        ),
        symbols={
            "XAUUSD": SymbolRiskSnapshot(
                symbol="XAUUSD",
                exposure=0.10,
                notional=1_000.0,
                used_margin=250.0,
                current_lots=0.05,
                current_side=1,
            ),
        },
        correlation={
            "XAUUSD": {
                "XAUUSD": 1.0,
            },
        },
    )


def portfolio_context() -> PortfolioContext:
    context = risk_context()

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


def portfolio_request(
    portfolio_risk: float,
) -> RiskRequest:
    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="decision-001",
        capital_allocation={},
        margin_allocation={},
        target_exposure={
            "XAUUSD": 0.10,
        },
        target_signals={},
        portfolio_risk=portfolio_risk,
        reason_codes=(),
        model=MODEL,
    )

    return RiskRequest(
        decision=decision,
        portfolio=portfolio_context(),
        risk_context=risk_context(),
    )

def test_projected_portfolio_risk_below_limit_is_approved() -> None:
    limits = RiskLimits(
        max_portfolio_risk=0.50,
    )
    engine = RiskEngine(limits=limits)

    result = engine.evaluate(
        portfolio_request(0.49),
    )

    assert result.status == RiskDecisionStatus.APPROVED
    assert result.target_exposure["XAUUSD"] == pytest.approx(0.10)


def test_projected_portfolio_risk_equal_to_limit_is_approved() -> None:
    limits = RiskLimits(
        max_portfolio_risk=0.50,
    )
    engine = RiskEngine(limits=limits)

    result = engine.evaluate(
        portfolio_request(0.50),
    )

    assert result.status == RiskDecisionStatus.APPROVED
    assert result.target_exposure["XAUUSD"] == pytest.approx(0.10)


def test_projected_portfolio_risk_above_limit_is_rejected() -> None:
    limits = RiskLimits(
        max_portfolio_risk=0.50,
    )
    engine = RiskEngine(limits=limits)

    result = engine.evaluate(
        portfolio_request(0.5000001),
    )

    assert result.status == RiskDecisionStatus.REJECTED


def test_projected_portfolio_risk_rejection_carries_expected_violation() -> None:
    limits = RiskLimits(
        max_portfolio_risk=0.50,
    )
    engine = RiskEngine(limits=limits)

    result = engine.evaluate(
        portfolio_request(0.75),
    )

    assert result.status == RiskDecisionStatus.REJECTED
    assert any(
        violation.code == "MAX_PORTFOLIO_RISK"
        for violation in result.violations
    )


def test_custom_portfolio_risk_limit_is_honored() -> None:
    limits = RiskLimits(
        max_portfolio_risk=0.25,
    )
    engine = RiskEngine(limits=limits)

    approved = engine.evaluate(
        portfolio_request(0.25),
    )
    rejected = engine.evaluate(
        portfolio_request(0.2500001),
    )

    assert approved.status == RiskDecisionStatus.APPROVED
    assert rejected.status == RiskDecisionStatus.REJECTED

    assert any(
        violation.code == "MAX_PORTFOLIO_RISK"
        for violation in rejected.violations
    )


def test_portfolio_risk_rejection_returns_no_trade_allocation() -> None:
    limits = RiskLimits(
        max_portfolio_risk=0.50,
    )
    engine = RiskEngine(limits=limits)

    result = engine.evaluate(
        portfolio_request(0.80),
    )

    assert result.status == RiskDecisionStatus.REJECTED
    assert result.target_exposure == {}
    assert result.capital_allocation == {}
    assert result.margin_allocation == {}
