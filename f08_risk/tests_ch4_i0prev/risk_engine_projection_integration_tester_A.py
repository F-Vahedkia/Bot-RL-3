# f08_risk/tests_ch4_i0prev/risk_engine_projection_integration_tester_A.py (11)
#
# Run: pytest -v -s f08_risk/tests_ch4_i0prev/risk_engine_projection_integration_tester_A.py


from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
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
    model_name="integration-model",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="integration-exp",
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
                exposure=0.20,
                notional=2_000.0,
                used_margin=500.0,
                current_lots=0.10,
                current_side=1,
            ),
            "EURUSD": SymbolRiskSnapshot(
                symbol="EURUSD",
                exposure=0.10,
                notional=1_000.0,
                used_margin=500.0,
                current_lots=0.20,
                current_side=1,
            ),
        },
        correlation={
            "XAUUSD": {
                "XAUUSD": 1.0,
                "EURUSD": 0.90,
            },
            "EURUSD": {
                "XAUUSD": 0.90,
                "EURUSD": 1.0,
            },
        },
    )


def portfolio() -> PortfolioContext:
    return PortfolioContext(
        timestamp=TS,
        equity=10_000.0,
        balance=10_000.0,
        used_margin=1_000.0,
        free_margin=9_000.0,
        margin_level=1_000.0,
        drawdown=0.0,
        daily_drawdown=0.0,
        exposure={
            "XAUUSD": 0.20,
            "EURUSD": 0.10,
        },
        concentration={
            "XAUUSD": 0.20,
            "EURUSD": 0.10,
        },
        correlation={
            "XAUUSD": {
                "EURUSD": 0.90,
            },
            "EURUSD": {
                "XAUUSD": 0.90,
            },
        },
        risk_blocked=False,
        mode=DecisionMode.BACKTEST,
    )


def decision(
    exposures: dict[str, float],
) -> PortfolioDecision:
    return PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="integration-001",
        capital_allocation={
            symbol: abs(value)
            for symbol, value in exposures.items()
        },
        margin_allocation={
            symbol: abs(value) * 5_000.0
            for symbol, value in exposures.items()
        },
        target_exposure=exposures,
        target_signals={
            symbol: 1
            for symbol in exposures
        },
        portfolio_risk=0.10,
        reason_codes=(),
        model=MODEL,
    )


def request(
    exposures: dict[str, float],
) -> RiskRequest:
    return RiskRequest(
        decision=decision(exposures),
        portfolio=portfolio(),
        risk_context=risk_context(),
    )


def test_engine_uses_projected_evaluation_path():

    result = RiskEngine().evaluate(
        request(
            {
                "XAUUSD": 0.20,
                "EURUSD": 0.10,
            }
        )
    )

    assert result.status == RiskDecisionStatus.APPROVED
    assert result.metadata["evaluation_path"] == "projected"
    assert "projected" in result.metadata


def test_projection_drives_total_exposure_modification():

    result = RiskEngine(
        limits=RiskLimits(
            max_total_exposure=0.40,
        )
    ).evaluate(
        request(
            {
                "XAUUSD": 0.30,
                "EURUSD": 0.30,
            }
        )
    )

    assert result.status == RiskDecisionStatus.MODIFIED

    total = sum(
        abs(value)
        for value in result.target_exposure.values()
    )

    assert total == pytest.approx(0.40)


def test_projection_drives_correlated_exposure_modification():

    result = RiskEngine(
        limits=RiskLimits(
            max_total_exposure=1.0,
            max_correlated_exposure=0.30,
        )
    ).evaluate(
        request(
            {
                "XAUUSD": 0.30,
                "EURUSD": 0.30,
            }
        )
    )

    assert result.status == RiskDecisionStatus.MODIFIED

    combined = (
        abs(result.target_exposure["XAUUSD"])
        +
        abs(result.target_exposure["EURUSD"])
    )

    assert combined == pytest.approx(0.30)


def test_projection_drives_margin_modification():

    result = RiskEngine(
        limits=RiskLimits(
            max_total_exposure=1.0,
            max_margin_utilization=0.10,
        )
    ).evaluate(
        request(
            {
                "XAUUSD": 0.40,
                "EURUSD": 0.40,
            }
        )
    )

    assert result.status == RiskDecisionStatus.MODIFIED

    projected = result.metadata["projected"]

    assert (
        projected["projected_margin_utilization"]
        <= 0.10 + 1e-12
    )


def test_existing_risk_block_remains_hard_reject():

    p = portfolio()

    context = replace(
        risk_context(),
        risk_blocked=True,
    )

    result = RiskEngine().evaluate(
        RiskRequest(
            decision=decision(
                {
                    "XAUUSD": 0.20,
                }
            ),
            portfolio=portfolio(),
            risk_context=context,
        )
    )

    assert result.status == RiskDecisionStatus.REJECTED


def test_projected_path_is_deterministic():

    engine_a = RiskEngine()
    engine_b = RiskEngine()

    req = request(
        {
            "XAUUSD": 0.30,
            "EURUSD": 0.30,
        }
    )

    result_a = engine_a.evaluate(req)
    result_b = engine_b.evaluate(req)

    assert result_a == result_b

