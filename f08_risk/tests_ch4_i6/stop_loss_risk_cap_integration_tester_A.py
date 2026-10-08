# f08_risk/tests_ch4_i6/stop_loss_risk_cap_integration_tester_A.py (42)
#
# Run: pytest -v -s f08_risk/tests_ch4_i6/stop_loss_risk_cap_integration_tester_A.py

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
from f08_risk.risk_engine import RiskEngine
from f08_risk.risk_context import (
    AccountRiskSnapshot,
    RiskContext,
    SymbolRiskSnapshot,
)
from f08_risk.stop_loss_position_sizing import (
    StopLossPositionSizingRequest,
)

TS = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)

MODEL = ModelIdentity(
    model_name="test-model",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="exp-sl-cap",
)


def make_context() -> RiskContext:
    equity = 10_000.0

    return RiskContext(
        timestamp=TS,
        account=AccountRiskSnapshot(
            balance=equity,
            equity=equity,
            used_margin=1_000.0,
            free_margin=9_000.0,
            margin_level=1_000.0,
            leverage=100.0,
            peak_equity=equity,
            day_start_equity=equity,
            open_position_count=0,
        ),
        symbols={
            "XAUUSD": SymbolRiskSnapshot(
                symbol="XAUUSD",
                exposure=0.0,
                notional=0.0,
                used_margin=0.0,
                current_lots=0.0,
                current_side=0,
            ),
        },
        correlation={
            "XAUUSD": {
                "XAUUSD": 1.0,
            },
        },
    )


def make_request(
    *,
    target_exposure: float,
    risk_per_trade: float = 0.01,
    entry_price: float = 2500.0,
    stop_price: float = 2490.0,
) -> RiskRequest:

    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="sl-cap-decision",
        capital_allocation={},
        margin_allocation={},
        target_exposure={
            "XAUUSD": target_exposure,
        },
        target_signals={},
        portfolio_risk=0.01,
        reason_codes=(),
        model=MODEL,
    )

    context = make_context()

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
            "XAUUSD": 0.0,
        },
        concentration={},
        correlation=context.correlation,
        risk_blocked=False,
        mode=DecisionMode.BACKTEST,
    )

    stop_loss_request = StopLossPositionSizingRequest(
        symbol="XAUUSD",
        equity=context.account.equity,
        risk_per_trade=risk_per_trade,
        entry_price=entry_price,
        stop_price=stop_price,
        contract_size=100.0,
        currency_conversion_rate=1.0,
    )

    return RiskRequest(
        decision=decision,
        portfolio=portfolio,
        risk_context=context,
        stop_loss_requests={
            "XAUUSD": stop_loss_request,
        },
    )


def test_exposure_within_stop_loss_budget_is_preserved() -> None:
    request = make_request(
        target_exposure=0.002,
    )

    result = RiskEngine().evaluate(request)

    assert result.status == RiskDecisionStatus.APPROVED
    assert result.target_exposure["XAUUSD"] == pytest.approx(0.002)
    assert "XAUUSD" in result.stop_loss_results
    assert not any(
        v.code == "STOP_LOSS_RISK_CAP"
        for v in result.violations
    )


def test_exposure_above_stop_loss_budget_is_reduced() -> None:
    request = make_request(
        target_exposure=0.10,
        risk_per_trade=0.0001,
    )
    result = RiskEngine().evaluate(request)
    assert result.status == RiskDecisionStatus.MODIFIED
    assert result.target_exposure["XAUUSD"] < 0.10
    assert any(
        v.code == "STOP_LOSS_RISK_CAP"
        for v in result.violations
    )


def test_stop_loss_cap_preserves_long_direction() -> None:
    request = make_request(
        target_exposure=0.10,
    )

    result = RiskEngine().evaluate(request)

    assert result.target_exposure["XAUUSD"] > 0.0


def test_stop_loss_cap_preserves_short_direction() -> None:
    request = make_request(
        target_exposure=-0.10,
        risk_per_trade=0.0001,
        entry_price=2500.0,
        stop_price=2510.0,
    )
    result = RiskEngine().evaluate(request)
    assert result.status == RiskDecisionStatus.MODIFIED
    assert result.target_exposure["XAUUSD"] < 0.0


def test_stop_loss_result_remains_available_after_exposure_cap() -> None:
    request = make_request(
        target_exposure=0.10,
    )

    result = RiskEngine().evaluate(request)

    sl_result = result.stop_loss_results["XAUUSD"]

    assert sl_result.symbol == "XAUUSD"
    assert sl_result.risk_budget == pytest.approx(100.0)
    assert sl_result.stop_distance == pytest.approx(10.0)


def test_zero_risk_per_trade_removes_exposure() -> None:
    request = make_request(
        target_exposure=0.10,
        risk_per_trade=0.0,
    )

    result = RiskEngine().evaluate(request)

    assert result.status == RiskDecisionStatus.MODIFIED
    assert result.target_exposure["XAUUSD"] == pytest.approx(0.0)


def test_stop_loss_cap_does_not_reverse_direction() -> None:
    request = make_request(
        target_exposure=-0.10,
        entry_price=2500.0,
        stop_price=2510.0,
    )

    result = RiskEngine().evaluate(request)

    assert result.target_exposure["XAUUSD"] <= 0.0


def test_stop_loss_cap_does_not_change_target_signal() -> None:
    decision = PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="sl-cap-signal",
        capital_allocation={},
        margin_allocation={},
        target_exposure={
            "XAUUSD": 0.10,
        },
        target_signals={
            "XAUUSD": 1,
        },
        portfolio_risk=0.01,
        reason_codes=(),
        model=MODEL,
    )

    context = make_context()

    request = RiskRequest(
        decision=decision,
        portfolio=PortfolioContext(
            timestamp=TS,
            equity=10_000.0,
            balance=10_000.0,
            used_margin=1_000.0,
            free_margin=9_000.0,
            margin_level=1_000.0,
            drawdown=0.0,
            daily_drawdown=0.0,
            exposure={"XAUUSD": 0.0},
            concentration={},
            correlation=context.correlation,
            risk_blocked=False,
            mode=DecisionMode.BACKTEST,
        ),
        risk_context=context,
        stop_loss_requests={
            "XAUUSD": StopLossPositionSizingRequest(
                symbol="XAUUSD",
                equity=10_000.0,
                risk_per_trade=0.01,
                entry_price=2500.0,
                stop_price=2490.0,
                contract_size=100.0,
                currency_conversion_rate=1.0,
            ),
        },
    )

    result = RiskEngine().evaluate(request)

    assert result.target_signals["XAUUSD"] == 1
