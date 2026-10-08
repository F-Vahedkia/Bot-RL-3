# f08_risk/tests_ch4_i6/stop_loss_exposure_alignment_integration_tester_A.py (41)
#
# Run: pytest -v -s f08_risk/tests_ch4_i6/stop_loss_exposure_alignment_integration_tester_A.py

# Purpose:
#   Validate Stop-Loss direction consistency with signed target exposure.
#
# Contract:
#
#   Long  exposure > 0  -> stop < entry
#   Short exposure < 0  -> stop > entry
#   Flat  exposure == 0 -> no active stop-loss request
#
# This tester validates the RiskEngine invariant and does not test broker
# execution or order placement.

from __future__ import annotations
from datetime import datetime, timezone
import pytest

from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PortfolioContext,
    PortfolioDecision,
)
from f08_risk.contracts import RiskRequest
from f08_risk.risk_engine import RiskEngine
from f08_risk.stop_loss_position_sizing import (
    StopLossPositionSizingRequest,
)
from f08_risk.risk_context import (
    AccountRiskSnapshot,
    RiskContext,
    SymbolRiskSnapshot,
)

TS = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)

MODEL = ModelIdentity(
    model_name="test-model",
    model_version="1",
    policy_version="1",
    config_version="1",
    experiment_id="exp-sl-alignment",
)


def make_decision(
    *,
    target_exposure: float,
) -> PortfolioDecision:
    return PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="sl-alignment-decision",
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


def make_portfolio(
    *,
    target_exposure: float,
) -> PortfolioContext:
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
            "XAUUSD": target_exposure,
        },
        concentration={},
        correlation={
            "XAUUSD": {
                "XAUUSD": 1.0,
            },
        },
        risk_blocked=False,
        mode=DecisionMode.BACKTEST,
    )


def make_risk_context(
    *,
    target_exposure: float,
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
            open_position_count=1,
        ),
        symbols={
            "XAUUSD": SymbolRiskSnapshot(
                symbol="XAUUSD",
                exposure=target_exposure,
                notional=abs(target_exposure) * 10_000.0,
                used_margin=250.0,
                current_lots=(
                    0.0
                    if target_exposure == 0.0
                    else 0.05
                ),
                current_side=(
                    0
                    if target_exposure == 0.0
                    else 1
                    if target_exposure > 0.0
                    else -1
                ),
            ),
        },
        correlation={
            "XAUUSD": {
                "XAUUSD": 1.0,
            },
        },
        risk_blocked=False,
    )


def make_sl_request(
    *,
    entry_price: float = 2_500.0,
    stop_price: float = 2_490.0,
) -> StopLossPositionSizingRequest:
    return StopLossPositionSizingRequest(
        symbol="XAUUSD",
        equity=10_000.0,
        risk_per_trade=0.01,
        entry_price=entry_price,
        stop_price=stop_price,
        contract_size=100.0,
        currency_conversion_rate=1.0,
    )


def make_request(
    *,
    target_exposure: float,
    entry_price: float = 2_500.0,
    stop_price: float = 2_490.0,
) -> RiskRequest:
    return RiskRequest(
        decision=make_decision(
            target_exposure=target_exposure,
        ),
        portfolio=make_portfolio(
            target_exposure=target_exposure,
        ),
        risk_context=make_risk_context(
            target_exposure=target_exposure,
        ),
        stop_loss_requests={
            "XAUUSD": make_sl_request(
                entry_price=entry_price,
                stop_price=stop_price,
            ),
        },
    )


def test_long_exposure_accepts_stop_below_entry() -> None:
    request = make_request(
        target_exposure=0.20,
        entry_price=2_500.0,
        stop_price=2_490.0,
    )

    RiskEngine()._validate_stop_loss_alignment(request)


def test_long_exposure_rejects_stop_above_entry() -> None:
    request = make_request(
        target_exposure=0.20,
        entry_price=2_500.0,
        stop_price=2_510.0,
    )

    with pytest.raises(
        ValueError,
        match="long exposure",
    ):
        RiskEngine()._validate_stop_loss_alignment(request)


def test_long_exposure_equal_stop_is_rejected_by_sl_contract() -> None:
    with pytest.raises(
        ValueError,
        match="entry_price and stop_price must not be equal",
    ):
        make_sl_request(
            entry_price=2_500.0,
            stop_price=2_500.0,
        )


def test_short_exposure_accepts_stop_above_entry() -> None:
    request = make_request(
        target_exposure=-0.20,
        entry_price=2_500.0,
        stop_price=2_510.0,
    )

    RiskEngine()._validate_stop_loss_alignment(request)


def test_short_exposure_rejects_stop_below_entry() -> None:
    request = make_request(
        target_exposure=-0.20,
        entry_price=2_500.0,
        stop_price=2_490.0,
    )

    with pytest.raises(
        ValueError,
        match="short exposure",
    ):
        RiskEngine()._validate_stop_loss_alignment(request)


def test_short_exposure_equal_stop_is_rejected_by_sl_contract() -> None:
    with pytest.raises(
        ValueError,
        match="entry_price and stop_price must not be equal",
    ):
        make_sl_request(
            entry_price=2_500.0,
            stop_price=2_500.0,
        )


def test_flat_exposure_rejects_active_stop_loss() -> None:
    request = make_request(
        target_exposure=0.0,
        entry_price=2_500.0,
        stop_price=2_490.0,
    )

    with pytest.raises(
        ValueError,
        match="flat symbol",
    ):
        RiskEngine()._validate_stop_loss_alignment(request)


def test_positive_exposure_is_based_on_sign_not_target_size() -> None:
    request = make_request(
        target_exposure=0.0001,
        entry_price=2_500.0,
        stop_price=2_499.0,
    )

    RiskEngine()._validate_stop_loss_alignment(request)


def test_negative_exposure_is_based_on_sign_not_target_size() -> None:
    request = make_request(
        target_exposure=-0.0001,
        entry_price=2_500.0,
        stop_price=2_501.0,
    )

    RiskEngine()._validate_stop_loss_alignment(request)


def test_projected_engine_enforces_alignment() -> None:
    request = make_request(
        target_exposure=0.20,
        entry_price=2_500.0,
        stop_price=2_510.0,
    )

    with pytest.raises(
        ValueError,
        match="long exposure",
    ):
        RiskEngine().evaluate(request)

