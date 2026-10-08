# f08_risk/tests_ch4_i6/stop_loss_risk_request_decision_integration_tester_A.py (40)
#
# Run: pytest -v -s f08_risk/tests_ch4_i6/stop_loss_risk_request_decision_integration_tester_A.py

# Purpose:
#   Validate the official RiskRequest <-> RiskDecision Stop-Loss contract.
#
# Contract under test:
#
#   PortfolioDecision
#        |
#        v
#   RiskRequest
#        |
#        +--> stop_loss_requests
#        |
#        v
#   RiskDecision
#        |
#        +--> stop_loss_results
#
# Important:
#   This tester intentionally does NOT exercise RiskEngine yet.
#   It locks the contract before RiskEngine integration.

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
    RiskDecision,
    RiskDecisionStatus,
    RiskRequest,
)
from f08_risk.stop_loss_position_sizing import (
    StopLossPositionSizingCalculator,
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
    experiment_id="exp-sl-contract",
)


def make_decision() -> PortfolioDecision:
    return PortfolioDecision(
        approved=True,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="decision-sl-001",
        capital_allocation={},
        margin_allocation={},
        target_exposure={
            "XAUUSD": 0.20,
        },
        target_signals={
            "XAUUSD": 1,
        },
        portfolio_risk=0.01,
        reason_codes=(),
        model=MODEL,
    )


def make_portfolio() -> PortfolioContext:
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


def make_risk_context() -> RiskContext:
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
        risk_blocked=False,
    )


def make_sl_request(
    symbol: str = "XAUUSD",
) -> StopLossPositionSizingRequest:
    return StopLossPositionSizingRequest(
        symbol=symbol,
        equity=10_000.0,
        risk_per_trade=0.01,
        entry_price=2_500.0,
        stop_price=2_490.0,
        contract_size=100.0,
        currency_conversion_rate=1.0,
    )


def make_risk_request(
    *,
    stop_loss_requests=None,
) -> RiskRequest:
    kwargs = {}

    if stop_loss_requests is not None:
        kwargs["stop_loss_requests"] = stop_loss_requests

    return RiskRequest(
        decision=make_decision(),
        portfolio=make_portfolio(),
        risk_context=make_risk_context(),
        **kwargs,
    )


def test_stop_loss_requests_default_to_empty_mapping() -> None:
    request = make_risk_request()

    assert request.stop_loss_requests == {}


def test_single_symbol_stop_loss_request_is_preserved() -> None:
    sl_request = make_sl_request()

    request = make_risk_request(
        stop_loss_requests={
            "XAUUSD": sl_request,
        }
    )

    assert request.stop_loss_requests["XAUUSD"] is sl_request


def test_multiple_symbol_stop_loss_requests_are_supported() -> None:
    xau = make_sl_request("XAUUSD")
    eur = make_sl_request("EURUSD")

    request = make_risk_request(
        stop_loss_requests={
            "XAUUSD": xau,
            "EURUSD": eur,
        }
    )

    assert set(request.stop_loss_requests) == {
        "XAUUSD",
        "EURUSD",
    }


def test_non_request_value_is_rejected() -> None:
    with pytest.raises(
        TypeError,
        match="StopLossPositionSizingRequest",
    ):
        make_risk_request(
            stop_loss_requests={
                "XAUUSD": object(),
            }
        )


def test_empty_symbol_key_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="empty symbol",
    ):
        make_risk_request(
            stop_loss_requests={
                "": make_sl_request("XAUUSD"),
            }
        )


def test_symbol_key_must_match_request_symbol() -> None:
    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        make_risk_request(
            stop_loss_requests={
                "EURUSD": make_sl_request("XAUUSD"),
            }
        )


def test_backward_compatibility_without_stop_loss_is_preserved() -> None:
    request = make_risk_request()

    assert request.stop_loss_requests == {}


def test_stop_loss_result_can_be_computed_for_one_symbol() -> None:
    request = make_sl_request()

    result = StopLossPositionSizingCalculator().calculate(request)

    assert result.symbol == "XAUUSD"
    assert result.risk_budget == pytest.approx(100.0)
    assert result.stop_distance == pytest.approx(10.0)
    assert result.loss_per_lot_at_stop == pytest.approx(1_000.0)
    assert result.volume_lots == pytest.approx(0.1)


def test_risk_decision_defaults_to_empty_stop_loss_results() -> None:
    decision = RiskDecision(
        status=RiskDecisionStatus.APPROVED,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="risk-decision-001",
        source_decision_id="decision-sl-001",
        capital_allocation={},
        margin_allocation={},
        target_exposure={
            "XAUUSD": 0.20,
        },
        target_signals={
            "XAUUSD": 1,
        },
        portfolio_risk=0.01,
        violations=(),
        model=MODEL,
    )

    assert decision.stop_loss_results == {}


def test_risk_decision_preserves_single_stop_loss_result() -> None:
    request = make_sl_request()

    result = StopLossPositionSizingCalculator().calculate(request)

    decision = RiskDecision(
        status=RiskDecisionStatus.APPROVED,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="risk-decision-002",
        source_decision_id="decision-sl-001",
        capital_allocation={},
        margin_allocation={},
        target_exposure={
            "XAUUSD": 0.20,
        },
        target_signals={
            "XAUUSD": 1,
        },
        portfolio_risk=0.01,
        violations=(),
        model=MODEL,
        stop_loss_results={
            "XAUUSD": result,
        },
    )

    assert decision.stop_loss_results["XAUUSD"] is result


def test_risk_decision_supports_multiple_stop_loss_results() -> None:
    calculator = StopLossPositionSizingCalculator()

    xau_result = calculator.calculate(
        make_sl_request("XAUUSD")
    )

    eur_result = calculator.calculate(
        make_sl_request("EURUSD")
    )

    decision = RiskDecision(
        status=RiskDecisionStatus.MODIFIED,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="risk-decision-003",
        source_decision_id="decision-sl-001",
        capital_allocation={},
        margin_allocation={},
        target_exposure={
            "XAUUSD": 0.20,
            "EURUSD": 0.10,
        },
        target_signals={
            "XAUUSD": 1,
            "EURUSD": 1,
        },
        portfolio_risk=0.02,
        violations=(),
        model=MODEL,
        stop_loss_results={
            "XAUUSD": xau_result,
            "EURUSD": eur_result,
        },
    )

    assert set(decision.stop_loss_results) == {
        "XAUUSD",
        "EURUSD",
    }


def test_stop_loss_result_symbol_is_retrievable_by_same_symbol_key() -> None:
    calculator = StopLossPositionSizingCalculator()

    result = calculator.calculate(
        make_sl_request("XAUUSD")
    )

    decision = RiskDecision(
        status=RiskDecisionStatus.APPROVED,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="risk-decision-004",
        source_decision_id="decision-sl-001",
        capital_allocation={},
        margin_allocation={},
        target_exposure={
            "XAUUSD": 0.20,
        },
        target_signals={
            "XAUUSD": 1,
        },
        portfolio_risk=0.01,
        violations=(),
        model=MODEL,
        stop_loss_results={
            "XAUUSD": result,
        },
    )

    stored = decision.stop_loss_results["XAUUSD"]

    assert stored.symbol == "XAUUSD"


def test_stop_loss_contract_does_not_change_target_exposure() -> None:
    request = make_risk_request(
        stop_loss_requests={
            "XAUUSD": make_sl_request(),
        }
    )

    assert request.decision.target_exposure["XAUUSD"] == pytest.approx(0.20)


def test_stop_loss_contract_does_not_change_target_signal() -> None:
    request = make_risk_request(
        stop_loss_requests={
            "XAUUSD": make_sl_request(),
        }
    )

    assert request.decision.target_signals["XAUUSD"] == 1


def test_stop_loss_request_remains_immutable() -> None:
    request = make_risk_request(
        stop_loss_requests={
            "XAUUSD": make_sl_request(),
        }
    )

    with pytest.raises(
        AttributeError,
    ):
        request.stop_loss_requests = {}


def test_risk_decision_stop_loss_results_remain_immutable() -> None:
    calculator = StopLossPositionSizingCalculator()

    result = calculator.calculate(
        make_sl_request()
    )

    decision = RiskDecision(
        status=RiskDecisionStatus.APPROVED,
        timestamp=TS,
        mode=DecisionMode.BACKTEST,
        decision_id="risk-decision-005",
        source_decision_id="decision-sl-001",
        capital_allocation={},
        margin_allocation={},
        target_exposure={
            "XAUUSD": 0.20,
        },
        target_signals={
            "XAUUSD": 1,
        },
        portfolio_risk=0.01,
        violations=(),
        model=MODEL,
        stop_loss_results={
            "XAUUSD": result,
        },
    )

    with pytest.raises(
        AttributeError,
    ):
        decision.stop_loss_results = {}

