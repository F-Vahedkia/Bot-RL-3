# f08_risk/tests_ch4_i6/stop_loss_risk_contract_tester_A.py (33)
#
# Run: pytest -v -s f08_risk/tests_ch4_i6/stop_loss_risk_contract_tester_A.py

# =============================================================================
# Contract / edge-case tester for f08_risk.stop_loss_risk
# =============================================================================

from __future__ import annotations

from math import isclose

import pytest

from f08_risk.stop_loss_risk import (
    StopLossRiskCalculator,
    StopLossRiskRequest,
    StopLossRiskResult,
)


def _make_request(
    *,
    equity: float = 10_000.0,
    risk_per_trade: float = 0.01,
    entry_price: float = 1.1000,
    stop_price: float = 1.0900,
) -> StopLossRiskRequest:

    return StopLossRiskRequest(
        symbol="EURUSD",
        equity=equity,
        risk_per_trade=risk_per_trade,
        entry_price=entry_price,
        stop_price=stop_price,
    )


def _assert_close(
    actual: float,
    expected: float,
) -> None:
    assert isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


# =============================================================================
# Dataclass contracts
# =============================================================================

def test_request_is_frozen_slots_dataclass() -> None:
    request = _make_request()

    assert hasattr(request, "__dataclass_fields__")
    assert "__dict__" not in dir(request)

    with pytest.raises(AttributeError):
        request.equity = 20_000.0


def test_result_is_frozen_slots_dataclass() -> None:
    result = StopLossRiskCalculator().calculate(
        _make_request()
    )

    assert isinstance(result, StopLossRiskResult)
    assert hasattr(result, "__dataclass_fields__")
    assert "__dict__" not in dir(result)

    with pytest.raises(AttributeError):
        result.risk_budget = 200.0


# =============================================================================
# Basic calculation
# =============================================================================

def test_risk_budget_is_equity_times_risk_per_trade() -> None:
    result = StopLossRiskCalculator().calculate(
        _make_request(
            equity=10_000.0,
            risk_per_trade=0.01,
        )
    )

    _assert_close(
        result.risk_budget,
        100.0,
    )


def test_stop_distance_is_absolute_price_difference() -> None:
    result = StopLossRiskCalculator().calculate(
        _make_request(
            entry_price=1.1000,
            stop_price=1.0900,
        )
    )

    _assert_close(
        result.stop_distance,
        0.0100,
    )


def test_stop_distance_is_direction_neutral() -> None:
    long_style = StopLossRiskCalculator().calculate(
        _make_request(
            entry_price=1.1000,
            stop_price=1.0900,
        )
    )

    short_style = StopLossRiskCalculator().calculate(
        _make_request(
            entry_price=1.0900,
            stop_price=1.1000,
        )
    )

    _assert_close(
        long_style.stop_distance,
        short_style.stop_distance,
    )


def test_zero_risk_per_trade_produces_zero_budget() -> None:
    result = StopLossRiskCalculator().calculate(
        _make_request(
            risk_per_trade=0.0,
        )
    )

    _assert_close(
        result.risk_budget,
        0.0,
    )


def test_zero_equity_produces_zero_budget() -> None:
    result = StopLossRiskCalculator().calculate(
        _make_request(
            equity=0.0,
        )
    )

    _assert_close(
        result.risk_budget,
        0.0,
    )


# =============================================================================
# Validation
# =============================================================================

@pytest.mark.parametrize(
    "field,value,error_match",
    [
        ("equity", -1.0, "equity"),
        ("risk_per_trade", -0.01, "risk_per_trade"),
        ("risk_per_trade", 1.01, "risk_per_trade"),
        ("entry_price", 0.0, "entry_price"),
        ("stop_price", 0.0, "stop_price"),
    ],
)
def test_invalid_numeric_values_are_rejected(
    field: str,
    value: float,
    error_match: str,
) -> None:

    kwargs = {
        "equity": 10_000.0,
        "risk_per_trade": 0.01,
        "entry_price": 1.1000,
        "stop_price": 1.0900,
    }

    kwargs[field] = value

    with pytest.raises(
        (ValueError, TypeError),
        match=error_match,
    ):
        StopLossRiskRequest(
            symbol="EURUSD",
            **kwargs,
        )


def test_equal_entry_and_stop_prices_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must not be equal",
    ):
        _make_request(
            entry_price=1.1000,
            stop_price=1.1000,
        )


def test_non_finite_equity_is_rejected() -> None:
    with pytest.raises(ValueError, match="equity"):
        _make_request(equity=float("nan"))


def test_non_finite_risk_per_trade_is_rejected() -> None:
    with pytest.raises(ValueError, match="risk_per_trade"):
        _make_request(risk_per_trade=float("inf"))


def test_non_finite_entry_price_is_rejected() -> None:
    with pytest.raises(ValueError, match="entry_price"):
        _make_request(entry_price=float("nan"))


def test_non_finite_stop_price_is_rejected() -> None:
    with pytest.raises(ValueError, match="stop_price"):
        _make_request(stop_price=float("inf"))


# =============================================================================
# Type contract
# =============================================================================

def test_calculator_rejects_wrong_request_type() -> None:
    with pytest.raises(TypeError, match="StopLossRiskRequest"):
        StopLossRiskCalculator().calculate(
            "invalid"
        )

