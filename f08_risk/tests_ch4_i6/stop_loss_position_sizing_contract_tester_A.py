# f08_risk/tests_ch4_i6/stop_loss_position_sizing_contract_tester_A.py (39)
#
# Run: pytest -v -s f08_risk/tests_ch4_i6/stop_loss_position_sizing_contract_tester_A.py

# =============================================================================
# Contract / edge-case tester for integrated Stop-Loss position sizing.
# =============================================================================

from __future__ import annotations

from math import isclose

import pytest

from f08_risk.stop_loss_position_sizing import (
    StopLossPositionSizingCalculator,
    StopLossPositionSizingRequest,
    StopLossPositionSizingResult,
)


def _make_request(
    *,
    equity: float = 10_000.0,
    risk_per_trade: float = 0.01,
    entry_price: float = 1.1000,
    stop_price: float = 1.0900,
    contract_size: float = 100_000.0,
    currency_conversion_rate: float = 1.0,
) -> StopLossPositionSizingRequest:

    return StopLossPositionSizingRequest(
        symbol="EURUSD",
        equity=equity,
        risk_per_trade=risk_per_trade,
        entry_price=entry_price,
        stop_price=stop_price,
        contract_size=contract_size,
        currency_conversion_rate=currency_conversion_rate,
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
    result = StopLossPositionSizingCalculator().calculate(
        _make_request()
    )

    assert isinstance(
        result,
        StopLossPositionSizingResult,
    )
    assert hasattr(result, "__dataclass_fields__")
    assert "__dict__" not in dir(result)

    with pytest.raises(AttributeError):
        result.volume_lots = 1.0


# =============================================================================
# Integrated calculation
# =============================================================================

def test_full_pipeline_calculation() -> None:
    result = StopLossPositionSizingCalculator().calculate(
        _make_request(
            equity=10_000.0,
            risk_per_trade=0.01,
            entry_price=1.1000,
            stop_price=1.0900,
            contract_size=100_000.0,
            currency_conversion_rate=1.0,
        )
    )

    _assert_close(
        result.risk_budget,
        100.0,
    )

    _assert_close(
        result.stop_distance,
        0.0100,
    )

    _assert_close(
        result.loss_per_lot_at_stop,
        1_000.0,
    )

    _assert_close(
        result.volume_lots,
        0.1,
    )


def test_risk_budget_is_passed_into_final_sizing() -> None:
    result = StopLossPositionSizingCalculator().calculate(
        _make_request(
            equity=20_000.0,
            risk_per_trade=0.01,
            entry_price=1.1000,
            stop_price=1.0900,
            contract_size=100_000.0,
        )
    )

    _assert_close(
        result.risk_budget,
        200.0,
    )

    _assert_close(
        result.loss_per_lot_at_stop,
        1_000.0,
    )

    _assert_close(
        result.volume_lots,
        0.2,
    )


def test_wider_stop_reduces_volume() -> None:
    calculator = StopLossPositionSizingCalculator()

    tight = calculator.calculate(
        _make_request(
            entry_price=1.1000,
            stop_price=1.0950,
        )
    )

    wide = calculator.calculate(
        _make_request(
            entry_price=1.1000,
            stop_price=1.0900,
        )
    )

    assert wide.volume_lots < tight.volume_lots


def test_higher_risk_per_trade_increases_volume() -> None:
    calculator = StopLossPositionSizingCalculator()

    one_percent = calculator.calculate(
        _make_request(
            risk_per_trade=0.01,
        )
    )

    two_percent = calculator.calculate(
        _make_request(
            risk_per_trade=0.02,
        )
    )

    _assert_close(
        two_percent.volume_lots,
        one_percent.volume_lots * 2.0,
    )


def test_currency_conversion_affects_final_volume() -> None:
    calculator = StopLossPositionSizingCalculator()

    base = calculator.calculate(
        _make_request(
            currency_conversion_rate=1.0,
        )
    )

    converted = calculator.calculate(
        _make_request(
            currency_conversion_rate=2.0,
        )
    )

    _assert_close(
        converted.loss_per_lot_at_stop,
        base.loss_per_lot_at_stop * 2.0,
    )

    _assert_close(
        converted.volume_lots,
        base.volume_lots / 2.0,
    )


def test_direction_is_not_encoded() -> None:
    calculator = StopLossPositionSizingCalculator()

    long_style = calculator.calculate(
        _make_request(
            entry_price=1.1000,
            stop_price=1.0900,
        )
    )

    short_style = calculator.calculate(
        _make_request(
            entry_price=1.0900,
            stop_price=1.1000,
        )
    )

    _assert_close(
        long_style.stop_distance,
        short_style.stop_distance,
    )

    _assert_close(
        long_style.volume_lots,
        short_style.volume_lots,
    )


def test_zero_risk_produces_zero_volume() -> None:
    result = StopLossPositionSizingCalculator().calculate(
        _make_request(
            risk_per_trade=0.0,
        )
    )

    _assert_close(
        result.risk_budget,
        0.0,
    )

    _assert_close(
        result.volume_lots,
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
        ("contract_size", 0.0, "contract_size"),
        (
            "currency_conversion_rate",
            0.0,
            "currency_conversion_rate",
        ),
    ],
)
def test_invalid_values_are_rejected(
    field: str,
    value: float,
    error_match: str,
) -> None:

    kwargs = {
        "equity": 10_000.0,
        "risk_per_trade": 0.01,
        "entry_price": 1.1000,
        "stop_price": 1.0900,
        "contract_size": 100_000.0,
        "currency_conversion_rate": 1.0,
    }

    kwargs[field] = value

    with pytest.raises(
        ValueError,
        match=error_match,
    ):
        StopLossPositionSizingRequest(
            symbol="EURUSD",
            **kwargs,
        )


def test_equal_prices_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must not be equal",
    ):
        _make_request(
            entry_price=1.1000,
            stop_price=1.1000,
        )


# =============================================================================
# Type contract
# =============================================================================

def test_calculator_rejects_wrong_request_type() -> None:
    with pytest.raises(
        TypeError,
        match="StopLossPositionSizingRequest",
    ):
        StopLossPositionSizingCalculator().calculate(
            "invalid"
        )
