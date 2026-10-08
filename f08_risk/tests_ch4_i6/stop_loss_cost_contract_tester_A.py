# f08_risk/tests_ch4_i6/stop_loss_cost_contract_tester_A.py (37)
#
# Run: pytest -v -s f08_risk/tests_ch4_i6/stop_loss_cost_contract_tester_A.py

# =============================================================================
# Contract / edge-case tester for f08_risk.stop_loss_cost
# =============================================================================

from __future__ import annotations
from math import isclose
import pytest

from f08_risk.stop_loss_cost import (
    StopLossCostCalculator,
    StopLossCostRequest,
    StopLossCostResult,
)


def _make_request(
    *,
    entry_price: float = 1.1000,
    stop_price: float = 1.0900,
    contract_size: float = 100_000.0,
    currency_conversion_rate: float = 1.0,
) -> StopLossCostRequest:

    return StopLossCostRequest(
        symbol="EURUSD",
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
        request.entry_price = 1.2000


def test_result_is_frozen_slots_dataclass() -> None:
    result = StopLossCostCalculator().calculate(
        _make_request()
    )

    assert isinstance(result, StopLossCostResult)
    assert hasattr(result, "__dataclass_fields__")
    assert "__dict__" not in dir(result)

    with pytest.raises(AttributeError):
        result.loss_per_lot_at_stop = 1.0


# =============================================================================
# Core calculation
# =============================================================================

def test_stop_distance_is_absolute_price_difference() -> None:
    result = StopLossCostCalculator().calculate(
        _make_request(
            entry_price=1.1000,
            stop_price=1.0900,
        )
    )

    _assert_close(
        result.stop_distance,
        0.0100,
    )


def test_loss_per_lot_at_stop_is_calculated() -> None:
    result = StopLossCostCalculator().calculate(
        _make_request(
            entry_price=1.1000,
            stop_price=1.0900,
            contract_size=100_000.0,
            currency_conversion_rate=1.0,
        )
    )

    _assert_close(
        result.loss_per_lot_at_stop,
        1_000.0,
    )


def test_long_style_and_short_style_have_same_loss_cost() -> None:
    calculator = StopLossCostCalculator()

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
        long_style.loss_per_lot_at_stop,
        short_style.loss_per_lot_at_stop,
    )


def test_larger_stop_distance_produces_larger_loss_per_lot() -> None:
    calculator = StopLossCostCalculator()

    small_stop = calculator.calculate(
        _make_request(
            entry_price=1.1000,
            stop_price=1.0950,
        )
    )

    large_stop = calculator.calculate(
        _make_request(
            entry_price=1.1000,
            stop_price=1.0900,
        )
    )

    assert large_stop.loss_per_lot_at_stop > (
        small_stop.loss_per_lot_at_stop
    )


def test_larger_contract_size_produces_larger_loss() -> None:
    calculator = StopLossCostCalculator()

    base = calculator.calculate(
        _make_request(
            contract_size=100_000.0,
        )
    )

    doubled = calculator.calculate(
        _make_request(
            contract_size=200_000.0,
        )
    )

    _assert_close(
        doubled.loss_per_lot_at_stop,
        base.loss_per_lot_at_stop * 2.0,
    )


def test_currency_conversion_scales_loss() -> None:
    calculator = StopLossCostCalculator()

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


# =============================================================================
# Validation
# =============================================================================

@pytest.mark.parametrize(
    "entry_price,stop_price",
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (1.0, 1.0),
    ],
)
def test_invalid_prices_are_rejected(
    entry_price: float,
    stop_price: float,
) -> None:

    with pytest.raises(ValueError):
        _make_request(
            entry_price=entry_price,
            stop_price=stop_price,
        )


@pytest.mark.parametrize(
    "contract_size",
    [0.0, -1.0],
)
def test_non_positive_contract_size_is_rejected(
    contract_size: float,
) -> None:

    with pytest.raises(
        ValueError,
        match="contract_size",
    ):
        _make_request(
            contract_size=contract_size,
        )


@pytest.mark.parametrize(
    "currency_conversion_rate",
    [0.0, -1.0],
)
def test_non_positive_conversion_rate_is_rejected(
    currency_conversion_rate: float,
) -> None:

    with pytest.raises(
        ValueError,
        match="currency_conversion_rate",
    ):
        _make_request(
            currency_conversion_rate=currency_conversion_rate,
        )


def test_non_finite_entry_price_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="entry_price",
    ):
        _make_request(
            entry_price=float("nan"),
        )


def test_non_finite_stop_price_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="stop_price",
    ):
        _make_request(
            stop_price=float("inf"),
        )


def test_non_finite_contract_size_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="contract_size",
    ):
        _make_request(
            contract_size=float("nan"),
        )


def test_non_finite_conversion_rate_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="currency_conversion_rate",
    ):
        _make_request(
            currency_conversion_rate=float("inf"),
        )


# =============================================================================
# Type contract
# =============================================================================

def test_calculator_rejects_wrong_request_type() -> None:
    with pytest.raises(
        TypeError,
        match="StopLossCostRequest",
    ):
        StopLossCostCalculator().calculate(
            "invalid"
        )

