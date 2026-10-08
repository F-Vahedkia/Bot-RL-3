# f08_risk/tests_ch4_i1_i2/position_sizing_contract_tester_A.py (26)
#
# Run from project root:
#     python -m f08_risk.tests_ch4_i1_i2.position_sizing_contract_tester_A
#     pytest -v -s f08_risk/tests_ch4_i1_i2/position_sizing_contract_tester_A.py

"""
Contract / edge-case tester for f08_risk.position_sizing.

Scope
-----
This tester validates the CURRENT contract of PositionSizingCalculator.

Important:
    - PositionSizingCalculator currently calculates CONTINUOUS lot volume.
    - It does NOT apply broker execution constraints such as:
        min_lot
        lot_step
        max_lot
    - Therefore this tester intentionally verifies the mathematical sizing
    contract and does not expect broker-lot normalization.
"""

from __future__ import annotations

from math import isclose, isfinite

from f08_risk.position_sizing import (
    PositionSizingCalculator,
    PositionSizingRequest,
    PositionSizingResult,
)


def _assert_close(actual: float, expected: float, *, msg: str = "") -> None:
    assert isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (
        msg or f"expected {expected}, got {actual}"
    )


def _make_request(
    *,
    target_exposure: float = 0.10,
    equity: float = 10_000.0,
    price: float = 1.25,
    contract_size: float = 100_000.0,
    currency_conversion_rate: float = 1.0,
) -> PositionSizingRequest:
    return PositionSizingRequest(
        symbol="EURUSD",
        target_exposure=target_exposure,
        equity=equity,
        price=price,
        contract_size=contract_size,
        currency_conversion_rate=currency_conversion_rate,
    )


def test_request_is_frozen_slots_dataclass() -> None:
    request = _make_request()
    assert hasattr(request, "__dataclass_fields__")
    assert "__dict__" not in dir(request)
    try:
        request.equity = 20_000.0
    except AttributeError:
        pass
    else:
        raise AssertionError("PositionSizingRequest must be frozen")


def test_result_is_frozen_slots_dataclass() -> None:
    result = PositionSizingCalculator().calculate(_make_request())
    assert isinstance(result, PositionSizingResult)
    assert hasattr(result, "__dataclass_fields__")
    assert "__dict__" not in dir(result)
    try:
        result.volume_lots = 1.0
    except AttributeError:
        pass
    else:
        raise AssertionError("PositionSizingResult must be frozen")


def test_basic_fractional_sizing() -> None:
    request = _make_request(
        target_exposure=0.10,
        equity=10_000.0,
        price=1.25,
        contract_size=100_000.0,
        currency_conversion_rate=1.0,
    )
    result = PositionSizingCalculator().calculate(request)
    _assert_close(result.target_notional, 1_000.0)
    _assert_close(result.volume_lots, 0.008)
    assert result.symbol == "EURUSD"
    _assert_close(result.target_exposure, 0.10)
    _assert_close(result.equity, 10_000.0)


def test_zero_target_exposure_returns_zero_lots() -> None:
    result = PositionSizingCalculator().calculate(_make_request(target_exposure=0.0))
    _assert_close(result.target_notional, 0.0)
    _assert_close(result.volume_lots, 0.0)


def test_zero_equity_returns_zero_lots() -> None:
    result = PositionSizingCalculator().calculate(
        _make_request(target_exposure=0.25, equity=0.0)
    )
    _assert_close(result.target_notional, 0.0)
    _assert_close(result.volume_lots, 0.0)


def test_currency_conversion_rate_scales_denominator() -> None:
    calculator = PositionSizingCalculator()
    base = calculator.calculate(
        _make_request(
            target_exposure=0.20,
            equity=10_000.0,
            price=1.25,
            contract_size=100_000.0,
            currency_conversion_rate=1.0,
        )
    )
    converted = calculator.calculate(
        _make_request(
            target_exposure=0.20,
            equity=10_000.0,
            price=1.25,
            contract_size=100_000.0,
            currency_conversion_rate=2.0,
        )
    )
    _assert_close(converted.volume_lots, base.volume_lots / 2.0)


def test_large_exposure_is_continuous_and_not_capped_here() -> None:
    result = PositionSizingCalculator().calculate(
        _make_request(
            target_exposure=2.0,
            equity=10_000.0,
            price=1.25,
            contract_size=100_000.0,
        )
    )
    _assert_close(result.target_notional, 20_000.0)
    _assert_close(result.volume_lots, 0.16)


def test_very_small_exposure_is_not_rounded_to_broker_step() -> None:
    result = PositionSizingCalculator().calculate(
        _make_request(
            target_exposure=0.001,
            equity=10_000.0,
            price=1.25,
            contract_size=100_000.0,
        )
    )
    _assert_close(result.volume_lots, 0.00008)
    assert result.volume_lots < 0.01


def test_no_min_lot_is_applied() -> None:
    result = PositionSizingCalculator().calculate(
        _make_request(
            target_exposure=0.001,
            equity=10_000.0,
            price=1.25,
            contract_size=100_000.0,
        )
    )
    assert result.volume_lots < 0.01


def test_no_lot_step_rounding_is_applied() -> None:
    result = PositionSizingCalculator().calculate(
        _make_request(
            target_exposure=0.123456,
            equity=10_000.0,
            price=1.25,
            contract_size=100_000.0,
        )
    )
    expected = (10_000.0 * 0.123456) / (1.25 * 100_000.0)
    _assert_close(result.volume_lots, expected)
    assert result.volume_lots != round(result.volume_lots / 0.01) * 0.01


def test_no_max_lot_is_applied() -> None:
    result = PositionSizingCalculator().calculate(
        _make_request(
            target_exposure=100.0,
            equity=10_000.0,
            price=1.0,
            contract_size=1.0,
        )
    )
    _assert_close(result.volume_lots, 1_000_000.0)


def test_negative_target_exposure_is_rejected_by_current_contract() -> None:
    try:
        _make_request(target_exposure=-0.10)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "negative target_exposure must be rejected by PositionSizingRequest"
        )


def test_request_rejects_zero_or_negative_price() -> None:
    for price in (0.0, -1.0):
        try:
            _make_request(price=price)
        except ValueError:
            pass
        else:
            raise AssertionError(f"price={price} must be rejected")


def test_request_rejects_non_positive_contract_size() -> None:
    for contract_size in (0.0, -1.0):
        try:
            _make_request(contract_size=contract_size)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"contract_size={contract_size} must be rejected"
            )


def test_request_rejects_non_positive_conversion_rate() -> None:
    for rate in (0.0, -1.0):
        try:
            _make_request(currency_conversion_rate=rate)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"currency_conversion_rate={rate} must be rejected"
            )


def test_non_finite_target_exposure_is_rejected() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        try:
            _make_request(target_exposure=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"target_exposure={value!r} must be rejected")


def test_non_finite_equity_is_rejected() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        try:
            _make_request(equity=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"equity={value!r} must be rejected")


def test_non_finite_price_is_rejected() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        try:
            _make_request(price=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"price={value!r} must be rejected")


def test_non_finite_contract_size_is_rejected() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        try:
            _make_request(contract_size=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"contract_size={value!r} must be rejected")


def test_non_finite_conversion_rate_is_rejected() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        try:
            _make_request(currency_conversion_rate=value)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"currency_conversion_rate={value!r} must be rejected"
            )


def test_calculator_rejects_wrong_request_type() -> None:
    try:
        PositionSizingCalculator().calculate(None)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError(
            "calculate() must reject a non-PositionSizingRequest"
        )


def test_result_values_are_finite_and_non_negative() -> None:
    result = PositionSizingCalculator().calculate(_make_request())
    numeric_values = (
        result.target_exposure,
        result.equity,
        result.target_notional,
        result.price,
        result.contract_size,
        result.currency_conversion_rate,
        result.volume_lots,
    )
    assert all(isfinite(value) for value in numeric_values)
    assert result.target_exposure >= 0.0
    assert result.equity >= 0.0
    assert result.target_notional >= 0.0
    assert result.price > 0.0
    assert result.contract_size > 0.0
    assert result.currency_conversion_rate > 0.0
    assert result.volume_lots >= 0.0


def _run_all_tests() -> None:
    tests = [
        test_request_is_frozen_slots_dataclass,
        test_result_is_frozen_slots_dataclass,
        test_basic_fractional_sizing,
        test_zero_target_exposure_returns_zero_lots,
        test_zero_equity_returns_zero_lots,
        test_currency_conversion_rate_scales_denominator,
        test_large_exposure_is_continuous_and_not_capped_here,
        test_very_small_exposure_is_not_rounded_to_broker_step,
        test_no_min_lot_is_applied,
        test_no_lot_step_rounding_is_applied,
        test_no_max_lot_is_applied,
        test_negative_target_exposure_is_rejected_by_current_contract,
        test_request_rejects_zero_or_negative_price,
        test_request_rejects_non_positive_contract_size,
        test_request_rejects_non_positive_conversion_rate,
        test_non_finite_target_exposure_is_rejected,
        test_non_finite_equity_is_rejected,
        test_non_finite_price_is_rejected,
        test_non_finite_contract_size_is_rejected,
        test_non_finite_conversion_rate_is_rejected,
        test_calculator_rejects_wrong_request_type,
        test_result_values_are_finite_and_non_negative,
    ]

    passed = 0
    for test in tests:
        test()
        passed += 1
        print(f"[PASS] {test.__name__}")

    print()
    print(f"{passed}/{len(tests)} tests green.")


if __name__ == "__main__":
    _run_all_tests()
