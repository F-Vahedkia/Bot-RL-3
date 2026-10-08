# f08_risk/tests_ch4_i1_i2/margin_calculator_contract_tester_A.py (27)
#
# Run from project root:
#     python -m f08_risk.tests_ch4_i0prev.margin_calculator_contract_tester_A
#     pytest -v -s f08_risk/tests_ch4_i1_i2/margin_calculator_contract_tester_A.py

"""
Contract / edge-case tester for f08_risk.margin_calculator.

Scope
-----
This tester validates the CURRENT contract of:
    InstrumentMarginSpec
    MarginRequest
    MarginResult
    MarginCalculator

Important:
    - The calculator is broker-independent.
    - Default forex leverage mode resolves margin_rate as 1 / leverage.
    - Explicit margin_rate overrides leverage-derived margin rate.
    - currency_conversion_rate scales the notional value before margin is applied.
    - This tester does not introduce broker-specific margin rules.

"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from math import isclose, isfinite

from f08_risk.margin_calculator import (
    MARGIN_MODE_FOREX_LEVERAGE,
    InstrumentMarginSpec,
    MarginCalculator,
    MarginRequest,
    MarginResult,
)


# =============================================================================
# Helpers
# =============================================================================

def _assert_close(actual: float, expected: float, *, msg: str = "") -> None:
    assert isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ), msg or f"expected {expected}, got {actual}"


def _spec(
    *,
    symbol: str = "EURUSD",
    contract_size: float = 100_000.0,
    leverage: float = 100.0,
    margin_rate: float | None = None,
) -> InstrumentMarginSpec:
    return InstrumentMarginSpec(
        symbol=symbol,
        contract_size=contract_size,
        leverage=leverage,
        margin_rate=margin_rate,
    )


def _request(
    *,
    symbol: str = "EURUSD",
    volume_lots: float = 1.0,
    price: float = 1.25,
    spec: InstrumentMarginSpec | None = None,
    currency_conversion_rate: float = 1.0,
    margin_mode: str = MARGIN_MODE_FOREX_LEVERAGE,
) -> MarginRequest:
    actual_spec = spec or _spec(symbol=symbol)

    return MarginRequest(
        symbol=symbol,
        volume_lots=volume_lots,
        price=price,
        spec=actual_spec,
        currency_conversion_rate=currency_conversion_rate,
        margin_mode=margin_mode,
    )


# =============================================================================
# InstrumentMarginSpec contract
# =============================================================================

def test_spec_accepts_valid_leverage_without_explicit_margin_rate() -> None:
    spec = _spec(leverage=100.0)

    assert spec.symbol == "EURUSD"
    _assert_close(spec.contract_size, 100_000.0)
    _assert_close(spec.leverage, 100.0)
    assert spec.margin_rate is None


def test_spec_accepts_valid_explicit_margin_rate() -> None:
    spec = _spec(
        leverage=100.0,
        margin_rate=0.025,
    )

    _assert_close(spec.margin_rate, 0.025)


def test_spec_rejects_empty_symbol() -> None:
    try:
        _spec(symbol="")
    except ValueError:
        pass
    else:
        raise AssertionError("empty symbol must be rejected")


def test_spec_rejects_invalid_contract_size() -> None:
    for value in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        try:
            _spec(contract_size=value)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"contract_size={value!r} must be rejected"
            )


def test_spec_rejects_invalid_leverage() -> None:
    for value in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        try:
            _spec(leverage=value)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"leverage={value!r} must be rejected"
            )


def test_spec_rejects_invalid_explicit_margin_rate() -> None:
    for value in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        try:
            _spec(margin_rate=value)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"margin_rate={value!r} must be rejected"
            )


def test_spec_rejects_margin_rate_above_one() -> None:
    try:
        _spec(margin_rate=1.0000001)
    except ValueError:
        pass
    else:
        raise AssertionError("margin_rate > 1 must be rejected")


def test_spec_boundary_margin_rate_one_is_valid() -> None:
    spec = _spec(margin_rate=1.0)
    _assert_close(spec.margin_rate or 0.0, 1.0)


# =============================================================================
# MarginRequest contract
# =============================================================================

def test_request_is_frozen_slots_dataclass() -> None:
    request = _request()

    assert hasattr(request, "__dataclass_fields__")
    assert "__dict__" not in dir(request)

    try:
        request.price = 2.0
    except (AttributeError, FrozenInstanceError):
        pass
    else:
        raise AssertionError("MarginRequest must be frozen")


def test_request_accepts_zero_volume() -> None:
    request = _request(volume_lots=0.0)

    assert request.volume_lots == 0.0


def test_request_rejects_negative_volume() -> None:
    try:
        _request(volume_lots=-0.01)
    except ValueError:
        pass
    else:
        raise AssertionError("negative volume_lots must be rejected")


def test_request_rejects_invalid_price() -> None:
    for value in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        try:
            _request(price=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"price={value!r} must be rejected")


def test_request_rejects_invalid_conversion_rate() -> None:
    for value in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        try:
            _request(currency_conversion_rate=value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"currency_conversion_rate={value!r} must be rejected")


def test_request_requires_symbol_to_match_spec() -> None:
    spec = _spec(symbol="GBPUSD")

    try:
        _request(symbol="EURUSD", spec=spec)
    except ValueError:
        pass
    else:
        raise AssertionError("request symbol and spec symbol must match")


def test_request_rejects_unsupported_margin_mode() -> None:
    try:
        _request(margin_mode="unsupported_mode")
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported margin mode must be rejected")


# =============================================================================
# Core calculation
# =============================================================================

def test_leverage_based_margin_rate_is_one_over_leverage() -> None:
    request = _request(
        volume_lots=1.0,
        price=1.25,
        spec=_spec(
            contract_size=100_000.0,
            leverage=100.0,
            margin_rate=None,
        ),
    )

    result = MarginCalculator().calculate(request)

    # Notional = 1 * 100,000 * 1.25 = 125,000
    # Margin rate = 1 / 100 = 0.01
    # Required margin = 1,250
    _assert_close(result.notional_value, 125_000.0)
    _assert_close(result.margin_rate, 0.01)
    _assert_close(result.required_margin, 1_250.0)
    assert result.calculation_method == MARGIN_MODE_FOREX_LEVERAGE


def test_explicit_margin_rate_overrides_leverage() -> None:
    request = _request(
        volume_lots=1.0,
        price=1.25,
        spec=_spec(
            contract_size=100_000.0,
            leverage=100.0,
            margin_rate=0.025,
        ),
    )

    result = MarginCalculator().calculate(request)

    _assert_close(result.margin_rate, 0.025)
    _assert_close(result.required_margin, 3_125.0)
    assert result.calculation_method == MARGIN_MODE_FOREX_LEVERAGE


def test_currency_conversion_scales_notional_and_required_margin() -> None:
    base_request = _request(
        volume_lots=1.0,
        price=1.25,
        currency_conversion_rate=1.0,
    )
    converted_request = _request(
        volume_lots=1.0,
        price=1.25,
        currency_conversion_rate=2.0,
    )

    calculator = MarginCalculator()
    base = calculator.calculate(base_request)
    converted = calculator.calculate(converted_request)

    _assert_close(converted.notional_value, base.notional_value * 2.0)
    _assert_close(converted.required_margin, base.required_margin * 2.0)


def test_zero_volume_requires_zero_margin() -> None:
    result = MarginCalculator().calculate(
        _request(volume_lots=0.0)
    )
    _assert_close(result.notional_value, 0.0)
    _assert_close(result.required_margin, 0.0)
    assert result.volume_lots == 0.0


def test_required_margin_scales_linearly_with_volume() -> None:
    calculator = MarginCalculator()

    one_lot = calculator.calculate(
        _request(volume_lots=1.0)
    )
    two_lots = calculator.calculate(
        _request(volume_lots=2.0)
    )
    _assert_close(
        two_lots.required_margin,
        one_lot.required_margin * 2.0,
    )


def test_required_margin_scales_linearly_with_price() -> None:
    calculator = MarginCalculator()

    first = calculator.calculate(
        _request(volume_lots=1.0, price=1.0)
    )
    second = calculator.calculate(
        _request(volume_lots=1.0, price=2.0)
    )
    _assert_close(
        second.required_margin,
        first.required_margin * 2.0,
    )


def test_contract_size_scales_notional_and_margin() -> None:
    calculator = MarginCalculator()

    small = calculator.calculate(
        _request(
            spec=_spec(
                contract_size=10_000.0,
                leverage=100.0,
            )
        )
    )
    large = calculator.calculate(
        _request(
            spec=_spec(
                contract_size=100_000.0,
                leverage=100.0,
            )
        )
    )
    _assert_close(
        large.notional_value,
        small.notional_value * 10.0,
    )
    _assert_close(
        large.required_margin,
        small.required_margin * 10.0,
    )


# =============================================================================
# Result contract
# =============================================================================

def test_result_is_frozen_slots_dataclass() -> None:
    result = MarginCalculator().calculate(_request())

    assert isinstance(result, MarginResult)
    assert "__dict__" not in dir(result)

    try:
        result.required_margin = 1.0
    except (AttributeError, FrozenInstanceError):
        pass
    else:
        raise AssertionError("MarginResult must be frozen")


def test_result_values_are_finite_and_non_negative() -> None:
    result = MarginCalculator().calculate(_request())

    numeric_values = (
        result.volume_lots,
        result.price,
        result.notional_value,
        result.margin_rate,
        result.required_margin,
    )

    assert all(isfinite(value) for value in numeric_values)
    assert result.volume_lots >= 0.0
    assert result.price > 0.0
    assert result.notional_value >= 0.0
    assert result.margin_rate > 0.0
    assert result.required_margin >= 0.0


def test_calculator_rejects_wrong_request_type() -> None:
    try:
        MarginCalculator().calculate(None)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("calculate() must reject a non-MarginRequest")


# =============================================================================
# Runner
# =============================================================================

def _run_all_tests() -> None:
    tests = [
        test_spec_accepts_valid_leverage_without_explicit_margin_rate,
        test_spec_accepts_valid_explicit_margin_rate,
        test_spec_rejects_empty_symbol,
        test_spec_rejects_invalid_contract_size,
        test_spec_rejects_invalid_leverage,
        test_spec_rejects_invalid_explicit_margin_rate,
        test_spec_rejects_margin_rate_above_one,
        test_spec_boundary_margin_rate_one_is_valid,
        test_request_is_frozen_slots_dataclass,
        test_request_accepts_zero_volume,
        test_request_rejects_negative_volume,
        test_request_rejects_invalid_price,
        test_request_rejects_invalid_conversion_rate,
        test_request_requires_symbol_to_match_spec,
        test_request_rejects_unsupported_margin_mode,
        test_leverage_based_margin_rate_is_one_over_leverage,
        test_explicit_margin_rate_overrides_leverage,
        test_currency_conversion_scales_notional_and_required_margin,
        test_zero_volume_requires_zero_margin,
        test_required_margin_scales_linearly_with_volume,
        test_required_margin_scales_linearly_with_price,
        test_contract_size_scales_notional_and_margin,
        test_result_is_frozen_slots_dataclass,
        test_result_values_are_finite_and_non_negative,
        test_calculator_rejects_wrong_request_type,
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

