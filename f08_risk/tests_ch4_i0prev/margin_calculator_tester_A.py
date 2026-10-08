# f08_risk/tests_ch4_i0prev/margin_calculator_tester_A.py (13)
#
# Run: pytest -v -s f08_risk/tests_ch4_i0prev/margin_calculator_tester_A.py

from __future__ import annotations

import pytest

from f08_risk.margin_calculator import (
    MARGIN_MODE_FOREX_LEVERAGE,
    InstrumentMarginSpec,
    MarginCalculator,
    MarginRequest,
)


@pytest.fixture()
def calculator() -> MarginCalculator:
    return MarginCalculator()


# =============================================================================
# Basic calculation
# =============================================================================

def test_forex_leverage_margin(calculator: MarginCalculator) -> None:
    spec = InstrumentMarginSpec(
        symbol="EURUSD",
        contract_size=100_000.0,
        leverage=100.0,
    )

    request = MarginRequest(
        symbol="EURUSD",
        volume_lots=1.0,
        price=1.10,
        spec=spec,
    )

    result = calculator.calculate(request)

    assert result.notional_value == pytest.approx(110_000.0)
    assert result.margin_rate == pytest.approx(0.01)
    assert result.required_margin == pytest.approx(1_100.0)


def test_multiple_lots_scale_linearly(
    calculator: MarginCalculator,
) -> None:
    spec = InstrumentMarginSpec(
        symbol="EURUSD",
        contract_size=100_000.0,
        leverage=100.0,
    )

    request = MarginRequest(
        symbol="EURUSD",
        volume_lots=2.5,
        price=1.10,
        spec=spec,
    )

    result = calculator.calculate(request)

    assert result.notional_value == pytest.approx(275_000.0)
    assert result.required_margin == pytest.approx(2_750.0)


def test_currency_conversion_is_applied(
    calculator: MarginCalculator,
) -> None:
    spec = InstrumentMarginSpec(
        symbol="EURGBP",
        contract_size=100_000.0,
        leverage=50.0,
    )

    request = MarginRequest(
        symbol="EURGBP",
        volume_lots=1.0,
        price=0.85,
        spec=spec,
        currency_conversion_rate=1.20,
    )

    result = calculator.calculate(request)

    expected_notional = 100_000.0 * 0.85 * 1.20
    expected_margin = expected_notional / 50.0

    assert result.notional_value == pytest.approx(expected_notional)
    assert result.required_margin == pytest.approx(expected_margin)


# =============================================================================
# Explicit margin rate
# =============================================================================

def test_explicit_margin_rate_overrides_leverage(
    calculator: MarginCalculator,
) -> None:
    spec = InstrumentMarginSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        leverage=100.0,
        margin_rate=0.025,
    )

    request = MarginRequest(
        symbol="XAUUSD",
        volume_lots=1.0,
        price=2_000.0,
        spec=spec,
    )

    result = calculator.calculate(request)

    assert result.notional_value == pytest.approx(200_000.0)
    assert result.margin_rate == pytest.approx(0.025)
    assert result.required_margin == pytest.approx(5_000.0)


# =============================================================================
# Zero volume
# =============================================================================

def test_zero_volume_requires_zero_margin(
    calculator: MarginCalculator,
) -> None:
    spec = InstrumentMarginSpec(
        symbol="EURUSD",
        contract_size=100_000.0,
        leverage=100.0,
    )

    request = MarginRequest(
        symbol="EURUSD",
        volume_lots=0.0,
        price=1.10,
        spec=spec,
    )

    result = calculator.calculate(request)

    assert result.notional_value == pytest.approx(0.0)
    assert result.required_margin == pytest.approx(0.0)


# =============================================================================
# Validation
# =============================================================================

def test_rejects_symbol_mismatch() -> None:
    spec = InstrumentMarginSpec(
        symbol="EURUSD",
        contract_size=100_000.0,
        leverage=100.0,
    )

    with pytest.raises(ValueError, match="symbol mismatch"):
        MarginRequest(
            symbol="GBPUSD",
            volume_lots=1.0,
            price=1.25,
            spec=spec,
        )


def test_rejects_invalid_lots() -> None:
    spec = InstrumentMarginSpec(
        symbol="EURUSD",
        contract_size=100_000.0,
        leverage=100.0,
    )

    with pytest.raises(ValueError, match="volume_lots"):
        MarginRequest(
            symbol="EURUSD",
            volume_lots=-1.0,
            price=1.10,
            spec=spec,
        )


def test_rejects_invalid_price() -> None:
    spec = InstrumentMarginSpec(
        symbol="EURUSD",
        contract_size=100_000.0,
        leverage=100.0,
    )

    with pytest.raises(ValueError, match="price"):
        MarginRequest(
            symbol="EURUSD",
            volume_lots=1.0,
            price=0.0,
            spec=spec,
        )


def test_rejects_invalid_leverage() -> None:
    with pytest.raises(ValueError, match="leverage"):
        InstrumentMarginSpec(
            symbol="EURUSD",
            contract_size=100_000.0,
            leverage=0.0,
        )


def test_rejects_invalid_margin_rate() -> None:
    with pytest.raises(ValueError, match="margin_rate"):
        InstrumentMarginSpec(
            symbol="EURUSD",
            contract_size=100_000.0,
            leverage=100.0,
            margin_rate=1.5,
        )


# =============================================================================
# Type / mode
# =============================================================================

def test_result_contains_calculation_method(
    calculator: MarginCalculator,
) -> None:
    spec = InstrumentMarginSpec(
        symbol="EURUSD",
        contract_size=100_000.0,
        leverage=100.0,
    )

    request = MarginRequest(
        symbol="EURUSD",
        volume_lots=1.0,
        price=1.10,
        spec=spec,
        margin_mode=MARGIN_MODE_FOREX_LEVERAGE,
    )

    result = calculator.calculate(request)

    assert result.calculation_method == MARGIN_MODE_FOREX_LEVERAGE


def test_calculation_is_deterministic(
    calculator: MarginCalculator,
) -> None:
    spec = InstrumentMarginSpec(
        symbol="EURUSD",
        contract_size=100_000.0,
        leverage=100.0,
    )

    request = MarginRequest(
        symbol="EURUSD",
        volume_lots=1.5,
        price=1.12,
        spec=spec,
    )

    result_1 = calculator.calculate(request)
    result_2 = calculator.calculate(request)

    assert result_1 == result_2

