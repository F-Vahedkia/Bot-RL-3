# f08_risk/tests_ch4_i0prev/position_sizing_tester_A.py (15)
#
# Run: pytest -v -s f08_risk/tests_ch4_i0prev/position_sizing_tester_A.py

from __future__ import annotations
import pytest

from f08_risk.position_sizing import (
    PositionSizingCalculator,
    PositionSizingRequest,
)


@pytest.fixture()
def calculator() -> PositionSizingCalculator:
    return PositionSizingCalculator()


# =============================================================================
# Basic sizing
# =============================================================================

def test_target_exposure_to_notional(
    calculator: PositionSizingCalculator,
) -> None:
    request = PositionSizingRequest(
        symbol="EURUSD",
        target_exposure=0.20,
        equity=10_000.0,
        price=1.10,
        contract_size=100_000.0,
    )

    result = calculator.calculate(request)

    assert result.target_notional == pytest.approx(2_000.0)


def test_target_exposure_to_lots(
    calculator: PositionSizingCalculator,
) -> None:
    request = PositionSizingRequest(
        symbol="EURUSD",
        target_exposure=0.20,
        equity=10_000.0,
        price=1.10,
        contract_size=100_000.0,
    )

    result = calculator.calculate(request)

    expected_lots = 2_000.0 / 110_000.0

    assert result.volume_lots == pytest.approx(expected_lots)


def test_sizing_scales_linearly_with_exposure(
    calculator: PositionSizingCalculator,
) -> None:
    request_a = PositionSizingRequest(
        symbol="EURUSD",
        target_exposure=0.10,
        equity=10_000.0,
        price=1.10,
        contract_size=100_000.0,
    )

    request_b = PositionSizingRequest(
        symbol="EURUSD",
        target_exposure=0.20,
        equity=10_000.0,
        price=1.10,
        contract_size=100_000.0,
    )

    result_a = calculator.calculate(request_a)
    result_b = calculator.calculate(request_b)

    assert result_b.volume_lots == pytest.approx(
        result_a.volume_lots * 2.0
    )


def test_currency_conversion_is_applied(
    calculator: PositionSizingCalculator,
) -> None:
    request = PositionSizingRequest(
        symbol="EURGBP",
        target_exposure=0.20,
        equity=10_000.0,
        price=0.85,
        contract_size=100_000.0,
        currency_conversion_rate=1.20,
    )

    result = calculator.calculate(request)

    expected_lots = (
        2_000.0
        / (0.85 * 100_000.0 * 1.20)
    )

    assert result.volume_lots == pytest.approx(expected_lots)


# =============================================================================
# Edge cases
# =============================================================================

def test_zero_exposure_produces_zero_lots(
    calculator: PositionSizingCalculator,
) -> None:
    request = PositionSizingRequest(
        symbol="EURUSD",
        target_exposure=0.0,
        equity=10_000.0,
        price=1.10,
        contract_size=100_000.0,
    )

    result = calculator.calculate(request)

    assert result.target_notional == pytest.approx(0.0)
    assert result.volume_lots == pytest.approx(0.0)


def test_zero_equity_produces_zero_lots(
    calculator: PositionSizingCalculator,
) -> None:
    request = PositionSizingRequest(
        symbol="EURUSD",
        target_exposure=0.20,
        equity=0.0,
        price=1.10,
        contract_size=100_000.0,
    )

    result = calculator.calculate(request)

    assert result.target_notional == pytest.approx(0.0)
    assert result.volume_lots == pytest.approx(0.0)


# =============================================================================
# Validation
# =============================================================================

def test_negative_exposure_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="target_exposure",
    ):
        PositionSizingRequest(
            symbol="EURUSD",
            target_exposure=-0.1,
            equity=10_000.0,
            price=1.10,
            contract_size=100_000.0,
        )


def test_negative_equity_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="equity",
    ):
        PositionSizingRequest(
            symbol="EURUSD",
            target_exposure=0.20,
            equity=-1.0,
            price=1.10,
            contract_size=100_000.0,
        )


def test_invalid_price_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="price",
    ):
        PositionSizingRequest(
            symbol="EURUSD",
            target_exposure=0.20,
            equity=10_000.0,
            price=0.0,
            contract_size=100_000.0,
        )


def test_invalid_contract_size_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="contract_size",
    ):
        PositionSizingRequest(
            symbol="EURUSD",
            target_exposure=0.20,
            equity=10_000.0,
            price=1.10,
            contract_size=0.0,
        )


def test_invalid_conversion_rate_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="currency_conversion_rate",
    ):
        PositionSizingRequest(
            symbol="EURGBP",
            target_exposure=0.20,
            equity=10_000.0,
            price=0.85,
            contract_size=100_000.0,
            currency_conversion_rate=0.0,
        )


# =============================================================================
# Determinism
# =============================================================================

def test_calculation_is_deterministic(
    calculator: PositionSizingCalculator,
) -> None:
    request = PositionSizingRequest(
        symbol="EURUSD",
        target_exposure=0.25,
        equity=20_000.0,
        price=1.12,
        contract_size=100_000.0,
    )

    result_1 = calculator.calculate(request)
    result_2 = calculator.calculate(request)

    assert result_1 == result_2

