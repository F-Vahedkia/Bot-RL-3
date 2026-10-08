# f08_risk/tests_ch4_i0prev/projection_margin_tester_A.py (17)
#
# Run: pytest -v -s f08_risk/tests_ch4_i0prev/projection_margin_tester_A.py

from __future__ import annotations

import pytest

from f08_risk.margin_calculator import InstrumentMarginSpec
from f08_risk.projection_margin import (
    ProjectionMarginCalculator,
    ProjectionMarginRequest,
)


@pytest.fixture()
def calculator() -> ProjectionMarginCalculator:
    return ProjectionMarginCalculator()


# =============================================================================
# Single-symbol calculation
# =============================================================================

def test_projection_margin_calculation(
    calculator: ProjectionMarginCalculator,
) -> None:
    request = ProjectionMarginRequest(
        symbol="EURUSD",
        target_exposure=0.20,
        equity=10_000.0,
        price=1.10,
        instrument=InstrumentMarginSpec(
            symbol="EURUSD",
            contract_size=100_000.0,
            leverage=100.0,
        ),
    )

    result = calculator.calculate(request)

    assert result.target_notional == pytest.approx(2_000.0)

    expected_lots = 2_000.0 / 110_000.0

    assert result.target_lots == pytest.approx(expected_lots)

    # The resulting notional is 2,000 USD.
    assert result.required_margin == pytest.approx(20.0)

    assert result.margin_rate == pytest.approx(0.01)


def test_explicit_margin_rate(
    calculator: ProjectionMarginCalculator,
) -> None:
    request = ProjectionMarginRequest(
        symbol="XAUUSD",
        target_exposure=0.20,
        equity=10_000.0,
        price=2_000.0,
        instrument=InstrumentMarginSpec(
            symbol="XAUUSD",
            contract_size=100.0,
            leverage=100.0,
            margin_rate=0.025,
        ),
    )

    result = calculator.calculate(request)

    assert result.target_notional == pytest.approx(2_000.0)

    expected_lots = 2_000.0 / 200_000.0

    assert result.target_lots == pytest.approx(expected_lots)

    assert result.required_margin == pytest.approx(50.0)


def test_currency_conversion_propagates_through_pipeline(
    calculator: ProjectionMarginCalculator,
) -> None:
    request = ProjectionMarginRequest(
        symbol="EURGBP",
        target_exposure=0.20,
        equity=10_000.0,
        price=0.85,
        instrument=InstrumentMarginSpec(
            symbol="EURGBP",
            contract_size=100_000.0,
            leverage=50.0,
        ),
        currency_conversion_rate=1.20,
    )

    result = calculator.calculate(request)

    expected_lots = (
        2_000.0 / (0.85 * 100_000.0 * 1.20)
    )

    expected_margin = 2_000.0 / 50.0

    assert result.target_lots == pytest.approx(expected_lots)
    assert result.required_margin == pytest.approx(expected_margin)


def test_negative_exposure_uses_absolute_size(
    calculator: ProjectionMarginCalculator,
) -> None:
    request = ProjectionMarginRequest(
        symbol="EURUSD",
        target_exposure=-0.20,
        equity=10_000.0,
        price=1.10,
        instrument=InstrumentMarginSpec(
            symbol="EURUSD",
            contract_size=100_000.0,
            leverage=100.0,
        ),
    )

    result = calculator.calculate(request)

    expected_lots = 2_000.0 / 110_000.0

    assert result.target_exposure == pytest.approx(-0.20)
    assert result.target_notional == pytest.approx(2_000.0)
    assert result.target_lots == pytest.approx(expected_lots)
    assert result.required_margin == pytest.approx(20.0)

# =============================================================================
# Multiple symbols
# =============================================================================

def test_calculate_many(
    calculator: ProjectionMarginCalculator,
) -> None:
    requests = {
        "EURUSD": ProjectionMarginRequest(
            symbol="EURUSD",
            target_exposure=0.10,
            equity=10_000.0,
            price=1.10,
            instrument=InstrumentMarginSpec(
                symbol="EURUSD",
                contract_size=100_000.0,
                leverage=100.0,
            ),
        ),
        "XAUUSD": ProjectionMarginRequest(
            symbol="XAUUSD",
            target_exposure=0.20,
            equity=10_000.0,
            price=2_000.0,
            instrument=InstrumentMarginSpec(
                symbol="XAUUSD",
                contract_size=100.0,
                leverage=100.0,
            ),
        ),
    }

    results = calculator.calculate_many(requests)

    assert set(results) == {"EURUSD", "XAUUSD"}

    assert results["EURUSD"].target_notional == pytest.approx(1_000.0)
    assert results["XAUUSD"].target_notional == pytest.approx(2_000.0)

    assert results["EURUSD"].required_margin == pytest.approx(10.0)
    assert results["XAUUSD"].required_margin == pytest.approx(20.0)


# =============================================================================
# Zero exposure
# =============================================================================

def test_zero_exposure(
    calculator: ProjectionMarginCalculator,
) -> None:
    request = ProjectionMarginRequest(
        symbol="EURUSD",
        target_exposure=0.0,
        equity=10_000.0,
        price=1.10,
        instrument=InstrumentMarginSpec(
            symbol="EURUSD",
            contract_size=100_000.0,
            leverage=100.0,
        ),
    )

    result = calculator.calculate(request)

    assert result.target_notional == pytest.approx(0.0)
    assert result.target_lots == pytest.approx(0.0)
    assert result.required_margin == pytest.approx(0.0)


# =============================================================================
# Validation
# =============================================================================

def test_calculate_rejects_wrong_request_type(
    calculator: ProjectionMarginCalculator,
) -> None:
    with pytest.raises(TypeError, match="ProjectionMarginRequest"):
        calculator.calculate(object())  # type: ignore[arg-type]


def test_calculate_many_rejects_none(
    calculator: ProjectionMarginCalculator,
) -> None:
    with pytest.raises(ValueError, match="requests are required"):
        calculator.calculate_many(None)  # type: ignore[arg-type]


def test_calculate_many_rejects_key_mismatch(
    calculator: ProjectionMarginCalculator,
) -> None:
    request = ProjectionMarginRequest(
        symbol="EURUSD",
        target_exposure=0.10,
        equity=10_000.0,
        price=1.10,
        instrument=InstrumentMarginSpec(
            symbol="EURUSD",
            contract_size=100_000.0,
            leverage=100.0,
        ),
    )

    with pytest.raises(ValueError, match="request symbol mismatch"):
        calculator.calculate_many(
            {"GBPUSD": request}
        )


# =============================================================================
# Determinism
# =============================================================================

def test_calculation_is_deterministic(
    calculator: ProjectionMarginCalculator,
) -> None:
    request = ProjectionMarginRequest(
        symbol="EURUSD",
        target_exposure=0.20,
        equity=10_000.0,
        price=1.10,
        instrument=InstrumentMarginSpec(
            symbol="EURUSD",
            contract_size=100_000.0,
            leverage=100.0,
        ),
    )

    result_1 = calculator.calculate(request)
    result_2 = calculator.calculate(request)

    assert result_1 == result_2


def test_many_calculation_is_deterministic(
    calculator: ProjectionMarginCalculator,
) -> None:
    requests = {
        "EURUSD": ProjectionMarginRequest(
            symbol="EURUSD",
            target_exposure=0.10,
            equity=10_000.0,
            price=1.10,
            instrument=InstrumentMarginSpec(
                symbol="EURUSD",
                contract_size=100_000.0,
                leverage=100.0,
            ),
        ),
        "XAUUSD": ProjectionMarginRequest(
            symbol="XAUUSD",
            target_exposure=0.20,
            equity=10_000.0,
            price=2_000.0,
            instrument=InstrumentMarginSpec(
                symbol="XAUUSD",
                contract_size=100.0,
                leverage=100.0,
            ),
        ),
    }

    result_1 = calculator.calculate_many(requests)
    result_2 = calculator.calculate_many(requests)

    assert result_1 == result_2

