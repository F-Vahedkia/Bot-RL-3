# f08_risk/tests_ch4_i6/risk_based_position_sizing_contract_tester_A.py (35)
#
# Run: pytest -v -s f08_risk/tests_ch4_i6/risk_based_position_sizing_contract_tester_A.py

# =============================================================================
# Contract / edge-case tester for risk-based position sizing.
# =============================================================================

from __future__ import annotations

from math import isclose

import pytest

from f08_risk.risk_based_position_sizing import (
    RiskBasedPositionSizingCalculator,
    RiskBasedPositionSizingRequest,
    RiskBasedPositionSizingResult,
)


def _make_request(
    *,
    risk_budget: float = 100.0,
    loss_per_lot_at_stop: float = 200.0,
) -> RiskBasedPositionSizingRequest:

    return RiskBasedPositionSizingRequest(
        symbol="EURUSD",
        risk_budget=risk_budget,
        loss_per_lot_at_stop=loss_per_lot_at_stop,
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
        request.risk_budget = 200.0


def test_result_is_frozen_slots_dataclass() -> None:
    result = RiskBasedPositionSizingCalculator().calculate(
        _make_request()
    )

    assert isinstance(
        result,
        RiskBasedPositionSizingResult,
    )
    assert hasattr(result, "__dataclass_fields__")
    assert "__dict__" not in dir(result)

    with pytest.raises(AttributeError):
        result.volume_lots = 1.0


# =============================================================================
# Core calculation
# =============================================================================

def test_basic_risk_based_sizing() -> None:
    result = RiskBasedPositionSizingCalculator().calculate(
        _make_request(
            risk_budget=100.0,
            loss_per_lot_at_stop=200.0,
        )
    )

    _assert_close(
        result.volume_lots,
        0.5,
    )


def test_zero_risk_budget_returns_zero_volume() -> None:
    result = RiskBasedPositionSizingCalculator().calculate(
        _make_request(
            risk_budget=0.0,
        )
    )

    _assert_close(
        result.volume_lots,
        0.0,
    )


def test_larger_risk_budget_produces_proportionally_larger_volume() -> None:
    calculator = RiskBasedPositionSizingCalculator()

    base = calculator.calculate(
        _make_request(
            risk_budget=100.0,
            loss_per_lot_at_stop=200.0,
        )
    )

    larger = calculator.calculate(
        _make_request(
            risk_budget=200.0,
            loss_per_lot_at_stop=200.0,
        )
    )

    _assert_close(
        larger.volume_lots,
        base.volume_lots * 2.0,
    )


def test_larger_loss_per_lot_produces_smaller_volume() -> None:
    calculator = RiskBasedPositionSizingCalculator()

    base = calculator.calculate(
        _make_request(
            risk_budget=100.0,
            loss_per_lot_at_stop=200.0,
        )
    )

    larger_loss = calculator.calculate(
        _make_request(
            risk_budget=100.0,
            loss_per_lot_at_stop=400.0,
        )
    )

    _assert_close(
        larger_loss.volume_lots,
        base.volume_lots / 2.0,
    )


def test_symbol_is_preserved() -> None:
    result = RiskBasedPositionSizingCalculator().calculate(
        _make_request()
    )

    assert result.symbol == "EURUSD"


# =============================================================================
# Validation
# =============================================================================

@pytest.mark.parametrize(
    "risk_budget",
    [-1.0],
)
def test_negative_risk_budget_is_rejected(
    risk_budget: float,
) -> None:

    with pytest.raises(
        ValueError,
        match="risk_budget",
    ):
        _make_request(
            risk_budget=risk_budget,
        )


@pytest.mark.parametrize(
    "loss_per_lot_at_stop",
    [0.0, -1.0],
)
def test_non_positive_loss_per_lot_is_rejected(
    loss_per_lot_at_stop: float,
) -> None:

    with pytest.raises(
        ValueError,
        match="loss_per_lot_at_stop",
    ):
        _make_request(
            loss_per_lot_at_stop=loss_per_lot_at_stop,
        )


def test_non_finite_risk_budget_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="risk_budget",
    ):
        _make_request(
            risk_budget=float("nan"),
        )


def test_non_finite_loss_per_lot_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="loss_per_lot_at_stop",
    ):
        _make_request(
            loss_per_lot_at_stop=float("inf"),
        )


# =============================================================================
# Type contract
# =============================================================================

def test_calculator_rejects_wrong_request_type() -> None:
    with pytest.raises(
        TypeError,
        match="RiskBasedPositionSizingRequest",
    ):
        RiskBasedPositionSizingCalculator().calculate(
            "invalid"
        )
