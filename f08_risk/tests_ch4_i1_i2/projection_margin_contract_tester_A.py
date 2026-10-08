# f08_risk/tests_ch4_i1_i2/projection_margin_contract_tester_A.py (25)
#
# Run: pytest -v -s f08_risk/tests_ch4_i1_i2/projection_margin_contract_tester_A.py

"""
این Tester باید حداقل این‌ها را تثبیت کند:
    - ProjectionMarginRequest برای مقادیر پایه معتبر است.
    - exposure مثبت و منفی هر دو از نظر قرارداد معتبرند.
    - exposure صفر به zero lots و zero margin منتهی می‌شود.
    - target notional به‌درستی از equity و exposure به دست می‌آید.
    - conversion rate در sizing و margin یکسان اعمال می‌شود.
    - margin rate صریح بر 1 / leverage تقدم دارد.
    - وقتی margin rate صریح وجود ندارد، 1 / leverage استفاده می‌شود.
    - نتیجه required_margin دقیقاً با ترکیب sizing و margin calculator سازگار است.
    - علامت target_exposure در ProjectionMarginResult حفظ می‌شود.
    - calculate_many() mapping را درست حفظ می‌کند.
    - mismatch بین کلید mapping و request.symbol رد می‌شود.
    - type-check برای request انجام می‌شود.
    - خروجی چندنمادی deterministic است.

Purpose:
    - Validate the public contract and composition behavior of
    - ProjectionMarginCalculator.
Scope:
    - ProjectionMarginRequest
    - ProjectionMarginResult
    - ProjectionMarginCalculator
    - Composition of PositionSizingCalculator + MarginCalculator

This tester intentionally does NOT modify or prescribe production behavior.
It first establishes the contract that the implementation must satisfy.
"""
from __future__ import annotations

from dataclasses import fields
from math import isclose

import pytest

from f08_risk.margin_calculator import (
    InstrumentMarginSpec,
    MarginCalculator,
)
from f08_risk.position_sizing import (
    PositionSizingCalculator,
)
from f08_risk.projection_margin import (
    ProjectionMarginCalculator,
    ProjectionMarginRequest,
    ProjectionMarginResult,
)


# =============================================================================
# Fixtures / helpers
# =============================================================================

SPEC_LEVERAGE = InstrumentMarginSpec(
    symbol="EURUSD",
    contract_size=100_000.0,
    leverage=100.0,
)

SPEC_EXPLICIT_MARGIN = InstrumentMarginSpec(
    symbol="EURUSD",
    contract_size=100_000.0,
    leverage=100.0,
    margin_rate=0.02,
)


def make_request(
    *,
    target_exposure: float = 0.10,
    equity: float = 10_000.0,
    price: float = 1.25,
    instrument: InstrumentMarginSpec = SPEC_LEVERAGE,
    currency_conversion_rate: float = 1.0,
) -> ProjectionMarginRequest:
    return ProjectionMarginRequest(
        symbol="EURUSD",
        target_exposure=target_exposure,
        equity=equity,
        price=price,
        instrument=instrument,
        currency_conversion_rate=currency_conversion_rate,
    )


def calculator() -> ProjectionMarginCalculator:
    return ProjectionMarginCalculator(
        position_sizer=PositionSizingCalculator(),
        margin_calculator=MarginCalculator(),
    )


# =============================================================================
# Request contract
# =============================================================================

def test_request_is_dataclass_with_expected_fields() -> None:
    names = {
        field.name
        for field in fields(ProjectionMarginRequest)
    }

    assert names == {
        "symbol",
        "target_exposure",
        "equity",
        "price",
        "instrument",
        "currency_conversion_rate",
    }


def test_request_accepts_positive_exposure() -> None:
    request = make_request(target_exposure=0.20)

    assert request.target_exposure == pytest.approx(0.20)


def test_request_accepts_negative_exposure() -> None:
    request = make_request(target_exposure=-0.20)

    assert request.target_exposure == pytest.approx(-0.20)


def test_request_accepts_zero_exposure() -> None:
    request = make_request(target_exposure=0.0)

    assert request.target_exposure == pytest.approx(0.0)


# =============================================================================
# Result contract
# =============================================================================

def test_result_is_dataclass_with_expected_fields() -> None:
    names = {
        field.name
        for field in fields(ProjectionMarginResult)
    }

    assert names == {
        "symbol",
        "target_exposure",
        "target_notional",
        "target_lots",
        "required_margin",
        "margin_rate",
        "calculation_method",
    }


# =============================================================================
# Composition: Position Sizing
# =============================================================================

def test_target_notional_is_equity_times_exposure() -> None:
    request = make_request(
        target_exposure=0.25,
        equity=20_000.0,
    )

    result = calculator().calculate(request)

    assert result.target_notional == pytest.approx(5_000.0)


def test_target_lots_are_based_on_absolute_exposure() -> None:
    positive = calculator().calculate(
        make_request(target_exposure=0.10)
    )
    negative = calculator().calculate(
        make_request(target_exposure=-0.10)
    )

    assert positive.target_lots == pytest.approx(
        negative.target_lots
    )

    assert positive.target_exposure == pytest.approx(0.10)
    assert negative.target_exposure == pytest.approx(-0.10)


def test_zero_exposure_produces_zero_lots() -> None:
    result = calculator().calculate(
        make_request(target_exposure=0.0)
    )

    assert result.target_lots == pytest.approx(0.0)


# =============================================================================
# Composition: Margin
# =============================================================================

def test_default_margin_rate_is_one_over_leverage() -> None:
    result = calculator().calculate(
        make_request(
            target_exposure=0.10,
            instrument=SPEC_LEVERAGE,
        )
    )

    assert result.margin_rate == pytest.approx(0.01)


def test_explicit_margin_rate_takes_precedence_over_leverage() -> None:
    result = calculator().calculate(
        make_request(
            target_exposure=0.10,
            instrument=SPEC_EXPLICIT_MARGIN,
        )
    )

    assert result.margin_rate == pytest.approx(0.02)


def test_required_margin_matches_composed_calculations() -> None:
    request = make_request(
        target_exposure=0.10,
        equity=10_000.0,
        price=1.25,
        instrument=SPEC_LEVERAGE,
    )

    sizing = PositionSizingCalculator().calculate(
        request=__import__(
            "f08_risk.position_sizing",
            fromlist=["PositionSizingRequest"],
        ).PositionSizingRequest(
            symbol="EURUSD",
            target_exposure=0.10,
            equity=10_000.0,
            price=1.25,
            contract_size=100_000.0,
            currency_conversion_rate=1.0,
        )
    )

    expected_margin = (
        sizing.volume_lots
        * SPEC_LEVERAGE.contract_size
        * request.price
        * 1.0
        * (1.0 / SPEC_LEVERAGE.leverage)
    )

    result = calculator().calculate(request)

    assert result.required_margin == pytest.approx(
        expected_margin
    )


def test_conversion_rate_affects_notional_and_margin() -> None:
    base = calculator().calculate(
        make_request(
            target_exposure=0.10,
            currency_conversion_rate=1.0,
        )
    )

    converted = calculator().calculate(
        make_request(
            target_exposure=0.10,
            currency_conversion_rate=2.0,
        )
    )

    assert converted.target_notional == pytest.approx(
        base.target_notional
    )

    assert converted.target_lots == pytest.approx(
        base.target_lots / 2.0
    )

    assert converted.required_margin == pytest.approx(
        base.required_margin
    )


# =============================================================================
# Zero / edge composition
# =============================================================================

def test_zero_exposure_produces_zero_margin() -> None:
    result = calculator().calculate(
        make_request(target_exposure=0.0)
    )

    assert result.target_lots == pytest.approx(0.0)
    assert result.required_margin == pytest.approx(0.0)


def test_negative_exposure_preserves_sign_only_in_result() -> None:
    result = calculator().calculate(
        make_request(target_exposure=-0.15)
    )

    assert result.target_exposure == pytest.approx(-0.15)
    assert result.target_lots >= 0.0
    assert result.required_margin >= 0.0


# =============================================================================
# Request type validation
# =============================================================================

def test_calculate_rejects_wrong_request_type() -> None:
    with pytest.raises(TypeError, match="ProjectionMarginRequest"):
        calculator().calculate("bad")  # type: ignore[arg-type]


# =============================================================================
# calculate_many
# =============================================================================

def test_calculate_many_returns_all_requested_symbols() -> None:
    eurusd = ProjectionMarginRequest(
        symbol="EURUSD",
        target_exposure=0.10,
        equity=10_000.0,
        price=1.25,
        instrument=SPEC_LEVERAGE,
    )

    gbpusd_spec = InstrumentMarginSpec(
        symbol="GBPUSD",
        contract_size=100_000.0,
        leverage=50.0,
    )

    gbpusd = ProjectionMarginRequest(
        symbol="GBPUSD",
        target_exposure=0.05,
        equity=10_000.0,
        price=1.30,
        instrument=gbpusd_spec,
    )

    results = calculator().calculate_many(
        {
            "EURUSD": eurusd,
            "GBPUSD": gbpusd,
        }
    )

    assert set(results) == {
        "EURUSD",
        "GBPUSD",
    }

    assert results["EURUSD"].symbol == "EURUSD"
    assert results["GBPUSD"].symbol == "GBPUSD"


def test_calculate_many_rejects_mapping_key_mismatch() -> None:
    request = make_request()

    with pytest.raises(ValueError, match="request symbol mismatch"):
        calculator().calculate_many(
            {
                "GBPUSD": request,
            }
        )


def test_calculate_many_rejects_none() -> None:
    with pytest.raises(ValueError, match="requests are required"):
        calculator().calculate_many(None)  # type: ignore[arg-type]


def test_calculate_many_is_deterministic() -> None:
    request_a = make_request(target_exposure=0.10)

    request_b = ProjectionMarginRequest(
        symbol="GBPUSD",
        target_exposure=0.05,
        equity=10_000.0,
        price=1.30,
        instrument=InstrumentMarginSpec(
            symbol="GBPUSD",
            contract_size=100_000.0,
            leverage=50.0,
        ),
    )

    requests = {
        "EURUSD": request_a,
        "GBPUSD": request_b,
    }

    first = calculator().calculate_many(requests)
    second = calculator().calculate_many(requests)

    assert list(first.keys()) == list(second.keys())

    for symbol in first:
        assert first[symbol].symbol == second[symbol].symbol
        assert first[symbol].target_exposure == pytest.approx(
            second[symbol].target_exposure
        )
        assert first[symbol].target_notional == pytest.approx(
            second[symbol].target_notional
        )
        assert first[symbol].target_lots == pytest.approx(
            second[symbol].target_lots
        )
        assert first[symbol].required_margin == pytest.approx(
            second[symbol].required_margin
        )
