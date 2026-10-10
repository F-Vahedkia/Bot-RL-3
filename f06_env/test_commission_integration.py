# f06_env/test_commission_integration.py
#
# Run: pytest -q f06_env/test_commission_integration.py
"""
این تستر مسیر واقعی زیر را بررسی میکند:
    CommissionProfileCatalog → build_execution_costs → ExecutionSimulator.execute
و همچنین مواردی از جمله:
    - محاسبه بر مبنای لات،
    - کاهش بخشی از پوزیشن،
    - محاسبه بر مبنای ارزش اسمی،
    - تبدیل ارز وابسته به زمان،
    - خطای نرخ تبدیل مفقود و
    - سازگاری رفتار قدیمی.
را هم بررسی میکند.
"""

from datetime import datetime, timedelta, timezone

import pytest

from f03_data.instrument_specs import InstrumentSpec
from f05_transact_costs.commission_profiles import (
    CommissionProfileCatalog,
)
from f05_transact_costs.contracts import (
    CommissionModel,
    MarketQuote,
)
from f06_env.commission_adapter import build_execution_costs
from f06_env.contracts import PositionIntent
from f06_env.execution_simulator import (
    ExecutionCost,
    ExecutionSimulator,
)
from f06_env.portfolio_state import PositionState


_BASE_TIME = datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc)


def _instrument() -> InstrumentSpec:
    return InstrumentSpec(
        symbol="EURUSD",
        pip_size=0.0001,
        tick_size=0.00001,
        tick_value=1.0,
        tick_value_currency="USD",
        contract_size=100_000.0,
        currency_base="EUR",
        currency_profit="USD",
    )


def _quote(
    seconds: int = 0,
    *,
    bid: float = 1.1000,
    ask: float = 1.1002,
) -> MarketQuote:
    return MarketQuote(
        symbol="EURUSD",
        bid=bid,
        ask=ask,
        timestamp=_BASE_TIME + timedelta(seconds=seconds),
        source="simulation",
    )


def _round_turn_catalog() -> CommissionProfileCatalog:
    raw = {
        "active_broker": "broker_a",
        "active_account_type": "raw",
        "default": {
            "basis": "per_lot",
            "rate": 5.0,
            "currency": "USD",
            "verified": True,
        },
        "brokers": {
            "broker_a": {
                "accounts": {
                    "raw": {
                        "strict_symbols": True,
                        "symbols": {
                            "EURUSD": {
                                "basis": "per_lot",
                                "rate": 8.0,
                                "currency": "USD",
                                "billing_basis": "round_turn",
                                "notional_basis": "quote",
                                "verified": True,
                            }
                        },
                    }
                }
            }
        },
    }

    return CommissionProfileCatalog.from_mapping(
        raw,
        fallback_model=CommissionModel(
            basis="per_lot",
            rate=5.0,
            currency="USD",
        ),
    )


def test_per_lot_commission_is_charged_on_open_and_close():
    instrument = _instrument()
    simulator = ExecutionSimulator()
    position = PositionState(symbol="EURUSD")

    cost = ExecutionCost(
        commission_model=CommissionModel(
            basis="per_lot",
            rate=8.0,
            currency="USD",
        ),
        account_currency="USD",
    )

    opened = simulator.execute(
        position=position,
        intent=PositionIntent(
            symbol="EURUSD",
            target_side=1,
            target_lots=0.25,
        ),
        quote=_quote(),
        instrument=instrument,
        cost=cost,
    )

    assert opened["execution_type"] == "open"
    assert opened["commission"] == pytest.approx(2.0)
    assert position.lots == pytest.approx(0.25)

    closed = simulator.execute(
        position=position,
        intent=PositionIntent(
            symbol="EURUSD",
            target_side=0,
            target_lots=0.0,
        ),
        quote=_quote(60, bid=1.1010, ask=1.1012),
        instrument=instrument,
        cost=cost,
    )

    assert closed["execution_type"] == "close"
    assert closed["commission"] == pytest.approx(2.0)
    assert position.lots == pytest.approx(0.0)
    assert opened["commission"] + closed["commission"] == pytest.approx(4.0)


def test_partial_reduction_charges_only_closed_lots():
    simulator = ExecutionSimulator()
    instrument = _instrument()

    position = PositionState(
        symbol="EURUSD",
        side=1,
        lots=0.25,
        entry_price=1.1000,
    )

    cost = ExecutionCost(
        commission_model=CommissionModel(
            basis="per_lot",
            rate=8.0,
            currency="USD",
        ),
        account_currency="USD",
    )

    result = simulator.execute(
        position=position,
        intent=PositionIntent(
            symbol="EURUSD",
            target_side=1,
            target_lots=0.10,
        ),
        quote=_quote(60),
        instrument=instrument,
        cost=cost,
    )

    # Closed volume: 0.25 - 0.10 = 0.15 lots.
    assert result["execution_type"] == "reduce"
    assert result["commission"] == pytest.approx(1.20)
    assert position.lots == pytest.approx(0.10)


def test_percent_notional_commission_uses_fill_price():
    simulator = ExecutionSimulator()
    instrument = _instrument()
    position = PositionState(symbol="EURUSD")

    cost = ExecutionCost(
        commission_model=CommissionModel(
            basis="percent_notional",
            rate=0.001,
            currency="USD",
        ),
        account_currency="USD",
        commission_notional_basis="quote",
    )

    result = simulator.execute(
        position=position,
        intent=PositionIntent(
            symbol="EURUSD",
            target_side=1,
            target_lots=0.10,
        ),
        quote=_quote(),
        instrument=instrument,
        cost=cost,
    )

    # Quote-currency notional = contract_size * lots * fill price.
    expected_notional = 100_000.0 * 0.10 * 1.1002
    assert result["commission"] == pytest.approx(
        expected_notional * 0.001
    )


def test_timestamp_aware_currency_conversion_is_applied():
    simulator = ExecutionSimulator()
    instrument = _instrument()
    position = PositionState(symbol="EURUSD")
    observed_timestamps = []

    def eur_to_usd(timestamp: datetime) -> float:
        observed_timestamps.append(timestamp)
        return 1.10

    cost = ExecutionCost(
        commission_model=CommissionModel(
            basis="per_lot",
            rate=2.0,
            currency="EUR",
        ),
        account_currency="USD",
        commission_to_account_rate=eur_to_usd,
    )

    quote = _quote()
    result = simulator.execute(
        position=position,
        intent=PositionIntent(
            symbol="EURUSD",
            target_side=1,
            target_lots=0.50,
        ),
        quote=quote,
        instrument=instrument,
        cost=cost,
    )

    # 2 EUR/lot * 0.50 lot * 1.10 USD/EUR = 1.10 USD.
    assert result["commission"] == pytest.approx(1.10)
    assert observed_timestamps == [quote.timestamp]


def test_missing_commission_currency_conversion_fails():
    simulator = ExecutionSimulator()
    instrument = _instrument()
    position = PositionState(symbol="EURUSD")

    cost = ExecutionCost(
        commission_model=CommissionModel(
            basis="per_lot",
            rate=2.0,
            currency="EUR",
        ),
        account_currency="USD",
    )

    with pytest.raises(
        ValueError,
        match="commission_to_account_rate.*is required",
    ):
        simulator.execute(
            position=position,
            intent=PositionIntent(
                symbol="EURUSD",
                target_side=1,
                target_lots=0.50,
            ),
            quote=_quote(),
            instrument=instrument,
            cost=cost,
        )


def test_adapter_normalizes_round_turn_rate_exactly_once():
    catalog = _round_turn_catalog()

    costs = build_execution_costs(
        profiles=catalog,
        symbols=["EURUSD"],
        account_currency="USD",
        broker="broker_a",
        account_type="raw",
    )

    cost = costs["EURUSD"]

    assert cost.commission_model is not None
    assert cost.commission_model.rate == pytest.approx(4.0)
    assert cost.commission_billing_basis == "per_side"

    simulator = ExecutionSimulator()
    position = PositionState(symbol="EURUSD")

    opened = simulator.execute(
        position=position,
        intent=PositionIntent(
            symbol="EURUSD",
            target_side=1,
            target_lots=0.25,
        ),
        quote=_quote(),
        instrument=_instrument(),
        cost=cost,
    )

    closed = simulator.execute(
        position=position,
        intent=PositionIntent(
            symbol="EURUSD",
            target_side=0,
            target_lots=0.0,
        ),
        quote=_quote(60),
        instrument=_instrument(),
        cost=cost,
    )

    # 8 USD/lot round-turn * 0.25 lot = 2 USD total:
    # 1 USD opening + 1 USD closing.
    assert opened["commission"] == pytest.approx(1.0)
    assert closed["commission"] == pytest.approx(1.0)
    assert opened["commission"] + closed["commission"] == pytest.approx(2.0)


def test_legacy_fixed_commission_remains_supported():
    simulator = ExecutionSimulator()
    position = PositionState(symbol="EURUSD")

    result = simulator.execute(
        position=position,
        intent=PositionIntent(
            symbol="EURUSD",
            target_side=1,
            target_lots=0.01,
        ),
        quote=_quote(),
        instrument=_instrument(),
        cost=ExecutionCost(commission=1.5),
    )

    assert result["commission"] == pytest.approx(1.5)

