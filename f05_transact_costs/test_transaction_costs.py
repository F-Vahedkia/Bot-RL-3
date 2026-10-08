from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import math

import pytest

from .calculator import TransactionCostCalculator
from .config import TransactionCostConfig
from .contracts import (
    CommissionBasis, CommissionModel, CostSource, ExecutionFill, FillRole,
    InstrumentSpec, MarketQuote, TransactionCostRequest,
)
from .engine import LiveObservedExecutionCostEngine, SimulationExecutionCostEngine
from .ledger import TransactionCostLedger
from .models import FixedSlippageModel, NormalCappedSlippageModel
from .providers import MappingInstrumentResolver
from .round_trip import RoundTripCostEstimator

TS = datetime(2026, 9, 30, tzinfo=timezone.utc)


def raw_config() -> dict:
    return {
        "project": {"base_currency": "USD", "random_seed": 42},
        "transaction_costs": {
            "version": 1,
            "account_currency_source": "project.base_currency",
            "simulation": {
                "spread": {"source": "historical_bid_ask", "require_quote": True},
                "slippage": {"model": "normal_capped", "mean_pips": 2.0, "std_pips": 1.0, "cap_pips": 3.0},
                "commission": {"basis": "per_lot", "rate": 7.0, "currency": "USD", "minimum": 0.0},
                "additional_fee": {"value": 0.0, "currency": "USD"},
            },
            "live": {
                "spread": {"source": "broker"},
                "slippage": {"source": "broker", "cap_pips": 3.0, "cap_policy": "tolerance", "on_exceed": "flag_and_reconcile"},
                "commission": {"source": "broker"},
            },
            "accounting": {"charge_on_entry": True, "charge_on_exit": True, "price_costs_embedded_in_fill": True},
            "deterministic": {"seed_source": "project.random_seed"},
        },
    }


def instrument() -> InstrumentSpec:
    return InstrumentSpec("EURUSD", 0.0001, 0.00001, 1.0, "USD", 100000.0)


def quote(source: CostSource | str = CostSource.SIMULATION) -> MarketQuote:
    return MarketQuote("EURUSD", 1.10000, 1.10010, TS, source)


def commission() -> CommissionModel:
    return CommissionModel(CommissionBasis.PER_LOT, 7.0, "USD")


def req(*, side=1, role=FillRole.ENTRY, lots=1.0, slippage=0.00002) -> TransactionCostRequest:
    return TransactionCostRequest(
        symbol="EURUSD", side=side, role=role, lots=lots,
        quote=quote(), instrument=instrument(), tick_value_to_account_rate=1.0,
        commission_model=commission(), account_currency="USD", commission_to_account_rate=1.0,
        commission_notional=110000.0 * lots, additional_fee=0.0, additional_fee_currency="USD",
        additional_fee_to_account_rate=1.0, slippage_price=slippage,
    )


def test_config_roundtrip() -> None:
    cfg = TransactionCostConfig.from_mapping(raw_config())
    assert cfg.simulation.commission_rate == 7.0
    assert cfg.live.slippage_cap_pips == 3.0
    assert cfg.live.slippage_cap_policy == "tolerance"
    assert cfg.live.slippage_on_exceed == "flag_and_reconcile"
    assert TransactionCostConfig.resolve_account_currency(raw_config()) == "USD"


def test_quote_and_instrument() -> None:
    assert quote().executable_price(1) == 1.10010
    assert quote().executable_price(-1) == 1.10000
    assert math.isclose(instrument().value_per_price_unit_per_lot, 100000.0)


def test_calculator_reconciles_components() -> None:
    result = TransactionCostCalculator().calculate(req(), realization="expected", price_costs_embedded_in_fill=True)
    assert math.isclose(result.spread_cost, 5.0)
    assert math.isclose(result.slippage_cost, 2.0)
    assert math.isclose(result.commission, 7.0)
    assert math.isclose(result.total_cost, 14.0)
    assert math.isclose(result.cash_cost_total, 7.0)


def test_percent_commission() -> None:
    model = CommissionModel("percent_notional", 0.0001, "USD", 2.0)
    assert math.isclose(model.amount(lots=1.0, notional=110000.0), 11.0)
    assert math.isclose(model.amount(lots=1.0, notional=1000.0), 2.0)


def test_simulation_engine_is_adverse_only() -> None:
    engine = SimulationExecutionCostEngine(TransactionCostCalculator(), FixedSlippageModel(0.00002))
    fill = engine.execute(
        execution_id="e1", fill_id="f1", symbol="EURUSD", side=1, role=FillRole.ENTRY, lots=1,
        quote=quote(), instrument=instrument(), tick_value_to_account_rate=1.0, commission_model=commission(),
        commission_notional=110000.0, commission_to_account_rate=1.0, account_currency="USD",
        additional_fee=0.0, additional_fee_currency="USD", additional_fee_to_account_rate=1.0,
        event_key="e1", execution_timestamp=TS,
    )
    assert math.isclose(fill.fill_price, 1.10012)
    assert fill.cost.realization.value == "expected"


def test_live_engine_price_improvement_has_zero_adverse_slippage() -> None:
    engine = LiveObservedExecutionCostEngine(TransactionCostCalculator(), 3.0)
    fill = engine.observe(
        execution_id="e2", fill_id="bf1", broker_order_id="bo1", symbol="EURUSD", side=1, role=FillRole.ENTRY,
        lots=1, quote=quote(CostSource.BROKER), instrument=instrument(), tick_value_to_account_rate=1.0,
        fill_price=1.10009, observed_commission=7.0, observed_commission_currency="USD",
        commission_to_account_rate=1.0, account_currency="USD", execution_timestamp=TS,
    )
    assert fill.cost.slippage_price == 0.0
    assert fill.cost.realization.value == "actual"


def test_live_engine_cap_breach_preserves_authoritative_broker_fill() -> None:
    engine = LiveObservedExecutionCostEngine(
        TransactionCostCalculator(),
        3.0,
        cap_policy="tolerance",
        on_exceed="flag_and_reconcile",
    )
    fill = engine.observe(
        execution_id="e3", fill_id="bf3", broker_order_id="bo3", symbol="EURUSD", side=1, role=FillRole.ENTRY,
        lots=1, quote=quote(CostSource.BROKER), instrument=instrument(), tick_value_to_account_rate=1.0,
        fill_price=1.10050, observed_commission=7.0, observed_commission_currency="USD",
        commission_to_account_rate=1.0, account_currency="USD", execution_timestamp=TS,
    )
    assert fill.fill_price == 1.10050
    assert math.isclose(fill.adverse_slippage_price, 0.00040)
    assert fill.cost.realization is not None
    assert fill.slippage_cap_exceeded is True


def test_execution_fill_is_immutable() -> None:
    engine = LiveObservedExecutionCostEngine(TransactionCostCalculator(), 3.0)
    fill = engine.observe(
        execution_id="immut", fill_id="bf-immut", broker_order_id="bo-immut", symbol="EURUSD", side=1, role=FillRole.ENTRY,
        lots=1, quote=quote(CostSource.BROKER), instrument=instrument(), tick_value_to_account_rate=1.0,
        fill_price=1.10050, observed_commission=7.0, observed_commission_currency="USD",
        commission_to_account_rate=1.0, account_currency="USD", execution_timestamp=TS,
    )
    with pytest.raises(FrozenInstanceError):
        fill.fill_price = 1.10040


def test_fill_reconciles() -> None:
    calc = TransactionCostCalculator()
    cost = calc.calculate(req(), realization="expected", price_costs_embedded_in_fill=True)
    fill = ExecutionFill(
        execution_id="e4", fill_id="f4", symbol="EURUSD", side=1, role=FillRole.ENTRY, lots=1,
        bid=1.1, ask=1.1001, reference_price=1.10005, market_price=1.1001, fill_price=1.10012,
        quote_timestamp=TS, execution_timestamp=TS, cost=cost, source=CostSource.SIMULATION, model_version="fixed.v1",
    )
    assert math.isclose(fill.adverse_slippage_price, 0.00002)


def test_round_trip_two_legs() -> None:
    calc = TransactionCostCalculator()
    est = RoundTripCostEstimator(calc).estimate(req(role=FillRole.ENTRY), req(side=-1, role=FillRole.EXIT))
    assert math.isclose(est.total_cost, 28.0)


def test_deterministic_slippage_is_event_stable() -> None:
    a = NormalCappedSlippageModel(2, 1, 3, 42)
    assert a.sample_pips(event_key="x") == a.sample_pips(event_key="x")
    assert a.sample_pips(event_key="x") != a.sample_pips(event_key="y")


def test_mapping_instrument_resolver() -> None:
    resolver = MappingInstrumentResolver({"EURUSD": {"digits": 5, "trade_tick_size": 0.00001, "trade_tick_value": 1.0, "tick_value_currency": "USD"}})
    assert resolver.resolve(symbol="eurusd").symbol == "EURUSD"


def test_ledger_partial_fill_reconciliation() -> None:
    calc = TransactionCostCalculator(); ledger = TransactionCostLedger()
    sim_eng = SimulationExecutionCostEngine(calc, FixedSlippageModel(0.00001))
    live_eng = LiveObservedExecutionCostEngine(
        calc, 3.0, cap_policy="tolerance", on_exceed="flag_and_reconcile"
    )
    expected = sim_eng.execute(
        execution_id="e5", fill_id="exp", symbol="EURUSD", side=1, role=FillRole.ENTRY, lots=1.0,
        quote=quote(), instrument=instrument(), tick_value_to_account_rate=1.0, commission_model=commission(),
        commission_notional=110000, commission_to_account_rate=1, account_currency="USD", additional_fee=0,
        additional_fee_currency="USD", additional_fee_to_account_rate=1, event_key="exp", execution_timestamp=TS,
    )
    actual1 = live_eng.observe(
        execution_id="e5", fill_id="a1", broker_order_id="bo-a1", symbol="EURUSD", side=1, role=FillRole.ENTRY, lots=0.4,
        quote=quote(CostSource.BROKER), instrument=instrument(), tick_value_to_account_rate=1.0,
        fill_price=1.10011, observed_commission=7.0 * 0.4, observed_commission_currency="USD",
        commission_to_account_rate=1, account_currency="USD", execution_timestamp=TS,
    )
    actual2 = live_eng.observe(
        execution_id="e5", fill_id="a2", broker_order_id="bo-a2", symbol="EURUSD", side=1, role=FillRole.ENTRY, lots=0.6,
        quote=quote(CostSource.BROKER), instrument=instrument(), tick_value_to_account_rate=1.0,
        fill_price=1.10011, observed_commission=7.0 * 0.6, observed_commission_currency="USD",
        commission_to_account_rate=1, account_currency="USD", execution_timestamp=TS,
    )
    ledger.record(expected); assert ledger.count() == 1
    recon = TransactionCostLedger.reconcile_execution([expected], [actual1, actual2])
    assert math.isclose(recon.expected_lots, 1.0)
    assert math.isclose(recon.actual_lots, 1.0)
    assert recon.slippage_cap_exceeded is False
    with pytest.raises(ValueError): ledger.record(expected)


def test_ledger_reconciliation_propagates_cap_breach_without_changing_fill() -> None:
    calc = TransactionCostCalculator()
    ledger = TransactionCostLedger()
    sim_eng = SimulationExecutionCostEngine(calc, FixedSlippageModel(0.00001))
    live_eng = LiveObservedExecutionCostEngine(
        calc, 3.0, cap_policy="tolerance", on_exceed="flag_and_reconcile"
    )
    expected = sim_eng.execute(
        execution_id="e-cap", fill_id="exp-cap", symbol="EURUSD", side=1, role=FillRole.ENTRY, lots=1.0,
        quote=quote(), instrument=instrument(), tick_value_to_account_rate=1.0, commission_model=commission(),
        commission_notional=110000, commission_to_account_rate=1, account_currency="USD", additional_fee=0,
        additional_fee_currency="USD", additional_fee_to_account_rate=1, event_key="exp-cap", execution_timestamp=TS,
    )
    actual = live_eng.observe(
        execution_id="e-cap", fill_id="act-cap", broker_order_id="bo-cap", symbol="EURUSD", side=1, role=FillRole.ENTRY, lots=1.0,
        quote=quote(CostSource.BROKER), instrument=instrument(), tick_value_to_account_rate=1.0,
        fill_price=1.10050, observed_commission=7.0, observed_commission_currency="USD",
        commission_to_account_rate=1, account_currency="USD", execution_timestamp=TS,
    )
    ledger.record(expected)
    assert actual.fill_price == 1.10050
    recon = TransactionCostLedger.reconcile_execution([expected], [actual])
    assert recon.slippage_cap_exceeded is True
    assert actual.fill_price == 1.10050


def test_live_engine_reject_policy_still_rejects_cap_breach() -> None:
    engine = LiveObservedExecutionCostEngine(
        TransactionCostCalculator(),
        3.0,
        cap_policy="tolerance",
        on_exceed="reject",
    )
    with pytest.raises(ValueError):
        engine.observe(
            execution_id="e-reject", fill_id="bf-reject", broker_order_id="bo-reject",
            symbol="EURUSD", side=1, role=FillRole.ENTRY, lots=1,
            quote=quote(CostSource.BROKER), instrument=instrument(), tick_value_to_account_rate=1.0,
            fill_price=1.10050, observed_commission=7.0, observed_commission_currency="USD",
            commission_to_account_rate=1.0, account_currency="USD", execution_timestamp=TS,
        )


@pytest.mark.parametrize("field", ["lots", "bid", "ask", "slippage_price"])
def test_nonfinite_request_fields_rejected(field) -> None:
    values = {"lots": float("nan"), "bid": 1.1, "ask": 1.1001, "slippage_price": 0.0}
    values[field] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        req(**values)
