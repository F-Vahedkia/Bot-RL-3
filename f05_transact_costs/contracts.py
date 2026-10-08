from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any


class FillRole(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


class CommissionBasis(str, Enum):
    FIXED = "fixed"
    PER_LOT = "per_lot"
    PERCENT_NOTIONAL = "percent_notional"


class CostRealization(str, Enum):
    EXPECTED = "expected"
    ACTUAL = "actual"


class CostSource(str, Enum):
    SIMULATION = "simulation"
    BROKER = "broker"


def _finite(value: Any, name: str, *, minimum: float | None = None, strict: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        if strict and result <= minimum:
            raise ValueError(f"{name} must be > {minimum}")
        if not strict and result < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    return result


def _currency(value: Any, name: str) -> str:
    result = str(value).strip().upper()
    if len(result) != 3 or not result.isalpha():
        raise ValueError(f"{name} must be a 3-letter currency code")
    return result


def _symbol(value: Any) -> str:
    result = str(value).strip().upper()
    if not result:
        raise ValueError("symbol is required")
    return result


def _id(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        allowed = ", ".join(repr(v.value) for v in enum_type)
        raise ValueError(f"{name} must be one of: {allowed}") from exc


@dataclass(frozen=True, slots=True)
class MarketQuote:
    """Executable bid/ask quote supplied by an external quote provider."""

    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    source: CostSource | str

    def __post_init__(self) -> None:
        symbol = _symbol(self.symbol)
        bid = _finite(self.bid, "bid", minimum=0.0, strict=True)
        ask = _finite(self.ask, "ask", minimum=0.0, strict=True)
        if bid > ask:
            raise ValueError("bid must be <= ask")
        timestamp = _timestamp(self.timestamp, "timestamp")
        source = _enum(self.source, CostSource, "source")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "source", source)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_price(self) -> float:
        return self.ask - self.bid

    @property
    def half_spread_price(self) -> float:
        return self.spread_price / 2.0

    def executable_price(self, side: int) -> float:
        if side == 1:
            return self.ask
        if side == -1:
            return self.bid
        raise ValueError("side must be -1 or 1")


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Normalized instrument data resolved from the project's symbol-spec source."""

    symbol: str
    pip_size: float
    tick_size: float
    tick_value: float
    tick_value_currency: str
    contract_size: float | None = None
    volume_min: float | None = None
    volume_step: float | None = None
    volume_max: float | None = None

    def __post_init__(self) -> None:
        symbol = _symbol(self.symbol)
        pip_size = _finite(self.pip_size, "pip_size", minimum=0.0, strict=True)
        tick_size = _finite(self.tick_size, "tick_size", minimum=0.0, strict=True)
        tick_value = _finite(self.tick_value, "tick_value", minimum=0.0, strict=True)
        currency = _currency(self.tick_value_currency, "tick_value_currency")

        normalized: dict[str, float] = {}
        for name, value in (
            ("contract_size", self.contract_size),
            ("volume_min", self.volume_min),
            ("volume_step", self.volume_step),
            ("volume_max", self.volume_max),
        ):
            if value is not None:
                normalized[name] = _finite(value, name, minimum=0.0, strict=True)

        if "volume_min" in normalized and "volume_max" in normalized:
            if normalized["volume_min"] > normalized["volume_max"]:
                raise ValueError("volume_min must be <= volume_max")
        if "volume_step" in normalized and "volume_max" in normalized:
            if normalized["volume_step"] > normalized["volume_max"]:
                raise ValueError("volume_step must be <= volume_max")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "pip_size", pip_size)
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(self, "tick_value", tick_value)
        object.__setattr__(self, "tick_value_currency", currency)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    @property
    def value_per_price_unit_per_lot(self) -> float:
        return self.tick_value / self.tick_size


@dataclass(frozen=True, slots=True)
class CommissionModel:
    """Commission schedule expressed in its native currency."""

    basis: CommissionBasis | str
    rate: float
    currency: str
    minimum: float = 0.0

    def __post_init__(self) -> None:
        basis = _enum(self.basis, CommissionBasis, "commission.basis")
        rate = _finite(self.rate, "commission.rate", minimum=0.0)
        currency = _currency(self.currency, "commission.currency")
        minimum = _finite(self.minimum, "commission.minimum", minimum=0.0)
        if basis is CommissionBasis.PERCENT_NOTIONAL and rate > 1.0:
            raise ValueError("percent_notional commission rate must be <= 1")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "minimum", minimum)

    def amount(self, *, lots: float, notional: float) -> float:
        lots = _finite(lots, "lots", minimum=0.0, strict=True)
        notional = _finite(notional, "notional", minimum=0.0)
        if self.basis is CommissionBasis.FIXED:
            raw = self.rate
        elif self.basis is CommissionBasis.PER_LOT:
            raw = self.rate * lots
        else:
            if notional <= 0.0:
                raise ValueError("notional must be > 0 for percent_notional commission")
            raw = self.rate * notional
        return max(raw, self.minimum)


@dataclass(frozen=True, slots=True)
class TransactionCostRequest:
    """Resolved inputs needed to estimate the direct cost of one fill."""

    symbol: str
    side: int
    role: FillRole | str
    lots: float
    quote: MarketQuote
    instrument: InstrumentSpec
    tick_value_to_account_rate: float
    commission_model: CommissionModel
    account_currency: str
    commission_to_account_rate: float
    commission_notional: float
    additional_fee: float
    additional_fee_currency: str
    additional_fee_to_account_rate: float
    slippage_price: float

    def __post_init__(self) -> None:
        symbol = _symbol(self.symbol)
        try:
            side = int(self.side)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("side must be -1 or 1") from exc
        if side not in (-1, 1):
            raise ValueError("side must be -1 or 1")
        role = _enum(self.role, FillRole, "role")
        lots = _finite(self.lots, "lots", minimum=0.0, strict=True)
        if not isinstance(self.quote, MarketQuote):
            raise TypeError("quote must be MarketQuote")
        if self.quote.symbol != symbol:
            raise ValueError("quote symbol does not match request symbol")
        if not isinstance(self.instrument, InstrumentSpec):
            raise TypeError("instrument must be InstrumentSpec")
        if self.instrument.symbol != symbol:
            raise ValueError("instrument symbol does not match request symbol")
        tick_rate = _finite(self.tick_value_to_account_rate, "tick_value_to_account_rate", minimum=0.0, strict=True)
        commission_rate = _finite(self.commission_to_account_rate, "commission_to_account_rate", minimum=0.0, strict=True)
        notional = _finite(self.commission_notional, "commission_notional", minimum=0.0)
        fee = _finite(self.additional_fee, "additional_fee", minimum=0.0)
        fee_rate = _finite(self.additional_fee_to_account_rate, "additional_fee_to_account_rate", minimum=0.0, strict=True)
        slippage = _finite(self.slippage_price, "slippage_price", minimum=0.0)
        if not isinstance(self.commission_model, CommissionModel):
            raise TypeError("commission_model must be CommissionModel")
        if self.commission_model.basis is CommissionBasis.PERCENT_NOTIONAL and notional <= 0.0:
            raise ValueError("commission_notional must be > 0 for percent_notional commission")
        account_currency = _currency(self.account_currency, "account_currency")
        fee_currency = _currency(self.additional_fee_currency, "additional_fee_currency")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "lots", lots)
        object.__setattr__(self, "tick_value_to_account_rate", tick_rate)
        object.__setattr__(self, "commission_to_account_rate", commission_rate)
        object.__setattr__(self, "commission_notional", notional)
        object.__setattr__(self, "additional_fee", fee)
        object.__setattr__(self, "additional_fee_to_account_rate", fee_rate)
        object.__setattr__(self, "slippage_price", slippage)
        object.__setattr__(self, "account_currency", account_currency)
        object.__setattr__(self, "additional_fee_currency", fee_currency)

    @property
    def market_price(self) -> float:
        return self.quote.executable_price(self.side)

    @property
    def reference_price(self) -> float:
        return self.quote.mid

    @property
    def spread_price(self) -> float:
        return self.quote.half_spread_price

    @property
    def price_value_per_unit(self) -> float:
        return self.instrument.value_per_price_unit_per_lot * self.tick_value_to_account_rate


@dataclass(frozen=True, slots=True)
class TransactionCostBreakdown:
    """Reconciled direct cost components in account currency."""

    realization: CostRealization | str
    spread_price: float
    slippage_price: float
    spread_cost: float
    slippage_cost: float
    commission: float
    additional_fee: float
    price_cost_total: float
    cash_cost_total: float
    total_cost: float
    price_costs_embedded_in_fill: bool
    account_currency: str

    def __post_init__(self) -> None:
        realization = _enum(self.realization, CostRealization, "realization")
        values = {
            name: _finite(getattr(self, name), name, minimum=0.0)
            for name in (
                "spread_price", "slippage_price", "spread_cost", "slippage_cost",
                "commission", "additional_fee", "price_cost_total", "cash_cost_total", "total_cost",
            )
        }
        if not isinstance(self.price_costs_embedded_in_fill, bool):
            raise TypeError("price_costs_embedded_in_fill must be bool")
        account_currency = _currency(self.account_currency, "account_currency")

        expected_price = values["spread_cost"] + values["slippage_cost"]
        if not math.isclose(values["price_cost_total"], expected_price, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("price_cost_total does not reconcile")

        expected_cash = values["commission"] + values["additional_fee"]
        if not self.price_costs_embedded_in_fill:
            expected_cash += expected_price
        if not math.isclose(values["cash_cost_total"], expected_cash, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("cash_cost_total does not reconcile")

        expected_total = expected_price + values["commission"] + values["additional_fee"]
        if not math.isclose(values["total_cost"], expected_total, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("total_cost does not reconcile")

        object.__setattr__(self, "realization", realization)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "account_currency", account_currency)


@dataclass(frozen=True, slots=True)
class ExecutionFill:
    """One atomic fill with sufficient data for accounting and reconciliation."""

    execution_id: str
    fill_id: str
    symbol: str
    side: int
    role: FillRole | str
    lots: float
    bid: float
    ask: float
    reference_price: float
    market_price: float
    fill_price: float
    quote_timestamp: datetime
    execution_timestamp: datetime
    cost: TransactionCostBreakdown
    source: CostSource | str
    broker_order_id: str | None = None
    model_version: str = ""
    slippage_cap_exceeded: bool = False

    def __post_init__(self) -> None:
        execution_id = _id(self.execution_id, "execution_id")
        fill_id = _id(self.fill_id, "fill_id")
        symbol = _symbol(self.symbol)
        side = int(self.side)
        if side not in (-1, 1):
            raise ValueError("side must be -1 or 1")
        role = _enum(self.role, FillRole, "role")
        lots = _finite(self.lots, "lots", minimum=0.0, strict=True)
        bid = _finite(self.bid, "bid", minimum=0.0, strict=True)
        ask = _finite(self.ask, "ask", minimum=0.0, strict=True)
        if bid > ask:
            raise ValueError("bid must be <= ask")
        reference = _finite(self.reference_price, "reference_price", minimum=0.0, strict=True)
        market = _finite(self.market_price, "market_price", minimum=0.0, strict=True)
        fill = _finite(self.fill_price, "fill_price", minimum=0.0, strict=True)
        if not math.isclose(reference, (bid + ask) / 2.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("reference_price must equal quote mid")
        executable = ask if side > 0 else bid
        if not math.isclose(market, executable, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("market_price must equal side executable quote")
        quote_timestamp = _timestamp(self.quote_timestamp, "quote_timestamp")
        execution_timestamp = _timestamp(self.execution_timestamp, "execution_timestamp")
        if execution_timestamp < quote_timestamp:
            raise ValueError("execution_timestamp must be >= quote_timestamp")
        if not isinstance(self.cost, TransactionCostBreakdown):
            raise TypeError("cost must be TransactionCostBreakdown")
        source = _enum(self.source, CostSource, "source")
        expected_realization = CostRealization.EXPECTED if source is CostSource.SIMULATION else CostRealization.ACTUAL
        if self.cost.realization is not expected_realization:
            raise ValueError("fill source and cost realization do not match")

        signed = fill - market if side > 0 else market - fill
        if source is CostSource.SIMULATION and signed < -1e-12:
            raise ValueError("simulated fill cannot improve on executable quote")
        adverse = max(0.0, signed)
        if not math.isclose(adverse, self.cost.slippage_price, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("fill_price does not reconcile with slippage_price")

        model_version = str(self.model_version).strip()
        if not model_version:
            raise ValueError("model_version is required")
        if not isinstance(self.slippage_cap_exceeded, bool):
            raise TypeError("slippage_cap_exceeded must be bool")
        broker_order_id = None if self.broker_order_id is None else str(self.broker_order_id).strip() or None
        if source is CostSource.BROKER and broker_order_id is None:
            raise ValueError("broker_order_id is required for broker fills")

        object.__setattr__(self, "execution_id", execution_id)
        object.__setattr__(self, "fill_id", fill_id)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "lots", lots)
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)
        object.__setattr__(self, "reference_price", reference)
        object.__setattr__(self, "market_price", market)
        object.__setattr__(self, "fill_price", fill)
        object.__setattr__(self, "quote_timestamp", quote_timestamp)
        object.__setattr__(self, "execution_timestamp", execution_timestamp)
        object.__setattr__(self, "cost", self.cost)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "broker_order_id", broker_order_id)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "slippage_cap_exceeded", self.slippage_cap_exceeded)

    @property
    def adverse_slippage_price(self) -> float:
        signed = self.fill_price - self.market_price if self.side > 0 else self.market_price - self.fill_price
        return max(0.0, signed)


@dataclass(frozen=True, slots=True)
class CostReconciliation:
    """Expected-vs-actual reconciliation for one execution id."""

    execution_id: str
    expected_lots: float
    actual_lots: float
    expected_total: float
    actual_total: float
    variance: float
    account_currency: str
    slippage_cap_exceeded: bool = False

    def __post_init__(self) -> None:
        execution_id = _id(self.execution_id, "execution_id")
        expected_lots = _finite(self.expected_lots, "expected_lots", minimum=0.0)
        actual_lots = _finite(self.actual_lots, "actual_lots", minimum=0.0)
        expected_total = _finite(self.expected_total, "expected_total", minimum=0.0)
        actual_total = _finite(self.actual_total, "actual_total", minimum=0.0)
        variance = _finite(self.variance, "variance")
        account_currency = _currency(self.account_currency, "account_currency")
        if not isinstance(self.slippage_cap_exceeded, bool):
            raise TypeError("slippage_cap_exceeded must be bool")
        if not math.isclose(variance, actual_total - expected_total, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("variance does not reconcile")
        object.__setattr__(self, "execution_id", execution_id)
        object.__setattr__(self, "expected_lots", expected_lots)
        object.__setattr__(self, "actual_lots", actual_lots)
        object.__setattr__(self, "expected_total", expected_total)
        object.__setattr__(self, "actual_total", actual_total)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "account_currency", account_currency)
        object.__setattr__(self, "slippage_cap_exceeded", self.slippage_cap_exceeded)
