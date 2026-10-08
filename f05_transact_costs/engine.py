from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .calculator import TransactionCostCalculator
from .contracts import CommissionModel, CostRealization, CostSource, ExecutionFill, FillRole, InstrumentSpec, MarketQuote, TransactionCostRequest
from .models import SlippageModel


@dataclass(frozen=True, slots=True)
class SimulationExecutionCostEngine:
    calculator: TransactionCostCalculator
    slippage_model: SlippageModel

    def execute(
        self,
        *,
        execution_id: str,
        fill_id: str,
        symbol: str,
        side: int,
        role: FillRole | str,
        lots: float,
        quote: MarketQuote,
        instrument: InstrumentSpec,
        tick_value_to_account_rate: float,
        commission_model: CommissionModel,
        commission_notional: float,
        commission_to_account_rate: float,
        account_currency: str,
        additional_fee: float,
        additional_fee_currency: str,
        additional_fee_to_account_rate: float,
        event_key: str,
        execution_timestamp: datetime,
    ) -> ExecutionFill:
        if quote.source is not CostSource.SIMULATION:
            raise ValueError("simulation engine requires simulation quote source")
        slippage = self.slippage_model.sample_price(pip_size=instrument.pip_size, event_key=event_key)
        request = TransactionCostRequest(
            symbol=symbol,
            side=side,
            role=role,
            lots=lots,
            quote=quote,
            instrument=instrument,
            tick_value_to_account_rate=tick_value_to_account_rate,
            commission_model=commission_model,
            account_currency=account_currency,
            commission_to_account_rate=commission_to_account_rate,
            commission_notional=commission_notional,
            additional_fee=additional_fee,
            additional_fee_currency=additional_fee_currency,
            additional_fee_to_account_rate=additional_fee_to_account_rate,
            slippage_price=slippage,
        )
        cost = self.calculator.calculate(
            request,
            realization=CostRealization.EXPECTED,
            price_costs_embedded_in_fill=True,
        )
        market_price = request.market_price
        fill_price = market_price + slippage if side > 0 else market_price - slippage
        return ExecutionFill(
            execution_id=execution_id,
            fill_id=fill_id,
            symbol=symbol,
            side=side,
            role=role,
            lots=lots,
            bid=quote.bid,
            ask=quote.ask,
            reference_price=request.reference_price,
            market_price=market_price,
            fill_price=fill_price,
            quote_timestamp=quote.timestamp,
            execution_timestamp=execution_timestamp,
            cost=cost,
            source=CostSource.SIMULATION,
            model_version=self.slippage_model.model_version,
        )


@dataclass(frozen=True, slots=True)
class LiveObservedExecutionCostEngine:
    calculator: TransactionCostCalculator
    slippage_cap_pips: float
    cap_policy: str = "tolerance"
    on_exceed: str = "flag_and_reconcile"
    model_version: str = "broker_observed.v1"

    def __post_init__(self) -> None:
        import math

        cap = float(self.slippage_cap_pips)
        if not math.isfinite(cap) or cap < 0.0:
            raise ValueError("slippage_cap_pips must be finite and >= 0")
        cap_policy = str(self.cap_policy).strip().lower()
        if cap_policy != "tolerance":
            raise ValueError("cap_policy must be tolerance")
        on_exceed = str(self.on_exceed).strip().lower()
        if on_exceed not in {"flag_and_reconcile", "reject"}:
            raise ValueError("on_exceed must be flag_and_reconcile or reject")
        model_version = str(self.model_version).strip()
        if not model_version:
            raise ValueError("model_version is required")
        object.__setattr__(self, "slippage_cap_pips", cap)
        object.__setattr__(self, "cap_policy", cap_policy)
        object.__setattr__(self, "on_exceed", on_exceed)
        object.__setattr__(self, "model_version", model_version)

    def observe(
        self,
        *,
        execution_id: str,
        fill_id: str,
        broker_order_id: str,
        symbol: str,
        side: int,
        role: FillRole | str,
        lots: float,
        quote: MarketQuote,
        instrument: InstrumentSpec,
        tick_value_to_account_rate: float,
        fill_price: float,
        observed_commission: float,
        observed_commission_currency: str,
        commission_to_account_rate: float,
        account_currency: str,
        observed_fee: float = 0.0,
        observed_fee_currency: str | None = None,
        observed_fee_to_account_rate: float = 1.0,
        execution_timestamp: datetime | None = None,
    ) -> ExecutionFill:
        import math

        if quote.source is not CostSource.BROKER:
            raise ValueError("live observed engine requires broker quote source")
        if execution_timestamp is None:
            raise TypeError("execution_timestamp is required")
        fill_price = float(fill_price)
        if not math.isfinite(fill_price) or fill_price <= 0.0:
            raise ValueError("fill_price must be positive and finite")

        # The broker fill is authoritative. Never clamp or overwrite it.
        market_price = quote.executable_price(side)
        signed_slippage = fill_price - market_price if side > 0 else market_price - fill_price
        adverse_slippage = max(0.0, signed_slippage)
        cap_price = self.slippage_cap_pips * instrument.pip_size
        cap_exceeded = adverse_slippage > cap_price
        if cap_exceeded and self.on_exceed == "reject":
            raise ValueError("observed adverse slippage exceeds configured cap")

        fee_currency = account_currency if observed_fee == 0.0 and observed_fee_currency is None else observed_fee_currency
        if fee_currency is None:
            raise ValueError("observed_fee_currency is required when observed_fee is non-zero")
        commission_model = CommissionModel("fixed", observed_commission, observed_commission_currency)
        request = TransactionCostRequest(
            symbol=symbol,
            side=side,
            role=role,
            lots=lots,
            quote=quote,
            instrument=instrument,
            tick_value_to_account_rate=tick_value_to_account_rate,
            commission_model=commission_model,
            account_currency=account_currency,
            commission_to_account_rate=commission_to_account_rate,
            commission_notional=0.0,
            additional_fee=observed_fee,
            additional_fee_currency=fee_currency,
            additional_fee_to_account_rate=observed_fee_to_account_rate,
            slippage_price=adverse_slippage,
        )
        cost = self.calculator.calculate(
            request,
            realization=CostRealization.ACTUAL,
            price_costs_embedded_in_fill=True,
        )
        return ExecutionFill(
            execution_id=execution_id,
            fill_id=fill_id,
            broker_order_id=broker_order_id,
            symbol=symbol,
            side=side,
            role=role,
            lots=lots,
            bid=quote.bid,
            ask=quote.ask,
            reference_price=request.reference_price,
            market_price=market_price,
            fill_price=fill_price,
            quote_timestamp=quote.timestamp,
            execution_timestamp=execution_timestamp,
            cost=cost,
            source=CostSource.BROKER,
            model_version=self.model_version,
            slippage_cap_exceeded=cap_exceeded,
        )
