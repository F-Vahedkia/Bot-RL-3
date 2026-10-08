from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    CommissionBasis,
    CostRealization,
    ExecutionFill,
    TransactionCostBreakdown,
    TransactionCostRequest,
)
from datetime import datetime
import math


@dataclass(frozen=True, slots=True)
class TransactionCostCalculator:
    """Pure calculation layer: no file I/O, broker I/O, or hidden defaults."""

    def calculate(
        self,
        request: TransactionCostRequest,
        *,
        realization: CostRealization | str,
        price_costs_embedded_in_fill: bool,
    ) -> TransactionCostBreakdown:
        if not isinstance(request, TransactionCostRequest):
            raise TypeError("request must be TransactionCostRequest")
        if not isinstance(price_costs_embedded_in_fill, bool):
            raise TypeError("price_costs_embedded_in_fill must be bool")

        price_value = request.price_value_per_unit
        spread_cost = request.spread_price * price_value * request.lots
        slippage_cost = request.slippage_price * price_value * request.lots
        commission_native = request.commission_model.amount(
            lots=request.lots,
            notional=request.commission_notional,
        )
        commission = commission_native * request.commission_to_account_rate
        additional_fee = request.additional_fee * request.additional_fee_to_account_rate
        price_cost_total = spread_cost + slippage_cost
        cash_cost_total = commission + additional_fee
        if not price_costs_embedded_in_fill:
            cash_cost_total += price_cost_total
        total_cost = price_cost_total + commission + additional_fee

        return TransactionCostBreakdown(
            realization=realization,
            spread_price=request.spread_price,
            slippage_price=request.slippage_price,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            commission=commission,
            additional_fee=additional_fee,
            price_cost_total=price_cost_total,
            cash_cost_total=cash_cost_total,
            total_cost=total_cost,
            price_costs_embedded_in_fill=price_costs_embedded_in_fill,
            account_currency=request.account_currency,
        )

    def build_fill(
        self,
        request: TransactionCostRequest,
        *,
        timestamp: datetime,
        execution_id: str,
        fill_id: str,
        source: str,
        model_version: str,
        broker_order_id: str | None = None,
    ) -> ExecutionFill:
        cost = self.calculate(
            request,
            realization=CostRealization.EXPECTED if source == "simulation" else CostRealization.ACTUAL,
            price_costs_embedded_in_fill=True,
        )
        market_price = request.market_price
        fill_price = market_price + request.slippage_price if request.side > 0 else market_price - request.slippage_price
        if not math.isfinite(fill_price) or fill_price <= 0.0:
            raise ValueError("fill_price must be positive and finite")
        return ExecutionFill(
            execution_id=execution_id,
            fill_id=fill_id,
            symbol=request.symbol,
            side=request.side,
            role=request.role,
            lots=request.lots,
            bid=request.quote.bid,
            ask=request.quote.ask,
            reference_price=request.reference_price,
            market_price=market_price,
            fill_price=fill_price,
            quote_timestamp=request.quote.timestamp,
            execution_timestamp=timestamp,
            cost=cost,
            source=source,
            broker_order_id=broker_order_id,
            model_version=model_version,
        )
