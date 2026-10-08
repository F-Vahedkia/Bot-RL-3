# f06_env/execution_simulator.py

"""Simulation execution state-transition adapter for Bot-RL-3 V8.

f06_env owns position-state transition semantics and realized-PnL accounting.
f05_transact_costs owns transaction-cost calculation and immutable execution
fill contracts. This module bridges the two layers for simulation only.

Execution flow:
    PositionState + PositionIntent + f07 MarketQuote
        -> f07 SimulationExecutionCostEngine
        -> one or more ExecutionFill objects
        -> f04 position mutation + realized PnL

A REVERSE transition is represented as two atomic fills (EXIT + ENTRY) under
one logical execution_id.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict

import numpy as np

from f06_env.contracts import PositionIntent
from f06_env.portfolio_state import PositionState
from f05_transact_costs import (
    CommissionBasis,
    CommissionModel,
    CostSource,
    ExecutionFill,
    FillRole,
    InstrumentSpec,
    MarketQuote,
    SimulationExecutionCostEngine,
)


class ExecutionType(str, Enum):
    OPEN = "open"
    INCREASE = "increase"
    REDUCE = "reduce"
    REVERSE = "reverse"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class ExecutionCost:
    """Legacy compatibility container; no longer consumed by execution.

    Transaction-cost calculation is exclusively owned by f05_transact_costs.
    Existing imports may continue to resolve this class during the staged
    f04 migration, but passing it to ExecutionSimulator is intentionally no
    longer supported.
    """

    spread: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0

    def total_price_cost(self) -> float:
        return abs(float(self.spread)) + abs(float(self.slippage))


@dataclass(frozen=True, slots=True)
class TransactionCostExecutionContext:
    """Resolved f07 simulation inputs injected by the composition root."""

    cost_engine: SimulationExecutionCostEngine
    instrument: InstrumentSpec
    commission_model: CommissionModel
    tick_value_to_account_rate: float
    account_currency: str
    commission_to_account_rate: float
    additional_fee: float
    additional_fee_currency: str
    additional_fee_to_account_rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.cost_engine, SimulationExecutionCostEngine):
            raise TypeError("cost_engine must be SimulationExecutionCostEngine")
        if not isinstance(self.instrument, InstrumentSpec):
            raise TypeError("instrument must be InstrumentSpec")
        if not isinstance(self.commission_model, CommissionModel):
            raise TypeError("commission_model must be CommissionModel")

        tick_rate = float(self.tick_value_to_account_rate)
        commission_rate = float(self.commission_to_account_rate)
        fee = float(self.additional_fee)
        fee_rate = float(self.additional_fee_to_account_rate)
        if not np.isfinite(tick_rate) or tick_rate <= 0.0:
            raise ValueError("tick_value_to_account_rate must be positive and finite")
        if not np.isfinite(commission_rate) or commission_rate <= 0.0:
            raise ValueError("commission_to_account_rate must be positive and finite")
        if not np.isfinite(fee) or fee < 0.0:
            raise ValueError("additional_fee must be finite and >= 0")
        if not np.isfinite(fee_rate) or fee_rate <= 0.0:
            raise ValueError("additional_fee_to_account_rate must be positive and finite")

        account_currency = str(self.account_currency).strip().upper()
        fee_currency = str(self.additional_fee_currency).strip().upper()
        if len(account_currency) != 3 or not account_currency.isalpha():
            raise ValueError("account_currency must be a 3-letter currency code")
        if len(fee_currency) != 3 or not fee_currency.isalpha():
            raise ValueError("additional_fee_currency must be a 3-letter currency code")

        object.__setattr__(self, "tick_value_to_account_rate", tick_rate)
        object.__setattr__(self, "account_currency", account_currency)
        object.__setattr__(self, "commission_to_account_rate", commission_rate)
        object.__setattr__(self, "additional_fee", fee)
        object.__setattr__(self, "additional_fee_currency", fee_currency)
        object.__setattr__(self, "additional_fee_to_account_rate", fee_rate)


class ExecutionSimulator:
    """Apply target position intents using f07 simulation execution costs."""

    def __init__(
        self,
        *,
        contract_size: float | None = None,
        point_value: float = 1.0,
        transaction_cost_context: TransactionCostExecutionContext,
    ) -> None:
        if not isinstance(transaction_cost_context, TransactionCostExecutionContext):
            raise TypeError("transaction_cost_context is required")

        if contract_size is None:
            contract_size = transaction_cost_context.instrument.contract_size
            if contract_size is None:
                raise ValueError(
                    "contract_size must be supplied when instrument.contract_size is unavailable"
                )
        contract_size = float(contract_size)
        point_value = float(point_value)
        if not np.isfinite(contract_size) or contract_size <= 0.0:
            raise ValueError("contract_size must be > 0 and finite")
        if not np.isfinite(point_value) or point_value <= 0.0:
            raise ValueError("point_value must be > 0 and finite")

        instrument_contract_size = transaction_cost_context.instrument.contract_size
        if instrument_contract_size is not None and not np.isclose(
            contract_size,
            float(instrument_contract_size),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "contract_size must match transaction_cost_context.instrument.contract_size"
            )

        self.contract_size = contract_size
        self.point_value = point_value
        self.transaction_cost_context = transaction_cost_context

    @staticmethod
    def _side_sign(side: int) -> int:
        if side > 0:
            return 1
        if side < 0:
            return -1
        return 0

    @staticmethod
    def _classify_transition(
        *,
        old_side: int,
        old_lots: float,
        new_side: int,
        new_lots: float,
    ) -> ExecutionType:
        if old_side == 0 and new_side != 0:
            return ExecutionType.OPEN
        if old_side != 0 and new_side == 0:
            return ExecutionType.CLOSE
        if old_side == new_side:
            if new_lots > old_lots:
                return ExecutionType.INCREASE
            if new_lots < old_lots:
                return ExecutionType.REDUCE
            return ExecutionType.REDUCE
        if old_side != new_side:
            return ExecutionType.REVERSE
        raise RuntimeError("Invalid execution transition")

    def _commission_notional(
        self,
        *,
        side: int,
        lots: float,
        quote: MarketQuote,
    ) -> float:
        if self.transaction_cost_context.commission_model.basis is not CommissionBasis.PERCENT_NOTIONAL:
            return 0.0
        contract_size = self.transaction_cost_context.instrument.contract_size
        if contract_size is None:
            raise ValueError(
                "instrument.contract_size is required for percent_notional commission"
            )
        return quote.executable_price(side) * float(contract_size) * float(lots)

    def _fill(
        self,
        *,
        execution_id: str,
        fill_id: str,
        symbol: str,
        side: int,
        role: FillRole,
        lots: float,
        quote: MarketQuote,
        event_key: str,
        execution_timestamp: datetime,
    ) -> ExecutionFill:
        context = self.transaction_cost_context
        return context.cost_engine.execute(
            execution_id=execution_id,
            fill_id=fill_id,
            symbol=symbol,
            side=side,
            role=role,
            lots=lots,
            quote=quote,
            instrument=context.instrument,
            tick_value_to_account_rate=context.tick_value_to_account_rate,
            commission_model=context.commission_model,
            commission_notional=self._commission_notional(
                side=side,
                lots=lots,
                quote=quote,
            ),
            commission_to_account_rate=context.commission_to_account_rate,
            account_currency=context.account_currency,
            additional_fee=context.additional_fee,
            additional_fee_currency=context.additional_fee_currency,
            additional_fee_to_account_rate=context.additional_fee_to_account_rate,
            event_key=event_key,
            execution_timestamp=execution_timestamp,
        )

    def execute(
        self,
        position: PositionState,
        intent: PositionIntent,
        *,
        quote: MarketQuote,
        execution_id: str,
        execution_timestamp: datetime,
        event_key: str,
    ) -> dict[str, object]:
        """Apply one target-position intent and account using f07 fills.

        `quote` is the resolved historical bid/ask. `execution_id` identifies
        the logical execution; a reverse may produce EXIT and ENTRY fills that
        share it. `event_key` must be stable so f07 simulation slippage remains
        deterministic.
        """
        if not isinstance(position, PositionState):
            raise TypeError("position must be PositionState")
        if not isinstance(intent, PositionIntent):
            raise TypeError("intent must be PositionIntent")
        if not isinstance(quote, MarketQuote):
            raise TypeError("quote must be f05_transact_costs.MarketQuote")
        if quote.source is not CostSource.SIMULATION:
            raise ValueError("ExecutionSimulator requires a simulation MarketQuote")
        if not isinstance(execution_timestamp, datetime):
            raise TypeError("execution_timestamp must be datetime")
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("execution_id is required")
        if not isinstance(event_key, str) or not event_key.strip():
            raise ValueError("event_key is required")
        if position.symbol.upper() != intent.symbol.upper():
            raise ValueError(
                f"position symbol {position.symbol!r} does not match intent symbol {intent.symbol!r}"
            )
        if quote.symbol != intent.symbol.upper():
            raise ValueError("quote symbol does not match intent symbol")

        old_side = int(position.side)
        old_lots = float(position.lots)
        old_entry = position.entry_price
        new_side = int(intent.target_side)
        new_lots = float(intent.target_lots)
        if new_side != 0 and new_lots <= 0.0:
            raise ValueError("non-flat intent must have target_lots > 0")

        transition = self._classify_transition(
            old_side=old_side,
            old_lots=old_lots,
            new_side=new_side,
            new_lots=new_lots,
        )

        realized_pnl = 0.0
        fills: list[ExecutionFill] = []

        def cash_cost(fill: ExecutionFill) -> float:
            return float(fill.cost.cash_cost_total)

        # OPEN: one ENTRY fill.
        if transition is ExecutionType.OPEN:
            fill = self._fill(
                execution_id=execution_id,
                fill_id=f"{execution_id}:entry",
                symbol=intent.symbol,
                side=new_side,
                role=FillRole.ENTRY,
                lots=new_lots,
                quote=quote,
                event_key=f"{event_key}:entry",
                execution_timestamp=execution_timestamp,
            )
            fills.append(fill)
            realized_pnl -= cash_cost(fill)
            position.side = new_side
            position.lots = new_lots
            position.entry_price = fill.fill_price

        # CLOSE: one EXIT fill.
        elif transition is ExecutionType.CLOSE:
            if old_lots > 0.0 and old_entry is not None:
                fill = self._fill(
                    execution_id=execution_id,
                    fill_id=f"{execution_id}:exit",
                    symbol=intent.symbol,
                    side=-old_side,
                    role=FillRole.EXIT,
                    lots=old_lots,
                    quote=quote,
                    event_key=f"{event_key}:exit",
                    execution_timestamp=execution_timestamp,
                )
                fills.append(fill)
                realized_pnl += (
                    (fill.fill_price - float(old_entry))
                    * old_side
                    * old_lots
                    * self.contract_size
                    * self.point_value
                )
                realized_pnl -= cash_cost(fill)
            position.side = 0
            position.lots = 0.0
            position.entry_price = None

        # INCREASE: one ENTRY fill for the incremental quantity.
        elif transition is ExecutionType.INCREASE:
            added_lots = new_lots - old_lots
            fill = self._fill(
                execution_id=execution_id,
                fill_id=f"{execution_id}:entry",
                symbol=intent.symbol,
                side=old_side,
                role=FillRole.ENTRY,
                lots=added_lots,
                quote=quote,
                event_key=f"{event_key}:entry",
                execution_timestamp=execution_timestamp,
            )
            fills.append(fill)
            realized_pnl -= cash_cost(fill)
            new_entry = (
                fill.fill_price
                if old_entry is None
                else (
                    float(old_entry) * old_lots + fill.fill_price * added_lots
                )
                / new_lots
            )
            position.side = old_side
            position.lots = new_lots
            position.entry_price = new_entry

        # REDUCE: one EXIT fill for the closed quantity.
        elif transition is ExecutionType.REDUCE:
            closed_lots = old_lots - new_lots
            if closed_lots > 0.0 and old_entry is not None:
                fill = self._fill(
                    execution_id=execution_id,
                    fill_id=f"{execution_id}:exit",
                    symbol=intent.symbol,
                    side=-old_side,
                    role=FillRole.EXIT,
                    lots=closed_lots,
                    quote=quote,
                    event_key=f"{event_key}:exit",
                    execution_timestamp=execution_timestamp,
                )
                fills.append(fill)
                realized_pnl += (
                    (fill.fill_price - float(old_entry))
                    * old_side
                    * closed_lots
                    * self.contract_size
                    * self.point_value
                )
                realized_pnl -= cash_cost(fill)
            position.side = old_side
            position.lots = new_lots
            position.entry_price = old_entry

        # REVERSE: two independent fills sharing execution_id.
        else:
            if old_lots > 0.0 and old_entry is not None:
                exit_fill = self._fill(
                    execution_id=execution_id,
                    fill_id=f"{execution_id}:exit",
                    symbol=intent.symbol,
                    side=-old_side,
                    role=FillRole.EXIT,
                    lots=old_lots,
                    quote=quote,
                    event_key=f"{event_key}:exit",
                    execution_timestamp=execution_timestamp,
                )
                fills.append(exit_fill)
                realized_pnl += (
                    (exit_fill.fill_price - float(old_entry))
                    * old_side
                    * old_lots
                    * self.contract_size
                    * self.point_value
                )
                realized_pnl -= cash_cost(exit_fill)

            position.side = new_side
            position.lots = new_lots
            if new_lots > 0.0:
                entry_fill = self._fill(
                    execution_id=execution_id,
                    fill_id=f"{execution_id}:entry",
                    symbol=intent.symbol,
                    side=new_side,
                    role=FillRole.ENTRY,
                    lots=new_lots,
                    quote=quote,
                    event_key=f"{event_key}:entry",
                    execution_timestamp=execution_timestamp,
                )
                fills.append(entry_fill)
                realized_pnl -= cash_cost(entry_fill)
                position.entry_price = entry_fill.fill_price
            else:
                position.entry_price = None

        position.realized_pnl += realized_pnl
        return {
            "realized_pnl": float(realized_pnl),
            "commission": float(sum(fill.cost.commission for fill in fills)),
            "additional_fee": float(sum(fill.cost.additional_fee for fill in fills)),
            "cash_cost_total": float(sum(fill.cost.cash_cost_total for fill in fills)),
            "price_cost_total": float(sum(fill.cost.price_cost_total for fill in fills)),
            "spread": float(sum(fill.cost.spread_price for fill in fills)),
            "slippage": float(sum(fill.cost.slippage_price for fill in fills)),
            "changed": float(bool(fills)),
            "execution_type": transition.value,
            "execution_fills": tuple(fills),
        }

    def mark_to_market(
        self,
        *,
        positions: Dict[str, PositionState],
        prices: Dict[str, float],
    ) -> float:
        """Calculate total unrealized PnL at current prices."""
        total = 0.0
        for symbol, position in positions.items():
            if position.side == 0 or position.lots <= 0.0:
                position.unrealized_pnl = 0.0
                continue
            if position.entry_price is None:
                raise RuntimeError(f"Open position {symbol} has no entry_price")
            if symbol not in prices:
                raise KeyError(f"Missing market price for open symbol {symbol}")
            current = float(prices[symbol])
            if not np.isfinite(current) or current <= 0.0:
                raise ValueError(f"Invalid current market price for {symbol}")
            position.current_price = current
            pnl = (
                (current - float(position.entry_price))
                * self._side_sign(position.side)
                * position.lots
                * self.contract_size
                * self.point_value
            )
            position.unrealized_pnl = float(pnl)
            total += pnl
        return float(total)
