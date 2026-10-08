from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from f06_env.contracts import PositionIntent
from f06_env.portfolio_state import PositionState
from f05_transact_costs.contracts import InstrumentSpec, MarketQuote


class ExecutionType(str, Enum):
    OPEN = "open"
    INCREASE = "increase"
    REDUCE = "reduce"
    REVERSE = "reverse"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class ExecutionCost:
    """Already-resolved simulation cost input. Spread is never duplicated here."""

    slippage_price: float = 0.0
    commission: float = 0.0

    def __post_init__(self) -> None:
        slippage = float(self.slippage_price)
        commission = float(self.commission)
        if not np.isfinite(slippage) or slippage < 0.0:
            raise ValueError("slippage_price must be finite and >= 0")
        if not np.isfinite(commission) or commission < 0.0:
            raise ValueError("commission must be finite and >= 0")
        object.__setattr__(self, "slippage_price", slippage)
        object.__setattr__(self, "commission", commission)


class ExecutionSimulator:
    """Candle-time simulation state transition using executable bid/ask quotes."""

    def __init__(self, *, default_cost: ExecutionCost | None = None) -> None:
        self.default_cost = default_cost or ExecutionCost()

    @staticmethod
    def _classify(old_side: int, old_lots: float, new_side: int, new_lots: float) -> ExecutionType | None:
        if old_side == 0 and new_side != 0:
            return ExecutionType.OPEN
        if old_side != 0 and new_side == 0:
            return ExecutionType.CLOSE
        if old_side == new_side:
            if new_lots > old_lots:
                return ExecutionType.INCREASE
            if new_lots < old_lots:
                return ExecutionType.REDUCE
            return None
        return ExecutionType.REVERSE

    @staticmethod
    def _fill_price(quote: MarketQuote, side: int, slippage_price: float) -> float:
        base = quote.ask if side > 0 else quote.bid
        price = base + slippage_price if side > 0 else base - slippage_price
        if not np.isfinite(price) or price <= 0.0:
            raise ValueError("computed fill price must be positive and finite")
        return float(price)

    @staticmethod
    def _pnl(
        *,
        entry_price: float,
        exit_price: float,
        side: int,
        lots: float,
        instrument: InstrumentSpec,
    ) -> float:
        return float(
            (exit_price - entry_price)
            * side
            * lots
            * instrument.value_per_price_unit_per_lot
        )

    def execute(
        self,
        *,
        position: PositionState,
        intent: PositionIntent,
        quote: MarketQuote,
        instrument: InstrumentSpec,
        cost: ExecutionCost | None = None,
    ) -> dict[str, Any]:
        if not isinstance(position, PositionState):
            raise TypeError("position must be PositionState")
        if not isinstance(intent, PositionIntent):
            raise TypeError("intent must be PositionIntent")
        if not isinstance(quote, MarketQuote):
            raise TypeError("quote must be MarketQuote")
        if not isinstance(instrument, InstrumentSpec):
            raise TypeError("instrument must be InstrumentSpec")
        if position.symbol != intent.symbol or quote.symbol != intent.symbol or instrument.symbol != intent.symbol:
            raise ValueError("position, intent, quote and instrument symbols must match")
        cost = cost or self.default_cost

        old_side = int(position.side)
        old_lots = float(position.lots)
        old_entry = position.entry_price
        new_side = int(intent.target_side)
        new_lots = float(intent.target_lots)
        transition = self._classify(old_side, old_lots, new_side, new_lots)

        if transition is None:
            position.stop_price = intent.stop_price
            return {
                "changed": False,
                "execution_type": "hold",
                "realized_pnl": 0.0,
                "commission": 0.0,
                "slippage": 0.0,
                "fill_price": None,
            }

        realized_pnl = 0.0
        total_commission = 0.0
        total_slippage = 0.0
        fill_price: float | None = None

        if transition is ExecutionType.OPEN:
            fill_price = self._fill_price(quote, new_side, cost.slippage_price)
            position.side = new_side
            position.lots = new_lots
            position.entry_price = fill_price
            total_commission = cost.commission
            total_slippage = cost.slippage_price

        elif transition is ExecutionType.CLOSE:
            if old_entry is None and old_lots > 0.0:
                raise RuntimeError("open position has no entry_price")
            fill_price = self._fill_price(quote, -old_side, cost.slippage_price)
            if old_lots > 0.0 and old_entry is not None:
                realized_pnl = self._pnl(
                    entry_price=float(old_entry),
                    exit_price=fill_price,
                    side=old_side,
                    lots=old_lots,
                    instrument=instrument,
                )
            position.side = 0
            position.lots = 0.0
            position.entry_price = None
            position.stop_price = None
            total_commission = cost.commission
            total_slippage = cost.slippage_price

        elif transition is ExecutionType.INCREASE:
            if old_entry is None:
                raise RuntimeError("increase requires an existing entry_price")
            added_lots = new_lots - old_lots
            fill_price = self._fill_price(quote, old_side, cost.slippage_price)
            position.entry_price = (
                float(old_entry) * old_lots + fill_price * added_lots
            ) / new_lots
            position.lots = new_lots
            total_commission = cost.commission
            total_slippage = cost.slippage_price

        elif transition is ExecutionType.REDUCE:
            if old_entry is None:
                raise RuntimeError("reduce requires an existing entry_price")
            closed_lots = old_lots - new_lots
            fill_price = self._fill_price(quote, -old_side, cost.slippage_price)
            realized_pnl = self._pnl(
                entry_price=float(old_entry),
                exit_price=fill_price,
                side=old_side,
                lots=closed_lots,
                instrument=instrument,
            )
            position.lots = new_lots
            total_commission = cost.commission
            total_slippage = cost.slippage_price

        elif transition is ExecutionType.REVERSE:
            if old_entry is None and old_lots > 0.0:
                raise RuntimeError("reverse requires an existing entry_price")
            close_fill = self._fill_price(quote, -old_side, cost.slippage_price)
            if old_lots > 0.0 and old_entry is not None:
                realized_pnl = self._pnl(
                    entry_price=float(old_entry),
                    exit_price=close_fill,
                    side=old_side,
                    lots=old_lots,
                    instrument=instrument,
                )
            open_fill = self._fill_price(quote, new_side, cost.slippage_price)
            fill_price = open_fill
            position.side = new_side
            position.lots = new_lots
            position.entry_price = open_fill
            total_commission = 2.0 * cost.commission
            total_slippage = 2.0 * cost.slippage_price

        position.realized_pnl += realized_pnl - total_commission
        position.stop_price = intent.stop_price

        return {
            "changed": True,
            "execution_type": transition.value,
            "realized_pnl": float(realized_pnl),
            "commission": float(total_commission),
            "slippage": float(total_slippage),
            "fill_price": fill_price,
            "quote_timestamp": quote.timestamp,
        }

    def mark_to_market(
        self,
        *,
        positions: dict[str, PositionState],
        quotes: dict[str, MarketQuote],
        instruments: dict[str, InstrumentSpec],
    ) -> float:
        total = 0.0
        for symbol, position in positions.items():
            if position.side == 0 or position.lots <= 0.0:
                position.unrealized_pnl = 0.0
                position.current_price = None
                position.notional = 0.0
                position.used_margin = 0.0
                continue
            quote = quotes[symbol]
            instrument = instruments[symbol]
            current = quote.bid if position.side > 0 else quote.ask
            if position.entry_price is None:
                raise RuntimeError(f"open position {symbol} has no entry_price")
            pnl = self._pnl(
                entry_price=float(position.entry_price),
                exit_price=float(current),
                side=position.side,
                lots=position.lots,
                instrument=instrument,
            )
            position.current_price = float(current)
            if instrument.contract_size is not None:
                position.notional = abs(float(quote.mid) * position.lots * instrument.contract_size)
            else:
                position.notional = 0.0
            position.unrealized_pnl = pnl
            total += pnl
        return float(total)
