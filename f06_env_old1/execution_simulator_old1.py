# f06_env/execution_simulator.py (3)

"""
این فایل مشخص می‌کند وقتی یک PositionIntent جدید روی یک PositionState فعلی اعمال می‌شود،
پوزیشن فعلی دقیقاً چگونه تغییر کند و چه مقدار realized_pnl ایجاد شود.

این فایل دارای 3 کلاس است:
1) ExecutionType ===> این کلاس نوع تغییر پوزیشن را مشخص می‌کند.

2) ExecutionCost ===> این کلاس هزینه‌های اجرای فرضی را نگه می‌دارد. شامل: spread, slippage, commission

3) ExecutionSimulator ===> این کلاس هسته اصلی این فایل است و انجام ترید را شبیه سازی میکند.
    جریان کلی:

    PositionState فعلی
        +
    PositionIntent جدید
        +
    Market Price
        ↓
    ExecutionSimulator
        ↓
    OPEN / INCREASE / REDUCE / REVERSE / CLOSE
        ↓
    PositionState جدید
        +
    Realized PnL / Costs
"""
# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from enum import Enum

import numpy as np

from f06_env.contracts import PositionIntent
from f06_env.portfolio_state import PositionState

# =============================================================================
# Class-1
# =============================================================================
class ExecutionType(str, Enum):
    OPEN = "open"
    INCREASE = "increase"
    REDUCE = "reduce"
    REVERSE = "reverse"
    CLOSE = "close"

# =============================================================================
# Class-2
# =============================================================================
@dataclass(frozen=True, slots=True)
class ExecutionCost:
    """
    Cost model for simulated execution.

    Values are expressed in price units except commission,
    which is expressed directly in account currency.
    """

    spread: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0

    def total_price_cost(self) -> float:
        return abs(float(self.spread)) + abs(float(self.slippage))


# =============================================================================
# Class-3
# =============================================================================
class ExecutionSimulator:
    """
    Deterministic execution simulator.

    Responsibilities:
        - apply target position intents
        - calculate synthetic fill price
        - calculate realized PnL
        - calculate execution costs

    Non-responsibilities:
        - risk decisions
        - position sizing policy
        - reward calculation
        - broker communication
    """

    def __init__(
        self,
        *,
        contract_size: float = 1.0,
        point_value: float = 1.0,
    ) -> None:
        if contract_size <= 0.0:
            raise ValueError("contract_size must be > 0")

        if point_value <= 0.0:
            raise ValueError("point_value must be > 0")

        self.contract_size = float(contract_size)
        self.point_value = float(point_value)


    @staticmethod
    def _side_sign(side: int) -> int:
        if side > 0:
            return 1
        if side < 0:
            return -1
        return 0


    @staticmethod
    def _classify_transition(
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


    def execute(
        self,
        position: PositionState,
        intent: PositionIntent,
        market_price: float,
        cost: ExecutionCost | None = None,
    ) -> dict[str, float | str]:
        """
        Apply a target-position intent to the current position.

        Transition semantics
        --------------------
        OPEN:
            flat -> non-flat

        INCREASE:
            same side, larger size
            realized PnL = 0
            entry price = weighted average

        REDUCE:
            same side, smaller size
            realized PnL = PnL of closed quantity only
            entry price of remaining quantity is preserved

        REVERSE:
            opposite side
            entire old position is closed and realized
            new position is opened at current market price

        CLOSE:
            non-flat -> flat
            entire position is realized

        Notes
        -----
        spread and slippage are reported through the result dictionary but are
        not applied to fill price/PnL here. That remains outside this accounting
        correction.
        """
        if not isinstance(position, PositionState):
            raise TypeError("position must be PositionState")

        if not isinstance(intent, PositionIntent):
            raise TypeError("intent must be PositionIntent")

        if position.symbol.upper() != intent.symbol.upper():
            raise ValueError(
                f"position symbol {position.symbol!r} does not match "
                f"intent symbol {intent.symbol!r}"
            )

        price = float(market_price)
        if not np.isfinite(price) or price <= 0.0:
            raise ValueError("market_price must be a positive finite value")

        if cost is None:
            cost = ExecutionCost()

        price_cost = float(cost.total_price_cost())
        long_entry_price = price + price_cost
        short_entry_price = price - price_cost
        long_exit_price = price - price_cost
        short_exit_price = price + price_cost

        if not isinstance(cost, ExecutionCost):
            raise TypeError("cost must be ExecutionCost")

        old_side = int(position.side)
        old_lots = float(position.lots)
        old_entry = position.entry_price

        new_side = int(intent.target_side)
        new_lots = float(intent.target_lots)

        execution_type = self._classify_transition(
            old_side=old_side,
            old_lots=old_lots,
            new_side=new_side,
            new_lots=new_lots,
        )

        realized_pnl = 0.0
        changed = False

        # ------------------------------------------------------------------
        # OPEN
        # ------------------------------------------------------------------
        if old_side == 0 and new_side != 0:
            position.side = new_side
            position.lots = new_lots
            # position.entry_price = price
            position.entry_price = (
                long_entry_price if new_side > 0 else short_entry_price
            )
            changed = True

        # ------------------------------------------------------------------
        # CLOSE
        # ------------------------------------------------------------------
        elif old_side != 0 and new_side == 0:
            if old_lots > 0.0 and old_entry is not None:
                exit_price = (
                    long_exit_price if old_side > 0 else short_exit_price
                )

                realized_pnl = (
                    (exit_price - float(old_entry))
                    * old_side
                    * old_lots
                    * self.contract_size
                    * self.point_value
                )

            position.side = 0
            position.lots = 0.0
            position.entry_price = None
            changed = True

        # ------------------------------------------------------------------
        # SAME SIDE
        # ------------------------------------------------------------------
        elif old_side != 0 and new_side == old_side:

            # --------------------------------------------------------------
            # INCREASE
            # --------------------------------------------------------------
            if new_lots > old_lots:
                added_lots = new_lots - old_lots

                if old_entry is None:
                    new_entry = price
                else:
                    new_entry = (
                        float(old_entry) * old_lots
                        + 
                        (
                            long_entry_price if old_side > 0 else short_entry_price
                        ) * added_lots
                    ) / new_lots

                position.side = old_side
                position.lots = new_lots
                position.entry_price = new_entry
                changed = True

            # --------------------------------------------------------------
            # REDUCE
            # --------------------------------------------------------------
            elif new_lots < old_lots:
                closed_lots = old_lots - new_lots

                if old_entry is not None and closed_lots > 0.0:
                    realized_pnl = (
                        (
                            (long_exit_price if old_side > 0 else short_exit_price)
                            - float(old_entry)
                        )
                        * old_side
                        * closed_lots
                        * self.contract_size
                        * self.point_value
                    )

                position.side = old_side
                position.lots = new_lots
                position.entry_price = old_entry
                changed = True

            # --------------------------------------------------------------
            # SAME TARGET SIZE -> no accounting change
            # --------------------------------------------------------------
            else:
                position.side = old_side
                position.lots = old_lots
                position.entry_price = old_entry

        # ------------------------------------------------------------------
        # REVERSE
        # ------------------------------------------------------------------
        else:
            if old_lots > 0.0 and old_entry is not None:
                exit_price = (
                    long_exit_price if old_side > 0 else short_exit_price
                )

                realized_pnl = (
                    (exit_price - float(old_entry))
                    * old_side
                    * old_lots
                    * self.contract_size
                    * self.point_value
                )

            position.side = new_side
            position.lots = new_lots

            if new_lots > 0.0:
                position.entry_price = (
                    long_entry_price if new_side > 0 else short_entry_price
                )
            else:
                position.entry_price = None

            changed = True

        # ------------------------------------------------------------------
        # Commission
        # ------------------------------------------------------------------
        commission = float(cost.commission) if changed else 0.0

        position.realized_pnl += realized_pnl - commission

        return {
            "realized_pnl": float(realized_pnl),
            "commission": float(commission),
            "spread": float(cost.spread),
            "slippage": float(cost.slippage),
            "changed": float(changed),
            "execution_type": execution_type.value,
        }


    def mark_to_market(
        self,
        *,
        positions: Dict[str, PositionState],
        prices: Dict[str, float],
    ) -> float:
        """
        Calculate total unrealized PnL at current prices.
        """
        total = 0.0

        for symbol, position in positions.items():
            if position.side == 0 or position.lots <= 0.0:
                position.unrealized_pnl = 0.0
                continue

            if position.entry_price is None:
                raise RuntimeError(
                    f"Open position {symbol} has no entry_price"
                )

            if symbol not in prices:
                raise KeyError(
                    f"Missing market price for open symbol {symbol}"
                )

            current = float(prices[symbol])
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


# ============================================================================= END
