
# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from f06_env.contracts import PositionIntent
from f06_env.portfolio_state import PositionState
from f05_transact_costs.contracts import (
    CommissionBasis,
    CommissionModel,
    InstrumentSpec,
    MarketQuote,
)

# =============================================================================
# Functions
# =============================================================================

RateSource = float | Callable[[datetime], float]


def _resolve_rate_source(
    source: RateSource | None,
    *,
    timestamp: datetime,
    name: str,
) -> float:
    """
    این تابع نرخ تبدیل ارز را اعتبارسنجی می‌کند.
    منبع نرخ می‌تواند یک مقدار ثابت یا تابعی باشد که در زمان اجرای معامله آن را دریافت کند.
    برای بک‌تست، تابع باید نرخ تاریخی متناظر با همان زمان را برگرداند؛
    استفاده از نرخ فعلی برای داده‌های تاریخی صحیح نیست.
    """
    if source is None:
        raise ValueError(f"{name} is required when currencies differ")

    raw = source(timestamp) if callable(source) else source

    try:
        rate = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{name} must resolve to a numeric rate"
        ) from exc

    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(
            f"{name} must resolve to a positive finite rate"
        )

    return rate


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
    """ورودی هزینهٔ شبیه‌سازی؛ اسپرد دوباره به این بخش اضافه نمی‌شود."""

    slippage_price: float = 0.0
    commission: float = 0.0

    commission_model: CommissionModel | None = None
    commission_billing_basis: str = "per_side"
    commission_notional_basis: str = "quote"
    account_currency: str | None = None
    commission_to_account_rate: RateSource | None = None
    notional_to_commission_rate: RateSource | None = None

    def __post_init__(self) -> None:
        slippage = float(self.slippage_price)
        commission = float(self.commission)

        if not np.isfinite(slippage) or slippage < 0.0:
            raise ValueError("slippage_price must be finite and >= 0")

        if not np.isfinite(commission) or commission < 0.0:
            raise ValueError("commission must be finite and >= 0")

        if self.commission_model is not None:
            if not isinstance(self.commission_model, CommissionModel):
                raise TypeError(
                    "commission_model must be CommissionModel"
                )

            if commission != 0.0:
                raise ValueError(
                    "set either legacy commission or commission_model, "
                    "not both"
                )

            if self.account_currency is None:
                raise ValueError(
                    "account_currency is required when commission_model is set"
                )

            account_currency = str(self.account_currency).strip().upper()
            if len(account_currency) != 3 or not account_currency.isalpha():
                raise ValueError(
                    "account_currency must be a 3-letter currency code"
                )

            billing_basis = str(
                self.commission_billing_basis
            ).strip().lower()

            if billing_basis not in {"per_side", "round_turn"}:
                raise ValueError(
                    "commission_billing_basis must be per_side or round_turn"
                )

            notional_basis = str(
                self.commission_notional_basis
            ).strip().lower()

            if notional_basis not in {"base", "quote"}:
                raise ValueError(
                    "commission_notional_basis must be base or quote"
                )

            if (
                billing_basis == "round_turn"
                and self.commission_model.minimum != 0.0
            ):
                raise ValueError(
                    "round_turn profiles require minimum=0 until "
                    "round-turn minimum accounting across partial fills "
                    "is implemented"
                )

            object.__setattr__(
                self, "account_currency", account_currency
            )
            object.__setattr__(
                self, "commission_billing_basis", billing_basis
            )
            object.__setattr__(
                self, "commission_notional_basis", notional_basis
            )

        object.__setattr__(self, "slippage_price", slippage)
        object.__setattr__(self, "commission", commission)

# =============================================================================
# Class-3
# =============================================================================

class ExecutionSimulator:
    """Candle-time simulation state transition using executable bid/ask quotes."""

    def __init__(
        self,
        *,
        default_cost: ExecutionCost | None = None,
        charge_on_entry: bool = True,
        charge_on_exit: bool = True,
    ) -> None:
        if not isinstance(charge_on_entry, bool):
            raise TypeError("charge_on_entry must be bool")

        if not isinstance(charge_on_exit, bool):
            raise TypeError("charge_on_exit must be bool")

        self.default_cost = (
            default_cost if default_cost is not None else ExecutionCost()
        )
        self.charge_on_entry = charge_on_entry
        self.charge_on_exit = charge_on_exit


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


    def _commission_for_fill(
        self,
        *,
        cost: ExecutionCost,
        lots: float,
        fill_price: float,
        instrument: InstrumentSpec,
        timestamp: datetime,
    ) -> float:
        """محاسبهٔ کمیسیون یک fill برحسب ارز حساب."""

        model = cost.commission_model

        if model is None:
            # سازگاری با حالت قدیمی:
            # commission مبلغ ثابت هر انتقال است.
            return cost.commission

        account_currency = cost.account_currency
        if account_currency is None:
            raise RuntimeError(
                "account_currency is missing from commission cost"
            )

        notional = 0.0

        if model.basis is CommissionBasis.PERCENT_NOTIONAL:
            if instrument.contract_size is None:
                raise ValueError(
                    "contract_size is required for percent_notional "
                    f"commission: {instrument.symbol}"
                )

            contract_size = float(instrument.contract_size)

            if cost.commission_notional_basis == "base":
                notional_currency = instrument.currency_base
                notional_native = contract_size * lots
            else:
                notional_currency = instrument.currency_profit
                notional_native = (
                    contract_size * lots * fill_price
                )

            if notional_currency is None:
                raise ValueError(
                    "instrument currency is required for "
                    f"notional_basis={cost.commission_notional_basis}: "
                    f"{instrument.symbol}"
                )

            notional_currency = str(notional_currency).upper()

            if notional_currency == model.currency:
                notional_rate = 1.0
            else:
                notional_rate = _resolve_rate_source(
                    cost.notional_to_commission_rate,
                    timestamp=timestamp,
                    name=(
                        f"notional_to_commission_rate[{instrument.symbol}] "
                        f"({notional_currency}->{model.currency})"
                    ),
                )

            notional = notional_native * notional_rate

        commission_native = model.amount(
            lots=lots,
            notional=notional,
        )

        if model.currency == account_currency:
            commission_to_account = 1.0
        else:
            commission_to_account = _resolve_rate_source(
                cost.commission_to_account_rate,
                timestamp=timestamp,
                name=(
                    f"commission_to_account_rate[{instrument.symbol}] "
                    f"({model.currency}->{account_currency})"
                ),
            )

        billing_multiplier = (
            0.5
            if cost.commission_billing_basis == "round_turn"
            else 1.0
        )

        return float(
            commission_native
            * commission_to_account
            * billing_multiplier
        )


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

        # entry_commission = (
        #     cost.commission if self.charge_on_entry else 0.0
        # )
        # exit_commission = (
        #     cost.commission if self.charge_on_exit else 0.0
        # )
        def entry_commission(
            lots: float,
            fill_price: float,
        ) -> float:
            if not self.charge_on_entry:
                return 0.0

            return self._commission_for_fill(
                cost=cost,
                lots=lots,
                fill_price=fill_price,
                instrument=instrument,
                timestamp=quote.timestamp,
            )

        def exit_commission(
            lots: float,
            fill_price: float,
        ) -> float:
            if not self.charge_on_exit:
                return 0.0

            return self._commission_for_fill(
                cost=cost,
                lots=lots,
                fill_price=fill_price,
                instrument=instrument,
                timestamp=quote.timestamp,
            )


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
            total_commission = entry_commission(
                new_lots, float(fill_price)
            )
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
            total_commission = exit_commission(
                old_lots, float(fill_price)
            )
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
            total_commission = entry_commission(
                added_lots, float(fill_price)
            )
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
            total_commission = exit_commission(
                closed_lots, float(fill_price)
            )
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
            total_commission = (
                exit_commission(old_lots, float(close_fill)) +
                entry_commission(new_lots, float(open_fill))
            )
            total_slippage = 2.0 * cost.slippage_price

        accounting_realized_delta = realized_pnl - total_commission
        position.realized_pnl += accounting_realized_delta
        position.stop_price = intent.stop_price

        return {
            "changed": True,
            "execution_type": transition.value,
            "realized_pnl": float(realized_pnl),
            "accounting_realized_delta": float(accounting_realized_delta),
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


# ============================================================================= END
