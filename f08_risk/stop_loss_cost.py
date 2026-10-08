# f08_risk/stop_loss_cost.py (36)
# =============================================================================
# Bot-RL-3 v8
# Chapter 4 - Risk Layer
#
# Monetary loss of one lot when the Stop Loss is reached.
#
# Formula:
#
#     stop_distance = abs(entry_price - stop_price)
#
#     loss_per_lot_at_stop =
#         stop_distance
#         * contract_size
#         * currency_conversion_rate
#
# This module is deterministic and instrument-aware, but broker-independent.
#
# Non-responsibilities:
#   - selecting Stop Loss
#   - selecting trade direction
#   - position sizing
#   - broker execution
#   - lot normalization
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


# =============================================================================
# Request
# =============================================================================

@dataclass(frozen=True, slots=True)
class StopLossCostRequest:
    symbol: str
    entry_price: float
    stop_price: float
    contract_size: float
    currency_conversion_rate: float = 1.0

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")

        if not isfinite(self.entry_price):
            raise ValueError("entry_price must be finite")

        if self.entry_price <= 0.0:
            raise ValueError("entry_price must be > 0")

        if not isfinite(self.stop_price):
            raise ValueError("stop_price must be finite")

        if self.stop_price <= 0.0:
            raise ValueError("stop_price must be > 0")

        if self.entry_price == self.stop_price:
            raise ValueError(
                "entry_price and stop_price must not be equal"
            )

        if not isfinite(self.contract_size):
            raise ValueError(
                "contract_size must be finite"
            )

        if self.contract_size <= 0.0:
            raise ValueError(
                "contract_size must be > 0"
            )

        if not isfinite(self.currency_conversion_rate):
            raise ValueError(
                "currency_conversion_rate must be finite"
            )

        if self.currency_conversion_rate <= 0.0:
            raise ValueError(
                "currency_conversion_rate must be > 0"
            )


# =============================================================================
# Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class StopLossCostResult:
    symbol: str
    entry_price: float
    stop_price: float
    stop_distance: float
    contract_size: float
    currency_conversion_rate: float
    loss_per_lot_at_stop: float

    def __post_init__(self) -> None:
        values = (
            self.entry_price,
            self.stop_price,
            self.stop_distance,
            self.contract_size,
            self.currency_conversion_rate,
            self.loss_per_lot_at_stop,
        )

        if not all(isfinite(value) for value in values):
            raise ValueError(
                "StopLossCostResult contains non-finite values"
            )

        if self.entry_price <= 0.0:
            raise ValueError("entry_price must be > 0")

        if self.stop_price <= 0.0:
            raise ValueError("stop_price must be > 0")

        if self.stop_distance <= 0.0:
            raise ValueError("stop_distance must be > 0")

        if self.contract_size <= 0.0:
            raise ValueError("contract_size must be > 0")

        if self.currency_conversion_rate <= 0.0:
            raise ValueError(
                "currency_conversion_rate must be > 0"
            )

        if self.loss_per_lot_at_stop <= 0.0:
            raise ValueError(
                "loss_per_lot_at_stop must be > 0"
            )


# =============================================================================
# Calculator
# =============================================================================

class StopLossCostCalculator:
    """
    Pure deterministic calculator for monetary loss per one lot
    when the Stop Loss is reached.
    """

    def calculate(
        self,
        request: StopLossCostRequest,
    ) -> StopLossCostResult:

        if not isinstance(
            request,
            StopLossCostRequest,
        ):
            raise TypeError(
                "Expected StopLossCostRequest, "
                f"got {type(request).__name__}"
            )

        stop_distance = abs(
            request.entry_price
            - request.stop_price
        )

        loss_per_lot_at_stop = (
            stop_distance
            * request.contract_size
            * request.currency_conversion_rate
        )

        return StopLossCostResult(
            symbol=request.symbol,
            entry_price=request.entry_price,
            stop_price=request.stop_price,
            stop_distance=stop_distance,
            contract_size=request.contract_size,
            currency_conversion_rate=request.currency_conversion_rate,
            loss_per_lot_at_stop=loss_per_lot_at_stop,
        )

# ============================================================================= END
