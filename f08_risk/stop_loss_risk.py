# f08_risk/stop_loss_risk.py (32)
# =============================================================================
# Bot-RL-3 v8
# Chapter 4 - Risk Layer
#
# Stop-Loss / Risk-per-Trade mathematical contract.
#
# Responsibilities:
#   - validate entry/stop prices
#   - calculate stop distance
#   - calculate monetary risk budget from equity
#
# Non-responsibilities:
#   - broker communication
#   - broker symbol specifications
#   - lot normalization
#   - order placement
#   - portfolio constraints
#   - execution
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


# =============================================================================
# Request
# =============================================================================

@dataclass(frozen=True, slots=True)
class StopLossRiskRequest:
    """
    Input contract for Stop-Loss / Risk-per-Trade calculation.

    risk_per_trade:
        Fraction of current equity that may be lost if the stop is reached.
        Example:
            0.01 -> 1% of equity
            0.005 -> 0.5% of equity
    """

    symbol: str
    equity: float
    risk_per_trade: float
    entry_price: float
    stop_price: float

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")

        if not isfinite(self.equity):
            raise ValueError("equity must be finite")

        if self.equity < 0.0:
            raise ValueError("equity must be >= 0")

        if not isfinite(self.risk_per_trade):
            raise ValueError("risk_per_trade must be finite")

        if self.risk_per_trade < 0.0:
            raise ValueError("risk_per_trade must be >= 0")

        if self.risk_per_trade > 1.0:
            raise ValueError("risk_per_trade must be <= 1")

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


# =============================================================================
# Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class StopLossRiskResult:
    symbol: str
    equity: float
    risk_per_trade: float
    risk_budget: float
    entry_price: float
    stop_price: float
    stop_distance: float

    def __post_init__(self) -> None:
        values = (
            self.equity,
            self.risk_per_trade,
            self.risk_budget,
            self.entry_price,
            self.stop_price,
            self.stop_distance,
        )

        if not all(isfinite(value) for value in values):
            raise ValueError(
                "StopLossRiskResult contains non-finite values"
            )

        if self.equity < 0.0:
            raise ValueError("equity must be >= 0")

        if self.risk_per_trade < 0.0:
            raise ValueError("risk_per_trade must be >= 0")

        if self.risk_per_trade > 1.0:
            raise ValueError("risk_per_trade must be <= 1")

        if self.risk_budget < 0.0:
            raise ValueError("risk_budget must be >= 0")

        if self.entry_price <= 0.0:
            raise ValueError("entry_price must be > 0")

        if self.stop_price <= 0.0:
            raise ValueError("stop_price must be > 0")

        if self.stop_distance <= 0.0:
            raise ValueError("stop_distance must be > 0")


# =============================================================================
# Calculator
# =============================================================================

class StopLossRiskCalculator:
    """
    Pure deterministic calculator for Stop-Loss / Risk-per-Trade.

    Formula:

        risk_budget = equity * risk_per_trade

        stop_distance = abs(entry_price - stop_price)

    No direction is encoded here.
    """

    def calculate(
        self,
        request: StopLossRiskRequest,
    ) -> StopLossRiskResult:

        if not isinstance(request, StopLossRiskRequest):
            raise TypeError(
                "Expected StopLossRiskRequest, "
                f"got {type(request).__name__}"
            )

        risk_budget = (
            request.equity
            * request.risk_per_trade
        )

        stop_distance = abs(
            request.entry_price
            - request.stop_price
        )

        return StopLossRiskResult(
            symbol=request.symbol,
            equity=request.equity,
            risk_per_trade=request.risk_per_trade,
            risk_budget=risk_budget,
            entry_price=request.entry_price,
            stop_price=request.stop_price,
            stop_distance=stop_distance,
        )

