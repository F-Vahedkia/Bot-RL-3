# f08_risk/stop_loss_position_sizing.py (38)
# =============================================================================
# Bot-RL-3 v8
# Chapter 4 - Risk Layer
#
# Integrated Stop-Loss / Risk-per-Trade position sizing.
#
# Flow:
#
#   equity
#       +
#   risk_per_trade
#       +
#   entry_price
#       +
#   stop_price
#       +
#   contract_size
#       +
#   currency_conversion_rate
#             |
#             v
#       risk_budget
#             |
#             v
#   loss_per_lot_at_stop
#             |
#             v
#       volume_lots
#
# This module composes the mathematical contracts already established in:
#
#   stop_loss_risk.py
#   stop_loss_cost.py
#   risk_based_position_sizing.py
#
# Non-responsibilities:
#   - direction
#   - Stop-Loss selection
#   - broker communication
#   - broker lot normalization
#   - execution
#   - portfolio constraints
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from f08_risk.risk_based_position_sizing import (
    RiskBasedPositionSizingCalculator,
    RiskBasedPositionSizingRequest,
)
from f08_risk.stop_loss_cost import (
    StopLossCostCalculator,
    StopLossCostRequest,
)
from f08_risk.stop_loss_risk import (
    StopLossRiskCalculator,
    StopLossRiskRequest,
)

# =============================================================================
# Request
# =============================================================================

@dataclass(frozen=True, slots=True)
class StopLossPositionSizingRequest:
    symbol: str
    equity: float
    risk_per_trade: float
    entry_price: float
    stop_price: float
    contract_size: float
    currency_conversion_rate: float = 1.0

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

        if not isfinite(self.contract_size):
            raise ValueError("contract_size must be finite")

        if self.contract_size <= 0.0:
            raise ValueError("contract_size must be > 0")

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
class StopLossPositionSizingResult:
    symbol: str
    equity: float
    risk_per_trade: float
    risk_budget: float
    entry_price: float
    stop_price: float
    stop_distance: float
    contract_size: float
    currency_conversion_rate: float
    loss_per_lot_at_stop: float
    volume_lots: float

    def __post_init__(self) -> None:
        values = (
            self.equity,
            self.risk_per_trade,
            self.risk_budget,
            self.entry_price,
            self.stop_price,
            self.stop_distance,
            self.contract_size,
            self.currency_conversion_rate,
            self.loss_per_lot_at_stop,
            self.volume_lots,
        )

        if not all(isfinite(value) for value in values):
            raise ValueError("StopLossPositionSizingResult contains non-finite values")

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

        if self.contract_size <= 0.0:
            raise ValueError("contract_size must be > 0")

        if self.currency_conversion_rate <= 0.0:
            raise ValueError("currency_conversion_rate must be > 0")

        if self.loss_per_lot_at_stop <= 0.0:
            raise ValueError("loss_per_lot_at_stop must be > 0")

        if self.volume_lots < 0.0:
            raise ValueError("volume_lots must be >= 0")


# =============================================================================
# Calculator
# =============================================================================

class StopLossPositionSizingCalculator:
    """
    Integrated deterministic Stop-Loss / Risk-per-Trade calculator.

    No direction is encoded.

    Formula:

        risk_budget = equity * risk_per_trade

        stop_distance = abs(entry_price - stop_price)

        loss_per_lot_at_stop =
            stop_distance
            * contract_size
            * currency_conversion_rate

        volume_lots =
            risk_budget / loss_per_lot_at_stop
    """

    def __init__(
        self,
        *,
        risk_calculator: StopLossRiskCalculator | None = None,
        cost_calculator: StopLossCostCalculator | None = None,
        sizing_calculator: RiskBasedPositionSizingCalculator | None = None,
    ) -> None:
        self.risk_calculator = (
            risk_calculator
            or StopLossRiskCalculator()
        )

        self.cost_calculator = (
            cost_calculator
            or StopLossCostCalculator()
        )

        self.sizing_calculator = (
            sizing_calculator
            or RiskBasedPositionSizingCalculator()
        )

    def calculate(
        self,
        request: StopLossPositionSizingRequest,
    ) -> StopLossPositionSizingResult:

        if not isinstance(
            request,
            StopLossPositionSizingRequest,
        ):
            raise TypeError(
                "Expected StopLossPositionSizingRequest, "
                f"got {type(request).__name__}"
            )

        risk_result = self.risk_calculator.calculate(
            StopLossRiskRequest(
                symbol=request.symbol,
                equity=request.equity,
                risk_per_trade=request.risk_per_trade,
                entry_price=request.entry_price,
                stop_price=request.stop_price,
            )
        )

        cost_result = self.cost_calculator.calculate(
            StopLossCostRequest(
                symbol=request.symbol,
                entry_price=request.entry_price,
                stop_price=request.stop_price,
                contract_size=request.contract_size,
                currency_conversion_rate=(
                    request.currency_conversion_rate
                ),
            )
        )

        sizing_result = self.sizing_calculator.calculate(
            RiskBasedPositionSizingRequest(
                symbol=request.symbol,
                risk_budget=risk_result.risk_budget,
                loss_per_lot_at_stop=(
                    cost_result.loss_per_lot_at_stop
                ),
            )
        )

        return StopLossPositionSizingResult(
            symbol=request.symbol,
            equity=request.equity,
            risk_per_trade=request.risk_per_trade,
            risk_budget=risk_result.risk_budget,
            entry_price=request.entry_price,
            stop_price=request.stop_price,
            stop_distance=cost_result.stop_distance,
            contract_size=request.contract_size,
            currency_conversion_rate=(
                request.currency_conversion_rate
            ),
            loss_per_lot_at_stop=(
                cost_result.loss_per_lot_at_stop
            ),
            volume_lots=sizing_result.volume_lots,
        )

# ============================================================================= END
