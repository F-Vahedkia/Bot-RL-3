# f08_risk/risk_based_position_sizing.py (34)
# =============================================================================
# Bot-RL-3 v8
# Chapter 4 - Risk Layer
#
# Risk-budget based position sizing.
#
# Flow:
#
#   risk_budget
#       +
#   loss_per_lot_at_stop
#       ↓
#   volume_lots
#
# Responsibilities:
#   - convert monetary risk budget into continuous lot volume
#
# Non-responsibilities:
#   - Stop-Loss price selection
#   - broker communication
#   - broker lot constraints
#   - lot rounding
#   - execution
#   - portfolio risk constraints
#   - direction
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


# =============================================================================
# Request
# =============================================================================

@dataclass(frozen=True, slots=True)
class RiskBasedPositionSizingRequest:
    symbol: str
    risk_budget: float
    loss_per_lot_at_stop: float

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")

        if not isfinite(self.risk_budget):
            raise ValueError("risk_budget must be finite")

        if self.risk_budget < 0.0:
            raise ValueError("risk_budget must be >= 0")

        if not isfinite(self.loss_per_lot_at_stop):
            raise ValueError(
                "loss_per_lot_at_stop must be finite"
            )

        if self.loss_per_lot_at_stop <= 0.0:
            raise ValueError(
                "loss_per_lot_at_stop must be > 0"
            )


# =============================================================================
# Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class RiskBasedPositionSizingResult:
    symbol: str
    risk_budget: float
    loss_per_lot_at_stop: float
    volume_lots: float

    def __post_init__(self) -> None:
        values = (
            self.risk_budget,
            self.loss_per_lot_at_stop,
            self.volume_lots,
        )

        if not all(isfinite(value) for value in values):
            raise ValueError(
                "RiskBasedPositionSizingResult contains "
                "non-finite values"
            )

        if self.risk_budget < 0.0:
            raise ValueError("risk_budget must be >= 0")

        if self.loss_per_lot_at_stop <= 0.0:
            raise ValueError(
                "loss_per_lot_at_stop must be > 0"
            )

        if self.volume_lots < 0.0:
            raise ValueError(
                "volume_lots must be >= 0"
            )


# =============================================================================
# Calculator
# =============================================================================

class RiskBasedPositionSizingCalculator:
    """
    Pure deterministic risk-based position-sizing calculator.

    Formula:

        volume_lots =
            risk_budget / loss_per_lot_at_stop

    Direction is intentionally not represented.

    Broker constraints such as min_lot, lot_step and max_lot
    are intentionally outside this calculator.
    """

    def calculate(
        self,
        request: RiskBasedPositionSizingRequest,
    ) -> RiskBasedPositionSizingResult:

        if not isinstance(
            request,
            RiskBasedPositionSizingRequest,
        ):
            raise TypeError(
                "Expected RiskBasedPositionSizingRequest, "
                f"got {type(request).__name__}"
            )

        volume_lots = (
            request.risk_budget
            / request.loss_per_lot_at_stop
        )

        return RiskBasedPositionSizingResult(
            symbol=request.symbol,
            risk_budget=request.risk_budget,
            loss_per_lot_at_stop=request.loss_per_lot_at_stop,
            volume_lots=volume_lots,
        )

# ============================================================================= END
