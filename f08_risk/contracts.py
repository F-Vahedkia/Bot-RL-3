# f08_risk/contracts.py (1)
#
# Created: 1405/06/19

# Chapter 4 - Risk & Constraint Layer
#
# این فایل قراردادهای رسمی Risk Layer را تعریف می‌کند.
#
# معماری:
#     f07_agents
#          |
#          v
#     PortfolioDecision
#          |
#          v
#     RiskRequest
#          |
#          v
#     RiskEngine
#          |
#          v
#     RiskDecision
#
# Risk Layer:
#     - execution انجام نمی‌دهد
#     - broker را صدا نمی‌زند
#     - MT5 را نمی‌شناسد
#
# هدف:
#     بررسی و محدود کردن تصمیم Portfolio قبل از Execution.

"""
نکات مهم که chatGPT در دور دوم نوشته است:
-------------------------------------------
    - PortfolioDecision از f07_agents ورودی اصلی Risk Layer است.
    - مسیر رسمی فعلی:
        PortfolioDecision → RiskRequest → RiskEngine → RiskDecision
    - RiskRequest فعلی دارای risk_context اختیاری است و برای backward compatibility نگه داشته شده.
    - RiskViolation یک contract immutable برای ثبت violation/warning است.
    - RiskDecision خروجی نهایی Risk Layer است و هنوز Order نیست.
    - سه وضعیت رسمی داریم:
        APPROVED / MODIFIED / REJECTED
    - RiskDecision شناسهٔ decision_id و source_decision_id را برای traceability نگه می‌دارد.
    - timestamp باید timezone-aware باشد.
    - allocationها نمی‌توانند منفی باشند و target_exposure می‌تواند signed باشد.
    - portfolio_risk باید finite و نامنفی باشد.
    - RiskRequest تطابق timestamp و mode بین Decision و Portfolio و همچنین timestamp مربوط به RiskContext را enforce می‌کند.
"""
# =====================================================================
# Imports
# =====================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Mapping, Tuple,  TYPE_CHECKING

from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PortfolioDecision,
    PortfolioContext,
)
from f08_risk.stop_loss_position_sizing import (
    StopLossPositionSizingRequest,
    StopLossPositionSizingResult,
)
if TYPE_CHECKING:
    from f08_risk.risk_context import RiskContext

# =====================================================================
# Risk Decision Status
# =====================================================================

class RiskDecisionStatus(str, Enum):
    """
    نتیجه بررسی Risk Layer.
    """
    APPROVED = "approved"
    MODIFIED = "modified"
    REJECTED = "rejected"


# =====================================================================
# Risk Violation
# =====================================================================

@dataclass(frozen=True, slots=True)
class RiskViolation:
    """
    یک rule violation یا warning ثبت‌شده توسط Risk Engine.
    """
    code: str
    message: str
    symbol: str | None = None
    severity: str = "warning"

    def __post_init__(self) -> None:
        if not str(self.code).strip():
            raise ValueError("violation code is required")

        if not str(self.message).strip():
            raise ValueError("violation message is required")

        if self.severity not in ("info", "warning", "critical"):
            raise ValueError("severity must be info, warning or critical")


# =====================================================================
# Risk Request
# =====================================================================

@dataclass(frozen=True, slots=True)
class RiskRequest:
    """
    Input contract for one Risk Engine evaluation.

    `risk_context` is mandatory.
    The official Risk evaluation path is projected-state evaluation.
    """

    decision: PortfolioDecision
    portfolio: PortfolioContext
    risk_context: "RiskContext"
    #-----new-1-start
    stop_loss_requests: Mapping[
            str,
            StopLossPositionSizingRequest,
        ] = field(default_factory=dict)
    #-----new-1-end

    def __post_init__(self) -> None:
        if self.decision is None:
            raise ValueError("decision is required.")

        if self.portfolio is None:
            raise ValueError("portfolio is required.")

        if not isinstance(self.decision, PortfolioDecision):
            raise TypeError("decision must be PortfolioDecision.")

        if not isinstance(self.portfolio, PortfolioContext):
            raise TypeError("portfolio must be PortfolioContext.")

        if self.decision.timestamp != self.portfolio.timestamp:
            raise ValueError("Decision timestamp must match portfolio timestamp.")

        if self.decision.mode != self.portfolio.mode:
            raise ValueError("Decision mode must match portfolio mode.")

        from f08_risk.risk_context import RiskContext

        if self.risk_context is None:
            raise ValueError("risk_context is required.")

        if not isinstance(self.risk_context, RiskContext):
            raise TypeError("risk_context must be RiskContext.")

        if self.risk_context.timestamp != self.decision.timestamp:
            raise ValueError(
                "RiskContext timestamp must match decision timestamp."
            )

        #-----new-2 start
        for symbol, request in self.stop_loss_requests.items():
            if not str(symbol).strip():
                raise ValueError(
                    "stop_loss_requests contains an empty symbol."
                )

            if not isinstance(
                request,
                StopLossPositionSizingRequest,
            ):
                raise TypeError(
                    f"stop_loss_requests[{symbol}] must be "
                    "StopLossPositionSizingRequest."
                )

            normalized_symbol = str(symbol).strip().upper()

            if request.symbol.strip().upper() != normalized_symbol:
                raise ValueError(
                    f"stop_loss_requests[{symbol}] symbol does not match "
                    f"request.symbol={request.symbol}."
                )
        #-----new-2 end
# =====================================================================
# Risk Decision
# =====================================================================

@dataclass(frozen=True, slots=True)
class RiskDecision:
    """
    خروجی نهایی Risk Layer.

    این object هنوز Order نیست.

    آنچه در اینجا نگهداری می‌شود:
        - نتیجه approval
        - decision اصلاح‌شده
        - violationها
        - مدل و traceability
    """

    status: RiskDecisionStatus
    timestamp: datetime
    mode: DecisionMode
    decision_id: str
    source_decision_id: str
    capital_allocation: Mapping[str, float]
    margin_allocation: Mapping[str, float]
    target_exposure: Mapping[str, float]
    target_signals: Mapping[str, int]
    portfolio_risk: float
    violations: Tuple[RiskViolation, ...]
    model: ModelIdentity
    metadata: Mapping[str, float] = field(default_factory=dict)
    #-----new-3 start
    stop_loss_results: Mapping[
        str,
        StopLossPositionSizingResult,
    ] = field(default_factory=dict)
    #-----new-3 end

    @property
    def approved(self) -> bool:
        return self.status != RiskDecisionStatus.REJECTED

    @property
    def modified(self) -> bool:
        return self.status == RiskDecisionStatus.MODIFIED

    @property
    def rejected(self) -> bool:
        return self.status == RiskDecisionStatus.REJECTED


    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        if not str(self.decision_id).strip():
            raise ValueError("decision_id is required")

        if not str(self.source_decision_id).strip():
            raise ValueError("source_decision_id is required")

        if not isfinite(
            float(self.portfolio_risk)
        ):
            raise ValueError("portfolio_risk must be finite")

        if self.portfolio_risk < 0.0:
            raise ValueError("portfolio_risk must be >= 0")

        for symbol, value in (
            self.capital_allocation.items()
        ):
            if not isfinite(float(value)) or value < 0.0:
                raise ValueError(f"Invalid capital allocation for {symbol}")

        for symbol, value in (
            self.margin_allocation.items()
        ):
            if not isfinite(float(value)) or value < 0.0:
                raise ValueError(f"Invalid margin allocation for {symbol}")

        for symbol, value in (
            self.target_exposure.items()
        ):
            if not isfinite(float(value)):
                raise ValueError(f"Invalid target exposure for {symbol}")

        for symbol, signal in (
            self.target_signals.items()
        ):
            if signal not in (-1, 0, 1):
                raise ValueError(f"Invalid target signal for {symbol}")

# ============================================================================= END