# f07_agents/agent_state.py (2)
#
# Created: 1405/06/19
# Last Edited: 1405/06/23-21:

# Runtime state of Symbol-Agent and Meta-Agent.
#
# این فایل فقط state runtime را نگهداری می‌کند.
# هیچ تصمیمی در این فایل گرفته نمی‌شود.


from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Mapping

from f07_agents.contracts import (
    DecisionMode,
    ModelIdentity,
)


# =============================================================================
# Class-1: Symbol-Agent State
# =============================================================================

@dataclass(slots=True)
class SymbolAgentState:
    """
    وضعیت runtime یک Symbol-Agent.

    هر Symbol-Agent state مستقل خودش را دارد.
    """
    symbol: str
    decision_count: int = 0
    error_count: int = 0
    trades_count: int = 0
    last_signal: int = 0
    last_confidence: float = 0.0
    last_expected_return: float = 0.0
    last_risk_score: float = 0.0
    last_desired_exposure: float = 0.0
    last_stop_price: float | None = None
    cumulative_reward: float = 0.0
    last_decision_at: Optional[datetime] = None
    last_error: Optional[str] = None
    model: Optional[ModelIdentity] = None
    mode: Optional[DecisionMode] = None

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol).replace(" ","")

        if not self.symbol:
            raise ValueError("symbol is required")

        if self.last_signal not in (-1, 0, 1):
            raise ValueError("last_signal must be -1, 0 or 1")


    def record_decision(
        self,
        *,
        signal: int,
        confidence: float,
        expected_return: float,
        risk_score: float,
        desired_exposure: float,
        stop_price: float | None,
        timestamp: datetime,
        model: ModelIdentity,
        mode: DecisionMode,
    ) -> None:
        if signal not in (-1, 0, 1):
            raise ValueError(
                "signal must be -1, 0, or 1."
            )

        self.decision_count += 1
        self.last_signal = int(signal)
        self.last_confidence = float(confidence)
        self.last_expected_return = float(
            expected_return
        )
        self.last_risk_score = float(
            risk_score
        )
        self.last_desired_exposure = float(
            desired_exposure
        )
        self.last_stop_price = (
            None
            if stop_price is None
            else float(stop_price)
        )
        self.last_decision_at = timestamp
        self.last_error = None
        self.model = model
        self.mode = mode


    def record_error(self, error: Exception) -> None:

        self.error_count += 1
        self.last_error = (
            f"{type(error).__name__}: {error}"
        )


    def add_reward(self, reward: float) -> None:

        self.cumulative_reward += float(reward)


    def reset(self) -> None:

        self.decision_count = 0
        self.error_count = 0
        self.trades_count = 0
        self.last_signal = 0
        self.last_confidence = 0.0
        self.last_expected_return = 0.0
        self.last_risk_score = 0.0
        self.last_desired_exposure = 0.0
        self.last_stop_price = None
        self.cumulative_reward = 0.0
        self.last_decision_at = None
        self.last_error = None
        self.model = None
        self.mode = None


# =============================================================================
# Class-2: Meta-Agent State
# =============================================================================

@dataclass(slots=True)
class MetaAgentState:
    """
    وضعیت runtime Meta-Agent.

    این state مربوط به کل Portfolio است.
    """
    decision_count: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    error_count: int = 0
    cumulative_reward: float = 0.0
    equity: float = 0.0
    drawdown: float = 0.0
    total_exposure: float = 0.0

    capital_allocation: Dict[str, float] = field(default_factory=dict)
    margin_allocation: Dict[str, float] = field(default_factory=dict)
    target_stop_price: Dict[str, float | None] = field(default_factory=dict)

    last_decision_at: Optional[datetime] = None
    last_rejection_reason: Optional[str] = None
    last_error: Optional[str] = None
    model: Optional[ModelIdentity] = None
    mode: Optional[DecisionMode] = None


    def record_decision(
        self,
        *,
        approved: bool,
        equity: float,
        drawdown: float,
        total_exposure: float,
        capital_allocation: Mapping[str, float],
        margin_allocation: Mapping[str, float],
        target_stop_price: Mapping[str, float | None],
        timestamp: datetime,
        model: ModelIdentity,
        mode: DecisionMode,
        rejection_reason: Optional[str] = None,
    ) -> None:
        self.decision_count += 1

        if approved:
            self.approved_count += 1
        else:
            self.rejected_count += 1

        self.equity = float(equity)
        self.drawdown = float(drawdown)
        self.total_exposure = float(total_exposure)

        self.capital_allocation = {
            str(symbol).replace(" ", ""): float(value)
            for symbol, value in capital_allocation.items()
        }

        self.margin_allocation = {
            str(symbol).replace(" ", ""): float(value)
            for symbol, value in margin_allocation.items()
        }

        self.target_stop_price = {
            str(symbol).replace(" ", ""): (
                None
                if value is None
                else float(value)
            )
            for symbol, value in target_stop_price.items()
        }

        self.last_decision_at = timestamp
        self.last_rejection_reason = rejection_reason
        self.last_error = None
        self.model = model
        self.mode = mode


    def record_error(self, error: Exception) -> None:

        self.error_count += 1
        self.last_error = (
            f"{type(error).__name__}: {error}"
        )


    def add_reward(self, reward: float) -> None:

        self.cumulative_reward += float(reward)


    def reset(self) -> None:

        self.decision_count = 0
        self.approved_count = 0
        self.rejected_count = 0
        self.error_count = 0
        self.cumulative_reward = 0.0
        self.equity = 0.0
        self.drawdown = 0.0
        self.total_exposure = 0.0
        self.capital_allocation.clear()
        self.margin_allocation.clear()
        self.target_stop_price.clear()
        self.last_decision_at = None
        self.last_rejection_reason = None
        self.last_error = None
        self.model = None
        self.mode = None

# ============================================================================= END
