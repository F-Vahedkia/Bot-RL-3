# f08_risk/risk_projection.py (9)
# =============================================================================
# Bot-RL-3 v8
# Chapter 4 - Risk Layer
#
# Purpose:
#     - Calculate the portfolio state that would exist AFTER applying a proposed
#     - PortfolioDecision, without executing anything.
#
# Flow:
#     Current RiskContext + PortfolioDecision
#           ->
#     ProjectedRiskState
#
# Rules:
#     - No broker access
#     - No execution
#     - No mutation
#     - Deterministic
# =============================================================================
"""
نکاتی که در مرحله دوم بررسی ها chatGPT نوشته است:
----------------------------------------------------
    - این فایل برای فهم وضعیت فعلی فصل ۴ بسیار مهم است، چون قرارداد
        «اگر این PortfolioDecision اعمال می‌شد، وضعیت ریسک بعد از آن چه می‌شد؟»
        را پیاده می‌کند
    ...
    بقیه موارد را من در اینجا نیاوردم.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from f07_agents.contracts import PortfolioDecision
from f08_risk.projection_margin import (
    ProjectionMarginCalculator,
    ProjectionMarginRequest,
)
from f08_risk.risk_context import (
    RiskContext,
    SymbolRiskSnapshot,
)

# =============================================================================
# Validation helpers
# =============================================================================

def _finite(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _non_negative(name: str, value: float) -> float:
    value = _finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return value


# =============================================================================
# Projected Symbol Risk
# =============================================================================

@dataclass(frozen=True, slots=True)
class ProjectedSymbolRisk:
    """
    Projected risk state for one symbol after applying target exposure.
    """
    symbol: str
    current_exposure: float
    target_exposure: float
    projected_exposure_delta: float
    current_notional: float
    projected_notional: float
    current_used_margin: float
    projected_used_margin: float
    current_position_count: int
    projected_position_count: int

    def __post_init__(self) -> None:

        symbol = str(self.symbol).strip().upper()
        if not symbol:
            raise ValueError("symbol is required.")

        current_exposure = _finite("current_exposure", self.current_exposure)
        target_exposure = _finite("target_exposure", self.target_exposure)
        projected_exposure_delta = _finite("projected_exposure_delta", self.projected_exposure_delta)
        expected_delta = (target_exposure - current_exposure)

        if abs(projected_exposure_delta - expected_delta) > 1e-12:
            raise ValueError(
                "projected_exposure_delta does not match "
                "target_exposure - current_exposure."
            )
        current_notional = _non_negative("current_notional", self.current_notional)
        projected_notional = _non_negative("projected_notional", self.projected_notional)
        current_used_margin = _non_negative("current_used_margin", self.current_used_margin)
        projected_used_margin = _non_negative("projected_used_margin", self.projected_used_margin)

        if (
            isinstance(self.current_position_count, bool)
            or not isinstance(self.current_position_count, int)
        ):
            raise TypeError("current_position_count must be a non-negative integer.")

        if self.current_position_count < 0:
            raise ValueError("current_position_count must be >= 0.")

        if (
            isinstance(self.projected_position_count, bool)
            or not isinstance(self.projected_position_count, int)
        ):
            raise TypeError("projected_position_count must be a non-negative integer.")

        if self.projected_position_count < 0:
            raise ValueError("projected_position_count must be >= 0.")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "current_exposure", current_exposure)
        object.__setattr__(self, "target_exposure", target_exposure)
        object.__setattr__(self, "projected_exposure_delta", projected_exposure_delta)
        object.__setattr__(self, "current_notional", current_notional)
        object.__setattr__(self, "projected_notional", projected_notional)
        object.__setattr__(self, "current_used_margin", current_used_margin)
        object.__setattr__(self, "projected_used_margin", projected_used_margin)
        object.__setattr__(self, "current_position_count", self.current_position_count)
        object.__setattr__(self, "projected_position_count", self.projected_position_count)

# =============================================================================
# Projected Portfolio Risk
# =============================================================================

@dataclass(frozen=True, slots=True)
class ProjectedRiskState:
    """
    Portfolio-level projected state after applying a proposed decision.
    """
    timestamp: object
    decision_id: str
    equity: float
    current_used_margin: float
    projected_used_margin: float
    projected_free_margin: float
    current_total_exposure: float
    projected_total_exposure: float
    projected_position_count: int
    symbols: Mapping[str, ProjectedSymbolRisk]

    def __post_init__(self) -> None:
        if self.timestamp is None:
            raise ValueError("timestamp is required.")

        decision_id = str(self.decision_id).strip()
        if not decision_id:
            raise ValueError("decision_id is required.")

        equity = _non_negative("equity", self.equity)
        current_used_margin = _non_negative("current_used_margin", self.current_used_margin)
        projected_used_margin = _non_negative("projected_used_margin", self.projected_used_margin)
        projected_free_margin = _non_negative("projected_free_margin", self.projected_free_margin)
        current_total_exposure = _non_negative("current_total_exposure", self.current_total_exposure)
        projected_total_exposure = _non_negative("projected_total_exposure", self.projected_total_exposure)

        if int(self.projected_position_count) != self.projected_position_count:
            raise ValueError("projected_position_count must be an integer.")

        projected_position_count = int(self.projected_position_count)

        if projected_position_count < 0:
            raise ValueError("projected_position_count must be >= 0.")

        if projected_free_margin > equity + 1e-12:
            raise ValueError("projected_free_margin cannot exceed equity.")

        normalized_symbols = {}

        for symbol, snapshot in self.symbols.items():
            normalized_symbol = str(symbol).strip().upper()

            if not normalized_symbol:
                raise ValueError("symbols contains an empty symbol.")

            if not isinstance(snapshot, ProjectedSymbolRisk):
                raise TypeError(
                    f"symbols[{normalized_symbol}] must be "
                    "ProjectedSymbolRisk."
                )

            if snapshot.symbol != normalized_symbol:
                raise ValueError(
                    f"symbols[{normalized_symbol}] does not match "
                    f"snapshot.symbol={snapshot.symbol}."
                )

            normalized_symbols[normalized_symbol] = snapshot

        object.__setattr__(self, "decision_id", decision_id)
        object.__setattr__(self, "equity", equity)
        object.__setattr__(self, "current_used_margin", current_used_margin)
        object.__setattr__(self, "projected_used_margin", projected_used_margin)
        object.__setattr__(self, "projected_free_margin", projected_free_margin)
        object.__setattr__(self, "current_total_exposure", current_total_exposure)
        object.__setattr__(self, "projected_total_exposure", projected_total_exposure)
        object.__setattr__(self, "projected_position_count", projected_position_count)
        object.__setattr__(self, "symbols", MappingProxyType(normalized_symbols))

    @property
    def projected_margin_utilization(self) -> float:
        if self.equity <= 0.0:
            return 0.0

        return (
            self.projected_used_margin
            / self.equity
        )


# =============================================================================
# Risk Projection
# =============================================================================

class RiskProjection:
    """
    Pure deterministic risk-state projector.

    It combines:
        current RiskContext + proposed PortfolioDecision
    and returns:
        ProjectedRiskState

    Optional instrument-aware margin projection is supported through
    ProjectionMarginRequest mappings.

    No external state is modified.
    """

    def __init__(
        self,
        *,
        margin_calculator: ProjectionMarginCalculator | None = None,
    ) -> None:
        self._margin_calculator = (
            margin_calculator
            if margin_calculator is not None
            else ProjectionMarginCalculator()
        )


    def project(
        self,
        context: RiskContext,
        decision: PortfolioDecision,
        *,
        margin_requests: Mapping[str, ProjectionMarginRequest] | None = None,
    ) -> ProjectedRiskState:

        if context is None:
            raise ValueError("context is required.")

        if decision is None:
            raise ValueError("decision is required.")
    
        if decision.timestamp != context.timestamp:
            raise ValueError("Decision timestamp must match RiskContext timestamp.")

        # ---------------------------------------------------------------------
        # Optional instrument-aware margin projection
        # ---------------------------------------------------------------------
        if margin_requests is not None:
            normalized_margin_requests = {}

            for symbol, request in margin_requests.items():
                normalized_symbol = str(symbol).strip().upper()

                if not normalized_symbol:
                    raise ValueError("margin_requests contains an empty symbol.")

                if not isinstance(request, ProjectionMarginRequest):
                    raise TypeError(
                        f"margin_requests[{normalized_symbol}] must be "
                        "ProjectionMarginRequest."
                    )

                if request.symbol != normalized_symbol:
                    raise ValueError(
                        f"margin_requests[{normalized_symbol}] does not "
                        f"match request.symbol={request.symbol}."
                    )

                normalized_margin_requests[normalized_symbol] = request

            margin_requests = normalized_margin_requests

        # ---------------------------------------------------------------------
        # Normalize proposed targets
        # ---------------------------------------------------------------------

        target_exposures = {
            str(symbol).strip().upper(): float(exposure)
            for symbol, exposure
            in decision.target_exposure.items()
        }

        # ---------------------------------------------------------------------
        # Determine complete symbol universe
        # ---------------------------------------------------------------------

        symbols = set(context.symbols)
        symbols.update(target_exposures)

        # ---------------------------------------------------------------------
        # Project every symbol independently
        # ---------------------------------------------------------------------

        projected_symbols = {}

        for symbol in sorted(symbols):
            current = context.symbol_snapshot(symbol)

            # Symbol does not currently have a position.
            if current is None:
                current = SymbolRiskSnapshot(
                    symbol=symbol,
                    exposure=0.0,
                    notional=0.0,
                    used_margin=0.0,
                    current_lots=0.0,
                    current_side=0,
                )
            target_exposure = target_exposures.get(symbol, current.exposure)
            target_exposure = _finite(f"target_exposure[{symbol}]", target_exposure)

            current_position_count = current.position_count

            if abs(target_exposure) <= 1e-12:
                projected_position_count = 0
            elif abs(current.exposure) <= 1e-12:
                projected_position_count = 1
            else:
                projected_position_count = current.position_count

            # -----------------------------------------------------------------
            # Instrument-aware projection path
            #
            # This path is authoritative whenever margin_requests is supplied.
            # It calculates target notional and target margin directly from
            # instrument specifications.
            # -----------------------------------------------------------------
            if margin_requests is not None:
                if abs(target_exposure) <= 1e-12:
                    projected_notional = 0.0
                    projected_used_margin = 0.0
                else:
                    request = margin_requests.get(symbol)
                    if request is None:
                        raise ValueError(
                            f"Missing margin request for projected "
                            f"symbol {symbol!r}."
                        )
                    if abs(request.target_exposure - target_exposure) > 1e-12:
                        raise ValueError(
                            f"margin_requests[{symbol}] target_exposure "
                            "does not match PortfolioDecision."
                        )
                    margin_result = self._margin_calculator.calculate(request) 
                    projected_notional = margin_result.target_notional
                    projected_used_margin = margin_result.required_margin

            # -----------------------------------------------------------------
            # Legacy normalized projection path
            #
            # Used only when instrument-level margin information is absent.
            # Existing RiskProjection behavior is preserved exactly here.
            # -----------------------------------------------------------------
            else:
                if abs(current.exposure) <= 1e-12:
                    projected_notional = 0.0
                    projected_used_margin = 0.0
                else:
                    scale = abs(target_exposure) / abs(current.exposure)
                    projected_notional = current.notional * scale
                    projected_used_margin = current.used_margin * scale

                # Explicit close.
                if abs(target_exposure) <= 1e-12:
                    projected_notional = 0.0
                    projected_used_margin = 0.0

            projected_symbols[symbol] = (
                ProjectedSymbolRisk(
                    symbol=symbol,
                    current_exposure=current.exposure,
                    target_exposure=target_exposure,
                    projected_exposure_delta=target_exposure - current.exposure,
                    current_notional=current.notional,
                    projected_notional=projected_notional,
                    current_used_margin=current.used_margin,
                    projected_used_margin=projected_used_margin,
                    current_position_count=current_position_count,
                    projected_position_count=projected_position_count,
                )
            )
        # ---------------------------------------------------------------------
        # Portfolio aggregates
        # ---------------------------------------------------------------------

        current_total_exposure = (
            context.total_current_exposure
        )
        projected_total_exposure = sum(
            abs(item.target_exposure)
            for item in projected_symbols.values()
        )
        current_used_margin = (
            context.total_current_used_margin
        )
        projected_used_margin = sum(
            item.projected_used_margin
            for item in projected_symbols.values()
        )
        projected_free_margin = max(
            0.0,
            context.account.equity
            - projected_used_margin,
        )
        # projected_position_count = sum(
        #     1
        #     for item in projected_symbols.values()
        #     if abs(item.target_exposure) > 1e-12
        # )
        projected_position_count = sum(
            item.projected_position_count
            for item in projected_symbols.values()
        )
        return ProjectedRiskState(
            timestamp=context.timestamp,
            decision_id=decision.decision_id,
            equity=context.account.equity,
            current_used_margin=current_used_margin,
            projected_used_margin=projected_used_margin,
            projected_free_margin=projected_free_margin,
            current_total_exposure=current_total_exposure,
            projected_total_exposure=projected_total_exposure,
            projected_position_count=projected_position_count,
            symbols=projected_symbols,
        )

# ============================================================================= END