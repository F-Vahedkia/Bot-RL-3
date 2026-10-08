# f08_risk/risk_engine.py (4)
# =============================================================================
# Bot-RL-3 v8
# Chapter 4 - Risk Layer
#
# Production-oriented deterministic pre-trade risk engine.
#
# Flow:
#
#   RiskRequest
#       |
#       +--> RiskContext
#       |
#       +--> RiskProjection
#       |
#       +--> ProjectedRiskState
#       |
#       +--> hard constraints / modifications
#       |
#       +--> RiskDecision
#
# =============================================================================
"""
نکاتی که در مرحله دوم chatGPT اعلام کرده:
------------------------------------------
    - RiskEngine هستهٔ deterministic pre-trade risk است و
        مسئولیت‌هایش شامل exposure، concentration، correlation،
        margin utilization، portfolio-risk و projected-state evaluation 
        ست؛ execution و broker و position accounting خارج از مسئولیت آن هستند.
    - مسیر production اکنون از RiskContext و RiskProjection عبور می‌کند و
        در صورت وجود margin_requests می‌تواند از ProjectionMarginRequest برای
        مارجین محاسبه‌شده به‌شکل instrument-aware استفاده کند.
    - پس از هر modification مهم، projection دوباره محاسبه می‌شود؛
        بنابراین مسیر فعلی صرفاً «حد را cap کن و تمام» نیست.
    - محدودیت‌های فعلی در Engine شامل symbol exposure، total exposure،
        correlation، capital concentration و margin utilization هستند.
    - max_open_positions و max_positions_per_symbol اکنون در مسیر projected وارد شده‌اند.
        برای per-symbol، از projected.symbols[symbol].projected_position_count استفاده می‌شود.
    - در پایان، RiskDecision همراه با خلاصهٔ projected state تولید می‌شود و
        RiskEngineState نیز هر evaluation را ثبت می‌کند.
    - یک نکتهٔ مهم برای مرحلهٔ بعدی این است که risk_engine.py همین حالا
        ترتیب اعمال constraintها را به‌صورت اجرایی دارد؛
        بنابراین مورد «Hard vs Soft constraints + Conflict Resolution» بیشتر از جنس
        formalize کردن همین policy و تست سناریوهای تعارض است،
        نه اینکه ترتیب از صفر ساخته شود. این ترتیب در کد فعلی قابل مشاهده است.
"""
from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Dict, Mapping, Tuple
from uuid import uuid4

from f07_agents.contracts import PortfolioDecision

from f08_risk.contracts import (
    RiskDecision,
    RiskDecisionStatus,
    RiskRequest,
    RiskViolation,
)
from f08_risk.limits import RiskLimits
from f08_risk.risk_mode_policy import get_risk_mode_policy
from f08_risk.projection_margin import ProjectionMarginRequest
from f08_risk.risk_context import RiskContext
from f08_risk.risk_projection import (
    ProjectedRiskState,
    RiskProjection,
)
from f08_risk.stop_loss_position_sizing import (
    StopLossPositionSizingCalculator,
    StopLossPositionSizingResult,
)
from f08_risk.audit import (
    RiskAuditModification,
    RiskAuditRecord,
    RiskAuditTrail
)
from f08_risk.state import RiskEngineState
from f08_risk.position_sizing import (
    PositionSizingCalculator,
    PositionSizingRequest,
)

# =============================================================================
# Validation
# =============================================================================

def _finite(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value

# =============================================================================
# Risk Engine
# =============================================================================

class RiskEngine:
    """
    Deterministic portfolio pre-trade risk engine.

    Responsibilities:
        - hard risk blocks
        - exposure constraints
        - concentration limits
        - correlation limits
        - margin utilization
        - portfolio-risk budget
        - projected-state evaluation

    Non-responsibilities:
        - broker communication
        - order placement
        - fill simulation
        - position accounting
        - price discovery
    """

    def __init__(
        self,
        limits: RiskLimits | None = None,
        projector: RiskProjection | None = None,
        stop_loss_calculator: StopLossPositionSizingCalculator | None = None,
    ) -> None:

        self.limits = limits or RiskLimits()
        self.projector = projector or RiskProjection()
        self.stop_loss_calculator = (
            stop_loss_calculator
            or StopLossPositionSizingCalculator()
        )
        self.position_sizing_calculator = PositionSizingCalculator()
        self.state = RiskEngineState()
        self.audit_trail = RiskAuditTrail()
    # =========================================================================
    # Public API
    # ========================================================================= new method
    def evaluate(
        self,
        request: RiskRequest,
        *,
        margin_requests: Mapping[str, ProjectionMarginRequest] | None = None,
    ) -> RiskDecision:

        if request is None:
            raise ValueError("request is required.")

        if not isinstance(request, RiskRequest):
            raise TypeError("request must be RiskRequest.")

        policy = get_risk_mode_policy(
            request.decision.mode
        )

        if not policy.enforce_hard_constraints:
            raise RuntimeError(
                "Risk mode policy cannot disable hard constraints."
            )

        if not policy.enforce_soft_constraints:
            raise RuntimeError(
                "Risk mode policy cannot disable soft constraints."
            )

        if not policy.enforce_stop_loss:
            raise RuntimeError(
                "Risk mode policy cannot disable stop-loss risk."
            )

        if request.risk_context is None:
            raise RuntimeError(
                "RiskRequest.risk_context is required for "
                "projected Risk evaluation."
            )

        result = self._evaluate_projected(
            request,
            margin_requests=margin_requests,
        )

        metadata = dict(
            result.metadata
        )

        metadata["risk_mode_policy"] = {
            "mode": policy.mode,
            "enforce_hard_constraints": (
                policy.enforce_hard_constraints
            ),
            "enforce_soft_constraints": (
                policy.enforce_soft_constraints
            ),
            "enforce_stop_loss": (
                policy.enforce_stop_loss
            ),
            "record_audit": (
                policy.record_audit
            ),
        }

        result = replace(
            result,
            metadata=metadata,
        )

        if policy.record_audit:
            self._record_audit(
                request=request,
                result=result,
            )

        return result

    # =========================================================================
    # Account-level hard safety guards
    # =========================================================================

    def _account_hard_guard(
        self,
        *,
        equity: float,
        free_margin: float,
        used_margin: float,
        margin_level: float,
        leverage: float,
    ) -> RiskViolation | None:
        """
        Validate account-level hard safety conditions.

        This helper does not perform projection and does not modify anything.
        It only detects unsafe account-state conditions.
        """

        equity = _finite("equity", equity)
        free_margin = _finite("free_margin", free_margin)
        used_margin = _finite("used_margin", used_margin)
        margin_level = _finite("margin_level", margin_level)
        leverage = _finite("leverage", leverage)

        if equity <= 0.0:
            return RiskViolation(
                code="NO_USABLE_CAPITAL",
                message="Account equity is zero or negative; no usable capital remains.",
                severity="critical",
            )

        if free_margin < 0.0:
            return RiskViolation(
                code="INSUFFICIENT_FREE_MARGIN",
                message="Account free margin is negative.",
                severity="critical",
            )

        if used_margin > 0.0 and margin_level <= 0.0:
            return RiskViolation(
                code="INVALID_MARGIN_LEVEL",
                message="Margin level must be > 0 while used margin is positive.",
                severity="critical",
            )

        if leverage <= 0.0:
            return RiskViolation(
                code="INVALID_LEVERAGE",
                message="Account leverage must be > 0.",
                severity="critical",
            )

        if (
            self.limits.max_account_leverage is not None
            and leverage > self.limits.max_account_leverage
        ):
            return RiskViolation(
                code="MAX_ACCOUNT_LEVERAGE",
                message=("Account leverage exceeds the configured maximum leverage limit."),
                severity="critical",
            )

        return None

    # =========================================================================
    # Projected-state production path
    # =========================================================================

    def _evaluate_projected(
        self,
        request: RiskRequest,
        *,
        margin_requests: Mapping[str, ProjectionMarginRequest] | None = None,
    ) -> RiskDecision:
        
        decision = request.decision
        context = request.risk_context

        if context is None:
            raise RuntimeError("Projected evaluation requires risk_context.")

        # ---------------------------------------------------------------------
        # Account-level hard safety guards.
        # ---------------------------------------------------------------------

        account_guard = self._account_hard_guard(
            equity=context.account.equity,
            free_margin=context.account.free_margin,
            used_margin=context.account.used_margin,
            margin_level=context.account.margin_level,
            leverage=context.account.leverage,
        )
        if account_guard is not None:
            return self._reject(
                request,
                account_guard,
            )

        # ---------------------------------------------------------------------
        # Hard blocks that depend on current portfolio state.
        # ---------------------------------------------------------------------

        violations = []

        if context.risk_blocked:
            return self._reject(
                request,
                RiskViolation(
                    code="RISK_BLOCKED",
                    message="Portfolio is currently risk blocked.",
                    severity="critical",
                ),
            )

        if (
            context.account.drawdown
            >= self.limits.max_drawdown
        ):
            return self._reject(
                request,
                RiskViolation(
                    code="MAX_DRAWDOWN",
                    message="Portfolio drawdown is at or above the configured maximum.",
                    severity="critical",
                ),
            )

        if (
            context.account.daily_drawdown
            >= self.limits.max_daily_drawdown
        ):
            return self._reject(
                request,
                RiskViolation(
                    code="MAX_DAILY_DRAWDOWN",
                    message="Portfolio daily drawdown is at or above the configured maximum.",
                    severity="critical",
                ),
            )

        if (
            decision.portfolio_risk
            > self.limits.max_portfolio_risk
        ):
            return self._reject(
                request,
                RiskViolation(
                    code="MAX_PORTFOLIO_RISK",
                    message="Requested portfolio risk exceeds the configured portfolio-risk budget.",
                    severity="critical",
                ),
            )

        # ---------------------------------------------------------------------
        # Start from an immutable decision candidate.
        # ---------------------------------------------------------------------

        candidate = decision
        modified = False

        # ---------------------------------------------------------------------
        # Project initial candidate.
        # ---------------------------------------------------------------------

        projected = self._project_candidate(
            context=context,
            decision=candidate,
            margin_requests=margin_requests,
        )

        # ---------------------------------------------------------------------
        # Symbol exposure limits.
        # ---------------------------------------------------------------------

        target_exposure = dict(
            candidate.target_exposure
        )

        for symbol in sorted(target_exposure):

            exposure = float(
                target_exposure[symbol]
            )

            if abs(exposure) > self.limits.max_symbol_exposure:

                target_exposure[symbol] = (
                    self._cap_signed(
                        exposure,
                        self.limits.max_symbol_exposure,
                    )
                )

                violations.append(
                    RiskViolation(
                        code="MAX_SYMBOL_EXPOSURE",
                        message=(
                            f"Projected exposure for {symbol} "
                            "exceeds the symbol exposure limit."
                        ),
                        symbol=symbol,
                        severity="warning",
                    )
                )

                modified = True

        # ---------------------------------------------------------------------
        # Rebuild candidate with symbol exposure modifications.
        # ---------------------------------------------------------------------

        if modified:
            candidate = replace(
                candidate,
                target_exposure=target_exposure,
            )

            projected = self._project_candidate(
                context=context,
                decision=candidate,
                margin_requests=margin_requests,
            )

        # ---------------------------------------------------------------------
        # Total projected exposure.
        # ---------------------------------------------------------------------

        if (
            projected.projected_total_exposure
            > self.limits.max_total_exposure
        ):

            factor = (
                self.limits.max_total_exposure
                / projected.projected_total_exposure
            )

            target_exposure = {
                symbol: float(exposure) * factor
                for symbol, exposure
                in candidate.target_exposure.items()
            }

            candidate = replace(
                candidate,
                target_exposure=target_exposure,
            )

            projected = self._project_candidate(
                context=context,
                decision=candidate,
                margin_requests=margin_requests,
            )

            violations.append(
                RiskViolation(
                    code="MAX_TOTAL_EXPOSURE",
                    message=(
                        "Projected total exposure exceeds the "
                        "portfolio exposure limit and was scaled."
                    ),
                    severity="warning",
                )
            )

            modified = True

        # ---------------------------------------------------------------------
        # Correlation limits.
        # ---------------------------------------------------------------------

        target_exposure = dict(
            candidate.target_exposure
        )

        correlation_modified = False

        for symbol_a in sorted(context.correlation):

            row = context.correlation[symbol_a]
            for symbol_b in sorted(row):
                if symbol_a >= symbol_b:
                    continue

                correlation = float(
                    row[symbol_b]
                )

                if (
                    abs(correlation)
                    < self.limits.high_correlation_threshold
                ):
                    continue

                exposure_a = abs(
                    target_exposure.get(symbol_a, 0.0)
                )

                exposure_b = abs(
                    target_exposure.get(symbol_b, 0.0)
                )

                combined = (
                    exposure_a + exposure_b
                )

                if (
                    combined
                    <= self.limits.max_correlated_exposure
                ):
                    continue

                factor = (
                    self.limits.max_correlated_exposure
                    / combined
                )

                if symbol_a in target_exposure:
                    target_exposure[symbol_a] = (
                        float(
                            target_exposure[symbol_a]
                        )
                        * factor
                    )

                if symbol_b in target_exposure:
                    target_exposure[symbol_b] = (
                        float(
                            target_exposure[symbol_b]
                        )
                        * factor
                    )

                correlation_modified = True

                violations.append(
                    RiskViolation(
                        code="MAX_CORRELATED_EXPOSURE",
                        message=(
                            f"Highly correlated pair {symbol_a}/{symbol_b} "
                            "exceeds the combined exposure limit."
                        ),
                        severity="warning",
                    )
                )

        if correlation_modified:

            candidate = replace(
                candidate,
                target_exposure=target_exposure,
            )

            projected = self._project_candidate(
                context=context,
                decision=candidate,
                margin_requests=margin_requests,
            )

            modified = True

        # ---------------------------------------------------------------------
        # Capital concentration.
        # ---------------------------------------------------------------------

        capital_allocation = dict(
            candidate.capital_allocation
        )

        concentration_modified = False

        for symbol in sorted(capital_allocation):

            allocation = _finite(
                f"capital_allocation[{symbol}]",
                capital_allocation[symbol],
            )

            if abs(allocation) <= self.limits.max_symbol_concentration:
                continue

            capital_allocation[symbol] = (
                self._cap_signed(
                    allocation,
                    self.limits.max_symbol_concentration,
                )
            )

            concentration_modified = True

            violations.append(
                RiskViolation(
                    code="MAX_SYMBOL_CONCENTRATION",
                    message=(
                        f"Capital allocation for {symbol} "
                        "exceeds the symbol concentration limit."
                    ),
                    symbol=symbol,
                    severity="warning",
                )
            )

        if concentration_modified:

            candidate = replace(
                candidate,
                capital_allocation=capital_allocation,
            )

            modified = True

        # ---------------------------------------------------------------------
        # Projected margin utilization.
        #
        # Projection is intentionally based on normalized RiskContext values.
        # Broker-specific margin calculation remains outside this engine.
        # ---------------------------------------------------------------------

        if (
            projected.projected_margin_utilization
            > self.limits.max_margin_utilization
        ):

            current_projected_margin = (
                projected.projected_used_margin
            )

            if current_projected_margin > 0.0:

                factor = (
                    self.limits.max_margin_utilization
                    * context.account.equity
                    / current_projected_margin
                )

                factor = min(
                    max(factor, 0.0),
                    1.0,
                )

                target_exposure = {
                    symbol: float(exposure) * factor
                    for symbol, exposure
                    in candidate.target_exposure.items()
                }

                capital_allocation = {
                    symbol: float(allocation) * factor
                    for symbol, allocation
                    in candidate.capital_allocation.items()
                }

                margin_allocation = {
                    symbol: float(allocation) * factor
                    for symbol, allocation
                    in candidate.margin_allocation.items()
                }

                candidate = replace(
                    candidate,
                    target_exposure=target_exposure,
                    capital_allocation=capital_allocation,
                    margin_allocation=margin_allocation,
                )

                projected = self._project_candidate(
                    context=context,
                    decision=candidate,
                    margin_requests=margin_requests,
                )

                violations.append(
                    RiskViolation(
                        code="MAX_MARGIN_UTILIZATION",
                        message=(
                            "Projected margin utilization exceeds "
                            "the configured maximum and was scaled."
                        ),
                        severity="warning",
                    )
                )

                modified = True

        # ---------------------------------------------------------------------
        # Stop-loss risk caps.
        # ---------------------------------------------------------------------

        if request.stop_loss_requests:
            (
                candidate,
                stop_loss_violations,
                stop_loss_modified,
            ) = self._apply_stop_loss_risk_caps(
                request,
                candidate,
            )
            if stop_loss_modified:
                violations.extend(
                    stop_loss_violations
                )
                modified = True
                projected = self._project_candidate(
                    context=context,
                    decision=candidate,
                    margin_requests=margin_requests,
                )

        # ---------------------------------------------------------------------
        # Final projected hard safety checks.
        # ---------------------------------------------------------------------

        if (
            self.limits.max_open_positions is not None
            and projected.projected_position_count
            > self.limits.max_open_positions
        ):
            return self._reject(
                request,
                RiskViolation(
                    code="MAX_OPEN_POSITIONS",
                    message=(
                        "Projected open-position count exceeds "
                        "the configured portfolio position limit."
                    ),
                    severity="critical",
                ),
                extra_violations=tuple(violations),
            )

        # ======================= بلوک جدید- شروع
        if self.limits.max_positions_per_symbol is not None:
            for symbol in sorted(projected.symbols):
                symbol_projection = projected.symbols[symbol]

                if (
                    symbol_projection.projected_position_count
                    > self.limits.max_positions_per_symbol
                ):
                    return self._reject(
                        request,
                        RiskViolation(
                            code="MAX_POSITIONS_PER_SYMBOL",
                            message=(
                                f"Projected open-position count for "
                                f"{symbol} exceeds the configured "
                                "per-symbol position limit."
                            ),
                            symbol=symbol,
                            severity="critical",
                        ),
                        extra_violations=tuple(violations),
                    )
        # ======================= بلوک جدید- پایان

        if projected.projected_free_margin <= 0.0:
            return self._reject(
                request,
                RiskViolation(
                    code="NO_FREE_MARGIN",
                    message=(
                        "Projected free margin is zero or negative."
                    ),
                    severity="critical",
                ),
                extra_violations=tuple(violations),
            )
        
        # ---------------------------------------------------------------------
        # Build final RiskDecision.
        # ---------------------------------------------------------------------

        status = (
            RiskDecisionStatus.MODIFIED
            if modified
            else RiskDecisionStatus.APPROVED
        )

        metadata = {
            "evaluation_path": "projected",
            "projected": self._projected_summary(
                projected
            ),
            "risk_context_symbols": tuple(
                sorted(context.symbols)
            ),
        }

        self._validate_stop_loss_alignment(
            request
        )

        stop_loss_results = (
            self._calculate_stop_loss_results(request)
        )

        result = RiskDecision(
            status=status,
            timestamp=candidate.timestamp,
            mode=candidate.mode,
            decision_id=candidate.decision_id,
            source_decision_id=candidate.decision_id,
            capital_allocation=dict(
                candidate.capital_allocation
            ),
            margin_allocation=dict(
                candidate.margin_allocation
            ),
            target_exposure=dict(
                candidate.target_exposure
            ),
            target_signals=dict(
                candidate.target_signals
            ),
            portfolio_risk=float(
                candidate.portfolio_risk
            ),
            violations=tuple(
                violations
            ),
            model=candidate.model,
            metadata=metadata,
            stop_loss_results=stop_loss_results,
        )

        self.state.record(
            status=status,
            decision_id=result.decision_id,
            violation_codes=tuple(v.code for v in violations),
            timestamp=result.timestamp,
        )

        return result

    # =========================================================================
    # Projected-state helper
    # =========================================================================

    def _project_candidate(
        self,
        *,
        context: RiskContext,
        decision: PortfolioDecision,
        margin_requests: Mapping[
            str,
            ProjectionMarginRequest,
        ] | None,
    ) -> ProjectedRiskState:
        """
        Project one candidate decision.

        When margin_requests is absent, preserve the original normalized
        RiskProjection behavior.

        When margin_requests is supplied, align each active target exposure
        with its instrument-aware ProjectionMarginRequest before projection.
        """

        if margin_requests is None:
            return self.projector.project(
                context=context,
                decision=decision,
            )

        normalized_requests = {}

        for symbol, request in margin_requests.items():
            normalized_symbol = (
                str(symbol).strip().upper()
            )

            if not normalized_symbol:
                raise ValueError(
                    "margin_requests contains an empty symbol."
                )

            if not isinstance(
                request,
                ProjectionMarginRequest,
            ):
                raise TypeError(
                    f"margin_requests[{normalized_symbol}] must be "
                    "ProjectionMarginRequest."
                )

            if request.symbol != normalized_symbol:
                raise ValueError(
                    f"margin_requests[{normalized_symbol}] does not "
                    f"match request.symbol={request.symbol}."
                )

            if abs(
                float(request.equity)
                - float(context.account.equity)
            ) > 1e-12:
                raise ValueError(
                    f"margin_requests[{normalized_symbol}] equity does not "
                    "match RiskContext account equity."
                )

            normalized_requests[
                normalized_symbol
            ] = request

        aligned_requests = {}

        for symbol, exposure in decision.target_exposure.items():
            normalized_symbol = (
                str(symbol).strip().upper()
            )

            exposure = float(exposure)

            if abs(exposure) <= 1e-12:
                continue

            request = normalized_requests.get(
                normalized_symbol
            )

            if request is None:
                raise ValueError(
                    f"Missing margin request for projected "
                    f"symbol {normalized_symbol!r}."
                )

            aligned_requests[
                normalized_symbol
            ] = replace(
                request,
                target_exposure=exposure,
            )

        return self.projector.project(
            context=context,
            decision=decision,
            margin_requests=aligned_requests,
        )

    # =========================================================================
    # Reject
    # =========================================================================

    def _reject(
        self,
        request: RiskRequest,
        violation: RiskViolation,
        extra_violations: Tuple[RiskViolation, ...] = (),
    ) -> RiskDecision:

        violations = (
            tuple(extra_violations)
            + (violation,)
        )

        decision = request.decision

        result = RiskDecision(
            status=RiskDecisionStatus.REJECTED,
            timestamp=decision.timestamp,
            mode=decision.mode,
            decision_id=decision.decision_id,
            source_decision_id=decision.decision_id,
            capital_allocation={},
            margin_allocation={},
            target_exposure={},
            target_signals={},
            portfolio_risk=float(decision.portfolio_risk),
            violations=violations,
            model=decision.model,
            metadata={
                "evaluation_path": "projected",
            },
        )

        self.state.record(
            status=RiskDecisionStatus.REJECTED,
            decision_id=result.decision_id,
            violation_codes=tuple(v.code for v in violations),
            timestamp=result.timestamp,
        )

        return result


    # =========================================================================
    # Audit modification
    # =========================================================================

    @staticmethod
    def _build_audit_modifications(
        request: RiskRequest,
        result: RiskDecision,
    ) -> Tuple[RiskAuditModification, ...]:
        """
        Build coarse requested-to-final modification lineage from completed
        Risk violations.

        The current engine exposes requested and final states plus violation
        codes, but does not yet retain every intermediate state between
        constraints. Therefore this method records the final observable delta
        associated with each modifying violation.
        """

        decision = request.decision

        modifications: list[RiskAuditModification] = []

        for violation in result.violations:
            code = str(violation.code).strip().upper()
            symbol = violation.symbol

            if code == "MAX_SYMBOL_EXPOSURE":
                if not symbol:
                    continue

                normalized_symbol = str(symbol).strip().upper()

                requested = float(
                    decision.target_exposure.get(
                        normalized_symbol,
                        0.0,
                    )
                )

                final = float(
                    result.target_exposure.get(
                        normalized_symbol,
                        0.0,
                    )
                )

                if requested != final:
                    modifications.append(
                        RiskAuditModification(
                            constraint_code=code,
                            symbol=normalized_symbol,
                            field="target_exposure",
                            requested_value=requested,
                            final_value=final,
                        )
                    )

            elif code == "MAX_TOTAL_EXPOSURE":
                requested = sum(
                    abs(float(value))
                    for value in decision.target_exposure.values()
                )

                final = sum(
                    abs(float(value))
                    for value in result.target_exposure.values()
                )

                if requested != final:
                    modifications.append(
                        RiskAuditModification(
                            constraint_code=code,
                            symbol=None,
                            field="target_exposure",
                            requested_value=requested,
                            final_value=final,
                        )
                    )

            elif code == "MAX_SYMBOL_CONCENTRATION":
                if not symbol:
                    continue

                normalized_symbol = str(symbol).strip().upper()

                requested = float(
                    decision.capital_allocation.get(
                        normalized_symbol,
                        0.0,
                    )
                )

                final = float(
                    result.capital_allocation.get(
                        normalized_symbol,
                        0.0,
                    )
                )

                if requested != final:
                    modifications.append(
                        RiskAuditModification(
                            constraint_code=code,
                            symbol=normalized_symbol,
                            field="capital_allocation",
                            requested_value=requested,
                            final_value=final,
                        )
                    )

            elif code == "MAX_MARGIN_UTILIZATION":
                requested_exposure = sum(
                    abs(float(value))
                    for value in decision.target_exposure.values()
                )

                final_exposure = sum(
                    abs(float(value))
                    for value in result.target_exposure.values()
                )

                if requested_exposure != final_exposure:
                    modifications.append(
                        RiskAuditModification(
                            constraint_code=code,
                            symbol=None,
                            field="target_exposure",
                            requested_value=requested_exposure,
                            final_value=final_exposure,
                        )
                    )

                requested_capital = sum(
                    abs(float(value))
                    for value in decision.capital_allocation.values()
                )

                final_capital = sum(
                    abs(float(value))
                    for value in result.capital_allocation.values()
                )

                if requested_capital != final_capital:
                    modifications.append(
                        RiskAuditModification(
                            constraint_code=code,
                            symbol=None,
                            field="capital_allocation",
                            requested_value=requested_capital,
                            final_value=final_capital,
                        )
                    )

                requested_margin = sum(
                    abs(float(value))
                    for value in decision.margin_allocation.values()
                )

                final_margin = sum(
                    abs(float(value))
                    for value in result.margin_allocation.values()
                )

                if requested_margin != final_margin:
                    modifications.append(
                        RiskAuditModification(
                            constraint_code=code,
                            symbol=None,
                            field="margin_allocation",
                            requested_value=requested_margin,
                            final_value=final_margin,
                        )
                    )

            elif code == "STOP_LOSS_RISK_CAP":
                if not symbol:
                    continue

                normalized_symbol = str(symbol).strip().upper()

                requested = float(
                    decision.target_exposure.get(
                        normalized_symbol,
                        0.0,
                    )
                )

                final = float(
                    result.target_exposure.get(
                        normalized_symbol,
                        0.0,
                    )
                )

                if requested != final:
                    modifications.append(
                        RiskAuditModification(
                            constraint_code=code,
                            symbol=normalized_symbol,
                            field="target_exposure",
                            requested_value=requested,
                            final_value=final,
                        )
                    )

        return tuple(modifications)

    # =========================================================================
    # Audit
    # ========================================================================= new method

    def _record_audit(
        self,
        *,
        request: RiskRequest,
        result: RiskDecision,
    ) -> None:
        """
        Record one immutable audit snapshot for a completed Risk evaluation.

        RiskEngineState remains runtime statistics only. RiskAuditTrail stores
        one individual audit record per completed evaluate() call.
        """

        decision = request.decision

        evaluation_path = str(
            result.metadata.get(
                "evaluation_path",
                "unknown",
            )
        )

        modifications = self._build_audit_modifications(
            request,
            result,
        )

        record = RiskAuditRecord(
            audit_id=f"audit-{uuid4().hex}",
            timestamp=result.timestamp,
            mode=result.mode,
            source_decision_id=decision.decision_id,
            risk_decision_id=result.decision_id,
            status=result.status,
            model=result.model,
            requested_target_exposure=dict(
                decision.target_exposure
            ),
            final_target_exposure=dict(
                result.target_exposure
            ),
            requested_capital_allocation=dict(
                decision.capital_allocation
            ),
            final_capital_allocation=dict(
                result.capital_allocation
            ),
            requested_margin_allocation=dict(
                decision.margin_allocation
            ),
            final_margin_allocation=dict(
                result.margin_allocation
            ),
            requested_portfolio_risk=float(
                decision.portfolio_risk
            ),
            final_portfolio_risk=float(
                result.portfolio_risk
            ),
            violations=tuple(
                result.violations
            ),
            evaluation_path=evaluation_path,
            modifications=modifications,
            stop_loss_requests=dict(
                request.stop_loss_requests
            ),
            stop_loss_results=dict(
                result.stop_loss_results
            ),
        )

        self.audit_trail.append(record)

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _cap_signed(
        value: float,
        limit: float,
    ) -> float:

        if value > limit:
            return limit

        if value < -limit:
            return -limit

        return value

    @staticmethod
    def _projected_summary(
        projected: ProjectedRiskState,
    ) -> Mapping[str, object]:

        return {
            "current_total_exposure": projected.current_total_exposure,
            "projected_total_exposure": projected.projected_total_exposure,
            "current_used_margin": projected.current_used_margin,
            "projected_used_margin": projected.projected_used_margin,
            "projected_free_margin": projected.projected_free_margin,
            "projected_margin_utilization": projected.projected_margin_utilization,
            "projected_position_count": projected.projected_position_count,
            "symbols": {
                symbol: {
                    "current_exposure": snapshot.current_exposure,
                    "target_exposure": snapshot.target_exposure,
                    "delta": snapshot.projected_exposure_delta,
                    "current_notional": snapshot.current_notional,
                    "projected_notional": snapshot.projected_notional,
                    "current_used_margin": snapshot.current_used_margin,
                    "projected_used_margin": snapshot.projected_used_margin,
                    "current_position_count": snapshot.current_position_count,
                    "projected_position_count": snapshot.projected_position_count,
                }
                for symbol, snapshot
                in projected.symbols.items()
            },
        }

    # =========================================================================
    # Stop-Loss / Exposure Alignment
    # =========================================================================

    @staticmethod
    def _validate_stop_loss_alignment(
        request: RiskRequest,
    ) -> None:
        """
        Validate consistency between signed target exposure and stop-loss prices.

        Long:
            target_exposure > 0
            stop_price < entry_price

        Short:
            target_exposure < 0
            stop_price > entry_price

        Flat:
            target_exposure == 0
            no active stop-loss request is allowed.
        """

        for symbol, sl_request in request.stop_loss_requests.items():
            normalized_symbol = str(symbol).strip().upper()

            exposure = float(
                request.decision.target_exposure.get(
                    normalized_symbol,
                    0.0,
                )
            )

            entry_price = float(
                sl_request.entry_price
            )
            stop_price = float(
                sl_request.stop_price
            )

            if abs(exposure) <= 1e-12:
                raise ValueError(
                    f"Stop-loss request for flat symbol "
                    f"{normalized_symbol!r} is not allowed."
                )

            if exposure > 0.0 and stop_price >= entry_price:
                raise ValueError(
                    f"Stop-loss for long exposure must be below "
                    f"entry price for {normalized_symbol}."
                )

            if exposure < 0.0 and stop_price <= entry_price:
                raise ValueError(
                    f"Stop-loss for short exposure must be above "
                    f"entry price for {normalized_symbol}."
                )

    # =========================================================================
    # Stop-Loss Risk Sizing
    # =========================================================================

    def _calculate_stop_loss_results(
        self,
        request: RiskRequest,
    ) -> Mapping[
        str,
        StopLossPositionSizingResult,
    ]:
        """
        Calculate deterministic stop-loss risk sizing for all requested symbols.

        This method does not modify target_exposure, target_signals,
        capital_allocation, or margin_allocation.

        Stop-loss sizing is therefore an additional risk-sizing result,
        while exposure remains the authoritative portfolio risk representation.
        """
        results: dict[
            str,
            StopLossPositionSizingResult,
        ] = {}
        for symbol in sorted(request.stop_loss_requests):
            sl_request = request.stop_loss_requests[symbol]

            result = self.stop_loss_calculator.calculate(
                sl_request
            )
            results[symbol] = result
        return results
    

    # =========================================================================
    # Stop-Loss Risk Cap
    # =========================================================================

    def _apply_stop_loss_risk_caps(
        self,
        request: RiskRequest,
        candidate: PortfolioDecision,
    ) -> tuple[PortfolioDecision, list[RiskViolation], bool]:
        """
        Limit target exposure by the maximum position size permitted by the
        requested stop-loss risk budget.

        Exposure remains the authoritative portfolio representation.

        The stop-loss calculator defines the maximum lot volume that may be
        exposed to the configured loss budget. That volume is converted back
        into an equivalent exposure cap and compared with the candidate's
        absolute target exposure.
        """

        target_exposure = dict(
            candidate.target_exposure
        )

        violations: list[RiskViolation] = []
        modified = False

        for symbol in sorted(request.stop_loss_requests):
            normalized_symbol = str(symbol).strip().upper()

            if normalized_symbol not in target_exposure:
                continue

            exposure = float(
                target_exposure[normalized_symbol]
            )

            if abs(exposure) <= 1e-12:
                continue

            sl_request = request.stop_loss_requests[
                symbol
            ]

            sl_result = self.stop_loss_calculator.calculate(
                sl_request
            )

            exposure_request = PositionSizingRequest(
                symbol=normalized_symbol,
                target_exposure=abs(exposure),
                equity=sl_request.equity,
                price=sl_request.entry_price,
                contract_size=sl_request.contract_size,
                currency_conversion_rate=(
                    sl_request.currency_conversion_rate
                ),
            )

            exposure_result = (
                self.position_sizing_calculator.calculate(
                    exposure_request
                )
            )

            requested_volume = (
                exposure_result.volume_lots
            )

            allowed_volume = (
                sl_result.volume_lots
            )

            if requested_volume <= allowed_volume:
                continue

            if requested_volume <= 0.0:
                continue

            factor = (
                allowed_volume
                / requested_volume
            )

            factor = min(
                max(factor, 0.0),
                1.0,
            )

            target_exposure[normalized_symbol] = (
                exposure * factor
            )

            violations.append(
                RiskViolation(
                    code="STOP_LOSS_RISK_CAP",
                    message=(
                        f"Target exposure for {normalized_symbol} "
                        "exceeds the maximum exposure compatible "
                        "with the configured stop-loss risk budget."
                    ),
                    symbol=normalized_symbol,
                    severity="warning",
                )
            )

            modified = True

        if not modified:
            return candidate, violations, False

        candidate = replace(
            candidate,
            target_exposure=target_exposure,
        )

        return candidate, violations, True

    # =========================================================================
    # State
    # =========================================================================
    
    def reset(self) -> None:
        self.state.reset()

# ============================================================================= END