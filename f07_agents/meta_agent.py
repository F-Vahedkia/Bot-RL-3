# f07_agents/meta_agent.py (4)
#
# Created: 1405/06/19

# Production-grade Meta-Agent foundation.
#
# مسئولیت:
#     - دریافت خروجی Symbol-Agentها
#     - مشاهده وضعیت کل Portfolio
#     - اعمال Portfolio guardrails
#     - کنترل:
#           capital allocation
#           margin allocation
#           exposure
#           concentration
#           correlation
#           portfolio risk
#           drawdown
#
# عدم مسئولیت:
#     - broker
#     - order execution
#     - MT5
#     - market data
#     - feature calculation
#
# Meta-Agent خروجی Decision تولید می‌کند.
# اتصال این Decision به f06_env در یک Adapter مستقل انجام خواهد شد.


from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Protocol

from f07_agents.agent_state import (
    MetaAgentState,
)
from f07_agents.contracts import (
    DecisionMode,
    MetaPolicyOutput,
    ModelIdentity,
    PortfolioContext,
    PortfolioDecision,
    SymbolAgentOutput,
)

# =============================================================================
# Class-1: Meta Policy Interface
# =============================================================================

class MetaPolicy(Protocol):
    """
    قرارداد Policy سطح Portfolio.

    این Policy می‌تواند در آینده:
        - RL
        - Transformer
        - Attention
        - Ensemble
    باشد.
    """
    def predict(
        self,
        *,
        signals: Mapping[str, SymbolAgentOutput],
        portfolio: PortfolioContext,
    ) -> MetaPolicyOutput:
        ...


# =============================================================================
# Class-2: Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class MetaAgentConfig:
    """
    محدودیت‌های Portfolio-level.
    اینها hard guardrail هستند، نه Policy.
    """

    mode: DecisionMode
    model: ModelIdentity
    max_total_allocation: float = 1.0   # مجموع سرمایه تخصیص داده‌شده
    max_symbol_allocation: float = 0.25   # حداکثر تخصیص به یک symbol
    max_total_exposure: float = 1.0   # حداکثر exposure کل
    max_portfolio_risk: float = 1.0   # حداکثر risk کل Portfolio

    # اگر correlation مطلق دو نماد از این حد بیشتر شود،
    # آنها به عنوان exposure همبسته شدید در نظر گرفته می‌شوند.
    high_correlation_threshold: float = 0.85

    max_correlated_allocation: float = 0.50   # سقف مجموع allocation دو نماد heavily correlated
    max_drawdown: float = 1.0   # در صورت drawdown بالاتر از این مقدار، تصمیم جدید block می‌شود.


    def __post_init__(self) -> None:

        if not (
            0.0 < self.max_total_allocation <= 1.0
        ):
            raise ValueError("max_total_allocation must be in (0,1]")

        if not (
            0.0 < self.max_symbol_allocation <= 1.0
        ):
            raise ValueError("max_symbol_allocation must be in (0,1]")

        if not (
            0.0 < self.max_total_exposure <= 1.0
        ):
            raise ValueError("max_total_exposure must be in (0,1]")

        if not (
            0.0 <= self.max_portfolio_risk <= 1.0
        ):
            raise ValueError("max_portfolio_risk must be in [0,1]")

        if not (
            0.0 <= self.high_correlation_threshold <= 1.0
        ):
            raise ValueError("high_correlation_threshold must be in [0,1]")

        if not (
            0.0 < self.max_correlated_allocation <= 1.0
        ):
            raise ValueError("max_correlated_allocation must be in (0,1]")

        if not (
            0.0 <= self.max_drawdown <= 1.0
        ):
            raise ValueError("max_drawdown must be in [0,1]")


# =============================================================================
# Class-3: Meta-Agent
# =============================================================================

class MetaAgent:
    """
    Meta-Agent سطح Portfolio.

    Symbol-Agentها فقط پیشنهاد می‌دهند.
    تصمیم نهایی Portfolio در این کلاس شکل می‌گیرد.

    
    """
    def __init__(
        self,
        *,
        config: MetaAgentConfig,
        policy: MetaPolicy,
    ) -> None:

        self.config = config
        self.policy = policy
        self.state = MetaAgentState()

    # ===========================================
    # Decision
    # ===========================================

    def decide(
        self,
        *,
        signals: Mapping[str, SymbolAgentOutput],
        portfolio: PortfolioContext,
        decision_id: str,
    ) -> PortfolioDecision:
        decision_id = str(decision_id).strip()

        if not decision_id:
            raise ValueError("decision_id is required.")

        try:
            self._validate_signal_set(signals)

            if portfolio.risk_blocked:
                return self._rejected_decision(
                    portfolio=portfolio,
                    decision_id=decision_id,
                    reason_code="portfolio_risk_blocked",
                )

            if portfolio.drawdown >= self.config.max_drawdown:
                return self._rejected_decision(
                    portfolio=portfolio,
                    decision_id=decision_id,
                    reason_code="max_drawdown_reached",
                )

            proposal = self.policy.predict(
                signals=signals,
                portfolio=portfolio,
            )

            capital_allocation = dict(proposal.capital_allocation)
            margin_allocation = dict(proposal.margin_allocation)
            target_signals = dict(proposal.target_signals)
            target_exposure = dict(proposal.target_exposure)
            (
                target_stop_price,
                stop_price_reason_codes,
            ) = self._resolve_target_stop_prices(
                signals=signals,
                target_signals=target_signals,
                override=proposal.target_stop_price,
            )

            # -----------------------------------------------------------------
            # Guardrail 1: per-symbol capital allocation
            # -----------------------------------------------------------------
            capital_allocation = self._apply_symbol_cap(capital_allocation)

            # -----------------------------------------------------------------
            # Guardrail 2: correlation allocation cap
            # -----------------------------------------------------------------
            capital_allocation = self._apply_correlation_cap(
                allocations=capital_allocation,
                correlation=portfolio.correlation,
            )

            # -----------------------------------------------------------------
            # Guardrail 3: total capital allocation
            # -----------------------------------------------------------------
            capital_allocation = self._scale_to_limit(
                values=capital_allocation,
                limit=self.config.max_total_allocation,
            )

            # -----------------------------------------------------------------
            # Guardrail 4: total exposure
            # -----------------------------------------------------------------
            target_exposure = self._scale_to_limit(
                values=target_exposure,
                limit=self.config.max_total_exposure,
            )

            # -----------------------------------------------------------------
            # Guardrail 5: portfolio risk
            # -----------------------------------------------------------------
            portfolio_risk = min(
                max(
                    float(proposal.portfolio_risk),
                    0.0,
                ),
                self.config.max_portfolio_risk,
            )

            # -----------------------------------------------------------------
            # Margin allocation follows final capital allocation.
            # -----------------------------------------------------------------
            margin_allocation = self._resize_mapping(
                values=margin_allocation,
                reference=capital_allocation,
            )

            reason_codes = tuple(
                dict.fromkeys(
                    tuple(proposal.reason_codes)
                    + tuple(stop_price_reason_codes)
                )
            )

            total_exposure = sum(
                abs(float(value))
                for value in target_exposure.values()
            )

            self.state.record_decision(
                approved=True,
                equity=portfolio.equity,
                drawdown=portfolio.drawdown,
                total_exposure=total_exposure,
                capital_allocation=capital_allocation,
                margin_allocation=margin_allocation,
                target_stop_price=target_stop_price,
                timestamp=portfolio.timestamp,
                model=self.config.model,
                mode=self.config.mode,
            )

            return PortfolioDecision(
                approved=True,
                timestamp=portfolio.timestamp,
                mode=self.config.mode,
                decision_id=decision_id,
                capital_allocation=capital_allocation,
                margin_allocation=margin_allocation,
                target_exposure=target_exposure,
                target_signals=target_signals,
                portfolio_risk=portfolio_risk,
                reason_codes=reason_codes,
                model=self.config.model,
                target_stop_price=target_stop_price,
            )

        except Exception as exc:
            self.state.record_error(exc)
            raise

    # ===========================================
    # ???????
    # ===========================================

    def _resolve_target_stop_prices(
        self,
        *,
        signals: Mapping[str, SymbolAgentOutput],
        target_signals: Mapping[str, int],
        override: Mapping[str, float | None],
    ) -> tuple[
        dict[str, float | None],
        tuple[str, ...],
    ]:
        """
        Resolve portfolio-level stop prices.

        Rules:

            1. No Meta-Agent override:
                inherit Symbol-Agent stop_price.

            2. Explicit float override:
                replace Symbol-Agent stop_price.

            3. Explicit None override:
                remove stop_price intentionally.

            4. Final flat target signal:
                no active stop_price is retained.
        """

        result: dict[str, float | None] = {}
        reason_codes: list[str] = []

        for symbol, signal in signals.items():
            target_signal = int(
                target_signals.get(symbol, 0)
            )

            if target_signal == 0:
                result[symbol] = None
                continue

            if symbol in override:
                stop_price = override[symbol]

                if stop_price is None:
                    result[symbol] = None
                    reason_codes.append(
                        "meta_stop_price_explicitly_removed"
                    )
                else:
                    result[symbol] = float(stop_price)
                    reason_codes.append(
                        "meta_stop_price_overridden"
                    )

                continue

            result[symbol] = signal.stop_price

        return result, tuple(
            dict.fromkeys(reason_codes)
        )


    # ===========================================
    # Validation
    # ===========================================

    def _validate_signal_set(
        self,
        signals: Mapping[str, SymbolAgentOutput],
    ) -> None:
        if not signals:
            raise ValueError(
                "signals must not be empty."
            )

        for key, signal in signals.items():
            normalized_key = str(key).replace(" ", "")
            signal_symbol = str(
                signal.symbol
            ).replace(" ", "")

            if normalized_key != signal_symbol:
                raise ValueError(
                    "Signal mapping key does not match "
                    f"signal.symbol: "
                    f"key={key!r}, "
                    f"symbol={signal.symbol!r}"
                )


    # ===========================================
    # Symbol concentration cap
    # ===========================================

    def _apply_symbol_cap(
        self,
        values: Mapping[str, float],
    ) -> dict[str, float]:

        return {
            symbol: min(
                max(0.0, float(value)),
                self.config.max_symbol_allocation,
            )
            for symbol, value in values.items()
        }

    # ===========================================
    # Total limit scaling
    # ===========================================

    @staticmethod
    def _scale_to_limit(
        *,
        values: Mapping[str, float],
        limit: float,
    ) -> dict[str, float]:

        total = sum(
            max(0.0, float(value))
            for value in values.values()
        )

        if total <= limit:
            return dict(values)

        factor = limit / total

        return {
            symbol: max(0.0, float(value)) * factor
            for symbol, value in values.items()
        }

    # ===========================================
    # Correlation cap
    # ===========================================

    def _apply_correlation_cap(
        self,
        *,
        allocations: Mapping[str, float],
        correlation: Mapping[
            str,
            Mapping[str, float],
        ],
    ) -> dict[str, float]:

        result = {
            symbol: max(0.0, float(value))
            for symbol, value in allocations.items()
        }

        for symbol_a, symbol_b in combinations(result.keys(),2):

            corr_ab = float(
                correlation
                .get(symbol_a, {})
                .get(symbol_b, 0.0)
            )
            corr_ba = float(
                correlation
                .get(symbol_b, {})
                .get(symbol_a, corr_ab)
            )

            correlation_value = max(abs(corr_ab), abs(corr_ba))
            if correlation_value < self.config.high_correlation_threshold:
                continue

            combined = result[symbol_a] + result[symbol_b]
            if combined <= self.config.max_correlated_allocation:
                continue

            factor = self.config.max_correlated_allocation / combined
            result[symbol_a] *= factor
            result[symbol_b] *= factor

        return result

    # ===========================================
    # Margin Allocation
    # ===========================================

    @staticmethod
    def _resize_mapping(
        *,
        values: Mapping[str, float],
        reference: Mapping[str, float],
    ) -> dict[str, float]:

        reference_total = sum(
            max(0.0, value)
            for value in reference.values()
        )

        source_total = sum(
            max(0.0, float(value))
            for value in values.values()
        )

        if reference_total <= 0.0 or source_total <= 0.0:
            return {
                symbol: 0.0
                for symbol in reference
            }

        result: dict[str, float] = {}
        for symbol, ref_value in reference.items():

            original = max(
                0.0,
                float(values.get(symbol, 0.0)),
            )
            result[symbol] = (original * reference_total / source_total)

        return result

    # ===========================================
    # Rejection
    # ===========================================

    def _rejected_decision(
        self,
        *,
        portfolio: PortfolioContext,
        decision_id: str,
        reason_code: str,
    ) -> PortfolioDecision:
        self.state.record_decision(
            approved=False,
            equity=portfolio.equity,
            drawdown=portfolio.drawdown,
            total_exposure=0.0,
            capital_allocation={},
            margin_allocation={},
            target_stop_price={},
            timestamp=portfolio.timestamp,
            model=self.config.model,
            mode=self.config.mode,
            rejection_reason=reason_code,
        )
        return PortfolioDecision(
            approved=False,
            timestamp=portfolio.timestamp,
            mode=self.config.mode,
            decision_id=decision_id,
            capital_allocation={},
            margin_allocation={},
            target_exposure={},
            target_signals={},
            portfolio_risk=0.0,
            reason_codes=(reason_code,),
            model=self.config.model,
            target_stop_price={},
        )

    # ===========================================
    # Lifecycle
    # ===========================================

    def update_reward(self, reward: float) -> None:
        self.state.add_reward(reward)


    def reset(self) -> None:
        self.state.reset()

# ============================================================================= END