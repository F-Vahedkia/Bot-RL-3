# f07_agents/action_builder.py (between 5,6)
#
# Created: 1405/06/19

# Chapter 3 - Symbol-Agent + Meta-Agent
# تبدیل PortfolioDecision به PortfolioAction معنایی.
#
# این فایل:
#     - broker را نمی‌شناسد
#     - order ارسال نمی‌کند
#     - execution انجام نمی‌دهد
#
# فقط مرز بین Decision Layer و Environment Layer است.
#
# Architecture:
#
#     Meta-Agent
#          |
#          v
#     PortfolioDecision
#          |
#          v
#     PortfolioActionBuilder
#          |
#          v
#     f06_env.PortfolioAction
#
# نکته:
#     تبدیل exposure به lots به یک sizing policy نیاز دارد.
#     بنابراین lot sizing به صورت dependency injection انجام می‌شود.


from __future__ import annotations

from typing import Mapping, Protocol

from f06_env.contracts import (
    PortfolioAction,
    PositionIntent,
)
from f07_agents.contracts import (
    PortfolioDecision,
)

# =============================================================================
# Class-1: Position Sizing Interface
# =============================================================================

class PositionSizer(Protocol):
    """
    قرارداد تبدیل تصمیم exposure به target_lots.

    این interface عمداً از broker و execution مستقل است.

    در آینده می‌تواند بر اساس:
        - equity
        - volatility
        - contract specification
        - risk budget
        - leverage
        - stop distance
        - instrument margin
    حجم را محاسبه کند.
    """
    def size(
        self,
        *,
        symbol: str,
        target_exposure: float,
    ) -> float:
        ...

# =============================================================================
# Class-2: Portfolio Action Builder
# =============================================================================

class PortfolioActionBuilder:
    """
    تبدیل PortfolioDecision به PortfolioAction.
    این کلاس فقط semantic translation انجام می‌دهد.
    """
    def __init__(
        self,
        *,
        position_sizer: PositionSizer,
    ) -> None:

        self.position_sizer = position_sizer

    # ===========================================
    # Build
    # ===========================================

    def build(
        self,
        *,
        decision: PortfolioDecision,
        allowed_symbols: set[str] | None = None,
    ) -> PortfolioAction:
        """
        ساخت PortfolioAction از تصمیم Meta-Agent.

        اگر decision رد شده باشد، action خالی تولید می‌شود.

        allowed_symbols:
            مجموعه نمادهای مجاز در Environment.

        Symbol identity policy:
            فقط فاصله‌ها حذف می‌شوند.
            هیچ upper/lower normalization انجام نمی‌شود.
        """

        if not decision.approved:
            return PortfolioAction()

        allowed = (
            None
            if allowed_symbols is None
            else {
                str(symbol).replace(" ", "")
                for symbol in allowed_symbols
            }
        )

        normalized_signals: dict[str, int] = {}
        normalized_exposures: dict[str, float] = {}
        normalized_stop_prices: dict[str, float | None] = {}

        for symbol, side in decision.target_signals.items():
            normalized_symbol = str(symbol).replace(" ", "")

            if normalized_symbol in normalized_signals:
                raise ValueError(
                    "Decision contains duplicate symbols after "
                    f"space removal: {normalized_symbol!r}"
                )

            normalized_signals[normalized_symbol] = int(side)

            normalized_exposures[normalized_symbol] = float(
                decision.target_exposure.get(
                    symbol,
                    0.0,
                )
            )

            normalized_stop_prices[normalized_symbol] = (
                decision.target_stop_price.get(
                    symbol,
                    None,
                )
            )

        symbols = set(normalized_signals)

        if allowed is not None:
            unknown = symbols - allowed

            if unknown:
                raise ValueError(
                    "Decision contains symbols not allowed by "
                    f"Environment: {sorted(unknown)}"
                )

        intents: list[PositionIntent] = []

        for normalized_symbol in sorted(symbols):
            side = normalized_signals[normalized_symbol]

            exposure = normalized_exposures[
                normalized_symbol
            ]
            stop_price = normalized_stop_prices[
                normalized_symbol
            ]

            if side == 0:
                lots = 0.0
                stop_price = None
            else:
                lots = float(
                    self.position_sizer.size(
                        symbol=normalized_symbol,
                        target_exposure=abs(exposure),
                    )
                )

            if lots < 0.0:
                raise ValueError(
                    f"PositionSizer returned negative "
                    f"lots for {normalized_symbol}"
                )

            if side == 0 and lots != 0.0:
                raise ValueError(
                    f"Flat decision produced non-zero "
                    f"lots for {normalized_symbol}"
                )

            intents.append(
                PositionIntent(
                    symbol=normalized_symbol,
                    target_side=side,
                    target_lots=lots,
                    stop_price=stop_price,
                )
            )

        return PortfolioAction(
            intents=tuple(intents)
        )
    
# ============================================================================= END