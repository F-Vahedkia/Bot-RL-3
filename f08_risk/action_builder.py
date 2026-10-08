# f08_risk/action_builder.py (5)
#
# Created: 1405/06/19

# RiskDecision -> PortfolioAction
#
# Execution در این فایل انجام نمی‌شود.


from __future__ import annotations
from typing import Protocol

from f06_env.contracts import (
    PortfolioAction,
    PositionIntent,
)
from f08_risk.contracts import (
    RiskDecision,
    RiskDecisionStatus,
)


class RiskPositionSizer(Protocol):
    """
    تبدیل exposure نهایی Risk Layer به target_lots.
    در آینده می‌تواند به instrument specification و risk-based sizing واقعی متصل شود.
    """

    def size(
        self,
        *,
        symbol: str,
        target_exposure: float,
    ) -> float:
        ...


class RiskActionBuilder:
    """
    تبدیل RiskDecision به PortfolioAction.

    این کلاس:
        - execution نمی‌کند
        - broker را نمی‌شناسد
    """

    def __init__(
        self,
        *,
        position_sizer: RiskPositionSizer,
    ) -> None:

        self.position_sizer = position_sizer


    def build(
        self,
        *,
        decision: RiskDecision,
        allowed_symbols: set[str] | None = None,
    ) -> PortfolioAction:
        """
        ساخت Action فقط برای تصمیم Approved یا Modified.
        """

        if decision.status == RiskDecisionStatus.REJECTED:
            return PortfolioAction()

        allowed = (
            None
            if allowed_symbols is None
            else {
                str(symbol).upper().strip()
                for symbol in allowed_symbols
            }
        )

        symbols = (
            set(decision.target_signals)
            |
            set(decision.target_exposure)
        )


        if allowed is not None:
            unknown = symbols - allowed
            if unknown:
                raise ValueError(
                    "Risk decision contains symbols "
                    f"outside allowed set: {sorted(unknown)}"
                )

        intents: list[PositionIntent] = []

        for symbol in sorted(symbols):
            side = int(
                decision.target_signals.get(
                    symbol,
                    0,
                )
            )

            if side == 0:
                lots = 0.0
            else:
                exposure = abs(
                    float(
                        decision.target_exposure.get(
                            symbol,
                            0.0,
                        )
                    )
                )
                lots = float(
                    self.position_sizer.size(
                        symbol=symbol,
                        target_exposure=exposure,
                    )
                )

            if lots < 0.0:
                raise ValueError(f"Negative position size for {symbol}")

            intents.append(
                PositionIntent(
                    symbol=symbol,
                    target_side=side,
                    target_lots=lots,
                )
            )

        return PortfolioAction(
            intents=tuple(intents)
        )

# ============================================================================= END