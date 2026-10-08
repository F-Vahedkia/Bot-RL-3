# f07_agents/portfolio_context_adapter.py (6)
#
# Created: 1405/06/19

# Chapter 3 - Symbol-Agent + Meta-Agent
# هدف:
#     تبدیل PortfolioState واقعی f06_env به PortfolioContext
#     مورد استفاده Meta-Agent.
#
# این فایل:
#     - هیچ accounting انجام نمی‌دهد.
#     - PnL را محاسبه نمی‌کند.
#     - margin را محاسبه نمی‌کند.
#     - risk را محاسبه نمی‌کند.
#
# فقط state موجود را به قرارداد Decision Layer منتقل می‌کند.
#
# Architecture:
#
#     f06_env.PortfolioState
#             |
#             v
#     PortfolioContextAdapter
#             |
#             v
#     f07_agents.PortfolioContext
#             |
#             v
#         Meta-Agent
#
# exposure و correlation عمداً از خارج دریافت می‌شوند،
# زیرا معنای دقیق آنها به مدل ریسک و بازار پروژه وابسته است
# و نباید در این Adapter حدس زده شود.


from __future__ import annotations

from typing import Mapping

from f06_env.portfolio_state import (
    PortfolioState,
)
from f07_agents.contracts import (
    DecisionMode,
    PortfolioContext,
)


# =============================================================================
# Class-1: Portfolio Context Adapter
# =============================================================================

class PortfolioContextAdapter:
    """
    Adapter استاندارد بین PortfolioState و Meta-Agent.

    این کلاس فقط translation انجام می‌دهد.
    """

    @staticmethod
    def from_portfolio_state(
        *,
        portfolio: PortfolioState,
        timestamp,
        mode: DecisionMode,
        exposure: Mapping[str, float],
        correlation: Mapping[
            str,
            Mapping[str, float],
        ] | None = None,
        concentration: Mapping[str, float] | None = None,
        risk_blocked: bool = False,
    ) -> PortfolioContext:
        """
        ساخت PortfolioContext از PortfolioState.

        نکته:
            exposure و correlation محاسبه نمی‌شوند؛
            چون این دو مفهوم باید توسط لایه تخصصی مربوطه تولید شوند.
        """

        if not isinstance(
            portfolio,
            PortfolioState,
        ):
            raise TypeError("portfolio must be PortfolioState")

        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        normalized_exposure = {
            str(symbol).replace(" ",""): float(value)
            for symbol, value
            in exposure.items()
        }

        normalized_concentration = {
            str(symbol).replace(" ",""): float(value)
            for symbol, value
            in (concentration or {}).items()
        }

        normalized_correlation = {
            str(symbol_a).replace(" ",""): {
                str(symbol_b).replace(" ",""): float(value)
                for symbol_b, value
                in row.items()
            }
            for symbol_a, row
            in (correlation or {}).items()
        }

        return PortfolioContext(
            timestamp=timestamp,
            equity=float(portfolio.equity),
            balance=float(portfolio.balance),
            used_margin=float(portfolio.used_margin),
            free_margin=float(portfolio.free_margin),
            margin_level=portfolio.margin_level,
            drawdown=float(portfolio.total_drawdown),
            daily_drawdown=float(
                portfolio.daily_drawdown
            ),
            exposure=normalized_exposure,
            concentration=normalized_concentration,
            correlation=normalized_correlation,
            risk_blocked=bool(risk_blocked),
            mode=mode,
        )

# ============================================================================= END