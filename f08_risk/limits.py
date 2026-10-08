# f08_risk/limits.py (2)
#
# Created: 1405/06/19

# Risk limits / hard constraints.
#
# این فایل فقط configuration محدودیت‌های Risk Layer است.
"""
یادداشت خودم:
    - None یعنی policy برای leverage حساب تعریف نشده است.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True, slots=True)
class RiskLimits:
    """
    محدودیت‌های سطح Portfolio.

    این مقادیر بعداً می‌توانند بخشی از
    self-optimization باشند، اما هنگام اجرای
    live باید immutable باشند.
    """

    # حداکثر exposure کل Portfolio
    max_total_exposure: float = 1.0

    # حداکثر exposure یک symbol
    max_symbol_exposure: float = 0.50

    # حداکثر concentration یک symbol
    max_symbol_concentration: float = 0.50

    # حداکثر مجموع margin utilization
    max_margin_utilization: float = 0.80

    # حداکثر drawdown
    max_drawdown: float = 0.20

    # حداکثر daily drawdown
    max_daily_drawdown: float = 0.05

    # حداکثر portfolio risk score
    max_portfolio_risk: float = 0.50

    #--------------------
    # حداکثر leverage مجاز در سطح حساب
    # None یعنی این hard guard غیرفعال است.
    max_account_leverage: float | None = None
    #--------------------

    # correlation threshold
    high_correlation_threshold: float = 0.85

    # حداکثر exposure مجموع نمادهای heavily correlated
    max_correlated_exposure: float = 0.50

    # حداکثر تعداد کل پوزیشن‌های باز در Portfolio
    max_open_positions: int | None = None

    # حداکثر تعداد پوزیشن‌های باز برای هر symbol
    max_positions_per_symbol: int | None = None


    def __post_init__(self) -> None:

        values = {
            "max_total_exposure": self.max_total_exposure,
            "max_symbol_exposure": self.max_symbol_exposure,
            "max_symbol_concentration": self.max_symbol_concentration,
            "max_margin_utilization": self.max_margin_utilization,
            "max_drawdown": self.max_drawdown,
            "max_daily_drawdown": self.max_daily_drawdown,
            "max_portfolio_risk": self.max_portfolio_risk,
            "high_correlation_threshold": self.high_correlation_threshold,
            "max_correlated_exposure": self.max_correlated_exposure,
        }

        for name, value in values.items():
            value = float(value)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")
            
            if value > 1.0:
                raise ValueError(f"{name} must be <= 1")

        if self.max_open_positions is not None:
            if (
                not isinstance(self.max_open_positions, int)
                or isinstance(self.max_open_positions, bool)
            ):
                raise TypeError("max_open_positions must be an integer or None.")

            if self.max_open_positions < 0:
                raise ValueError("max_open_positions must be >= 0.")

        if self.max_positions_per_symbol is not None:
            if (
                not isinstance(self.max_positions_per_symbol, int)
                or isinstance(self.max_positions_per_symbol, bool)
            ):
                raise TypeError("max_positions_per_symbol must be an integer or None.")

            if self.max_positions_per_symbol < 0:
                raise ValueError("max_positions_per_symbol must be >= 0.")

        if self.max_account_leverage is not None:
            if not isfinite(float(self.max_account_leverage)):
                raise ValueError("max_account_leverage must be finite or None.")

            if self.max_account_leverage <= 0.0:
                raise ValueError("max_account_leverage must be > 0.")

# ============================================================================= END
