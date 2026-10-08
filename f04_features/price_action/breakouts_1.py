# -*- coding: utf-8 -*-
# f04_features/price_action/breakouts.py
# Status in (Bot-RL-3H): Completed

"""
Price Action — Breakouts
========================
تشخیص شکستِ رِنجِ rolling با تایید، Retest و Fail-break به‌صورت ضد-لوک‌اِهد.

خروجی‌ها:
- range_upper, range_lower
- breakout_up, breakout_down
- retest_up, retest_down
- fail_break_up, fail_break_down
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _rolling_range_bounds(high: pd.Series, low: pd.Series, window: int, min_periods: int) -> tuple[pd.Series, pd.Series]:
    rhi = high.rolling(window=window, min_periods=min_periods).max()
    rlo = low.rolling(window=window, min_periods=min_periods).min()
    return rhi, rlo


def build_breakouts(
    df: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    range_window: int = 20,
    min_periods: int = 10,
    confirm_closes: int = 1,
    retest_lookahead: int = 5,
    fail_break_lookahead: int = 3,
    anti_lookahead: bool = True,
) -> pd.DataFrame:
    """
    شکستِ باندهای رِنج rolling را شناسایی می‌کند با معیارهای:
      - Breakout: close > upper_prev  یا  close < lower_prev (upper/lower شیفت‌شده)
      - Confirm:  n کلوز متوالی تأییدکننده
      - Retest:   لمس مجدد باند شکسته‌شده در بازهٔ تعیین‌شده
      - Fail:     بازگشت سریع داخل رنج پس از شکست

    نکته: برای حذف هرگونه لوک‌اِهد، مقایسه‌ها با باندِ «کندل قبل» انجام می‌شود.
    """
    for col in (high_col, low_col, close_col):
        if col not in df.columns:
            raise ValueError(f"'{col}' not found in DataFrame.")

    out = df.copy()
    h, l, c = out[high_col].astype("float64"), out[low_col].astype("float64"), out[close_col].astype("float64")

    # باندهای رنج rolling
    upper, lower = _rolling_range_bounds(h, l, window=range_window, min_periods=min_periods)

    # *** نکتهٔ مهم: استفاده از باندهای «شیفت‌شده» برای مقایسه شکست ***
    upper_prev = upper.shift(1)
    lower_prev = lower.shift(1)

    # شروط خام شکست
    above_prev = c > upper_prev
    below_prev = c < lower_prev

    # تایید با n کلوز متوالی
    conf_up = (
        above_prev.rolling(confirm_closes, min_periods=confirm_closes)
        .apply(lambda x: 1.0 if bool(np.all(x)) else 0.0, raw=True)
        .fillna(0.0)
    )
    conf_dn = (
        below_prev.rolling(confirm_closes, min_periods=confirm_closes)
        .apply(lambda x: 1.0 if bool(np.all(x)) else 0.0, raw=True)
        .fillna(0.0)
    )

    breakout_up = (conf_up > 0).astype("int8")
    breakout_down = (conf_dn > 0).astype("int8")

    # Retest: لمس مجدد باندِ شکست‌خورده (با باند prev همان لحظه)
    retest_up = pd.Series(0, index=out.index, dtype="int8")
    retest_down = pd.Series(0, index=out.index, dtype="int8")

    for i in range(len(out)):
        if breakout_up.iat[i] == 1:
            hi = min(len(out) - 1, i + retest_lookahead)
            # لمس lower/upper؟ برای up، بازگشت تا نزدیکی upper_prev کافی است
            touched = (l.iloc[i+1:hi+1] <= upper_prev.iloc[i]).any()
            retest_up.iat[i] = 1 if touched else 0
        if breakout_down.iat[i] == 1:
            hi = min(len(out) - 1, i + retest_lookahead)
            touched = (h.iloc[i+1:hi+1] >= lower_prev.iloc[i]).any()
            retest_down.iat[i] = 1 if touched else 0

    # Fail-break: بازگشت سریع داخل رنج پس از شکست
    fail_break_up = pd.Series(0, index=out.index, dtype="int8")
    fail_break_down = pd.Series(0, index=out.index, dtype="int8")

    for i in range(len(out)):
        if breakout_up.iat[i] == 1:
            hi = min(len(out) - 1, i + fail_break_lookahead)
            failed = (c.iloc[i+1:hi+1] < upper_prev.iloc[i]).any()
            fail_break_up.iat[i] = 1 if failed else 0
        if breakout_down.iat[i] == 1:
            hi = min(len(out) - 1, i + fail_break_lookahead)
            failed = (c.iloc[i+1:hi+1] > lower_prev.iloc[i]).any()
            fail_break_down.iat[i] = 1 if failed else 0

    # ضد-لوک‌اِهد نهایی برای خروجی‌ها
    if anti_lookahead:
        breakout_up = breakout_up.shift(1).fillna(0).astype("int8")
        breakout_down = breakout_down.shift(1).fillna(0).astype("int8")
        retest_up = retest_up.shift(1).fillna(0).astype("int8")
        retest_down = retest_down.shift(1).fillna(0).astype("int8")
        fail_break_up = fail_break_up.shift(1).fillna(0).astype("int8")
        fail_break_down = fail_break_down.shift(1).fillna(0).astype("int8")

    out["range_upper"] = upper.astype("float32")
    out["range_lower"] = lower.astype("float32")
    out["breakout_up"] = breakout_up
    out["breakout_down"] = breakout_down
    out["retest_up"] = retest_up
    out["retest_down"] = retest_down
    out["fail_break_up"] = fail_break_up
    out["fail_break_down"] = fail_break_down

    return out
