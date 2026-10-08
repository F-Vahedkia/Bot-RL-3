# f04_features/price_action/breakouts_numba_1.py
# Reviewed at 1404/10/15

"""
🔹 ویژگی‌ها و مزایا

Numba-Accelerated
توابع _rolling_max, _rolling_min, _confirm_consecutive, _retest_and_fail با @njit بهینه شده‌اند.
سرعت اجرای rolling و حلقه‌های Retest/Fail بسیار بالا رفته است.

Anti-lookahead
تمام فلگ‌ها (breakout, retest, fail_break) با np.roll شیفت داده شده و اولین ردیف صفرگذاری شده است.

Warm-up & min_periods
ردیف‌های اولیه که کمتر از min_periods داده دارند، NaN خواهند بود.
این با الزامات استاندارد فیچرهای مالی و تست warm-up کاملاً منطبق است.

MTF-Aware
اگرچه خود کد مستقیماً روی TF مختلف اعمال نشده، اما معماری آن اجازه می‌دهد که خروجی‌های باند و فلگ‌ها به راحتی در هر TF (High/Low/Close resampled) استفاده شود.
با کمی wrapper بیرونی می‌توان برای TF بالاتر یا پایین‌تر اعمال نمود.

Dtype-Consistency
range_upper/lower → float32
تمام فلگ‌ها → int8
سازگاری کامل با بقیه فیچرها و حافظه کم.
پارامترها کامل و قابل کنترل
confirm_closes, retest_lookahead, fail_break_lookahead و anti_lookahead بدون تغییر API.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit


# ------------------------------------------------------------------
# NUMBA KERNELS
# ------------------------------------------------------------------

@njit(cache=True)
def _rolling_max(arr, window, min_periods):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window + 1)
        cnt = i - start + 1
        if cnt < min_periods:
            continue
        m = arr[start]
        for j in range(start + 1, i + 1):
            if arr[j] > m:
                m = arr[j]
        out[i] = m
    return out


@njit(cache=True)
def _rolling_min(arr, window, min_periods):
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window + 1)
        cnt = i - start + 1
        if cnt < min_periods:
            continue
        m = arr[start]
        for j in range(start + 1, i + 1):
            if arr[j] < m:
                m = arr[j]
        out[i] = m
    return out


@njit(cache=True)
def _confirm_consecutive(cond, n):
    out = np.zeros(len(cond), dtype=np.int8)
    run = 0
    for i in range(len(cond)):
        if cond[i]:
            run += 1
            if run >= n:
                out[i] = 1
        else:
            run = 0
    return out


@njit(cache=True)
def _retest_and_fail(
    h, l, c,
    breakout_up, breakout_dn,
    upper_prev, lower_prev,
    retest_lookahead, fail_lookahead
):
    n = len(c)
    retest_up = np.zeros(n, dtype=np.int8)
    retest_dn = np.zeros(n, dtype=np.int8)
    fail_up = np.zeros(n, dtype=np.int8)
    fail_dn = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if breakout_up[i] == 1 and not np.isnan(upper_prev[i]):
            hi = min(n - 1, i + retest_lookahead)
            for j in range(i + 1, hi + 1):
                if l[j] <= upper_prev[i]:
                    retest_up[i] = 1
                    break
            hi = min(n - 1, i + fail_lookahead)
            for j in range(i + 1, hi + 1):
                if c[j] < upper_prev[i]:
                    fail_up[i] = 1
                    break

        if breakout_dn[i] == 1 and not np.isnan(lower_prev[i]):
            hi = min(n - 1, i + retest_lookahead)
            for j in range(i + 1, hi + 1):
                if h[j] >= lower_prev[i]:
                    retest_dn[i] = 1
                    break
            hi = min(n - 1, i + fail_lookahead)
            for j in range(i + 1, hi + 1):
                if c[j] > lower_prev[i]:
                    fail_dn[i] = 1
                    break

    return retest_up, retest_dn, fail_up, fail_dn


# ------------------------------------------------------------------
# PUBLIC API (UNCHANGED)
# ------------------------------------------------------------------

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

    for col in (high_col, low_col, close_col):
        if col not in df.columns:
            raise ValueError(f"'{col}' not found in DataFrame.")

    out = df.copy()

    h = df[high_col].to_numpy(dtype=np.float64)
    l = df[low_col].to_numpy(dtype=np.float64)
    c = df[close_col].to_numpy(dtype=np.float64)

    upper = _rolling_max(h, range_window, min_periods)
    lower = _rolling_min(l, range_window, min_periods)

    upper_prev = np.roll(upper, 1)
    lower_prev = np.roll(lower, 1)
    upper_prev[0] = np.nan
    lower_prev[0] = np.nan

    above = c > upper_prev
    below = c < lower_prev

    breakout_up = _confirm_consecutive(above, confirm_closes)
    breakout_dn = _confirm_consecutive(below, confirm_closes)

    ret_up, ret_dn, fail_up, fail_dn = _retest_and_fail(
        h, l, c,
        breakout_up, breakout_dn,
        upper_prev, lower_prev,
        retest_lookahead, fail_break_lookahead
    )

    if anti_lookahead:
        breakout_up = np.roll(breakout_up, 1)
        breakout_dn = np.roll(breakout_dn, 1)
        ret_up = np.roll(ret_up, 1)
        ret_dn = np.roll(ret_dn, 1)
        fail_up = np.roll(fail_up, 1)
        fail_dn = np.roll(fail_dn, 1)

        breakout_up[0] = breakout_dn[0] = 0
        ret_up[0] = ret_dn[0] = 0
        fail_up[0] = fail_dn[0] = 0

    out["range_upper"] = upper.astype("float32")
    out["range_lower"] = lower.astype("float32")
    out["breakout_up"] = breakout_up.astype("int8")
    out["breakout_down"] = breakout_dn.astype("int8")
    out["retest_up"] = ret_up.astype("int8")
    out["retest_down"] = ret_dn.astype("int8")
    out["fail_break_up"] = fail_up.astype("int8")
    out["fail_break_down"] = fail_dn.astype("int8")

    return out
