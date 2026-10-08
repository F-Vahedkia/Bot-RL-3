# f04_features/indicators/divergences.py
# Status in (Bot-RL-3): Final Review 1405/01/23

from __future__ import annotations
import numpy as np
import pandas as pd
# pd.set_option('future.no_silent_downcasting', True)
from typing import Dict, Literal
from .core import rsi, macd
from collections import deque
from numba import njit

#------------------------------------------------------------------------------ 1- ok
"""
تشخیص pivot به صورت causal (بدون استفاده از داده‌های آینده)
pivot زمانی تایید می‌شود که k کندل بعد از آن تشکیل شده باشند
بنابراین خروجی pivot با shift(k) به جلو منتقل می‌شود
"""
def pivots_numpy(series: pd.Series, k: int = 2):
    x = series.to_numpy(dtype=np.float64, copy=False)
    n = len(x)

    ph = np.zeros(n, dtype=np.int8)
    pl = np.zeros(n, dtype=np.int8)

    maxdq = deque()
    mindq = deque()

    for i in range(n):
        # maintain max deque
        while maxdq and x[maxdq[-1]] <= x[i]:
            maxdq.pop()
        maxdq.append(i)
        if maxdq[0] <= i - (2 * k + 1):
            maxdq.popleft()

        # maintain min deque
        while mindq and x[mindq[-1]] >= x[i]:
            mindq.pop()
        mindq.append(i)
        if mindq[0] <= i - (2 * k + 1):
            mindq.popleft()

        # confirmation happens k bars after the center
        if i >= 2 * k:
            c = i - k
            if maxdq[0] == c:
                ph[i] = 1
            if mindq[0] == c:
                pl[i] = 1

    return ph, pl

@njit
def _pivots_njit_core(x, k):
    n = len(x)
    ph = np.zeros(n, dtype=np.int8)
    pl = np.zeros(n, dtype=np.int8)

    maxdq = []
    mindq = []
    for i in range(n):

        # maintain max deque
        while len(maxdq) > 0 and x[maxdq[-1]] <= x[i]:
            maxdq.pop()
        maxdq.append(i)
        if maxdq[0] <= i - (2 * k + 1):
            maxdq.pop(0)

        # maintain min deque
        while len(mindq) > 0 and x[mindq[-1]] >= x[i]:
            mindq.pop()
        mindq.append(i)
        if mindq[0] <= i - (2 * k + 1):
            mindq.pop(0)

        # confirmation happens k bars after the center
        if i >= 2 * k:
            c = i - k
            if maxdq[0] == c:
                ph[i] = 1
            if mindq[0] == c:
                pl[i] = 1

    return ph, pl

def pivots_njit(series: pd.Series, k: int = 2):
    # این تابع فقط برای مقایسه سرعت باقیمانده است. وگرنه باید حذف میشد
    x = series.to_numpy(dtype=np.float64)
    ph, pl = _pivots_njit_core(x, k)
    return ph, pl

# WRAPPER:
def pivots(series: pd.Series, k: int = 2):
    # نسخه numpy برای داده‌های کوچک سریع‌تر است
    if series.size < 700_000:
        ph, pl = pivots_numpy(series, k)
    else:
        ph, pl = pivots_njit(series, k)
    return (
        pd.Series(ph, index=series.index, name="ph"),
        pd.Series(pl, index=series.index, name="pl"),
    )

#------------------------------------------------------------------------------ 2- ok
# تشخیص واگرایی کلاسیک و مخفی بین قیمت و اسیلاتور
# خروجی به صورت فلگ باینری (مناسب برای استفاده در سیگنال‌ها)
def divergence_flags_old(price: pd.Series, osc: pd.Series, k: int = 2, mode: str = "classic"):
    """
    این تابع برای بازار زنده مناسب نیست و فقط برای مرور استراتژی کدنویسی موضوع نگه داشته شده
    """
    ph, pl = pivots(price, k)
    oh, ol = pivots(osc, k)

    bull = pd.Series(0, index=price.index, dtype="int8")
    bear = pd.Series(0, index=price.index, dtype="int8")

    last_ph = None
    last_pl = None
    last_oh = None
    last_ol = None

    p = price.values
    o = osc.values
    phv = ph.values
    plv = pl.values
    ohv = oh.values
    olv = ol.values

    for i in range(len(price)):

        if phv[i]:
            if last_ph is not None and last_oh is not None:

                if mode == "classic":
                    if p[last_ph] < p[i] and o[last_oh] > o[i]:
                        bear.iloc[i] = 1

                elif mode == "hidden":
                    if p[last_ph] > p[i] and o[last_oh] < o[i]:
                        bear.iloc[i] = 1

            last_ph = i

            if ohv[i]:
                last_oh = i


        if plv[i]:
            if last_pl is not None and last_ol is not None:

                if mode == "classic":
                    if p[last_pl] > p[i] and o[last_ol] < o[i]:
                        bull.iloc[i] = 1

                elif mode == "hidden":
                    if p[last_pl] < p[i] and o[last_ol] > o[i]:
                        bull.iloc[i] = 1

            last_pl = i

            if olv[i]:
                last_ol = i

    return bull, bear

def divergence_flags_numpy(price: pd.Series, osc: pd.Series, k: int = 2, mode: str = "classic"):
    """
    شناسایی واگرایی (divergence) بین price و oscillator
    نسخه سریع بر پایه numpy برای داده‌های < 1M
    کاملاً causal و مناسب هم برای training و هم برای live
    """

    # --- pivots ---
    ph, pl = pivots(price, k)
    oh, ol = pivots(osc, k)

    p = price.to_numpy(dtype=np.float64)
    o = osc.to_numpy(dtype=np.float64)
    phv = ph.to_numpy(dtype=np.int8)
    plv = pl.to_numpy(dtype=np.int8)
    ohv = oh.to_numpy(dtype=np.int8)
    olv = ol.to_numpy(dtype=np.int8)

    bull = np.zeros_like(p, dtype=np.int8)
    bear = np.zeros_like(p, dtype=np.int8)

    mode_flag = 0 if mode == "classic" else 1

    # --- حالت last pivot ها ---
    last_ph = -1
    last_pl = -1
    last_oh = -1
    last_ol = -1

    # چون حلقه سبک با numpy indexها کار می‌کند، برای n<1M سریع است (~0.05s)
    for i in range(p.size):
        if phv[i]:
            if last_ph != -1 and last_oh != -1:
                if mode_flag == 0:  # classic
                    if p[last_ph] < p[i] and o[last_oh] > o[i]:
                        bear[i] = 1
                else:  # hidden
                    if p[last_ph] > p[i] and o[last_oh] < o[i]:
                        bear[i] = 1
            last_ph = i
            if ohv[i]:
                last_oh = i

        if plv[i]:
            if last_pl != -1 and last_ol != -1:
                if mode_flag == 0:  # classic
                    if p[last_pl] > p[i] and o[last_ol] < o[i]:
                        bull[i] = 1
                else:  # hidden
                    if p[last_pl] < p[i] and o[last_ol] > o[i]:
                        bull[i] = 1
            last_pl = i
            if olv[i]:
                last_ol = i

    return bull, bear

@njit
def _divergence_flags_njit_core(p, o, ph, pl, oh, ol, mode_flag):
    n = p.size
    bull = np.zeros(n, dtype=np.int8)
    bear = np.zeros(n, dtype=np.int8)

    last_ph = -1
    last_pl = -1
    last_oh = -1
    last_ol = -1

    for i in range(n):

        # --- price high pivots ---
        if ph[i] == 1:
            if last_ph != -1 and last_oh != -1:

                if mode_flag == 0:  # classic
                    if p[last_ph] < p[i] and o[last_oh] > o[i]:
                        bear[i] = 1

                else:  # hidden
                    if p[last_ph] > p[i] and o[last_oh] < o[i]:
                        bear[i] = 1

            last_ph = i
            if oh[i] == 1:
                last_oh = i

        # --- price low pivots ---
        if pl[i] == 1:
            if last_pl != -1 and last_ol != -1:

                if mode_flag == 0:  # classic
                    if p[last_pl] > p[i] and o[last_ol] < o[i]:
                        bull[i] = 1

                else:  # hidden
                    if p[last_pl] < p[i] and o[last_ol] > o[i]:
                        bull[i] = 1

            last_pl = i
            if ol[i] == 1:
                last_ol = i

    return bull, bear

def divergence_flags_njit(price: pd.Series, osc: pd.Series, k: int = 2, mode: str = "classic"):
    # --- pivots (causal, fast, already optimized) ---
    ph, pl = pivots(price, k)
    oh, ol = pivots(osc, k)

    p = price.to_numpy(dtype=np.float64)
    o = osc.to_numpy(dtype=np.float64)
    phv = ph.to_numpy(dtype=np.int8)   # price high value
    plv = pl.to_numpy(dtype=np.int8)
    ohv = oh.to_numpy(dtype=np.int8)   # oscilator high value
    olv = ol.to_numpy(dtype=np.int8)

    mode_flag = 0 if mode == "classic" else 1

    bull, bear = _divergence_flags_njit_core(p, o, phv, plv, ohv, olv, mode_flag)

    return bull, bear

# WRAPPER:
def divergence_flags(price: pd.Series,
                     osc: pd.Series,
                     k: int = 2,
                     mode: Literal ["classic", "hidden"] = "classic",
):
    # --- نوع داده باید Series باشد ---
    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas Series")
    if not isinstance(osc, pd.Series):
        raise TypeError("osc must be a pandas Series")

    # --- هر دو باید 1D باشند ---
    if price.ndim != 1:
        raise ValueError("price must be 1-dimensional")
    if osc.ndim != 1:
        raise ValueError("osc must be 1-dimensional")

    # --- سری‌ها باید هم‌طول باشند ---
    if price.size != osc.size:
        raise ValueError("price and osc must have the same length")

    # --- سری‌ها باید هم‌index باشند ---
    if not price.index.equals(osc.index):
        raise ValueError("price and osc must have the same index")

    # --- نوع داده باید قابل تبدیل به float باشد ---
    # این کنترل سبک، سریع و امن است
    try:
        price.astype("float64")
        osc.astype("float64")
    except Exception:
        raise TypeError("price and osc must contain float‑compatible numeric data")

    # --- انتخاب موتور بر اساس حجم داده ---
    if price.size < 1_000_000:
        bull, bear = divergence_flags_numpy(price, osc, k, mode)
    else:
        bull, bear = divergence_flags_njit(price, osc, k, mode)
    
    if mode == "classic":
        abb_mod = "cls"  # abbreviated mode: "cls" for "classic"
    else:
        abb_mod = "hid"  # abbreviated mode: "hid" for "hidden"    
    return (
        pd.Series(bull, index=price.index, dtype="int8", name=f"div_bull_{abb_mod}"),
        pd.Series(bear, index=price.index, dtype="int8", name=f"div_bear_{abb_mod}"),
    )

#------------------------------------------------------------------------------ 3- ok
def registry_flag() -> Dict[str, callable]:
    # رجیستری فیچرهای باینری واگرایی
    
    def cast32(d: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        return {k: pd.Series(v, copy=False).astype("float32") for k, v in d.items()}
    
    def make_div_macd(
        df,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        k: int = 2,
        mode: str = "classic",
        **_,
    ):
        line, _, _ = macd(df["close"], fast, slow, signal)
        bull, bear = divergence_flags(df["close"], line, k=k, mode=mode)
        return cast32({
            f"div_macd_bull_{fast}_{slow}_{signal}_{k}_{mode}": bull,
            f"div_macd_bear_{fast}_{slow}_{signal}_{k}_{mode}": bear,
        })

    def make_div_rsi(
        df,
        period: int = 14,
        k: int = 2,
        mode: str = "classic",
        **_,
    ):
        osc = rsi(df["close"], period)
        bull, bear = divergence_flags(df["close"], osc, k=k, mode=mode)
        return cast32({
            f"div_rsi_bull_{period}_{k}_{mode}": bull,
            f"div_rsi_bear_{period}_{k}_{mode}": bear,
        })

    return {
        "div_macd": make_div_macd,
        "div_rsi": make_div_rsi,
    }

#------------------------------------------------------------------------------ 4- ok
def divergence_values_old(price: pd.Series, osc: pd.Series, k: int = 2, mode: str = "classic"):
    """
    این تابع برای بازار زنده مناسب نیست و فقط برای مرور استراتژی کدنویسی موضوع نگه داشته شده
    """
    # محاسبه مقدار عددی واگرایی (feature خام برای مدل‌های ML / RL)
    # مقدار divergence به صورت اختلاف اسیلاتور نسبت به pivot قبلی

    ph, pl = pivots(price, k)
    oh, ol = pivots(osc, k)

    bull = pd.Series(np.nan, index=price.index, dtype="float64")
    bear = pd.Series(np.nan, index=price.index, dtype="float64")

    last_ph = None
    last_pl = None
    last_oh = None
    last_ol = None

    p = price.values
    o = osc.values
    phv = ph.values
    plv = pl.values
    ohv = oh.values
    olv = ol.values

    for i in range(len(price)):

        if phv[i]:
            if last_ph is not None and last_oh is not None:

                if mode == "classic":
                    if p[last_ph] < p[i] and o[last_oh] > o[i]:
                        bear.iloc[i] = o[i] - o[last_oh]

                elif mode == "hidden":
                    if p[last_ph] > p[i] and o[last_oh] < o[i]:
                        bear.iloc[i] = o[i] - o[last_oh]

            last_ph = i

            if ohv[i]:
                last_oh = i


        if plv[i]:
            if last_pl is not None and last_ol is not None:

                if mode == "classic":
                    if p[last_pl] > p[i] and o[last_ol] < o[i]:
                        bull.iloc[i] = o[i] - o[last_ol]

                elif mode == "hidden":
                    if p[last_pl] < p[i] and o[last_ol] > o[i]:
                        bull.iloc[i] = o[i] - o[last_ol]

            last_pl = i

            if olv[i]:
                last_ol = i
    return bull, bear

def divergence_values(price: pd.Series, osc: pd.Series, k: int = 2, mode: str = "classic"):
    # محاسبه مقدار عددی واگرایی (feature خام برای مدل‌های ML / RL)
    # مقدار divergence به صورت اختلاف اسیلاتور نسبت به pivot قبلی

    ph, pl = pivots(price, k)
    oh, ol = pivots(osc, k)

    p = price.to_numpy(dtype=np.float64, copy=False)
    o = osc.to_numpy(dtype=np.float64, copy=False)

    phv = ph.to_numpy()
    plv = pl.to_numpy()
    ohv = oh.to_numpy()
    olv = ol.to_numpy()

    n = p.size

    bull = np.full(n, np.nan, dtype=np.float64)
    bear = np.full(n, np.nan, dtype=np.float64)

    last_ph = -1
    last_pl = -1
    last_oh = -1
    last_ol = -1

    for i in range(n):

        if phv[i]:

            if last_ph != -1 and last_oh != -1:

                if mode == "classic":
                    if p[last_ph] < p[i] and o[last_oh] > o[i]:
                        bear[i] = o[i] - o[last_oh]

                elif mode == "hidden":
                    if p[last_ph] > p[i] and o[last_oh] < o[i]:
                        bear[i] = o[i] - o[last_oh]

            last_ph = i

            if ohv[i]:
                last_oh = i


        if plv[i]:

            if last_pl != -1 and last_ol != -1:

                if mode == "classic":
                    if p[last_pl] > p[i] and o[last_ol] < o[i]:
                        bull[i] = o[i] - o[last_ol]

                elif mode == "hidden":
                    if p[last_pl] < p[i] and o[last_ol] > o[i]:
                        bull[i] = o[i] - o[last_ol]

            last_pl = i

            if olv[i]:
                last_ol = i

    return (
        pd.Series(bull, index=price.index),
        pd.Series(bear, index=price.index),
    )

#------------------------------------------------------------------------------ 5- ok
def registry() -> Dict[str, callable]:
    # رجیستری فیچرهای عددی واگرایی (برای استفاده مستقیم در RL)
    def cast32(d: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        return {k: pd.Series(v, copy=False).astype("float32") for k, v in d.items()}
    
    def make_div_macd(
        df,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        k: int = 2,
        mode: str = "classic",
        **_,
    ):
        line, _, _ = macd(df["close"], fast, slow, signal)
        b, s = divergence_values(df["close"], line, k=k, mode=mode)
        return cast32({
            f"div_macd_bull_{fast}_{slow}_{signal}_{k}_{mode}": b,
            f"div_macd_bear_{fast}_{slow}_{signal}_{k}_{mode}": s,
        })

    def make_div_rsi(
        df,
        period: int = 14,
        k: int = 2,
        mode: str = "classic",
        **_,
    ):
        osc = rsi(df["close"], period)
        b, s = divergence_values(df["close"], osc, k=k, mode=mode)
        return cast32({
            f"div_rsi_bull_{period}_{k}_{mode}": b,
            f"div_rsi_bear_{period}_{k}_{mode}": s,
        })

    return {
        "div_macd": make_div_macd,
        "div_rsi": make_div_rsi,
    }

#------------------------------------------------------------------------------

# =====================================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# =====================================================================================
""" Func Names                           Used in Functions: ...
                       1   2   3   4   5   6   7   8   9  10  11  11  12  13  14
1  pivots             --  ok  --  ok  --
2  divergence_flags   --  --  ok  --  --   Only For Live Trading
3  registry_flag      --  --  --  --  --   Only For Live Trading
4  divergence_values  --  --  --  --  ok
5  registry           --  --  --  --  --   MAIN REGISTRY FUNCTION
"""
