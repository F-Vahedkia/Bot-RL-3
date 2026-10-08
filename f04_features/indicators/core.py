# f04_features/indicators/core.py
# Status in (Bot-RL-3): Final Review 1405/01/21
#                       Reviewed at  1405/02/19
"""
اندیکاتورهای پایه - Anti look-ahead
شامل اندیکاتورهای:
sma, ema, wma, rsi, roc, atr, macd, bollinger, keltner, stochastic,
cci, mfi, obv, williams_r, parabolic_sar, heikin_ashi
"""

# =============================================================================
# Imports & Logger
# =============================================================================
from __future__ import annotations
from typing import Dict, Tuple, Literal, Callable
import numpy as np
import pandas as pd
from numba import njit
import logging

# ------------------ Importing Internal Modules ---------------------
from .utils import (
    true_range,
    ema as ema_utils,
    compute_atr
)

# -------------------- Logger for this module -----------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
# فرضهای لازم برای توابع هسته:
#  - نوع ورودی این توابع یک یا چند سری پانداس است
#  - نوع float64 در ابتدای این توابع، enforce میشود
#  - خروجی این توابع:
#  - خروجی این توابع یک یا چند سری است
#  - در خروجی این توابع نوع 32 یا 64 تعیین نمیشود
#  - در خروجی این توابع برای سری یا دیتافریم نام گذاری نمیشود
#  - خروجی رجیستری:
#  - در خروجی API های توابع در رجیستری، هم float32 و هم نام سری ها تعین میشود
#  - هرکدام از توابع رجیستری، یک دیکشنری از سری/سریها را برمیگرداند
# ============================================================================= 1,2,3
# Part-1: Averages
# ============================================================================= OK
# تابع زیر فقط بخاطر سرعت بیشتری که دارد نگهداشته شده است
# وگر چون min_periods را شامل نمیشود، باید حذف شود
def sma_old(s: pd.Series, n: int) -> pd.Series:      # was faster by numpy
    """
    Simple Moving Average (SMA)
    - بدون look‑ahead
    - warm-up صحیح
    - پیاده‌سازی بسیار سریع مناسب دیتای میلیون کندلی
    """
    if s is None or len(s) < n:
        logger.warning("sma() from core.py returns empty series.")
        return pd.Series(dtype="float64", name=f"sma_{n}")

    # ---------- Core computation ----------
    values = s.to_numpy(dtype="float64")
    length = len(values)
    csum = np.cumsum(values)
    out = np.full(length, np.nan, dtype="float64")
    out[n-1:] = (csum[n-1:] - np.concatenate(([0.0], csum[:-n]))) / n
    return pd.Series(out, index=s.index).rename(f"sma_{n}")

def sma(s: pd.Series, n: int, min_periods: int = None) -> pd.Series:
    """
    Simple Moving Average (SMA)
    - بدون look-ahead
    - warm-up صحیح مطابق rolling(window=n, min_periods=...)
    - سریع با numpy
    """
    if s is None or n <= 0:
        return pd.Series(dtype="float64", name=f"sma_{n}")

    # اگر min_periods داده نشده باشد، همان رفتار قدیمی را حفظ می‌کنیم
    if min_periods is None:
        min_periods = n

    # min_periods باید بین 1 و n باشد
    if min_periods <= 0:
        min_periods = 1
    if min_periods > n:
        min_periods = n

    values = s.to_numpy(dtype="float64")
    length = len(values)
    if length < min_periods:
        logger.warning("sma() from core.py returns empty series.")
        return pd.Series(np.full(length, np.nan, dtype="float64"), index=s.index).rename(f"sma_{n}")

    # sums[i] = sum(values[:i])  (prefix sum with leading 0)
    sums = np.concatenate(([0.0], np.cumsum(values)))

    out = np.full(length, np.nan, dtype="float64")

    # --- warm-up: window sizes 1..(n-1), but only from min_periods ---
    # out[t] = mean(values[t-k+1 : t+1]) for k = min(t+1, n)
    warm_end = min(n - 1, length - 1)
    if warm_end >= (min_periods - 1):
        t = np.arange(min_periods - 1, warm_end + 1)
        k = t + 1  # window size for warm-up (since t < n)
        out[t] = (sums[t + 1] - sums[t + 1 - k]) / k

    # --- steady-state: full window n from index n-1 onward ---
    if length >= n:
        t2 = np.arange(n - 1, length)
        out[t2] = (sums[t2 + 1] - sums[t2 + 1 - n]) / n

    return pd.Series(out, index=s.index).rename(f"sma_{n}")


# ---------------------------------------------------------
def ema(s: pd.Series, n: int, min_periods: int = None) -> pd.Series:
    """
    Exponential Moving Average (EMA)
    - بدون look-ahead
    - warm-up استاندارد و قابل تنظیم با min_periods
    """

    # ---------- Input validation ----------
    if s is None:
        logger.warning("ema(): input is None. Returning empty series.")
        return pd.Series(dtype="float64", name=f"ema_{n}")

    if len(s) == 0:
        logger.warning("ema(): empty input series.")
        return pd.Series(dtype="float64", index=s.index, name=f"ema_{n}")

    return ema_utils(s=s, n=n, min_periods=min_periods)


# ---------------------------------------------------------
def wma(s: pd.Series, n: int) -> pd.Series:      # is faster
    """
    Weighted Moving Average (WMA)
    - بدون look‑ahead
    - پیاده‌سازی برداری بسیار سریع (بدون rolling.apply)
    - مناسب دیتای بسیار بزرگ
    """
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if s is None:
        logger.warning("wma(): input is None. Returning empty series.")
        return pd.Series(dtype="float64", name=f"wma_{n}")

    # Case 2: insufficient length → return aligned empty series
    if len(s) < n:
        logger.warning("wma(): input series too short. Returning empty aligned series.")
        return pd.Series(index=s.index, dtype="float64", name=f"wma_{n}")
    
    # ---------- Core computation ----------
    values = np.nan_to_num(s.to_numpy(dtype="float64"), nan=0.0)
    length = len(values)

    weights = np.arange(1, n + 1, dtype="float64")
    weights /= weights.sum()

    conv = np.convolve(values, weights, mode="valid")

    result = np.full(length, np.nan, dtype="float64")
    result[n-1:] = conv

    return pd.Series(result, index=s.index, name=f"wma_{n}")


# ============================================================================= 4,5
# Part-2: Ocilators
# ============================================================================= OK
# --- Canonical RSI (core-level) — Single Source ----------
def rsi(
    close: pd.Series,
    length: int = 14,
    method: Literal["ema", "wilders"] = "ema"
) -> pd.Series:

    """ Compute RSI (Relative Strength Index) with proper warm-up handling.
    - close: pd.Series of closing prices
    - length: lookback period
    - method: 'ema' (default) or 'wilders'
    Returns pd.Series with NaN for warm-up period.
    """

    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if close is None:
        logger.warning("rsi(): input is None. Returning empty series.")
        return pd.Series(dtype="float64", name=f"rsi_{length}")


    # Case 2: insufficient length → return aligned empty series
    if len(close) < length + 1:
        logger.warning("rsi(): input series too short. Returning empty aligned series.")
        return pd.Series(index=close.index, dtype="float64", name=f"rsi_{length}")

    # ---------- Core computation ----------
    c = pd.Series(close).astype("float64", copy=False)
    delta = c.diff()

    # gains and losses
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))

    if method.lower() == "wilders":
        # Wilder's smoothing (alpha = 1/length)
        gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    else:
        # EMA-like smoothing
        gain = gain.ewm(span=length, adjust=False, min_periods=length).mean()
        loss = loss.ewm(span=length, adjust=False, min_periods=length).mean()

    # RS
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
        rsi_values = 100 - 100 / (1 + rs)

    # skeep initial NaN for warm-up (first 'length' values)
    rsi_values[:length] = np.nan

    return rsi_values.rename(f"rsi_{length}")

# ---------------------------------------------------------
def roc(close: pd.Series, n: int = 10) -> pd.Series:
    """
    Rate of Change (percentage) over n periods.
    - Returns percent change * 100 as float64.
    - First n rows are NaN (warm-up).
    """
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if close is None:
        logger.warning("roc(): one input are None. Returning empty series.")
        return pd.Series(dtype="float64", name=f"roc_{n}")
    
    # Case 2: insufficient length → return aligned empty series
    if len(close) < n + 1:
        logger.warning("roc(): input series too short. Returning empty aligned series.")
        return pd.Series(index=close.index, dtype="float64", name=f"roc_{n}")
    
    # ---------- Core computation ----------
    s = pd.Series(close).astype("float64", copy=False)
    out = s.pct_change(periods=n, fill_method=None) * 100.0
    out[:n] = np.nan
    return out.rename(f"roc_{n}")

# ============================================================================= 6
# Part-3: ATR/TR
# ============================================================================= OK (New Changes:050124)
def atr_old(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    # Use this function for system decisions only
    """
    Average True Range (ATR) using EWM smoothing with alpha = 1/n (Wilder-like).
    - TR is computed per-bar (True Range).
    - ATR uses ewm(alpha=1/n, adjust=False, min_periods=n).
    - First n rows are NaN to ensure proper warm-up.
    """
    
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if high is None or low is None or close is None:
        logger.warning("atr(): one or more inputs are None. Returning empty series.")
        return pd.Series(dtype="float64", name=f"atr_{n}")

    # Case 2: insufficient length → return aligned empty series
    if len(high) < n or len(low) < n or len(close) < n:
        logger.warning("atr(): input series too short. Returning empty aligned series.")
        return pd.Series(index=close.index, dtype="float64", name=f"atr_{n}")

    # ---------- Core computation ----------
    # rely on true_range(high, low, close) existing in module scope
    tr = true_range(high, low, close).astype("float64")
    # EWM with alpha=1/n, require min_periods=n
    if len(tr) < n:
        logger.warning("atr() from core.py returns empty series.")
        return pd.Series(dtype="float64", name=f"atr_{n}")
    atr_series = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    atr_series[:n] = np.nan  # warm-up
    return atr_series.rename(f"atr_{n}")


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 14, method: str = "wilder", min_periods: int = None) -> pd.Series:
    return compute_atr(df=pd.concat([high, low, close], axis=1),
                       window=n, method=method, min_periods=min_periods)

# ============================================================================= 7
# Part-4: MACD
# ============================================================================= OK
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD line, Signal line, and Histogram with proper warm-up handling.
    Returns pd.Series with NaN for initial warm-up rows.
    """
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if close is None:
        logger.warning("macd(): one inputs are None. Returning empty series.")
        return (
            pd.Series(dtype="float64", name=f"macd_{fast}_{slow}_{signal}"), 
            pd.Series(dtype="float64", name=f"macd_signal_{fast}_{slow}_{signal}"), 
            pd.Series(dtype="float64", name=f"macd_hist_{fast}_{slow}_{signal}")
        )
    # Case 2: insufficient length → return aligned empty series
    if len(close) < slow + 1:
        logger.warning("macd(): input series too short. Returning empty aligned series.")
        return (
            pd.Series(index=close.index, dtype="float64", name=f"macd_{fast}_{slow}_{signal}"), 
            pd.Series(index=close.index, dtype="float64", name=f"macd_signal_{fast}_{slow}_{signal}"), 
            pd.Series(index=close.index, dtype="float64", name=f"macd_hist_{fast}_{slow}_{signal}")
        )

    # ---------- Core computation ----------
    # EMA fast and slow with min_periods for correct warm-up
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean().astype("float64")
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean().astype("float64")

    # MACD line
    macd_line = (ema_fast - ema_slow)
    macd_line[:slow] = np.nan  # ensure first 'slow' rows are NaN

    # Signal line: EMA on MACD line
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    signal_line[:slow + signal - 1] = np.nan  # first value appears after (slow + signal -1)

    # Histogram: MACD - Signal
    hist = (macd_line - signal_line)
    hist[:slow + signal - 1] = np.nan  # align "histogram" with "signal_line"

    # Rename series
    macd_line   = macd_line  .rename(       f"macd_{fast}_{slow}_{signal}")
    signal_line = signal_line.rename(f"macd_signal_{fast}_{slow}_{signal}")
    hist        = hist       .rename(  f"macd_hist_{fast}_{slow}_{signal}")

    return macd_line, signal_line, hist

# ============================================================================= 8,9
# Part-5: Bounds
# ============================================================================= OK
def bollinger(close: pd.Series, n: int = 20, k: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands (mid, upper, lower).
    - mid: SMA(n) (first n-1 rows NaN)
    - upper/lower: mid ± k * rolling_std(n) (same warm-up as mid)
    - returns (mid, upper, lower) as float64 Series with names.
    """
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if close is None:
        logger.warning("bollinger(): one input are None. Returning empty series.")
        return (
            pd.Series(dtype="float64", name=f"bb_up_{n}_{k}"), 
            pd.Series(dtype="float64", name=f"bb_mid_{n}_{k}"), 
            pd.Series(dtype="float64", name=f"bb_lo_{n}_{k}")
        )
    
    # Case 2: insufficient length → return aligned empty series
    if len(close) < n:
        logger.warning("bollinger(): input series too short. Returning empty aligned series.")
        return (
            pd.Series(index=close.index, dtype="float64", name=f"bb_up_{n}_{k}"), 
            pd.Series(index=close.index, dtype="float64", name=f"bb_mid_{n}_{k}"), 
            pd.Series(index=close.index, dtype="float64", name=f"bb_lo_{n}_{k}")
        )
    
    # ---------- Core computation ----------
    mid = sma(close, n).astype("float64")  # uses sma() which ensures warm-up
    # use population std (ddof=0) for stability; require same min_periods as mid
    sd = close.astype("float64").rolling(window=n, min_periods=n).std(ddof=0)
    # ensure identical warm-up alignment
    sd[:n-1] = np.nan
    upper = (mid + k * sd)
    lower = (mid - k * sd)

    upper = upper.rename(f"bb_up_{n}_{k}")
    mid = mid.rename(f"bb_mid_{n}_{k}")
    lower = lower.rename(f"bb_lo_{n}_{k}")
    return upper, mid, lower

# ---------------------------------------------------------
def keltner(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, m: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channel (mid = EMA(n), upper = mid + atr_mult * ATR(n), lower = mid - m*ATR).
    - Ensures warm-up: first n rows NaN (ATR uses n; EMA uses n -> we align to max).
    - Returns (mid, upper, lower) as float32 Series with names.
    - m: atr_mult
    """
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if high is None or low is None or close is None:
        logger.warning("keltner(): one or more inputs are None. Returning empty series.")
        return (
            pd.Series(dtype="float64", name=f"kelt_up_{n}_{m}"),
            pd.Series(dtype="float64", name=f"kelt_mid_{n}_{m}"),
            pd.Series(dtype="float64", name=f"kelt_lo_{n}_{m}"),
        )
    
    # Case 2: insufficient length → return aligned empty series
    if len(high) < n or len(low) < n or len(close) < n:
        logger.warning("keltner(): input series too short. Returning empty aligned series.")
        return (
            pd.Series(index=close.index, dtype="float64", name=f"kelt_up_{n}_{m}"), 
            pd.Series(index=close.index, dtype="float64", name=f"kelt_mid_{n}_{m}"), 
            pd.Series(index=close.index, dtype="float64", name=f"kelt_lo_{n}_{m}")
        )

    # ---------- Core computation ----------
    mid = ema(close, n).astype("float64")   # ema ensures warm-up
    atr_val = atr(high, low, close, n).astype("float64")
    mid[:n - 1]     = np.nan   # warm-up
    atr_val[:n - 1] = np.nan   # warm-up
    upper = (mid + m * atr_val)
    lower = (mid - m * atr_val)

    upper = upper.rename(f"kelt_up_{n}_{m}")
    mid   =   mid.rename(f"kelt_mid_{n}_{m}")
    lower = lower.rename(f"kelt_lo_{n}_{m}")
    return upper, mid, lower

# ============================================================================= 10
# Part-6: Stochastic
# ============================================================================= OK
def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K and %D) – professional grade version.
    - %K = 100 * (close - lowest_n) / (highest_n - lowest_n)
    - %D = SMA(%K, d)
    - First n rows of %K are NaN (warm-up)
    - First n + d - 1 rows of %D are NaN (warm-up)
    - Both returned as float64 Series with proper naming.
    """
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if high is None or low is None or close is None:
        logger.warning("stochastic(): one or more inputs are None. Returning empty series.")
        return (
            pd.Series(dtype="float64", name=f"stoch_k_{n}_{d}"),
            pd.Series(dtype="float64", name=f"stoch_d_{n}_{d}")
        )
    
    # Case 2: insufficient length → return aligned empty series
    if len(high) < n or len(low) < n or len(close) < n:
        logger.warning("stochastic(): input series too short. Returning empty aligned series.")
        return (
            pd.Series(index=close.index, dtype="float64", name=f"stoch_k_{n}_{d}"),
            pd.Series(index=close.index, dtype="float64", name=f"stoch_d_{n}_{d}")
        )

    # ---------- Core computation ----------
    high_f = high.astype("float64")
    low_f = low.astype("float64")
    close_f = close.astype("float64")

    # Rolling highest / lowest
    lowest = low_f.rolling(window=n, min_periods=n).min()
    highest = high_f.rolling(window=n, min_periods=n).max()

    # %K calculation
    # Prevent divide-by-zero using replace
    k = 100.0 * (close_f - lowest) / (highest - lowest).replace(0, np.nan)
    k[:n - 1] = np.nan                     # correct warm-up
    k = k.rename(f"stoch_k_{n}_{d}")

    # %D calculation (SMA over K)
    # dline = k.rolling(window=d, min_periods=d).mean()
    dline = sma(k, d)
    dline[:n -1 + d - 1] = np.nan          # correct warm-up for %D
    dline = dline.rename(f"stoch_d_{n}_{d}")

    return k, dline

# ============================================================================= 11,12,13,14
# Part-7: CCI/MFI/OBV/Williams%R
# ============================================================================= OK
def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if high is None or low is None or close is None:
        logger.warning("cci(): one or more inputs are None. Returning empty series.")
        return pd.Series(dtype="float64", name=f"cci_{n}")
    
    # Case 2: insufficient length → return aligned empty series
    if len(high) < n or len(low) < n or len(close) < n:
        logger.warning("cci(): input series too short. Returning empty aligned series.")
        return pd.Series(index=close.index, dtype="float64", name=f"cci_{n}")

    # ---------- Core computation ----------
    tp = ((high + low + close) / 3.0).astype("float64")
    ma = sma(tp, n)
    md = (tp - ma).abs().rolling(window=n, min_periods=n).mean()
    c = (tp - ma) / (0.015 * md.replace(0, np.nan))
    c[: n - 1] = np.nan
    return c.rename(f"cci_{n}")

# ---------------------------------------------------------
def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if high is None or low is None or close is None or volume is None:
        logger.warning("mfi(): one or more inputs are None. Returning empty series.")
        return pd.Series(dtype="float64", name=f"mfi_{n}")
    
    # Case 2: insufficient length → return aligned empty series
    if len(high) < n or len(low) < n or len(close) < n or len(volume) < n:
        logger.warning("mfi(): input series too short. Returning empty aligned series.")
        return pd.Series(index=close.index, dtype="float64", name=f"mfi_{n}")
    
    # ---------- Core computation ----------
    tp = ((high + low + close) / 3.0).astype("float64")
    vol = volume.fillna(0.0).astype("float64")
    mf = tp * vol
    pos = pd.Series(np.where(tp.diff() > 0, mf, 0.0), index=tp.index).rolling(window=n, min_periods=n).sum()
    neg = pd.Series(np.where(tp.diff() < 0, mf, 0.0), index=tp.index).rolling(window=n, min_periods=n).sum()
    ratio = pos / neg.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + ratio))
    out[:n] = np.nan
    return out.rename(f"mfi_{n}")

# ---------------------------------------------------------
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (OBV)
    - OBV requires at least 1 previous close → warm-up 1 row (NaN)
    - Uses cumsum of volume * direction
    - Fully deterministic, float32 output
    """
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if close is None or volume is None:
        logger.warning("obv(): one or more inputs are None. Returning empty series.")
        return pd.Series(dtype="float64", name=f"obv")
    
    # Case 2: insufficient length → return aligned empty series
    if len(close) < 2 or len(volume) < 2:
        logger.warning("obv(): input series too short. Returning empty aligned series.")
        return pd.Series(index=close.index, dtype="float64", name=f"obv")
    
    # ---------- Core computation ----------
    close_f = close.astype("float64")
    vol_f = volume.fillna(0.0).astype("float64")

    # direction of price movement
    direction = np.sign(close_f.diff())
    # ensure the first value is NaN (no previous candle)
    direction.iloc[0] = np.nan

    obv_raw = (direction * vol_f).cumsum()
    obv_raw.iloc[0] = np.nan  # proper warm-up: first row NaN

    return pd.Series(obv_raw, index=close.index, dtype="float64", name=f"obv")

# ---------------------------------------------------------
def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """
    Williams %R (WILLR)
    - First n rows NaN (warm-up)
    - Uses (highest_n - close) / (highest_n - lowest_n)
    - Returns float32, deterministic
    """
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if high is None or low is None or close is None:
        logger.warning("williams_r(): one or more inputs are None. Returning empty series.")
        return pd.Series(dtype="float64", name=f"wr_{n}")
    
    # Case 2: insufficient length → return aligned empty series
    if len(high) < n or len(low) < n or len(close) < n:
        logger.warning("williams_r(): input series too short. Returning empty aligned series.")
        return pd.Series(index=close.index, dtype="float64", name=f"wr_{n}")
    
    # ---------- Core computation ----------
    high_f = high.astype("float64")
    low_f = low.astype("float64")
    close_f = close.astype("float64")

    highest = high_f.rolling(window=n, min_periods=n).max()
    lowest = low_f.rolling(window=n, min_periods=n).min()

    wr = -100.0 * (highest - close_f) / (highest - lowest).replace(0, np.nan)
    wr[ : n - 1] = np.nan  # proper warm-up

    return wr.rename(f"wr_{n}")

# ============================================================================= 15,16
# Part-8: PSAR & heikin_ashi
# ============================================================================= OK
def parabolic_sar_orig(
    high: pd.Series,
    low: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2
) -> pd.Series:
    """
    Professional-grade Parabolic SAR (Wilder)
    - First TWO rows are NaN (true warm-up)
    - No fake values, fully deterministic
    - Exact EP/AF reset behavior on trend reversal
    """
    h = high.astype("float64").values
    l = low.astype("float64").values
    n = len(h)
    out = np.full(n, np.nan, dtype="float64")

    # --- determine initial trend (Wilder rule)
    # if current close > previous close → uptrend; else downtrend
    uptrend = h[1] > h[0]

    # --- initialize EP & SAR
    if uptrend:
        ep = h[1]          # highest high so far
        sar = l[0]         # SAR starts from previous low
    else:
        ep = l[1]          # lowest low so far
        sar = h[0]         # SAR starts from previous high

    af = af_start

    # warm-up: first two rows NaN (professional rule)
    out[0] = np.nan
    out[1] = np.nan

    # --- main loop from candle #2
    for i in range(2, n):
        # compute next SAR
        sar = sar + af * (ep - sar)
        # clamp SAR into allowed region (to avoid penetration)
        if uptrend:
            sar = min(sar, l[i-1], l[i-2])
        else:
            sar = max(sar, h[i-1], h[i-2])

        # --- Trend logic
        if uptrend:
            # reversal?
            if l[i] < sar:
                # switch to downtrend
                uptrend = False
                sar = ep       # SAR jumps to previous EP
                ep = l[i]      # new EP
                af = af_start  # reset AF
            else:
                # continue uptrend
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:
            # reversal?
            if h[i] > sar:
                # switch to uptrend
                uptrend = True
                sar = ep
                ep = h[i]
                af = af_start
            else:
                # continue downtrend
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)
        out[i] = sar
    return out

@njit
def _parabolic_sar_njit_core(h, l, af_start, af_step, af_max):

    n = h.size
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        out[i] = np.nan

    if n < 3:
        return out

    uptrend = h[1] > h[0]

    if uptrend:
        ep = h[1]
        sar = l[0]
    else:
        ep = l[1]
        sar = h[0]

    af = af_start

    for i in range(2, n):
        sar = sar + af * (ep - sar)
        if uptrend:
            if sar > l[i-1]:
                sar = l[i-1]
            if sar > l[i-2]:
                sar = l[i-2]
        else:
            if sar < h[i-1]:
                sar = h[i-1]
            if sar < h[i-2]:
                sar = h[i-2]

        if uptrend:
            if l[i] < sar:
                uptrend = False
                sar = ep
                ep = l[i]
                af = af_start
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = af + af_step
                    if af > af_max:
                        af = af_max
        else:
            if h[i] > sar:
                uptrend = True
                sar = ep
                ep = h[i]
                af = af_start
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = af + af_step
                    if af > af_max:
                        af = af_max
        out[i] = sar
    return out

def parabolic_sar_njit(
    high: pd.Series,
    low: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2
) -> pd.Series:

    h = high.to_numpy(dtype=np.float64, copy=False)
    l = low.to_numpy(dtype=np.float64, copy=False)
    out = _parabolic_sar_njit_core(h, l, af_start, af_step, af_max)
    return out

# WRAPPER:
def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2
) -> pd.Series:
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series
    if high is None or low is None:
        logger.warning("sar(): one or more inputs are None. Returning empty series.")
        return pd.Series(dtype="float64", name="sar")
    
    # Case 2: insufficient length → return aligned empty series
    if len(high) < 3 or len(low) < 3:   # SAR نیاز به حداقل ۳ کندل دارد
        logger.warning("sar(): input series too short. Returning empty aligned series.")
        return pd.Series(index=high.index, dtype="float64", name="sar")
    
    # ---------- Core computation ----------    
    n = high.size
    if n < 1_000_000:
        sar = parabolic_sar_orig(high, low, af_start, af_step, af_max)
    else:
        sar = parabolic_sar_njit(high, low, af_start, af_step, af_max)
    return pd.Series(sar, index=high.index, dtype="float64", name="sar") 

# ---------------------------------------------------------
def heikin_ashi_numpy(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    
    n = len(close)

    # convert to numpy (fast, zero-copy)
    o = open_.to_numpy(dtype=np.float64, copy=False)
    h = high.to_numpy(dtype=np.float64, copy=False)
    l = low.to_numpy(dtype=np.float64, copy=False)
    c = close.to_numpy(dtype=np.float64, copy=False)

    # ha_close vectorized
    ha_close = (o + h + l + c) * 0.25

    # ha_open iterative but in pure NumPy (fast)
    ha_open = np.empty(n, dtype=np.float64)
    ha_open[0] = (o[0] + c[0]) * 0.5
    for i in range(1, n):
        ha_open[i] = 0.5 * (ha_open[i-1] + ha_close[i-1])

    # ha_high / ha_low vectorized
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low  = np.minimum.reduce([l, ha_open, ha_close])

    return ha_open, ha_high, ha_low, ha_close

@njit(cache=True)
def _heikin_ashi_njit_core(o, h, l, c):
    n = len(o)
    ha_open = np.empty(n, dtype=np.float64)
    ha_close = (o + h + l + c) * 0.25
    ha_open[0] = (o[0] + c[0]) * 0.5
    
    for i in range(1, n):
        ha_open[i] = 0.5 * (ha_open[i-1] + ha_close[i-1])
    
    # numba‑safe operations
    ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
    ha_low  = np.minimum(l, np.minimum(ha_open, ha_close))

    # ha_high = np.maximum.reduce([h, ha_open, ha_close])
    # ha_low  = np.minimum.reduce([l, ha_open, ha_close])
    return ha_open, ha_high, ha_low, ha_close

def heikin_ashi_njit(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
   
    o = open_.to_numpy(np.float64, copy=False)
    h = high.to_numpy(np.float64, copy=False)
    l = low.to_numpy(np.float64, copy=False)
    c = close.to_numpy(np.float64, copy=False)

    ha_open, ha_high, ha_low, ha_close = _heikin_ashi_njit_core(o, h, l, c)
    return ha_open, ha_high, ha_low, ha_close

# WRAPPER:
def heikin_ashi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    
    # ---------- Input validation ----------
    # Case 1: any input is None → return truly empty float64 series    
    if open_ is None or high is None or low is None or close is None or len(close) == 0:
        logger.warning("heikin_ashi(): one or more inputs are None. Returning empty series.")
        return (
            pd.Series(dtype="float64", name="ha_open"),
            pd.Series(dtype="float64", name="ha_high"),
            pd.Series(dtype="float64", name="ha_low"),
            pd.Series(dtype="float64", name="ha_close"),
        )    
    # ---------- Core computation ----------    
    n = close.size
    if n < 2_000_000:
        ha_o, ha_h, ha_l, ha_c = heikin_ashi_numpy(open_, high, low, close)
    else:
        ha_o, ha_h, ha_l, ha_c = heikin_ashi_njit(open_, high, low, close)

    return (
        pd.Series(ha_o, dtype="float64", name="ha_open"),
        pd.Series(ha_h, dtype="float64", name="ha_high"),
        pd.Series(ha_l, dtype="float64", name="ha_low"),
        pd.Series(ha_c, dtype="float64", name="ha_close"),
    )

# ============================================================================= 
# Part-9: Registry
# =============================================================================
# رجیستریِ core (نام → تابع سازندهٔ map از series)
IndicatorMap = Dict[str, Callable]

def registry() -> IndicatorMap:
    # def wrap(name: str, s: pd.Series) -> Dict[str, pd.Series]:
    #     return {name: s.astype("float32")}
    def cast32(d: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        return {k: pd.Series(v, copy=False).astype("float32") for k, v in d.items()}
    
    #--- 1,2,3 ---
    def make_sma(df: pd.DataFrame, col: str = "close", period: int = 20, **_):
        return cast32({
            f"sma_{col}_{period}": sma(df[col], period)
            })
    def make_ema(df: pd.DataFrame, col: str = "close", period: int = 20, **_):
        return cast32({
            f"ema_{col}_{period}": ema(df[col], period)
            })
    def make_wma(df: pd.DataFrame, col: str = "close", period: int = 20, **_):
        return cast32({
            f"wma_{col}_{period}": wma(df[col], period)
            })
    
    #--- 4,5,6 ---
    def make_rsi(df: pd.DataFrame, period: int = 14, **_):
        return cast32({
            f"rsi_{period}": rsi(df["close"], period)
            })
    def make_roc(df: pd.DataFrame, n: int = 10, **_):
        return cast32({
            f"roc_{n}": roc(df["close"], n)
            })
    def make_atr(df: pd.DataFrame, n: int = 14, **_):
        return cast32({
            f"atr_{n}": atr(df["high"], df["low"], df["close"], n)
            })
    
    #--- 7,8,9 ---
    def make_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **_):
        line, sig, hist = macd(df["close"], fast, slow, signal)
        return cast32({f"macd_{fast}_{slow}_{signal}": line,
                f"macd_signal_{fast}_{slow}_{signal}": sig,
                f"macd_hist_{fast}_{slow}_{signal}": hist})
    def make_bbands(df: pd.DataFrame, col: str = "close", period: int = 20, k: float = 2.0, **_):
        up, mid, lo = bollinger(df[col], period, k)
        return cast32({f"bb_mid_{period}_{k}": mid,
                f"bb_up_{period}_{k}": up,
                f"bb_lo_{period}_{k}": lo})
    def make_keltner(df: pd.DataFrame, period: int = 20, m: float = 2.0, **_):
        up, mid, lo = keltner(df["high"], df["low"], df["close"], period, m)
        return cast32({f"kelt_mid_{period}_{m}": mid,
                f"kelt_up_{period}_{m}": up,
                f"kelt_lo_{period}_{m}": lo})
    
    #--- 10,11,12 ---
    def make_stoch(df: pd.DataFrame, n: int = 14, d: int = 3, **_):
        k, dline = stochastic(df["high"], df["low"], df["close"], n, d)
        return cast32({
            f"stoch_k_{n}_{d}": k,
            f"stoch_d_{n}_{d}": dline
            })
    def make_cci(df: pd.DataFrame, n: int = 20, **_):
        return cast32({
            f"cci_{n}": cci(df["high"], df["low"], df["close"], n)
            })
    def make_mfi(df: pd.DataFrame, n: int = 14, **_):
        return cast32({
            f"mfi_{n}": mfi(df["high"], df["low"], df["close"], df["volume"], n)
            })
    
    #--- 13,14,15 ---
    def make_obv(df: pd.DataFrame, **_):
        return cast32({
            "obv": obv(df["close"], df["volume"])
            })
    def make_wr(df: pd.DataFrame, n: int = 14, **_):
        return cast32({
            f"wr_{n}": williams_r(df["high"], df["low"], df["close"], n)
            })
    def make_sar(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2, **_):
        return cast32({
            f"sar_{af_start}_{af_step}_{af_max}": parabolic_sar(df["high"], df["low"], af_start, af_step, af_max)
            })
    
    #--- 16,17 ---
    def make_ha(df: pd.DataFrame, **_):
        o,h,l,c = heikin_ashi(df["open"], df["high"], df["low"], df["close"])
        return cast32({
            "ha_open": o,
            "ha_high": h,
            "ha_low": l,
            "ha_close": c
            })
    def make_tr(df: pd.DataFrame, **_):
        return cast32({
            "tr": true_range(df["high"], df["low"], df["close"])
            })

    
    return {
        "sma": make_sma,
        "ema": make_ema,
        "wma": make_wma,

        "rsi": make_rsi,
        "roc": make_roc,
        "atr": make_atr,

        "macd": make_macd,
        "bbands": make_bbands,
        "keltner": make_keltner,

        "stoch": make_stoch,
        "cci": make_cci,
        "mfi": make_mfi,

        "obv": make_obv,
        "wr": make_wr,
        "sar": make_sar,

        "ha": make_ha,
        "tr": make_tr,  # 17: true_range
    }


# =====================================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# =====================================================================================
""" Func Names                           Used in Functions: ...
                    1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
1  sma             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
2  ema             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
3  wma             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
4  rsi             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
5  roc             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
6  atr             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
7  macd            --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
8  bollinger       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
9  keltner         --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
10 stochastic      --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
11 cci             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
12 mfi             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
13 obv             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
14 williams_r      --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
15 parabolic_sar   --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
16 heikin_ashi     --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
17 registry        --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
"""