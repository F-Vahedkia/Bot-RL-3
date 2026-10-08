

# =============================================================================
# From: extras_trend.py
# =============================================================================
from __future__ import annotations
from typing import Dict, Literal, Mapping, Callable
import numpy as np
import pandas as pd
from numba import njit
import logging
from datetime import datetime

from f04_features.indicators.core import sma, atr
from f04_features.indicators.utils import ema

# -----------------------------------------------
def _safe_div(num: pd.Series, den: pd.Series, eps: float = 1e-12) -> pd.Series:
    den_safe = den.where(den.abs() > eps, np.nan)
    out = num.divide(den_safe)
    return out.replace([np.inf, -np.inf], np.nan)


# --- One-Step Slope ----------------------------
def ma_slope(
    time_series: pd.Series,
    window: int = 20,
    method: Literal["sma", "ema"] = "ema",
    norm: Literal["none", "price", "stdev"] = "stdev",
    eps: float = 1e-12,
) -> pd.Series:
    """
    Normalized moving-average slope, suitable as a trend-strength feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, must contain `price_col`.
    price_col : str, default "close"
        Column to use for price.
    window : int, default 20
        MA and normalization window.
    method : {"sma", "ema"}, default "ema"
        Moving average type.
    norm : {"stdev", "price", "none"}, default "stdev"
        Normalization scheme:
          - "stdev": slope / rolling std(price)
          - "price": slope / |price|
          - "none" : raw slope
    eps : float, default 1e-12
        Numerical epsilon for denominator.

    Returns
    -------
    slope : pd.Series (float32)
        Normalized slope with name f"ma_slope_{method}_{window}".
    """
    # 1) --- initial checks ---
    # if price_col not in df.columns:
    #     raise ValueError(f"df must contain column '{price_col}'")

    # 2) --- extract price series ---
    # px = df[price_col].astype("float64")
    px = time_series.astype("float64")

    # 3) --- calculate moving average ---
    if method.lower() == "sma":
        ma = px.rolling(window=window, min_periods=max(2, window // 2)              ).mean()
    elif method.lower() == "ema":
        ma = px.ewm    (  span=window, min_periods=max(2, window // 2), adjust=False).mean()
    else:
        raise ValueError("method must be 'sma' or 'ema'")

    # 4) --- multi-step slope ---
    slope = ma.diff()

    # 5) --- normalization ---
    if norm == "none":
        pass
    elif norm == "price":
        slope = _safe_div(slope, px.abs(), eps)
    elif norm == "stdev":
        denom = px.rolling(window=window, min_periods=max(2, window // 2) ).std()
        slope = _safe_div(slope, denom, eps)
    else:
        raise ValueError("norm must be 'stdev', 'price', or 'none'")

    # 6) --- final result ---
    slope = slope.astype("float32")
    slope.name = f"ma_slope_{method}_{window}"
    return slope

# --- Multi-Step Slope --------------------------
def ma_slope_multistep(
    time_series: pd.Series,
    window: int = 20,
    method: Literal["sma", "ema"] = "ema",
    step: int = 5,
    norm: Literal["none", "price", "stdev"] = "stdev",
    eps: float = 1e-12,
) -> pd.Series:
    
    # 1) --- initial checks ---
    # if price_col not in df.columns:
    #     raise ValueError(f"df must contain column '{price_col}'")
    if step < 1:
        raise ValueError("step must be >= 1")

    # 2) --- extract price series ---
    # px = df[price_col].astype("float64")
    px = time_series.astype("float64")

    # 3) --- calculate moving average ---
    if method.lower() == "sma":
        ma = px.rolling(window=window, min_periods=max(2,window//2)).mean()
    elif method.lower() == "ema":
        ma = px.ewm(span=window, adjust=False, min_periods=max(2,window//2)).mean()
    else:
        raise ValueError("method must be 'sma' or 'ema'")

    # 4) --- multi-step slope ---
    slope = (ma - ma.shift(step)) / float(step)

    # 5) --- normalization ---
    if norm == "none":
        pass
    elif norm == "price":
        slope = _safe_div(slope, px.abs(), eps)
    if norm == "stdev":
        denom = px.rolling(window=window, min_periods=max(2,window//2)).std()
        slope = _safe_div(slope, denom, eps)
    else:
        raise ValueError("norm must be 'stdev', 'price', or 'none'")

    # 6) --- final result ---
    slope = slope.astype("float32")
    slope.name = f"ma_slope_{method}_{window}_step{step}"
    return slope


def ma_slope_multistep_selective_1(
    time_series: pd.Series,
    window: int,
    method: Literal["sma", "ema"] = "sma",
    step: int = 3,
    norm: Literal["none", "price", "stdev", "atr", "selective"] = "selective",
    norm_window: int = None,
    # -------------
    vol_mode: Literal["atr", "stdev"] = "atr",                    #| for norm = "selective"
    vol_threshold: float = 0.8,    # regime threshold (z-score)   #| for norm = "selective"
    eps: float = 1e-9,             #| for using in: _safe_div()
    min_periods: int = None,
    high: pd.Series = None,
    low: pd.Series = None,
    close: pd.Series = None,
):
    """
    Production-grade MA slope with selective normalization.
    slope = (MA_t - MA_{t-step}) / step
    """
    # 1) --- Initial check ---
    if norm_window is None:
        norm_window = window

    # 2) --- MA ---
    if method.lower() == "sma":
        ma = sma(time_series, window, min_periods=min_periods)
    elif method.lower() == "ema":
        ma = ema(time_series, window, min_periods=min_periods)
    else:
        raise ValueError("method must be 'sma' or 'ema'")
    
    # 3) --- multi-step slope ---
    slope = (ma - ma.shift(step)) / float(step)       # <== هسته اصلی

    # 4) --- normalization ---
    if norm == "none":
        out = slope

    elif norm == "price":
        out = _safe_div(num=slope, den=time_series, eps=eps)

    elif norm == "stdev":
        st = time_series.rolling(norm_window, min_periods=min_periods).std()
        out = _safe_div(num=slope, den=st, eps=eps)

    elif norm == "atr":
        if high is None or low is None or close is None:
            raise ValueError("ATR normalization requires high/low/close series.")
        atr_val = atr(high, low, close, norm_window, min_periods=min_periods)
        out = _safe_div(num=slope, den=atr_val, eps=eps)

    elif norm == "selective":
        # 4-A) --- volatility measure ---
        if vol_mode == "atr":
            if high is None or low is None or close is None:
                # fallback to "stdev" if "HLC" not available
                vol = time_series.rolling(norm_window, min_periods=min_periods).std()
            else:
                vol = atr(high, low, close, norm_window, min_periods=min_periods)
        elif vol_mode == "stdev":
            vol = time_series.rolling(norm_window, min_periods=min_periods).std()

        # 4-B) --- volatility regime z-score ---
        vol_mean = vol.rolling(norm_window, min_periods=min_periods).mean()
        vol_std  = vol.rolling(norm_window, min_periods=min_periods).std()
        # vol_z  = (vol - vol_mean) / (vol_std + eps)                    # old
        vol_z    = _safe_div(num=(vol - vol_mean), den=vol_std, eps=eps) # new

        # 4-C) --- regime rule ---
        high_vol = vol_z > vol_threshold

        # 4_D) --- normalize ---
        norm_high = vol          # ATR or stdev
        norm_low  = time_series  # price scaling

        # out = slope / np.where(high_vol, norm_high, norm_low)                          # old
        out = _safe_div(num=slope, den=np.where(high_vol, norm_high, norm_low), eps=eps) # new

    else:
        raise ValueError(f"Unknown norm mode: {norm}")

    # 5) --- output ---
    return ma, atr, out.astype("float32")
    # return out.astype("float32")

def ma_slope_multistep_selective_1B(
    time_series: pd.Series,
    window: int,
    method: Literal["sma", "ema"] = "sma",
    step: int = 3,
    norm: Literal["none", "price", "stdev", "atr", "selective"] = "selective",
    norm_window: int = None,
    # -------------
    vol_mode: Literal["atr", "stdev"] = "atr",                    #| for norm = "selective"
    vol_threshold: float = 0.8,    # regime threshold (z-score)   #| for norm = "selective"
    eps: float = 1e-9,             #| for using in: _safe_div()
    min_periods_ma: int = None,
    min_periods_norm: int = None,
    min_periods_volat: int = None,
    high: pd.Series = None,
    low: pd.Series = None,
    close: pd.Series = None,
):
    """
    Production-grade MA slope with selective normalization.
    slope = (MA_t - MA_{t-step}) / step
    """
    # 1) --- Initial check ---
    if norm_window is None:
        norm_window = window

    ma       = pd.Series(index=time_series.index, dtype="float32")
    vol      = pd.Series(index=time_series.index, dtype="float32")
    vol_mean = pd.Series(index=time_series.index, dtype="float32")
    vol_std  = pd.Series(index=time_series.index, dtype="float32")
    vol_z    = pd.Series(index=time_series.index, dtype="float32")
    out      = pd.Series(index=time_series.index, dtype="float32")

    # 2) --- MA ---
    if method.lower() == "sma":
        ma = sma(time_series, window, min_periods=min_periods_ma)
    elif method.lower() == "ema":
        ma = ema(time_series, window, min_periods=min_periods_ma)
    else:
        raise ValueError("method must be 'sma' or 'ema'")
    
    # 3) --- multi-step slope ---
    slope = (ma - ma.shift(step)) / float(step)       # <== هسته اصلی

    # 4) --- normalization ---
    if norm == "none":
        out = slope

    elif norm == "price":
        out = _safe_div(num=slope, den=time_series, eps=eps)

    elif norm == "stdev":
        st = time_series.rolling(norm_window, min_periods=min_periods_norm).std()
        out = _safe_div(num=slope, den=st, eps=eps)

    elif norm == "atr":
        if high is None or low is None or close is None:
            raise ValueError("ATR normalization requires high/low/close series.")
        atr_val = atr(high, low, close, norm_window, min_periods=min_periods_norm)
        out = _safe_div(num=slope, den=atr_val, eps=eps)

    elif norm == "selective":
        # 4-A) --- volatility measure ---
        if vol_mode == "atr":
            if high is None or low is None or close is None:
                # fallback to "stdev" if "HLC" not available
                vol = time_series.rolling(norm_window, min_periods=min_periods_norm).std()
            else:
                vol = atr(high, low, close, norm_window, min_periods=min_periods_norm)
        elif vol_mode == "stdev":
            vol = time_series.rolling(norm_window, min_periods=min_periods_norm).std()

        # 4-B) --- volatility regime z-score ---
        vol_mean = vol.rolling(norm_window, min_periods=min_periods_volat).mean()
        vol_std  = vol.rolling(norm_window, min_periods=min_periods_volat).std()
        # vol_z  = (vol - vol_mean) / (vol_std + eps)                    # old
        vol_z    = _safe_div(num=(vol - vol_mean), den=vol_std, eps=eps) # new

        # 4-C) --- regime rule ---
        high_vol = vol_z > vol_threshold

        # 4_D) --- normalize ---
        norm_high = vol          # ATR or stdev
        norm_low  = time_series  # price scaling

        # out = slope / np.where(high_vol, norm_high, norm_low)                          # old1
        # out = _safe_div(num=slope, den=np.where(high_vol, norm_high, norm_low), eps=eps) # old2
        den = norm_high.where(high_vol, norm_low)     # new
        out = _safe_div(num=slope, den=den, eps=eps)  # new

    else:
        raise ValueError(f"Unknown norm mode: {norm}")

    # 5) --- output ---
    return ma, vol, vol_mean, vol_std, vol_z, out.astype("float32")
    # return out.astype("float32")

def ma_slope_multistep_selective_2(
    time_series: pd.Series,
    window: int,
    method: Literal["sma", "ema"] = "sma",
    step: int = 3,
    norm: Literal["none", "price", "stdev", "atr", "selective"] = "selective",
    norm_window: int = None,
    # vol_mode: Literal["atr", "stdev"] = "atr",
    vol_threshold: float = 0.8,
    eps: float = 1e-9,
    min_periods: int = None,
    high: pd.Series = None,
    low: pd.Series = None,
    close: pd.Series = None,
):

    if norm_window is None:
        norm_window = window

    # ---------- MA ----------
    if method == "sma":
        ma = sma(time_series, window, min_periods=min_periods)
    elif method == "ema":
        ma = ema(time_series, window, min_periods=min_periods)
    else:
        raise ValueError("method must be 'sma' or 'ema'")

    # ---------- slope ----------
    slope = (ma - ma.shift(step)) / float(step)

    # ---------- simple modes ----------
    if norm == "none":
        return slope.astype("float32")

    if norm == "price":
        out = _safe_div(slope, time_series, eps)
        return out.astype("float32")

    # ---------- volatility (single source) ----------
    vol = None

    if norm in ("stdev", "selective"): # and vol_mode == "stdev":
        vol = time_series.rolling(norm_window, min_periods=min_periods).std()

    if norm in ("atr", "selective"): # and vol_mode == "atr":
        if high is None or low is None or close is None:
            raise ValueError("ATR requires high/low/close")
        vol = atr(high, low, close, norm_window, min_periods=min_periods)

    # ---------- pure volatility normalization ----------
    if norm in ("stdev", "atr"):
        out = _safe_div(slope, vol, eps)
        return ma, vol, out.astype("float32")
        # return out.astype("float32")

    # ---------- selective normalization ----------
    if norm == "selective":

        # volatility regime
        vol_mean = vol.rolling(norm_window, min_periods=min_periods).mean()
        vol_std  = vol.rolling(norm_window, min_periods=min_periods).std()
        vol_z = _safe_div(vol - vol_mean, vol_std, eps)

        high_vol = vol_z > vol_threshold  # filter
        norm_high = vol             # for high volatility
        norm_low  = time_series     # for low volatility

        den = np.where(high_vol, norm_high, norm_low)
        out = _safe_div(slope, den, eps)

        return out.astype("float32")

    raise ValueError(f"Unknown norm mode: {norm}")



# =============================================================================
# From: utils.py
# =============================================================================
from typing import Optional
from f04_features.indicators.zigzag import zigzag_wrapper as zigzag
from f04_features.indicators.utils import detect_swings
""" --------------------------------------------------------------------------- OK Func13
Online: وضعیت لحظه‌ای قبل از تأیید پیوت =>  مخصوص UI و live.
Leg-Based: تحلیل لگ‌ها بعد از پیوتها=> مناسب فیچرهای پیشرفته.
MTF: نگاشت Swing های HTF به LTF => برای multi-timeframe features.
"""
def detect_swings_online(
    high: pd.Series,
    low: pd.Series,
    *,
    depth: int,
    deviation: float = 0.0,
    backstep: Optional[int] = None,
    atr: Optional[pd.Series] = None,
    atr_mult: Optional[float] = None,
    tf: Optional[str] = None,
) -> dict:
    """
    کاربرد: پردازش زنده (Real-Time)

    زمانی که هنوز کندل‌ها کامل نشده‌اند.
    اندیکاتور یا ربات باید حرکت در حال توسعه را هم ببیند.
    مناسب برای:
    نمایش لحظه‌ای در UI
    تصمیمات intrabar
    تحلیل جریان سفارشات (flow)
    نه برای فیچرهای ML.
    """

    """
    Online swing detector.
    Returns current active leg, plus confirmed pivots.

    Output:
      {
        'confirmed_swings': DataFrame([...]),
        'developing_leg': {
            'direction': 'up' | 'down',
            'start_time': Timestamp,
            'start_price': float,
            'current_price': float,
            'progress_ratio': float
        }
      }
    """

    if not high.index.equals(low.index) or len(high) < depth + 2:
        return {"confirmed_swings": pd.DataFrame(), "developing_leg": None}

    zz = zigzag(high=high, low=low, depth=depth, deviation=deviation, backstep=backstep or depth)
    piv = zz.loc[zz["state"] != 0, ["state", "high_zz", "low_zz"]].copy()

    piv["kind"] = np.where(piv["state"] == 1, "H", "L")
    piv["price"] = np.where(piv["state"] == 1, piv["high_zz"], piv["low_zz"])
    piv["tf"] = tf
    piv.index = pd.to_datetime(piv.index, utc=True)

    # ATR filter
    if atr is not None and atr_mult and atr_mult > 0:
        piv["atr"] = atr.reindex(piv.index)
        prev = piv["price"].shift(1)
        piv = piv[(piv["price"] - prev).abs() >= piv["atr"] * atr_mult]

    # developing leg
    if len(piv) >= 1:
        last_kind = piv["kind"].iloc[-1]
        direction = "up" if last_kind == "L" else "down"
        start_time = piv.index[-1]
        start_price = piv["price"].iloc[-1]
        current_price = high.iloc[-1] if direction == "up" else low.iloc[-1]
        progress_ratio = abs(current_price - start_price) / (atr.iloc[-1] if atr is not None else (abs(current_price - start_price) + 1e-9))
        developing_leg = dict(direction=direction, start_time=start_time, start_price=start_price,
                              current_price=current_price, progress_ratio=float(progress_ratio))
    else:
        developing_leg = None

    return {"confirmed_swings": piv, "developing_leg": developing_leg}

def detect_swing_legs(
    high: pd.Series,
    low: pd.Series,
    *,
    depth: int,
    deviation: float = 0.0,
    backstep: Optional[int] = None,
    atr: Optional[pd.Series] = None,
    tf: Optional[str] = None,
) -> pd.DataFrame:
    """
    کاربرد: تحلیل ساختاری و فیزیک حرکت قیمت

    بعد از pivotها ساخته می‌شود.
    خروجی: طولِ لگ، دامنه، زاویه، سرعت، مدت، شتاب.
    مناسب برای:
    ساخت فیچرهای advanced
    تحلیل momentum/energy
    الگوهای موجی
    بازسازی ساختار CHoCH / BOS
    منبع آن: pivotهای نسخه Base.
    """
    
    """
    Convert zigzag pivots to explicit swing legs.
    Each row represents a movement between two pivots.
    """

    zz = zigzag(high=high, low=low, depth=depth, deviation=deviation, backstep=backstep or depth)
    piv = zz.loc[zz["state"] != 0, ["state", "high_zz", "low_zz"]].copy()
    piv["price"] = np.where(piv["state"] == 1, piv["high_zz"], piv["low_zz"])
    piv.index = pd.to_datetime(piv.index, utc=True)

    legs = []
    for i in range(1, len(piv)):
        start_idx, end_idx = piv.index[i - 1], piv.index[i]
        start_price, end_price = piv["price"].iloc[i - 1], piv["price"].iloc[i]
        direction = "up" if end_price > start_price else "down"
        delta = end_price - start_price
        pct = (delta / start_price) * 100
        duration = (end_idx - start_idx).total_seconds()
        atr_val = atr[end_idx] if atr is not None and end_idx in atr.index else np.nan
        legs.append([start_idx, end_idx, start_price, end_price, direction, delta, pct, duration, atr_val])

    out = pd.DataFrame(legs, columns=["start", "end", "price_start", "price_end",
                                     "direction", "delta", "pct_change", "duration_s", "atr"])
    out["tf"] = tf
    return out

def detect_swings_mtf(
    data_dict: dict,
    *,
    depth_map: dict,
    deviation_map: Optional[dict] = None,
    atr_map: Optional[dict] = None,
    atr_mult: Optional[float] = None,
) -> dict:
    """
    کاربرد: دریافت Swing های تایم‌فریم بالاتر روی داده‌های تایم‌فریم پایین‌تر

    برای ساخت فیچرهای سلسله‌مراتبی (HTF+LTF).
    مناسب برای:
    Multi-Resolution Signals
    Context Building
    ترکیب ساختار روزانه/ساعتی با تریگر دقیقه‌ای
    نیازمند: خروجی نسخه Base در تایم‌فریم‌های مختلف.
    """

    """
    Multi-Timeframe swing detector.

    Parameters:
        data_dict    : {'1h': (high_1h, low_1h), '4h': (...), ...}
        depth_map    : {'1h': 20, '4h': 30, ...}
        deviation_map: optional deviations per TF
        atr_map      : optional ATR series per TF
        atr_mult     : optional ATR filtering multiplier

    Returns:
        {'1h': swings_df_1h, '4h': swings_df_4h, ...}
    """

    results = {}

    for tf, (high, low) in data_dict.items():
        depth = depth_map.get(tf, 20)
        deviation = deviation_map.get(tf, 0.0) if deviation_map else 0.0
        atr = atr_map.get(tf) if atr_map else None

        df = detect_swings(
            high=high,
            low=low,
            depth=depth,
            deviation=deviation,
            atr=atr,
            atr_mult=atr_mult,
            tf=tf,
        )

        results[tf] = df

    return results



# =============================================================================
# From: zigzag.py    این نسخه ها فقط برای تست بودند. از آنها فقط v3 انتخاب شد.
# =============================================================================
def _zigzag_mql_numpy_complete_v2(
    high: np.ndarray,
    low: np.ndarray,
    depth: int = 12,
    deviation: float = 0.05,
    backstep: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    بله — در این نسخه همچنان از همان Hard‑Confirmation استفاده شده است.
    اما با یک تفاوت مهم:
    در زیگزاگ کلاسیک (MQL)، hard‑confirmation یعنی:
    «وقتی یک pivot در جهت مخالف پیدا شد، pivot قبلی را تأیید کن و اجازه بده که repaint رخ دهد.»
    در نسخه‌ی من:
    «وقتی یک pivot در جهت مخالف مشاهده شد، تنها زمانِ تأیید (confirmed_at[pos] = i) ثبت می‌شود،
    اما هیچ تغییری در گذشته (zz_buffer یا state_series قبل از i) داده نمی‌شود.»

    نتیجه
    نوع تأیید همان Hard‑Confirmation است(تأیید فقط وقتی رخ می‌دهد که جهت واقعاً عوض شود)
    اما اثر جانبی مخربش حذف شده است(یعنی دیگر repaint / پاک شدن pivot قبلی نداریم)
    اگر بخواهم در یک خط بگویم:
    همان Hard‑Confirmation زیگزاگ را داریم، اما بدون اینکه چیزی از گذشته تغییر کند.
    """
    n = len(high)
    zz_buffer    = np.zeros(n)
    state_series = np.zeros(n, dtype=np.int8)
    
    confirmed_at   = np.full(n, -1, dtype=np.int32)
    developing_leg = np.zeros(n, dtype=np.int8)

    # ---------------------------------
    # phase-1: calculating local highs/lows
    #          by means of depth, deviation, backstep
    # ---------------------------------
    low_windows  = np.lib.stride_tricks.sliding_window_view(low, depth)
    high_windows = np.lib.stride_tricks.sliding_window_view(high, depth)

    low_idx_in_window  = np.argmin(low_windows, axis=1)
    high_idx_in_window = np.argmax(high_windows, axis=1)

    low_vals  = low[np.arange(depth - 1, n) - (depth - 1) + low_idx_in_window]
    high_vals = high[np.arange(depth - 1, n) - (depth - 1) + high_idx_in_window]

    # بافرهای اصلی
    low_map  = np.zeros(n)
    high_map = np.zeros(n)

    for i in range(depth - 1, n):
        # فیلتر LOW با deviation و backstep
        l_val = low_vals[i - (depth - 1)]
        if (low[i] - l_val) <= deviation:
            back_range = slice(max(0, i - backstep), i)
            mask = (low_map[back_range] != 0) & (low_map[back_range] > l_val)
            low_map[np.where(mask)[0] + back_range.start] = 0.0
        if low[i] == l_val: low_map[i] = l_val

        # فیلتر HIGH با deviation و backstep
        h_val = high_vals[i - (depth - 1)]
        if (h_val - high[i]) <= deviation:
            back_range = slice(max(0, i - backstep), i)
            mask = (high_map[back_range] != 0) & (high_map[back_range] < h_val)
            high_map[np.where(mask)[0] + back_range.start] = 0.0
        if high[i] == h_val: high_map[i] = h_val

    # ---------------------------------
    # مرحله ۲: Resolve نهایی (بدون Look-ahead)
    # ---------------------------------
    last_high = 0.0
    last_low = 0.0
    last_high_pos = -1
    last_low_pos = -1
    search_mode = 0 

    for i in range(depth - 1, n):
        l_cand = low_map[i]  # کاندیدای فیلتر شده
        h_cand = high_map[i] # کاندیدای فیلتر شده

        if search_mode == 1: developing_leg[i] = 1
        elif search_mode == -1: developing_leg[i] = -1

        if search_mode == 0:
            if h_cand != 0:
                last_high = h_cand; last_high_pos = i
                zz_buffer[i] = h_cand; state_series[i] = -1; search_mode = -1
            elif l_cand != 0:
                last_low = l_cand; last_low_pos = i
                zz_buffer[i] = l_cand; state_series[i] = 1; search_mode = 1

        elif search_mode == 1: # دنبال کف هستیم
            if l_cand != 0 and (last_low == 0 or l_cand < last_low):
                # در حالت لایو، ما فقط پایین‌ترین کف دیده شده در این لگ را ثبت می‌کنیم
                last_low = l_cand; last_low_pos = i
                zz_buffer[i] = l_cand; state_series[i] = 1
            elif h_cand != 0:
                # تایید بازگشت: وقتی سقف جدیدی در جهت مخالف پیدا شود
                if last_low_pos >= 0: confirmed_at[last_low_pos] = i
                last_high = h_cand; last_high_pos = i
                zz_buffer[i] = h_cand; state_series[i] = -1; search_mode = -1

        elif search_mode == -1: # دنبال سقف هستیم
            if h_cand != 0 and (last_high == 0 or h_cand > last_high):
                last_high = h_cand; last_high_pos = i
                zz_buffer[i] = h_cand; state_series[i] = -1
            elif l_cand != 0:
                if last_high_pos >= 0: confirmed_at[last_high_pos] = i
                last_low = l_cand; last_low_pos = i
                zz_buffer[i] = l_cand; state_series[i] = 1; search_mode = 1

    high_actual = np.where(state_series == -1, high, 0)
    low_actual  = np.where(state_series == +1, low , 0)

    return state_series, high_actual, low_actual, confirmed_at, developing_leg


def _zigzag_mql_numpy_complete_v3(
    high: np.ndarray,
    low: np.ndarray,
    depth: int = 12,
    deviation: float = 0.05,
    backstep: int = 3,
    mode_confirmation: Literal ["harh", "soft"]  = "soft",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    n = len(high)

    zz_buffer    = np.zeros(n)
    state_series = np.zeros(n, dtype=np.int8)

    confirmed_at   = np.full(n, -1, dtype=np.int32)
    developing_leg = np.zeros(n, dtype=np.int8)

    # -----------------------------
    # phase 1 : candidate detection
    # -----------------------------
    low_windows  = np.lib.stride_tricks.sliding_window_view(low, depth)
    high_windows = np.lib.stride_tricks.sliding_window_view(high, depth)

    low_idx_in_window  = np.argmin(low_windows, axis=1)
    high_idx_in_window = np.argmax(high_windows, axis=1)

    base = np.arange(depth - 1, n) - (depth - 1)

    low_vals  = low[base + low_idx_in_window]
    high_vals = high[base + high_idx_in_window]

    low_map  = np.zeros(n)
    high_map = np.zeros(n)

    for i in range(depth - 1, n):

        l_val = low_vals[i - (depth - 1)]
        if (low[i] - l_val) <= deviation:
            r = slice(max(0, i - backstep), i)
            m = (low_map[r] != 0) & (low_map[r] > l_val)
            low_map[np.where(m)[0] + r.start] = 0.0
        if low[i] == l_val:
            low_map[i] = l_val

        h_val = high_vals[i - (depth - 1)]
        if (h_val - high[i]) <= deviation:
            r = slice(max(0, i - backstep), i)
            m = (high_map[r] != 0) & (high_map[r] < h_val)
            high_map[np.where(m)[0] + r.start] = 0.0
        if high[i] == h_val:
            high_map[i] = h_val

    # -----------------------------
    # phase 2 : resolve
    # -----------------------------
    last_high = 0.0
    last_low = 0.0
    last_high_pos = -1
    last_low_pos = -1

    search_mode = 0

    for i in range(depth - 1, n):

        l_cand = low_map[i]
        h_cand = high_map[i]

        if search_mode == 1:
            developing_leg[i] = 1
        elif search_mode == -1:
            developing_leg[i] = -1

        # -----------------
        # initial pivot
        # -----------------
        if search_mode == 0:

            if h_cand != 0:
                last_high = h_cand
                last_high_pos = i

                zz_buffer[i] = h_cand
                state_series[i] = -1
                search_mode = -1

            elif l_cand != 0:
                last_low = l_cand
                last_low_pos = i

                zz_buffer[i] = l_cand
                state_series[i] = 1
                search_mode = 1

        # -----------------
        # searching LOW
        # -----------------
        elif search_mode == 1:

            if l_cand != 0 and (last_low == 0 or l_cand < last_low):

                # حذف extreme قبلی همان لگ
                if last_low_pos >= 0:
                    zz_buffer[last_low_pos] = 0
                    state_series[last_low_pos] = 0

                last_low = l_cand
                last_low_pos = i

                zz_buffer[i] = l_cand
                state_series[i] = 1

            confirm = False

            if h_cand != 0:
                confirm = True

            if mode_confirmation == "soft":
                if last_low_pos >= 0 and (high[i] - last_low) >= deviation:
                    confirm = True

            if confirm and h_cand != 0:
                if last_low_pos >= 0:
                    confirmed_at[last_low_pos] = i

                last_high = h_cand
                last_high_pos = i

                zz_buffer[i] = h_cand
                state_series[i] = -1

                search_mode = -1

        # -----------------
        # searching HIGH
        # -----------------
        elif search_mode == -1:

            if h_cand != 0 and (last_high == 0 or h_cand > last_high):

                if last_high_pos >= 0:
                    zz_buffer[last_high_pos] = 0
                    state_series[last_high_pos] = 0

                last_high = h_cand
                last_high_pos = i

                zz_buffer[i] = h_cand
                state_series[i] = -1

            confirm = False

            if l_cand != 0:
                confirm = True

            if mode_confirmation == "soft":
                if last_high_pos >= 0 and (last_high - low[i]) >= deviation:
                    confirm = True

            if confirm and l_cand != 0:

                if last_high_pos >= 0:
                    confirmed_at[last_high_pos] = i

                last_low = l_cand
                last_low_pos = i

                zz_buffer[i] = l_cand
                state_series[i] = 1

                search_mode = 1

    high_actual = np.where(state_series == -1, high, 0)
    low_actual  = np.where(state_series == 1 , low , 0)

    return state_series, high_actual, low_actual, confirmed_at, developing_leg


def _zigzag_mql_numpy_complete_v4(
    high: np.ndarray,
    low: np.ndarray,
    depth: int = 12,
    deviation: float = 0.05,
    backstep: int = 3,
    mode_confirmation: str = "hard",
):
    """
    ZigZag غیر ری‌پینت کامل — نسخه v4
    • همیشه بهترین کف/سقف بین دو pivot انتخاب می‌شود
    • پشتیبانی از hard و soft confirmation بدون ری‌پینت
    """

    n = len(high)

    zz = np.zeros(n)
    state = np.zeros(n, dtype=np.int8)
    confirmed_at = np.full(n, -1, dtype=np.int32)
    developing_leg = np.zeros(n, dtype=np.int8)

    # --------------------------
    # مرحله ۱: پیدا کردن کاندیدها
    # --------------------------
    low_windows = np.lib.stride_tricks.sliding_window_view(low, depth)
    high_windows = np.lib.stride_tricks.sliding_window_view(high, depth)

    low_idx_in = np.argmin(low_windows, axis=1)
    high_idx_in = np.argmax(high_windows, axis=1)

    base = np.arange(depth - 1, n) - (depth - 1)

    low_vals = low[base + low_idx_in]
    high_vals = high[base + high_idx_in]

    low_map = np.zeros(n)
    high_map = np.zeros(n)

    for i in range(depth - 1, n):

        # کف
        lv = low_vals[i - (depth - 1)]
        if (low[i] - lv) <= deviation:
            r = slice(max(0, i - backstep), i)
            m = (low_map[r] != 0) & (low_map[r] > lv)
            low_map[np.where(m)[0] + r.start] = 0
        if low[i] == lv:
            low_map[i] = lv

        # سقف
        hv = high_vals[i - (depth - 1)]
        if (hv - high[i]) <= deviation:
            r = slice(max(0, i - backstep), i)
            m = (high_map[r] != 0) & (high_map[r] < hv)
            high_map[np.where(m)[0] + r.start] = 0
        if high[i] == hv:
            high_map[i] = hv

    # --------------------------
    # مرحله ۲: Resolve با انتخاب بهترین نقطه لگ
    # --------------------------

    search_mode = 0   # 0=unknown, 1=low, -1=high

    # بهترین نقطهٔ لگ فعلی:
    leg_min = float('inf')
    leg_min_pos = -1

    leg_max = -float('inf')
    leg_max_pos = -1

    for i in range(depth - 1, n):

        l_cand = low_map[i]
        h_cand = high_map[i]

        # --------------------------
        # انتخاب اولین pivot
        # --------------------------
        if search_mode == 0:

            if h_cand != 0:
                search_mode = -1
                leg_max = h_cand
                leg_max_pos = i

            elif l_cand != 0:
                search_mode = 1
                leg_min = l_cand
                leg_min_pos = i

            continue

        # --------------------------
        # جستجوی کف
        # --------------------------
        if search_mode == 1:

            developing_leg[i] = 1

            # همیشه بهترین کف کل لگ را ذخیره کن
            if low[i] < leg_min:
                leg_min = low[i]
                leg_min_pos = i

            confirm = False

            # hard confirm → سقف واقعی
            if h_cand != 0:
                confirm = True

            # soft confirm → برگشت قیمت
            if mode_confirmation == "soft":
                if (high[i] - leg_min) >= deviation:
                    confirm = True

            if confirm:

                zz[leg_min_pos] = leg_min
                state[leg_min_pos] = 1
                confirmed_at[leg_min_pos] = i

                # شروع لگ جدید (جستجوی سقف)
                search_mode = -1
                leg_max = -float('inf')
                leg_max_pos = -1

            continue

        # --------------------------
        # جستجوی سقف
        # --------------------------
        if search_mode == -1:

            developing_leg[i] = -1

            if high[i] > leg_max:
                leg_max = high[i]
                leg_max_pos = i

            confirm = False
            if l_cand != 0:
                confirm = True

            if mode_confirmation == "soft":
                if (leg_max - low[i]) >= deviation:
                    confirm = True

            if confirm:

                zz[leg_max_pos] = leg_max
                state[leg_max_pos] = -1
                confirmed_at[leg_max_pos] = i

                search_mode = 1
                leg_min = float('inf')
                leg_min_pos = -1

            continue

    # --------------------------
    # خروجی High/Low Actual
    # --------------------------
    high_actual = np.where(state == -1, high, 0)
    low_actual = np.where(state == 1, low, 0)

    return state, high_actual, low_actual, confirmed_at, developing_leg



# =============================================================================
# From: 
# =============================================================================
