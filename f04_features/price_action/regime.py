# -*- coding: utf-8 -*-
# f04_features/price_action/regime.py
# Status in (Bot-RL-3H): Delete _true_range() at 05/01/31

"""
Price Action — Regime Detector
==============================
این ماژول سه رِژیم ساختاری را برچسب‌گذاری می‌کند: Range / Spike / Channel
و شدت هر کدام را به صورت [0..1] برمی‌گرداند. برای پرهیز از دوباره‌کاری، در صورت
وجود ستون‌های کمکی (ATR، بازار ساختار از market_structure، یا اندیکاتورهای آماده)
از آن‌ها استفاده می‌کند؛ در غیر این صورت، محاسبات داخلی امن انجام می‌شود.

خروجی‌های استاندارد:
- regime_range      : شدتِ رِنج [0..1]
- regime_spike      : شدتِ اسپایک [0..1]
- regime_channel    : شدتِ کانال [0..1]
- regime_label      : {'range','spike','channel', None}
- regime_confidence : اطمینان برچسب [0..1]

نکات:
- anti_lookahead=True → شیفت +1 روی خروجی‌ها.
- پیام‌های runtime انگلیسی هستند.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from f04_features.indicators.utils import true_range

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _ensure_series(s, name=None) -> pd.Series:
    if isinstance(s, pd.Series):
        return s
    return pd.Series(s, name=name)

# def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
#     """ATR primitive: TR = max(high-low, |high-prevClose|, |low-prevClose|)."""
#     prev_close = close.shift(1)
#     tr = pd.concat([
#         (high - low).abs(),
#         (high - prev_close).abs(),
#         (low - prev_close).abs(),
#     ], axis=1).max(axis=1)
#     return tr

def _rolling_iqr_width(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Approx channel/range width via IQR over a rolling window."""
    r = x.rolling(window=window, min_periods=min_periods)
    q1 = r.quantile(0.25)
    q3 = r.quantile(0.75)
    return (q3 - q1)

def _norm01(s: pd.Series) -> pd.Series:
    return (s - s.min(skipna=True)) / (s.max(skipna=True) - s.min(skipna=True) + 1e-9)

def _clip01(s: pd.Series) -> pd.Series:
    return s.clip(0.0, 1.0).astype("float32")


# ------------------------------------------------------------
# Core detector
# ------------------------------------------------------------
def build_regime(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_window: int = 14,
    width_window: int = 20,
    slope_window: int = 12,
    spike_thr_atr: float = 1.8,
    channel_slope_thr: float = 0.20,
    anti_lookahead: bool = True,
) -> pd.DataFrame:
    """
    برچسب‌گذاری سه رِژیم: Range / Spike / Channel

    منطق خلاصه:
      - Range: عرض توزیع (IQR) پایین نسبت به ATR و نوسان حول میانگین → شدت بالا.
      - Spike: TR/ATR بسیار بزرگ (range expansion) → شدت بالا.
      - Channel: شیب پایدار close (یا سوئینگ‌ها) + جهت‌داری → شدت بالا.

    پارامترها:
      atr_window         : پنجره‌ی ATR داخلی
      width_window       : پنجره‌ی ارزیابی عرض (IQR) برای Range
      slope_window       : پنجره‌ی شیب نرم برای Channel
      spike_thr_atr      : آستانه نسبت TR/ATR برای تشخیص اسپایک
      channel_slope_thr  : آستانه‌ی بزرگی شیب نرمال برای کانال

    خروجی: ستون‌های regime_* و برچسب نهایی
    """
    if not {price_col, high_col, low_col}.issubset(df.columns):
        raise ValueError("Required columns not found in input DataFrame.")

    out = df.copy()
    c = out[price_col].astype("float64")
    h = out[high_col].astype("float64")
    l = out[low_col].astype("float64")

    # ---- ATR (داخلی در صورت نبود ستون atr) ----
    if "atr" in out.columns:
        atr = out["atr"].astype("float64")
    else:
        tr = true_range(h, l, c)
        # جلوگیری از هشدار «Mean of empty slice» هنگام داده‌ی کم
        if len(tr) < atr_window:
            atr = pd.Series(np.nan, index=tr.index, dtype="float32")
        else:
            atr = tr.rolling(window=atr_window, min_periods=max(2, atr_window // 2)).mean()


    # ---- Range intensity (عرض توزیع پایین → Range بیشتر) ----
    if len(c) < width_window:
        width = pd.Series(np.nan, index=c.index, dtype="float32")
    else:
        width = _rolling_iqr_width(c, window=width_window, min_periods=max(2, width_window // 2))
    
    
    # نسبت عرض به ATR → هر چه کمتر، رنج‌تر
    rel_width = (width / (atr + 1e-9)).replace([np.inf, -np.inf], np.nan)
    range_intensity = _clip01(1.0 - _norm01(rel_width.fillna(rel_width.median())))

    # ---- Spike intensity (TR/ATR بزرگ) ----
    tr = true_range(h, l, c)
    ratio_tr_atr = (tr / (atr + 1e-9)).replace([np.inf, -np.inf], np.nan)
    spike_intensity = _clip01(_norm01(ratio_tr_atr))
    # تقویت نقاطی که از آستانه spike_thr_atr عبور می‌کنند
    spike_intensity = _clip01(spike_intensity + (ratio_tr_atr > spike_thr_atr).astype("float32") * 0.25)

    # ---- Channel intensity (شیب نرم و جهت‌داری) ----
    # شیب جمع‌شونده‌ی کوتاه‌مدت
    slope = c.diff().rolling(window=slope_window, min_periods=max(2, slope_window // 2)).sum()
    # نرمال‌سازی با IQR
    r = slope.rolling(window=slope_window, min_periods=max(2, slope_window // 2))
    med = r.median()
    q1 = r.quantile(0.25)
    q3 = r.quantile(0.75)
    iqr = (q3 - q1).replace(0.0, np.nan)
    z_slope = ((slope - med) / iqr).clip(-2.0, 2.0) / 2.0  # [-1..+1]
    channel_intensity = _clip01(z_slope.abs())
    # تقویت اگر قدر مطلق شیب > آستانه
    channel_intensity = _clip01(channel_intensity + (z_slope.abs() > channel_slope_thr).astype("float32") * 0.15)

    # ---- ضد-لوک‌اِهد ----
    if anti_lookahead:
        range_intensity = range_intensity.shift(1)
        spike_intensity = spike_intensity.shift(1)
        channel_intensity = channel_intensity.shift(1)

    # ---- برچسب نهایی و اعتماد ----
    stack = pd.concat([range_intensity, spike_intensity, channel_intensity], axis=1)
    stack.columns = ["regime_range", "regime_spike", "regime_channel"]
    label_idx = stack.values.argmax(axis=1)
    map_label = {0: "range", 1: "spike", 2: "channel"}
    regime_label = pd.Series([map_label.get(i) if i==i else None for i in label_idx], index=out.index)
    confidence = stack.max(axis=1).fillna(0.0).astype("float32")

    out["regime_range"     ] = range_intensity  .fillna(0.0).astype("float32")
    out["regime_spike"     ] = spike_intensity  .fillna(0.0).astype("float32")
    out["regime_channel"   ] = channel_intensity.fillna(0.0).astype("float32")
    out["regime_label"     ] = regime_label                 .astype("object")
    out["regime_confidence"] = confidence

    return out

