# -*- coding: utf-8 -*-
# f04_features/price_action/mtf_context.py
# Status in (Bot-RL-3H): Completed

"""
Multi-Timeframe (MTF) Context for Price Action
==============================================

این ماژول سیگنال‌های زمینه‌ای بین‌تایم‌فریمی تولید می‌کند تا در کنار فیچرهای پرایس‌اکشن
به‌عنوان قیود/تقویت‌کنندهٔ تصمیم استفاده شوند.

خروجی‌های اصلی:
- mtf_bias             : بایاس نرم [-1, +1] در تایم‌فریم بالادستی (Higher TF)
- mtf_bias_local       : بایاس نرم [-1, +1] در تایم‌فریم پایه (Base TF)
- mtf_conflict         : پرچم تضاد جهت‌ها (int8: 0/1)
- mtf_confluence_score : امتیاز همگرایی [0..1] (1 یعنی هم‌جهت کامل)
- mtf_strength         : شدت بایاس بالادستی [0..1] (بِیس‌لاین برای وزن‌دهی)
- mtf_meta_coverage    : پوشش دادهٔ مؤثر پس از warmup و شیفت (0..1)

نکات:
- ضد لوک‌اِهد واقعی: خروجی‌ها به‌صورت پیش‌فرض با shift(+1) تولید می‌شوند.
- بدون وابستگی بیرون از دامنهٔ f01..f14.
- طراحی امن: اگر df_higher ارائه نشود، از close پایه برای برآورد بایاس بالادستی استفاده می‌شود.

پیام‌های runtime انگلیسی هستند؛ توضیحات فارسی تنها در کامنت‌ها آمده‌اند.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = [
    "build_mtf_context",
    "compute_soft_bias",
    "compute_confluence",
]

# ================================================================
# Utilities (توابع کمکی داخلی)
# ================================================================
def _safe_series(x) -> pd.Series:
    """Ensure pandas Series."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        return pd.Series(x)
    raise TypeError("Input must be a pandas Series or array-like.")


def _robust_scale(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    مقیاس‌گذاری robust با IQR برای کاهش اثر نوفه/آوتلایر.
    خروجی می‌تواند NaN داشته باشد که در بالادست مدیریت می‌شود.
    """
    r = s.rolling(window=window, min_periods=min_periods)
    med = r.median()
    q1 = r.quantile(0.25)
    q3 = r.quantile(0.75)
    iqr = (q3 - q1).replace(0.0, np.nan)
    z = (s - med) / iqr
    return z


def _align_to_base(higher_series: pd.Series, base_index: pd.Index) -> pd.Series:
    """
    هم‌راستاسازی تقریبی سری Higher TF با ایندکس Base.
    روش: قطعه‌بندی ساده و ffill؛ برای جلوگیری از وابستگی به resample بیرونی.
    """
    s = _safe_series(higher_series).copy()
    s = s.reset_index(drop=True).ffill()
    n_base = len(base_index)
    n_high = max(1, len(s))
    # نسبت ساده‌ی نگاشت
    step = max(1, int(np.floor(n_base / n_high)))
    aligned = pd.Series(index=range(n_base), dtype="float32")
    j = 0
    for i in range(n_base):
        aligned.iloc[i] = s.iloc[min(j, n_high - 1)]
        if (i + 1) % step == 0 and j < n_high - 1:
            j += 1
    aligned.index = base_index
    return aligned.astype("float32")


# ================================================================
# Public helpers (توابع عمومی قابل استفاده مجدد)
# ================================================================
def compute_soft_bias(
    close: pd.Series,
    *,
    window: int = 8,
    min_periods: int = 4,
    clip_z: float = 2.0,
) -> pd.Series:
    """
    محاسبهٔ بایاس نرم در بازهٔ [-1, +1] از سری close.
    روش: شیب تجمعی کوتاه‌مدت → مقیاس‌گذاری robust (IQR) → کلیپ → نگاشت به [-1, +1]

    ورودی‌ها:
        close        : سری قیمت پایانی
        window       : پنجرهٔ محاسبهٔ شیب/آمار
        min_periods  : حداقل نمونه برای شروع محاسبات
        clip_z       : کلیپ کردن نمرهٔ Z در ±clip_z

    خروجی:
        pd.Series dtype float32 در بازهٔ [-1, +1]
    """
    c = _safe_series(close).astype("float64")
    grad = c.diff().rolling(window=window, min_periods=min_periods).sum()
    z = _robust_scale(grad, window=window, min_periods=min_periods)
    bias = (z.clip(-clip_z, clip_z) / clip_z).astype("float32")
    return bias


def compute_confluence(
    bias_local: pd.Series,
    bias_higher: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    محاسبهٔ (conflict_flag, confluence_score).
    - conflict_flag: 1 اگر علامت‌ها مخالف باشند، 0 در غیر این صورت.
    - confluence_score: 1 - |diff|/2 در بازهٔ [0..1]
    """
    bl = _safe_series(bias_local).fillna(0.0)
    bh = _safe_series(bias_higher).fillna(0.0)
    conflict = (np.sign(bl) * np.sign(bh) < 0).astype("int8")
    confluence = (1.0 - (np.abs(bl - bh) / 2.0)).clip(0.0, 1.0).astype("float32")
    return conflict, confluence


# ================================================================
# Main API
# ================================================================
def build_mtf_context(
    df_base: pd.DataFrame,
    *,
    df_higher: pd.DataFrame | None = None,
    base_close_col: str = "close",
    higher_close_col: str = "close",
    bias_window_base: int = 8,
    bias_window_higher: int = 8,
    anti_lookahead: bool = True,
    fillna_confluence: float = 0.5,
) -> pd.DataFrame:
    """
    تولید سیگنال‌های MTF برای پرایس‌اکشن.

    ورودی‌ها:
        df_base            : دیتافریم پایه (حداقل ستون 'close')
        df_higher          : دیتافریم بالادستی (اختیاری؛ در صورت نبود، از close پایه استفاده می‌شود)
        base_close_col     : نام ستون close در دیتای پایه
        higher_close_col   : نام ستون close در دیتای بالادستی
        bias_window_base   : پنجرهٔ محاسبهٔ بایاس پایه
        bias_window_higher : پنجرهٔ محاسبهٔ بایاس بالادستی
        anti_lookahead     : اعمال shift(+1) روی خروجی‌ها برای حذف لوک‌اِهد
        fillna_confluence  : مقدار جایگزین برای NaN در امتیاز همگرایی

    خروجی:
        df_base با ستون‌های افزوده‌شدهٔ:
            - mtf_bias
            - mtf_bias_local
            - mtf_conflict
            - mtf_confluence_score
            - mtf_strength
            - mtf_meta_coverage
    """
    if base_close_col not in df_base.columns:
        raise ValueError(f"'{base_close_col}' not found in df_base columns.")

    out = df_base.copy()

    # ----- Local bias (Base TF)
    bias_local = compute_soft_bias(
        out[base_close_col],
        window=bias_window_base,
        min_periods=max(2, bias_window_base // 2),
    )

    # ----- Higher bias (Higher TF or fallback)
    if (df_higher is not None) and (higher_close_col in df_higher.columns) and (len(df_higher) > 0):
        bias_higher_raw = compute_soft_bias(
            df_higher[higher_close_col],
            window=bias_window_higher,
            min_periods=max(2, bias_window_higher // 2),
        )
        mtf_bias = _align_to_base(bias_higher_raw, out.index)
    else:
        mtf_bias = compute_soft_bias(
            out[base_close_col],
            window=bias_window_higher,
            min_periods=max(2, bias_window_higher // 2),
        )

    # ----- Conflict & Confluence
    conflict, confluence = compute_confluence(bias_local, mtf_bias)

    # ----- Strength (شدت بایاس بالادستی) → [0..1]
    mtf_strength = mtf_bias.abs().clip(0.0, 1.0).astype("float32")

    # ----- Anti-lookahead
    if anti_lookahead:
        mtf_bias = mtf_bias.shift(1)
        bias_local = bias_local.shift(1)
        conflict = conflict.shift(1).fillna(0).astype("int8")
        confluence = confluence.shift(1)
        mtf_strength = mtf_strength.shift(1)

    # ----- Finalize columns
    out["mtf_bias"] = mtf_bias.astype("float32")
    out["mtf_bias_local"] = bias_local.astype("float32")
    out["mtf_conflict"] = conflict.astype("int8")
    out["mtf_confluence_score"] = confluence.fillna(fillna_confluence).clip(0.0, 1.0).astype("float32")
    out["mtf_strength"] = mtf_strength.fillna(0.0).astype("float32")

    # ----- Meta: coverage پس از warmup/shift
    valid_cols = ["mtf_bias", "mtf_bias_local", "mtf_confluence_score"]
    valid_mask = pd.Series(True, index=out.index)
    for c in valid_cols:
        valid_mask &= out[c].notna()
    coverage = float(valid_mask.mean()) if len(valid_mask) else 0.0
    out["mtf_meta_coverage"] = pd.Series([coverage] * len(out), index=out.index, dtype="float32")

    return out
