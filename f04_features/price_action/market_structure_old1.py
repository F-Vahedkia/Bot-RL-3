# -*- coding: utf-8 -*-
# f04_features/price_action/market_structure.py
# Status in (Bot-RL-3H): Completed

"""
Market Structure Features (Price Action)
========================================
- تشخیص سوئینگ‌های HH, HL, LH, LL
- شناسایی BOS و CHoCH
"""

import pandas as pd
from f04_features.indicators.levels import fractal_points  # از ماژول موجود استفاده می‌کنیم


def _extract_fractal_flags(swings, df: pd.DataFrame) -> pd.DataFrame:
    """
    استخراج پرچم‌های فراکتال high/low از خروجی‌های مختلفِ fractal_points.
    این تابع با چند حالت رایج سازگار است:
      1) DataFrame با ستون‌های 'fractal_high' و 'fractal_low'
      2) تاپل (fractal_high_series, fractal_low_series)
      3) DataFrame با نام‌های متفاوت ولی شامل دو ستون باینری
    """
    df = df.copy()
    fh = None
    fl = None

    # حالت 1: DataFrame با نام‌های استاندارد
    if isinstance(swings, pd.DataFrame):
        cols = [c.lower() for c in swings.columns]
        mapping = {c: col for c, col in zip(swings.columns, cols)}
        if "fractal_high" in cols:
            fh = swings[mapping[[c for c in swings.columns if mapping[c] == "fractal_high"][0]]]
        if "fractal_low" in cols:
            fl = swings[mapping[[c for c in swings.columns if mapping[c] == "fractal_low"][0]]]
        # اگر نام‌ها متفاوت بود، سعی کن دو ستون باینری پیدا کنی
        if fh is None or fl is None:
            # انتخاب دو ستون باینری/عددی
            bin_cols = [c for c in swings.columns if pd.api.types.is_integer_dtype(swings[c]) or pd.api.types.is_bool_dtype(swings[c])]
            if len(bin_cols) >= 2:
                fh = swings[bin_cols[0]]
                fl = swings[bin_cols[1]]

    # حالت 2: تاپل/لیست از دو سری
    elif isinstance(swings, (tuple, list)) and len(swings) >= 2:
        fh, fl = swings[0], swings[1]

    # نهایی‌سازی: اگر هنوز None هستند، ستون‌های صفر بساز
    if fh is None:
        fh = pd.Series([0] * len(df), index=df.index)
    if fl is None:
        fl = pd.Series([0] * len(df), index=df.index)

    df["__fractal_high__"] = (fh.astype("int8") > 0).astype("int8")
    df["__fractal_low__"] = (fl.astype("int8") > 0).astype("int8")
    return df


def detect_swings(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    تشخیص نقاط سوئینگ با تکیه بر fractal_points موجود در indicators.
    توجه: برخی نسخه‌ها پارامتر lookback را نمی‌پذیرند. اینجا سازگارانه عمل می‌کنیم.
    """
    base = df[["high", "low"]].copy()

    # تلاش برای عبور lookback؛ اگر پشتیبانی نشد، بدون آن فراخوانی می‌کنیم.
    try:
        swings = fractal_points(base["high"], base["low"], lookback)  # نوع 1: با آرگومان position-based
    except TypeError:
        try:
            swings = fractal_points(base["high"], base["low"])  # نوع 2: بدون lookback
        except TypeError:
            # آخرین شانس: شاید API فقط یک آرگومان DataFrame می‌گیرد
            swings = fractal_points(base)

    work = _extract_fractal_flags(swings, base)
    out = df.copy()
    out["swing_type"] = None

    last_high = None
    last_low = None

    for idx, row in work.iterrows():
        if row["__fractal_high__"] == 1:
            cur_high = df.at[idx, "high"]
            if last_high is None or cur_high > last_high:
                out.at[idx, "swing_type"] = "HH"
            else:
                out.at[idx, "swing_type"] = "LH"
            last_high = cur_high

        elif row["__fractal_low__"] == 1:
            cur_low = df.at[idx, "low"]
            if last_low is None or cur_low < last_low:
                out.at[idx, "swing_type"] = "LL"
            else:
                out.at[idx, "swing_type"] = "HL"
            last_low = cur_low

    return out


def detect_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
    """
    تشخیص Break of Structure و Change of Character بر مبنای برچسب‌های سوئینگ.
    """
    out = df.copy()
    out["bos_up"] = 0
    out["bos_down"] = 0
    out["choch_up"] = 0
    out["choch_down"] = 0

    last_structure = None

    for idx, swing in out["swing_type"].items():
        if swing is None:
            continue

        if last_structure is None:
            last_structure = swing
            continue

        # BOS صعودی: تداوم ساختار صعودی
        if swing == "HH" and last_structure in ("HL", "HH"):
            out.at[idx, "bos_up"] = 1

        # BOS نزولی: تداوم ساختار نزولی
        elif swing == "LL" and last_structure in ("LH", "LL"):
            out.at[idx, "bos_down"] = 1

        # CHoCH صعودی: تغییر شخصیت از نزولی به صعودی
        elif swing == "HH" and last_structure in ("LH", "LL"):
            out.at[idx, "choch_up"] = 1

        # CHoCH نزولی: تغییر شخصیت از صعودی به نزولی
        elif swing == "LL" and last_structure in ("HL", "HH"):
            out.at[idx, "choch_down"] = 1

        last_structure = swing

    return out


def build_market_structure(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    اجرای کامل تحلیل ساختار بازار: سوئینگ + BOS/CHoCH.
    """
    x = detect_swings(df, lookback=lookback)
    x = detect_bos_choch(x)
    return x

