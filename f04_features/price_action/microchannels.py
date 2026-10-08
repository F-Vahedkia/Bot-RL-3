# -*- coding: utf-8 -*-
# f04_features/price_action/microchannels.py
# Status in (Bot-RL-3H): Completed

"""
Price Action — Micro Channels
=============================
تشخیص میکروکانال‌های صعودی/نزولی.
'open' دیگر الزامی نیست؛ اگر نبود، از close.shift(1) به‌عنوان جایگزین امن استفاده می‌شود.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def build_microchannels(
    df: pd.DataFrame,
    *,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    min_len: int = 3,
    near_extreme_thr: float = 0.2,
    anti_lookahead: bool = True,
) -> pd.DataFrame:
    """
    تشخیص میکروکانال‌ها با معیارهای ساده و قابل‌گسترش.

    - near_extreme_thr: اگر (high-close)/range <= thr → close near high
                        اگر (close-low)/range <= thr → close near low
    - اگر ستون 'open' وجود نداشت، از close.shift(1) به‌عنوان جایگزین استفاده می‌شود.
    """
    # فقط high/low/close الزامی‌اند
    for col in (high_col, low_col, close_col):
        if col not in df.columns:
            raise ValueError(f"'{col}' not found in DataFrame.")

    out = df.copy()
    # fallback امن برای open (در صورت نبود)
    if open_col in out.columns:
        o = out[open_col].astype("float64")
    else:
        # جایگزین: کلوز کندل قبلی (یا خود کلوز در اولین ردیف)
        o = out[close_col].astype("float64").shift(1).fillna(out[close_col].astype("float64"))

    h = out[high_col].astype("float64")
    l = out[low_col].astype("float64")
    c = out[close_col].astype("float64")

    rng = (h - l).replace(0.0, np.nan)
    near_high = ((h - c).abs() / rng) <= near_extreme_thr
    near_low = ((c - l).abs() / rng) <= near_extreme_thr

    # جهت‌داری low/high
    higher_low = l > l.shift(1)
    lower_high = h < h.shift(1)

    up_flag = pd.Series(0, index=out.index, dtype="int8")
    dn_flag = pd.Series(0, index=out.index, dtype="int8")
    length = pd.Series(0, index=out.index, dtype="int32")
    quality = pd.Series(0.0, index=out.index, dtype="float32")

    up_len = 0
    dn_len = 0
    for i in range(len(out)):
        up_condition = bool((near_high.iloc[i] if pd.notna(near_high.iloc[i]) else False) or
                            (higher_low.iloc[i] if pd.notna(higher_low.iloc[i]) else False))
        if up_condition:
            up_len += 1
        else:
            up_len = 0

        dn_condition = bool((near_low.iloc[i] if pd.notna(near_low.iloc[i]) else False) or
                            (lower_high.iloc[i] if pd.notna(lower_high.iloc[i]) else False))
        if dn_condition:
            dn_len += 1
        else:
            dn_len = 0

        if up_len >= min_len and up_len >= dn_len:
            up_flag.iat[i] = 1
            length.iat[i] = up_len
            span = range(max(0, i - up_len + 1), i + 1)
            nh = near_high.iloc[list(span)].fillna(False).mean()
            hl = higher_low.iloc[list(span)].fillna(False).mean()
            quality.iat[i] = float(np.clip(0.5 * nh + 0.5 * hl, 0.0, 1.0))
            dn_flag.iat[i] = 0
        elif dn_len >= min_len:
            dn_flag.iat[i] = 1
            length.iat[i] = dn_len
            span = range(max(0, i - dn_len + 1), i + 1)
            nl = near_low.iloc[list(span)].fillna(False).mean()
            lh = lower_high.iloc[list(span)].fillna(False).mean()
            quality.iat[i] = float(np.clip(0.5 * nl + 0.5 * lh, 0.0, 1.0))
            up_flag.iat[i] = 0
        else:
            up_flag.iat[i] = 0
            dn_flag.iat[i] = 0
            length.iat[i] = 0
            quality.iat[i] = 0.0

    if anti_lookahead:
        up_flag = up_flag.shift(1).fillna(0).astype("int8")
        dn_flag = dn_flag.shift(1).fillna(0).astype("int8")
        length = length.shift(1).fillna(0).astype("int32")
        quality = pd.Series(quality).shift(1).fillna(0.0).astype("float32")

    out["micro_channel_up"] = up_flag
    out["micro_channel_down"] = dn_flag
    out["micro_channel_len"] = length
    out["micro_channel_quality"] = quality

    return out
