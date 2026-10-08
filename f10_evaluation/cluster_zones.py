# -*- coding: utf-8 -*-
# f10_evaluation/cluster_zones.py
# Status in (Bot-RL-3): Completed

r"""
تبدیل خروجی fib_cluster به «زون/سیگنال قابل‌مصرف» (Bot-RL-3)

کاربری:
- دریافت DataFrame خوشه‌ها (price_min/price_max/price_mean/members/score/...)
- ساخت زون‌ها (low/high/mid/width) + محاسبهٔ فاصله تا قیمت فعلی
- تخمین SL/TP و RR برای لانگ/شورت بر مبنای ATR یا پهنای زون

سیاست:
- کامنت‌ها فارسی؛ این ماژول خروجی/لاگ چاپ نمی‌کند (کتابخانه‌ای)
- وابستگی‌ها: numpy, pandas (بدون اتکا به ماژول‌های داخلی دیگر)
"""

from __future__ import annotations
from typing import Optional, Iterable, Dict, Any, List, Tuple
import numpy as np
import pandas as pd

__all__ = ["select_top_zones", "clusters_to_zones"]

# ------------------------ Helpers ------------------------

def _at_or_before(s: Optional[pd.Series], ts: Optional[pd.Timestamp]) -> float:
    """آخرین مقدار سری تا زمان ts (یا آخرین مقدار سری اگر ts=None)."""
    if s is None or len(s) == 0:
        return np.nan
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    if ts is None:
        return float(s.iloc[-1])
    ts = pd.to_datetime(ts, utc=True)
    s2 = s.loc[:ts]
    return float(s2.iloc[-1]) if len(s2) else np.nan

def _safe_div(n: float, d: float) -> float:
    """تقسیم امن (در صورت تقسیم‌برصفر → NaN)."""
    try:
        d = float(d)
        return float(n) / d if d != 0.0 else np.nan
    except Exception:
        return np.nan

# ------------------------ API ----------------------------

def select_top_zones(
    clusters: pd.DataFrame,
    top_k: int = 10,
    min_members: int = 1,
    min_score: Optional[float] = None,
) -> pd.DataFrame:
    """
    فیلتر و مرتب‌سازی خوشه‌ها:
    - حذف خوشه‌های با members < min_members
    - اگر min_score داده شود، حذف score کمتر
    - مرتب‌سازی: score نزولی، سپس price_mean صعودی
    - بازگرداندن top_k
    """
    if clusters is None or clusters.empty:
        return pd.DataFrame(columns=[
            "price_min","price_max","price_mean","members","tfs","ratios","score"
        ])

    df = clusters.copy()
    if "members" in df.columns:
        df = df[df["members"].fillna(0).astype(float) >= float(min_members)]
    if min_score is not None and "score" in df.columns:
        df = df[df["score"].fillna(-np.inf) >= float(min_score)]
    if df.empty:
        return df

    by: List[str] = []
    asc: List[bool] = []
    if "score" in df.columns:
        by.append("score"); asc.append(False)
    if "price_mean" in df.columns:
        by.append("price_mean"); asc.append(True)

    if by:
        df = df.sort_values(by=by, ascending=asc)

    return df.head(int(top_k)).reset_index(drop=True)


def clusters_to_zones(
    clusters: pd.DataFrame,
    current_price: float,
    atr_series: Optional[pd.Series] = None,
    ref_time: Optional[pd.Timestamp] = None,
    *,
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.0,
    top_k: int = 5,
    min_members: int = 1,
    min_score: Optional[float] = None,
    proximity_pct: float = 0.20,  # درصد: نزدیکی به قیمت (مثلاً 0.20%)
) -> pd.DataFrame:
    """
    تبدیل خوشه‌ها به زون معاملاتی + محاسبهٔ SL/TP و RR لانگ/شورت.

    ورودی‌ها:
      clusters: خروجی fib_cluster
      current_price: قیمت فعلی
      atr_series/ref_time: برای نمونه‌برداری ATR (اختیاری)
      ضرایب SL/TP: sl_atr_mult / tp_atr_mult
      فیلتر: top_k / min_members / min_score
      proximity_pct: پنجرهٔ نزدیکی به قیمت فعلی (٪)

    خروجی: DataFrame با ستون‌های
      zone_low, zone_high, zone_mid, zone_width,
      score, members, tfs, ratios,
      distance_to_price, within_proximity,
      long_entry_hint, long_sl, long_tp, long_rr,
      short_entry_hint, short_sl, short_tp, short_rr
    """
    sel = select_top_zones(clusters, top_k=top_k, min_members=min_members, min_score=min_score).copy()
    if sel.empty:
        return pd.DataFrame(columns=[
            "zone_low","zone_high","zone_mid","zone_width",
            "score","members","tfs","ratios",
            "distance_to_price","within_proximity",
            "long_entry_hint","long_sl","long_tp","long_rr",
            "short_entry_hint","short_sl","short_tp","short_rr",
        ])

    # ویژگی‌های پایهٔ زون
    sel["zone_low"]  = sel["price_min"].astype(float)
    sel["zone_high"] = sel["price_max"].astype(float)
    sel["zone_mid"]  = 0.5 * (sel["zone_low"] + sel["zone_high"])
    sel["zone_width"]= (sel["zone_high"] - sel["zone_low"]).astype(float)

    # فاصله تا قیمت و پرچم نزدیکی
    current_price = float(current_price)
    sel["distance_to_price"] = (sel["zone_mid"] - current_price).abs().astype(float)
    band_abs = current_price * (float(proximity_pct) / 100.0)
    sel["within_proximity"] = (
        (current_price >= (sel["zone_low"] - band_abs)) &
        (current_price <= (sel["zone_high"] + band_abs))
    )

    # ATR نمونه‌برداری‌شده (در صورت نبود → fallback به میانگین پهنای زون‌ها)
    atr_val = _at_or_before(atr_series, ref_time) if atr_series is not None else np.nan
    if (atr_series is None) or np.isnan(atr_val) or atr_val <= 0:
        zmean = float(sel["zone_width"].mean())
        atr_val = zmean if zmean > 0 else 0.0

    # لانگ
    sel["long_entry_hint"] = sel["zone_mid"].astype(float)
    sel["long_sl"] = (sel["zone_low"]  - sl_atr_mult * atr_val).astype(float)
    sel["long_tp"] = (sel["zone_high"] + tp_atr_mult * atr_val).astype(float)
    sel["long_rr"] = [
        _safe_div((tp - current_price), (current_price - sl))
        for tp, sl in zip(sel["long_tp"].tolist(), sel["long_sl"].tolist())
    ]

    # شورت
    sel["short_entry_hint"] = sel["zone_mid"].astype(float)
    sel["short_sl"] = (sel["zone_high"] + sl_atr_mult * atr_val).astype(float)
    sel["short_tp"] = (sel["zone_low"]  - tp_atr_mult * atr_val).astype(float)
    sel["short_rr"] = [
        _safe_div((current_price - tp), (sl - current_price))
        for tp, sl in zip(sel["short_tp"].tolist(), sel["short_sl"].tolist())
    ]

    # مرتب‌سازی: score↓ و سپس فاصله به قیمت ↑
    sort_by: List[str] = []; asc: List[bool] = []
    if "score" in sel.columns: sort_by.append("score"); asc.append(False)
    sort_by.append("distance_to_price"); asc.append(True)
    sel = sel.sort_values(by=sort_by, ascending=asc).reset_index(drop=True)

    keep = [
        "zone_low","zone_high","zone_mid","zone_width",
        "score","members",
    ]
    if "tfs" in sel.columns:    keep.append("tfs")
    if "ratios" in sel.columns: keep.append("ratios")
    keep += [
        "distance_to_price","within_proximity",
        "long_entry_hint","long_sl","long_tp","long_rr",
        "short_entry_hint","short_sl","short_tp","short_rr",
    ]
    return sel[keep]
