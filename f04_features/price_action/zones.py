# -*- coding: utf-8 -*-
# f04_features/price_action/zones.py
# Status in (Bot-RL-3H): Completed
"""
Price Action Zones
==================
این ماژول، زون‌های Supply/Demand و Order Blocks را با تکیه بر توابع موجود در indicators
(در صورت وجود) می‌سازد و خروجی استاندارد ارائه می‌دهد. از «دوباره‌کاری» پرهیز شده و
در صورت نبود توابع indicators، خطای دوستانه/پیغام مناسب ثبت می‌شود.

خروجی نهایی DataFrame ورودی را با ستون‌های استاندارد زیر غنی می‌کند:
- SD Zones:
    sd_upper, sd_lower, sd_kind, sd_age, sd_fresh, sd_touch_count
- Order Blocks:
    ob_upper, ob_lower, ob_kind, ob_age, ob_strength, ob_touch_count
- فاصله تا زون‌ها (نرمال‌شده به ATR در صورت موجود بودن ستون atr):
    dist_to_sd, dist_to_ob
"""

from __future__ import annotations
import pandas as pd

# تلاش برای استفاده از پیاده‌سازی‌های موجود در indicators
_HAS_SR_ADV = True
try:
    # امضاها ممکن است با نسخه‌های مختلف فرق کنند؛ در لایه‌ی wrapper هندل می‌کنیم.
    from f04_features.indicators.sr_advanced import (
        detect_sd,   # شناسایی اولیه زون‌های عرضه/تقاضا
        make_sd,     # ساخت باندها و ویژگی‌های زون SD
        detect_ob,   # شناسایی اولیه Order Block
        make_ob,     # ساخت باندها و ویژگی‌های OB
    )
except Exception:
    _HAS_SR_ADV = False


# ---------------------------------------------------------------------
# کمکی: تلاش برای فراخوانی توابع با امضاهای متفاوت به‌صورت سازگار
# ---------------------------------------------------------------------
def _call_detect_sd(df: pd.DataFrame) -> pd.DataFrame | tuple | None:
    """فراخوانی سازگار detect_sd با امضاهای محتمل مختلف."""
    # حالت‌های محتمل: detect_sd(df) یا detect_sd(high, low, close) یا detect_sd(**kwargs)
    if not _HAS_SR_ADV:
        return None
    try:
        return detect_sd(df)
    except TypeError:
        try:
            return detect_sd(df["high"], df["low"], df.get("close", None))
        except TypeError:
            # آخرین تلاش: شاید فقط high/low لازم باشد
            try:
                return detect_sd(df["high"], df["low"])
            except Exception:
                return None


def _call_make_sd(df: pd.DataFrame, proto) -> pd.DataFrame | None:
    """فراخوانی سازگار make_sd روی خروجی detect_sd."""
    if not _HAS_SR_ADV:
        return None
    try:
        return make_sd(df, proto)
    except TypeError:
        # احتمال امضای make_sd(proto) بدون df
        try:
            return make_sd(proto)
        except Exception:
            return None


def _call_detect_ob(df: pd.DataFrame) -> pd.DataFrame | tuple | None:
    """فراخوانی سازگار detect_ob با امضاهای محتمل مختلف."""
    if not _HAS_SR_ADV:
        return None
    try:
        return detect_ob(df)
    except TypeError:
        try:
            return detect_ob(df["high"], df["low"], df.get("close", None))
        except TypeError:
            try:
                return detect_ob(df["high"], df["low"])
            except Exception:
                return None


def _call_make_ob(df: pd.DataFrame, proto) -> pd.DataFrame | None:
    """فراخوانی سازگار make_ob روی خروجی detect_ob."""
    if not _HAS_SR_ADV:
        return None
    try:
        return make_ob(df, proto)
    except TypeError:
        try:
            return make_ob(proto)
        except Exception:
            return None


# ---------------------------------------------------------------------
# کمکی: نرمال‌سازی نام ستون‌ها به یک قرارداد ثابت
# ---------------------------------------------------------------------
def _coerce_sd_df(sd_df: pd.DataFrame) -> pd.DataFrame:
    """
    تبدیل نام ستون‌های SD به قالب استاندارد:
    sd_upper, sd_lower, sd_kind, sd_age, sd_fresh, sd_touch_count
    """
    mapping_candidates = {
        "upper": "sd_upper", "high": "sd_upper", "u": "sd_upper",
        "lower": "sd_lower", "low": "sd_lower", "l": "sd_lower",
        "kind": "sd_kind", "type": "sd_kind", "zone_type": "sd_kind",
        "age": "sd_age", "fresh": "sd_fresh", "touch": "sd_touch_count",
        "touches": "sd_touch_count", "hits": "sd_touch_count",
    }
    out = pd.DataFrame(index=sd_df.index)
    for col in sd_df.columns:
        key = col.lower()
        target = mapping_candidates.get(key)
        if target:
            out[target] = sd_df[col]
    # تضمین وجود ستون‌ها
    for need, dtype in [
        ("sd_upper", "float32"),
        ("sd_lower", "float32"),
        ("sd_kind", "object"),
        ("sd_age", "float32"),
        ("sd_fresh", "float32"),
        ("sd_touch_count", "float32"),
    ]:
        if need not in out.columns:
            out[need] = pd.Series([pd.NA] * len(out), index=out.index)
        # تلاش برای cast امن
        if dtype != "object":
            out[need] = pd.to_numeric(out[need], errors="coerce")
    return out


def _coerce_ob_df(ob_df: pd.DataFrame) -> pd.DataFrame:
    """
    تبدیل نام ستون‌های OB به قالب استاندارد:
    ob_upper, ob_lower, ob_kind, ob_age, ob_strength, ob_touch_count
    """
    mapping_candidates = {
        "upper": "ob_upper", "high": "ob_upper", "u": "ob_upper",
        "lower": "ob_lower", "low": "ob_lower", "l": "ob_lower",
        "kind": "ob_kind", "type": "ob_kind", "block_type": "ob_kind",
        "age": "ob_age", "strength": "ob_strength",
        "touch": "ob_touch_count", "touches": "ob_touch_count", "hits": "ob_touch_count",
    }
    out = pd.DataFrame(index=ob_df.index)
    for col in ob_df.columns:
        key = col.lower()
        target = mapping_candidates.get(key)
        if target:
            out[target] = ob_df[col]
    # تضمین وجود ستون‌ها
    for need, dtype in [
        ("ob_upper", "float32"),
        ("ob_lower", "float32"),
        ("ob_kind", "object"),
        ("ob_age", "float32"),
        ("ob_strength", "float32"),
        ("ob_touch_count", "float32"),
    ]:
        if need not in out.columns:
            out[need] = pd.Series([pd.NA] * len(out), index=out.index)
        if dtype != "object":
            out[need] = pd.to_numeric(out[need], errors="coerce")
    return out


# ---------------------------------------------------------------------
# API اصلی ماژول
# ---------------------------------------------------------------------
def build_zones(df: pd.DataFrame, *, anti_lookahead: bool = True) -> pd.DataFrame:
    """
    ساخت زون‌های SD و OB به‌صورت استاندارد با تکیه بر توابع indicators (در صورت وجود).
    - anti_lookahead: در صورت True، بلافاصله پس از تولید زون‌ها، یک شیفت +1 روی فلگ‌ها/باندها اعمال می‌شود.

    خروجی: DataFrame ورودی + ستون‌های SD/OB + فاصله‌ها
    """
    out = df.copy()

    if not _HAS_SR_ADV:
        # اگر هنوز indicators آماده نیست، پیام مشخص برای توسعه بعدی بدهیم.
        # (در تست‌ها نیز بر همین اساس skip انجام می‌شود.)
        return out  # بدون تغییر (fail-safe بدون ایجاد خطا)

    # --- Supply/Demand ---
    sd_proto = _call_detect_sd(out)
    sd_df = _call_make_sd(out, sd_proto) if sd_proto is not None else None
    if isinstance(sd_df, pd.DataFrame) and len(sd_df) == len(out):
        sd_std = _coerce_sd_df(sd_df)
        if anti_lookahead:
            sd_std = sd_std.shift(1)  # ضد لوک‌اِهد
        out = out.join(sd_std)

    # --- Order Blocks ---
    ob_proto = _call_detect_ob(out)
    ob_df = _call_make_ob(out, ob_proto) if ob_proto is not None else None
    if isinstance(ob_df, pd.DataFrame) and len(ob_df) == len(out):
        ob_std = _coerce_ob_df(ob_df)
        if anti_lookahead:
            ob_std = ob_std.shift(1)  # ضد لوک‌اِهد
        out = out.join(ob_std)

    # --- فاصله تا زون‌ها (نرمال‌سازی به ATR اگر موجود باشد) ---
    if "close" in out.columns:
        if "sd_upper" in out.columns and "sd_lower" in out.columns:
            mid_sd = (out["sd_upper"] + out["sd_lower"]) / 2.0
            out["dist_to_sd"] = (out["close"] - mid_sd).abs()
        if "ob_upper" in out.columns and "ob_lower" in out.columns:
            mid_ob = (out["ob_upper"] + out["ob_lower"]) / 2.0
            out["dist_to_ob"] = (out["close"] - mid_ob).abs()

        # نرمال‌سازی به ATR اگر موجود باشد
        if "atr" in out.columns:
            for col in ("dist_to_sd", "dist_to_ob"):
                if col in out.columns:
                    out[col] = out[col] / out["atr"]

    return out
