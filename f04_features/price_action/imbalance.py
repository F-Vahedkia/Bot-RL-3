# -*- coding: utf-8 -*-
# f04_features/price_action/imbalance.py
# Status in (Bot-RL-3H): Completed

"""
Price Action: Imbalance & Liquidity
===================================
این ماژول ویژگی‌های مربوط به عدم تعادل (FVG) و نقدینگی (Liquidity Pools / Sweeps) را
با تکیه بر توابع موجود در indicators (در صورت وجود) تولید می‌کند.

خروجی استاندارد:
- FVG:
    fvg_upper, fvg_lower, fvg_mid, fvg_thickness, fvg_age, fvg_filled_ratio
- Liquidity Pools (LP):
    lp_high_price, lp_low_price
- Sweeps:
    sweep_up, sweep_down
- Distances (نرمال‌شده به ATR اگر حاضر باشد):
    dist_to_fvg_mid, dist_to_lp_high, dist_to_lp_low
"""

from __future__ import annotations
import pandas as pd

# تلاش برای استفاده از پیاده‌سازی‌های موجود در indicators (بدون حدس؛ با wrapper سازگار)
_HAS_SR_ADV = True
try:
    from f04_features.indicators.sr_advanced import (
        detect_fvg,    # تشخیص اولیه FVG
        make_fvg,      # ساخت باندها و ویژگی‌های FVG
        detect_liq_sweep,  # تشخیص اولیه Sweep
        make_liq_sweep,    # ساخت فلگ‌های نهایی Sweep
    )
except Exception:
    _HAS_SR_ADV = False

_HAS_LP = True
try:
    # برخی پروژه‌ها LP را جدا پیاده‌سازی می‌کنند؛ اگر موجود نباشد، امن عبور می‌کنیم.
    # نام‌های محتمل؛ wrapper پایین تلاش می‌کند امضاهای مختلف را هندل کند.
    from f04_features.indicators.sr_advanced import detect_liquidity_pools as _detect_lp  # اختیاری
except Exception:
    _HAS_LP = False
    _detect_lp = None


# ---------------------------------------------------------------------
# سازگارساز فراخوانی‌ها (بدون فرض امضا)
# ---------------------------------------------------------------------
def _call_detect_fvg(df: pd.DataFrame):
    if not _HAS_SR_ADV:
        return None
    try:
        return detect_fvg(df)
    except TypeError:
        try:
            return detect_fvg(df["high"], df["low"], df.get("close", None))
        except Exception:
            return None


def _call_make_fvg(df: pd.DataFrame, proto):
    if not _HAS_SR_ADV or proto is None:
        return None
    try:
        return make_fvg(df, proto)
    except TypeError:
        try:
            return make_fvg(proto)
        except Exception:
            return None


def _call_detect_lp(df: pd.DataFrame):
    if not _HAS_LP or _detect_lp is None:
        return None
    try:
        return _detect_lp(df)
    except TypeError:
        try:
            return _detect_lp(df["high"], df["low"])
        except Exception:
            return None


def _call_detect_sweep(df: pd.DataFrame):
    if not _HAS_SR_ADV:
        return None
    try:
        return detect_liq_sweep(df)
    except TypeError:
        try:
            return detect_liq_sweep(df["high"], df["low"], df.get("close", None))
        except Exception:
            return None


def _call_make_sweep(df: pd.DataFrame, proto):
    if not _HAS_SR_ADV or proto is None:
        return None
    try:
        return make_liq_sweep(df, proto)
    except TypeError:
        try:
            return make_liq_sweep(proto)
        except Exception:
            return None


# ---------------------------------------------------------------------
# نرمال‌سازی ستون‌ها به قرارداد ثابت
# ---------------------------------------------------------------------
def _coerce_fvg_df(fvg_df: pd.DataFrame) -> pd.DataFrame:
    """
    استانداردسازی نام ستون‌های FVG:
      fvg_upper, fvg_lower, fvg_mid, fvg_thickness, fvg_age, fvg_filled_ratio
    """
    mapping = {
        "upper": "fvg_upper", "high": "fvg_upper", "u": "fvg_upper",
        "lower": "fvg_lower", "low": "fvg_lower", "l": "fvg_lower",
        "mid": "fvg_mid", "center": "fvg_mid",
        "thickness": "fvg_thickness", "width": "fvg_thickness",
        "age": "fvg_age",
        "filled": "fvg_filled_ratio", "fill_ratio": "fvg_filled_ratio",
    }
    out = pd.DataFrame(index=fvg_df.index)
    for col in fvg_df.columns:
        t = mapping.get(col.lower())
        if t:
            out[t] = fvg_df[col]

    # تضمین وجود ستون‌ها با نوع مناسب
    ensure = [
        ("fvg_upper", "float32"),
        ("fvg_lower", "float32"),
        ("fvg_mid", "float32"),
        ("fvg_thickness", "float32"),
        ("fvg_age", "float32"),
        ("fvg_filled_ratio", "float32"),
    ]
    for name, dtype in ensure:
        if name not in out.columns:
            out[name] = pd.Series([pd.NA] * len(out), index=out.index)
        out[name] = pd.to_numeric(out[name], errors="coerce")
    return out


def _coerce_lp_df(lp_df: pd.DataFrame) -> pd.DataFrame:
    """
    استانداردسازی نام ستون‌های Liquidity Pools:
      lp_high_price, lp_low_price
    """
    mapping = {
        "lp_high": "lp_high_price", "equal_high": "lp_high_price",
        "lp_low": "lp_low_price", "equal_low": "lp_low_price",
        "high_price": "lp_high_price", "low_price": "lp_low_price",
    }
    out = pd.DataFrame(index=lp_df.index)
    for col in lp_df.columns:
        t = mapping.get(col.lower())
        if t:
            out[t] = lp_df[col]

    for name in ("lp_high_price", "lp_low_price"):
        if name not in out.columns:
            out[name] = pd.Series([pd.NA] * len(out), index=out.index)
        out[name] = pd.to_numeric(out[name], errors="coerce")
    return out


def _coerce_sweep_df(sw_df: pd.DataFrame) -> pd.DataFrame:
    """
    استانداردسازی نام ستون‌های Sweep:
      sweep_up, sweep_down (int8: 0/1)
    """
    mapping = {
        "sweep_up": "sweep_up",
        "sweep_down": "sweep_down",
        "stop_run_up": "sweep_up",
        "stop_run_down": "sweep_down",
    }
    out = pd.DataFrame(index=sw_df.index)
    for col in sw_df.columns:
        t = mapping.get(col.lower())
        if t:
            out[t] = (sw_df[col].astype("int8") > 0).astype("int8")

    for name in ("sweep_up", "sweep_down"):
        if name not in out.columns:
            out[name] = pd.Series([0] * len(out), index=out.index, dtype="int8")
        else:
            out[name] = (out[name].astype("int8") > 0).astype("int8")
    return out


# ---------------------------------------------------------------------
# API اصلی
# ---------------------------------------------------------------------
def build_imbalance_liquidity(df: pd.DataFrame, *, anti_lookahead: bool = True) -> pd.DataFrame:
    """
    ساخت فیچرهای Imbalance & Liquidity:
    - FVG (در صورت حضور توابع indicators)
    - Liquidity Pools (اگر موجود باشد)
    - Sweeps (اگر موجود باشد)
    - فاصله‌ها و نرمال‌سازی به ATR (اگر ستون atr موجود باشد)

    خروجی: DataFrame ورودی به‌علاوه ستون‌های استاندارد بالا.
    """
    out = df.copy()

    # ---------- FVG ----------
    fvg_proto = _call_detect_fvg(out)
    fvg_df = _call_make_fvg(out, fvg_proto) if fvg_proto is not None else None
    if isinstance(fvg_df, pd.DataFrame) and len(fvg_df) == len(out):
        fvgs = _coerce_fvg_df(fvg_df)
        if anti_lookahead:
            fvgs = fvgs.shift(1)
        out = out.join(fvgs)

    # ---------- Liquidity Pools ----------
    lp_raw = _call_detect_lp(out)
    if isinstance(lp_raw, pd.DataFrame) and len(lp_raw) == len(out):
        lps = _coerce_lp_df(lp_raw)
        if anti_lookahead:
            lps = lps.shift(1)
        out = out.join(lps)

    # ---------- Sweeps ----------
    sw_proto = _call_detect_sweep(out)
    sw_df = _call_make_sweep(out, sw_proto) if sw_proto is not None else None
    if isinstance(sw_df, pd.DataFrame) and len(sw_df) == len(out):
        sweeps = _coerce_sweep_df(sw_df)
        if anti_lookahead:
            sweeps = sweeps.shift(1)
        out = out.join(sweeps)

    # ---------- Distances ----------
    if "close" in out.columns:
        if "fvg_mid" in out.columns:
            out["dist_to_fvg_mid"] = (out["close"] - out["fvg_mid"]).abs()
        if "lp_high_price" in out.columns:
            out["dist_to_lp_high"] = (out["close"] - out["lp_high_price"]).abs()
        if "lp_low_price" in out.columns:
            out["dist_to_lp_low"] = (out["close"] - out["lp_low_price"]).abs()

        # نرمال‌سازی به ATR
        if "atr" in out.columns:
            for col in ("dist_to_fvg_mid", "dist_to_lp_high", "dist_to_lp_low"):
                if col in out.columns:
                    out[col] = out[col] / out["atr"]

    return out
