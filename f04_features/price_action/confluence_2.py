# -*- coding: utf-8 -*-
# f04_features/price_action/confluence.py
# Status: Final — Professional, Complete, Class-World Level
"""
Confluence Engine for Price Action — FINAL
==========================================

این ماژول امتیاز هم‌گرایی (Confluence) را با ترکیب سیگنال‌های پرایس‌اکشن و زمینهٔ MTF
و سیگنال‌های Extras (Regime/Breakouts/Microchannels) محاسبه می‌کند و نمره‌ای استاندارد
[0..1] به ازای هر ردیف تولید می‌کند.

ویژگی‌ها:
- ضد لوک‌اِهد (anti_lookahead)
- مقاوم به NaN / warm-up robust
- استفاده از ستون‌های موجود، بدون دوباره‌کاری
- آماده برای داده‌های UTC و آخرین کندل بسته شده
- Multi Time Frame-aware
- وزن‌دهی پویا به هر سبد
- کامنت‌های جامع و حرفه‌ای برای هر بخش
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ==========================
# --- Utility Functions ---
# ==========================
def _safe(arr, default=0.0):
    """بازگرداندن float امن از ورودی ممکن (NaN → default)"""
    if arr is None:
        return default
    try:
        x = float(arr)
        if math.isnan(x):
            return default
        return x
    except Exception:
        return default


def _clip01(s: pd.Series | float) -> pd.Series | float:
    """کلیپ کردن عدد یا Series در بازه 0..1"""
    if isinstance(s, pd.Series):
        return s.clip(0.0, 1.0)
    return min(1.0, max(0.0, float(s)))


def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    """بررسی وجود همه ستون‌ها در DataFrame"""
    return all(c in df.columns for c in cols)


def _norm_distance(series: pd.Series) -> pd.Series:
    """
    نگاشت فاصله‌ها به نمره 0..1
    - فاصله کمتر → امتیاز بالاتر
    - NaN → 0
    score = 1 / (1 + distance)
    """
    s = pd.to_numeric(series, errors="coerce")
    return (1.0 / (1.0 + s)).fillna(0.0).astype("float32")


# ==========================
# --- Main Function ---
# ==========================
def build_confluence(
    df: pd.DataFrame,
    *,
    weights: dict | None = None,
    strong_entry_threshold: float = 0.7,
    filter_threshold: float = 0.35,
    anti_lookahead: bool = True,
) -> pd.DataFrame:
    """
    محاسبه نمره Confluence نهایی

    پارامترها:
    ------------
    df : DataFrame
        باید شامل ستون‌های پایه price action و MTF باشد.
    weights : dict
        وزن سبدها: {"structure", "zones", "imbalance", "mtf", "extras"}
    strong_entry_threshold : float
        آستانه پرچم strong entry
    filter_threshold : float
        آستانه فیلتر عبور حداقلی
    anti_lookahead : bool
        فعال‌سازی شیفت برای جلوگیری از lookahead

    خروجی:
    --------
    df با ستون‌های جدید:
    - conf_components_structure
    - conf_components_zones
    - conf_components_imbalance
    - conf_components_mtf
    - conf_components_extras
    - conf_score
    - conf_flag_strong_entry
    - conf_flag_filter_pass
    """
    out = df.copy()

    # ---------------------------
    # 1) تنظیم وزن‌ها
    # ---------------------------
    if weights is None:
        weights = {"structure": 0.30, "zones": 0.30, "imbalance": 0.20,
                   "mtf": 0.20, "extras": 0.0}
    w_struct = _safe(weights.get("structure", 0.30))
    w_zones = _safe(weights.get("zones", 0.30))
    w_imbal = _safe(weights.get("imbalance", 0.20))
    w_mtf = _safe(weights.get("mtf", 0.20))
    w_extras = _safe(weights.get("extras", 0.0))
    w_sum = max(1e-9, w_struct + w_zones + w_imbal + w_mtf + w_extras)

    # =========================================================
    # 2) Structure (BOS/CHOCH + Swing) → 0..1
    # =========================================================
    s_score = pd.Series(0.0, index=out.index, dtype="float32")
    if _has_cols(out, ["bos_up","bos_down","choch_up","choch_down"]):
        s_score = (out["bos_up"].fillna(0).astype("int8")*0.5
                  + out["bos_down"].fillna(0).astype("int8")*0.5
                  + out["choch_up"].fillna(0).astype("int8")*0.35
                  + out["choch_down"].fillna(0).astype("int8")*0.35)
        s_score = _clip01(s_score)

    if "swing_type" in out.columns:
        boost = out["swing_type"].astype("object").fillna("").map({"HH":0.15,"LL":0.15}).fillna(0.0)
        s_score = _clip01(s_score + boost)

    out["conf_components_structure"] = s_score.astype("float32")

    # =========================================================
    # 3) Zones (SD/OB) → 0..1
    # =========================================================
    z_score = pd.Series(0.0, index=out.index, dtype="float32")
    if "dist_to_sd" in out.columns:
        z_score += _norm_distance(out["dist_to_sd"])*0.6
    if "dist_to_ob" in out.columns:
        z_score += _norm_distance(out["dist_to_ob"])*0.6
    z_score = _clip01(z_score)
    out["conf_components_zones"] = z_score

    # =========================================================
    # 4) Imbalance (FVG + Sweeps) → 0..1
    # =========================================================
    i_score = pd.Series(0.0, index=out.index, dtype="float32")
    if "dist_to_fvg_mid" in out.columns:
        i_score += _norm_distance(out["dist_to_fvg_mid"])*0.6
    if "sweep_up" in out.columns:
        i_score += out["sweep_up"].fillna(0).astype("int8")*0.25
    if "sweep_down" in out.columns:
        i_score += out["sweep_down"].fillna(0).astype("int8")*0.25
    i_score = _clip01(i_score)
    out["conf_components_imbalance"] = i_score

    # =========================================================
    # 5) MTF Context → 0..1
    # =========================================================
    m_score = pd.Series(0.0, index=out.index, dtype="float32")
    if "mtf_confluence_score" in out.columns:
        m_score += out["mtf_confluence_score"].fillna(0.5)*0.8
    if "mtf_conflict" in out.columns:
        pen = (out["mtf_conflict"].fillna(0).astype("int8")>0).astype("int8")*0.5
        m_score = _clip01(m_score - pen)
    if "mtf_strength" in out.columns:
        m_score = _clip01(m_score + out["mtf_strength"].fillna(0.0)*0.2)
    out["conf_components_mtf"] = m_score.astype("float32")

    # =========================================================
    # 6) Extras: Regime + Breakouts + MicroChannels
    # =========================================================
    e_score = pd.Series(0.0, index=out.index, dtype="float32")
    # --- Regime
    if "regime_channel" in out.columns:
        e_score += out["regime_channel"].fillna(0.0)*0.45
    if "regime_spike" in out.columns:
        e_score += out["regime_spike"].fillna(0.0)*0.25
    if "regime_range" in out.columns:
        e_score -= out["regime_range"].fillna(0.0)*0.20
    # --- Breakouts
    if "breakout_up" in out.columns:
        e_score += out["breakout_up"].fillna(0).astype("int8")*0.20
    if "breakout_down" in out.columns:
        e_score += out["breakout_down"].fillna(0).astype("int8")*0.20
    if "retest_up" in out.columns:
        e_score += out["retest_up"].fillna(0).astype("int8")*0.10
    if "retest_down" in out.columns:
        e_score += out["retest_down"].fillna(0).astype("int8")*0.10
    if "fail_break_up" in out.columns:
        e_score -= out["fail_break_up"].fillna(0).astype("int8")*0.15
    if "fail_break_down" in out.columns:
        e_score -= out["fail_break_down"].fillna(0).astype("int8")*0.15
    # --- Micro-channels
    if "micro_channel_up" in out.columns:
        e_score += out["micro_channel_up"].fillna(0).astype("int8")*0.15
    if "micro_channel_down" in out.columns:
        e_score += out["micro_channel_down"].fillna(0).astype("int8")*0.15
    if "micro_channel_quality" in out.columns:
        e_score += out["micro_channel_quality"].fillna(0.0)*0.20
    e_score = _clip01(e_score)
    out["conf_components_extras"] = e_score

    # =========================================================
    # 7) محاسبه conf_score نهایی با وزن‌ها
    # =========================================================
    conf = (s_score*(w_struct/w_sum)
            + z_score*(w_zones/w_sum)
            + i_score*(w_imbal/w_sum)
            + m_score*(w_mtf/w_sum)
            + e_score*(w_extras/w_sum)).astype("float32")
    conf = _clip01(conf)

    # =======================
    # 8) پرچم‌ها
    # =======================
    flag_strong = (conf>=strong_entry_threshold).astype("int8")
    flag_filter = (conf>=filter_threshold).astype("int8")

    # =======================
    # 9) Anti-lookahead → shift(+1)
    # =======================
    if anti_lookahead:
        for c in ["conf_components_structure","conf_components_zones",
                  "conf_components_imbalance","conf_components_mtf",
                  "conf_components_extras"]:
            out[c] = out[c].shift(1)
        conf = conf.shift(1)
        flag_strong = flag_strong.shift(1).fillna(0).astype("int8")
        flag_filter = flag_filter.shift(1).fillna(0).astype("int8")

    # =======================
    # 10) تضمین بازه و عدم NaN
    # =======================
    conf = conf.fillna(0.5)
    conf = _clip01(conf)
    out["conf_score"] = conf
    out["conf_flag_strong_entry"] = flag_strong
    out["conf_flag_filter_pass"] = flag_filter

    return out
