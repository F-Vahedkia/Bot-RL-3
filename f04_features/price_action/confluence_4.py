# -*- coding: utf-8 -*-
# f04_features/price_action/confluence_4.py
# Status: FINAL — Canonical, Production-Grade, MTF-aware

"""
Confluence Engine for Price Action — CANONICAL
==============================================

این ماژول امتیاز هم‌گرایی (Confluence) را با ترکیب:
- ساختار بازار (BOS / CHOCH / Swing)
- زون‌ها (Supply / Demand / OrderBlock)
- عدم‌تعادل و نقدینگی (FVG / Sweeps)
- زمینه‌ی چند-تایم‌فریم (MTF)
- Extras (Regime / Breakouts / Micro-Channels)

به صورت یک نمره‌ی استاندارد در بازه [0..1] محاسبه می‌کند.

ویژگی‌ها:
- MTF-aware با جریمه‌ی تضاد و پاداش هم‌راستایی
- Anti-lookahead (shift کامل خروجی‌ها)
- مقاوم به NaN و warm-up
- بدون حدس: نبود ستون ⇒ سهم صفر
- Vectorized و مناسب دیتای حجیم
- Type-safe (float32 / int8)
- Release-ready و production-grade
"""

from __future__ import annotations
import math
import pandas as pd
import numpy as np

# =========================================================
# Utility Helpers (Safe, Vectorized)
# =========================================================

def _safe(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def _clip01(x):
    if isinstance(x, pd.Series):
        return x.clip(0.0, 1.0)
    return min(1.0, max(0.0, float(x)))


def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def _to_float(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).astype("float32")


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype("int8")


def _norm_distance(series: pd.Series) -> pd.Series:
    """
    نگاشت فاصله به امتیاز ۰..۱
    نزدیک‌تر ⇒ امتیاز بیشتر
    score = 1 / (1 + distance)
    """
    s = pd.to_numeric(series, errors="coerce")
    return (1.0 / (1.0 + s)).fillna(0.0).astype("float32")


# =========================================================
# Main Builder
# =========================================================

def build_confluence(
    df: pd.DataFrame,
    *,
    weights: dict | None = None,
    strong_entry_threshold: float = 0.70,
    filter_threshold: float = 0.35,
    anti_lookahead: bool = True,
) -> pd.DataFrame:
    """
    محاسبه نمره نهایی Confluence به‌صورت کاملاً برداری و MTF-aware.
    """

    out = df.copy()

    # -----------------------------------------------------
    # 1) Weights
    # -----------------------------------------------------
    if weights is None:
        weights = {
            "structure": 0.30,
            "zones": 0.30,
            "imbalance": 0.20,
            "mtf": 0.20,
            "extras": 0.00,
        }

    w_struct = _safe(weights.get("structure", 0.30))
    w_zones  = _safe(weights.get("zones", 0.30))
    w_imbal  = _safe(weights.get("imbalance", 0.20))
    w_mtf    = _safe(weights.get("mtf", 0.20))
    w_extra  = _safe(weights.get("extras", 0.00))
    w_sum = max(1e-9, w_struct + w_zones + w_imbal + w_mtf + w_extra)

    # -----------------------------------------------------
    # 2) Structure (BOS / CHOCH / Swing)
    # -----------------------------------------------------
    s_score = pd.Series(0.0, index=out.index, dtype="float32")

    if _has_cols(out, ["bos_up", "bos_down", "choch_up", "choch_down"]):
        s_score = (
            _to_int(out["bos_up"]) * 0.50
            + _to_int(out["bos_down"]) * 0.50
            + _to_int(out["choch_up"]) * 0.35
            + _to_int(out["choch_down"]) * 0.35
        ).clip(0.0, 1.0)

    if "swing_type" in out.columns:
        swing_boost = (
            out["swing_type"]
            .astype("object")
            .fillna("")
            .map({"HH": 0.15, "LL": 0.15})
            .fillna(0.0)
            .astype("float32")
        )
        s_score = _clip01(s_score + swing_boost)

    out["conf_components_structure"] = s_score

    # -----------------------------------------------------
    # 3) Zones (Supply / Demand / OB)
    # -----------------------------------------------------
    z_score = pd.Series(0.0, index=out.index, dtype="float32")

    if "dist_to_sd" in out.columns:
        z_score += _norm_distance(out["dist_to_sd"]) * 0.60
    if "dist_to_ob" in out.columns:
        z_score += _norm_distance(out["dist_to_ob"]) * 0.60

    z_score = z_score.clip(0.0, 1.0)
    out["conf_components_zones"] = z_score

    # -----------------------------------------------------
    # 4) Imbalance / Liquidity (FVG + Sweeps)
    # -----------------------------------------------------
    i_score = pd.Series(0.0, index=out.index, dtype="float32")

    if "dist_to_fvg_mid" in out.columns:
        i_score += _norm_distance(out["dist_to_fvg_mid"]) * 0.60
    if "sweep_up" in out.columns:
        i_score += _to_int(out["sweep_up"]) * 0.25
    if "sweep_down" in out.columns:
        i_score += _to_int(out["sweep_down"]) * 0.25

    i_score = i_score.clip(0.0, 1.0)
    out["conf_components_imbalance"] = i_score

    # -----------------------------------------------------
    # 5) MTF Context (Aware + Conflict Penalty)
    # -----------------------------------------------------
    m_score = pd.Series(0.0, index=out.index, dtype="float32")

    if "mtf_confluence_score" in out.columns:
        m_score += _to_float(out["mtf_confluence_score"], 0.5) * 0.80

    if "mtf_conflict" in out.columns:
        penalty = (_to_int(out["mtf_conflict"]) > 0).astype("float32") * 0.50
        m_score = (m_score - penalty).clip(0.0, 1.0)

    if "mtf_strength" in out.columns:
        m_score = _clip01(m_score + _to_float(out["mtf_strength"]) * 0.20)

    out["conf_components_mtf"] = m_score

    # -----------------------------------------------------
    # 6) Extras (Regime / Breakouts / Micro-Channels)
    # -----------------------------------------------------
    e_score = pd.Series(0.0, index=out.index, dtype="float32")

    if "regime_channel" in out.columns:
        e_score += _to_float(out["regime_channel"]) * 0.45
    if "regime_spike" in out.columns:
        e_score += _to_float(out["regime_spike"]) * 0.25
    if "regime_range" in out.columns:
        e_score -= _to_float(out["regime_range"]) * 0.20

    if "breakout_up" in out.columns:
        e_score += _to_int(out["breakout_up"]) * 0.20
    if "breakout_down" in out.columns:
        e_score += _to_int(out["breakout_down"]) * 0.20
    if "retest_up" in out.columns:
        e_score += _to_int(out["retest_up"]) * 0.10
    if "retest_down" in out.columns:
        e_score += _to_int(out["retest_down"]) * 0.10
    if "fail_break_up" in out.columns:
        e_score -= _to_int(out["fail_break_up"]) * 0.15
    if "fail_break_down" in out.columns:
        e_score -= _to_int(out["fail_break_down"]) * 0.15

    if "micro_channel_up" in out.columns:
        e_score += _to_int(out["micro_channel_up"]) * 0.15
    if "micro_channel_down" in out.columns:
        e_score += _to_int(out["micro_channel_down"]) * 0.15
    if "micro_channel_quality" in out.columns:
        e_score += _to_float(out["micro_channel_quality"]) * 0.20

    e_score = e_score.clip(0.0, 1.0)
    out["conf_components_extras"] = e_score

    # -----------------------------------------------------
    # 7) Final Confluence Score
    # -----------------------------------------------------
    conf = (
        s_score * (w_struct / w_sum)
        + z_score * (w_zones / w_sum)
        + i_score * (w_imbal / w_sum)
        + m_score * (w_mtf / w_sum)
        + e_score * (w_extra / w_sum)
    ).clip(0.0, 1.0).astype("float32")

    flag_strong = (conf >= strong_entry_threshold).astype("int8")
    flag_filter = (conf >= filter_threshold).astype("int8")

    # -----------------------------------------------------
    # 8) Anti-lookahead
    # -----------------------------------------------------
    if anti_lookahead:
        for c in (
            "conf_components_structure",
            "conf_components_zones",
            "conf_components_imbalance",
            "conf_components_mtf",
            "conf_components_extras",
        ):
            out[c] = out[c].shift(1)
        conf = conf.shift(1)
        flag_strong = flag_strong.shift(1).fillna(0).astype("int8")
        flag_filter = flag_filter.shift(1).fillna(0).astype("int8")

    # -----------------------------------------------------
    # 9) Final Columns
    # -----------------------------------------------------
    out["conf_score"] = conf.fillna(0.5).clip(0.0, 1.0).astype("float32")
    out["conf_flag_strong_entry"] = flag_strong
    out["conf_flag_filter_pass"] = flag_filter

    return out
