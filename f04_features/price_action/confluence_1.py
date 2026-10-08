# -*- coding: utf-8 -*-
# f04_features/price_action/confluence.py
# Status in (Bot-RL-3H): Completed
"""
Confluence Engine for Price Action
==================================
این ماژول امتیاز هم‌گرایی (Confluence) را با ترکیب سیگنال‌های پرایس‌اکشن و زمینهٔ MTF
به صورت یک نمره‌ی استاندارد [0..1] تولید می‌کند.

ورودی‌ها: DataFrame که قبلاً توسط ماژول‌های زیر غنی شده:
- market_structure:  swing_type, bos_up, bos_down, choch_up, choch_down
- zones:             sd_upper/lower, ob_upper/lower, dist_to_sd, dist_to_ob
- imbalance:         fvg_* ، sweep_up/sweep_down ، dist_to_fvg_mid
- mtf_context:       mtf_bias, mtf_bias_local, mtf_conflict, mtf_confluence_score, mtf_strength

خروجی‌ها:
- conf_score              : نمره نهایی ۰..۱
- conf_components_*       : نمره‌ی هر سبد (ساختار، زون، عدم‌تعادل/نقدینگی، MTF)
- conf_flag_strong_entry  : پرچم ورود قوی (int8)
- conf_flag_filter_pass   : پرچم عبور از فیلترهای حداقلی (int8)

نکات:
- ضد لوک‌اِهد: در صورت anti_lookahead=True، خروجی‌ها shift(+1) می‌شوند.
- بدون دوباره‌کاری: از ستون‌های موجود استفاده می‌کنیم؛ اگر ستونی نبود، سهمش صفر می‌شود.
- پیام‌های runtime انگلیسی هستند.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd


def _safe(arr, default=0.0):
    """برگرداندن مقدار float امن از ورودی‌های ممکن (NaN → default)."""
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
    """کلیپ در بازه [0..1]."""
    if isinstance(s, pd.Series):
        return s.clip(0.0, 1.0)
    return min(1.0, max(0.0, float(s)))


def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def _norm_distance(series: pd.Series) -> pd.Series:
    """
    نگاشت فاصله‌ها به نمرهٔ ۰..۱ (دوری کمتر → امتیاز بیشتر).
    score = 1 / (1 + distance)
    اگر NaN باشد → 0
    """
    s = pd.to_numeric(series, errors="coerce")
    return (1.0 / (1.0 + s)).fillna(0.0).astype("float32")


def build_confluence(
    df: pd.DataFrame,
    *,
    weights: dict | None = None,
    strong_entry_threshold: float = 0.7,
    filter_threshold: float = 0.35,
    anti_lookahead: bool = True,
) -> pd.DataFrame:
    """
    ساخت نمرهٔ Confluence نهایی.

    پارامترها:
        weights: دیکشنری وزن‌ها برای چهار سبد
            {
              "structure": 0.30,
              "zones": 0.30,
              "imbalance": 0.20,
              "mtf": 0.20
            }
        strong_entry_threshold: آستانهٔ نمره برای پرچم ورود قوی (conf_score ≥ ...)
        filter_threshold: آستانهٔ حداقلی برای عبور از فیلترها (conf_score ≥ ...)
        anti_lookahead: شیفت خروجی‌ها برای حذف لوک‌اِهد

    خروجی:
        df به‌علاوهٔ ستون‌های:
          conf_components_structure, conf_components_zones,
          conf_components_imbalance, conf_components_mtf,
          conf_score, conf_flag_strong_entry, conf_flag_filter_pass
    """
    out = df.copy()

    # ---------- وزن‌ها ----------
    if weights is None:
        weights = {
            "structure": 0.30,
            "zones": 0.30,
            "imbalance": 0.20,
            "mtf": 0.20,
            "extras": 0.00,  # ← جدید: رژیم/بریک‌اوت/میکروکانال
        }
    w_struct = _safe(weights.get("structure", 0.30))
    w_zones = _safe(weights.get("zones", 0.30))
    w_imbal = _safe(weights.get("imbalance", 0.20))
    w_mtf = _safe(weights.get("mtf", 0.20))
    w_extras = _safe(weights.get("extras", 0.00))
    w_sum = max(1e-9, w_struct + w_zones + w_imbal + w_mtf + w_extras)

    # =========================================================
    # 1) ساختار (Structure) → از BOS/CHOCH و swing_type
    # =========================================================
    # امتیاز پایه از BOS/CHOCH
    s_score = pd.Series(0.0, index=out.index, dtype="float32")
    if _has_cols(out, ["bos_up", "bos_down", "choch_up", "choch_down"]):
        s_score = (
            out["bos_up"].fillna(0).astype("int8") * 0.5
            + out["bos_down"].fillna(0).astype("int8") * 0.5
            + out["choch_up"].fillna(0).astype("int8") * 0.35
            + out["choch_down"].fillna(0).astype("int8") * 0.35
        )
        # نرمال به ۰..۱
        s_score = (s_score / 1.0).clip(0.0, 1.0).astype("float32")

    # تقویت جزئی با نوع سوئینگ
    if "swing_type" in out.columns:
        swing = out["swing_type"].astype("object").fillna("")
        # اگر HH/LL بیاید، سبد ساختار کمی تقویت می‌شود
        #boost = swing.map({"HH": 0.15, "LL": 0.15}).fillna(0.0).astype("float32")
        boost = out["swing_type"].astype("object").fillna("").map({"HH": 0.15, "LL": 0.15}).fillna(0.0).astype("float32")
        s_score = _clip01(s_score + boost)

    out["conf_components_structure"] = s_score.astype("float32")

    # =========================================================
    # 2) زون‌ها (Zones) → SD/OB proximity
    # =========================================================
    z_score = pd.Series(0.0, index=out.index, dtype="float32")
    # فاصله تا SD و OB (هرچه نزدیک‌تر → امتیاز بالاتر)
    if "dist_to_sd" in out.columns:
        z_score += _norm_distance(out["dist_to_sd"]) * 0.6
    if "dist_to_ob" in out.columns:
        z_score += _norm_distance(out["dist_to_ob"]) * 0.6
    # نرمال‌سازی (ممکن است هر دو نباشند)
    z_score = z_score.clip(0.0, 1.0).astype("float32")
    out["conf_components_zones"] = z_score

    # =========================================================
    # 3) عدم تعادل/نقدینگی (Imbalance) → FVG و Sweeps
    # =========================================================
    i_score = pd.Series(0.0, index=out.index, dtype="float32")
    if "dist_to_fvg_mid" in out.columns:
        i_score += _norm_distance(out["dist_to_fvg_mid"]) * 0.6
    # سوئیپ‌ها (وجود سیگنال) → تقویت
    if "sweep_up" in out.columns:
        i_score += out["sweep_up"].fillna(0).astype("int8") * 0.25
    if "sweep_down" in out.columns:
        i_score += out["sweep_down"].fillna(0).astype("int8") * 0.25
    i_score = i_score.clip(0.0, 1.0).astype("float32")
    out["conf_components_imbalance"] = i_score

    # =========================================================
    # 4) MTF Context → mtf_confluence_score و penalties
    # =========================================================
    m_score = pd.Series(0.0, index=out.index, dtype="float32")
    if "mtf_confluence_score" in out.columns:
        m_score += out["mtf_confluence_score"].fillna(0.5) * 0.8
    if "mtf_conflict" in out.columns:
        # تضاد، جریمه دارد
        pen = (out["mtf_conflict"].fillna(0).astype("int8") > 0).astype("int8") * 0.5
        m_score = (m_score - pen).clip(0.0, 1.0)
    if "mtf_strength" in out.columns:
        # بایاس قوی‌تر → کمی پاداش
        m_score = _clip01(m_score + out["mtf_strength"].fillna(0.0) * 0.2)
    out["conf_components_mtf"] = m_score.astype("float32")

    # =========================================================
    # 5) Extras (جدید): Regime + Breakouts + Micro-channels
    # =========================================================
    e_score = pd.Series(0.0, index=out.index, dtype="float32")

    # Regime: کانال و اسپایک امتیاز، رنج جریمه
    if "regime_channel" in out.columns:
        e_score += out["regime_channel"].fillna(0.0) * 0.45
    if "regime_spike" in out.columns:
        e_score += out["regime_spike"].fillna(0.0) * 0.25
    if "regime_range" in out.columns:
        e_score -= out["regime_range"].fillna(0.0) * 0.20

    # Breakouts: بریک‌اوت + ریتست پاداش، فیل‌بریک جریمه
    if "breakout_up" in out.columns:
        e_score += out["breakout_up"].fillna(0).astype("int8") * 0.20
    if "breakout_down" in out.columns:
        e_score += out["breakout_down"].fillna(0).astype("int8") * 0.20
    if "retest_up" in out.columns:
        e_score += out["retest_up"].fillna(0).astype("int8") * 0.10
    if "retest_down" in out.columns:
        e_score += out["retest_down"].fillna(0).astype("int8") * 0.10
    if "fail_break_up" in out.columns:
        e_score -= out["fail_break_up"].fillna(0).astype("int8") * 0.15
    if "fail_break_down" in out.columns:
        e_score -= out["fail_break_down"].fillna(0).astype("int8") * 0.15

    # Micro-channels: پرچم + کیفیت
    if "micro_channel_up" in out.columns:
        e_score += out["micro_channel_up"].fillna(0).astype("int8") * 0.15
    if "micro_channel_down" in out.columns:
        e_score += out["micro_channel_down"].fillna(0).astype("int8") * 0.15
    if "micro_channel_quality" in out.columns:
        e_score += out["micro_channel_quality"].fillna(0.0) * 0.20

    e_score = e_score.clip(0.0, 1.0).astype("float32")
    out["conf_components_extras"] = e_score

    # =========================================================
    # امتیاز نهایی با وزن‌ها (اکنون ۵ سبد)
    # =========================================================
    conf = (
        s_score * (w_struct / w_sum)
        + z_score * (w_zones / w_sum)
        + i_score * (w_imbal / w_sum)
        + m_score * (w_mtf / w_sum)
        + e_score * (w_extras / w_sum)  # ← جدید
    ).astype("float32")
    conf = conf.clip(0.0, 1.0)

    # پرچم‌ها
    flag_strong = (conf >= float(strong_entry_threshold)).astype("int8")
    flag_filter = (conf >= float(filter_threshold)).astype("int8")

    # ضد-لوک‌اِهد
    if anti_lookahead:
        for c in (
            "conf_components_structure",
            "conf_components_zones",
            "conf_components_imbalance",
            "conf_components_mtf",
            "conf_components_extras",  # ← جدید
        ):
            out[c] = out[c].shift(1)
        conf = conf.shift(1)
        flag_strong = flag_strong.shift(1).fillna(0).astype("int8")
        flag_filter = flag_filter.shift(1).fillna(0).astype("int8")
    
    # ✅ تضمین بازه و بدون NaN برای تست رِنج
    conf = conf.fillna(0.5).clip(0.0, 1.0).astype("float32")

    #out["conf_score"] = conf.astype("float32")
    out["conf_score"] = conf
    out["conf_flag_strong_entry"] = flag_strong
    out["conf_flag_filter_pass"] = flag_filter

    return out

