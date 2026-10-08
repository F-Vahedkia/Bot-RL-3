# -*- coding: utf-8 -*-
# f04_features/price_action/config_wiring.py
# Status in (Bot-RL-3H): Completed

"""
Price Action — Config Wiring
============================
این ماژول، پارامترهای پرایس‌اکشن را از دیکت کانفیگ (بارگذاری‌شده از YAML)
استخراج کرده و به صورت امن به ارکستریتور پرایس‌اکشن پاس می‌دهد.
قوانین پروژه رعایت شده‌اند:
- عدمِ حدس: فقط کلیدهایی که وجود دارند خوانده می‌شوند؛ مقدارِ پیش‌فرض خودِ ماژول‌ها حفظ می‌شود.
- عدمِ دوباره‌کاری: به هسته‌ی بیلدرها دست نمی‌زنیم؛ فقط وایرینگ انجام می‌دهیم.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import copy
import pandas as pd

from f04_features.price_action import registry as pa_reg


# ---------------------------------------------------------------------
# کمکـی: واکشی امن از دیکت کانفیگ (بدون حدس و بدون KeyError)
# ---------------------------------------------------------------------
def _get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    واکشی امن با مسیرهای نقطه‌ای مثل: 'features.price_action_params.regime.slope_window'
    اگر هر بخش از مسیر نبود، default برمی‌گرداند.
    """
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _maybe_put(d: Dict[str, Any], key: str, val: Any) -> None:
    """اگر val تهی/None نبود، در دیکشنری مقصد قرار می‌دهیم (از overwrite بیهوده پرهیز)."""
    if val is not None:
        d[key] = val


# ---------------------------------------------------------------------
# API 1: استخراج kwargs های پرایس‌اکشن از روی کانفیگ
# ---------------------------------------------------------------------
def extract_pa_kwargs_from_config(
        cfg: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any],
            Dict[str, Any], Dict[str, Any], Dict[str, Any],
            Dict[str, Any], Dict[str, Any], bool]:
    """
    ورودی:
        cfg: دیکشنری کانفیگ بارگذاری‌شده از YAML
    خروجی:
        (market_structure_kwargs, regime_kwargs, zones_kwargs, imbalance_kwargs,
         breakouts_kwargs, microchannels_kwargs, mtf_context_kwargs, confluence_kwargs,
         anti_lookahead_flag)

    نکته: تنها کلیدهایی که در cfg موجودند استخراج می‌شوند؛ بقیه به پیش‌فرض‌های بیلدرها سپرده می‌شوند.
    """
    # namespace پایه برای پارامترهای PA
    base_ns = "features.price_action_params"

    anti_lookahead = bool(_get(cfg, f"{base_ns}.anti_lookahead", True))


    # --- market_structure ---
    ms_kw: Dict[str, Any] = {}
    _maybe_put(ms_kw, "lookback", _get(cfg, f"{base_ns}.market_structure.lookback"))


    # --- regime ---
    rg_kw: Dict[str, Any] = {"anti_lookahead": anti_lookahead}
    _maybe_put(rg_kw, "atr_window", _get(cfg, f"{base_ns}.regime.atr_window"))
    _maybe_put(rg_kw, "width_window", _get(cfg, f"{base_ns}.regime.width_window"))
    _maybe_put(rg_kw, "slope_window", _get(cfg, f"{base_ns}.regime.slope_window"))
    _maybe_put(rg_kw, "spike_thr_atr", _get(cfg, f"{base_ns}.regime.spike_thr_atr"))
    _maybe_put(rg_kw, "channel_slope_thr", _get(cfg, f"{base_ns}.regime.channel_slope_thr"))


    # --- zones ---
    zn_kw: Dict[str, Any] = {"anti_lookahead": anti_lookahead}


    # --- imbalance ---
    im_kw: Dict[str, Any] = {"anti_lookahead": anti_lookahead}


    # --- breakouts ---
    bo_kw: Dict[str, Any] = {"anti_lookahead": anti_lookahead}
    _maybe_put(bo_kw, "range_window", _get(cfg, f"{base_ns}.breakouts.range_window"))
    _maybe_put(bo_kw, "min_periods", _get(cfg, f"{base_ns}.breakouts.min_periods"))
    _maybe_put(bo_kw, "confirm_closes", _get(cfg, f"{base_ns}.breakouts.confirm_closes"))
    _maybe_put(bo_kw, "retest_lookahead", _get(cfg, f"{base_ns}.breakouts.retest_lookahead"))
    _maybe_put(bo_kw, "fail_break_lookahead", _get(cfg, f"{base_ns}.breakouts.fail_break_lookahead"))


    # --- microchannels ---
    mc_kw: Dict[str, Any] = {"anti_lookahead": anti_lookahead}
    _maybe_put(mc_kw, "min_len", _get(cfg, f"{base_ns}.microchannels.min_len"))
    _maybe_put(mc_kw, "near_extreme_thr", _get(cfg, f"{base_ns}.microchannels.near_extreme_thr"))


    # --- mtf_context ---
    mtf_kw: Dict[str, Any] = {"anti_lookahead": anti_lookahead}
    _maybe_put(mtf_kw, "bias_window_base", _get(cfg, f"{base_ns}.mtf.bias_window_base"))
    _maybe_put(mtf_kw, "bias_window_higher", _get(cfg, f"{base_ns}.mtf.bias_window_higher"))
    _maybe_put(mtf_kw, "fillna_confluence", _get(cfg, f"{base_ns}.mtf.fillna_confluence"))


    # --- confluence ---
    cf_kw: Dict[str, Any] = {"anti_lookahead": anti_lookahead}
    weights = _get(cfg, f"{base_ns}.confluence.weights")
    if isinstance(weights, dict):
        cf_kw["weights"] = copy.deepcopy(weights)
    _maybe_put(cf_kw, "strong_entry_threshold", _get(cfg, f"{base_ns}.confluence.strong_entry_threshold"))
    _maybe_put(cf_kw, "filter_threshold", _get(cfg, f"{base_ns}.confluence.filter_threshold"))


    return ms_kw, rg_kw, zn_kw, im_kw, bo_kw, mc_kw, mtf_kw, cf_kw, anti_lookahead


# ---------------------------------------------------------------------
# API 2: بیلد پرایس‌اکشن از روی کانفیگ (یک‌جا)
# ---------------------------------------------------------------------
def build_pa_features_from_config(
    df_base: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    df_higher: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    بیلد کل بستهٔ پرایس‌اکشن از روی cfg و صدا زدن ارکستریتور رسمی.

    ورودی‌ها:
        df_base : دیتای تایم‌فریم پایه (اجباری)
        cfg     : دیکت کانفیگ (بارگذاری از YAML)
        df_higher: دیتای تایم‌فریم بالادستی (اختیاری)

    خروجی:
        df_base غنی‌شده با تمام ستون‌های پرایس‌اکشن
    """
    (ms_kw, rg_kw, zn_kw, im_kw, bo_kw, mc_kw, mtf_kw, cf_kw, anti_lookahead) = extract_pa_kwargs_from_config(cfg)

    # پاس دادن kwargs فقط در صورت وجودشان؛ سایر پارامترها از پیش‌فرض‌های ماژول‌ها تأمین می‌شود.
    out = pa_reg.build_all_price_action_features(
        df_base,
        df_higher=df_higher,
        anti_lookahead=anti_lookahead,
        market_structure_kwargs=ms_kw if ms_kw else None,
        regime_kwargs=rg_kw if rg_kw else None,
        zones_kwargs=zn_kw if zn_kw else None,
        imbalance_kwargs=im_kw if im_kw else None,
        breakouts_kwargs=bo_kw if bo_kw else None,
        microchannels_kwargs=mc_kw if mc_kw else None,
        mtf_context_kwargs=mtf_kw if mtf_kw else None,
        confluence_kwargs=cf_kw if cf_kw else None,
    )
    return out
