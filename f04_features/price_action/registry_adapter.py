# f04_features/price_action/registry_adapter.py
# Status in (Bot-RL-3): Reviewed at 1404/10/15

"""
این ماژول بدون دستکاری رجیستری اندیکاتورهای موجود، یک آداپتر ارائه می‌کند تا
بیلدرهای پرایس‌اکشن را به رجیستری مرکزی (mapping name -> builder) اضافه کنید.

API:
- get_price_action_registry() -> dict[str, callable]
- register_price_action_to_indicators_registry(registry_dict: dict) -> dict

نکات:
- اگر کلیدها قبلاً در رجیستری وجود داشته باشند، با پیشوند جلوگیری از تداخل انجام می‌دهد.
"""

#==============================================================================
# Imports
#==============================================================================
from __future__ import annotations
from typing import Callable, Dict, Optional
import pandas as pd

from f04_features.price_action import (
    breakouts_1 as _bo,
    confluence_1 as _cf,
    imbalance as _im,
    market_structure as _ms,
    microchannels as _mc,
    mtf_context as _mtf,
    regime as _rg,
    zones as _zn,
)

# -------------------------------------------------------------------
# نگاشت استاندارد نام→بیلدر برای PA
# نام‌ها را ساده و شفاف نگه می‌داریم؛ در صورت نیاز می‌توانند در DSL نیز استفاده شوند.
# -------------------------------------------------------------------
_PA_REGISTRY: Dict[str, Callable] = {
    "pa_breakouts":        _bo.build_breakouts,
    "pa_confluence":       _cf.build_confluence,
    "pa_imbalance":        _im.build_imbalance_liquidity,
    "pa_market_structure": _ms.build_market_structure,
    "pa_microchannels":    _mc.build_microchannels,
    "pa_mtf_context":     _mtf.build_mtf_context,
    "pa_regime":           _rg.build_regime,
    "pa_zones":            _zn.build_zones,
    # "pa_regime_state":     _ms.build_regime_state_pro,
}

# -------------------------------------------------------------------
# دسترسی به لیست نام‌ها و بیلدرها (merge شده از registry.py)
# -------------------------------------------------------------------
def list_pa_features() -> list[str]:
    return list(_PA_REGISTRY.keys())


def get_pa_builder(name: str) -> Callable:
    if name not in _PA_REGISTRY:
        raise KeyError(f"Unknown PA feature builder: {name}")
    return _PA_REGISTRY[name]

# -------------------------------------------------------------------
# برگرداندن نگاشت کامل بیلدرهای پرایس‌اکشن.
# -------------------------------------------------------------------
def register_price_action_to_indicators_registry(registry_dict: Dict[str, Callable]) -> Dict[str, Callable]:
    if registry_dict is None or not isinstance(registry_dict, dict):
        raise TypeError("registry_dict must be a dict mapping from str to callable.")
    for k, fn in _PA_REGISTRY.items():
        if k in registry_dict and registry_dict.get(k) is not fn:
            alt = f"pa__{k}"
            if alt not in registry_dict:
                registry_dict[alt] = fn
        else:
            registry_dict.setdefault(k, fn)
    return registry_dict


def get_price_action_registry() -> Dict[str, Callable]:
    return dict(_PA_REGISTRY)

# -----------------------------------------------------------------------------
# Orchestrator: ساخت یک‌جای بستهٔ پرایس‌اکشن
# ----------------------------------------------------------------------------- از اینجا مانده است
def build_all_price_action_features(
    df_base: pd.DataFrame,
    *,
    df_higher: Optional[pd.DataFrame] = None,
    anti_lookahead: bool = True,
    # پارامترهای اختیاری عبوری به زیرماژول‌ها:
    market_structure_kwargs: Optional[dict] = None,
    zones_kwargs: Optional[dict] = None,
    imbalance_kwargs: Optional[dict] = None,
    mtf_context_kwargs: Optional[dict] = None,
    confluence_kwargs: Optional[dict] = None,
    regime_kwargs: Optional[dict] = None,
    breakouts_kwargs: Optional[dict] = None,
    microchannels_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    ترتیب اجرای پیشنهادی:
    1) Market Structure
    2) Regime Detector
    3) Zones (SD/OB)
    4) Imbalance & Liquidity (FVG, Sweeps)
    5) Breakouts (وابسته به رنج/کانال rolling)
    6) Micro-channels
    7) MTF Context (با df_higher اختیاری)
    8) Confluence (تجمیع نهایی)
    """
    ms_kw = dict(lookback=3)
    if market_structure_kwargs: ms_kw.update(market_structure_kwargs)

    rg_kw = dict(anti_lookahead=anti_lookahead)
    if regime_kwargs: rg_kw.update(regime_kwargs)

    zn_kw = dict(anti_lookahead=anti_lookahead)
    if zones_kwargs: zn_kw.update(zones_kwargs)

    im_kw = dict(anti_lookahead=anti_lookahead)
    if imbalance_kwargs: im_kw.update(imbalance_kwargs)

    bo_kw = dict(anti_lookahead=anti_lookahead)
    if breakouts_kwargs: bo_kw.update(breakouts_kwargs)

    mc_kw = dict(anti_lookahead=anti_lookahead)
    if microchannels_kwargs: mc_kw.update(microchannels_kwargs)

    mtf_kw = dict(
                    df_higher=df_higher,
                    anti_lookahead=anti_lookahead,
                    base_close_col="close",
                    higher_close_col="close")
    if mtf_context_kwargs: mtf_kw.update(mtf_context_kwargs)

    cf_kw = dict(
                    anti_lookahead=anti_lookahead,
                    strong_entry_threshold=0.7,
                    filter_threshold=0.35)
    if confluence_kwargs: cf_kw.update(confluence_kwargs)

    out = df_base.copy()

    out = _ms. build_market_structure   (out, **ms_kw)
    out = _rg. build_regime             (out, **rg_kw)
    out = _zn. build_zones              (out, **zn_kw)
    out = _im. build_imbalance_liquidity(out, **im_kw)
    out = _bo. build_breakouts          (out, **bo_kw)
    out = _mc. build_microchannels      (out, **mc_kw)
    out = _mtf.build_mtf_context        (out, **mtf_kw)
    out = _cf. build_confluence         (out, **cf_kw)

    return out

# -----------------------------------------------------------------------------
