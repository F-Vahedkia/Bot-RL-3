# -*- coding: utf-8 -*-
# f14_optimization/hparam_bridge.py
# Status in (Bot-RL-3): Completed

"""
Hyperparameter Bridge for Price Action
======================================
این ماژول یک پل ایمن است تا دیکشنری نمونه‌ی هایپرپارامترها (کلیدها با
فرمت 'pa.*') را به ساختار config شما تزریق کند (features.price_action_params.*).
- بدون تغییر f14_optimization
- بدون حدس: فقط کلیدهای موجود در sample ست می‌شوند
- سازگار با وزن‌های confluence و وزن extras
"""

from __future__ import annotations
from typing import Any, Dict, List


_BASE_NS = ("features", "price_action_params")


def _ensure_path(cfg: Dict[str, Any], path: List[str]) -> Dict[str, Any]:
    cur = cfg
    for k in path:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    return cur


def _set_nested(cfg: Dict[str, Any], path: List[str], value: Any) -> None:
    parent = _ensure_path(cfg, path[:-1])
    parent[path[-1]] = value


def apply_pa_hparams_to_config(cfg: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    ورودی:
      cfg    : دیکشنری کانفیگ (از YAML)
      sample : دیکشنری خروجی sampler با کلیدهایی مثل:
               'pa.breakouts.range_window', 'pa.confluence.weights', ...

    خروجی:
      همان cfg با مقادیر اعمال‌شده (reference-preserving)
    """
    for k, v in (sample or {}).items():
        if not isinstance(k, str) or not k.startswith("pa."):
            continue
        parts = k.split(".")
        # مثال: pa.breakouts.range_window  →  features.price_action_params.breakouts.range_window
        path = list(_BASE_NS) + parts[1:]

        # نگاشت خاص وزن‌های confluence (لیست → دیکشنری)
        if k == "pa.confluence.weights":
            # انتظار یک لیست با 4 یا 5 وزن: [structure, zones, imbalance, mtf] (+extras اختیاری)
            if not isinstance(v, (list, tuple)) or len(v) < 4:
                continue
            w = {
                "structure": float(v[0]),
                "zones": float(v[1]),
                "imbalance": float(v[2]),
                "mtf": float(v[3]),
            }
            if len(v) >= 5:
                w["extras"] = float(v[4])
            _set_nested(cfg, list(_BASE_NS) + ["confluence", "weights"], w)
            continue

        _set_nested(cfg, path, v)

    return cfg
