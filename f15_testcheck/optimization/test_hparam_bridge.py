# f16_tests/optimization/test_hparam_bridge.py
"""
Unit Tests — Price Action Hyperparameter Bridge
===============================================
این تست‌ها بررسی می‌کنند که:
- نمونهٔ هایپرپارامترها با کلیدهای pa.* به درستی به config تزریق شود
- نگاشت وزن‌های confluence (لیست → دیکشنری) درست اعمال شود
- خروجی با وایرینگ قبلی build_pa_features_from_config سازگار باشد
"""
# Run: pytest f16_tests/optimization/test_hparam_bridge.py

import pandas as pd
from f14_optimization.hparam_bridge import apply_pa_hparams_to_config
from f04_features.price_action.config_wiring import build_pa_features_from_config


def test_apply_hparams_and_build():
    # config اولیه (بخش‌های لازم)
    cfg = {
        "features": {
            "price_action_params": {
                "anti_lookahead": True,
                "market_structure": {"lookback": 3},
                "confluence": {
                    "weights": {"structure": 0.25, "zones": 0.25, "imbalance": 0.20, "mtf": 0.10, "extras": 0.20},
                    "strong_entry_threshold": 0.70,
                    "filter_threshold": 0.35,
                },
            }
        }
    }

    # نمونهٔ هایپرپارامتر از sampler
    sample = {
        "pa.market_structure.lookback": 4,
        "pa.breakouts.range_window": 8,
        "pa.breakouts.min_periods": 3,
        "pa.breakouts.confirm_closes": 1,
        "pa.microchannels.min_len": 3,
        "pa.confluence.weights": [0.30, 0.30, 0.20, 0.10, 0.10],  # با extras
        "pa.confluence.strong_entry_threshold": 0.68,
    }

    cfg_out = apply_pa_hparams_to_config(cfg, sample)

    # اعتبارسنجی نگاشت
    w = cfg_out["features"]["price_action_params"]["confluence"]["weights"]
    assert w == {"structure": 0.30, "zones": 0.30, "imbalance": 0.20, "mtf": 0.10, "extras": 0.10}
    assert cfg_out["features"]["price_action_params"]["market_structure"]["lookback"] == 4
    assert cfg_out["features"]["price_action_params"]["breakouts"]["range_window"] == 8

    # سازگاری با بیلدر اصلی
    base = pd.DataFrame({
        "high":  [10, 12, 11, 13, 15, 14, 16, 17, 16, 18],
        "low":   [ 9, 10, 10, 11, 13, 13, 14, 15, 14, 15],
        "close": [ 9.5, 11, 11, 12.8, 14.8, 13.5, 15.5, 16.5, 15.2, 17.2],
        "atr":   [ 0.5, 0.6, 0.55, 0.7, 0.8, 0.75, 0.85, 0.9, 0.88, 0.95],
    })
    out = build_pa_features_from_config(base, cfg_out, df_higher=None)

    must_cols = ["swing_type", "breakout_up", "conf_score"]
    for c in must_cols:
        assert c in out.columns
        assert len(out[c]) == len(base)
