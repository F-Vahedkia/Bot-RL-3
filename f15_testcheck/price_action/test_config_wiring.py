# f16_tests/price_action/test_config_wiring.py
"""
Unit Tests — Price Action Config Wiring
=======================================
این تست‌ها بررسی می‌کنند که:
- پارامترها به‌صورت امن از cfg استخراج می‌شوند (بدون نیاز به وجود همهٔ کلیدها)
- build_pa_features_from_config داده را با طول صحیح و ستون‌های کلیدی برمی‌گرداند
"""
# Run: pytest f16_tests/price_action/test_config_wiring.py -q

import pandas as pd
from f04_features.price_action import config_wiring as w


def _minimal_cfg_with_pa():
    # فقط چند کلید؛ بقیه باید از پیش‌فرض‌های ماژول‌ها تأمین شوند
    return {
        "features": {
            "price_action_params": {
                "anti_lookahead": True,
                "market_structure": {"lookback": 3},
                "breakouts": {"range_window": 6, "min_periods": 3, "confirm_closes": 1},
                "confluence": {
                    "weights": {"structure": 0.30, "zones": 0.30, "imbalance": 0.20, "mtf": 0.20},
                    "strong_entry_threshold": 0.7,
                    "filter_threshold": 0.35,
                },
            }
        }
    }


def test_extract_kwargs_safe_paths():
    cfg = _minimal_cfg_with_pa()
    ms_kw, rg_kw, zn_kw, im_kw, bo_kw, mc_kw, mtf_kw, cf_kw, anti = w.extract_pa_kwargs_from_config(cfg)

    assert anti is True
    assert ms_kw.get("lookback") == 3
    assert bo_kw.get("range_window") == 6
    assert bo_kw.get("min_periods") == 3
    assert bo_kw.get("confirm_closes") == 1
    assert isinstance(cf_kw.get("weights"), dict)
    # بعضی کیت‌ها عمداً تنظیم نشده‌اند؛ نباید خطا بدهد:
    assert "atr_window" not in rg_kw or isinstance(rg_kw.get("atr_window"), int) or rg_kw.get("atr_window") is None


def test_build_pa_features_from_config_end_to_end():
    cfg = _minimal_cfg_with_pa()
    base = pd.DataFrame({
        "high":  [10, 12, 11, 13, 15, 14, 16, 17, 16, 18, 20, 19],
        "low":   [ 9, 10, 10, 11, 13, 13, 14, 15, 14, 15, 17, 18],
        "close": [ 9.5, 11, 11, 12.8, 14.8, 13.5, 15.5, 16.5, 15.2, 17.2, 19.2, 18.5],
        "atr":   [ 0.5, 0.6, 0.55, 0.7, 0.8, 0.75, 0.85, 0.9, 0.88, 0.95, 1.0, 0.98],
    })
    higher = pd.DataFrame({"close": [9.5, 10.5, 10.2, 10.9, 11.3, 11.0]})

    out = w.build_pa_features_from_config(base, cfg, df_higher=higher)

    # چند ستون کلیدی باید اضافه شده باشد
    must_cols = [
        "swing_type",           # market_structure
        "breakout_up",          # breakouts
        "mtf_confluence_score", # mtf
        "conf_score",           # confluence
    ]
    for c in must_cols:
        assert c in out.columns
        assert len(out[c]) == len(base)
