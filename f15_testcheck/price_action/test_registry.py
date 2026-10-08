# f16_tests/price_action/test_registry.py
"""
Unit Tests — Price Action Registry & Orchestration (Extended)
=============================================================
"""
# Run: pytest f16_tests/price_action/test_registry.py -q

import pandas as pd
from f04_features.price_action import registry as pa_reg


def test_list_and_get_builders_extended():
    names = pa_reg.list_pa_features()
    assert isinstance(names, list) and len(names) >= 8
    for key in (
        "pa_market_structure","pa_zones","pa_imbalance","pa_mtf_context",
        "pa_confluence","pa_regime","pa_breakouts","pa_microchannels"
    ):
        assert key in names
        fn = pa_reg.get_pa_builder(key)
        assert callable(fn)


def test_build_all_price_action_features_end_to_end_extended():
    base = pd.DataFrame({
        "high":  [10, 12, 11, 13, 15, 14, 16, 17, 16, 18, 20, 19, 20, 21, 21.5, 22],
        "low":   [ 9, 10, 10, 11, 13, 13, 14, 15, 14, 15, 17, 18, 19, 20, 20.5, 21],
        "close": [ 9.5, 11, 11, 12.8, 14.8, 13.5, 15.5, 16.5, 15.2, 17.2, 19.2, 18.5, 19.5, 20.8, 21.2, 21.6],
        "atr":   [ 0.5, 0.6, 0.55, 0.7, 0.8, 0.75, 0.85, 0.9, 0.88, 0.95, 1.0, 0.98, 0.9, 0.92, 0.94, 0.96],
    })
    higher = pd.DataFrame({"close": [9.5, 10.5, 10.2, 10.9, 11.3, 11.0, 11.6]})

    out = pa_reg.build_all_price_action_features(
        base,
        df_higher=higher,
        anti_lookahead=True,
        # نمونه‌ای از override پارامترها (اختیاری)
        breakouts_kwargs=dict(confirm_closes=1, range_window=6, min_periods=3),
        regime_kwargs=dict(slope_window=6, width_window=8),
    )

    # چند ستون کلیدی از هر ماژول
    expected_some = [
        # market_structure
        "swing_type", "bos_up", "bos_down",
        # regime
        "regime_label", "regime_confidence",
        # zones
        "range_upper", "range_lower", "breakout_up", "breakout_down",  # از breakouts نیز انتظار داریم
        # imbalance
        # (اختیاری؛ بسته به indicators ممکن است ساخته شود)
        # mtf
        "mtf_bias", "mtf_confluence_score",
        # microchannels
        "micro_channel_up", "micro_channel_down",
        # confluence
        "conf_score", "conf_flag_filter_pass",
    ]
    for col in expected_some:
        assert col in out.columns
        assert len(out[col]) == len(base)
