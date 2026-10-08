"""
Unit Tests for Price Action: Imbalance & Liquidity
==================================================
این تست‌ها ماژول imbalance را کنترل می‌کنند. اگر توابع indicators موجود نباشند،
تست‌ها همچنان اجرا می‌شوند اما ستون‌های مربوطه ممکن است حاضر نباشند؛
در این صورت assertها به‌صورت شرطی عمل می‌کنند.
"""
# روش اجرای برنامه:
# pytest f16_tests/price_action/test_imbalance.py -q

import importlib.util
import pandas as pd
import pytest

from f04_features.price_action import imbalance as imba


def _has_sr_adv() -> bool:
    return importlib.util.find_spec("f04_features.indicators.sr_advanced") is not None


@pytest.fixture
def sample_df():
    data = {
        "high":  [10, 12, 11, 13, 15, 14, 16, 17, 16, 18, 20, 19],
        "low":   [ 9, 10, 10, 11, 13, 13, 14, 15, 14, 15, 17, 18],
        "close": [ 9.5, 11, 11, 12.8, 14.8, 13.5, 15.5, 16.5, 15.2, 17.2, 19.2, 18.5],
        "atr":   [ 0.5, 0.6, 0.55, 0.7, 0.8, 0.75, 0.85, 0.9, 0.88, 0.95, 1.0, 0.98],
    }
    return pd.DataFrame(data)


def test_build_imbalance_liquidity_columns(sample_df):
    out = imba.build_imbalance_liquidity(sample_df, anti_lookahead=True)
    assert len(out) == len(sample_df)

    # اگر sr_advanced/LP حاضر باشد، این ستون‌ها ممکن است وجود داشته باشند
    fvg_cols = {"fvg_upper", "fvg_lower", "fvg_mid", "fvg_thickness", "fvg_age", "fvg_filled_ratio"}
    lp_cols = {"lp_high_price", "lp_low_price"}
    sw_cols = {"sweep_up", "sweep_down"}
    dist_cols = {"dist_to_fvg_mid", "dist_to_lp_high", "dist_to_lp_low"}

    # حضور ستون‌ها اختیاری است؛ اگر بودند طولشان باید درست باشد
    for grp in (fvg_cols, lp_cols, sw_cols, dist_cols):
        present = [c for c in grp if c in out.columns]
        for c in present:
            assert len(out[c]) == len(sample_df)


def test_antilookahead_shift_behavior(sample_df):
    out0 = imba.build_imbalance_liquidity(sample_df, anti_lookahead=False)
    out1 = imba.build_imbalance_liquidity(sample_df, anti_lookahead=True)

    # اگر ستون‌هایی ساخته شده باشند، نسخه‌ی shift باید در اولین ردیف NaN/متفاوت باشد
    candidate_cols = [
        "fvg_upper", "fvg_lower", "fvg_mid",
        "lp_high_price", "lp_low_price",
        "sweep_up", "sweep_down",
    ]
    for col in candidate_cols:
        if col in out0.columns and col in out1.columns:
            # برای فلگ‌های باینری هم انتظار می‌رود ردیف اول shift شده باشد
            if pd.notna(out0[col].iloc[0]):
                assert pd.isna(out1[col].iloc[0]) or out0[col].iloc[0] != out1[col].iloc[0]
