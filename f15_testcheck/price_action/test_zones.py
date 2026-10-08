"""
Unit Tests for Price Action Zones (Supply/Demand & Order Blocks)
================================================================
این تست‌ها ماژول zones را کنترل می‌کنند. اگر ماژول sr_advanced در indicators موجود نباشد،
تست‌ها به‌صورت تمیز skip می‌شوند (بدون خطا).
"""
# روش اجرای برنامه:
# pytest f16_tests/price_action/test_zones.py -q

import pandas as pd
import pytest

# اگر ماژول indicators.sr_advanced موجود نباشد، این تست‌ها skip می‌شوند.
sr_adv = pytest.importorskip(
    "f04_features.indicators.sr_advanced",
    reason="sr_advanced module not available; zones rely on existing indicators."
)

from f04_features.price_action import zones as zn


@pytest.fixture
def sample_df():
    # دیتای ساختگی با high/low/close و اختیاری atr
    data = {
        "high":  [10, 12, 11, 13, 15, 14, 16, 17, 16, 18, 20, 19],
        "low":   [ 9, 10, 10, 11, 13, 13, 14, 15, 14, 15, 17, 18],
        "close": [ 9.5, 11, 11, 12.8, 14.8, 13.5, 15.5, 16.5, 15.2, 17.2, 19.2, 18.5],
        "atr":   [ 0.5, 0.6, 0.55, 0.7, 0.8, 0.75, 0.85, 0.9, 0.88, 0.95, 1.0, 0.98],
    }
    return pd.DataFrame(data)


def test_build_zones_columns(sample_df):
    out = zn.build_zones(sample_df, anti_lookahead=True)

    # ستون‌های SD اگر indicators خروجی ساخته باشد
    sd_cols = {"sd_upper", "sd_lower", "sd_kind", "sd_age", "sd_fresh", "sd_touch_count"}
    ob_cols = {"ob_upper", "ob_lower", "ob_kind", "ob_age", "ob_strength", "ob_touch_count"}

    # وجود یا عدم وجود ستون‌ها به در دسترس بودن sr_advanced بستگی دارد
    # اما اگر باشند، طول باید برابر باشد.
    for grp in (sd_cols, ob_cols):
        present = [c for c in grp if c in out.columns]
        if present:
            assert len(out) == len(sample_df)

    # فاصله‌ها اگر ساخته شده باشند، باید طول برابر داشته باشند
    for col in ("dist_to_sd", "dist_to_ob"):
        if col in out.columns:
            assert len(out[col]) == len(sample_df)


def test_antilookahead_shift(sample_df):
    out_noshift = zn.build_zones(sample_df, anti_lookahead=False)
    out_shift  = zn.build_zones(sample_df, anti_lookahead=True)

    # اگر ستون sd/ob ساخته شده باشد، نسخه‌ی shift باید در اولین اندیس مقدار NaN بدهد
    for col in ("sd_upper", "sd_lower", "ob_upper", "ob_lower"):
        if col in out_noshift.columns and col in out_shift.columns:
            # مقایسه‌ی اولین ردیف: با shift باید NaN باشد یا متفاوت از نسخه‌ی بدون shift
            if pd.notna(out_noshift[col].iloc[0]):
                assert pd.isna(out_shift[col].iloc[0]) or out_shift[col].iloc[0] != out_noshift[col].iloc[0]
