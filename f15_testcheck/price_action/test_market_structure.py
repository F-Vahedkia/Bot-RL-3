"""
Unit Tests for Market Structure (Price Action)
==============================================
این تست‌ها رفتار ماژول market_structure را کنترل می‌کنند.
- تست تشخیص سوئینگ‌ها (HH, HL, LH, LL)
- تست تشخیص BOS / CHoCH
- تست اجرای کامل build_market_structure
"""
# روش اجرای برنامه:
# pytest f16_tests/price_action/test_market_structure.py -q

import pandas as pd
import pytest, sys
from pathlib import Path
from f04_features.price_action import market_structure as ms

# ================================================================
# داده نمونه برای تست‌ها
# ================================================================
@pytest.fixture
def sample_df():
    """
    دیتافریم ساده با داده‌های ساختگی high/low برای تست.
    """
    data = {
        "high": [10, 12, 11, 13, 15, 14, 16, 14, 13, 17],
        "low":  [ 9, 10, 10, 11, 13, 13, 14, 12, 11, 15],
    }
    return pd.DataFrame(data)


# ================================================================
# تست: تشخیص سوئینگ‌ها
# ================================================================
def test_detect_swings(sample_df):
    df = ms.detect_swings(sample_df, lookback=2)
    # ستون باید وجود داشته باشد
    assert "swing_type" in df.columns
    # مقدار باید None یا یکی از نوع‌های ساختاری باشد
    valid_types = [None, "HH", "HL", "LH", "LL"]
    assert df["swing_type"].apply(lambda x: x in valid_types).all()


# ================================================================
# تست: تشخیص BOS / CHoCH
# ================================================================
def test_detect_bos_choch(sample_df):
    df = ms.detect_swings(sample_df, lookback=2)
    df = ms.detect_bos_choch(df)

    # ستون‌های BOS/CHoCH باید وجود داشته باشند
    for col in ["bos_up", "bos_down", "choch_up", "choch_down"]:
        assert col in df.columns
        # فقط مقادیر 0 یا 1 مجاز هستند
        assert set(df[col].unique()).issubset({0, 1})


# ================================================================
# تست: اجرای کامل build_market_structure
# ================================================================
def test_build_market_structure(sample_df):
    df = ms.build_market_structure(sample_df, lookback=2)

    # همه ستون‌های کلیدی باید ساخته شوند
    expected_cols = ["swing_type", "bos_up", "bos_down", "choch_up", "choch_down"]
    for col in expected_cols:
        assert col in df.columns

    # دیتافریم باید همان طول ورودی را داشته باشد
    assert len(df) == len(sample_df)
