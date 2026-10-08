# f16_tests/price_action/test_mtf_context.py
"""
Unit Tests — MTF Context (Price Action)
=======================================
تست‌های ماژول MTF Context:
- تولید ستون‌های کلیدی
- رفتار anti_lookahead
- محدودهٔ معتبر امتیاز همگرایی [0..1]
- هم‌راستاسازی دادهٔ Higher TF با Base
"""
# روش اجرای برنامه:
# pytest f16_tests/price_action/test_mtf_context.py -q

import pandas as pd
from f04_features.price_action import mtf_context as mtf


def test_mtf_context_without_higher():
    base = pd.DataFrame({
        "close": [10, 10.2, 10.4, 10.35, 10.6, 10.8, 10.75, 10.9, 11.0, 11.1]
    })
    out = mtf.build_mtf_context(base, df_higher=None, anti_lookahead=True)

    # ستون‌های اصلی
    for col in ("mtf_bias", "mtf_bias_local", "mtf_conflict", "mtf_confluence_score", "mtf_strength", "mtf_meta_coverage"):
        assert col in out.columns
        assert len(out[col]) == len(base)

    # امتیاز همگرایی باید در بازه [0..1] باشد
    assert out["mtf_confluence_score"].between(0.0, 1.0, inclusive="both").all()

    # coverage باید عددی بین 0 و 1 باشد
    cov = out["mtf_meta_coverage"].iloc[0]
    assert 0.0 <= cov <= 1.0


def test_mtf_context_with_higher_alignment():
    # Base: 20 نمونه، Higher: 5 نمونه (باید align شود)
    base = pd.DataFrame({"close": [100 + i * 0.5 for i in range(20)]})
    higher = pd.DataFrame({"close": [100, 101, 100.5, 101.2, 102.0]})
    out = mtf.build_mtf_context(base, df_higher=higher, anti_lookahead=True)

    # طول برابر با Base
    assert len(out) == len(base)

    # ستون‌های اصلی
    for col in ("mtf_bias", "mtf_bias_local", "mtf_conflict", "mtf_confluence_score", "mtf_strength", "mtf_meta_coverage"):
        assert col in out.columns

    # امتیاز همگرایی معتبر
    assert out["mtf_confluence_score"].between(0.0, 1.0, inclusive="both").all()

    # strength باید بین 0 و 1 باشد
    assert out["mtf_strength"].between(0.0, 1.0, inclusive="both").all()


def test_antilookahead_shift_effect():
    base = pd.DataFrame({"close": [10, 10.1, 10.2, 10.15, 10.3, 10.5, 10.45, 10.6]})
    out0 = mtf.build_mtf_context(base, anti_lookahead=False)
    out1 = mtf.build_mtf_context(base, anti_lookahead=True)

    # اثر شیفت: انتظار داریم اولین مقدار در نسخهٔ anti_lookahead به‌ازای برخی ستون‌ها NaN/تغییر کرده باشد
    for col in ("mtf_bias", "mtf_bias_local", "mtf_confluence_score"):
        assert len(out0[col]) == len(out1[col])
        # اجازه می‌دهیم NaN یا تغییرِ مقدار رخ دهد
        if pd.notna(out0[col].iloc[0]):
            assert (pd.isna(out1[col].iloc[0])) or (out0[col].iloc[0] != out1[col].iloc[0])
