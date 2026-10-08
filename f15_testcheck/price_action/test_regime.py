# f16_tests/price_action/test_regime.py
"""
Unit Tests — Price Action: Regime Detector
==========================================
تست‌های ماژول build_regime:
- وجود ستون‌های کلیدی و طول برابر با ورودی
- رفتار anti_lookahead (شیفت +1)
- دامنهٔ مقادیر شدت‌ها [0..1] و معتبر بودن برچسب‌ها
"""
# Run: pytest f16_tests/price_action/test_regime.py -q

import pandas as pd
from f04_features.price_action import regime as rg


def _sample_df(n=60):
    # دادهٔ ساختگی با فراز و فرود برای پوشش هر سه رِژیم
    base = [100 + i*0.05 for i in range(n//3)]                 # شبه کانال ملایم
    spike = [base[-1] + i*0.5 for i in range(n//6)]             # جهش تند (اسپایک)
    chop  = [spike[-1] + (-1)**i * 0.15 for i in range(n - len(base) - len(spike))]  # رنج
    close = base + spike + chop
    high  = [c + 0.1 for c in close]
    low   = [c - 0.1 for c in close]
    return pd.DataFrame({"close": close, "high": high, "low": low})


def test_regime_columns_ranges():
    df = _sample_df()
    out = rg.build_regime(df, anti_lookahead=True)

    # ستون‌ها
    for col in ("regime_range", "regime_spike", "regime_channel", "regime_label", "regime_confidence"):
        assert col in out.columns
        assert len(out[col]) == len(df)

    # شدت‌ها در بازهٔ [0..1]
    for col in ("regime_range", "regime_spike", "regime_channel", "regime_confidence"):
        s = out[col]
        assert s.min() >= 0.0 - 1e-9
        assert s.max() <= 1.0 + 1e-9

    # برچسب‌ها معتبر باشند
    valid = {"range", "spike", "channel", None}
    assert set(out["regime_label"].drop_duplicates().tolist()).issubset(valid)


def test_regime_antilookahead():
    df = _sample_df()
    out0 = rg.build_regime(df, anti_lookahead=False)
    out1 = rg.build_regime(df, anti_lookahead=True)

    for col in ("regime_range", "regime_spike", "regime_channel"):
        assert len(out0[col]) == len(out1[col])
        if pd.notna(out0[col].iloc[0]):
            # پس از شیفت، اولین ردیف باید 0/NaN/تغییر کرده باشد
            v1 = out1[col].iloc[0]
            assert (pd.isna(v1)) or (abs(out0[col].iloc[0] - v1) > 1e-12) or (v1 == 0.0)
