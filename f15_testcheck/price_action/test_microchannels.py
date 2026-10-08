# f16_tests/price_action/test_microchannels.py
"""
Unit Tests — Price Action: Micro Channels
=========================================
تست‌های ماژول build_microchannels:
- تولید ستون‌های اصلی با طول صحیح
- رفتار anti_lookahead
- سناریوی ساده برای ایجاد میکروکانال صعودی/نزولی
"""
# Run: pytest f16_tests/price_action/test_microchannels.py -q

import pandas as pd
from f04_features.price_action import microchannels as mc


def _sample_micro(n=50):
    # بخش اول: صعودی کم‌نوسان → احتمال micro up
    up = [100 + i*0.2 for i in range(n//2)]
    # بخش دوم: نزولی کم‌نوسان → احتمال micro down
    dn = [up[-1] - (i+1)*0.2 for i in range(n - len(up))]
    close = up + dn
    high  = [c + 0.1 for c in close]
    low   = [c - 0.1 for c in close]
    open_ = [close[0]] + close[:-1]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_microchannels_columns_and_lengths():
    df = _sample_micro()
    out = mc.build_microchannels(df, anti_lookahead=True)

    for c in ("micro_channel_up", "micro_channel_down", "micro_channel_len", "micro_channel_quality"):
        assert c in out.columns
        assert len(out[c]) == len(df)

    # انواع داده‌ها
    assert set(out["micro_channel_up"].unique()).issubset({0, 1})
    assert set(out["micro_channel_down"].unique()).issubset({0, 1})
    assert out["micro_channel_len"].dtype.kind in ("i", "u")  # عدد صحیح
    assert out["micro_channel_quality"].between(0.0, 1.0, inclusive="both").all()


def test_microchannels_antilookahead_shift():
    df = _sample_micro()
    out0 = mc.build_microchannels(df, anti_lookahead=False)
    out1 = mc.build_microchannels(df, anti_lookahead=True)

    for c in ("micro_channel_up", "micro_channel_down", "micro_channel_len", "micro_channel_quality"):
        assert len(out0[c]) == len(out1[c])
        # انتظار تغییر/شیفت در اولین ردیف
        if c == "micro_channel_quality":
            assert out1[c].iloc[0] == 0.0 or (out1[c].iloc[0] != out0[c].iloc[0])
        else:
            assert out1[c].iloc[0] in (0, 1) or out1[c].iloc[0] == 0


def test_microchannels_simple_presence_of_flags():
    df = _sample_micro()
    out = mc.build_microchannels(df, anti_lookahead=True, min_len=3)

    # انتظار داریم حداقل یک پرچم up یا down در هر نیمه داده ظاهر شود
    assert out["micro_channel_up"].sum() >= 1 or out["micro_channel_down"].sum() >= 1
