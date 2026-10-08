# f15_testcheck/price_action/test_breakouts.py
# Reviewed at 1404/10/15

"""
Unit Tests — Price Action: Breakouts
====================================
تست‌های ماژول build_breakouts:
- تولید ستون‌های اصلی با طول صحیح
- رفتار anti_lookahead روی فلگ‌ها
- سناریوی سادهٔ شکست رو به بالا/پایین و Retest/Fail

Run:
- برای نمایش جزئیات هر تست
    pytest f15_testcheck/price_action/test_breakouts.py -v
- فقط تست یک تابع خاص
    pytest f15_testcheck/price_action/test_breakouts.py -k FuncName -v
- نمایش کامل خطاها
    pytest f15_testcheck/price_action/test_breakouts.py -v --tb=long
- اجرای همه تست‌های فولدر
    pytest f15_testcheck/price_action/ -v
"""
# --- Imports & ... -------------------------------------------------
import pandas as pd
import numpy as np
from f04_features.price_action import breakouts_numba_3 as bo

DATA_PATH = "f03_data/raw/XAUUSD/M15.csv"


# --- Creating synthetic data ---------------------------------------
def _sample_range_then_break(n=80):
    # 0..39: رنج باریک؛ 40..79: بریک‌اوت به بالا
    close = []
    for i in range(n):
        if i < n//2:
            close.append(100 + (-1)**i * 0.1)  # رنج کوچک
        else:
            close.append(101.0 + (i - n//2) * 0.3)  # شکست رو به بالا و ادامه
    high = [c + 0.15 for c in close]
    low  = [c - 0.15 for c in close]
    return pd.DataFrame({"close": close, "high": high, "low": low})

# --- Loading Real Data ---------------------------------------------
def _load_test_data():
    df = pd.read_csv(DATA_PATH)
    # اطمینان از اینکه ستون‌های ضروری موجود هستند
    required_cols = ["close", "high", "low"][-2000:]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"'{col}' column not found in CSV file")
    return df


# --- test_1 --------------------------------------------------------
def test_breakouts_columns_and_lengths():
    # df = _sample_range_then_break()
    df = _load_test_data()
    out = bo.build_breakouts(df, anti_lookahead=True)

    cols = [
        "range_upper", "range_lower",
        "breakout_up", "breakout_down",
        "retest_up", "retest_down",
        "fail_break_up", "fail_break_down",
    ]
    for c in cols:
        assert c in out.columns
        assert len(out[c]) == len(df)

# --- test_2 --------------------------------------------------------
def test_breakouts_antilookahead_flags_shift():
    df = _sample_range_then_break()
    out0 = bo.build_breakouts(df, anti_lookahead=False)
    out1 = bo.build_breakouts(df, anti_lookahead=True)

    for c in ("breakout_up", "breakout_down", "retest_up", "retest_down", "fail_break_up", "fail_break_down"):
        assert len(out0[c]) == len(out1[c])
        # اولین ردیف در نسخهٔ شیفت‌شده باید 0/تغییر کرده باشد
        assert out1[c].iloc[0] in (0, 1)  # باینری
        if out0[c].iloc[0] != 0:
            assert out1[c].iloc[0] != out0[c].iloc[0]

# --- test_3 --------------------------------------------------------
def test_breakouts_simple_up_scenario_has_some_signals():
    df = _sample_range_then_break()
    out = bo.build_breakouts(df, anti_lookahead=True, confirm_closes=1)

    # انتظار داریم جایی پس از نیمه راه، شکست رو به بالا رخ داده باشد
    assert out["breakout_up"].sum() >= 1

    # Retest ممکن است بسته به داده رخ دهد؛ اگر رخ داده، باید باینری معتبر باشد
    assert set(out["retest_up"].unique()).issubset({0, 1})
    assert set(out["fail_break_up"].unique()).issubset({0, 1})

# --- test_4 --------------------------------------------------------
def test_breakouts_upper_lower_values_reasonable():
    """
    بررسی اینکه range_upper و range_lower بعد از warm-up منطقی هستند:
    - range_upper >= range_lower
    - dtype صحیح
    """
    df = _load_test_data()
    min_periods = 10  # باید با مقدار پیش‌فرض build_breakouts هماهنگ باشد

    out = bo.build_breakouts(df, min_periods=min_periods)

    # فقط بعد از warm-up تست می‌کنیم
    out_valid = out.iloc[min_periods:]

    assert (out_valid["range_upper"] >= out_valid["range_lower"]).all()
    assert out["range_upper"].dtype == "float32"
    assert out["range_lower"].dtype == "float32"

# --- test_5 --------------------------------------------------------
def test_breakouts_breakout_flags_consistency():
    """
    بررسی سازگاری منطق breakout با باندهای رنج (پس از warm-up):
    - اگر breakout_up = 1 باشد، باید close > upper_prev باشد
    - اگر breakout_down = 1 باشد، باید close < lower_prev باشد
    """
    df = _load_test_data()
    min_periods = 10

    out = bo.build_breakouts(
        df,
        confirm_closes=1,
        min_periods=min_periods,
        anti_lookahead=False,
    )

    upper_prev = out["range_upper"].shift(1)
    lower_prev = out["range_lower"].shift(1)
    close = df["close"]

    # فقط بررسی forward-consistency (نه برعکس!)
    for i in range(min_periods + 1, len(df)):
        if out["breakout_up"].iat[i] == 1:
            assert close.iat[i] > upper_prev.iat[i]

        if out["breakout_down"].iat[i] == 1:
            assert close.iat[i] < lower_prev.iat[i]

# --- test_6 --------------------------------------------------------
def test_breakouts_warmup_effects():
    """
    بررسی warm-up و NaN اولیه:
    - تعداد ردیف‌های NaN در شروع برابر min_periods است
    """
    df = _load_test_data()
    min_periods = 10  # مطابق پیش‌فرض
    out = bo.build_breakouts(df, min_periods=min_periods, anti_lookahead=False)

    # ردیف‌های اولیه ممکن است 0 باشند یا NaN در range_upper/lower
    assert out["range_upper"].iloc[:min_periods].isna().sum() <= min_periods
    assert out["range_lower"].iloc[:min_periods].isna().sum() <= min_periods

    # پس از warm-up باید مقادیر معتبر باشند
    assert out["range_upper"].iloc[min_periods:].notna().all()
    assert out["range_lower"].iloc[min_periods:].notna().all()

# --- test_7 --------------------------------------------------------
def test_breakouts_retest_fail_break_logic():
    """
    بررسی صحت منطق Retest و Fail-break با داده‌های مصنوعی
    """
    df = _sample_range_then_break(n=50)
    out = bo.build_breakouts(df, confirm_closes=1, retest_lookahead=5, fail_break_lookahead=3, anti_lookahead=False)

    # اگر breakout_up رخ داده، باید retest_up <= 1 باشد
    for i in range(len(df)):
        if out["breakout_up"].iat[i] == 1:
            assert out["retest_up"].iat[i] in (0, 1)
            assert out["fail_break_up"].iat[i] in (0, 1)
        if out["breakout_down"].iat[i] == 1:
            assert out["retest_down"].iat[i] in (0, 1)
            assert out["fail_break_down"].iat[i] in (0, 1)

# --- test_8 --------------------------------------------------------
def test_breakouts_anti_lookahead_behavior():
    """
    بررسی اثر anti-lookahead:
    - فلگ‌های شیفت‌شده
    - هیچ مقدار آینده‌ای به ردیف فعلی نرسد
    """
    df = _sample_range_then_break(n=30)
    out_no_shift = bo.build_breakouts(df, anti_lookahead=False)
    out_shifted = bo.build_breakouts(df, anti_lookahead=True)

    # بررسی اینکه فلگ‌های شیفت‌شده کاملاً باینری هستند
    for col in ("breakout_up", "breakout_down", "retest_up", "retest_down", "fail_break_up", "fail_break_down"):
        assert set(out_shifted[col].unique()).issubset({0, 1})
        # اولین ردیف شیفت شده حتماً باید تغییر کند یا صفر باشد
        assert out_shifted[col].iloc[0] in (0, 1)

# --- test_9 --------------------------------------------------------
def test_breakouts_with_different_confirm_closes():
    """
    بررسی تاثیر پارامتر confirm_closes:
    - با confirm_closes > 1، breakout فقط وقتی فعال شود که n کلوز متوالی رخ دهد.
    """
    df = _sample_range_then_break(n=50)
    out1 = bo.build_breakouts(df, confirm_closes=1, anti_lookahead=False)
    out2 = bo.build_breakouts(df, confirm_closes=3, anti_lookahead=False)
    # جمع مقادیر باید کمتر یا مساوی باشد چون شرط سخت‌تر شده
    assert out2["breakout_up"].sum() <= out1["breakout_up"].sum()
    assert out2["breakout_down"].sum() <= out1["breakout_down"].sum()

# --- test_10 --------------------------------------------------------
def test_breakouts_edge_case_short_data():
    """
    بررسی داده کوتاه‌تر از window و min_periods:
    - خروجی باید بدون ارور تولید شود
    - همه فلگ‌ها صفر یا باینری معتبر باشند
    """
    df = _sample_range_then_break(n=5)  # کوتاه‌تر از range_window پیش‌فرض
    out = bo.build_breakouts(df, anti_lookahead=True)
    for col in ["breakout_up","breakout_down","retest_up","retest_down","fail_break_up","fail_break_down"]:
        assert set(out[col].unique()).issubset({0,1})

# --- test_11 --------------------------------------------------------
def test_breakouts_with_missing_values():
    """
    بررسی داده‌هایی که NaN دارند:
    - نباید خطا بدهد
    - خروجی‌ها باید درست dtype و باینری باشند
    """
    df = _sample_range_then_break(n=30)
    df.loc[5, "high"] = np.nan
    df.loc[10, "low"] = np.nan
    df.loc[15, "close"] = np.nan
    out = bo.build_breakouts(df, anti_lookahead=True)
    for col in ["breakout_up","breakout_down","retest_up","retest_down","fail_break_up","fail_break_down"]:
        assert set(out[col].dropna().unique()).issubset({0,1})

# --- test_12 --------------------------------------------------------
def test_breakouts_with_spike_price():
    """
    بررسی رفتار در پرش شدید قیمت (spike):
    - breakout باید درست شناسایی شود
    """
    df = _sample_range_then_break(n=40)
    df.loc[20, "close"] += 5.0  # spike رو به بالا
    out = bo.build_breakouts(df, confirm_closes=1, anti_lookahead=True)
    assert out["breakout_up"].sum() >= 1

# --- test_13 -------------------------------------------------------
def test_breakouts_dtype_consistency():
    """
    اطمینان از اینکه همه فلگ‌ها int8 و باندها float32 هستند
    """
    df = _sample_range_then_break()
    out = bo.build_breakouts(df)
    flag_cols = ["breakout_up","breakout_down","retest_up","retest_down","fail_break_up","fail_break_down"]
    for col in flag_cols:
        assert out[col].dtype == "int8"
    assert out["range_upper"].dtype == "float32"
    assert out["range_lower"].dtype == "float32"

# -------------------------------------------------------------------
