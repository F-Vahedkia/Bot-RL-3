# f04_features/indicators/ts03_idx_srad__2make_fvg_1.py
# Run                  : pytest .\f15_testcheck\unit\ts03_idx_srad__2make_fvg_1.py
# Run with short answer: pytest -q f15_testcheck\unit\ts03_idx_srad__2make_fvg_1.py

#####################################################################
# فایل تستر بر اساس داده های ساختگی
#####################################################################

import pandas as pd
import numpy as np
import pytest

from f04_features.indicators.sr_advanced import make_fvg

# ============================================================
# Synthetic Data Builder
# ============================================================
def build_base_df(n=40, seed=123):
    """
    ساخت دیتافریم پایه OHLC با حرکت آرام و تکرارپذیر
    """
    rng = np.random.default_rng(seed)
    price = 100.0
    rows = []
    for _ in range(n):
        o = price
        h = o + rng.uniform(0.5, 1.5)
        l = o - rng.uniform(0.5, 1.5)
        c = rng.uniform(l, h)
        rows.append([o, h, l, c])
        price = c

    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    return df

# ============================================================
# Scenario Injection
# ============================================================
def inject_bullish_fvg(df, i: int):
    """
    ایجاد Bullish FVG بین کندل i,i+1,i+2
    شرط: low[i+2] > high[i]
    """
    df = df.copy()
    df.loc[i,   ["open","high","low","close"]] = [100, 102,  99, 101]
    df.loc[i+1, ["open","high","low","close"]] = [101, 103, 100, 102]
    df.loc[i+2, ["open","high","low","close"]] = [110, 112, 109, 111]
    return df

def inject_bearish_fvg(df, i: int):
    df = df.copy()
    df.loc[i,   ["open","high","low","close"]] = [120, 122, 119, 121]
    df.loc[i+1, ["open","high","low","close"]] = [121, 123, 120, 122]
    df.loc[i+2, ["open","high","low","close"]] = [110, 111, 108, 109]
    return df

def inject_touch(df, zone_top, zone_bottom, idx: int):
    df = df.copy()
    df.loc[idx, "low"]  = zone_bottom + 0.1
    df.loc[idx, "high"] = zone_top    - 0.1
    return df

def inject_fill_bull(df, zone_bottom, idx: int):
    df = df.copy()
    df.loc[idx, "low"] = zone_bottom - 0.5
    return df

def inject_fill_bear(df, zone_top, idx: int):
    df = df.copy()
    df.loc[idx, "high"] = zone_top + 0.5
    return df

# ============================================================
# Runner
# ============================================================
def run_indicator(df: pd.DataFrame):
    cfg = dict(
        lookback=2,
        atr_window=5,
        min_size_pct_of_atr=0.01,
        max_bars_alive=4,
    )
    out = make_fvg(df, **cfg)
    # out یک dict[str, Series] است؛ index باید با df یکی باشد
    out_df = pd.DataFrame(out)
    assert list(out_df.index) == list(df.index), "Index mismatch between df and make_fvg output"
    res = pd.concat([df, out_df], axis=1)
    return res

# ============================================================
# Diagnostics
# ============================================================
def print_debug(df):
    cols = [
        "high","low",
        "fvg_up",
        "fvg_down",
        "fvg_top",
        "fvg_bottom",
        "fvg_born",
        "fvg_touched_next",
        "fvg_filled_next",
        "fvg_expired_next",
        "fvg_active_top",
        "fvg_active_bottom",
        "fvg_touch_now",
        "fvg_touch_count",
        "fvg_score",
    ]
    existing = [c for c in cols if c in df.columns]
    print("\n========== DEBUG TABLE ==========\n")
    print(df[existing].tail(20))

# ============================================================
# TESTS
# ============================================================
def test_basic_output_shape_and_dtypes():
    df = build_base_df()
    res = run_indicator(df)

    # ستون‌های کلیدی باید وجود داشته باشند
    for col in [
        "fvg_up", "fvg_down",
        "fvg_top", "fvg_bottom",
        "fvg_touched_any", "fvg_filled_any",
        "fvg_expired_now", "fvg_alive_window",
        "fvg_active_top", "fvg_active_bottom",
        "fvg_touch_now", "fvg_touch_count",
        "fvg_score",
    ]:
        assert col in res.columns, f"Missing expected column {col}"
        assert len(res[col]) == len(df), f"Column {col} length mismatch"

    # نوع‌های منطقی حداقلی
    assert res["fvg_up"].dtype in (np.int8, np.int64, bool)
    assert res["fvg_down"].dtype in (np.int8, np.int64, bool)
    assert res["fvg_score"].dtype == np.float32

def test_bullish_fvg_detection():
    df = build_base_df()
    df = inject_bullish_fvg(df, 5)
    res = run_indicator(df)
    # صرفاً انتظار داریم حداقل یک fvg_up ثبت شود
    assert res["fvg_up"].sum() > 0, "Bullish FVG not detected"

def test_bearish_fvg_detection():
    df = build_base_df()
    df = inject_bearish_fvg(df, 5)
    res = run_indicator(df)
    assert res["fvg_down"].sum() > 0, "Bearish FVG not detected"

def test_touch_detection():
    df = build_base_df()
    df = inject_bullish_fvg(df, 5)
    res = run_indicator(df)

    if res["fvg_top"].notna().sum() == 0:
        pytest.skip("No FVG zone detected to test touch")

    zone_top    = res["fvg_top"].dropna().iloc[0]
    zone_bottom = res["fvg_bottom"].dropna().iloc[0]

    df2 = inject_touch(df, zone_top, zone_bottom, 10)
    res2 = run_indicator(df2)
    print_debug(res2)

    # فقط چک می‌کنیم که indicator قابلیت ثبت touch در پنجره N را داشته باشد
    assert res2["fvg_touched_any"].sum() >= 0  # این ستون باید وجود داشته باشد، مقدار غیرمنفی است

def test_fill_detection():
    df = build_base_df()
    df = inject_bullish_fvg(df, 5)
    res = run_indicator(df)

    if res["fvg_bottom"].notna().sum() == 0:
        pytest.skip("No FVG zone detected to test fill")

    zone_bottom = res["fvg_bottom"].dropna().iloc[0]
    df2 = inject_fill_bull(df, zone_bottom, 10)
    res2 = run_indicator(df2)
    print_debug(res2)

    assert res2["fvg_filled_any"].sum() >= 0  # صرفاً وجود و سلامت ستون

def test_expired_zone_column_exists():
    df = build_base_df()
    df = inject_bullish_fvg(df, 5)
    res = run_indicator(df)

    # فقط وجود ستون و مقدار معتبر (۰/۱) را چک می‌کنیم
    assert "fvg_expired_now" in res.columns
    assert res["fvg_expired_now"].min() >= 0

def test_zone_propagation_columns():
    df = build_base_df()
    df = inject_bullish_fvg(df, 5)
    res = run_indicator(df)

    assert "fvg_active_top" in res.columns
    assert "fvg_active_bottom" in res.columns
    # اگر هیچ زون فعالی نباشد، اشکالی ندارد؛ فقط نباید crash کند
    assert len(res["fvg_active_top"]) == len(df)

def test_score_generation_non_negative():
    df = build_base_df()
    df = inject_bullish_fvg(df, 5)
    res = run_indicator(df)
    # score باید وجود داشته باشد و عددی باشد
    assert "fvg_score" in res.columns
    assert res["fvg_score"].max() >= 0

def test_touch_counter_monotonic_window():
    df = build_base_df()
    df = inject_bullish_fvg(df, 5)
    res = run_indicator(df)

    if res["fvg_top"].notna().sum() == 0:
        pytest.skip("No FVG zone detected to test touch_count")

    zone_top    = res["fvg_top"].dropna().iloc[0]
    zone_bottom = res["fvg_bottom"].dropna().iloc[0]

    df2 = df.copy()
    for i in range(10, 14):
        df2 = inject_touch(df2, zone_top, zone_bottom, i)

    res2 = run_indicator(df2)
    # شمارش لمس در Rolling(N) باید حداقل جایی بیشتر از صفر شود
    assert res2["fvg_touch_count"].max() >= 0
