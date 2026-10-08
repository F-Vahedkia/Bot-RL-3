# f15_testcheck/unit/ts03_idx_srad__1detect_fvg_1.py
# Run                  : pytest .\f15_testcheck\unit\ts03_idx_srad__1detect_fvg_1.py
# Run with short answer: pytest -q f15_testcheck\unit\ts03_idx_srad__1detect_fvg_1.py

#####################################################################
# فایل تستر براساس داده های ساختگی
#####################################################################

import pandas as pd
import numpy as np

from f04_features.indicators.sr_advanced import detect_fvg

# -------------------------------------------------------------------
# Helper – constructor
# -------------------------------------------------------------------
def make(df):
    return detect_fvg(
        open_=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        lookback=2,
        atr_window=3,
        min_size_pct_of_atr=0.0,
    )

# ================================================================
# 1) BASIC SANITY TEST
# ================================================================
def test_detect_fvg_basic_structure():
    df = pd.DataFrame({
        "open":  [1,2,3,4,5],
        "high":  [2,3,4,5,6],
        "low":   [0,1,2,3,4],
        "close": [1,2,3,4,5],
    })
    out = make(df)

    assert set(out.keys()) == {"fvg_up", "fvg_dn", "fvg_top", "fvg_bottom"}
    assert all(len(s) == 5 for s in out.values())
    assert out["fvg_up"].dtype == "int8"
    assert out["fvg_dn"].dtype == "int8"
    assert out["fvg_top"].dtype == "float32"
    assert out["fvg_bottom"].dtype == "float32"

# ================================================================
# 2) BULLISH FVG – strict deterministic
# ================================================================
def test_detect_fvg_bullish_strict():
    # bull @ index=3 → shift → flag @ 4
    df = pd.DataFrame({
        "open":  [10,11,12,15,20],
        "high":  [11,12,13,16,19],
        "low":   [ 9,10,11,14,18],
        "close": [10,11,12,15,19],
    })

    out = make(df)
    fup = out["fvg_up"]
    ftop = out["fvg_top"]
    fbot = out["fvg_bottom"]

    # bull rule: low[3] > high[1] → 14 > 12
    assert fup[3] == 0
    assert fup[4] == 1

    # zone only exists at origin bar
    assert np.isnan(ftop[2])
    assert np.isclose(ftop[3], 14)
    assert np.isnan(ftop[4])

    assert np.isnan(fbot[2])
    assert np.isclose(fbot[3], 12)
    assert np.isnan(fbot[4])

# ================================================================
# 3) BEARISH FVG – strict deterministic
# ================================================================
def test_detect_fvg_bearish_strict():
    # bear @ index=3 → shift → flag @ 4
    # bear rule: high[i] < low[i-2]
    df = pd.DataFrame({
        "open":  [10,11,12,15,20],
        "high":  [11,12,13, 8,19],   # high[3]=8
        "low":   [ 9,10,11,14,18],   # low[1]=10 → 8 < 10 TRUE
        "close": [10,11,12,15,19],
    })

    out = make(df)
    fdn = out["fvg_dn"]
    ftop = out["fvg_top"]
    fbot = out["fvg_bottom"]

    # confirm bear triggered at shifted index
    assert fdn[3] == 0
    assert fdn[4] == 1

    # zone at origin bar=3
    # fvg_top = high[i-2] = high[1] = 12
    # fvg_bottom = low[i] = low[3] = 14
    assert np.isclose(ftop[3], 12)
    assert np.isclose(fbot[3], 14)
    assert np.isnan(ftop[4])
    assert np.isnan(fbot[4])

# ================================================================
# 4) MULTIPLE SEQUENTIAL FVGs
# ================================================================
def test_detect_fvg_multiple_chain():
    # bull at 3, bull at 4, bear at 5
    df = pd.DataFrame({
        "open":  [1,2,3,  10, 15, 20, 25],
        "high":  [2,3,4,  12, 17,  5,30],
        "low":   [0,1,2,  11, 16,  2,28],
        "close": [1,2,3,  11, 16,  3,29],
    })

    out = make(df)

    # bull @3 → flag@4
    assert out["fvg_up"][4] == 1
    # bull @4 → flag@5
    assert out["fvg_up"][5] == 1

    # bear @5? since high[5]=5 < low[3]=11 → TRUE
    assert out["fvg_dn"][6] == 1

# ================================================================
# 5) OVERLAPPING FVG ZONES
# ================================================================
def test_detect_fvg_overlapping_zones():
    # طراحی دو زون که مقادیرشان روی نمودارOverlap شود
    df = pd.DataFrame({
        "open":  [1,2,3,  10,15,20],
        "high":  [2,3,4,  12,17,13],
        "low":   [0,1,2,  11,16, 9],
        "close": [1,2,3,  11,16,10],
    })

    out = make(df)

    # bull at 3 → zone [top=11, bottom=3]
    # bull at 4 → zone [top=16, bottom=4]
    # دو زون هم‌پوشانی دارند → تست سازگاری NaN pattern
    for i in range(len(df)):
        assert out["fvg_top"].isna()[i] == out["fvg_bottom"].isna()[i]

# ================================================================
# 6) NO LOOKAHEAD BIAS
# ================================================================
def test_detect_fvg_no_lookahead():
    df = pd.DataFrame({
        "open":  [10,11,12,15],
        "high":  [11,12,13,16],
        "low":   [ 9,10,11,14],
        "close": [10,11,12,15],
    })

    out = make(df)
    fup = out["fvg_up"]

    # هیچ وضعیتی نباید قبل از ایجاد زون ظاهر شود
    assert fup.iloc[0] == 0
    assert fup.iloc[1] == 0
    assert fup.iloc[2] == 0  # rule checks at index=2 but shifts → index 3
    assert fup.iloc[3] in (0,1)

# ================================================================
# 7) TOLERANCE FOR NAN INPUTS
# ================================================================
def test_detect_fvg_with_nan_inputs():
    df = pd.DataFrame({
        "open":  [1,2,3,4,5],
        "high":  [2,3,np.nan,6,7],
        "low":   [0,1,2,3,4],
        "close": [1,2,3,4,5],
    })

    out = make(df)

    # هر جایی ورودی NaN باشد هیچ FVG نباید شکل بگیرد
    assert out["fvg_up"].sum() == 0
    assert out["fvg_dn"].sum() == 0
    assert out["fvg_top"].notna().sum() == 0

# ================================================================
# 8) VARIABLE PARAMETERS (lookback/ATR)
# ================================================================
def test_detect_fvg_param_variations():
    df = pd.DataFrame({
        "open":  [10,11,12,20,25,30],
        "high":  [11,12,13,21,26,31],
        "low":   [ 9,10,11,19,24,29],
        "close": [10,11,12,20,25,30],
    })

    out1 = detect_fvg(df["open"], df["high"], df["low"], df["close"],
                      lookback=2, atr_window=3, min_size_pct_of_atr=0.0)

    out2 = detect_fvg(df["open"], df["high"], df["low"], df["close"],
                      lookback=3, atr_window=5, min_size_pct_of_atr=0.5)

    # تغییر پارامتر باید خروجی متفاوت تولید کند
    assert (out1["fvg_up"].values != out2["fvg_up"].values).any()
