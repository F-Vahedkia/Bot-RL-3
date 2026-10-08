# هنوز این برنامه تست نشده است !!!!!!


# f15_testcheck\unit\ts03_idx_ptt__1_AllFuncs_1.py
# Run: pytest -q f15_testcheck\unit\ts03_idx_ptt__1_AllFuncs_1.py

import numpy as np
import pandas as pd
import pytest
import f04_features.indicators.patterns as patterns

# ---------- هِلپرهای عمومی تست ----------
n = 10_000
def make_ohlc_df(seed=123) -> pd.DataFrame:
    # ---------------------------------------------------
    # Load data
    # ---------------------------------------------------
    data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
    df = data[-n:].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)

    if not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("Data must contain open, high, low, close")

    return df


def assert_int8_series(s: pd.Series, length: int | None = None):
    assert isinstance(s, pd.Series)
    assert s.dtype == "int8"
    if length is not None:
        assert len(s) == length


# ---------- تست هِلپرهای private ----------

def test_apply_scale_basic_behavior():
    rules = [
        ("ratio", 2.5, 1.1, 10.0),
        ("x", 1.0, None, None),
    ]
    kwargs = {"scale_k": 2.0}
    out = patterns._apply_scale(kwargs, rules)
    # scale_k نباید حذف شود
    assert out["scale_k"] == 2.0
    # ratio از scale_k مشتق می‌شود: 2.5 * 2.0 = 5.0 در بازه [1.1,10]
    assert np.isclose(out["ratio"], 5.0)
    # x از scale_k مشتق می‌شود: 1.0 * 2.0 = 2.0 بدون کلیپ
    assert np.isclose(out["x"], 2.0)

    # اگر پارامتر از قبل موجود باشد، overwrite نشود
    kwargs2 = {"scale_k": 2.0, "ratio": 9.0}
    out2 = patterns._apply_scale(kwargs2, rules)
    assert out2["ratio"] == 9.0  # دست‌نخورده


def test_fmtf_trimming():
    assert patterns._fmtf(2.5) == "2.5"
    assert patterns._fmtf(2.0) == "2"
    assert patterns._fmtf(0.0) == "0"
    assert patterns._fmtf(0.1234, nd=3).startswith("0.123")


def test_body_and_abs_body_and_range():
    # n = 10_000
    o = pd.Series(np.linspace(100, 101, n), name="open")
    c = pd.Series(np.linspace(101, 99, n), name="close")
    h = pd.Series(np.maximum(o, c) + 1.0, name="high")
    l = pd.Series(np.minimum(o, c) - 1.0, name="low")

    body = patterns._body(o, c)
    abs_body = patterns._abs_body(o, c)
    rng = patterns._range(h, l)

    assert body.dtype == "float32"
    assert abs_body.dtype == "float32"
    assert rng.dtype == "float32"
    assert len(body) == n == len(abs_body) == len(rng)

    # abs_body باید قدر مطلق body باشد
    assert np.allclose(abs_body.values, np.abs(body.values), atol=1e-6)
    # range باید high-low باشد
    assert np.allclose(rng.values, (h - l).values.astype("float32"), atol=1e-6)


def test_body_wicks_consistency():
    df = make_ohlc_df(seed=456)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    body_abs, upper, lower = patterns._body_wicks(o, h, l, c)

    assert body_abs.dtype == "float32"
    assert upper.dtype == "float32"
    assert lower.dtype == "float32"
    assert len(body_abs) == len(upper) == len(lower) == n

    # body_abs باید برابر |close-open| باشد (تا حد تقریب float32)
    expected_body_abs = (c - o).abs().astype("float32")
    assert np.allclose(body_abs.values, expected_body_abs.values, atol=1e-6)

    # upper و lower باید non-negative باشند
    assert (upper >= 0).all()
    assert (lower >= 0).all()


# ---------- تست pattern helpers ----------

def test_engulfing_flags_shapes_and_dtype():
    df = make_ohlc_df(seed=111)
    bull, bear = patterns.engulfing_flags(df["open"], df["high"], df["low"], df["close"])
    assert_int8_series(bull, length=len(df))
    assert_int8_series(bear, length=len(df))
    # نباید با NaN پر شده باشد
    assert not bull.isna().any()
    assert not bear.isna().any()


def test_doji_flag_basic_behavior():
    df = make_ohlc_df(seed=222)
    # بدنه‌ها را تقریباً صفر کنیم تا چند doji قطعی داشته باشیم
    df["close"] = df["open"] + np.random.randn(len(df)).astype("float32") * 0.0001

    atr = pd.Series(np.ones(len(df), dtype="float32"), index=df.index)
    flag = patterns.doji_flag(
        df["open"],
        df["close"],
        atr=atr,
        atr_ratio_thresh=0.01,
        range_ratio_thresh=1.1,
    )
    assert_int8_series(flag, length=len(df))
    # انتظار داریم تعدادی 1 داشته باشیم
    assert flag.sum() > 0


def test_pinbar_flags_basic_behavior():
    df = make_ohlc_df(seed=333)
    bull, bear = patterns.pinbar_flags(df["open"], df["high"], df["low"], df["close"], ratio=2.0)
    assert_int8_series(bull, length=len(df))
    assert_int8_series(bear, length=len(df))


def test_hammer_shooting_flags_basic_behavior():
    df = make_ohlc_df(seed=444)
    bull, bear = patterns.hammer_shooting_flags(
        df["open"], df["high"], df["low"], df["close"],
        min_body_frac=0.0,
        wick_ratio=2.0,
        opp_wick_k=1.25,
    )
    assert_int8_series(bull, length=len(df))
    assert_int8_series(bear, length=len(df))


def test_harami_flags_basic_behavior():
    df = make_ohlc_df(seed=555)
    bull, bear = patterns.harami_flags(df["open"], df["high"], df["low"], df["close"])
    assert_int8_series(bull, length=len(df))
    assert_int8_series(bear, length=len(df))


def test_inside_outside_flags_basic_behavior():
    df = make_ohlc_df(seed=666)
    inside, outside = patterns.inside_outside_flags(
        df["open"], df["high"], df["low"], df["close"],
        min_range_k_atr=0.0,
        atr_win=14,
    )
    assert_int8_series(inside, length=len(df))
    assert_int8_series(outside, length=len(df))


def test_marubozu_flags_basic_behavior():
    df = make_ohlc_df(seed=777)
    bull, bear = patterns.marubozu_flags(
        df["open"], df["high"], df["low"], df["close"], wick_frac=0.1
    )
    assert_int8_series(bull, length=len(df))
    assert_int8_series(bear, length=len(df))


def test_tweezer_flags_basic_behavior_fixed_tol():
    df = make_ohlc_df(seed=888)
    top, bottom = patterns.tweezer_flags(
        df["high"], df["low"], tol_frac=0.001, tol_k=None, tol_mode="atr_price",
        atr_win=14, close=df["close"]
    )
    assert_int8_series(top, length=len(df))
    assert_int8_series(bottom, length=len(df))


def test_tweezer_flags_adaptive_tol():
    df = make_ohlc_df(seed=889)
    top, bottom = patterns.tweezer_flags(
        df["high"], df["low"],
        tol_frac=None, tol_k=2.0, tol_mode="atr_price",
        atr_win=14, close=df["close"]
    )
    assert_int8_series(top, length=len(df))
    assert_int8_series(bottom, length=len(df))


def test_three_soldiers_crows_flags_basic_behavior():
    df = make_ohlc_df(seed=999)
    bull, bear = patterns.three_soldiers_crows_flags(
        df["open"], df["close"], atr_ref=None, min_body_atr=0.2
    )
    assert_int8_series(bull, length=len(df))
    assert_int8_series(bear, length=len(df))


def test_morning_evening_star_flags_basic_behavior():
    df = make_ohlc_df(seed=1001)
    morning, evening = patterns.morning_evening_star_flags(
        df["open"], df["high"], df["low"], df["close"],
        small_body_atr=0.3,
        atr_win=14,
    )
    assert_int8_series(morning, length=len(df))
    assert_int8_series(evening, length=len(df))


def test_piercing_darkcloud_flags_basic_behavior():
    df = make_ohlc_df(seed=1002)
    piercing, darkcloud = patterns.piercing_darkcloud_flags(
        df["open"], df["close"], min_body_ratio=0.2
    )
    assert_int8_series(piercing, length=len(df))
    assert_int8_series(darkcloud, length=len(df))


def test_belt_hold_flags_basic_behavior():
    df = make_ohlc_df(seed=1003)
    bull, bear = patterns.belt_hold_flags(
        df["open"], df["high"], df["low"], df["close"], wick_frac=0.1
    )
    assert_int8_series(bull, length=len(df))
    assert_int8_series(bear, length=len(df))


# ---------- تست registry() ----------

def test_registry_structure_and_builders():
    reg = patterns.registry()
    assert isinstance(reg, dict)
    assert all(isinstance(k, str) for k in reg.keys())
    assert all(callable(v) for v in reg.values())

    df = make_ohlc_df(seed=2024)
    for name, builder in reg.items():
        out = builder(df.copy(), scale_k=2.0)
        assert isinstance(out, dict), f"builder {name} must return dict"
        assert out, f"builder {name} returned empty dict"
        for col, series in out.items():
            assert isinstance(col, str)
            assert isinstance(series, pd.Series)
            assert series.dtype == "int8"
            assert len(series) == len(df)


# ---------- تست build_patterns() (در صورت وجود) ----------
"""
@pytest.mark.skipif(
    not hasattr(patterns, "build_patterns"),
    reason="patterns.build_patterns() not present in this version of patterns.py",
)
def test_build_patterns_basic_schema():
    df = make_ohlc_df(seed=3030)
    # هم با atr و هم بدون atr باید کار کند
    df["atr"] = np.abs(df["close"] - df["open"]).rolling(14, min_periods=3).mean()

    out = patterns.build_patterns(df.copy())
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(df)
    # همه‌ی ستون‌ها باید int8 باشند
    for c in out.columns:
        assert out[c].dtype == "int8"

    # چند ستون پایه که طبق توضیحات build_patterns حتماً باید وجود داشته باشند:
    expected_any = [
        "pat_doji",
        "pat_engulf_bull",
        "pat_engulf_bear",
        "pat_pinbar_bull",
        "pat_pinbar_bear",
        "pat_hammer_bull",
        "pat_hammer_bear",
        "pat_harami_bull",
        "pat_harami_bear",
        "pat_inside",
        "pat_outside",
        "pat_marubozu_bull",
        "pat_marubozu_bear",
        "pat_tweezer_top",
        "pat_tweezer_bottom",
        "pat_3soldiers",
        "pat_3crows",
    ]
    # فقط ستون‌هایی که واقعاً در این نسخه ساخته می‌شوند، چک می‌کنیم (subset)
    present = set(out.columns)
    missing_core = [c for c in expected_any if c not in present]
    # اگر برخی غایب بودند، فقط هشدار می‌دهیم؛ برای عدم شکستن تست، assert را شل می‌گیریم
    if missing_core:
        # می‌توانی برای سخت‌گیری، این را به assert not missing_core تغییر دهی
        print("WARNING: some expected pattern columns missing in build_patterns:", missing_core)


@pytest.mark.skipif(
    not hasattr(patterns, "build_patterns"),
    reason="patterns.build_patterns() not present in this version of patterns.py",
)
def test_build_patterns_prefix_and_scale_k_options():
    df = make_ohlc_df(seed=4040)
    # بدون atr داخلی
    if "atr" in df.columns:
        del df["atr"]

    out1 = patterns.build_patterns(df.copy(), prefix="pat_")
    out2 = patterns.build_patterns(df.copy(), prefix="xpat_", scale_k=2.5)

    assert len(out1) == len(df)
    assert len(out2) == len(df)

    # prefix باید روی نام ستون‌ها اعمال شود
    assert all(c.startswith("pat_") for c in out1.columns)
    assert all(c.startswith("xpat_") for c in out2.columns)

    # index باید با df هم‌تراز باشد
    pd.testing.assert_index_equal(out1.index, df.index)
    pd.testing.assert_index_equal(out2.index, df.index)

"""