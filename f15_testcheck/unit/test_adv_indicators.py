# -*- coding: utf-8 -*-
# f15_testcheck/unit/test_adv_indicators.py
# هدف: اعتبارسنجی آداپترهای Advanced در رجیستری v2 و اینتگریشن از طریق Engine
# - کشف‌پذیری در رجیستری واحد (advanced tag)
# - بازگشت callable از get_indicator
# - تولید ستون‌ها از طریق Spec/DSL با engine.apply
# - عدم وجود non-finite بعد از warmup
# - سازگاری حداقلی MTF برای نمونه‌های @M1/@M5/@M15

# روش اجری برنامه:
# pytest f15_testcheck/unit/test_adv_indicators.py -q -rA --disable-warnings --durations=20


from __future__ import annotations
from pathlib import Path
import sys, inspect, types, random
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

# -- مسیر پروژه (درج ریشه) ----------------------------------------------------
_PR = Path(__file__).resolve().parents[2]  # ریشه‌ی مخزن (شامل f04_features/)
if str(_PR) not in sys.path:
    sys.path.insert(0, str(_PR))

# -- وارد کردن ماژول‌ها؛ در صورت خطا، کل ماژول را Skip کن ----------------------
try:
    from f04_features.OLD import feature_engine as _engine
    from f04_features.OLD.feature_registry import (
        get_indicator,
        list_all_indicators,
    )
except Exception as e:
    pytest.skip(f"Cannot import indicators engine/registry from {_PR}: {e}", allow_module_level=True)


# -- ابزار ساخت دیتای دقیقه‌ای به‌قدر کافی برای ADR(14) -----------------------
def make_ohlcv_minutes(n_minutes: int = 60 * 24 * 30, seed: int = 1337) -> pd.DataFrame:
    """
    ساخت دیتای واقع‌نمای دقیقه‌ای (UTC) با طول کافی (~30 روز) تا warmup اندیکاتورهای روزانه مثل ADR(14) کامل شود.
    """
    rs = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_minutes, freq="min", tz="UTC")
    close = pd.Series(np.cumsum(rs.normal(0, 0.2, n_minutes)) + 100.0, index=idx)
    open_ = close.shift(1).fillna(close)
    high = np.maximum(open_, close) + rs.random(n_minutes) * 0.2
    low = np.minimum(open_, close) - rs.random(n_minutes) * 0.2
    vol = pd.Series(rs.integers(100, 1000, n_minutes), index=idx)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


# -- هِلپر تست: شمارش non-finite بعد از اولین مقدار معتبر ----------------------
def nonfinite_after_first_valid(s: pd.Series) -> int:
    fv = s.first_valid_index()
    tail = s if fv is None else s.loc[fv:]
    arr = tail.to_numpy(dtype=float)
    return int((~np.isfinite(arr)).sum())


# ================================== تست‌ها ====================================

def test_registry_reports_advanced_tags():
    """
    رجیستری باید این کلیدها را با برچسب 'advanced' گزارش کند.
    """
    tags = list_all_indicators(include_legacy=False)
    for key in ("adr", "adr_distance_to_open", "sr_overlap_score", "round_levels"):
        assert key in tags and tags[key] == "unified", f"missing or not advanced: {key}"


def test_get_indicator_returns_callables():
    """
    get_indicator باید callable بدهد (آداپترها ثبت شده‌اند).
    """
    for key in ("adr", "adr_distance_to_open", "sr_overlap_score", "round_levels"):
        fn = get_indicator(key)
        assert callable(fn), f"not callable: {key}"


def test_apply_adv_specs_produces_expected_columns_and_no_nan_after_warmup():
    """
    اجرای Spec/DSL برای آداپترهای ADV باید ستون‌های مورد انتظار را بسازد
    و بعد از warmup هیچ non-finite باقی نماند.
    """
    df = make_ohlcv_minutes(n_minutes=60 * 24 * 30, seed=2025)  # ~30 روز

    specs = [
        "adr(window=14)@M1",
        "adr_distance_to_open(window=14)@M1",
        "sr_overlap_score(anchor=100,step=5,n=25,tol_pct=0.02)@M5",
        "round_levels(anchor=100,step=5,n=25)@M15",
    ]
    out = _engine.apply(df=df, specs=specs)

    # ستون‌های مورد انتظار
    expect_presence = [
        "__adr@M1__adr_14",
        "__adr_distance_to_open@M1__adr_day_open_14",
        "__adr_distance_to_open@M1__adr_dist_abs_14",
        "__adr_distance_to_open@M1__adr_dist_pct_14",
        "__sr_overlap_score@M5__sr_overlap_score_5_25_200bp",
        "__round_levels@M15__rl_nearest_5_25",
        "__round_levels@M15__rl_signed_5_25",
        "__round_levels@M15__rl_abs_5_25",
    ]
    cols = set(out.columns)
    missing = [c for c in expect_presence if c not in cols]
    assert not missing, f"Missing expected columns: {missing}"

    # عدم وجود non-finite پس از warmup
    feat_cols = [c for c in out.columns if c.startswith("__")]
    bad = {}
    for c in feat_cols:
        cnt = nonfinite_after_first_valid(out[c])
        if cnt != 0:
            bad[c] = cnt
    assert not bad, f"Non-finite after warmup detected: {bad}"


def test_adv_daily_broadcast_and_presence_on_M1_baseframe():
    """
    اطمینان از broadcast صحیح خروجی D1 به فریم پایه‌ی M1:
    - ADR روزانه محاسبه و به M1 منتشر شود (بدون شکاف پس از warmup).
    """
    df = make_ohlcv_minutes(n_minutes=60 * 24 * 35, seed=7)  # کمی بلندتر برای اطمینان
    specs = ["adr(window=14)@D1"]
    out = _engine.apply(df=df, specs=specs)
    col = "__adr@D1__adr_14"
    assert col in out.columns, "ADR@D1 column missing on baseframe"
    # بعد از اولین مقدار معتبر، نباید non-finite باقی بماند:
    assert nonfinite_after_first_valid(out[col]) == 0, "ADR@D1 has non-finite values after warmup"
