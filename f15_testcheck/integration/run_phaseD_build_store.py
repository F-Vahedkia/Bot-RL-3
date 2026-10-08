# -*- coding: utf-8 -*-
"""
Phase D — Build & Save Feature Store (Sample)
- تولید دیتای دقیقه‌ای ~60 روز
- اجرای Specهای چند-تایم‌فریمی (شامل 12 اندیکاتور + 4 اندیکاتور ADV)
- ذخیرهٔ خروجی در Parquet + متادیتا (CSV/JSON)
"""
# روش اجرای برنامه:
# python -m f15_testcheck.integration.run_phaseD_build_store

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from f04_features.OLD.feature_store_old1 import build_and_save_feature_store


def make_minutes_df(n_minutes: int = 60 * 24 * 60, seed: int = 2026) -> pd.DataFrame:
    """ساخت دیتای مصنوعی دقیقه‌ای (UTC) با طول کافی برای warmup روزانه/هفتگی."""
    rs = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_minutes, freq="min", tz="UTC")
    close = pd.Series(np.cumsum(rs.normal(0, 0.2, n_minutes)) + 100.0, index=idx)
    open_ = close.shift(1).fillna(close)
    high = np.maximum(open_, close) + rs.random(n_minutes) * 0.2
    low = np.minimum(open_, close) - rs.random(n_minutes) * 0.2
    vol = pd.Series(rs.integers(100, 1000, n_minutes), index=idx)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def main():
    df = make_minutes_df()

    # === Specها (نمونهٔ بزرگ 12×8 + 4 ADV) ===
    core12 = [
        "sma(close,20)", "ema(close,50)", "rsi(14)", "macd(12,26,9)",
        "atr(14)", "bbands(20,2)", "cci(20)", "mfi(14)",
        "stoch(14,3,3)", "wma(close,20)", "wr(14)", "sar(0.02,0.2)",
    ]
    adv4 = [
        "adr(window=14)", "adr_distance_to_open(window=14)",
        "sr_overlap_score(anchor=100,step=5,n=25,tol_pct=0.02)",
        "round_levels(anchor=100,step=5,n=25)",
    ]
    tfs8 = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]

    specs = [f"{s}@{tf}" for s in (core12 + adv4) for tf in tfs8]

    out_paths = build_and_save_feature_store(
        df=df,
        specs=specs,
        out_dir=Path("./f15_testcheck/_reports/feature_store"),
        base_name="SYNTH_2022Q1_12x8_ADV",
        data_format="parquet",
        compression="snappy",
    )
    print("Feature store artifacts:", out_paths)


if __name__ == "__main__":
    main()
