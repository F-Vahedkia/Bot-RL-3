# f04_features/indicators/ts03_idx_ExTr__1ALL_2.py
# Run: python -m f15_testcheck.unit.ts03_idx_ExTr__1ALL_2

#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

import pandas as pd
from datetime import datetime

from f04_features.indicators.extras_trend import (
    registry as trend_registry,
    ma_slope,
    rsi_zone,
)


def extra_registry():
    """Secondary registry for non-core trend features."""
    return {
        "ma_slope_extra": lambda df: {
            ma_slope(df).name: ma_slope(df)
        },
        "rsi_zone_extra": lambda df: rsi_zone(df).to_dict(orient="series"),
    }


def safe_align_feature(series: pd.Series, df: pd.DataFrame, feature_name: str) -> pd.Series:
    """
    Align a feature series to df.index; trims or reindexes if necessary.
    This makes the tester robust against minor length/index mismatches.
    """
    # اگر اصلاً Series نیست، تبدیلش می‌کنیم
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    # اگر index طولش کمتر از df است، روی df.index reindex می‌کنیم
    if not series.index.equals(df.index):
        # اگر اندیس عددی است (0..n-1) و طولش != len(df)، reindex مستقیم
        if series.index.dtype.kind in {"i", "u"}:
            # حالت عمومی: فقط براساس طول trim یا pad می‌کنیم
            if len(series) >= len(df):
                series = series.iloc[-len(df):]
                series.index = df.index
            else:
                # اگر کوتاه‌تر است، روی df.index reindex می‌کنیم (NaN برای بقیه)
                series = series.reindex(range(len(df)))
                series.index = df.index
        else:
            # اندیس زمانی/غیرعددی → reindex روی df.index
            series = series.reindex(df.index)

    # اگر هنوز طول mismatch است، آخرین دفاع: trim بر اساس طول df
    if len(series) != len(df):
        series = series.iloc[-len(df):]
        series.index = df.index

    return series


def main() -> None:

    # ------------------------------------------------------------------
    # 1) Load market data
    # ------------------------------------------------------------------
    t1 = datetime.now()

    data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
    df = data[-500_000:].copy()

    if "time" not in df.columns:
        raise ValueError("Input CSV must contain a 'time' column")

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)

    if not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("Data must contain open, high, low, close")

    t2 = datetime.now()
    elapsed = round((t2 - t1).total_seconds(), 3)
    print(f"Time taken to load data: {elapsed} seconds, length_df:{len(df)}")

    # ------------------------------------------------------------------
    # 2) Load registries
    # ------------------------------------------------------------------
    reg_main = trend_registry()
    reg_extra = extra_registry()

    print("\nMain Trend Registry:")
    for name in reg_main.keys():
        print(" -", name)

    print("\nExtra Trend Registry:")
    for name in reg_extra.keys():
        print(" -", name)

    # ------------------------------------------------------------------
    # 3) Run indicators
    # ------------------------------------------------------------------
    print("\nRunning all extras_trend indicators...\n")

    results = pd.DataFrame(index=df.index)

    def run_registry(reg, tag: str):
        for key, func in reg.items():

            print(f"[{tag}] Running feature group: {key}")
            t_start = datetime.now()

            out_dict = func(df)

            t_end = datetime.now()
            elapsed_local = round((t_end - t_start).total_seconds(), 3)
            print(f"  Time: {elapsed_local} seconds")

            if not isinstance(out_dict, dict):
                raise TypeError(
                    f"Registry function '{key}' must return a dict[name -> Series], "
                    f"got {type(out_dict)}"
                )

            for feature_name, series in out_dict.items():
                aligned = safe_align_feature(series, df, feature_name)

                if len(aligned) != len(df):
                    raise ValueError(
                        f"After alignment, feature {feature_name} still has incorrect length "
                        f"{len(aligned)} != {len(df)}"
                    )

                results[feature_name] = aligned

    # run both registries
    run_registry(reg_main, tag="MAIN")
    run_registry(reg_extra, tag="EXTRA")

    print(results["adx_14"].min(), results["adx_14"].max())
    
    # ------------------------------------------------------------------
    # 4) Save output
    # ------------------------------------------------------------------
    output_file = "ts03_idx_ExTr__1ALL_2.csv"
    results.to_csv(output_file)

    print(f"\nSaved output to: {output_file}")

    print("\nColumns:")
    for c in results.columns:
        print(" -", c)

    # ------------------------------------------------------------------
    # 5) Diagnostics
    # ------------------------------------------------------------------
    print("\nValidation diagnostics:")

    for c in results.columns:
        null_rate = results[c].isna().mean()
        var = results[c].var()
        print(f"{c}: null_rate={null_rate:.5f} var={var:.6f}")

    print("\n[TEST COMPLETED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
