# f04_features/indicators/ts03_idx_ExTr__1ALL_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_ExTr__1ALL_1

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
        "ma_slope": lambda df: {
            ma_slope(df).name: ma_slope(df)
        },
        "rsi_zone": lambda df: rsi_zone(df).to_dict(orient="series"),
    }


def main() -> None:

    # ------------------------------------------------------------------
    # 1) Load market data
    # ------------------------------------------------------------------
    t1 = datetime.now()

    data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
    df = data[-12_000:].copy()

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

    def run_registry(reg):
        for key, func in reg.items():

            print(f"Running feature group: {key}")
            t_start = datetime.now()

            out_dict = func(df)

            t_end = datetime.now()
            elapsed = round((t_end - t_start).total_seconds(), 3)
            print(f"  Time: {elapsed} seconds")

            for feature_name, series in out_dict.items():

                if len(series) != len(df):
                    raise ValueError(
                        f"Feature {feature_name} produced incorrect length "
                        f"{len(series)} != {len(df)}"
                    )

                results[feature_name] = series

    # run both registries
    run_registry(reg_main)
    run_registry(reg_extra)

    # ------------------------------------------------------------------
    # 4) Save output
    # ------------------------------------------------------------------
    output_file = "ts03_idx_ExTr__1ALL_1.csv"
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
