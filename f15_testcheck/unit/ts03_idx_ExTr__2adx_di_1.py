# f15_testcheck/unit/ts03_idx_ExTr__2adx_di_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_ExTr__2adx_di_1

from datetime import datetime
import pandas as pd

from f04_features.indicators.extras_trend import adx_di

#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

# --- Load data -----------------------------------------------------
t1 = datetime.now()
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
t2 = datetime.now()

# --- Preparing data ------------------------------------------------
data["time"] = pd.to_datetime(data["time"], utc=True)
data.set_index("time", inplace=True)
data = data.astype("float64")

elapsed = round((t2 - t1).total_seconds(), 4)
print(f"Time taken to load data: {elapsed} seconds, length_df:{len(data)}")

if not {"open", "high", "low", "close"}.issubset(data.columns):
    raise ValueError("Data must contain open, high, low, close")


# --- Calling functions ---------------------------------------------
funcs = {
    "adx_di": adx_di,
}
# --- list of iterations -----------------------------
# candle_nums = [10_000, 30_000, 100_000, 300_000, 1_000_000, 5_000_000]
candle_nums = [1_000]
results_df = None  # برای ذخیره خروجی نهایی در حالت multi-Series
result = None  # برای ذخیره خروجی نهایی در حالت DataFrame

for n in candle_nums:
    # --- slicing data ----------------
    df = data[- n:].copy()

    # --- Calling TR functions --------
    print("\n")
    for name, func in funcs.items():
        t1 = datetime.now()
        pdi, mdi, adx, adxr = func(df["high"], df["low"], df["close"], 14)
        # result = func(df["high"], df["low"], df["close"], 14)
        t2 = datetime.now()

        elapsed = round((t2 - t1).total_seconds(), 3)
        print(f"Time taken to run {name}: {elapsed} seconds, length_df:{len(df)}")

        if n == 1_000:
            if results_df is None:
                # create DataFrame
                results_df = pd.DataFrame(index=df.index)

            # add columns
            results_df[f"{name}_pdi"] = pdi.astype("float64")
            results_df[f"{name}_mdi"] = mdi.astype("float64")
            results_df[f"{name}_adx"] = adx.astype("float64")
            results_df[f"{name}_adxr"] = adxr.astype("float64")
            

# --- Save to CSV --------------------------------------------------
if results_df is not None:
    pd.concat(
        [df[["open", "high", "low", "close"]], results_df, result],
          axis=1
    ).to_csv("ts03_idx_ExTr__2adx_di_1.csv")

    print("Saved results for n=1000 to ts03_idx_ExTr__2adx_di_1.csv")

else:
    print("Error: no results were generated for n=1000")

