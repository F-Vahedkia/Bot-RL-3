# f15_testcheck/unit/ts03_idx_ExTr__5EMA_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_ExTr__5EMA_1

import numpy as np
import pandas as pd
from datetime import datetime
from f04_features.indicators.extras_trend import dema, tema, hma

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


# --- Calling functions --------------------------------------------- TEST-1
funcs = {
    "dema": dema,
    "tema": tema,
    "hma": hma,
}
# --- list of iterations -----------------------------
candle_nums = [10_000, 30_000, 100_000, 300_000, 1_000_000, 5_000_000]
# candle_nums = []
results_df = None  # برای ذخیره خروجی نهایی در حالت multi-Series
df_for_csv = None       # برای ذخیره داده sliced مربوط به n = 10_000

for n in candle_nums:
    # --- slicing data ----------------
    df = data[- n:].copy()

    # --- Calling TR functions --------
    print("\n")
    for name, func in funcs.items():
        t1 = datetime.now()
        res = func(df["close"], n = 10)
        t2 = datetime.now()

        elapsed = round((t2 - t1).total_seconds(), 3)
        print(f"Time taken to run {name}: {elapsed} seconds, length_df:{len(df)}")

        if n == 10_000:
            if results_df is None:
                results_df = pd.DataFrame(index=df.index)
                df_for_csv = df.copy()

            results_df[f"{name}"] = res.astype("float64")
            
# --- Save to CSV ---------------------------------------------------
if results_df is not None:
    final_df = pd.concat([df_for_csv["close"], results_df], axis=1)
    final_df.to_csv("ts03_idx_ExTr__5EMA_1.csv")
    print("Saved results for n=10000 to ts03_idx_ExTr__5EMA_1.csv")


##################################################################### TEST-2
# df = data[- 10_000:].copy()
# chang, volat, ef, sc, out = kama_orig(df["close"], n = 10, fast=2, slow=30)
# pd.concat([df["close"], chang, volat, ef, sc, out], axis=1).to_csv("ts03_idx_ExTr__4kama_2.csv")

