# f15_testcheck/unit/ts03_idx_utl__13detect_swings_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_utl__13detect_swings_1

import numpy as np
import pandas as pd
from datetime import datetime
from f04_features.indicators.utils import (
    detect_swings,
)

_PATH = "f16_test_results/"
_PATH = ""

# --- Load data --------------------------------------
t1 = datetime.now()
data_ = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
t2 = datetime.now()

total_candles = len(data_)
elapsed = round((t2 - t1).total_seconds(), 4)
print(f"Time taken to load data: {elapsed} seconds, length_df:{total_candles}")

# --- preparing data ------------------
data = data_[- total_candles:].copy()
data["time"] = pd.to_datetime(data["time"], utc=True)
data.set_index("time", inplace=True)

# --- Dict of functions ------------------------------
tr_funcs = {
    "detect_swings": detect_swings,
}

# --- list of iterations -----------------------------
# candle_nums = [5_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 5_000_000]
candle_nums = [5_000]

# --- دیتافریم برای ذخیره همه نتایج ----------------
results = pd.DataFrame()

for n in candle_nums:
    # --- slicing data ----------------
    df = data[- n:].copy()

    # --- Calling TR functions --------
    for name, func in tr_funcs.items():
        t1 = datetime.now()
        result = func(df=df, depth=12, deviation=0.05, backstep=5)
        t2 = datetime.now()

        elapsed = round((t2 - t1).total_seconds(), 3)
        print(f"Time taken to run {name}: {elapsed} seconds, length_df:{len(df)}")

        if n==5_000:
            results = result
    print("\n")


# --- Save results --------------------------------------------------
results.to_csv("ts03_idx_utl__13detect_swings_1.csv")
