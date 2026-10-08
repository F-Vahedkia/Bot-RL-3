# f15_testcheck/unit/ts03_idx_lvl__7compute_adr_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_lvl__7compute_adr_1

import pandas as pd
import numpy as np
from datetime import datetime
from f04_features.indicators.levels import compute_adr

_PATH = "f16_test_results/"
# --- بارگذاری داده واقعی -------------------------------------------
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

# --- ساخت compute_adr -----------------------------------------------
result = compute_adr(df, window=6, tz="UTC")

# --- اجرای هر سه تابع ----------------------------------------------

result.to_csv(f"{_PATH}ts03_idx_lvl__7compute_adr_1.csv")
print("----------------------------------------")
print("Added 1 test result files to f16_test_results:")
print("     ts03_idx_lvl__7compute_adr_1.csv")
print("----------------------------------------")
