# f15_testcheck/unit/ts03_idx_lvl__8adr_distance_to_open_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_lvl__8adr_distance_to_open_1

import pandas as pd
import numpy as np
from datetime import datetime
from f04_features.indicators.levels import compute_adr, adr_distance_to_open

_PATH = "f16_test_results/"
# --- بارگذاری داده واقعی -------------------------------------------
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)
df.to_csv(f"{_PATH}ts03_idx_lvl__8adr_distance_to_open_1_data.csv")

# --- ساخت compute_adr -----------------------------------------------
adr = compute_adr(df, window=6, tz="UTC")
adr_dist = adr_distance_to_open(df, adr=adr, tz="UTC")

# --- ذخیره جواب ها--------------------------------------------------
adr.to_csv(f"{_PATH}ts03_idx_lvl__8adr_distance_to_open_1_adr.csv")
adr_dist.to_csv(f"{_PATH}ts03_idx_lvl__8adr_distance_to_open_1.csv")
print("----------------------------------------")
print("Added 8 test result files to f16_test_results:")
print("✅   ts03_idx_lvl__8adr_distance_to_open_1_data.csv")
print("✅   ts03_idx_lvl__8adr_distance_to_open_1_adr.csv")
print("✅   ts03_idx_lvl__8adr_distance_to_open_1.csv")
print("----------------------------------------")
