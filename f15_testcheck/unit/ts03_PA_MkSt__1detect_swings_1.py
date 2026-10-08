# f15_testcheck/unit/ts03_PA_MkSt__1detect_swings_1.py
# Run: python -m f15_testcheck.unit.ts03_PA_MkSt__1detect_swings_1

import pandas as pd
# import numpy as np

from f04_features.price_action.market_structure import detect_swings
from f04_features.indicators.zigzag import zigzag_wrapper as zigzag

_PATH = "f16_test_results/"
# ---------------------------------------------------
# Load data
# ---------------------------------------------------
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

if not {"high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain high, low, close")

df.to_csv(f"{_PATH}ts03_PA_MkSt__1detect_swings_1_data.csv")

# ---------------------------------------------------
# Generate swings from detect_swings
# ---------------------------------------------------
swings = detect_swings(
    df,
    depth=12,
    deviation=5.0,
    backstep=10,
    point=0.01,
)

zzg = zigzag(
    high=df["high"],
    low=df["low"],
    depth=12,
    deviation=5.0,
    backstep=10,
    point=0.01,
    addmeta=False,
    final_check=True,
)

# ---------------------------------------------------
# Validation
# ---------------------------------------------------
fault_counter = 0

# 1) swing_high consistency
mask_high_mismatch = swings["swing_high"] != (zzg["state"] == 1)
fault_counter += mask_high_mismatch.sum()

# 2) swing_low consistency
mask_low_mismatch = swings["swing_low"] != (zzg["state"] == -1)
fault_counter += mask_low_mismatch.sum()

# 3) swing_price consistency for highs
high_price_mismatch = (
    swings["swing_high"]
    & (swings["swing_price"] != zzg["high"])
)
fault_counter += high_price_mismatch.sum()

# 4) swing_price consistency for lows
low_price_mismatch = (
    swings["swing_low"]
    & (swings["swing_price"] != zzg["low"])
)
fault_counter += low_price_mismatch.sum()

# 5) Non-pivot rows must be NaN
non_pivot_mismatch = (
    ~(swings["swing_high"] | swings["swing_low"])
    & swings["swing_price"].notna()
)
fault_counter += non_pivot_mismatch.sum()

print(f"fault_counter = {fault_counter}")

pd.concat([swings, zzg], axis=1).to_csv(f"{_PATH}ts03_PA_MkSt__1detect_swings_1.csv")

print("--------------------------------------------------")
print("Added 2 test result files to f16_test_results:")
print("✅   ts03_PA_MkSt__1detect_swings_1_data.csv")
print("✅   ts03_PA_MkSt__1detect_swings_1.csv")
print("--------------------------------------------------")
