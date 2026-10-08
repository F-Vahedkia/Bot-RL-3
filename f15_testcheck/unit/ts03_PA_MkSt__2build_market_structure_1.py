# f15_testcheck/unit/ts03_PA_MkSt__2build_market_structure_1.py
# Run: python -m f15_testcheck.unit.ts03_PA_MkSt__2build_market_structure_1

import pandas as pd
# import numpy as np

from f04_features.price_action.market_structure import (
    build_market_structure,
    detect_swings,
)

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

# ---------------------------------------------------
# Generate structure
# ---------------------------------------------------
structure = build_market_structure(
    df,
    depth=12,
    deviation=5.0,
    backstep=10,
    point=0.01,
)

swings = detect_swings(
    df,
    depth=12,
    deviation=5.0,
    backstep=10,
    point=0.01,
)

# ---------------------------------------------------
# Validation
# ---------------------------------------------------
fault_counter = 0

# 1) Structure columns must exist
required_cols = ["HH", "HL", "LH", "LL"]
for col in required_cols:
    if col not in structure.columns:
        raise ValueError(f"{col} column missing")

# 2) Structure flags only allowed on pivot rows
non_pivot_mask = ~(structure["swing_high"] | structure["swing_low"])
for col in required_cols:
    invalid_flag = non_pivot_mask & (structure[col] == True)
    fault_counter += invalid_flag.sum()

# 3) Only one of HH/HL/LH/LL allowed per row
multi_flag = (
    structure[required_cols].sum(axis=1) > 1
)
fault_counter += multi_flag.sum()

# 4) Logical consistency check (local validation)
pivots = structure[structure["swing_high"] | structure["swing_low"]]

prev_price = None
prev_type = None

for idx, row in pivots.iterrows():

    current_price = row["swing_price"]
    current_type = "high" if row["swing_high"] else "low"

    if prev_price is not None:

        if current_type == "high" and prev_type == "high":
            if current_price > prev_price and not row["HH"]:
                fault_counter += 1
            if current_price < prev_price and not row["LH"]:
                fault_counter += 1

        elif current_type == "low" and prev_type == "low":
            if current_price > prev_price and not row["HL"]:
                fault_counter += 1
            if current_price < prev_price and not row["LL"]:
                fault_counter += 1

    prev_price = current_price
    prev_type = current_type

print(f"fault_counter = {fault_counter}")

structure.to_csv(f"{_PATH}ts03_PA_MkSt__2build_market_structure_1.csv")
print("--------------------------------------------------")
print("Added 1 test result files to f16_test_results:")
print("✅   ts03_PA_MkSt__2build_market_structure_1.csv")
print("--------------------------------------------------")
