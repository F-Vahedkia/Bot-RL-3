# f15_testcheck/unit/ts03_PA_MkSt__3detect_bos_choch_1.py
# Run: python -m f15_testcheck.unit.ts03_PA_MkSt__3detect_bos_choch_1

import pandas as pd
from f04_features.price_action.market_structure import (
    build_market_structure,
    detect_bos_choch,
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
# Build structure first
# ---------------------------------------------------
structure = build_market_structure(
    df,
    depth=12,
    deviation=5.0,
    backstep=10,
    point=0.01,
)

# ---------------------------------------------------
# Detect BOS / CHOCH
# ---------------------------------------------------
bos = detect_bos_choch(structure, df)

# ---------------------------------------------------
# Validation for price-break version
# ---------------------------------------------------
fault_counter = 0
required_cols = ["bos_up", "bos_down", "choch_up", "choch_down"]
for col in required_cols:
    if col not in bos.columns:
        raise ValueError(f"{col} column missing")

# 1) BOS/CHOCH must only appear on structure rows
structure_mask = bos[["HH", "HL", "LH", "LL"]].any(axis=1)
for col in required_cols:
    invalid_flag = (~structure_mask) & (bos[col] == 1)
    fault_counter += invalid_flag.sum()

# 2) No simultaneous opposite signals
opposite_conflict = (
    (bos["bos_up"] == 1) & (bos["bos_down"] == 1)
) | (
    (bos["choch_up"] == 1) & (bos["choch_down"] == 1)
)
fault_counter += opposite_conflict.sum()

# 3) Logical replay: check price-break regime consistency
last_regime = 0  # 1 bullish, -1 bearish
last_swing_high = None
last_swing_low = None

for idx, row in bos.iterrows():
    # Update last swing levels
    if row.get("swing_high", False):
        last_swing_high = row.get("swing_price")
    if row.get("swing_low", False):
        last_swing_low = row.get("swing_price")

    # Check bullish signals
    if row["bos_up"] == 1 or row["choch_up"] == 1:
        if last_regime == 1 and row["choch_up"] == 1:
            fault_counter += 1
        last_regime = 1

    # Check bearish signals
    if row["bos_down"] == 1 or row["choch_down"] == 1:
        if last_regime == -1 and row["choch_down"] == 1:
            fault_counter += 1
        last_regime = -1

print(f"fault_counter = {fault_counter}")


# ---------------------------------------------------
# Second Part of Test File
# ---------------------------------------------------
df_bos = bos.copy()

# Pivot masks
high_mask = df_bos["swing_high"]
low_mask = df_bos["swing_low"]

# ستون‌های BOS/CHOCH
cols = ["bos_up", "bos_down", "choch_up", "choch_down"]

# نتایج
results = {}

for col in cols:
    count_total = df_bos[col].sum()
    
    # بررسی اینکه قبل از هر 1 حداقل یک swing قبلی وجود داشته باشد
    count_with_swing = 0
    last_high = False
    last_low = False
    
    for _, row in df_bos.iterrows():
        if row["swing_high"]:
            last_high = True
        if row["swing_low"]:
            last_low = True
        
        if row[col] == 1:
            if col in ["bos_up", "choch_up"] and last_high:
                count_with_swing += 1
            elif col in ["bos_down", "choch_down"] and last_low:
                count_with_swing += 1
    
    results[col] = (count_total, count_with_swing)

# نمایش نتیجه
for col, (total, valid) in results.items():
    print(f"{col}: total={total}, valid_with_swing={valid}")

bos.to_csv(f"{_PATH}ts03_PA_MkSt__3detect_bos_choch_1.csv")
print("--------------------------------------------------")
print("Added 1 test result files to f16_test_results:")
print("✅   ts03_PA_MkSt__3detect_bos_choch_1.csv")
print("--------------------------------------------------")
