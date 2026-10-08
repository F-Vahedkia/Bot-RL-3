# f04_features/indicators/ts03_idx_div__3div_value_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_div__3div_value_1

#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

import pandas as pd
from datetime import datetime

from f04_features.indicators.divergences import (
    registry,
)

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

t1 = datetime.now()
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-500_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)
t2 = datetime.now()

if not {"open", "high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain open, high, low, close")

elapsed = round((t2 - t1).total_seconds(), 3)
print(f"Time taken to load data: {elapsed} seconds, length_df:{len(df)}")

# -------------------------------------------------------------------
# Load Registry
# -------------------------------------------------------------------

registry = registry()

print("Registry loaded:")
for name in registry.keys():
    print(f" - {name}")

# -------------------------------------------------------------------
# Run all registry features
# -------------------------------------------------------------------

results = pd.DataFrame(index=df.index)

for key, func in registry.items():
    print(f"Running registry feature: {key}")

    t_start = datetime.now()
    out_dict = func(df)    # ← نکته مهم: registry maker ها ورودی‌شان df کامل است
    t_end = datetime.now()

    elapsed = round((t_end - t_start).total_seconds(), 3)
    print(f"  Time: {elapsed} seconds")

    # out_dict شامل ۲ سری است (bull / bear)
    for feature_name, series in out_dict.items():
        results[feature_name] = series

# -------------------------------------------------------------------
# Save Output
# -------------------------------------------------------------------

output_file = "ts03_idx_div__3div_value_1.csv"
results.to_csv(output_file)

print(f"\nSaved output to: {output_file}")
print("Columns:")
for c in results.columns:
    print("  -", c)

# -------------------------------------------------------------------
# Basic Validation
# -------------------------------------------------------------------

print("\nValidation:")
for c in results.columns:
    null_rate = results[c].isna().mean()
    print(f"{c}: null_rate={null_rate:.4f}")
    