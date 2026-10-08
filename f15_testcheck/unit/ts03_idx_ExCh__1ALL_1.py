# f04_features/indicators/ts03_idx_ExCh__1ALL_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_ExCh__1ALL_1

#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

import pandas as pd
from datetime import datetime

from f04_features.indicators.extras_channel import registry


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

t1 = datetime.now()
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-50_000:].copy()
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

print("\nRegistry loaded:")
for name in registry.keys():
    print(f" - {name}")


# -------------------------------------------------------------------
# Run all registry features
# -------------------------------------------------------------------

print("\nRunning all extras_channel indicators...\n")
results = pd.DataFrame(index=df.index)

for key, func in registry.items():

    print(f"Running registry feature: {key}")
    t_start = datetime.now()
    
    # --- registry maker functions expect df as input
    out_dict = func(df)
    t_end = datetime.now()
    elapsed = round((t_end - t_start).total_seconds(), 3)
    print(f"  Time: {elapsed} seconds")

    # --- out_dict may contain multiple feature series
    for feature_name, series in out_dict.items():
        if len(series) != len(df):
            raise ValueError(
                f"Feature {feature_name} produced incorrect length "
                f"{len(series)} != {len(df)}"
            )
        results[feature_name] = series


# -------------------------------------------------------------------
# Save Output
# -------------------------------------------------------------------

output_file = "ts03_idx_ExCh__1ALL_1.csv"
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
    print(f"{c}: null_rate={null_rate:.5f}")

print("\n[TEST COMPLETED SUCCESSFULLY]")

