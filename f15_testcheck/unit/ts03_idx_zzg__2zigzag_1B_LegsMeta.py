# f15_testcheck/unit/ts03_idx_zzg__2zigzag_1B_LegsMeta.py
# Run: python -m f15_testcheck.unit.ts03_idx_zzg__2zigzag_1B_LegsMeta

import pandas as pd
from pathlib import Path
from f04_features.indicators.zigzag import zigzag_wrapper as zigzag

_PATH = "f16_test_results/"
_PATH = ""
# ------------------------------------------------------------
# Loading Data
# ------------------------------------------------------------
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

# ------------------------------------------------------------
# Run ZigZag
# ------------------------------------------------------------
zzg = zigzag(
    high=df["high"], low=df["low"],
    depth=12, deviation=0.05, backstep=10, addmeta=True,
)
zzg.reset_index().to_csv(f"{_PATH}ts03_idx_zzg__2zigzag_1B_LegsMeta_Result.csv", index_label="no.")

df[["high","low"]].reset_index().to_csv(f"{_PATH}ts03_idx_zzg__2zigzag_1B_LegsMeta_orig_HL.csv", index_label="no.")

# ------------------------------------------------------------
# Extract legs metadata
# ------------------------------------------------------------
legs = zzg.attrs.get("legs", [])
legs_df = pd.DataFrame(legs)

# ------------------------------------------------------------
# Save to CSV
# ------------------------------------------------------------
legs_df.to_csv(f"{_PATH}ts03_idx_zzg__2zigzag_1B_LegsMeta.csv")

print("--------------------------------------------------")
print("Added 3 test result files to f16_test_results:")
print("     ts03_idx_zzg__2zigzag_1B_LegsMeta_Result.csv")
print("     ts03_idx_zzg__2zigzag_1B_LegsMeta_orig_HL.csv")
print("     ts03_idx_zzg__2zigzag_1B_LegsMeta.csv")
print("--------------------------------------------------")
