# f15_testcheck/unit/ts03_PA_MkSt__full_pipeline_1.py
# Run: python -m f15_testcheck.unit.ts03_PA_MkSt__full_pipeline_1

import pandas as pd
from f04_features.price_action.market_structure import (
    market_structure_pipeline,
    build_regime_state,
)

_PATH = "f16_test_results/"
# ---------------------------------------------------
# Load data
# ---------------------------------------------------
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

required_cols = {"open", "high", "low", "close"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Data must contain columns: {required_cols}")

# ---------------------------------------------------
# Run Market Structure Pipeline
# ---------------------------------------------------
ms_df = market_structure_pipeline(df)

# ---------------------------------------------------
# Build Regime State
# ---------------------------------------------------
final_df = build_regime_state(ms_df)

# ---------------------------------------------------
# Inspect results
# ---------------------------------------------------
print(final_df.tail(20))
print("Unique regimes:", final_df["regime"].unique())
print("Count Bullish:", (final_df["regime"] == 1).sum())
print("Count Bearish:", (final_df["regime"] == -1).sum())
print("Count Neutral:", (final_df["regime"] == 0).sum())

# ---------------------------------------------------
# Save output to CSV
# ---------------------------------------------------
final_df.to_csv(f"{_PATH}ts03_PA_MkSt__full_pipeline_1.csv")

print("--------------------------------------------------")
print("Added 1 test result files to f16_test_results:")
print("✅   ts03_PA_MkSt__full_pipeline_1.csv")
print("--------------------------------------------------")