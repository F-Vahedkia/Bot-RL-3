# f15_testcheck/unit/ts03_PA_MkSt__5build_regime_state_1.py
# Run: python -m f15_testcheck.unit.ts03_PA_MkSt__5build_regime_state_1

import pandas as pd
from f04_features.price_action.market_structure import market_structure_pipeline, build_regime_state

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


# --- اجرای pipeline ---
ms_df = market_structure_pipeline(df)   # تولید swings, HH/HL/LH/LL, BOS/CHoCH
result = build_regime_state(ms_df)      # حالا regime درست ساخته می‌شود

# --- بررسی نتایج ---
print(result)
# print("\nRegime sequence:", result["regime"].tolist())
print("Any Bullish:", (result["regime"] == 1).any())
print("Any Bearish:", (result["regime"] == -1).any())
print("Any Neutral:", (result["regime"] == 0).any())

result.to_csv(f"{_PATH}ts03_PA_MkSt__5build_regime_state_1.csv")
print("--------------------------------------------------")
print("Added 1 test result files to f16_test_results:")
print("✅   ts03_PA_MkSt__5build_regime_state_1.csv")
print("--------------------------------------------------")