# f15_testcheck/unit/ts03_PA_MkSt__6build_regime_state_pro_1.py
# Run: python -m f15_testcheck.unit.ts03_PA_MkSt__6build_regime_state_pro_1

import pandas as pd
from f04_features.price_action.market_structure import build_regime_state_pro

_PATH = "f16_test_results/"

# ---------------------------------------------------
# Load raw data
# ---------------------------------------------------
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

if not {"high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain high, low, close")

# ---------------------------------------------------
# Run advanced regime builder
# ---------------------------------------------------
result = build_regime_state_pro(
    df,
    atr_window=14,
    adr_window=14,
    tf_higher="5min",
    smoothing="atr",
    atr_method="wilder",
)

# ---------------------------------------------------
# Quick sanity checks
# ---------------------------------------------------
print(result.head(10))
print("Any Bullish (+1):", (result["regime"] == 1).any())
print("Any Bearish (-1):", (result["regime"] == -1).any())
print("Any Neutral (0):", (result["regime"] == 0).any())

# ---------------------------------------------------
# Save output to CSV
# ---------------------------------------------------
result.to_csv(f"{_PATH}ts03_PA_MkSt__6build_regime_state_pro_1.csv")

print("--------------------------------------------------")
print("Added 1 test result files to f16_test_results:")
print("✅   ts03_PA_MkSt__6build_regime_state_pro_1.csv")
print("--------------------------------------------------")