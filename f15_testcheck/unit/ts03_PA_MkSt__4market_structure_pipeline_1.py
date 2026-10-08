# f15_testcheck/unit/ts03_PA_MkSt__4market_structure_pipeline_1.py
# Run: python -m f15_testcheck.unit.ts03_PA_MkSt__4market_structure_pipeline_1

import pandas as pd
from f04_features.price_action.market_structure import market_structure_pipeline

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
result = market_structure_pipeline(df)

# --- بررسی نتایج ---
print(result)
print("\nColumns:", result.columns.tolist())
print("Any HH:", result["HH"].any(), "Any HL:", result["HL"].any())
print("Any bos_up:", result["bos_up"].any(), "Any choch_down:", result["choch_down"].any())

result.to_csv(f"{_PATH}ts03_PA_MkSt__4market_structure_pipeline_1.csv")
print("--------------------------------------------------")
print("Added 1 test result files to f16_test_results:")
print("✅   ts03_PA_MkSt__4market_structure_pipeline_1.csv")
print("--------------------------------------------------")