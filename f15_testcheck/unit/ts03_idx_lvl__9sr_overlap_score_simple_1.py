# f15_testcheck/unit/ts03_idx_lvl__9sr_overlap_score_simple_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_lvl__9sr_overlap_score_simple_1

import pandas as pd
import numpy as np
from f04_features.indicators.levels import sr_overlap_score_simple

_PATH = "f16_test_results/"
# --- بارگذاری داده واقعی -------------------------------------------
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-20:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

# برای تست فقط از قیمت بسته شدن کندل استفاده می‌کنیم
if "close" not in df.columns:
    raise ValueError("Column 'close' not found in data")

df.to_csv(f"{_PATH}ts03_idx_lvl__9sr_overlap_score_simple_1_data.csv")

# --- تعریف سطوح S/R تست --------------------------------------------
# نمونه سطوح (می‌توانی از سیستم واقعی خودت بگیری)
sr_levels = [
    df["close"].iloc[-1] * 0.995,
    df["close"].iloc[-1] * 1.000,
    df["close"].iloc[-1] * 1.005,
]

tol_pct = 0.003  # 0.3%  # tolerance percentage

# --- محاسبه امتیاز برای هر کندل -----------------------------------
scores = []

for price in df["close"].values:
    score = sr_overlap_score_simple(
        price=price,
        sr_levels=sr_levels,
        tol_pct=tol_pct,
    )
    scores.append(score)

df["sr_overlap_score"] = scores

# --- ذخیره خروجی ---------------------------------------------------
df[["close", "sr_overlap_score"]].to_csv(f"{_PATH}ts03_idx_lvl__9sr_overlap_score_simple_1.csv")

print("✅ results saved to 'a1___sr_overlap_score.csv'.")
print("Score stats:")
print(df["sr_overlap_score"].describe())

print("----------------------------------------")
print("Added 2 test result files to f16_test_results:")
print("✅   ts03_idx_lvl__9sr_overlap_score_simple_1_data.csv")
print("✅   ts03_idx_lvl__9sr_overlap_score_simple_1.csv")
print("----------------------------------------")
