# f15_testcheck/unit/ts03_idx_lvl__5fibo_levels_from_legs_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_lvl__5fibo_levels_from_legs_1

import pandas as pd
import numpy as np
from datetime import datetime
from f04_features.indicators.zigzag import zigzag_mtf_adapter
from f04_features.indicators.levels import (
    fibo_levels_from_legs_orig,
    fibo_levels_from_legs_njit,
    fibo_levels_from_legs,
)

_PATH = "f16_test_results/"
# --- بارگذاری داده واقعی -------------------------------------------
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

# --- ساخت ZigZag ----------------------------------------------------
zz_df = zigzag_mtf_adapter(
    high=df["high"],
    low=df["low"],
    tf_higher="5min",
    depth=12,
    deviation=5.0,
    backstep=10,
    point=0.01,
    mode= "forward_fill",     # روی نتیجه توابع مورد تست تأثیر ندارد
    extend_last_leg=False,    # روی نتیجه توابع مورد تست تأثیر ندارد
    use_timeshift=True,    # تأثیر دارد
)

# --- اجرای هر سه تابع ----------------------------------------------
results = {}
timings = {}

functions = {
    "fibo_orig": fibo_levels_from_legs_orig,
    "fibo_njit": fibo_levels_from_legs_njit,
    "fibo_wrapper": fibo_levels_from_legs,
}

n = len(df)

for name, func in functions.items():
    t1 = datetime.now()
    res = func(df, zz_df, extend_last_leg=True)
    t2 = datetime.now()
    results[name] = res
    timings[name] = (t2 - t1).total_seconds()
    print(f"{name} execution time: {timings[name]:.6f} sec")

# --- مقایسه orig و njit --------------------------------------------
fibo_df_orig = results["fibo_orig"].fillna(0)
fibo_df_njit = results["fibo_njit"].fillna(0)

diff_df = (fibo_df_orig - fibo_df_njit).abs()
max_diff = diff_df.max().max()

print(f"Max absolute difference (orig vs njit): {max_diff:.10f}")

# --- ترکیب برای خروجی CSV ------------------------------------------
compare_df = pd.concat([
    fibo_df_orig.add_suffix("_orig"),
    fibo_df_njit.add_suffix("_njit"),
    diff_df.add_suffix("_diff"),
], axis=1)

compare_df.to_csv(f"{_PATH}ts03_idx_lvl__5fibo_levels_from_legs_1.csv")
print("----------------------------------------")
print("Added 1 test result files to f16_test_results:")
print("     ts03_idx_lvl__5fibo_levels_from_legs_1.csv")
print("----------------------------------------")
