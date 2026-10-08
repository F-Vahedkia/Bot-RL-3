# f15_testcheck/unit/ts03_idx_zzg__1zigzag_new_wrap_1E.py
# Run: python -m f15_testcheck.unit.ts03_idx_zzg__1zigzag_new_wrap_1E

import numpy as np
import pandas as pd
from datetime import datetime
from f04_features.indicators.zigzag import zigzag_new_wrapper
from f04_features.indicators.utils import compute_atr

_PATH = "f16_test_results/"
_PATH = ""

# ============================================================
# Paramaters
# ============================================================
depth=12
deviation=0.05
backstep=10
# ---
soft_confirmation = True
deviation_soft_mult = 1.0
time_threshold = 10
atr_mult = 1.5

# ============================================================
# Load data
# ============================================================
t1 = datetime.now()
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
t2 = datetime.now()
df = data[-10_000:].copy()
t3 = datetime.now()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)
t4 = datetime.now()
# print(f"Time taken to read CSV with {len(data)} rows: {round((t2 - t1).total_seconds(), 1)} seconds")
# print(f"Time taken to slice data with {len(df)} rows: {round((t3 - t2).total_seconds(), 1)} seconds")
# print(f"Time taken to process datetime/index: {round((t4 - t3).total_seconds(), 1)} seconds")
# print("============================================================================")

# ============================================================
# Call functions
# ============================================================
zigzag_funcs = {
                "new_wrap": zigzag_new_wrapper,
                }

check = pd.DataFrame(df[["high", "low"]])
# --- containers for fault positions ---

for key in zigzag_funcs.keys():
    # continue
    df_index = df.index
    func = zigzag_funcs[key]
    t1 = datetime.now()
    state, high_actual, low_actual, confirmed_at, developing_leg = \
        func(
            df["high"].values,
            df["low"].values,
            depth=depth, deviation=deviation, backstep=backstep,

            soft_confirmation = True,
            atr = compute_atr(df[["high", "low", "close"]], window=depth),
            deviation_soft_mult = 1.0,
            time_threshold = 10,
            atr_mult = 1.5,
        )
    t2 = datetime.now()
    diff_time = round((t2 - t1).total_seconds(), 3)
    print(f"Time taken to run zigzag ({key}): {diff_time} seconds, length_df:{len(df)}")
    df_new = pd.DataFrame(
        index = df_index,
        data = {
            "high": df["high"],
            "low": df["low"],
            "state": state,
            "high_actual": high_actual,
            "low_actual": low_actual,
            "confirmed_at": confirmed_at,
            "developing_leg": developing_leg,
        }
    )
    df_new.reset_index().to_csv(f"{_PATH}ts03_idx_zzg__1zigzag_new_1E_{key}.csv", index_label="no.")



# ============================================================
# saving wrapper results to .csv
# ============================================================

# temp = pd.concat([df[["high", "low"]], zzg_df], axis=1)
# temp.reset_index().to_csv(f"{_PATH}ts03_idx_zzg__1zigzag_1B_wrapper.csv", index_label="no.")
# pd.DataFrame(zzg_df.attrs["legs"]).to_csv(f"{_PATH}ts03_idx_zzg__1zigzag_1B_legs.csv")

