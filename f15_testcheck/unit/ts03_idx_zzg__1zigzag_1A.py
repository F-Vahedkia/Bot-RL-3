# f15_testcheck/unit/ts03_idx_zzg__1zigzag_1A.py
# Run: python -m f15_testcheck.unit.ts03_idx_zzg__1zigzag_1A

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from f04_features.indicators.zigzag import (
    _zigzag_mql_numpy_complete,
    _zigzag_mql_njit_loopwise_complete,
    zigzag_wrapper as zigzag,
)
_PATH = "f16_test_results/"
_PATH = ""
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
# Call _zigzag_mql_numpy()
# ============================================================
zigzag_funcs = {
                "no_njit": _zigzag_mql_numpy_complete,
                "by_njit":  _zigzag_mql_njit_loopwise_complete,
                }

bytes_used = df["high"].nbytes + df["low"].nbytes

for key in zigzag_funcs.keys():
    # continue
    df_index = df.index
    func = zigzag_funcs[key]
    t1 = datetime.now()
    zzg, h_act, l_act, confirmed_at, developing_leg= \
        func(
            df["high"].values,
            df["low"].values,
            depth=12, deviation=0.05, backstep=13
        )
    t2 = datetime.now()
    print(f"Time taken to run zigzag ({key}): {round((t2 - t1).total_seconds(), 3)} seconds, \
          bytes_used:{bytes_used}, \
          length_df:{len(df)}")
    df_new = pd.DataFrame(
        index = df_index,
        data = {
            "high": df["high"],
            "low": df["low"],
            "state": zzg,
            "high_actual": h_act,
            "low_actual": l_act,
            "confirmed_at": confirmed_at,
            "developing_leg": developing_leg,
        }
    )
    df_new.reset_index().to_csv(f"{_PATH}ts03_idx_zzg__1zigzag_1A_{key}.csv", index_label="no.")

# ============================================================
# Call _zigzag_mql_numpy()
# ============================================================
# t1 = datetime.now()
# zzg_df = zigzag(df["high"], df["low"], depth=12, deviation=0.05, backstep=5, final_check=False)
# t2 = datetime.now()
# temp = pd.concat([df[["high", "low"]], zzg_df], axis=1)
# print(f"Time taken to run zigzag for {len(df)} candles: {round((t2 - t1).total_seconds(), 3)} seconds")
# temp.reset_index().to_csv(f"{_PATH}ts03_idx_zzg__1zigzag_1A.csv", index_label="no.")
# pd.DataFrame(zzg_df.attrs["legs"]).to_csv(f"{_PATH}ts03_idx_zzg__1zigzag_1A_legs.csv")

# print("--------------------------------------------------")
# print("Added 4 test result files to f16_test_results:")
# print("     ts03_idx_zzg__1zigzag_1A_no_njit.csv")
# print("     ts03_idx_zzg__1zigzag_1A_by_njit.csv")
# print("     ts03_idx_zzg__1zigzag_1A.csv")
# print("     ts03_idx_zzg__1zigzag_1A_legs.csv")
# print("--------------------------------------------------")
