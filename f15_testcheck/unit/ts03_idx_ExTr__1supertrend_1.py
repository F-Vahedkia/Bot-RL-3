# f04_features/indicators/ts03_idx_ExTr__1supertrend_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_ExTr__1supertrend_1

#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

import pandas as pd
from datetime import datetime

# from f04_features.indicators.core              import atr, ema, sma, wma
from f04_features.indicators.extras_trend_old1 import supertrend as supertrend_1
from f04_features.indicators.extras_trend_old2 import supertrend as supertrend_2
from f04_features.indicators.extras_trend_old3 import supertrend as supertrend_3
from f04_features.indicators.extras_trend      import (
    supertrend_numpy as supertrend_numpy,
    supertrend_njit as supertrend_njit,
    supertrend as supertrend,

)


# --- Load data -----------------------------------------------------
t1 = datetime.now()
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[- 50_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

if not {"open", "high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain open, high, low, close")
t2 = datetime.now()
elapsed = round((t2 - t1).total_seconds(), 3)
print(f"Time taken to load data: {elapsed} seconds, length_df:{len(df)}")


# --- Calling functions ---------------------------------------------
funcs = {
    # "suptr_1": supertrend_1,
    # "suptr_2": supertrend_2,
    # "suptr_3": supertrend_3,
    "suptr_numpy"  : supertrend_numpy,
    "suptr_njit"  : supertrend_njit,
    "suptr"  : supertrend,
}
# دیتافریم برای ذخیره همه نتایج
results_df = pd.DataFrame(index=df.index)

for name, func in funcs.items():
    t1 = datetime.now()
    result = func(df["high"], df["low"], df["close"], period=10, multiplier=3)
    t2 = datetime.now()

    elapsed = round((t2 - t1).total_seconds(), 3)
    print(f"Time taken to run {name}: {elapsed} seconds, length_df:{len(df)}")

    results_df[f"{name}"] = result

# --- Save results --------------------------------------------------
results_df.to_csv("ts03_idx_ExTr__1supertrend_1.csv")

