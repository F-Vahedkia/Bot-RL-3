# f04_features/indicators/ts03_idx_cor__3heikin_ashi_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_cor__3heikin_ashi_1

#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

import pandas as pd
from datetime import datetime

from f04_features.indicators.core import (
    # heikin_ashi,
    heikin_ashi_numpy,
    heikin_ashi_njit,
)

# --- Load data -----------------------------------------------------
t1 = datetime.now()
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-50_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

if not {"open", "high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain open, high, low, close")
t2 = datetime.now()
elapsed = round((t2 - t1).total_seconds(), 3)
print(f"Time taken to load data: {elapsed} seconds, length_df:{len(df)}")


# --- Calling functions ---------------------------------------------
funcs = {
    # "heikin_ashi": heikin_ashi,
    "heikin_ashi_numpy": heikin_ashi_numpy,
    "heikin_ashi_njit": heikin_ashi_njit,
}
# دیتافریم برای ذخیره همه نتایج
results_df = pd.DataFrame(index=df.index)

for name, func in funcs.items():
    t1 = datetime.now()
    ha_open, ha_high, ha_low, ha_close = func(
        df["open"], df["high"], df["low"], df["close"]
    )
    t2 = datetime.now()

    elapsed = round((t2 - t1).total_seconds(), 3)
    print(f"Time taken to run {name}: {elapsed} seconds, length_df:{len(df)}")

    results_df[f"{name}_open"] = ha_open
    results_df[f"{name}_high"] = ha_high
    results_df[f"{name}_low"] = ha_low
    results_df[f"{name}_close"] = ha_close

# --- Save results --------------------------------------------------
results_df.to_csv("ts03_idx_cor__3heikin_ashi_1.csv")
