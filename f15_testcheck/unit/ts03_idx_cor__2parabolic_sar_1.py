# f04_features/indicators/ts03_idx_cor__2parabolic_sar_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_cor__2parabolic_sar_1

#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

import pandas as pd
from datetime import datetime

from f04_features.indicators.core import (
    parabolic_sar_orig,
    parabolic_sar_njit,
    parabolic_sar
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
    "parabolic_sar_orig": parabolic_sar_orig,
    "parabolic_sar_njit": parabolic_sar_njit,
    "parabolic_sar": parabolic_sar,
}
# دیتافریم برای ذخیره همه نتایج
results_df = pd.DataFrame(index=df.index)

for name, func in funcs.items():
    t1 = datetime.now()
    result = func(df["high"], df["low"])
    t2 = datetime.now()

    elapsed = round((t2 - t1).total_seconds(), 3)
    print(f"Time taken to run {name}: {elapsed} seconds, length_df:{len(df)}")
    # ذخیره خروجی در دیتافریم مشترک
    results_df[name] = result


# --- Save results --------------------------------------------------
results_df.to_csv("ts03_idx_cor__2parabolic_sar_1.csv")
