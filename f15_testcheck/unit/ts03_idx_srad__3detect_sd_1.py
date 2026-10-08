# f04_features/indicators/ts03_idx_srad__3detect_sd_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_srad__3detect_sd_1

#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

import pandas as pd
from datetime import datetime

from f04_features.indicators.sr_advanced import (
    detect_sd,
    detect_sd_numba,
)

# --- Load data -----------------------------------------------------
t1 = datetime.now()
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

if not {"open", "high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain open, high, low, close")
t2 = datetime.now()
print(f"Time taken to run load data: {round((t2 - t1).total_seconds(), 3)} seconds, \
          length_df:{len(df)}")


# --- Calling detect_fvg --------------------------------------------
t1 = datetime.now()
my_detect_fvg = detect_sd(
    df["open"], df["high"], df["low"], df["close"],
    base_len = 2,
    atr_window = 14,
    base_atr_max = 0.6,
    impulse_atr_min = 1.2,
)

t2 = datetime.now()
print(f"Time taken to run detect_sd: {round((t2 - t1).total_seconds(), 3)} seconds, \
          length_df:{len(df)}")
pd.DataFrame(my_detect_fvg).to_csv("ts03_idx_srad__3detect_sd.csv")

