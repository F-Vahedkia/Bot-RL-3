# f04_features/indicators/ts03_idx_srad__2make_fvg_2.py
# Run: python -m f15_testcheck.unit.ts03_idx_srad__2make_fvg_2

#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

import pandas as pd
from datetime import datetime

from f04_features.indicators.sr_advanced import (
    detect_fvg_optimized,
    make_fvg
)

# --- Load data -----------------------------------------------------
t1 = datetime.now()
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-300:-200].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

if not {"open", "high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain open, high, low, close")
t2 = datetime.now()
print(f"Time taken to run load data: {round((t2 - t1).total_seconds(), 3)} seconds, \
          length_df:{len(df)}")


# --- Calling detect_fvg --------------------------------------------
t1 = datetime.now()
my_detect_fvg = detect_fvg_optimized(
    df["open"], df["high"], df["low"], df["close"],
    lookback = 2,
    atr_window = 14,
    min_size_pct_of_atr = 0.50,
    use_middle_filter = True,
    debug_mode = False,
)

t2 = datetime.now()
print(f"Time taken to run detect_fvg: {round((t2 - t1).total_seconds(), 3)} seconds, \
          length_df:{len(df)}")
pd.DataFrame(my_detect_fvg).to_csv("ts03_idx_srad__1detect_fvg.csv")


# --- Calling make_fvg ----------------------------------------------
t1 = datetime.now()
my_make_fvg = make_fvg(df)
t2 = datetime.now()
print(f"Time taken to run make_fvg: {round((t2 - t1).total_seconds(), 3)} seconds, \
          length_df:{len(df)}")
temp_df = pd.concat([df[["open","high","low","close"]], pd.DataFrame(my_make_fvg)],
                    axis=1,
                     )
temp_df.replace(0,None).to_csv("ts03_idx_srad__2make_fvg_2.csv")