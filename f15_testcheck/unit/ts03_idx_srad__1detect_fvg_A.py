# Run: python -m f15_testcheck.unit.ts03_idx_srad__1detect_fvg_0

#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime

from f04_features.indicators.core import atr as atr_core
from f04_features.indicators.sr_advanced import (
    detect_fvg_legacy as detect_fvg_1,
    detect_fvg_optimized as detect_fvg_2,
    )

# --- Logging -----------------------------------------------------
import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


# --- Load data -----------------------------------------------------
t1 = datetime.now()
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-100_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

if not {"high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain high, low, close")
t2 = datetime.now()
print(f"Time taken to run load data: {round((t2 - t1).total_seconds(), 3)} seconds, \
          length_df:{len(df)}")

# --- Calling detect_fvg --------------------------------------------
t1 = datetime.now()
my_fvg = detect_fvg_1(
    df["open"], df["high"], df["low"], df["close"],
    lookback = 6,
    atr_window = 14,
    min_size_pct_of_atr = 0.50,
    use_middle_filter = True,
    debug_mode = False,
)
t2 = datetime.now()
print(f"Time taken to run detect_fvg_1: {round((t2 - t1).total_seconds(), 3)} seconds, \
          length_df:{len(df)}")
pd.DataFrame(my_fvg).to_csv("ts03_idx_srad__1detect_fvg_1.csv")

# --- Calling detect_fvg --------------------------------------------
t1 = datetime.now()
my_fvg = detect_fvg_2(
    df["open"], df["high"], df["low"], df["close"],
    lookback = 6,
    atr_window = 14,
    min_size_pct_of_atr = 0.50,
    use_middle_filter = True,
    debug_mode = False,
)
t2 = datetime.now()
print(f"Time taken to run detect_fvg_2: {round((t2 - t1).total_seconds(), 3)} seconds, \
          length_df:{len(df)}")
pd.DataFrame(my_fvg).to_csv("ts03_idx_srad__1detect_fvg_2.csv")


