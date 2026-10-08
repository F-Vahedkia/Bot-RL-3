# f15_testcheck/unit/ts03_idx_lvl__3sr_distance_from_levels_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_lvl__3sr_distance_from_levels_1
# Date modified: 1404/11/20

import pandas as pd
import numpy as np

from datetime import datetime
from f04_features.indicators.levels import (
    sr_from_zigzag_legs,
    sr_distance_from_levels,
)

_PATH = "f16_test_results/"
# _PATH = ""
#------------------------------------------------------------------------------
def main():

    #--- ساختن داده های واقعی ---
    data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
    df = data[-10_000:].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)

    # --- build SR levels ---
    sr = sr_from_zigzag_legs(
        df,
        tf="5min",
        depth=12,
        deviation=5.0,
        backstep=10,
        extend_last_leg=True,
    )

    # --- compute distances ---
    t1 = datetime.now()
    dist = sr_distance_from_levels(
        df,
        sr=sr,
    )
    t2 = datetime.now()
    temp = round((t2 - t1).total_seconds(), 1)
    print(f"Time taken to run 'sr_distance_from_levels' with {len(df)} rows: {temp} seconds")

    # --- save result ---
    out = pd.concat([df["close"], sr, dist], axis=1)
    out.to_csv(f"{_PATH}ts03_idx_lvl__3sr_distance_from_levels_1.csv")

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

    print("----------------------------------------")
    print("Added 1 test result files to f16_test_results:")
    print("     ts03_idx_lvl__3sr_distance_from_levels_1.csv")
    print("----------------------------------------")
