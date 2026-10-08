# f15_testcheck/unit/ts03_idx_lvl__4zigzag_leg_mask_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_lvl__4zigzag_leg_mask_1

# Date modified: 1405/11/20

import pandas as pd
import numpy as np
import time

from f04_features.indicators.zigzag import zigzag_mtf_adapter
from f04_features.indicators.levels import _zigzag_leg_mask, _zigzag_leg_mask_njit

_PATH = "f16_test_results/"
#------------------------------------------------------------------------------
def main():
    # --- بارگذاری داده واقعی ---
    data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
    df = data[-10_000:].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)

    high = df["high"]
    low  = df["low"]

    # --- اجرای zigzag_mtf_adapter برای تولید لگ‌های سازگار با ماسک ---
    zz_ltf = zigzag_mtf_adapter(
        high=high,
        low=low,
        tf_higher="1h",
        depth=5,
        deviation=5.0,
        backstep=3,
        point=0.01,
        mode="forward_fill"
    )

    # --- سری خروجی ماسک ---
    series = zz_ltf

    # --- اجرای ماسک نسخه عادی ---
    t0 = time.perf_counter()
    mask_original = _zigzag_leg_mask(series)
    t1 = time.perf_counter()

    # --- اجرای ماسک نسخه njit ---
    t2 = time.perf_counter()
    mask_njit = _zigzag_leg_mask_njit(series)
    t3 = time.perf_counter()

    # --- بررسی برابری ماسک‌ها ---
    if not mask_original.equals(mask_njit):
        print("⚠️ Warning: Masks do NOT match!")
    else:
        print("✅ Masks match exactly.")

    print(f"Original mask time for {len(df)}: {t1-t0:.6f} sec")
    print(f"Numba njit mask time for {len(df)}: {t3-t2:.6f} sec")

    # --- ذخیره خروجی CSV ---
    df_out = pd.DataFrame({
        "datetime": series.index,
        "mask_original": mask_original.values,
        "mask_njit": mask_njit.values
    })
    df_out.to_csv(f"{_PATH}ts03_idx_lvl__4zigzag_leg_mask_1.csv", index=False)

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

    print("----------------------------------------")
    print("Added 1 test result files to f16_test_results:")
    print("     ts03_idx_lvl__4zigzag_leg_mask_1.csv")
    print("----------------------------------------")
