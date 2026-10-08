# f15_testcheck/unit/ts03_idx_lvl__2sr_from_zigzag_legs_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_lvl__2sr_from_zigzag_legs_1
# Date modified: 1404/11/19

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import itertools

from f04_features.indicators.levels import (
    sr_from_zigzag_legs_orig_1,
    sr_from_zigzag_legs_njit_1,
    sr_from_zigzag_legs_orig,
    sr_from_zigzag_legs_njit,
    sr_from_zigzag_legs,
)

_PATH = "f16_test_results/"
# --- Load real data ----------------------------------------------------------
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()

df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

# --- ZigZag parameters -------------------------------------------------------
tf_higher = "5min"
depth = 12
deviation = 5.0
backstep = 10

# --- Functions ---------------------------------------------------------------
functions = {
    # "sr_orig_1": sr_from_zigzag_legs_orig_1,
    # "sr_njit_1": sr_from_zigzag_legs_njit_1,
    "sr_orig": sr_from_zigzag_legs_orig,
    "sr_njit": sr_from_zigzag_legs_njit,
    "sr_wrapper": sr_from_zigzag_legs,
}

results = {}

# --- Run all implementations -------------------------------------------------
for name, func in functions.items():
    t1 = datetime.now()

    result = func(
        df,
        tf=tf_higher,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        extend_last_leg=True,
    )

    t2 = datetime.now()
    elapsed = round((t2 - t1).total_seconds(), 3)

    print(f"{name} runtime ({len(df)} rows): {elapsed} sec")

    result.reset_index().to_csv(f"{_PATH}ts03_idx_lvl__2sr_from_zigzag_legs_1_{name}.csv", index_label="no.")
    results[name] = result

# --- Utility: NaN-safe equality ---------------------------------------------
def allclose_nan(a, b, rtol=1e-5, atol=1e-8):
    if a.shape != b.shape:
        return False

    nan_mask = np.isnan(a) & np.isnan(b)
    a_clean = a[~nan_mask]
    b_clean = b[~nan_mask]

    if len(a_clean) == 0:
        return True

    return np.allclose(a_clean, b_clean, rtol=rtol, atol=atol)


# --- Pairwise comparison -----------------------------------------------------
print("\n--- Pairwise comparison ---")

for (name_a, df_a), (name_b, df_b) in itertools.combinations(results.items(), 2):

    print(f"\nComparing: {name_a}  <-->  {name_b}")

    # Shape test
    assert df_a.shape == df_b.shape, \
        f"Shape mismatch between {name_a} and {name_b}"

    # Support test
    sup_equal = allclose_nan(
        df_a["sr_support"].values,
        df_b["sr_support"].values,
    )

    # Resistance test
    res_equal = allclose_nan(
        df_a["sr_resistance"].values,
        df_b["sr_resistance"].values,
    )

    if not (sup_equal and res_equal):
        raise AssertionError(
            f"Value mismatch between {name_a} and {name_b}"
        )

    print("✔ Match")


print("\nAll implementations are fully consistent.")

print("----------------------------------------")
print("Added 3 test result files to f16_test_results:")
print("     ts03_idx_lvl__2sr_from_zigzag_legs_1_sr_orig.csv")
print("     ts03_idx_lvl__2sr_from_zigzag_legs_1_sr_njit.csv")
print("     ts03_idx_lvl__2sr_from_zigzag_legs_1_sr_wrapper.csv")
print("----------------------------------------")
