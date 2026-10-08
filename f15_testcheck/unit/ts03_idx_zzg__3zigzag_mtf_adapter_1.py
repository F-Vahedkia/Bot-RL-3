# f15_testcheck/unit/ts03_idx_zzg__3zigzag_mtf_adapter_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_zzg__3zigzag_mtf_adapter_1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from f04_features.indicators.zigzag import zigzag_mtf_adapter

_PATH = "f16_test_results/"
_PATH = ""
#--- ساختن داده های واقعی ----------------------------- Start
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)
#-------------------------------------------------------

high = df["high"]
low  = df["low"]
tf_higher = "5min"

zz_last = zigzag_mtf_adapter(
    high=high, low=low, tf_higher=tf_higher,
    depth=12, deviation=0.05, backstep=10,
    mode="last",
    extend_last_leg=False,
    use_timeshift=True,
)
zz_last.reset_index().to_csv(f"{_PATH}ts03_idx_zzg__3zigzag_mtf_adapter_1_zz_last.csv", index_label="no.")
legs = pd.DataFrame(zz_last.attrs["legs"])
legs.to_csv(f"{_PATH}ts03_idx_zzg__3zigzag_mtf_adapter_1_legs_last.csv", index_label="no.")


zz_ffill = zigzag_mtf_adapter(
    high=high, low=low, tf_higher=tf_higher,
    depth=12, deviation=5.0, backstep=10,
    mode="forward_fill",
    extend_last_leg=True,
    use_timeshift=True,
)
zz_ffill.reset_index().to_csv(f"{_PATH}ts03_idx_zzg__3zigzag_mtf_adapter_1_zz_ffill.csv", index_label="no.")
legs = pd.DataFrame(zz_ffill.attrs["legs"])
legs.to_csv(f"{_PATH}ts03_idx_zzg__3zigzag_mtf_adapter_1_legs_ffill.csv", index_label="no.")

print("--------------------------------------------------")
print("Added 4 test result files to f16_test_results:")
print("     ts03_idx_zzg__3zigzag_mtf_adapter_1_zz_last")
print("     ts03_idx_zzg__3zigzag_mtf_adapter_1_legs_last.csv")
print("     ts03_idx_zzg__3zigzag_mtf_adapter_1_zz_ffill.csv")
print("     ts03_idx_zzg__3zigzag_mtf_adapter_1_legs_ffill.csv")
print("--------------------------------------------------")
