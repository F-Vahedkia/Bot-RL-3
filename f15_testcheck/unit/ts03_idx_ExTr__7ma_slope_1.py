# f15_testcheck/unit/ts03_idx_ExTr__7ma_slope_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_ExTr__7ma_slope_1

import numpy as np
import pandas as pd
from datetime import datetime
# from f04_features.indicators.extras_trend import _ema, _sma, _atr
from f04_features.indicators.core import ema, sma, atr
# from f04_features.indicators.utils import compute_atr as atr

_PATH = "f16_test_results/"
# _PATH = ""
#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

# --- Load data -----------------------------------------------------
t1 = datetime.now()
data = pd.read_csv("f03_data/raw/XAUUSD/H1.csv")
t2 = datetime.now()

# --- Preparing data ------------------------------------------------
data["time"] = pd.to_datetime(data["time"], utc=True)
data.set_index("time", inplace=True)
data = data.astype("float64")

elapsed = round((t2 - t1).total_seconds(), 4)
print(f"Time taken to load data: {elapsed} seconds, length_df:{len(data)}")

if not {"open", "high", "low", "close"}.issubset(data.columns):
    raise ValueError("Data must contain open, high, low, close")


# --- Calling functions ---------------------------------------------
funcs = {
    "sma": sma,
    # "_sma": _sma,
    "ema": ema,
    # "_ema": _ema,
    "atr": atr,
    # "_atr": _atr,

}
# --- list of iterations -----------------------------
# candle_nums = [10_000, 30_000, 100_000, 300_000, 1_000_000, 5_000_000]
candle_nums = [1000]
res = []
results_df = None  # برای ذخیره خروجی نهایی در حالت multi-Series
df_for_csv = []  # برای ذخیره داده sliced مربوط به n = 10_000

for n in candle_nums:
    # --- slicing data ----------------
    df = data[- n:].copy()

    # --- result dataframe (base) -----
    results_df = df[["high", "low", "close"]].copy()

    print(f"\n===== Running indicators for n={n} =====")

    # --- Calling TR functions --------
    for name, func in funcs.items():
        t1 = datetime.now()
        # -------------------
        if name in {"sma", "ema"}:
            res = func(s=df["close"], n=10)
        if name in {"_sma", "_ema"}:
            res = func(series=df["close"], window=10, min_periods=10)
        if name == "atr":
            res = func(high=df["high"], low=df["low"], close=df["close"], n=10, method="classic")
        if name == "_atr":
            res = func(high=df["high"], low=df["low"], close=df["close"], window=10, min_periods=10)
        # -------------------

        t2 = datetime.now()

        elapsed = round((t2 - t1).total_seconds(), 3)
        print(f"{name:<10} | candles={n:<8} | time={elapsed*1000} milisec")

        # --- add result to dataframe ---
        if n == 1_000:
            results_df[f"{name}"] = res

    # --- Save to CSV only for specific candle count ---
    if n == 1_000:
        full_path = f"{_PATH}ts03_idx_ExTr__7ma_slope_1.csv"
        results_df.to_csv(full_path)
        print(f"\nCSV saved for n={n} -> {full_path}")
