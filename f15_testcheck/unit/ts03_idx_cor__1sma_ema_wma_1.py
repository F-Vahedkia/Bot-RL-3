# f04_features/indicators/ts03_idx_cor__1sma_ema_wma_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_cor__1sma_ema_wma_1

import pandas as pd
from datetime import datetime

from f04_features.indicators.core import (
    sma_old, sma
    # ema_old, ema, ema_fast,
    # wma_slow, wma
)
# from f04_features.indicators.extras_trend import _sma
# from f04_features.indicators.utils import (
#     _ema_numpy,
#     ema
# )

_PATH = "f16_test_results/"
# _PATH = ""
#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

# --- Load data -----------------------------------------------------
t1 = datetime.now()
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
df = data[-5_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

if not {"open", "high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain open, high, low, close")
t2 = datetime.now()
print(f"Time taken to load data: {round((t2 - t1).total_seconds(), 5)} seconds, \
          length_df:{len(df)}")


# --- Calling functions ---------------------------------------------
funcs = {
    "sma_old": sma_old,
    "sma": sma,
    # "_sma": _sma,

    # "ema_old": ema_old,
    # "ema": ema,
    # "_ema_numpy": _ema_numpy,
    # "ema": ema,
    # "ema_new1": ema_new1,
    # "ema_new2": ema_new2,

    # "wwma_slowa": wma_slow,
    # "wma": wma,
}
# دیتافریم برای ذخیره همه نتایج
results_df = pd.DataFrame(data=df["close"] ,index=df.index)

for name, func in funcs.items():
    t1 = datetime.now()
    if name == "sma_old":
        result = func(df["close"], n=14)
    if name in {"sma", "_sma"}:
        result = func(df["close"], 14, 7)
    t2 = datetime.now()

    # t1 = datetime.now()
    # if name in {"ema_utils"}:
    #     result = func(df["close"], 14)
    # if name in {"ema_fast", "ema_complete"}:
    #     result = func(df["close"], 14, 7)
    # t2 = datetime.now()

    # t1 = datetime.now()
    # result = func(s=df["close"], n=14, min_periods=7)
    # t2 = datetime.now()


    elapsed = round((t2 - t1).total_seconds(), 3)
    print(f"Time taken to run {name}: {elapsed} seconds, length_df:{len(df)}")
    # ذخیره خروجی در دیتافریم مشترک
    results_df[name] = result


# --- Save results --------------------------------------------------
full_path = f"{_PATH}ts03_idx_cor__1sma_ema_wma_1.csv"
results_df.to_csv(full_path)
print(f"\nCSV saved -> {full_path}")
