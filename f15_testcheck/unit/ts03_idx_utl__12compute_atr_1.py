# f04_features/indicators/ts03_idx_utl__10compute_atr_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_utl__10compute_atr_1

import pandas as pd
from datetime import datetime


from f04_features.indicators.utils        import compute_atr
# from f04_features.indicators.extras_trend import _atr

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
    "compute_atr": compute_atr,
    # "_atr": _atr
}
# دیتافریم برای ذخیره همه نتایج
results_df = pd.DataFrame(data=df[["high", "low", "close"]] ,index=df.index)

# for name, func in funcs.items():
    # t1 = datetime.now()
    # if name == "compute_atr":
    #     result = func(df[["high", "low", "close"]], window=14, method="classic", min_periods=5)
    # if name == "_atr":
    #     result = func(df["high"], df["low"], df["close"], window=14, min_periods=5)
    # t2 = datetime.now()

    # elapsed = round((t2 - t1).total_seconds(), 3)
    # print(f"Time taken to run {name}: {elapsed} seconds, length_df:{len(df)}")
    # # ذخیره خروجی در دیتافریم مشترک
    # results_df[name] = result


# t1 = datetime.now()
# res = _atr(df["high"], df["low"], df["close"], window=14, min_periods=5)
# t2 = datetime.now()
# elapsed = round((t2 - t1).total_seconds(), 3)
# print(f"Time taken to run {"_atr"}: {elapsed} seconds, length_df:{len(df)}")
# results_df["_atr"] = res

t1 = datetime.now()
res = compute_atr(df[["high", "low", "close"]], window=14, method="classic", min_periods=5)
t2 = datetime.now()
elapsed = round((t2 - t1).total_seconds(), 3)
print(f"Time taken to run {"compute_atr"}: {elapsed} seconds, length_df:{len(df)}")
results_df["compute_atr"] = res


# --- Save results --------------------------------------------------
full_path = f"{_PATH}ts03_idx_utl__10compute_atr_1.csv"
results_df.to_csv(full_path)
print(f"\nCSV saved -> {full_path}")