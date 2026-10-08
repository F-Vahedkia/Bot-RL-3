# f15_testcheck/unit/ts03_idx_zzg__1zigzag_new_1C.py
# Run: python -m f15_testcheck.unit.ts03_idx_zzg__1zigzag_new_1C

import numpy as np
import pandas as pd
from datetime import datetime
from f04_features.indicators.zigzag import (
    _zigzag_numpy,
    _zigzag_njit,
)
_PATH = "f16_test_results/"
_PATH = ""
depth=10
deviation=0.0
backstep=10

# ============================================================
# Check functions
# ============================================================
""" بررسی میکند که کفی که بین دو سقف متوالی قرار دارد،
پایین ترین کف در محدوده بین آن دو سقف باشد و بالعکس """

def check_zigzag_extrema_consistency(
    high: np.ndarray,
    low: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """
    ورودی:
        high, low : آرایه‌ی قیمت‌ها
        state     : خروجی زیگزاگ (همان state تابع v4)
                    -1: سقف (pivot high)
                    +1: کف  (pivot low)
                        0: بدون pivot

    خروجی:
        violations : np.ndarray با طول n
                        پیش‌فرض 0 است؛
                        هرجا نقض شرط رخ دهد مقدار 1 می‌شود.

        تعریف نقض:
        - بین دو سقف متوالی، کفی که انتخاب شده (اگر وجود دارد)
            کمترین low در بازه‌ی بین دو سقف نباشد.
        - بین دو کف متوالی، سقفی که انتخاب شده (اگر وجود دارد)
            بیشترین high در بازه‌ی بین دو کف نباشد.
    """
    n = len(high)
    violations = np.zeros(n, dtype=np.int8)
    fault_poss: list[int] = []

    # ایندکس همه‌ی pivotها
    pivot_idx = np.nonzero(state != 0)[0]
    if len(pivot_idx) < 3:
        return violations, fault_poss  # pivot کافی برای بررسی نیست

    for k in range(len(pivot_idx) - 2):
        i0 = pivot_idx[k]
        i1 = pivot_idx[k + 1]
        i2 = pivot_idx[k + 2]

        s0 = state[i0]
        s1 = state[i1]
        s2 = state[i2]

        # --- الگوی High-Low-High ---
        if s0 == -1 and s1 == 1 and s2 == -1:
            # بازه‌ی بین دو سقف (بدون خود سقف‌ها)
            if i2 - i0 > 1:  # حداقل 1 کندل بینشان باشد
                segment_lows = low[i0+1:i2]  # شامل i1 نیز هست
                min_low = segment_lows.min()
                # low انتخاب‌شده در pivot میانی
                pivot_low = low[i1]
                # اگر pivot_low کمترین low نباشد → نقض
                if pivot_low > min_low + 1e-12:  # تلورانس کوچک عدی
                    violations[i1] = 1
                    fault_poss.append(i1)

        # --- الگوی Low-High-Low ---
        if s0 == 1 and s1 == -1 and s2 == 1:
            if i2 - i0 > 1:
                segment_highs = high[i0+1:i2]
                max_high = segment_highs.max()
                pivot_high = high[i1]
                if pivot_high < max_high - 1e-12:
                    violations[i1] = 1
                    fault_poss.append(i1)

    return violations, fault_poss


""" بررسی میکند که در ستون state بعد از هر سقف یک کف باشد و بالعکس """
def check_zigzag_state_alternation(
    state: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """
    ورودی:
        state : خروجی زیگزاگ
                -1: سقف (pivot high)
                +1: کف  (pivot low)
                 0: بدون pivot

    خروجی:
        violations : np.ndarray با طول n
                     پیش‌فرض 0 است؛
                     اگر دو pivot متوالی علامت یکسان داشته باشند
                     مقدار 1 در pivot دوم ثبت می‌شود.

    تعریف نقض:
        در ZigZag باید pivotها به صورت زیر باشند:

            ... High → Low → High → Low ...

        یعنی state باید بین -1 و +1 به‌صورت یکی‌درمیان تغییر کند.

        اگر دو pivot پشت‌سرهم هر دو:
            -1  (High-High)
        یا
            +1  (Low-Low)

        باشند → violation.
    """

    n = len(state)
    violations = np.zeros(n, dtype=np.int8)
    fault_poss: list[int] = []

    pivot_idx = np.nonzero(state != 0)[0]

    if len(pivot_idx) < 2:
        return violations, fault_poss

    for k in range(len(pivot_idx) - 1):
        i0 = pivot_idx[k]
        i1 = pivot_idx[k + 1]

        s0 = state[i0]
        s1 = state[i1]

        # اگر دو pivot پشت سر هم علامت یکسان داشته باشند
        if s0 == s1:
            violations[i1] = 1
            fault_poss.append(i1)

    return violations, fault_poss


""" بررسی میکند که کفی که بین دو سقف پیدا شده است، از هر دو سقف پایینتر باشد و بالعکس """
def check_zigzag_triplet_order(
    high: np.ndarray,
    low: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """
    ورودی:
        high, low : آرایه قیمت‌ها
        state     : خروجی زیگزاگ
                    -1: pivot high
                    +1: pivot low
                     0: بدون pivot

    خروجی:
        violations : np.ndarray با طول n
                     پیش‌فرض 0 است؛
                     اگر ترتیب سه pivot متوالی نقض شود مقدار 1
                     روی pivot میانی ثبت می‌شود.

    تعریف نقض:

    حالت HLH:
        High → Low → High

        باید برقرار باشد:
            low[i1] < low[i0]
            low[i1] < low[i2]

        اگر برقرار نباشد → violation

    حالت LHL:
        Low → High → Low

        باید برقرار باشد:
            high[i1] > high[i0]
            high[i1] > high[i2]

        اگر برقرار نباشد → violation
    """

    n = len(state)
    violations = np.zeros(n, dtype=np.int8)
    fault_poss: list[int] = []

    pivot_idx = np.nonzero(state != 0)[0]

    if len(pivot_idx) < 3:
        return violations, fault_poss


    for k in range(len(pivot_idx) - 2):

        i0 = pivot_idx[k]
        i1 = pivot_idx[k + 1]
        i2 = pivot_idx[k + 2]

        s0 = state[i0]
        s1 = state[i1]
        s2 = state[i2]

        # --- HLH ---
        if s0 == -1 and s1 == 1 and s2 == -1:
            l0 = low[i0]
            l1 = low[i1]
            l2 = low[i2]
            if not (l1 < l0 and l1 < l2):
                violations[i1] = 1
                fault_poss.append(i1)

        # --- LHL ---
        if s0 == 1 and s1 == -1 and s2 == 1:
            h0 = high[i0]
            h1 = high[i1]
            h2 = high[i2]
            if not (h1 > h0 and h1 > h2):
                violations[i1] = 1
                fault_poss.append(i1)

    return violations, fault_poss


# ============================================================
# Load data
# ============================================================
t1 = datetime.now()
data = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
t2 = datetime.now()
df = data[-100_000:].copy()
t3 = datetime.now()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)
t4 = datetime.now()
# print(f"Time taken to read CSV with {len(data)} rows: {round((t2 - t1).total_seconds(), 1)} seconds")
# print(f"Time taken to slice data with {len(df)} rows: {round((t3 - t2).total_seconds(), 1)} seconds")
# print(f"Time taken to process datetime/index: {round((t4 - t3).total_seconds(), 1)} seconds")
# print("============================================================================")

# ============================================================
# Call functions
# ============================================================
zigzag_funcs = {
                "no_njit": _zigzag_numpy,
                "by_njit":  _zigzag_njit,
                }

check = pd.DataFrame(df[["high", "low"]])
# --- containers for fault positions ---
extr_fault_poss = {}
alt_fault_poss = {}
trip_fault_poss = {}

for key in zigzag_funcs.keys():
    # continue
    df_index = df.index
    func = zigzag_funcs[key]
    t1 = datetime.now()
    state, high_actual, low_actual, confirmed_at, developing_leg = \
        func(
            df["high"].values,
            df["low"].values,
            depth=depth, deviation=deviation, backstep=backstep,
        )
    t2 = datetime.now()
    diff_time = round((t2 - t1).total_seconds(), 3)
    print(f"Time taken to run zigzag ({key}): {diff_time} seconds, length_df:{len(df)}")
    df_new = pd.DataFrame(
        index = df_index,
        data = {
            "high": df["high"],
            "low": df["low"],
            "state": state,
            "high_actual": high_actual,
            "low_actual": low_actual,
            "confirmed_at": confirmed_at,
            "developing_leg": developing_leg,
        }
    )
    df_new.reset_index().to_csv(f"{_PATH}ts03_idx_zzg__1zigzag_new_1C_{key}.csv", index_label="no.")

    # --- Check functions ---
    if key == "":
        state = -1 * state
    
    violations, fault_poss = check_zigzag_extrema_consistency(
        df["high"].values, df["low"].values, state)
    check[key + "_extr"] = violations
    extr_fault_poss[key] = fault_poss


    violations, fault_poss = check_zigzag_state_alternation(state)
    check[key + "_alt"] = violations
    alt_fault_poss[key] = fault_poss


    violations, fault_poss = check_zigzag_triplet_order(
        df["high"].values, df["low"].values, state)
    check[key + "_trip"] = violations
    trip_fault_poss[key] = fault_poss

# ============================================================
# Calling wrapper
# ============================================================
# t1 = datetime.now()
# zzg_df = zigzag(
#     df["high"], df["low"],
#     depth=depth, deviation=deviation, backstep=backstep,
#     final_check = True,
# )
# t2 = datetime.now()


# diff_time = round((t2 - t1).total_seconds(), 3)
# print(f"Time taken to run zigzag ({key}): {diff_time} seconds, length_df:{len(df)}")


# --- Check functions ---
# state_wrapper = -1 * zzg_df["state"].values

# violations, fault_poss = check_zigzag_extrema_consistency(
#     df["high"].values, df["low"].values, state_wrapper
# )
# check["wrapper_extr"] = violations
# extr_fault_poss["wrapper"] = fault_poss

# violations, fault_poss = check_zigzag_state_alternation(state_wrapper)
# check["wrapper_alt"] = violations
# alt_fault_poss["wrapper"] = fault_poss

# violations, fault_poss = check_zigzag_triplet_order(
#     df["high"].values, df["low"].values, state_wrapper
# )
# check["wrapper_trip"] = violations
# trip_fault_poss["wrapper"] = fault_poss

# -----------------------------------------------
# saving chech results to .csv
# -----------------------------------------------
check.to_csv(f"{_PATH}ts03_idx_zzg__1zigzag_new_1C_check.csv", index_label="no.")

NUMBER = 20
# -----------------------------------------------
# print check-1 at terminal
# -----------------------------------------------
print("\n" + "="*60)
print("  EXTREMA CONSISTENCY FAULTS (extr)")
print("="*60)

for key in ["no_njit", "by_njit"]:
    col = key + "_extr"
    count = int(check[col].sum())
    poss = extr_fault_poss.get(key, [])

    poss = poss[:NUMBER] if len(poss) > NUMBER else poss

    if poss: # and len(poss) > 0:
        print(f"{col:<18} {count:4d}   ({', '.join(map(str, poss))})")
    else:
        print(f"{col:<18} {count:4d}")

# -----------------------------------------------
# print check-2 at terminal
# -----------------------------------------------
print("\n" + "="*60)
print("  STATE ALTERNATION FAULTS (alt)")
print("="*60)

for key in ["no_njit", "by_njit"]:
    col = key + "_alt"
    count = int(check[col].sum())
    poss = alt_fault_poss.get(key, [])

    poss = poss[:NUMBER] if len(poss) > NUMBER else poss

    if poss: # and len(poss) > 0:
        print(f"{col:<18} {count:4d}   ({', '.join(map(str, poss))})")
    else:
        print(f"{col:<18} {count:4d}")

# -----------------------------------------------
# print check-3 at terminal
# -----------------------------------------------
print("\n" + "="*60)
print("  TRIPLET ORDER FAULTS (trip)")
print("="*60)

for key in ["no_njit", "by_njit"]:
    col = key + "_trip"
    count = int(check[col].sum())
    poss = trip_fault_poss.get(key, [])

    poss = poss[:NUMBER] if len(poss) > NUMBER else poss

    if poss: # and len(poss) > 0:
        print(f"{col:<18} {count:4d}   ({', '.join(map(str, poss))})")
    else:
        print(f"{col:<18} {count:4d}")


# --- saving wrapper results to .csv ---
# temp = pd.concat([df[["high", "low"]], zzg_df], axis=1)
# temp.reset_index().to_csv(f"{_PATH}ts03_idx_zzg__1zigzag_1B_wrapper.csv", index_label="no.")
# pd.DataFrame(zzg_df.attrs["legs"]).to_csv(f"{_PATH}ts03_idx_zzg__1zigzag_1B_legs.csv")

