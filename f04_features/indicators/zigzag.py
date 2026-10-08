# f04_features/indicators/zigzag.py
# Created at (1404/10/--)
# Check: By use of f15_testcheck/unit/test_z1A_zigzag.py
# Check: By use of f15_testcheck/unit/test_z1B_zigzag_LegsMetadata.py
# Check: By use of f15_testcheck/unit/test_z2_zigzag_mtf_adapter.py
# Status in (Bot-RL-3): Reviewed at 1404/12/--

import numpy as np
import pandas as pd
from numba import njit
from typing import Literal

#====================================================================
# Vectorized by Numpy
#==================================================================== Func1
def _zigzag_mql_numpy_complete(
    high: np.ndarray,
    low: np.ndarray,
    depth: int = 12,
    deviation: float = 0.05,
    backstep: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy-based ZigZag with confirmed and developing leg tracking (retrospective confirmed_at).

    Returns:
        state_series: +1/-1 for lows/highs
        high_actual: values at highs
        low_actual: values at lows
        confirmed_at: index where pivot is confirmed (retrospective)
        developing_leg: +1/-1 provisional leg before confirmation
    """
    
    n = len(high)

    # بافرهای اصلی
    zz_buffer    = np.zeros(n)                 # نقاط ZigZag
    state_series = np.zeros(n, dtype=np.int8)  # -1=high, 1=low, 0=none

    # confirmed و developing
    confirmed_at   = np.full(n, -1, dtype=np.int32)
    developing_leg = np.zeros(n, dtype=np.int8)

    # ---------------------------------
    # phase-1: calculating local highs/lows
    #          by means of depth, deviation, backstep
    # ---------------------------------
    # دو سطر زیر یک آرایهٔ دوبعدی می‌سازد که شامل تمام پنجره‌های پشت‌سر‌هم
    # (sliding windows) از آرایهٔ low/high با طول ثابت depth است
    low_windows  = np.lib.stride_tricks.sliding_window_view(low, depth)  # len = (n - depth + 1)
    high_windows = np.lib.stride_tricks.sliding_window_view(high, depth)

    # برای هر پنجرهٔ depth، ایندکس (کمترین کف/بیشترین سقف) را داخل همان پنجره پیدا می‌کند
    low_idx_in_window  = np.argmin(low_windows, axis=1)
    high_idx_in_window = np.argmax(high_windows, axis=1)

    # تبدیل اندکسهای کف/سقف داخل هر پنجره به اندکس کل داده اصلی
    # low_idx  = np.arange(depth - 1, n) - (depth - 1) + low_idx_in_window
    # high_idx = np.arange(depth - 1, n) - (depth - 1) + high_idx_in_window
    low_idx  = np.arange(0, n - depth + 1) + low_idx_in_window
    high_idx = np.arange(0, n - depth + 1) + high_idx_in_window

    low_vals  = low[low_idx]
    high_vals = high[high_idx]

    # بافرهای اصلی
    low_map  = np.zeros(n)  # candidate lows
    high_map = np.zeros(n)  # candidate highs

    for i in range(depth - 1, n):
        # LOW
        val = low_vals[i - (depth - 1)]                                     # بدست آوردن مقدار کف هر پنجره
        if (low[i] - val) <= deviation:
            back_range = slice(max(0, i - backstep), i)                     # ساخت بازه: [i - backstep, i - 1]
            mask = (low_map[back_range] != 0) & (low_map[back_range] > val) # ساخت فیلتر برای حذف کف های بی اعتبار شده
            idx = np.where(mask)[0] + back_range.start                      # یک لیست است
            low_map[idx] = 0.0
        if low[i] == val:      # اگر کف کندل فعلی دقیقاً برابر با کف پنجره‌ بود،
            low_map[i] = val   # مقدار کف پنجره را در همین اندکس از آرایه جواب ذخیره کن.

        # HIGH
        val = high_vals[i - (depth - 1)]
        if (val - high[i]) <= deviation:
            back_range = slice(max(0, i - backstep), i)
            mask = (high_map[back_range] != 0) & (high_map[back_range] < val)
            idx = np.where(mask)[0] + back_range.start
            high_map[idx] = 0.0
        if high[i] == val:
            high_map[i] = val

    # ---------------------------------
    # phase-2: resolve zigzag + developing (retrospective confirmed_at)
    # ---------------------------------
    last_high = 0.0
    last_low = 0.0
    last_high_pos = -1
    last_low_pos = -1
    search_mode = 0  # 0=Extremum, 1=Search next peak, -1=Search next bottom

    for i in range(depth - 1, n):
        l_val = low_map[i]
        h_val = high_map[i]

        # developing_leg
        if search_mode == 1: developing_leg[i] = 1
        elif search_mode == -1: developing_leg[i] = -1

        if search_mode == 0:
            if last_high == 0.0 and h_val != 0.0:
                last_high = h_val
                last_high_pos = i
                zz_buffer[i] = h_val
                state_series[i] = -1
                search_mode = -1
            elif last_low == 0.0 and l_val != 0.0:
                last_low = l_val
                last_low_pos = i
                zz_buffer[i] = l_val
                state_series[i] = 1
                search_mode = 1

        elif search_mode == 1:  # Peak -> دنبال Low
            if l_val != 0.0 and l_val < last_low and h_val == 0.0:
                if last_low_pos >= 0:
                    confirmed_at[last_low_pos] = i
                    zz_buffer[last_low_pos] = 0.0
                    state_series[last_low_pos] = 0
                last_low = l_val
                last_low_pos = i
                zz_buffer[i] = l_val
                state_series[i] = 1
            elif h_val != 0.0:
                if last_low_pos >= 0:
                    confirmed_at[last_low_pos] = i
                last_high = h_val
                last_high_pos = i
                zz_buffer[i] = h_val
                state_series[i] = -1
                search_mode = -1

        elif search_mode == -1:  # Bottom -> دنبال High
            if h_val != 0.0 and h_val > last_high and l_val == 0.0:
                if last_high_pos >= 0:
                    confirmed_at[last_high_pos] = i
                    zz_buffer[last_high_pos] = 0.0
                    state_series[last_high_pos] = 0
                last_high = h_val
                last_high_pos = i
                zz_buffer[i] = h_val
                state_series[i] = -1
            elif l_val != 0.0:
                if last_high_pos >= 0:
                    confirmed_at[last_high_pos] = i
                last_low = l_val
                last_low_pos = i
                zz_buffer[i] = l_val
                state_series[i] = 1
                search_mode = 1

    high_actual = np.where(state_series == -1, high, 0)
    low_actual  = np.where(state_series == +1, low , 0)

    return state_series, high_actual, low_actual, confirmed_at, developing_leg
    # return state_series, high_actual, low_actual

#====================================================================
# Loop-wise and njit
#==================================================================== Func2
def _zigzag_mql_njit_loopwise_complete(
    high: np.ndarray,
    low: np.ndarray,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 3,
    # point: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    high_arr = high.astype(np.float64)
    low_arr  = low.astype(np.float64)

    @njit(cache=True)
    def _zigzag_core(high_arr, low_arr, depth, deviation, backstep):  #, point):
        n = high_arr.shape[0]

        low_map     = np.zeros(n, dtype=np.float64)
        high_map    = np.zeros(n, dtype=np.float64)
        zz_buffer   = np.zeros(n, dtype=np.float64)
        state       = np.zeros(n, dtype=np.int8)
        confirmed_at = np.full(n, -1, dtype=np.int32)
        developing_leg = np.zeros(n, dtype=np.int8)

        # -----------------------------
        # Stage 1: local extrema
        # -----------------------------
        for i in range(depth - 1, n):
            # LOW
            lo_val = low_arr[i]
            lo_idx = i
            for j in range(i - 1, i - depth, -1):
                if j < 0: break
                if low_arr[j] < lo_val:
                    lo_val = low_arr[j]
                    lo_idx = j

            if (low_arr[i] - lo_val) <= deviation:  # * point:
                for b in range(1, backstep + 1):
                    j = i - b
                    if j >= 0 and low_map[j] != 0.0 and low_map[j] > lo_val:
                        low_map[j] = 0.0
            else:
                lo_val = 0.0

            if low_arr[i] == lo_val:
                low_map[i] = lo_val

            # HIGH
            hi_val = high_arr[i]
            hi_idx = i
            for j in range(i - 1, i - depth, -1):
                if j < 0: break
                if high_arr[j] > hi_val:
                    hi_val = high_arr[j]
                    hi_idx = j

            if (hi_val - high_arr[i]) <= deviation:  # * point:
                for b in range(1, backstep + 1):
                    j = i - b
                    if j >= 0 and high_map[j] != 0.0 and high_map[j] < hi_val:
                        high_map[j] = 0.0
            else:
                hi_val = 0.0

            if high_arr[i] == hi_val:
                high_map[i] = hi_val

        # -----------------------------
        # Stage 2: zigzag resolve + developing + retrospective confirmed_at
        # -----------------------------
        last_high = 0.0
        last_low = 0.0
        last_hp = -1
        last_lp = -1
        mode = 0  # 0=Extremum, 1=Search Low, -1=Search High

        for i in range(depth - 1, n):
            # developing_leg
            if mode == 1:
                developing_leg[i] = 1
            elif mode == -1:
                developing_leg[i] = -1

            if mode == 0:
                if last_high == 0.0 and high_map[i] != 0.0:
                    last_high = high_map[i]
                    last_hp = i
                    zz_buffer[i] = last_high
                    state[i] = -1
                    mode = -1
                elif last_low == 0.0 and low_map[i] != 0.0:
                    last_low = low_map[i]
                    last_lp = i
                    zz_buffer[i] = last_low
                    state[i] = 1
                    mode = 1

            elif mode == 1:
                if low_map[i] != 0.0 and low_map[i] < last_low and high_map[i] == 0.0:
                    if last_lp >= 0:
                        confirmed_at[last_lp] = i
                        zz_buffer[last_lp] = 0.0
                        state[last_lp] = 0
                    last_low = low_map[i]
                    last_lp = i
                    zz_buffer[i] = last_low
                    state[i] = 1
                elif high_map[i] != 0.0:
                    if last_lp >= 0:
                        confirmed_at[last_lp] = i
                    last_high = high_map[i]
                    last_hp = i
                    zz_buffer[i] = last_high
                    state[i] = -1
                    mode = -1

            elif mode == -1:
                if high_map[i] != 0.0 and high_map[i] > last_high and low_map[i] == 0.0:
                    if last_hp >= 0:
                        confirmed_at[last_hp] = i
                        zz_buffer[last_hp] = 0.0
                        state[last_hp] = 0
                    last_high = high_map[i]
                    last_hp = i
                    zz_buffer[i] = last_high
                    state[i] = -1
                elif low_map[i] != 0.0:
                    if last_hp >= 0:
                        confirmed_at[last_hp] = i
                    last_low = low_map[i]
                    last_lp = i
                    zz_buffer[i] = last_low
                    state[i] = 1
                    mode = 1

        return state, zz_buffer, high_map, low_map, confirmed_at, developing_leg

    state, zz, hmap, lmap, confirmed_at, developing_leg = _zigzag_core(
        high_arr, low_arr, depth, deviation, backstep  #, point
    )

    high_actual = np.zeros_like(high_arr)
    low_actual  = np.zeros_like(low_arr)

    for i in range(state.shape[0]):
        if state[i] == -1:
            high_actual[i] = high_arr[i]
        elif state[i] == 1:
            low_actual[i] = low_arr[i]

    return state, high_actual, low_actual, confirmed_at, developing_leg
    # return state, high_actual, low_actual

#====================================================================
# Wrapper function to choose between njit and non-njit based on data size
#==================================================================== Func3
def zigzag_wrapper(
    high: pd.Series,
    low: pd.Series,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    # point: float = 0.01,
    addmeta: bool = True,
    final_check: bool = True,   # Added: 1404/12/01
) -> pd.DataFrame:
    
    n = len(high)

    idx = high.index
    high_np = np.ascontiguousarray(high.values, dtype=np.float64)
    low_np  = np.ascontiguousarray(low.values,  dtype=np.float64)
    bytes_used = high_np.nbytes + low_np.nbytes

    # Using "bytes_used <= 1_280_000"  isntead of "len(high_np) <= 80_000"
    if bytes_used <= 1_280_000:
        state, high_actual, low_actual, confirmed_at, developing_leg = \
            _zigzag_mql_numpy_complete(
                # high_np, low_np, depth, deviation, backstep, point)
                high_np, low_np, depth, deviation, backstep)

    else:
        state, high_actual, low_actual, confirmed_at, developing_leg = \
            _zigzag_mql_njit_loopwise_complete(
                # high_np, low_np, depth, deviation, backstep, point)
                high_np, low_np, depth, deviation, backstep)

    # --- state correction ----------------------------------------------------
    # از اینجا به بعد وضعیت سوئینگ را از (جستجوی آینده) به (وضعیت فعلی) اصلاح میکنیم
    # یعنی در سقفها برابر با +1 است و در کفها برابر با -1 است    
    state = -1 * state
    
    # --- main result of this function ----------------------------------------
    zz_df = pd.DataFrame(
        index=idx,
        data={
            "state": state.astype(np.int8),        # at HIGHs: state = +1, at LOWs: state = -1
            "high_zz" : high_actual.astype(float),
            "low_zz"  : low_actual.astype(float),
            "confirmed_at": confirmed_at.astype(np.int32),
            "developing_leg": developing_leg.astype(np.int8),
        }
    )
    # --- build zigzag legs metadata ------------------------------------------
    # leg = فاصله بین دو اکسترمم متوالی با جهت مشخص
    legs = []
    # precompute positional map (O(1) lookup)
    pos_map = {ts: i for i, ts in enumerate(idx)}

    last_idx = None
    last_state = 0
    for idx_val, state_val in zip(idx, state):
        if state_val == 0:
            continue

        if last_idx is not None:
            start_pos = pos_map[last_idx]
            end_pos   = pos_map[idx_val]
            price_st = high_np[start_pos] if last_state == 1 else low_np[start_pos]
            price_ed = high_np[end_pos]   if state_val  == 1 else low_np[end_pos]

            if end_pos > start_pos:
                legs.append({
                    "start_ts": last_idx,               # timestamp
                    "end_ts": idx_val,                  # timestamp
                    "direction": int(state_val),        # -1/+1
                    "start_extr": price_st,             # price at "start extremum"
                    "end_extr":   price_ed,             # price at "end extremum"
                    "price_diff": price_ed - price_st, 
                    "start_pos": int(start_pos),                  # pos = position
                    "end_pos": int(end_pos),                      # pos = position
                    "start_confirmed_at": int(confirmed_at[start_pos]),          # pos
                    "end_confirmed_at": int(confirmed_at[end_pos]),              # pos
                    "start_developing_leg": int(developing_leg[start_pos]),      # pos
                    "end_developing_leg": int(developing_leg[end_pos]),          # pos
                })

        last_idx = idx_val
        last_state = state_val
    
    # --- final check at wrapper ---------------------------------------------- added 1404/12/01
    if (legs is not None) and (final_check==True):
        merged_legs = []
        temp_leg = legs[0]

        for leg in legs[1:]:
            # اگر اختلاف قیمت (لگ موقت) هم‌علامت است با (لگ قبلی)، ادغام صورت بگیرد
            if np.sign(leg["price_diff"]) == np.sign(temp_leg["price_diff"]):
                
                # --- اصلاح خروجی گرفته شده از تابع زیگزاگ ---
                e = temp_leg["end_pos"]
                zz_df.loc[zz_df.index[e], "state"] = 0
                zz_df.loc[zz_df.index[e], "high_zz"] = 0
                zz_df.loc[zz_df.index[e], "low_zz"] = 0

                # --- بسط لگ فعلی: end_ts، end_extr، end_pos، price_diff ---
                temp_leg["end_ts"] = leg["end_ts"]
                temp_leg["end_extr"] = leg["end_extr"]
                temp_leg["end_pos"] = leg["end_pos"]
                temp_leg["price_diff"] = temp_leg["end_extr"] - temp_leg["start_extr"]
                temp_leg["direction"] = np.sign(temp_leg["price_diff"])  # این سطر شاید لازم نباشد
            else:
                merged_legs.append(temp_leg)
                temp_leg = leg

        merged_legs.append(temp_leg)  # آخرین لگ
        legs = merged_legs

    # --- attach metadata -----------------------------------------------------
    if not addmeta:
        return zz_df
    
    zz_df.attrs["legs"] = legs
    return zz_df

#====================================================================
# Multi-Timeframe zigzag adapter
"""
- Computes ZigZag on higher timeframe (HTF)
- Projects HTF swing points onto lower timeframe (LTF)

Parameters
----------
high, low : pd.Series
    LTF price series (DatetimeIndex required)
tf_higher : str
    Pandas resample rule (e.g. '5min', '15min', '1h', '4h', '1d')
mode : 'last' | 'forward_fill'
    Projection mode to LTF
    'last':         → Marks the swing only at the first LTF candle where the HTF swing is confirmed.
                (فقط نقطه شروع swing در LTF علامت می‌خورد)
    'forward_fill': → Propagates the HTF swing signal forward on all LTF candles until the next HTF swing appears.
                (سیگنال swing در تمام کندل‌های LTF تا ظهور swing بعدی ادامه می‌یابد)
    
Returns
-------
pd.Series
    MTF-aware ZigZag aligned to LTF index
"""
#==================================================================== Func4
def zigzag_mtf_adapter(
    high: pd.Series,
    low: pd.Series,
    tf_higher: str,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    # point: float = 0.01,
    mode: Literal["last", "forward_fill"] = "forward_fill",
    extend_last_leg: bool = False,  # True: فرض میکند که آخرین لگ تا انتهای دیتاها ادامه می یابد
    use_timeshift: bool = False,
) -> pd.Series:
    
    # --- Validation ---------------------------------------------------------- ok1
    if not isinstance(high.index, pd.DatetimeIndex) or not isinstance(low.index, pd.DatetimeIndex):
        raise ValueError("high/low must have DatetimeIndex")
    
    if not high.index.equals(low.index):
        raise ValueError("high and low must share identical index")
    
    # --- Build HTF OHLC ------------------------------------------------------ ok2
    htf = (
        pd.DataFrame({"high": high, "low": low})
            .resample(tf_higher)
            .agg({"high": "max", "low": "min"})
            .dropna()
    )
    if htf.empty:
        raise ValueError("HTF resample produced empty dataframe")
    
    # --- Run original ZigZag on HTF ------------------------------------------ ok3
    zz_htf = zigzag_wrapper(
        high=htf["high"],
        low=htf["low"],
        depth=depth, deviation=deviation, backstep=backstep, # point=point,
        addmeta=True,
        final_check=True,
    )
    legs_htf = zz_htf.attrs.get("legs", [])
    if len(legs_htf) == 0:
        return pd.Series(
            data=0.0,
            index=high.index,
            name=f"zigzag_mtf_{tf_higher}",
            dtype=np.float32,
        )
   
    # --- For debuging --------------------------------------------------------
    # (تابع ریسِت ایندکس سبب ایجاد تغییرات دائمی روی دیتافریم نمی شود)
    #
    # zz_htf.reset_index().to_csv("1__zz_htf.csv", index_label="no.")
    # pd.DataFrame(legs_htf).to_csv("2__legs_htf.csv", index_label="no.")
    
    # --- Prepare LTF container ----------------------------------------------- ok4
    # build an empty series with known time index, for "final main result"
    zz_ltf = pd.Series(
        data=0.0,
        index=high.index,
        name=f"zigzag_mtf_{tf_higher}",
        dtype=np.float32
    )
    
    ltf_index = zz_ltf.index
    n = len(ltf_index)
    ltf_legs =[]

    # --- Time shift ----------------------------------------------------------
    # طول HTF
    tf_htf = pd.to_timedelta(tf_higher)

    if use_timeshift:
        # فاصله واقعی LTF
        tf_ltf = ltf_index.to_series().diff().dropna().iloc[0]
        # انتقال از open به آخرین LTF داخل بازه
        timeshift = tf_htf - tf_ltf
    else:
        tf_ltf = pd.Timedelta(0)
        timeshift = pd.Timedelta(0)

    # --- last mode -----------------------------------------------------------
    if mode == "last":  
        zz_ltf.name = f"{zz_ltf.name}_status"
        for leg in legs_htf:

            start_ts = pd.to_datetime(leg["start_ts"], utc=True) + timeshift
            end_ts   = pd.to_datetime(leg["end_ts"],   utc=True) + timeshift

            direction = int(leg["direction"])

            # --- causal mapping (بدون شیفت آینده) ---
            ltf_start_pos = ltf_index.searchsorted(start_ts, side="right") - 1
            ltf_end_pos   = ltf_index.searchsorted(end_ts,   side="right") - 1

            # --- clamp to valid bounds ---
            if ltf_start_pos < 0:
                ltf_start_pos = 0
            if ltf_end_pos >= n:
                ltf_end_pos = n - 1
            if ltf_end_pos < ltf_start_pos:
                continue

            # --- apply last mode ---
            zz_ltf.iloc[ltf_start_pos] = -1 * direction
            if ltf_end_pos < n:
                zz_ltf.iloc[ltf_end_pos] = direction

            # --- for (legs) ---
            _start_ts = ltf_index[ltf_start_pos] + tf_ltf
            _end_ts   = ltf_index[ltf_end_pos  ] + tf_ltf
            _start_pos = ltf_index.searchsorted(_start_ts, side="right") - 1
            _end_pos   = ltf_index.searchsorted(_end_ts  , side="right") - 1
            # --- correcting probability errors ---
            _start_ts = ltf_index[_start_pos]
            _end_ts   = ltf_index[_end_pos  ]

            # --- store mapped leg metadata ---
            ltf_legs.append({
                "ltf_start_ts": _start_ts,
                "ltf_end_ts": _end_ts,
                "direction": direction,
                
                "ltf_start_extr": leg["start_extr"],
                "ltf_end_extr":   leg["end_extr"],
                # "ltf_price_diff": leg["price_diff"], 
                
                "ltf_start_pos": int(_start_pos),
                "ltf_end_pos": int(_end_pos),
            })
    
    # --- forward_fill mode ---------------------------------------------------
    if mode == "forward_fill":
        zz_ltf.name = f"{zz_ltf.name}_direction"
        for leg in legs_htf:

            start_ts = pd.to_datetime(leg["start_ts"], utc=True) + timeshift
            end_ts   = pd.to_datetime(leg["end_ts"],   utc=True) + timeshift

            direction = int(leg["direction"])

            # --- causal mapping (بدون شیفت آینده) ---
            ltf_start_pos = ltf_index.searchsorted(start_ts, side="right") - 1
            ltf_end_pos   = ltf_index.searchsorted(end_ts,   side="right") - 1

            # --- clamp to valid bounds ---
            if ltf_start_pos < 0:
                ltf_start_pos = 0
            if ltf_end_pos >= n:
                ltf_end_pos = n - 1
            if ltf_end_pos < ltf_start_pos:
                continue

            # --- for (forward_fill mode) & (legs) ---
            _start_ts = ltf_index[ltf_start_pos] + tf_ltf
            _end_ts   = ltf_index[ltf_end_pos  ] + tf_ltf
            _start_pos = ltf_index.searchsorted(_start_ts, side="right") - 1
            _end_pos   = ltf_index.searchsorted(_end_ts  , side="right") - 1
            # --- correcting probability errors ---
            _start_ts = ltf_index[_start_pos]
            _end_ts   = ltf_index[_end_pos  ]

            # --- apply forward_fill mode ---
            if mode == "forward_fill":
                zz_ltf.iloc[_start_pos : _end_pos] = direction

            # --- store mapped leg metadata ---
            ltf_legs.append({
                "ltf_start_ts": _start_ts,
                "ltf_end_ts": _end_ts,
                "direction": direction,
                
                "ltf_start_extr": leg["start_extr"],
                "ltf_end_extr":   leg["end_extr"],
                # "ltf_price_diff": leg["price_diff"], 
                
                "ltf_start_pos": int(_start_pos),
                "ltf_end_pos": int(_end_pos),
            })

        # --- Extend last leg if requested ---
        if extend_last_leg and ltf_legs:
            last = ltf_legs[-1]
            zz_ltf.iloc[last["ltf_end_pos"]:] = last["direction"]

    # --- Attach metadata -----------------------------------------------------
    zz_ltf.attrs["legs"] = ltf_legs

    return zz_ltf

#====================================================================
# Zigzag Legs
#==================================================================== Func5
def zigzag_legs(
    high: pd.Series,
    low: pd.Series,
    tf: str,   # این تایم فریم باید از تایم فریم مربوط به داده ها بزرگتر باشد
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    # point: float = 0.01,
    # mode: Literal["last", "forward_fill"] = "forward_fill",
    extend_last_leg: bool = False,
    use_timeshift: bool = False,
) -> pd.DataFrame:
    """
    Returns ZigZag legs in unified LTF schema.

    Output columns:
        ltf_start_ts
        ltf_end_ts
        direction
        ltf_start_extr
        ltf_end_extr
        ltf_start_pos
        ltf_end_pos
    """

    required_cols = [
        "ltf_start_ts",
        "ltf_end_ts",
        "direction",
        "ltf_start_extr",
        "ltf_end_extr",
        "ltf_start_pos",
        "ltf_end_pos",
    ]

    # --- validation ------------------------------------------------
    if not isinstance(high.index, pd.DatetimeIndex) or not isinstance(low.index, pd.DatetimeIndex):
        raise ValueError("high/low must have DatetimeIndex")

    if not high.index.equals(low.index):
        raise ValueError("high and low must share identical index")

    if len(high) < 2:
        return pd.DataFrame(columns=required_cols)

    # --- timeframe detection ---------------------------------------
    tf_df = high.index.to_series().diff().dropna().iloc[0]
    tf_df = pd.to_timedelta(tf_df)
    tf_td = pd.to_timedelta(tf)

    # --- lower TF requested (invalid) ------------------------------
    if tf_td < tf_df:
        return pd.DataFrame(columns=required_cols)

    # --- equal TF --------------------------------------------------
    if tf_td == tf_df:
        zz = zigzag_wrapper(
            high=high,
            low=low,
            depth=depth,
            deviation=deviation,
            backstep=backstep,
            # point=point,
            addmeta=True,
            final_check=True,
        )

        legs = zz.attrs.get("legs", [])
        if not legs:
            return pd.DataFrame(columns=required_cols)

        mapped = []
        for leg in legs:
            mapped.append({
                "ltf_start_ts": leg["start_ts"],
                "ltf_end_ts": leg["end_ts"],
                "direction": int(leg["direction"]),
                "ltf_start_extr": leg["start_extr"],
                "ltf_end_extr": leg["end_extr"],
                "ltf_start_pos": int(leg["start_pos"]),
                "ltf_end_pos": int(leg["end_pos"]),
            })

        return pd.DataFrame(mapped, columns=required_cols)

    # --- higher TF ---
    zz_mtf = zigzag_mtf_adapter(
        high=high,
        low=low,
        tf_higher=tf,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        # point=point,
        # mode=mode,
        extend_last_leg=extend_last_leg,
        use_timeshift=use_timeshift,
    )

    legs = zz_mtf.attrs.get("legs", [])
    if not legs:
        return pd.DataFrame(columns=required_cols)

    return pd.DataFrame(legs, columns=required_cols)


# =====================================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# =====================================================================================
""" Func Names                           Used in Functions: ...
                                        1   2   3   4   5   6   7   8   9  10
1  _zigzag_mql_numpy_complete          --  --  ok  --  --  --  --  --  --  --
2  _zigzag_mql_njit_loopwise_complete  --  --  ok  --  --  --  --  --  --  --
3  zigzag                              --  --  --  ok  ok  --  --  --  --  -- For Internal, External Use
4  zigzag_mtf_adapter                  --  --  --  --  ok  --  --  --  --  -- For Internal, External Use
5  zigzag_legs                         --  --  --  --  --  --  --  --  --  -- For External Use
"""


#########################################################################################
# NEW ZIGZAG FILE
#########################################################################################

#####################################################################
# Layer 1 — New Zigzag Core
#####################################################################

#================================================ func-1
# Numpy function
#================================================
def _zigzag_numpy(
    high: np.ndarray,
    low: np.ndarray,
    depth: int = 12,
    deviation: float = 0.05,
    backstep: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    n = len(high)

    zz_buffer    = np.zeros(n)
    state_series = np.zeros(n, dtype=np.int8)

    confirmed_at   = np.full(n, -1, dtype=np.int32)
    developing_leg = np.zeros(n, dtype=np.int8)

    # -----------------------------
    # phase 1 : candidate detection
    # -----------------------------
    low_windows  = np.lib.stride_tricks.sliding_window_view(low, depth)
    high_windows = np.lib.stride_tricks.sliding_window_view(high, depth)

    low_idx_in_window  = np.argmin(low_windows, axis=1)
    high_idx_in_window = np.argmax(high_windows, axis=1)

    base = np.arange(depth - 1, n) - (depth - 1)

    low_vals  = low[base + low_idx_in_window]
    high_vals = high[base + high_idx_in_window]

    low_map  = np.zeros(n)
    high_map = np.zeros(n)

    for i in range(depth - 1, n):

        l_val = low_vals[i - (depth - 1)]
        if (low[i] - l_val) <= deviation:
            r = slice(max(0, i - backstep), i)
            m = (low_map[r] != 0) & (low_map[r] > l_val)
            low_map[np.where(m)[0] + r.start] = 0.0
        if low[i] == l_val:
            low_map[i] = l_val

        h_val = high_vals[i - (depth - 1)]
        if (h_val - high[i]) <= deviation:
            r = slice(max(0, i - backstep), i)
            m = (high_map[r] != 0) & (high_map[r] < h_val)
            high_map[np.where(m)[0] + r.start] = 0.0
        if high[i] == h_val:
            high_map[i] = h_val

    # -----------------------------
    # phase 2 : resolve
    # -----------------------------
    last_high = 0.0
    last_low = 0.0
    last_high_pos = -1
    last_low_pos = -1

    search_mode = 0

    for i in range(depth - 1, n):

        l_cand = low_map[i]
        h_cand = high_map[i]

        if search_mode == 1:
            developing_leg[i] = 1
        elif search_mode == -1:
            developing_leg[i] = -1

        # -----------------
        # initial pivot
        # -----------------
        if search_mode == 0:

            if h_cand != 0:
                last_high = h_cand
                last_high_pos = i

                zz_buffer[i] = h_cand
                state_series[i] = -1
                search_mode = -1

            elif l_cand != 0:
                last_low = l_cand
                last_low_pos = i

                zz_buffer[i] = l_cand
                state_series[i] = 1
                search_mode = 1

        # -----------------
        # searching LOW
        # -----------------
        elif search_mode == 1:

            if l_cand != 0 and (last_low == 0 or l_cand < last_low):

                # حذف extreme قبلی همان لگ
                if last_low_pos >= 0:
                    zz_buffer[last_low_pos] = 0
                    state_series[last_low_pos] = 0

                last_low = l_cand
                last_low_pos = i

                zz_buffer[i] = l_cand
                state_series[i] = 1


            if h_cand != 0:

                if last_low_pos >= 0:
                    confirmed_at[last_low_pos] = i

                last_high = h_cand
                last_high_pos = i

                zz_buffer[i] = h_cand
                state_series[i] = -1

                search_mode = -1

        # -----------------
        # searching HIGH
        # -----------------
        elif search_mode == -1:

            if h_cand != 0 and (last_high == 0 or h_cand > last_high):

                if last_high_pos >= 0:
                    zz_buffer[last_high_pos] = 0
                    state_series[last_high_pos] = 0

                last_high = h_cand
                last_high_pos = i

                zz_buffer[i] = h_cand
                state_series[i] = -1


            if l_cand != 0:

                if last_high_pos >= 0:
                    confirmed_at[last_high_pos] = i

                last_low = l_cand
                last_low_pos = i

                zz_buffer[i] = l_cand
                state_series[i] = 1

                search_mode = 1

    high_actual = np.where(state_series == -1, high, 0)
    low_actual  = np.where(state_series == 1 , low , 0)

    return state_series, high_actual, low_actual, confirmed_at, developing_leg


#================================================ func-2
# Numba function
#================================================
@njit(cache=True, fastmath=True)
def _zigzag_njit(
    high: np.ndarray,
    low: np.ndarray,
    depth: int = 12,
    deviation: float = 0.05,
    backstep: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    n = len(high)

    zz_buffer = np.zeros(n)
    state_series = np.zeros(n, dtype=np.int8)

    confirmed_at = np.full(n, -1, dtype=np.int32)
    developing_leg = np.zeros(n, dtype=np.int8)

    low_map = np.zeros(n)
    high_map = np.zeros(n)

    # -------------------------------------------------
    # phase 1 : candidate detection (manual sliding window)
    # -------------------------------------------------
    for i in range(depth - 1, n):

        start = i - depth + 1

        min_low = low[start]
        max_high = high[start]

        for j in range(start + 1, i + 1):
            if low[j] < min_low:
                min_low = low[j]

            if high[j] > max_high:
                max_high = high[j]

        # LOW candidate
        if (low[i] - min_low) <= deviation:

            b_start = i - backstep
            if b_start < 0:
                b_start = 0

            for k in range(b_start, i):
                if low_map[k] != 0 and low_map[k] > min_low:
                    low_map[k] = 0.0

        if low[i] == min_low:
            low_map[i] = min_low

        # HIGH candidate
        if (max_high - high[i]) <= deviation:

            b_start = i - backstep
            if b_start < 0:
                b_start = 0

            for k in range(b_start, i):
                if high_map[k] != 0 and high_map[k] < max_high:
                    high_map[k] = 0.0

        if high[i] == max_high:
            high_map[i] = max_high

    # -------------------------------------------------
    # phase 2 : resolve zigzag
    # -------------------------------------------------

    last_high = 0.0
    last_low = 0.0

    last_high_pos = -1
    last_low_pos = -1

    search_mode = 0

    for i in range(depth - 1, n):

        l_cand = low_map[i]
        h_cand = high_map[i]

        if search_mode == 1:
            developing_leg[i] = 1
        elif search_mode == -1:
            developing_leg[i] = -1

        # ---------------------------------
        # initial pivot
        # ---------------------------------
        if search_mode == 0:

            if h_cand != 0.0:

                last_high = h_cand
                last_high_pos = i

                zz_buffer[i] = h_cand
                state_series[i] = -1

                search_mode = -1

            elif l_cand != 0.0:

                last_low = l_cand
                last_low_pos = i

                zz_buffer[i] = l_cand
                state_series[i] = 1

                search_mode = 1

        # ---------------------------------
        # searching LOW
        # ---------------------------------
        elif search_mode == 1:

            # improving low
            if l_cand != 0.0 and (last_low == 0.0 or l_cand < last_low):

                if last_low_pos >= 0:
                    zz_buffer[last_low_pos] = 0.0
                    state_series[last_low_pos] = 0

                last_low = l_cand
                last_low_pos = i

                zz_buffer[i] = l_cand
                state_series[i] = 1

            # RAW confirmation: only if opposite candidate appears
            if h_cand != 0.0:

                if last_low_pos >= 0:
                    confirmed_at[last_low_pos] = i

                last_high = h_cand
                last_high_pos = i

                zz_buffer[i] = h_cand
                state_series[i] = -1

                search_mode = -1

        # ---------------------------------
        # searching HIGH
        # ---------------------------------
        elif search_mode == -1:

            #  improving high
            if h_cand != 0.0 and (last_high == 0.0 or h_cand > last_high):

                if last_high_pos >= 0:
                    zz_buffer[last_high_pos] = 0.0
                    state_series[last_high_pos] = 0

                last_high = h_cand
                last_high_pos = i

                zz_buffer[i] = h_cand
                state_series[i] = -1

            # RAW confirmation: only if opposite candidate appears
            if l_cand != 0.0:

                if last_high_pos >= 0:
                    confirmed_at[last_high_pos] = i

                last_low = l_cand
                last_low_pos = i

                zz_buffer[i] = l_cand
                state_series[i] = 1

                search_mode = 1

    # --------------------------------------
    # generate high_actual / low_actual
    # --------------------------------------
    high_actual = np.zeros(n)
    low_actual = np.zeros(n)

    for i in range(n):
        if state_series[i] == -1:
            high_actual[i] = high[i]
        elif state_series[i] == 1:
            low_actual[i] = low[i]

    return state_series, high_actual, low_actual, confirmed_at, developing_leg


#================================================ func-3
# Wrapper function
#================================================
def zigzag_new_wrapper(
    high: pd.Series,
    low: pd.Series,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    # ---
    soft_confirmation: bool = True,
    atr: pd.Series = None,
    deviation_soft_mult: float = 1.0,
    time_threshold: int = 10,
    atr_mult: float = 1.5,
    # ---
    addmeta: bool = True,
    final_check: bool = False, # لازم است بخش مربوط به این، به قبل از (تاییدهای نرم) منتقل بشود
        # همچنین فعلاً برای جلوگیری از دستکاری گذشته، باداشتن اطلاعات آینده، بهتر است اعمال نشود
) -> pd.DataFrame:
    
    n = len(high)

    idx = high.index
    high_np = np.ascontiguousarray(high.values, dtype=np.float64)
    low_np  = np.ascontiguousarray(low.values,  dtype=np.float64)
    bytes_used = high_np.nbytes + low_np.nbytes

    # Using "bytes_used <= 1_280_000"  isntead of "len(high_np) <= 80_000"
    if bytes_used <= 1_280_000:
        state, high_actual, low_actual, confirmed_at, developing_leg = \
            _zigzag_numpy(high_np, low_np, depth, deviation, backstep)
    else:
        state, high_actual, low_actual, confirmed_at, developing_leg = \
            _zigzag_njit(high_np, low_np, depth, deviation, backstep)

    # --- soft confirmation ---------------------------------------------------
    if soft_confirmation:
        sc_deviation, sc_time, sc_structure, sc_atr = \
            zigzag_layer2_soft_confirmations(
                high=high,
                low=low,
                state_series=state,
                high_pivot=high_actual,
                low_pivot=low_actual,
                # ---
                atr=atr,
                deviation_soft_mult = deviation_soft_mult,
                time_threshold=time_threshold,
                atr_mult=atr_mult
            )
    # --- state correction ----------------------------------------------------
    # از اینجا به بعد وضعیت سوئینگ را از (جستجوی آینده) به (وضعیت فعلی) اصلاح میکنیم
    # یعنی در سقفها برابر با +1 است و در کفها برابر با -1 است    
    state = -1 * state
    
    # --- main result of this function ----------------------------------------
    zz_df = pd.DataFrame(
        index=idx,
        data={
            "state": state.astype(np.int8),        # at HIGHs: state = +1, at LOWs: state = -1
            "high_zz" : high_actual.astype(float),
            "low_zz"  : low_actual.astype(float),
            "confirmed_at": confirmed_at.astype(np.int32),
            "developing_leg": developing_leg.astype(np.int8),

            "sc_deviation": sc_deviation,
            "sc_time": sc_time,
            "sc_structure": sc_structure,
            "sc_atr": sc_atr,
        }
    )
    # --- build zigzag legs metadata ------------------------------------------
    # leg = فاصله بین دو اکسترمم متوالی با جهت مشخص
    legs = []
    # precompute positional map (O(1) lookup)
    pos_map = {ts: i for i, ts in enumerate(idx)}

    last_idx = None
    last_state = 0
    for idx_val, state_val in zip(idx, state):
        if state_val == 0:
            continue

        if last_idx is not None:
            start_pos = pos_map[last_idx]
            end_pos   = pos_map[idx_val]
            price_st = high_np[start_pos] if last_state == 1 else low_np[start_pos]
            price_ed = high_np[end_pos]   if state_val  == 1 else low_np[end_pos]

            if end_pos > start_pos:
                legs.append({
                    "start_ts": last_idx,               # timestamp
                    "end_ts": idx_val,                  # timestamp
                    "direction": int(state_val),        # -1/+1
                    "start_extr": price_st,             # price at "start extremum"
                    "end_extr":   price_ed,             # price at "end extremum"
                    "price_diff": price_ed - price_st, 
                    "start_pos": int(start_pos),                  # pos = position
                    "end_pos": int(end_pos),                      # pos = position
                    "start_confirmed_at": int(confirmed_at[start_pos]),          # pos
                    "end_confirmed_at": int(confirmed_at[end_pos]),              # pos
                    "start_developing_leg": int(developing_leg[start_pos]),      # pos
                    "end_developing_leg": int(developing_leg[end_pos]),          # pos
                })

        last_idx = idx_val
        last_state = state_val
    
    # --- final check at wrapper ---------------------------------------------- added 1404/12/01
    if (legs is not None) and (final_check == True):
        merged_legs = []
        temp_leg = legs[0]

        for leg in legs[1:]:
            # اگر اختلاف قیمت (لگ موقت) هم‌علامت است با (لگ قبلی)، ادغام صورت بگیرد
            if np.sign(leg["price_diff"]) == np.sign(temp_leg["price_diff"]):
                
                # --- اصلاح خروجی گرفته شده از تابع زیگزاگ ---
                e = temp_leg["end_pos"]
                zz_df.loc[zz_df.index[e], "state"] = 0
                zz_df.loc[zz_df.index[e], "high_zz"] = 0
                zz_df.loc[zz_df.index[e], "low_zz"] = 0

                # --- بسط لگ فعلی: end_ts، end_extr، end_pos، price_diff ---
                temp_leg["end_ts"] = leg["end_ts"]
                temp_leg["end_extr"] = leg["end_extr"]
                temp_leg["end_pos"] = leg["end_pos"]
                temp_leg["price_diff"] = temp_leg["end_extr"] - temp_leg["start_extr"]
                temp_leg["direction"] = np.sign(temp_leg["price_diff"])  # این سطر شاید لازم نباشد
            else:
                merged_legs.append(temp_leg)
                temp_leg = leg

        merged_legs.append(temp_leg)  # آخرین لگ
        legs = merged_legs

    # --- attach metadata -----------------------------------------------------
    if not addmeta:
        return zz_df
    
    zz_df.attrs["legs"] = legs
    return zz_df


#####################################################################
# Layer 2 — Soft Confirmation Engine
##################################################################### func-4
def zigzag_layer2_soft_confirmations(
    high: np.ndarray,
    low: np.ndarray,
    # close: np.ndarray,

    state_series: np.ndarray,
    high_pivot: np.ndarray,
    low_pivot: np.ndarray,
    # confirmed_at: np.ndarray,
    # developing_leg: np.ndarray,

    atr: np.ndarray | None = None,

    deviation_soft_mult: float = 1.0,
    time_threshold: int = 10,
    atr_mult: float = 1.5
):

    n = len(high)

    sc_deviation = np.zeros(n, dtype=np.int8)
    sc_time      = np.zeros(n, dtype=np.int8)
    sc_structure = np.zeros(n, dtype=np.int8)
    sc_atr       = np.zeros(n, dtype=np.int8)

    last_pivot_price = None
    last_pivot_index = -1
    last_pivot_type  = 0

    prev_high_pivot = None
    prev_low_pivot  = None

    for i in range(n):

        # ---------------------
        # detect new pivot
        # ---------------------
        if state_series[i] == -1:

            prev_high_pivot = high_pivot[i]

            last_pivot_price = high_pivot[i]
            last_pivot_index = i
            last_pivot_type  = -1

        elif state_series[i] == 1:

            prev_low_pivot = low_pivot[i]

            last_pivot_price = low_pivot[i]
            last_pivot_index = i
            last_pivot_type  = 1

        if last_pivot_index < 0:
            continue

        # ------------------------------
        # 1- deviation soft confirmation
        # ------------------------------
        if last_pivot_type == 1:  # from low
            move = high[i] - last_pivot_price
        else:                     # from high
            move = last_pivot_price - low[i]

        if move >= deviation_soft_mult:
            sc_deviation[i] = 1

        # ------------------------------
        # 2- time soft confirmation
        # ------------------------------
        bars_since = i - last_pivot_index

        if bars_since >= time_threshold:
            sc_time[i] = 1

        # ------------------------------
        # 3- structure soft confirmation
        # ------------------------------
        if state_series[i] == -1 and prev_high_pivot is not None:
            if high_pivot[i] < prev_high_pivot:
                sc_structure[i] = 1

        if state_series[i] == 1 and prev_low_pivot is not None:
            if low_pivot[i] > prev_low_pivot:
                sc_structure[i] = 1

        # -----------------------------
        # 4- ATR soft confirmation
        # ------------------------------
        if atr is not None:

            if last_pivot_type == 1:
                atr_move = high[i] - last_pivot_price
            else:
                atr_move = last_pivot_price - low[i]

            if atr_move >= atr_mult * atr[i]:
                sc_atr[i] = 1

    return sc_deviation, sc_time, sc_structure, sc_atr  # sc: soft_confirmation


#####################################################################
# Layer 3 — Final Candle‑Level Market State
##################################################################### func-5
def zigzag_layer3_market_state(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,

    state_series: np.ndarray,
    confirmed_at: np.ndarray,
    developing_leg: np.ndarray,

    sc_deviation: np.ndarray,
    sc_time: np.ndarray,
    sc_structure: np.ndarray,
    sc_atr: np.ndarray,
):

    n = len(close)

    # بخش‌های اصلی state لایه سوم
    trend_state = np.zeros(n, dtype=np.int8)
    pivot_age = np.full(n, -1, dtype=np.int32)
    last_pivot_price_arr = np.zeros(n)
    distance_from_pivot = np.zeros(n)
    soft_strength = np.zeros(n, dtype=np.int8)

    last_pivot_index = -1
    last_pivot_price = None

    for i in range(n):

        # ---------------------------
        # 1) تعیین trend_state
        # ---------------------------
        # developing_leg = خروجی لایه اول
        trend_state[i] = developing_leg[i]

        # ---------------------------
        # 2) pivot detection
        # ---------------------------
        if state_series[i] == 1:       # low pivot
            last_pivot_price = low[i]
            last_pivot_index = i

        elif state_series[i] == -1:    # high pivot
            last_pivot_price = high[i]
            last_pivot_index = i

        # ---------------------------
        # 3) pivot age
        # ---------------------------
        if last_pivot_index >= 0:
            pivot_age[i] = i - last_pivot_index

        # ---------------------------
        # 4) distance from pivot
        # ---------------------------
        if last_pivot_price is not None:
            last_pivot_price_arr[i] = last_pivot_price
            distance_from_pivot[i] = abs(close[i] - last_pivot_price)

        # ---------------------------
        # 5) soft-strength
        # ---------------------------
        soft_strength[i] = (
            sc_deviation[i]
            + sc_time[i]
            + sc_structure[i]
            + sc_atr[i]
        )

    return {
        "trend_state": trend_state,
        "pivot_age": pivot_age,
        "last_pivot_price": last_pivot_price_arr,
        "distance_from_pivot": distance_from_pivot,
        "soft_strength": soft_strength,
    }


