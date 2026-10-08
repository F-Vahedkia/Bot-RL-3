# f04_features/indicators/levels.py
# Status in (Bot-RL-3): Reviewed at 1404/12/07
#                       Reviewed at 1405/02/12

#==============================================================================
# Imports & Logger
#==============================================================================
from __future__ import annotations
import numpy as np
import pandas as pd
from numba import njit
from typing import Sequence, Dict, Optional

from .utils import compute_atr
from .zigzag import zigzag_mtf_adapter, zigzag_legs

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# from f02_utils.config_loader import ConfigLoader
# cfg = ConfigLoader().get_all()
# _loader = ConfigLoader()
# _CONFIG_CACHE = _loader.get_all()

""" =========================================================================== OK Func1
"""
def pivots_classic(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, ...]:
    """
    speed=OK
    Pivot های کلاسیک
    """
    pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3.0
    r1 = 2*pivot - low.shift(1)
    s1 = 2*pivot - high.shift(1)
    r2 = pivot + (high.shift(1) - low.shift(1))
    s2 = pivot - (high.shift(1) - low.shift(1))
    r3 = high.shift(1) + 2*(pivot - low.shift(1))
    s3 = low.shift(1) - 2*(high.shift(1) - pivot)
    return \
        pivot.astype("float32"), \
        r1.astype("float32"), \
        s1.astype("float32"), \
        r2.astype("float32"), \
        s2.astype("float32"), \
        r3.astype("float32"), \
        s3.astype("float32")

""" =========================================================================== OK Func2
سطوح حمایت و مقاومت ثابت را از روی لِگ‌های زیگزاگ می‌سازد و آن‌ها را روی تمام کندل‌های هر لگ پخش می‌کند.

اگر لگ صعودی باشد → مقدار کف ابتدای لگ = Support
اگر لگ نزولی باشد → مقدار سقف ابتدای لگ = Resistance

این مقدار از ابتدای لگ تا انتهای همان لگ، برای همهٔ کندل‌ها ثابت می‌ماند.
در این توابع شیفت زمانی برابر با صفر درنظر گرفته شده است.
در این توابع فقط از ضمیمه لگها استفاده میشود.
"""
def sr_from_zigzag_legs_orig_1(
    df: pd.DataFrame,
    *,
    tf: str,    # tf_higher
    depth: int,
    deviation: float,
    backstep: int,
    extend_last_leg: bool = False,
) -> pd.DataFrame:
    """ نسخه اصلی بدون شیفت زمانی:
    در این نسخه قیمتهای ابتدا و انتهای لگ ها از دیتافریم لگ ها گرفته نمی شود،
    بلکه از دیتافریم اصلی داده ها استخراج میشود
    
    extend_last_leg = True: فرض میکند که انتهای آخرین لگ نقش اس/آر را برای دیتاهای بعد از لگ آخر بازی میکند.
    نقش این آرگومان با آرگومان هم نامی که در تابع زیگزاگ چند تایمفریمی است، متفاوت می باشد.
        """
    zz = zigzag_mtf_adapter(
        high=df["high"],
        low=df["low"],
        tf_higher=tf,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        use_timeshift=False,
    )

    legs = zz.attrs.get("legs", [])
    idx = df.index
    n = len(df)

    # --- Building containers ---
    sup = pd.Series(np.nan, index=idx, dtype=np.float32)
    res = pd.Series(np.nan, index=idx, dtype=np.float32)
    print(df["low"].iloc[86:92])
    for leg in legs:
        s = leg["ltf_start_pos"]
        e = leg["ltf_end_pos"]
        ts = leg["ltf_start_ts"]

        if leg["direction"] > 0:
            sup.iloc[s:e] = np.float32(df.at[ts, "low"])
        else:
            res.iloc[s:e] = np.float32(df.at[ts, "high"])

    # --- extend last valid leg to end of dataframe (ONLY ONCE) using last extremum ---
    if legs and extend_last_leg:
        last = legs[-1]
        s = last["ltf_end_pos"]    # اولین کندل بعد از آخرین کندل واقعی لگ آخر
        e = n                      # انتهای دیتافریم

        if last["direction"] < 0:
            sup.iloc[s:e] = np.float32(df["low"].iloc[s])
        else:
            res.iloc[s:e] = np.float32(df["high"].iloc[s])

    return pd.DataFrame(
        {"sr_support": sup, "sr_resistance": res},
        index=idx,
        dtype=np.float32,
    )

# -------------------------------------------------------------------
def sr_from_zigzag_legs_njit_1(
    df: pd.DataFrame,
    *,
    tf: str,    # tf_higher
    depth: int,
    deviation: float,
    backstep: int,
    extend_last_leg: bool = False,
) -> pd.DataFrame:
    """
    نسخه انجیت بدون شیفت زمانی:
    در این نسخه قیمتهای ابتدا و انتهای لگ ها از دیتافریم لگ ها گرفته نمی شود،
    بلکه از دیتافریم اصلی داده ها اساخراج میشود
    """

    # --- Run zigzag_mtf_adapter ---
    zz = zigzag_mtf_adapter(
        high=df["high"],
        low=df["low"],
        tf_higher=tf,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        use_timeshift=False,
    )

    legs = zz.attrs.get("legs", [])
    n = len(df)

    # --- Building containers ---
    sup_arr = np.full(n, np.nan, dtype=np.float32)
    res_arr = np.full(n, np.nan, dtype=np.float32)

    if not legs:
        return pd.DataFrame({"sr_support": sup_arr, "sr_resistance": res_arr},
                            index=df.index, dtype=np.float32)

    # Prepare leg data for Numba
    leg_array = np.array([
        (leg["ltf_start_pos"], leg["ltf_end_pos"], leg["direction"])
        for leg in legs
        if leg["ltf_start_pos"] < n
    ], dtype=np.int64)

    high_vals = df["high"].values.astype(np.float32)
    low_vals = df["low"].values.astype(np.float32)

    @njit
    def fill_sr(sup_arr, res_arr, leg_array, high_vals, low_vals):
        for i in range(leg_array.shape[0]):
            s = leg_array[i, 0]
            e = leg_array[i, 1]
            direction = leg_array[i, 2]
            if e > len(sup_arr):
                e = len(sup_arr)
            if direction > 0:
                val = low_vals[s]
                for j in range(s, e):
                    sup_arr[j] = val
            else:
                val = high_vals[s]
                for j in range(s, e):
                    res_arr[j] = val
        return sup_arr, res_arr

    sup_arr, res_arr = fill_sr(sup_arr, res_arr, leg_array, high_vals, low_vals)

    # --- extend last valid leg to end of dataframe (ONLY ONCE) using last extremum ---
    if legs and extend_last_leg:
        last_leg = legs[-1]
        s_last = last_leg["ltf_end_pos"]   # اولین کندل بعد از آخرین کندل واقعی لگ آخر
        e_last = n                         # انتهای دیتافریم
        direction_last = last_leg["direction"]

        if s_last < n:
            if direction_last < 0:
                val = low_vals[s_last]
                sup_arr[s_last:e_last] = val
            else:
                val = high_vals[s_last]
                res_arr[s_last:e_last] = val

    return pd.DataFrame({"sr_support": sup_arr, "sr_resistance": res_arr},
                        index=df.index, dtype=np.float32)

# -------------------------------------------------------------------
def sr_from_zigzag_legs_orig(
    df: pd.DataFrame,
    *,
    tf: str,    # tf_higher
    depth: int,
    deviation: float,
    backstep: int,
    extend_last_leg: bool = False,
) -> pd.DataFrame:
    """ نسخه اصلی با شیفت زمانی:
    در این نسخه قیمتهای ابتدا و انتهای لگ ها از دیتافریم لگ ها گرفته می شود،
    extend_last_leg = True: فرض میکند که انتهای آخرین لگ نقش اس/آر را برای دیتاهای بعد از لگ آخر بازی میکند.
    نقش این آرگومان با آرگومان هم نامی که در تابع زیگزاگ چند تایمفریمی است، متفاوت می باشد.
    """
    # zz = zigzag_mtf_adapter(
    #     high=df["high"],
    #     low=df["low"],
    #     tf_higher=tf,
    #     depth=depth,
    #     deviation=deviation,
    #     backstep=backstep,
    #     use_timeshift=True,
    # )
    # legs = zz.attrs.get("legs", [])
    legs = zigzag_legs(
        high=df["high"],
        low=df["low"],
        tf=tf,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        # point=0.01,
        # mode: Literal["last", "forward_fill"] = "forward_fill",
        extend_last_leg=extend_last_leg,
        use_timeshift=True,
    )

    if isinstance(legs, pd.DataFrame):
        legs = legs.to_dict("records")

    idx = df.index
    n = len(df)

    # --- Building containers ---
    sup = pd.Series(np.nan, index=idx, dtype=np.float32)
    res = pd.Series(np.nan, index=idx, dtype=np.float32)

    for leg in legs:
        s = leg["ltf_start_pos"]
        e = leg["ltf_end_pos"]
        ts = leg["ltf_start_ts"]

        if leg["direction"] > 0:
            # sup.iloc[s:e] = np.float32(df.at[ts, "low"])    # 1404/12/02
            sup.iloc[s:e] = np.float32(leg["ltf_start_extr"])    # 1404/12/02
        else:
            # res.iloc[s:e] = np.float32(df.at[ts, "high"])    # 1404/12/02
            res.iloc[s:e] = np.float32(leg["ltf_start_extr"])    # 1404/12/02

    # --- extend last valid leg to end of dataframe (ONLY ONCE) using last extremum ---
    if legs and extend_last_leg:
        last = legs[-1]
        s = last["ltf_end_pos"]    # اولین کندل بعد از آخرین کندل واقعی لگ آخر
        e = n                      # انتهای دیتافریم

        if last["direction"] < 0:
            # sup.iloc[s:e] = np.float32(df["low"].iloc[s])    # 1404/12/02
            sup.iloc[s:e] = np.float32(last["ltf_end_extr"])    # 1404/12/02
        else:
            # res.iloc[s:e] = np.float32(df["high"].iloc[s])    # 1404/12/02
            res.iloc[s:e] = np.float32(last["ltf_end_extr"])    # 1404/12/02

    return pd.DataFrame(
        {"sr_support": sup, "sr_resistance": res},
        index=idx,
        dtype=np.float32,
    )

# -------------------------------------------------------------------
def sr_from_zigzag_legs_njit(
    df: pd.DataFrame,
    *,
    tf: str,    # tf_higher
    depth: int,
    deviation: float,
    backstep: int,
    extend_last_leg: bool = False,
) -> pd.DataFrame:
    """
    نسخه انجیت با شیفت زمانی:
    در این نسخه قیمتهای ابتدا و انتهای لگ ها از دیتافریم لگ ها گرفته می شود،
    """

    # --- Run zigzag_mtf_adapter ---
    # zz = zigzag_mtf_adapter(
    #     high=df["high"],
    #     low=df["low"],
    #     tf_higher=tf,
    #     depth=depth,
    #     deviation=deviation,
    #     backstep=backstep,
    #     use_timeshift=True,
    # )
    # legs = zz.attrs.get("legs", [])
    legs = zigzag_legs(
        high=df["high"],
        low=df["low"],
        tf=tf,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        # point=0.01,
        # mode: Literal["last", "forward_fill"] = "forward_fill",
        extend_last_leg=extend_last_leg,
        use_timeshift=True,
    )

    if isinstance(legs, pd.DataFrame):
        legs = legs.to_dict("records")

    n = len(df)

    # --- Building containers ---
    sup_arr = np.full(n, np.nan, dtype=np.float32)
    res_arr = np.full(n, np.nan, dtype=np.float32)

    if not legs:
        return pd.DataFrame({"sr_support": sup_arr, "sr_resistance": res_arr},
                            index=df.index, dtype=np.float32)

    # Prepare leg data for Numba (with extremum prices)
    leg_array = np.array([
        (
            leg["ltf_start_pos"],
            leg["ltf_end_pos"],
            leg["direction"],
            leg["ltf_start_extr"],
        )
        for leg in legs
        if leg["ltf_start_pos"] < n
    ], dtype=np.float32)

    high_vals = df["high"].values.astype(np.float32)
    low_vals = df["low"].values.astype(np.float32)

    @njit
    def fill_sr(sup_arr, res_arr, leg_array):
        for i in range(leg_array.shape[0]):

            s = int(leg_array[i, 0])
            e = int(leg_array[i, 1])
            direction = int(leg_array[i, 2])
            val = leg_array[i, 3]

            if e > len(sup_arr):
                e = len(sup_arr)

            if direction > 0:
                for j in range(s, e):
                    sup_arr[j] = val
            else:
                for j in range(s, e):
                    res_arr[j] = val
        return sup_arr, res_arr

    sup_arr, res_arr = fill_sr(sup_arr, res_arr, leg_array)

    # --- extend last valid leg to end of dataframe (ONLY ONCE) using last extremum ---
    if legs and extend_last_leg:
        last_leg = legs[-1]
        s_last = last_leg["ltf_end_pos"]   # اولین کندل بعد از آخرین کندل واقعی لگ آخر
        e_last = n                         # انتهای دیتافریم
        direction_last = last_leg["direction"]
        val_last = last_leg["ltf_end_extr"]

        if s_last < n:
            if direction_last < 0:
                sup_arr[s_last:e_last] = val_last
            else:
                res_arr[s_last:e_last] = val_last

    return pd.DataFrame({"sr_support": sup_arr, "sr_resistance": res_arr},
                        index=df.index, dtype=np.float32)

# --- نسخه رپر ------------------------------------------------------
def sr_from_zigzag_legs(
    df: pd.DataFrame,
    *,
    tf: str,    # tf_higher
    depth: int,
    deviation: float,
    backstep: int,
    extend_last_leg: bool = False,
    _njit_threshold: int = 1_000_000,
) -> pd.DataFrame:
    """
    Smart wrapper:
    - small DF  -> pandas implementation
    - large DF  -> njit implementation
    """
    if len(df) < _njit_threshold:
        return sr_from_zigzag_legs_orig(
            df,
            tf=tf,
            depth=depth,
            deviation=deviation,
            backstep=backstep,
            extend_last_leg=extend_last_leg,
        )   # output: indexed, float32
    else:
        return sr_from_zigzag_legs_njit(
            df,
            tf=tf,
            depth=depth,
            deviation=deviation,
            backstep=backstep,
            extend_last_leg=extend_last_leg,
        )   # output: indexed, float32

""" =========================================================================== OK Func3
"""
def sr_distance_from_levels(
    df: pd.DataFrame,
    sr: pd.DataFrame,
    *,
    atr_window: int = 14,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    فاصله نرمال‌شده قیمت پایانی تا سطوح حمایت و مقاومت فعال.
    نرمال‌سازی بر اساس ATR (Average True Range) انجام می‌شود.

    پارامترها:
    -----------
    df : pd.DataFrame
        دیتافریم شامل ستون‌های ['high', 'low', 'close'] و ایندکس زمانی
    sr : pd.DataFrame
        دیتافریم خروجی sr_from_zigzag_legs شامل ستون‌های
        ['sr_support', 'sr_resistance']
    atr_window : int
        طول پنجره برای محاسبه ATR
    eps : float
        مقدار کوچک برای جلوگیری از تقسیم بر صفر

    خروجی:
    -------
    pd.DataFrame با ستون‌های:
    - dist_to_support_norm
    - dist_to_resistance_norm
    """

    close = df["close"]
    
    # --- ATR (Single Source of Truth) ---------- Added 1404/12/02
    atr = compute_atr(df, window=atr_window, method="classic")
    
    # --- فاصله نرمال شده ---
    dist_sup = ((close - sr["sr_support"]) / (atr + eps)).astype(np.float32)
    dist_res = ((sr["sr_resistance"] - close) / (atr + eps)).astype(np.float32)

    # --- NaN propagation ---
    dist_sup[sr["sr_support"].isna()] = np.nan
    dist_res[sr["sr_resistance"].isna()] = np.nan

    return pd.DataFrame(
        {
            "dist_to_support_norm": dist_sup,
            "dist_to_resistance_norm": dist_res,
            # "atr": atr,     # for debug
        },
        index=df.index,
        dtype=np.float32,
    )

""" =========================================================================== OK Func4
برای استفاده داخلی است
در خروجی این توابع، اندکسهای متناظری که تحت پوشش لگهای زیگزاگ هستند True میشوند
"""
# -------------------------------------------------------------------
def _zigzag_leg_mask_orig(zz: pd.Series) -> pd.Series:
    """
    Returns boolean mask True for all indices covered by zigzag legs.
    """
    legs = zz.attrs.get("legs", [])
    if not legs:
        return pd.Series(False, index=zz.index, dtype=bool)

    mask = pd.Series(False, index=zz.index)
    n = len(zz)

    for leg in legs:
        s = leg["ltf_start_pos"]
        e = leg["ltf_end_pos"]
        if s >= n:
            continue
        if e > n:
            e = n
        mask.iloc[s:e] = True
    return mask.astype(bool)

# -------------------------------------------------------------------
def _zigzag_leg_mask_njit(zz: pd.Series) -> pd.Series:
    """
    Numba-optimized version of zigzag_leg_mask.
    Returns boolean mask True for all indices covered by zigzag legs.
    """
    n = len(zz)
    mask = np.zeros(n, dtype=np.bool_)

    legs = zz.attrs.get("legs", [])
    if not legs:
        return pd.Series(mask, index=zz.index, dtype=bool)

    # Prepare leg data as NumPy array for njit
    leg_array = np.array(
        [(leg["ltf_start_pos"], leg["ltf_end_pos"]) for leg in legs],
        dtype=np.int64
    )

    @njit
    def fill_mask(mask_arr, leg_arr):
        for i in range(leg_arr.shape[0]):
            s = leg_arr[i, 0]
            e = leg_arr[i, 1]
            if s >= mask_arr.shape[0]:
                continue
            if e > mask_arr.shape[0]:
                e = mask_arr.shape[0]
            for j in range(s, e):
                mask_arr[j] = True
        return mask_arr

    mask = fill_mask(mask, leg_array)
    return pd.Series(mask, index=zz.index, dtype=bool)

# --- نسخه رپر ------------------------------------------------------
def _zigzag_leg_mask(
    zz: pd.Series,
    _njit_threshold: int = 1_400_000,
) -> pd.Series:
    if len(zz) < _njit_threshold:
        return _zigzag_leg_mask_orig(zz=zz)
    else:
        return _zigzag_leg_mask_njit(zz=zz)

""" =========================================================================== OK Func5
"""
def fibo_levels_from_legs_orig(
    df: pd.DataFrame,
    zz: pd.Series,
    ratios: Optional[Sequence[float]] = None,
    extend_last_leg: bool = False,
) -> pd.DataFrame:
    """
    Compute Fibonacci retracement levels from zigzag leg metadata.
    Uses leg extrema stored in zz.attrs["legs"].
    """

    if ratios is None:
        from f02_utils.config_loader import load_config
        cfg = load_config()
        feat = cfg.get("features", {})
        fibo = feat.get("fibonacci", {})
        ratios = fibo.get("retracement_ratios", None)
        if ratios is None:
            ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

    legs = zz.attrs.get("legs", [])
    n = len(df)

    cols = [f"fibo_{r:.3f}" for r in ratios]
    out = np.full(shape=(n, len(ratios)), fill_value=np.nan, dtype=np.float32)

    ratios_arr = np.asarray(ratios, dtype=np.float32)

    for leg in legs:

        s = leg["ltf_start_pos"]
        e = leg["ltf_end_pos"]
        direction = int(leg["direction"])

        if s < 0 or e <= s or s >= n:
            continue
        if e > n:
            e = n

        start_price = float(leg["ltf_start_extr"])
        end_price   = float(leg["ltf_end_extr"])

        rng = abs(end_price - start_price)
        if rng == 0.0:
            continue

        if direction > 0:
            # bullish leg → retracement below high
            levels = end_price - ratios_arr * rng
        else:
            # bearish leg → retracement above low
            levels = end_price + ratios_arr * rng

        out[s:e, :] = levels.astype(np.float32)
    
    # --- extend last leg --------------------------------- start
    if extend_last_leg and legs:

        last_leg = legs[-1]

        s = last_leg["ltf_start_pos"]
        e = last_leg["ltf_end_pos"]
        direction = int(last_leg["direction"])

        if s >= 0 and s < n:

            start_price = float(last_leg["ltf_start_extr"])
            end_price   = float(last_leg["ltf_end_extr"])

            rng = abs(end_price - start_price)

            if rng != 0.0:

                if direction > 0:
                    levels = end_price - ratios_arr * rng
                else:
                    levels = end_price + ratios_arr * rng

                e = min(e, n)
                out[e:n, :] = levels.astype(np.float32)
    # --- extend last leg --------------------------------- end

    return pd.DataFrame(out, index=df.index, columns=cols, dtype=np.float32)

# -------------------------------------------------------------------
def fibo_levels_from_legs_njit(
    df: pd.DataFrame,
    zz: pd.Series,
    ratios: Optional[Sequence[float]] = None,
    extend_last_leg: bool = False,
) -> pd.DataFrame:

    # --- njit core -------------------------------------------------
    @njit(cache=True, fastmath=True)
    def _fibo_levels_from_legs_core_fast(
        n: int,
        start_pos_arr: np.ndarray,
        end_pos_arr: np.ndarray,
        direction_arr: np.ndarray,
        start_price_arr: np.ndarray,
        end_price_arr: np.ndarray,
        ratios_arr: np.ndarray,
    ):
        n_ratios = ratios_arr.shape[0]
        out = np.empty((n, n_ratios), dtype=np.float32)

        # fill NaN manually (numba-safe)
        for i in range(n):
            for j in range(n_ratios):
                out[i, j] = np.nan

        n_legs = start_pos_arr.shape[0]

        for i in range(n_legs):

            s = start_pos_arr[i]
            e = end_pos_arr[i]

            if s < 0 or e <= s or s >= n:
                continue
            if e > n:
                e = n

            direction = direction_arr[i]
            start_price = start_price_arr[i]
            end_price   = end_price_arr[i]

            rng = end_price - start_price
            if rng < 0:
                rng = -rng
            if rng == 0.0:
                continue

            # precompute levels once per leg
            for j in range(n_ratios):
                r = ratios_arr[j]
                if direction > 0:
                    level = end_price - r * rng
                else:
                    level = end_price + r * rng

                level32 = np.float32(level)

                # contiguous memory write (fast)
                for k in range(s, e):
                    out[k, j] = level32

        return out
    
    # ---------------------------------------------------------------
    if ratios is None:
        from f02_utils.config_loader import load_config
        cfg = load_config()
        feat = cfg.get("features", {})
        fibo = feat.get("fibonacci", {})
        ratios = fibo.get("retracement_ratios", None)
        if ratios is None:
            ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

    legs = zz.attrs.get("legs", [])
    n = len(df)

    cols = [f"fibo_{r:.3f}" for r in ratios]

    if not legs:
        return pd.DataFrame(
            np.full((n, len(ratios)), np.nan, dtype=np.float32),
            index=df.index,
            columns=cols,
        )

    start_pos_arr   = np.asarray([leg["ltf_start_pos"]  for leg in legs], dtype=np.int32)
    end_pos_arr     = np.asarray([leg["ltf_end_pos"]    for leg in legs], dtype=np.int32)
    direction_arr   = np.asarray([leg["direction"]      for leg in legs], dtype=np.int8)
    start_price_arr = np.asarray([leg["ltf_start_extr"] for leg in legs], dtype=np.float64)
    end_price_arr   = np.asarray([leg["ltf_end_extr"]   for leg in legs], dtype=np.float64)

    ratios_arr = np.asarray(ratios, dtype=np.float32)

    out = _fibo_levels_from_legs_core_fast(
        n,
        start_pos_arr,
        end_pos_arr,
        direction_arr,
        start_price_arr,
        end_price_arr,
        ratios_arr,
    )

    # --- extend last leg --------------------------------- start
    if extend_last_leg and legs:

        last_leg = legs[-1]

        s = last_leg["ltf_start_pos"]
        e = last_leg["ltf_end_pos"]
        direction = int(last_leg["direction"])

        if 0 <= s < n:

            start_price = float(last_leg["ltf_start_extr"])
            end_price   = float(last_leg["ltf_end_extr"])

            rng = abs(end_price - start_price)

            if rng != 0.0:

                if direction > 0:
                    levels = end_price - ratios_arr * rng
                else:
                    levels = end_price + ratios_arr * rng

                e = min(e, n)

                # extend to end of dataframe
                out[e:n, :] = levels.astype(np.float32)
    # --- extend last leg --------------------------------- end

    return pd.DataFrame(out, index=df.index, columns=cols, dtype=np.float32)

# --- نسخه رپر ------------------------------------------------------
def fibo_levels_from_legs(
    df: pd.DataFrame,
    zz: pd.Series,
    ratios: Optional[Sequence[float]] = None,
    threshold_bytes: int = 12_800_000,
    extend_last_leg: bool = False,
) -> pd.DataFrame:
    """
    Wrapper for Fibonacci-from-legs.
    Selects orig (numpy/pandas) or njit version based on memory footprint.

    Parameters
    ----------
    df : pd.DataFrame
    zz : pd.Series (with attrs["legs"])
    ratios : optional retracement ratios
    threshold_bytes : int
        If estimated memory usage exceeds this value → use njit version.
    """

    # --- estimate memory footprint (high/low only, same philosophy as zigzag) ---
    if "high" in df.columns and "low" in df.columns:
        high_np = np.ascontiguousarray(df["high"].values, dtype=np.float64)
        low_np  = np.ascontiguousarray(df["low"].values,  dtype=np.float64)
        bytes_used = high_np.nbytes + low_np.nbytes

    else:
        # fallback: approximate using index length
        bytes_used = len(df) * 16  # float64 * 2 assumption

    # --- choose implementation ---
    if bytes_used <= threshold_bytes:
        return fibo_levels_from_legs_orig(df, zz, ratios, extend_last_leg=extend_last_leg)   # output: indexed, float32
    else:
        return fibo_levels_from_legs_njit(df, zz, ratios, extend_last_leg=extend_last_leg)   # output: indexed, float32

""" =========================================================================== =Func6
"""
def registry() -> Dict[str, callable]:
    
    # --- func1 -------------------------------------------
    def make_pivots(df, **_):
        p, r1, s1, r2, s2, r3, s3 = pivots_classic(df["high"], df["low"], df["close"])
        return {"pivot": p,
                "pivot_r1": r1, "pivot_s1": s1,
                "pivot_r2": r2, "pivot_s2": s2,
                "pivot_r3": r3, "pivot_s3": s3
                }
    
    # --- func2 -------------------------------------------
    def make_sr_zigzag(df, tf, depth, deviation, backstep, **_):
        return sr_from_zigzag_legs(
            df,
            tf=tf,
            depth=depth,
            deviation=deviation,
            backstep=backstep,
        ).to_dict(orient="series")

    # --- func3 -------------------------------------------
    def make_sr_distance(df, tf, depth, deviation, backstep, atr_window=14, **_):
        # ابتدا SR levels بساز
        sr = sr_from_zigzag_legs(
            df,
            tf=tf,
            depth=depth,
            deviation=deviation,
            backstep=backstep,
        )
        # سپس فاصله نرمال‌شده تا سطوح
        return sr_distance_from_levels(
            df=df,
            sr=sr,
            atr_window=atr_window
        ).to_dict(orient="series")

    # --- func4 -------------------------------------------
    def make_fibo_from_legs(df, zz, ratios=None, **_):
        return fibo_levels_from_legs(
            df,
            zz,
            ratios=ratios
        ).to_dict(orient="series")

    return {
        "pivots": make_pivots,
        "sr": make_sr_zigzag,
        "sr_distance": make_sr_distance,
        "fibo": make_fibo_from_legs,
    }


""" =========================================================================== OK Func7
"""
def compute_adr(df: pd.DataFrame, window: int = 14, tz: str = "UTC") -> pd.Series:
    """
    ADR (Average Daily Range)
    Classic ADR: میانگینِ (High-Low) روزانه روی پنجرهٔ rolling.
    - ابتدا OHLC روزانه را می‌سازد (بر اساس resample('1D'))
    - سپس میانگین rolling از دامنهٔ روزانه را می‌گیرد
    - در پایان سری ADR روزانه را به تایم‌استمپ‌های درون‌روزی ffill می‌کند

    ورودی: df با ستون‌های high/low (و بهتر است close برای resample صحیح)
    خروجی: Series با نام 'ADR_{window}' هم‌تراز با df.index
    """
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError("DF must contain at least: high, low")

    # تبدیل به فریم روزانه
    daily = df[["high", "low"]].copy()
    daily = daily.tz_convert(tz) if (daily.index.tz is not None) else daily.tz_localize(tz)
    daily_ohl = pd.DataFrame({
        "hi": daily["high"].resample("1D", label="left", closed="left").max(),
        "lo": daily["low" ].resample("1D", label="left", closed="left").min(),
    }).dropna()
    daily_range = (daily_ohl["hi"] - daily_ohl["lo"]).rename("daily_range")
    adr_daily = daily_range.shift(1).rolling(window=window, min_periods=max(2, window // 2)).mean()
    adr_daily.name = f"ADR_{window}"

    # نگاشت ADR روزانه به ایندکس درون‌روزی با ffill
    adr_intraday = adr_daily.reindex(df.index, method="ffill")
    return adr_intraday.astype("float32")

""" =========================================================================== OK Func8
"""
def adr_distance_to_open(df: pd.DataFrame, adr: pd.Series, tz: str = "UTC") -> pd.DataFrame:
    """
    فاصلهٔ قیمت تا «بازِ روز» نرمال‌شده به ADR.
    خروجی ستون‌ها:
      - day_open: بازِ روز (نخستین close هر روز)
      - dist_abs: |price - day_open|
      - dist_pct_of_adr: 100 * dist_abs / ADR
    """
    if "close" not in df.columns:
        raise ValueError("DF must contain 'close' to compute day_open distance")

    px = df["close"].copy()
    px = px.tz_convert(tz) if (px.index.tz is not None) else px.tz_localize(tz)

    # بازِ روز = قیمت بسته شدن اولین کندل در بازه همان روز در تایم فریم اصلی (پایین تر)
    day_open_daily = px.resample("1D", label="left", closed="left").first().rename("day_open")
    day_open = day_open_daily.reindex(px.index, method="ffill")

    dist_abs = (px - day_open).abs().rename("dist_abs")
    adr_safe = adr.copy()
    adr_safe.replace(0.0, np.nan, inplace=True)
    dist_pct = (100.0 * dist_abs / adr_safe).rename("dist_pct_of_adr")

    out = pd.concat([day_open, dist_abs, dist_pct], axis=1)
    return out.astype("float32")

""" =========================================================================== OK Func9
"""
def sr_overlap_score_simple(price: float, sr_levels: Sequence[float], tol_pct: float = 0.05) -> np.float32:
    """
    S/R Overlap Score (0..1)
    امتیاز همپوشانی قیمت با سطوح S/R:
      - اگر نزدیک‌ترین سطح در فاصلهٔ tol_pct (نسبت به قیمت) باشد → امتیاز ۰..۱ (هرچه نزدیک‌تر، امتیاز بالاتر)
      - اگر چند سطح داخل tol باشند، یک پاداش کوچک اضافه می‌شود (clip به ۱)

    پارامترها:
      price: قیمتِ ارزیابی
      sr_levels: لیست سطوح S/R
      tol_pct: آستانهٔ نسبی (مثلاً 0.05 یعنی 5%)

    خروجی: نمرهٔ 0..1
    """
    if not sr_levels:
        return np.float32(0.0)

    # print(f" ---> price= {price}")               ##### for debug
    # print(f" ---> sr_levels= {sr_levels}")       ##### for debug
    # print(f" ---> tol_pct= {tol_pct}")           ##### for debug

    tol_abs = abs(price) * tol_pct
    # print(f" ---> tol_abs= {tol_abs}")           ##### for debug
    diffs = np.array([price - lv for lv in sr_levels], dtype=float)
    abs_diffs = np.abs(diffs)
    # print(f" ---> abs_diffs= {abs_diffs}")       ##### for debug

    j = int(np.argmin(abs_diffs))
    min_dist = float(abs_diffs[j])
    # print(f" ---> min_dist= {min_dist}")         ##### for debug

    if min_dist > tol_abs or tol_abs == 0.0:
        return np.float32(0.0)

    # امتیاز پایه: نزدیکی خطی تا ۱
    base = 1.0 - (min_dist / tol_abs)
    # print(f" ---> base= {base}")                 ##### for debug
    # پاداش کوچک بابت تعداد سطوح در محدودهٔ tol
    k = int(np.sum(abs_diffs <= tol_abs))
    # print(f" ---> k= {k}")                       ##### for debug
    bonus = 0.1 * max(0, k - 1)
    # print(f" ---> bonus= {bonus}")               ##### for debug

    score = min(1.0, max(0.0, base + bonus))
    # print(f" ---> score= {score}")               ##### for debug

    return np.float32(score)

# ----------------------------------------------------------------------------- OK Fun10
def sr_overlap_score(
    price: float,
    sr_levels: Sequence[float],
    tol_pct: float = 0.05,
    sr_weights: Optional[Sequence[float]] = None,
) -> np.float32:
    """
    امتیاز همپوشانی قیمت با سطوح S/R (نسخه پیشرفته)
    
    ویژگی‌ها:
      - نزدیک‌ترین سطح: تابع غیرخطی (sigmoid) برای حساسیت بیشتر
      - همه سطوح داخل tolerance: وزن‌دهی فاصله‌ای
      - امکان وزن‌دهی متفاوت برای سطوح S و R

    پارامترها:
      price: قیمت فعلی
      sr_levels: لیست سطوح S/R
      tol_pct: آستانه نسبی (مثلاً 0.05 یعنی ±5%)
      sr_weights: لیست وزن برای هر سطح در sr_levels (اختیاری)

    خروجی:
      score بین 0 و 1
    """ 
    # --- Normalize inputs (robust against list / tuple / ndarray) ---
    sr_levels = np.asarray(sr_levels, dtype=float)

    if sr_levels.size == 0:
        return np.float32(0.0)
    if not np.isfinite(price):
        return np.float32(0.0)

    # آستانه مطلق
    tol_abs = abs(price) * float(tol_pct)
    if tol_abs <= 0.0:
        return np.float32(0.0)

    # فاصله‌ها از قیمت
    abs_diffs = np.abs(price - sr_levels)

    # اگر وزن‌دهی جدا S/R ارائه نشده باشد، همه وزن برابر با یک فرض میشوند
    if sr_weights is None:
        sr_weights = np.ones(sr_levels.size, dtype=float)
    else:
        sr_weights = np.asarray(sr_weights, dtype=float)
        if sr_weights.size != sr_levels.size:
            raise ValueError("sr_weights must match sr_levels length")
        
    # نزدیک‌ترین سطح
    j = int(np.argmin(abs_diffs))
    min_dist = float(abs_diffs[j])
    weight_closest = sr_weights[j]

    # اگر نزدیک‌ترین سطح خارج tol_abs بود → score = 0
    if min_dist > tol_abs:
        return np.float32(0.0)

    # --- امتیاز غیرخطی برای نزدیک‌ترین سطح ---
    # برای حساسیت بیشتر به نزدیکی، از سیگموئید ساده استفاده می‌کنیم
    
    x = 1 - (min_dist / tol_abs)              # 1: یعنی قیمت روی سطح مورد نظر قرار دارد
                                              # 0: یعنی آن قیمت با سطح مورد نظر، به اندازه تلرانس تعیین شده فاصله دارد
    
    base = 1 / (1 + np.exp(-12 * (x - 0.5)))  # sigmoid بین ~0 و ~1
    base *= weight_closest                    # اعمال وزن سطح

    mask = abs_diffs <= tol_abs               # Build True/False array as a filter
    bonus = np.zeros_like(abs_diffs)          # Build array same type & same shape as "abs_diffs"

    bonus[mask] = (1.0 - (abs_diffs[mask] / tol_abs)) * sr_weights[mask]  # Calculate only for "mask == True"
    bonus[j] = 0.0                            # remove closest (already counted)

    score = base + np.sum(bonus)
    
    return np.float32(min(1.0, max(0.0, score)))



from f04_features.indicators.utils import (
    round_levels as _round_levels,
    nearest_level_distance as _nearest_level_distance,
)
from f02_utils.config_operations import _deep_get

# --- adr --------------------------------------------------------------------- Func.4 (edited 041204) 
def _adv_adr(df, **cfg) -> Dict[str, pd.Series]:
    # --- read config ----------------- start
    # params = _deep_get(_CONFIG_CACHE, "features.adr", {}) or {}
    params = _deep_get(cfg, "features.adr", {}) or {}

    enabled   = bool(params.get("enabled", False))
    window    = int(params.get("window", params.get("w", 14)))
    tz        = params.get("tz", "UTC")
    normalize = bool(params.get("normalize", False))
    # --- read config ----------------- end

    key = f"adr_{window}"

    # اگر غیرفعال است
    if not enabled:
        return {key: pd.Series(0.0, index=df.index, dtype="float32")}

    # محاسبه ADR
    s = compute_adr(df, window=window, tz=tz)
    if not isinstance(s, pd.Series):
        s = pd.Series(s, index=df.index)

    # اعمال normalize در صورت فعال بودن
    if normalize:
        close = df["close"].astype(float).replace(0, np.nan)
        s = s.astype(float) / close

    s = s.astype("float32")

    return {key: s}


# --- adr_distance_to_open ---------------------------------------------------- Func.5 (edited 041204)
def _adv_adr_distance_to_open(df, **cfg) -> Dict[str, pd.Series]:
    """
    Wrapper برای adapter اصلی `adr_distance_to_open` با اعمال تنظیمات کانفیگ.

    کلیدهای کانفیگ استفاده شده:
        enabled     : فعال/غیرفعال کردن feature
        window      : طول پنجره ADR
        use_percent : آیا فاصله نرمال‌شده بر اساس ADR محاسبه شود یا خیر

    خروجی:
        {
            'adr_day_open_<w>': Series open روز
            'adr_dist_abs_<w>': Series فاصله مطلق close تا open
            'adr_dist_pct_<w>': Series فاصله درصدی (در صورت فعال بودن use_percent)
        }
    """

    # --- خواندن تنظیمات از کانفیگ
    # params = (_deep_get(_CONFIG_CACHE, "features.adr_distance_to_open", {}) or {})
    params = (_deep_get(cfg, "features.adr.distance_to_open", {}) or {})
    enabled     = bool(params.get("enabled", False))
    window      = int(params.get("window", 14))
    tz          = params.get("tz", "UTC")
    use_percent = bool(params.get("use_percent", True))

    w = window

    # --- اگر feature غیرفعال است، ستون صفر برگردان
    if not enabled:
        zero = pd.Series(0.0, index=df.index, dtype="float32")
        return {
            f"adr_day_open_{w}": zero.copy(),
            f"adr_dist_abs_{w}": zero.copy(),
            f"adr_dist_pct_{w}": zero.copy(),
        }

    # --- فراخوانی adapter اصلی
    out = adr_distance_to_open(df, window=w, tz=tz)

    # اگر Series بازگردد، تبدیل به DataFrame می‌کنیم
    if isinstance(out, pd.Series):
        out = out.to_frame("dist_abs")

    # --- استخراج ستون‌ها
    s_open = out.get("day_open", df["open"].astype(float)).reindex(df.index)
    s_abs  = out.get("dist_abs", (df["close"].astype(float) - s_open).abs()).reindex(df.index)

    # --- محاسبه درصد نرمال‌شده فقط در صورت use_percent=True
    if use_percent:
        s_pct = out.get("dist_pct")
        if s_pct is None:
            # fallback: dist_abs / adr
            adr = compute_adr(df, window=w, tz=tz)
            s_pct = (s_abs / adr.replace(0, np.nan)).fillna(0.0)
        s_pct = s_pct.reindex(df.index)
    else:
        s_pct = pd.Series(0.0, index=df.index)

    # --- تبدیل dtype به float32
    s_open = s_open.astype("float32")
    s_abs  = s_abs.astype("float32")
    s_pct  = s_pct.astype("float32")

    return {
        f"adr_day_open_{w}": s_open,
        f"adr_dist_abs_{w}": s_abs,
        f"adr_dist_pct_{w}": s_pct,
    }


# --- sr_overlap_score -------------------------------------------------------- Func.6 (edited 041204)
def _adv_sr_overlap_score(df, **cfg) -> Dict[str, pd.Series]:
    """
    محاسبه امتیاز همپوشانی close با سطوح S/R رُند
    خروجی:
        {'sr_overlap_score_<step>_<n>_<tolbp>bp': Series}
    
    رفتار:
    - اگر feature غیرفعال باشد → سری صفر برمی‌گردد
    - normalize=True → امتیاز به [0,1] scale می‌شود
    """

    # --- خواندن تنظیمات از کانفیگ
    # params = (_deep_get(_CONFIG_CACHE, "features.sr_overlap_score", {}) or {})
    params = (_deep_get(cfg, "features.sr_overlap_score", {}) or {})
    enabled   = bool(params.get("enabled", False))
    anchor    = float(params.get("anchor", params.get("a", 0.0)))
    step      = float(params.get("step", 10))
    n         = int(params.get("n", 5))
    tol_pct   = float(params.get("tolerance_pct", params.get("tol_pct", 0.05)))
    normalize = bool(params.get("normalize", True))

    # اگر غیرفعال است → صفر برگردان
    if not enabled:
        zero = pd.Series(0.0, index=df.index, dtype="float32")
        key = f"sr_overlap_score_{str(step).replace('.','_')}_{int(n)}_{int(round(tol_pct*10000))}bp"
        return {key: zero}

    # --- تولید سطوح رُند
    levels = _round_levels(anchor=anchor, step=step, n=n)

    # --- محاسبه سری امتیاز
    close = df["close"].astype(float)
    vals = [float(sr_overlap_score(px, levels, tol_pct=tol_pct)) for px in close]
    s = pd.Series(vals, index=df.index, dtype="float32")

    # --- اعمال normalize
    if normalize:
        max_val = s.max()
        if max_val > 0:
            s = s / max_val

    # --- کلید امن
    tolbp = int(round(tol_pct * 10000))
    key = f"sr_overlap_score_{str(step).replace('.','_')}_{int(n)}_{tolbp}bp"

    return {key: s}


# --- round_levels (nearest distance) ----------------------------------------- Func.7
def _adv_round_levels(df, anchor: float, step: float, n: int = 10, **_) -> Dict[str, pd.Series]:
    """
    آداپتر round_levels: برای هر close فاصله تا نزدیک‌ترین سطح رُند.
    - levels = round_levels(anchor, step, n)
    - nearest_level_distance(price_t, levels) → {'nearest_level','signed','abs'}
    خروجی: سه سری هم‌نام با suffix:
       {'rl_nearest_<step>_<n>', 'rl_signed_<step>_<n>', 'rl_abs_<step>_<n>'}
    """
    levels = _round_levels(anchor=anchor, step=step, n=int(n))
    close = df["close"].astype(float)
    nearest_list = []
    signed_list = []
    abs_list = []
    for px in close:
        d = _nearest_level_distance(float(px), levels)
        nearest_list.append(d["nearest_level"])
        signed_list.append(d["signed"])
        abs_list.append(d["abs"])
    s_nearest = pd.Series(nearest_list, index=df.index, name="rl_nearest").astype("float32")
    s_signed  = pd.Series(signed_list,  index=df.index, name="rl_signed").astype("float32")
    s_abs     = pd.Series(abs_list,     index=df.index, name="rl_abs").astype("float32")
    tag = f"{str(step).replace('.','_')}_{int(n)}"
    return {
        f"rl_nearest_{tag}": s_nearest,
        f"rl_signed_{tag}":  s_signed,
        f"rl_abs_{tag}":     s_abs,
    }

# =============================================================================
# SR config injection (common/component/overrides by (Symbol, TF))
# =============================================================================
# استنتاج اختیاری Symbol/TF (اگر در کانفیگ باشد)
def _infer_symbol_tf(cfg_all, default_sym=None, default_tf=None):
    # کلیدهای رایج؛ اگر نبودند، overrides اعمال نمی‌شود (بدون خطا)
    sym_keys = ["data.symbol", "dataset.symbol", "active.symbol", "symbol"]
    tf_keys  = ["data.base_timeframe", "dataset.base_timeframe", "active.timeframe", "timeframe", "base_timeframe"]
    sym = next(( _deep_get(cfg_all, k) for k in sym_keys if _deep_get(cfg_all, k) is not None), default_sym)
    tf  = next(( _deep_get(cfg_all, k) for k in tf_keys  if _deep_get(cfg_all, k) is not None), default_tf)
    return sym, tf


def merge_sr_kwargs(name: str, cfg: dict, df: pd.DataFrame) -> dict:
    # cfg_all = _loader.get_all()
    cfg_all = cfg
    # 1) مشترک (فقط کلیدهای عمومی)
    base = _deep_get(cfg_all, "features.support_resistance.sr_advanced.common", {}) or {}
    # 2) پیش‌فرض‌های سطح کامپوننت (fvg/supply_demand/...)
    comp = _deep_get(cfg_all, f"features.support_resistance.sr_advanced.{name}", {}) or {}
    # 3) پروفایل اختیاری (Symbol/TF) اگر تعریف شده باشد
    sym, tf = _infer_symbol_tf(cfg_all)
    over = {}
    if sym and tf:
        over = _deep_get(cfg_all, f"features.support_resistance.sr_advanced.overrides.{sym}.{tf}.{name}", {}) or {}
    # 4) ادغام نهایی: پیش‌فرض کد ← base ← component ← overrides ← پارامترهای صریح کاربر
    return {**base, **comp, **over, **(cfg or {})}

