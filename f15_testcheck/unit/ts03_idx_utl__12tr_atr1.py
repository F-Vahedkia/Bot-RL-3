# f15_testcheck/unit/ts03_idx_utl__10tr_atr1.py
# Run: python -m f15_testcheck.unit.ts03_idx_utl__10tr_atr1

import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime
from f04_features.indicators.utils import (
    true_range        as tr_utils_hybrid,
    true_range_pandas as tr_utils_pandas,
    compute_atr       as atr_utils,
)
from f04_features.indicators.core import atr as atr_core

# --- NUMBA VERSION ---------------------------------------------------------
@njit
def _tr_numba_core(high, low, close):
    length = high.size
    tr = np.empty(length)
    tr[0] = high[0] - low[0]
    for i in range(1, length):
        hl = abs(high[i] - low[i])
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    return tr

def _tr_numba(high, low, close):
    idx = close.index
    high = high.to_numpy(np.float64)
    low = low.to_numpy(np.float64)
    close = close.to_numpy(np.float64)
    result = _tr_numba_core(high, low, close)
    return pd.Series(result, index=idx, dtype="float64")


# --- NUMPY VERSION ---------------------------------------------------------
def _tr_numpy(high, low, close):
    hl = np.abs(high - low)
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    hc.iloc[0] = hl.iloc[0]
    lc.iloc[0] = hl.iloc[0]
    result = np.maximum(hl, np.maximum(hc, lc))
    return pd.Series(result, index=close.index, dtype="float64")

# --- PANDAS VERSION --------------------------------------------------------
def _tr_pandas(high, low, close):
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    result = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return pd.Series(result, index=close.index, dtype="float64")


# --- HYBRID WRAPPER --------------------------------------------------------
def true_range(high, low, close, th_pandas=50_000, th_numpy=500_000):
    """
    Auto-select: Pandas < NumPy < Numba
    - if input is pandas.Series and len < th_pandas → pandas
    - if numpy arrays and len < th_numpy → numpy
    - if very large → numba
    """

    size = len(high)

    # Pandas path
    if isinstance(high, pd.Series) and size <= th_pandas:
        return _tr_pandas(high, low, close)

    # Convert pandas → numpy if needed
    if isinstance(high, pd.Series):
        high = high.to_numpy(dtype=np.float64)
        low = low.to_numpy(dtype=np.float64)
        close = close.to_numpy(dtype=np.float64)

    # NumPy path
    if size <= th_numpy:
        return _tr_numpy(high, low, close)

    # Numba path (only profitable for very large arrays)
    return _tr_numba(high, low, close)

###############################################################################

def atr_wilder(tr: pd.Series, n: int) -> pd.Series:
    tr = tr.astype("float64")
    out = tr.copy()

    # step-1: first_atr = average of first n candles
    out.iloc[:n] = np.nan
    if len(tr) >= n:
        first_atr = tr.iloc[:n].mean()
        out.iloc[n-1] = first_atr

        # step-2: RMA recursion
        prev = first_atr
        for i in range(n, len(tr)):
            prev = (prev * (n - 1) + tr.iloc[i]) / n
            out.iloc[i] = prev

    return out.astype("float64")

def atr_wilder_wrp(df: pd.DataFrame, n: int = 14):
    tr = _tr_numba(high=df["high"], low=df["low"], close=df["close"])
    return atr_wilder(tr=tr, n=n)

############################################################################### +++++
@njit
def _true_range_numba(high, low, close):
    length = high.size
    tr = np.empty(length)
    tr[0] = high[0] - low[0]  # first bar: standard TR definition

    for i in range(1, length):
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i - 1])
        l_pc = abs(low[i] - close[i - 1])
        tr[i] = max(h_l, h_pc, l_pc)
    return tr

def tr_wrapper(high: pd.Series, low: pd.Series, close: pd.Series):
    idx = close.index
    high = high.to_numpy(np.float64)
    low = low.to_numpy(np.float64)
    close = close.to_numpy(np.float64)
    out = _true_range_numba(high, low, close)
    return pd.Series(out, index=idx, dtype="float64")   #, name="tr_wrapper")

@njit
def _rma_wilder(tr, window):
    n = len(tr)
    out = np.empty(n, dtype=np.float64)
    alpha = 1.0 / window

    # warmup: SMA initial
    s = 0.0
    for i in range(window):
        s += tr[i]
    r = s / window
    out[window - 1] = r

    # recursive Wilder RMA
    for i in range(window, n):
        r = (tr[i] * alpha) + (r * (1.0 - alpha))
        out[i] = r

    # leading values before window-1 = NaN
    for i in range(window - 1):
        out[i] = np.nan
    return out

@njit
def _sma_classic(tr, window):
    n = len(tr)
    out = np.empty(n, dtype=np.float64)
    s = 0.0

    for i in range(window - 1):
        out[i] = np.nan
        s += tr[i]

    s += tr[window - 1]
    out[window - 1] = s / window

    # sliding window SMA
    for i in range(window, n):
        s += tr[i]
        s -= tr[i - window]
        out[i] = s / window

    return out

@njit
def _ema(tr, window):
    n = len(tr)
    out = np.empty(n, dtype=np.float64)
    alpha = 2.0 / (window + 1)

    # warmup: SMA initial
    s = 0.0
    for i in range(window):
        s += tr[i]
    e = s / window
    out[window - 1] = e

    # recursive EMA
    for i in range(window, n):
        e = (tr[i] * alpha) + (e * (1 - alpha))
        out[i] = e

    # leading NaN
    for i in range(window - 1):
        out[i] = np.nan
    return out


def atr_hybrid(df: pd.DataFrame, window=14, method="wilder"):
    """
    Production-grade ATR function (hybrid interface).

    Parameters
    ----------
    df : ['high', 'low', 'close'] pandas Dataframe
    window : int, ATR window
    method : str, one of ['wilder', 'classic', 'ema']

    Returns
    -------
    ATR : pd.Series float64
    """
    # --- نگهبان‌های ورودی ---
    if window < 1:
        raise ValueError("window must be >= 1")
    if not {"high", "low", "close"}.issubset(set(df.columns)):
        raise ValueError("DF must contain columns: high, low, close")

    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    # --- True Range ---
    tr = _true_range_numba(high, low, close)

    # --- نرمال‌سازی روش ---
    m = (method or "wilder").strip().lower()

    if m == "wilder":
        out = _rma_wilder(tr, window)
    elif m == "classic":
        out = _sma_classic(tr, window)
    elif m == "ema":
        out = _ema(tr, window)
    else:
        raise ValueError(f"Invalid ATR mode: {method}. Use wilder/classic/ema.")

    return pd.Series(out.astype(np.float64), index=df.index, name="atr_hybrid")

############################################################################### +++++
# --- Load data --------------------------------------
t1 = datetime.now()
data_ = pd.read_csv("f03_data/raw/XAUUSD/M1.csv")
t2 = datetime.now()

total_candles = len(data_)
elapsed = round((t2 - t1).total_seconds(), 4)
print(f"Time taken to load data: {elapsed} seconds, length_df:{total_candles}")

# --- preparing data ------------------
data = data_[- total_candles:].copy()
data["time"] = pd.to_datetime(data["time"], utc=True)
data.set_index("time", inplace=True)

# --- Dict of functions ------------------------------
tr_funcs = {
    "tr_utils_hybrid": tr_utils_hybrid,
    "tr_utils_pandas": tr_utils_pandas,
    # "tr_wrapper": tr_wrapper,
    # "_tr_numba" : _tr_numba,
    # "_tr_numpy" : _tr_numpy,
    # "_tr_pandas": _tr_pandas,
}
atr_funcs = {
    # "atr_utils" : atr_utils,
    # "atr_hybrid": atr_hybrid,
    # "atr_core"  : atr_core,
}
# --- list of iterations -----------------------------
# candle_nums = [10_000, 30_000, 100_000, 300_000, 1_000_000, 5_000_000]
candle_nums = [500_010, 1_000_000, 5_000_000]

# --- دیتافریم برای ذخیره همه نتایج ----------------
TR_results = pd.DataFrame()
ATR_results = pd.DataFrame()

for n in candle_nums:
    # --- slicing data ----------------
    df = data[- n:].copy()

    # --- Calling TR functions --------
    for name, func in tr_funcs.items():
        t1 = datetime.now()
        TR_result = func(df["high"], df["low"], df["close"])
        t2 = datetime.now()

        elapsed = round((t2 - t1).total_seconds(), 3)
        print(f"Time taken to run {name}: {elapsed} seconds, length_df:{len(df)}")

        if n==500_010:
            TR_results[f"{name}"] = TR_result.astype("float64")
    print("\n")

    # --- Calling ATR functions -------
    for name, func in atr_funcs.items():
        t1 = datetime.now()
        if name == "atr_core":
            ATR_result = func(df["high"], df["low"], df["close"], n=14)
        else:
            ATR_result = func(df=df, window=14, method="wilder")
        t2 = datetime.now()

        elapsed = round((t2 - t1).total_seconds(), 3)
        print(f"Time taken to run {name}: {elapsed} seconds, length_df:{len(df)}")

        if n==500_010:
            ATR_results[f"{name}"] = ATR_result.astype("float64")
    print("\n")

# --- Save results --------------------------------------------------
# TR_results .to_csv("ts03_idx_utl__10tr_atr1.csv")
# ATR_results.to_csv("ts03_idx_utl__10tr_atr2.csv")
pd.concat([TR_results, ATR_results], axis=1).to_csv("ts03_idx_utl__10tr_atr1.csv")
