# save as fetch_mt5_registry.py
import MetaTrader5 as mt5
import pandas as pd, numpy as np, datetime as dt
from f04_features.indicators import core as core_module
'''
# --- helpers --------------------------------------------- OK
def mt5_tf_const(tf_str):
    return getattr(mt5, "TIMEFRAME_"+tf_str)
# --------------------------------------------------------- OK
def ensure_init():
    if not mt5.initialize():
        raise RuntimeError("mt5.initialize() failed")
'''
#====================================================================
# Helper
#====================================================================
''' Try to read indicator value from MT5 using common i-functions.
    If not available or fails, raise.
'''
def read_indicator_from_mt5(fn_name, symbol, tf_const, params, n_bars):
    # fn_name examples: "iMA","iRSI","iMomentum","iATR","iMACD","iBands","iStochastic","iCCI","iMFI","iWPR","iSAR"
    fn = getattr(mt5, fn_name, None)
    if fn is None:
        raise AttributeError(f"{fn_name} not in mt5 API")
    out = []
    # mt5 functions typically return a single double for given shift, so we query shift = i for each bar
    # we'll query bars from newest (shift=0) to older; then reverse to align with dataframe order
    for shift in range(n_bars-1, -1, -1):
        try:
            val = fn(symbol, tf_const, *params, shift)
        except Exception:
            # some functions (e.g. iMACD) may return tuple; try calling without extra params
            val = fn(symbol, tf_const, *params, shift)
        out.append(val)
    return np.array(out[::-1])  # now oldest->newest
#====================================================================
# Main Function
#====================================================================
def fetch_registry_from_mt5(symbol: str, timeframe: str, start_iso: str, end_iso: str):
    # --- ensure init -------------------------------------
    if not mt5.initialize():
        raise RuntimeError("mt5.initialize() failed")    
    
    # --- construct mt5 timeframe -------------------------
    tf_const = getattr(mt5, "TIMEFRAME_"+timeframe)

    # --- Getting candles between 2 times -----------------
    start = dt.datetime.fromisoformat(start_iso)
    end = dt.datetime.fromisoformat(end_iso)
    rates = mt5.copy_rates_range(symbol, tf_const, start, end)
    if rates is None or len(rates) == 0:
        raise RuntimeError("no rates returned from mt5")
    
    # --- Transfering to DataFrame ------------------------
    df = pd.DataFrame(rates)
    # --- Changing times to UTC ---------------------------
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    # --- Defining time column as index -------------------
    df.set_index('time', inplace=True)

    # --- map available mt5 volume cols into a single 'volume' column used by core.py
    if ("volume" not in df.columns) and ("tick_volume" in df.columns):
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

    # --- calculating: n, registry, out -------------------
    n = len(df)
    registry = core_module.registry()
    out = df.copy()


    # === Tracking indicator source: 'mt5' or 'core' ===
    sources = {}  # dict: column_name -> 'mt5'|'core'

    # mapping registry name -> attempt to read from mt5 function (name of mt5 fn, params builder)
    mt5_attempts = {
        "sma": ("iMA", lambda period: (period, 0, mt5.MODE_SMA, mt5.PRICE_CLOSE)),
        "ema": ("iMA", lambda period: (period, 0, mt5.MODE_EMA, mt5.PRICE_CLOSE)),
        "wma": ("iMA", lambda period: (period, 0, mt5.MODE_LWMA, mt5.PRICE_CLOSE)),
        "rsi": ("iRSI", lambda period: (period, mt5.PRICE_CLOSE)),
        "roc": ("iMomentum", lambda n: (n, mt5.PRICE_CLOSE)),
        "atr": ("iATR", lambda n: (n,)),
        "macd": ("iMACD", lambda fast, slow, signal: (fast, slow, signal, mt5.PRICE_CLOSE)),
        "bbands": ("iBands", lambda period, k: (period, k, 0, mt5.PRICE_CLOSE)),
        "stoch": ("iStochastic", lambda n, d: (n, d, 3, mt5.STO_LOWHIGH, mt5.PRICE_CLOSE)),
        "cci": ("iCCI", lambda n: (n,)),
        "mfi": ("iMFI", lambda n: (n,)),
        "obv":  ("iOBV", lambda vol_type=1: (vol_type,)),  # builtin OBV
        "wr": ("iWPR", lambda n: (n,)),
        "sar": ("iSAR", lambda af_start, af_step, af_max: (af_start, af_step, af_max)),
        # keltner, ha, tr not direct mt5 functions -> fallback
    }

    for name, make_fn in registry.items():
        # 1) compute core values (always)
        try:
            core_vals = make_fn(df)  # dict or pd.Series
        except Exception as e:
            core_vals = {}
            print(f"core computation failed for {name}: {e}")

        if isinstance(core_vals, dict):
            core_map = core_vals
        else:
            core_map = {name: core_vals}

        # assign core columns with suffix _core
        for col, series in core_map.items():
            out[f"{col}_core"] = series

        '''
        # 2) attempt MT5 read if mapping exists
        mt5_entry = mt5_attempts.get(name)
        if mt5_entry is not None and mt5_entry[0] is not None:
            fn_name, param_builder = mt5_entry
            try:
                params = param_builder()  # use defaults or explicit args
            except TypeError:
                params = ()
            try:
                mt5_res = read_indicator_from_mt5(fn_name, symbol, tf_const, params, n)
                mt5_cols = list(core_map.keys())
                if isinstance(mt5_res, tuple):
                    if len(mt5_res) == len(mt5_cols):
                        for c, arr in zip(mt5_cols, mt5_res):
                            out[f"{c}_mt5"] = arr
                    else:
                        for idx, arr in enumerate(mt5_res):
                            out[f"{name}_mt5_{idx}"] = arr
                else:
                    out[f"{mt5_cols[0]}_mt5"] = mt5_res
            except Exception as e:
                for c in core_map.keys():
                    out[f"{c}_mt5"] = pd.Series([np.nan]*len(out), index=out.index)
                print(f"mt5 read failed for {name} (fn={fn_name} params={params}): {e}")
        else:
            for c in core_map.keys():
                out[f"{c}_mt5"] = pd.Series([np.nan]*len(out), index=out.index)
        '''
    return out

#====================================================================
# Example Usage
#====================================================================
df = fetch_registry_from_mt5("XAUUSD","H1","2025-10-01T00:00:00","2025-10-10T00:00:00")

# Save to CSV in project root
csv_path = "f15_testcheck/fetch_registry_output.csv"
df.to_csv(csv_path, index=False)
print(f"Output saved to {csv_path}")