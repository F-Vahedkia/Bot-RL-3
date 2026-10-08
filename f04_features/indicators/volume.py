# -*- coding: utf-8 -*-
# f04_features/indicators/volume.py
# Status in (Bot-RL-3): Completed

"""اندیکاتورهای حجمی: VWAP (روزانه/رولینگ)، ADL، Chaikin Money Flow"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

# VWAP با ریست روزانه (UTC)
def vwap_daily(high, low, close, volume):
    tp = (high + low + close) / 3.0
    day = pd.to_datetime(close.index).tz_convert("UTC").date
    df = pd.DataFrame({"tp": tp, "vol": volume}, index=close.index)
    ctp = (df["tp"] * df["vol"]).groupby(day).cumsum()
    cvol = df["vol"].groupby(day).cumsum().replace(0, np.nan)
    return (ctp / cvol).astype("float32")

# VWAP رولینگ (پنجره n)
def vwap_rolling(high, low, close, volume, n: int = 100):
    tp = (high + low + close) / 3.0
    num = (tp * volume).rolling(n, min_periods=n).sum()
    den = volume.rolling(n, min_periods=n).sum().replace(0, np.nan)
    return (num/den).astype("float32")

# Accumulation/Distribution Line (ADL)
def adl(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = (mfm * volume.fillna(0.0))
    return mfv.cumsum().astype("float32")

# Chaikin Money Flow (CMF)
def cmf(high, low, close, volume, n: int = 20):
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = (mfm * volume.fillna(0.0))
    num = mfv.rolling(n, min_periods=n).sum()
    den = volume.rolling(n, min_periods=n).sum().replace(0, np.nan)
    return (num/den).astype("float32")

def registry() -> Dict[str, callable]:
    
    def make_vwap_daily(df, **_):
        return {"vwap_daily": vwap_daily(df["high"], df["low"], df["close"], df["volume"])}
    def make_vwap_roll(df, n: int = 100, **_):
        return {f"vwap_roll_{n}": vwap_rolling(df["high"], df["low"], df["close"], df["volume"], n)}
    def make_adl(df, **_):
        return {"adl": adl(df["high"], df["low"], df["close"], df["volume"])}
    def make_cmf(df, n: int = 20, **_):
        return {f"cmf_{n}": cmf(df["high"], df["low"], df["close"], df["volume"], n)}
    
    return {
        "vwap": make_vwap_daily,
        "vwap_roll": make_vwap_roll,
        "adl": make_adl,
        "cmf": make_cmf,
    }

