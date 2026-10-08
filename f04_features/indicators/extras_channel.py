# f04_features/indicators/extras_channel.py
# Status in (Bot-RL-3): Final Reviewed at 1405/01/24

"""Channel and volatility derived features (production-grade)."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

from .core import bollinger, keltner

# =============================================================================
# Utilities
# =============================================================================

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0.0, np.nan)
    return (a / b).astype("float64")

# =============================================================================
# Donchian Channel
# =============================================================================

def donchian(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20):
    high_f = high.astype("float64")
    low_f = low.astype("float64")
    close_f = close.astype("float64")
    
    # --- Channel limits ---
    upper = high_f.rolling(n, min_periods=n).max()
    lower = low_f.rolling(n, min_periods=n).min()
    mid = (upper + lower) / 2.0
    
    # --- Channel width ---
    width = upper - lower
    
    # --- Position of "close" in channel ---
    width_n = width.replace(0.0, np.nan)
    pos = (close_f - lower) / width_n

    # --- Breakout ---
    brk_up = (close > upper.shift(1))   # .astype("float32")
    brk_dn = (close < lower.shift(1))   # .astype("float32")

    return (
        upper.rename(f"donch_up_{n}"),
        mid  .rename(f"donch_md_{n}"),
        lower.rename(f"donch_lo_{n}"),
        width.rename(f"donch_width_{n}"),
        pos  .rename(f"donch_pos_{n}"),
        brk_up.rename(f"donch_brk_up_{n}"),
        brk_dn.rename(f"donch_brk_dn_{n}"),
    )

# =============================================================================
# Chaikin Volatility
# =============================================================================

def chaikin_volatility(high: pd.Series, low: pd.Series, n: int = 10, roc: int = 10):

    hl = high - low
    ema_range = hl.ewm(span=n, adjust=False, min_periods=n).mean()
    prev = ema_range.shift(roc).replace(0.0, np.nan)
    cv = 100.0 * (ema_range - prev) / prev
    return cv.rename(f"chaikin_vol_{n}_{roc}")

# =============================================================================
# Bollinger derived features
# =============================================================================

def bollinger_position(close: pd.Series, n: int = 20, k: float = 2.0):

    upper, mid, lower = bollinger(close, n=n, k=k)
    width = (upper - lower).replace(0.0, np.nan)
    pos = (close - lower) / width
    return pos.rename(f"boll_pos_{n}_{k}")

# =============================================================================
# Keltner derived features
# =============================================================================

def keltner_position(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, atr_mult: float = 2.0):

    upper, mid, lower = keltner(high, low, close, n=n, m=atr_mult)
    width = (upper - lower).replace(0.0, np.nan)
    pos = (close - lower) / width
    return pos.rename(f"kelt_pos_{n}_{atr_mult}")

# =============================================================================
# Registry
# =============================================================================

def registry() -> Dict[str, callable]:

    def cast32(d: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        return {k: pd.Series(v, copy=False).astype("float32") for k, v in d.items()}

    def make_donchian(df, n: int = 20, **_):
        up, mid, lo, width, pos, brk_up, brk_dn = \
            donchian(df["high"], df["low"], df["close"], n)
        return cast32({
            f"donch_up_{n}": up,
            f"donch_mid_{n}": mid,
            f"donch_lo_{n}": lo,
            f"donch_width_{n}": width,
            f"donch_pos_{n}": pos,
            f"donch_brk_up_{n}": brk_up,
            f"donch_brk_dn_{n}": brk_dn,
        })

    def make_ch_vol(df, n: int = 10, roc: int = 10, **_):
        return cast32({
            f"chaikin_vol_{n}_{roc}": chaikin_volatility(df["high"], df["low"], n, roc)
        })

    def make_boll_pos(df, n: int = 20, k: float = 2.0, **_):
        return cast32({
            f"boll_pos_{n}_{k}": bollinger_position(df["close"], n=n, k=k)
        })

    def make_kelt_pos(df, n: int = 20, atr_mult: float = 2.0, **_):
        return cast32({
            f"kelt_pos_{n}_{atr_mult}": keltner_position(
                df["high"], df["low"], df["close"], n=n, atr_mult=atr_mult
            )
        })

    return {
        "donchian": make_donchian,
        "chaikin_vol": make_ch_vol,
        "bollinger_position": make_boll_pos,
        "keltner_position": make_kelt_pos,
    }
