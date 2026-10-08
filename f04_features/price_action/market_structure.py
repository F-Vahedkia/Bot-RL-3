# f04_features/price_action/market_structure.py
# Modified at 1404/12/07

"""
Market Structure Features (Price Action)
========================================
- تشخیص سوئینگ‌های HH, HL, LH, LL
- شناسایی BOS و CHoCH
"""

import pandas as pd
import numpy as np
from typing import Literal

from f04_features.indicators.zigzag import zigzag_wrapper as zigzag, zigzag_mtf_adapter
from f04_features.indicators.utils import compute_atr
from f04_features.indicators.levels import compute_adr

# ============================================================================= Test at 04/12/08
# Swing Detection
# ============================================================================= Func1
def detect_swings(
    df: pd.DataFrame,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01
) -> pd.DataFrame:
    """
    Detect swing highs and lows using zigzag indicator.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLC dataframe.
    **zigzag_kwargs :
        Parameters forwarded directly to zigzag().

    Returns
    -------
    pd.DataFrame
        Columns:
            - swing_high (bool)
            - swing_low  (bool)
            - swing_price (float)
    """

    # --- Run zigzag ---
    zz = zigzag(
        df["high"],
        df["low"],
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        point=point,
    )

    # --- Extract swing states ---
    swing_high = zz["state"] == 1
    swing_low  = zz["state"] == -1

    # --- Build result ---
    result = pd.DataFrame(index=df.index)

    result["swing_high"] = swing_high
    result["swing_low"]  = swing_low

    result["swing_price"] = np.where(
        swing_high,
        zz["high"],
        np.where(swing_low, zz["low"], np.nan)
    )

    return result


# ============================================================================= Test at 04/12/08
# Market Structure (HH / HL / LH / LL)
# ============================================================================= Func2
def build_market_structure(
    df: pd.DataFrame,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01
) -> pd.DataFrame:

    swings = detect_swings(
        df,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        point=point,
    )

    result = swings.copy()

    result["HH"] = False
    result["HL"] = False
    result["LH"] = False
    result["LL"] = False

    pivots = result[result["swing_high"] | result["swing_low"]]

    if len(pivots) < 2:
        return result

    last_high_price = None
    last_low_price = None

    for idx, row in pivots.iterrows():

        price = row["swing_price"]

        # ---------------------------
        # HIGH pivot (state == +1)
        # ---------------------------
        if row["swing_high"]:

            if last_high_price is not None:
                if price > last_high_price:
                    result.at[idx, "HH"] = True
                elif price < last_high_price:
                    result.at[idx, "LH"] = True

            last_high_price = price

        # ---------------------------
        # LOW pivot (state == -1)
        # ---------------------------
        elif row["swing_low"]:

            if last_low_price is not None:
                if price > last_low_price:
                    result.at[idx, "HL"] = True
                elif price < last_low_price:
                    result.at[idx, "LL"] = True

            last_low_price = price

    return result


# ============================================================================= Test at 04/12/08
# BOS / CHOCH Detection
# ============================================================================= Func3
def detect_bos_choch(
    structure: pd.DataFrame,
    price_df: pd.DataFrame,
    eps: float = 1e-6
) -> pd.DataFrame:

    out = structure.copy()

    out["bos_up"] = 0
    out["bos_down"] = 0
    out["choch_up"] = 0
    out["choch_down"] = 0

    last_swing_high = None
    last_swing_low = None
    regime = 0

    for idx, row in out.iterrows():

        close_price = price_df.loc[idx, "close"]

        if row.get("swing_high", False) and not pd.isna(row.get("swing_price")):
            last_swing_high = row["swing_price"]

        if row.get("swing_low", False) and not pd.isna(row.get("swing_price")):
            last_swing_low = row["swing_price"]

        if last_swing_high is not None and close_price >= last_swing_high * (1 + eps):
            if regime == 1:
                out.at[idx, "bos_up"] = 1
            else:
                out.at[idx, "choch_up"] = 1
            regime = 1

        elif last_swing_low is not None and close_price <= last_swing_low * (1 - eps):
            if regime == -1:
                out.at[idx, "bos_down"] = 1
            else:
                out.at[idx, "choch_down"] = 1
            regime = -1

    return out


# ============================================================================= Test at 04/12/08
# Unified Pipeline (Optional Convenience)
# ============================================================================= Func4
def market_structure_pipeline(
    df: pd.DataFrame,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01
) -> pd.DataFrame:
    """
    Full pipeline:
        zigzag → swings → structure → bos/choch
    """

    structure = build_market_structure(
        df,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        point=point,
    )
    final = detect_bos_choch(structure, df)

    return final


# ============================================================================= Test at 04/12/08
# Regime State Machine (Bull / Bear)
# ============================================================================= Func5
def build_regime_state(
    df: pd.DataFrame
) -> pd.DataFrame:
    
    """
    Build persistent market regime state machine.

    Requires:
        bos_up, bos_down, choch_up, choch_down

    Returns
    -------
    pd.DataFrame
        Adds:
            - regime (int)
                1  = bullish
               -1  = bearish
                0  = neutral (before first structure event)
    """

    out = df.copy()
    regime = 0
    regimes = []

    for _, row in out.iterrows():
        if row.get("bos_up", 0) == 1 or row.get("choch_up", 0) == 1:
            regime = 1
        elif row.get("bos_down", 0) == 1 or row.get("choch_down", 0) == 1:
            regime = -1
        regimes.append(regime)
    out["regime"] = regimes

    return out


# ============================================================================= Test at 04/12/08
# Advanced Regime Builder - Global Class
# ============================================================================= Func6
def build_regime_state_pro(
    df: pd.DataFrame,
    atr_window: int = 14,
    adr_window: int = 14,
    tf_higher: str = "5min",
    smoothing: Literal["atr", "adr", None] = "atr",
    atr_method: Literal["classic", "wilder", "ema"] = "wilder",
) -> pd.DataFrame:
    """
    Advanced Market Regime Builder (Bull / Bear / Neutral)
    ----------------------------------------
    - Uses market_structure_pipeline for swing, HH/HL/LH/LL, BOS/CHoCH
    - Smooths regime triggers using ATR, ADR, or Multi-Timeframe ZigZag
    - Outputs 'regime' column:
        0  = neutral (before first trigger)
        1  = bullish
       -1  = bearish

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: high, low, close
    atr_window : int
        Window for ATR smoothing
    adr_window : int
        Window for ADR smoothing
    tf_higher : str
        Higher timeframe for ZigZag smoothing
    smoothing : str
        Method for smoothing triggers: 'atr', 'adr', None
    atr_method : str
        Method for compute_atr: 'classic', 'wilder', 'ema'

    Returns
    -------
    pd.DataFrame
        Adds 'regime' column
    """

    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("DF must contain high, low, close")

    # --- Step 1: Run pipeline to get structure + BOS/CHoCH ---
    structure = market_structure_pipeline(df)

    # --- Step 2: Prepare smoothing series if needed ---
    if smoothing == "atr":
        smooth_factor = compute_atr(df, window=atr_window, method=atr_method)
    elif smoothing == "adr":
        smooth_factor = compute_adr(df, window=adr_window)
    elif smoothing is None:
        smooth_factor = pd.Series(1.0, index=df.index, dtype="float32")
    else:
        raise ValueError(f"Unknown smoothing: {smoothing!r}")

    # Prevent division by zero
    smooth_factor = smooth_factor.replace(0, 1e-8)

    # --- Step 3: Generate raw trigger series ---
    bull_trigger = ((structure["bos_up"] + structure["choch_up"]) / smooth_factor).fillna(0)
    bear_trigger = ((structure["bos_down"] + structure["choch_down"]) / smooth_factor).fillna(0)

    # --- Optional: Multi-Timeframe ZigZag smoothing ---
    zz_bull = zigzag_mtf_adapter(df["high"], df["low"], tf_higher, mode="forward_fill")
    zz_bear = -1 * zigzag_mtf_adapter(df["high"], df["low"], tf_higher, mode="forward_fill")

    # Combine triggers with MTF ZigZag
    bull_trigger += (zz_bull > 0).astype(float)
    bear_trigger += (zz_bear < 0).astype(float)

    # --- Step 4: Build regime series ---
    regime = pd.Series(0, index=df.index, dtype="int8")
    current = 0
    for idx in df.index:
        if bull_trigger.loc[idx] > 0:
            current = 1
        elif bear_trigger.loc[idx] > 0:
            current = -1
        regime.loc[idx] = current

    # --- Step 5: Attach to dataframe and return ---
    out = df.copy()
    out["regime_state_pro"] = regime

    return out


