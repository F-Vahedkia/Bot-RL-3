# f04_features/time_features/time_feature_engine.py
# freezed:       1405/05/27-15:56
# Date reviewed: 1405/05/26-10:00

# =============================================================================
# Time Feature Engine
# =============================================================================

"""
Time Feature Engine

Responsibilities:
- Apply configured time features to MTFDataset.
- Select features according to configuration.
- Execute registered time-feature functions.

Input:
    MTFDataset

Output:
    MTFDataset

This module does NOT:
- calculate time features directly,
- contain feature formulas,
- manage configuration.
"""

from __future__ import annotations
from typing import Dict, List, Any
import pandas as pd
from f03_data.mtf_dataset import MTFDataset
from f04_features.time_features.time_feature_registry import (
    get_time_feature,
    validate_time_feature_timeframe,
    validate_time_feature_live,
)

# =============================================================================
# Apply Time Features To Dataset
# ============================================================================= 1
def apply_time_features(
    dataset: MTFDataset,
    time_feature_config: Dict[str, List[str]],
) -> MTFDataset:
    """
    Apply configured time features to selected symbol/timeframes.
    Config example:
    {
        "XAUUSD": [
            "hour",
            "day_of_week",
            "sin_hour"
        ]
    }
    """
    if dataset is None:
        raise ValueError("dataset is required")
    symbol = dataset.symbol.upper()
    requested_features = time_feature_config.get(symbol)
    if requested_features is None:
        return dataset
    
    base_tf = dataset.base_tf.upper()
    for feature_name in requested_features:
        validate_time_feature_timeframe(
            feature_name,
            base_tf,
        )

    df = dataset.get(base_tf)
    if df is None or df.empty:
        return dataset

    updated_df = add_time_features_to_df(
        df=df,
        features=requested_features,
    )
    dataset.add(
        base_tf,
        updated_df,
    )
    return dataset

# =============================================================================
# Add Features To One DataFrame
# ============================================================================= 2
def add_time_features_to_df(
    df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    """
    Add selected time features to dataframe.

    Calculation is delegated completely
    to registered functions.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "Time features require DatetimeIndex"
        )
    result = df.copy()
    for feature_name in features:
        spec = get_time_feature(feature_name)
        result = spec.function(result)
        
    return result

# =============================================================================
# Apply Time Features To One Live DataFrame
# ============================================================================= 3
def add_time_features_to_live_df(
    df: pd.DataFrame,
    features: List[str],
    timeframe: str,
    previous_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Apply selected time features to live dataframe.

    The dataframe may contain one newly arrived candle or
    a small live window.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "Time features require DatetimeIndex"
        )
    result = df.copy()
    for feature_name in features:
        spec = validate_time_feature_live(
            feature_name,
            timeframe,
        )

        if feature_name.strip().lower() == "is_new_day":
            result = spec.function(
                result,
                previous_timestamp=previous_timestamp,
            )
        else:
            result = spec.function(result)

    return result

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    "apply_time_features",
    "add_time_features_to_df",
    "add_time_features_to_live_df",
]

# ============================================================================= END