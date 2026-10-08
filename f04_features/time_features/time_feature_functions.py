# f04_features/time_features/time_feature_functions.py
# freezed:       1404/05/27-15:56
# freezed:       1405/05/26-19:28
# Date reviewed: 1405/05/26-10:00

"""
Time Feature Functions
=====================
Pure functions for generating calendar/time based features.
Input:
    pandas.DataFrame
    DatetimeIndex required
Output:
    pandas.DataFrame with added feature columns

No configuration.
No symbol logic.
No timeframe logic.
No orchestration.
"""

# ===================================================================
# Imports
# ===================================================================
from __future__ import annotations
import numpy as np
import pandas as pd

# ===================================================================
# Basic calendar features
# =================================================================== 1-8
def add_hour(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["hour"] = result.index.hour
    return result

def add_minute(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["minute"] = result.index.minute
    return result

def add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["day_of_week"] = result.index.dayofweek
    return result

def add_day_of_month(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["day_of_month"] = result.index.day
    return result

def add_day_of_year(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["day_of_year"] = result.index.dayofyear
    return result

def add_week_of_year(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["week_of_year"] = (
        result.index.isocalendar().week.astype(int)
    )
    return result

def add_month(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["month"] = result.index.month
    return result

def add_quarter(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["quarter"] = result.index.quarter
    return result

# ===================================================================
# Market session features
# =================================================================== 9-11
def add_session_asia(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    hour = result.index.hour
    result["session_asia"] = (
        ((hour >= 0) & (hour < 8))
        .astype(int)
    )
    return result

def add_session_london(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    hour = result.index.hour
    result["session_london"] = (
        ((hour >= 8) & (hour < 16))
        .astype(int)
    )
    return result

def add_session_newyork(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    hour = result.index.hour
    result["session_newyork"] = (
        ((hour >= 13) & (hour < 21))
        .astype(int)
    )
    return result

# ===================================================================
# Calendar boolean features
# =================================================================== 12-16
def add_is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["is_weekend"] = (
        result.index.dayofweek >= 5
    ).astype(int)
    return result

def add_is_month_start(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["is_month_start"] = (
        result.index.is_month_start
    ).astype(int)
    return result

def add_is_month_end(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["is_month_end"] = (
        result.index.is_month_end
    ).astype(int)
    return result

def add_is_quarter_start(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["is_quarter_start"] = (
        result.index.is_quarter_start
    ).astype(int)
    return result

def add_is_quarter_end(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["is_quarter_end"] = (
        result.index.is_quarter_end
    ).astype(int)
    return result

# ===================================================================
# Cyclic encoding features
# =================================================================== 17-22
def add_sin_hour(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["sin_hour"] = np.sin(
        2 * np.pi * result.index.hour / 24
    )
    return result

def add_cos_hour(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["cos_hour"] = np.cos(
        2 * np.pi * result.index.hour / 24
    )
    return result

def add_sin_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["sin_day_of_week"] = np.sin(
        2 * np.pi * result.index.dayofweek / 7
    )
    return result

def add_cos_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["cos_day_of_week"] = np.cos(
        2 * np.pi * result.index.dayofweek / 7
    )
    return result

def add_sin_month(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["sin_month"] = np.sin(
        2 * np.pi * (result.index.month - 1) / 12
    )
    return result

def add_cos_month(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["cos_month"] = np.cos(
        2 * np.pi * (result.index.month - 1) / 12
    )
    return result

# ===================================================================
# Trading specific time features
# =================================================================== 23-25
def add_market_open_flag(
    df: pd.DataFrame,
    open_hour: int = 8
) -> pd.DataFrame:
    result = df.copy()
    result["market_open_flag"] = (
        result.index.hour == open_hour
    ).astype(int)
    return result

def add_market_close_flag(
    df: pd.DataFrame,
    close_hour: int = 21
) -> pd.DataFrame:
    result = df.copy()
    result["market_close_flag"] = (
        result.index.hour == close_hour
    ).astype(int)
    return result

def add_is_new_day_old1(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    previous_day = result.index.to_series().dt.date.shift(1)
    result["is_new_day"] = (
        result.index.to_series().dt.date != previous_day
    ).astype(int).values
    return result

def add_is_new_day(
    df: pd.DataFrame,
    previous_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    
    result = df.copy()

    if not isinstance(result.index, pd.DatetimeIndex):
        raise TypeError("is_new_day requires DatetimeIndex")

    if len(result) == 0:
        result["is_new_day"] = pd.Series(
            dtype="int64",
            index=result.index,
        )
        return result

    if previous_timestamp is not None:
        previous_timestamp = pd.Timestamp(previous_timestamp)

        current_day = result.index[0].date()
        previous_day = previous_timestamp.date()
        values = [int(current_day != previous_day)]
        if len(result) > 1:
            values.extend(
                (
                    result.index[1:].date
                    != result.index[:-1].date
                ).astype(int)
            )
        result["is_new_day"] = values
        return result

    previous_day = result.index.to_series().dt.date.shift(1)

    result["is_new_day"] = (
        result.index.to_series().dt.date != previous_day
    ).astype(int).values

    return result

# ===================================================================
# Export list
# ===================================================================
__all__ = [

    # basic calendar
    "add_hour",
    "add_minute",
    "add_day_of_week",
    "add_day_of_month",
    "add_day_of_year",
    "add_week_of_year",
    "add_month",
    "add_quarter",

    # sessions
    "add_session_asia",
    "add_session_london",
    "add_session_newyork",

    # calendar flags
    "add_is_weekend",
    "add_is_month_start",
    "add_is_month_end",
    "add_is_quarter_start",
    "add_is_quarter_end",

    # cyclic encoding
    "add_sin_hour",
    "add_cos_hour",
    "add_sin_day_of_week",
    "add_cos_day_of_week",
    "add_sin_month",
    "add_cos_month",

    # trading flags
    "add_market_open_flag",
    "add_market_close_flag",
    "add_is_new_day",
]

# ============================================================================= END
