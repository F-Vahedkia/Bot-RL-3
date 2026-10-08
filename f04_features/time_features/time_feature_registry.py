# f04_features/time_features/time_feature_registry.py
# freezed:       1404/05/27-15:56
# Date reviewed: 1405/05/26-10:00

# =============================================================================
# Time Feature Registry
# =============================================================================

"""
Time Feature Registry
=====================

Single source of truth for available time-based features.

Responsibilities:
- Define available time features.
- Connect feature name to calculation function.
- Store metadata of each feature.
- Validate requested features.

This module does NOT:
- modify DataFrames directly,
- read config,
- execute pipeline logic,
- manage datasets.
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Any

from .time_feature_functions import (

    # basic calendar
    add_hour,
    add_minute,
    add_day_of_week,
    add_day_of_month,
    add_day_of_year,
    add_week_of_year,
    add_month,
    add_quarter,

    # sessions
    add_session_asia,
    add_session_london,
    add_session_newyork,

    # calendar flags
    add_is_weekend,
    add_is_month_start,
    add_is_month_end,
    add_is_quarter_start,
    add_is_quarter_end,

    # cyclic
    add_sin_hour,
    add_cos_hour,
    add_sin_day_of_week,
    add_cos_day_of_week,
    add_sin_month,
    add_cos_month,

    # trading flags
    add_market_open_flag,
    add_market_close_flag,
    add_is_new_day,
)

# =============================================================================
# Types
# =============================================================================
TimeFeatureFunction = Callable[..., Any]

# =============================================================================
# Specification
# =============================================================================
@dataclass(frozen=True, slots=True)
class TimeFeatureSpec:
    """
    Definition of one time feature.
    """
    name: str
    function: TimeFeatureFunction
    output_columns: tuple[str, ...]
    valid_timeframes: tuple[str, ...]
    description: str = ""

# =============================================================================
# Registry
# =============================================================================
_all_tfs = [
    "M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15", "M20", "M30",
    "H1", "H2", "H3", "H4", "H6", "H8", "H12",
    "D1", "W1", "MN1",
]

_intraday_tfs = [
    "M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15", "M20", "M30",
    "H1", "H2", "H3", "H4", "H6", "H8", "H12",
]

TIME_FEATURE_TIMEFRAME_GROUP = {
    # ---------------------------------
    # Intraday only
    # ---------------------------------
    "minute": _intraday_tfs,
    "hour": _intraday_tfs,

    "session_asia": _intraday_tfs,
    "session_london": _intraday_tfs,
    "session_newyork": _intraday_tfs,

    "sin_hour": _intraday_tfs,
    "cos_hour": _intraday_tfs,

    "market_open_flag": _intraday_tfs,
    "market_close_flag": _intraday_tfs,

    # ---------------------------------
    # All timeframes
    # ---------------------------------
    "day_of_week": _all_tfs,
    "day_of_month": _all_tfs,
    "day_of_year": _all_tfs,
    "week_of_year": _all_tfs,

    "month": _all_tfs,
    "quarter": _all_tfs,

    "is_weekend": _all_tfs,
    "is_month_start": _all_tfs,
    "is_month_end": _all_tfs,
    "is_quarter_start": _all_tfs,
    "is_quarter_end": _all_tfs,

    "sin_day_of_week": _all_tfs,
    "cos_day_of_week": _all_tfs,

    "sin_month": _all_tfs,
    "cos_month": _all_tfs,

    "is_new_day": _all_tfs,
}

TIME_FEATURE_REGISTRY: Dict[str, TimeFeatureSpec] = {
    # ---------------------------------
    # Basic calendar
    # ---------------------------------
    "minute": TimeFeatureSpec(
        name="minute",
        function=add_minute,
        output_columns=("minute",),
        valid_timeframes=_intraday_tfs,
        description="Minute of hour from timestamp",
    ),
    "hour": TimeFeatureSpec(
        name="hour",
        function=add_hour,
        output_columns=("hour",),
        valid_timeframes=_intraday_tfs,
        description="Hour of day from timestamp",
    ),
    "day_of_week": TimeFeatureSpec(
        name="day_of_week",
        function=add_day_of_week,
        output_columns=("day_of_week",),
        valid_timeframes=_all_tfs,
        description="Day of week number",
    ),
    "day_of_month": TimeFeatureSpec(
        name="day_of_month",
        function=add_day_of_month,
        output_columns=("day_of_month",),
        valid_timeframes=_all_tfs,
        description="Day of month number",
    ),
    "day_of_year": TimeFeatureSpec(
        name="day_of_year",
        function=add_day_of_year,
        output_columns=("day_of_year",),
        valid_timeframes=_all_tfs,
        description="Day of year number",
    ),
    "week_of_year": TimeFeatureSpec(
        name="week_of_year",
        function=add_week_of_year,
        output_columns=("week_of_year",),
        valid_timeframes=_all_tfs,
        description="ISO week number",
    ),
    "month": TimeFeatureSpec(
        name="month",
        function=add_month,
        output_columns=("month",),
        valid_timeframes=_all_tfs,
        description="Month number",
    ),
    "quarter": TimeFeatureSpec(
        name="quarter",
        function=add_quarter,
        output_columns=("quarter",),
        valid_timeframes=_all_tfs,
        description="Quarter number",
    ),

    # ---------------------------------
    # Market sessions
    # ---------------------------------
    "session_asia": TimeFeatureSpec(
        name="session_asia",
        function=add_session_asia,
        output_columns=("session_asia",),
        valid_timeframes=_intraday_tfs,
        description="Asia trading session flag",
    ),
    "session_london": TimeFeatureSpec(
        name="session_london",
        function=add_session_london,
        output_columns=("session_london",),
        valid_timeframes=_intraday_tfs,
        description="London trading session flag",
    ),
    "session_newyork": TimeFeatureSpec(
        name="session_newyork",
        function=add_session_newyork,
        output_columns=("session_newyork",),
        valid_timeframes=_intraday_tfs,
        description="New York trading session flag",
    ),

    # ---------------------------------
    # Calendar flags
    # ---------------------------------
    "is_weekend": TimeFeatureSpec(
        name="is_weekend",
        function=add_is_weekend,
        output_columns=("is_weekend",),
        valid_timeframes=_all_tfs,
        description="Weekend flag",
    ),
    "is_month_start": TimeFeatureSpec(
        name="is_month_start",
        function=add_is_month_start,
        output_columns=("is_month_start",),
        valid_timeframes=_all_tfs,
        description="Month start flag",
    ),
    "is_month_end": TimeFeatureSpec(
        name="is_month_end",
        function=add_is_month_end,
        output_columns=("is_month_end",),
        valid_timeframes=_all_tfs,
        description="Month end flag",
    ),
    "is_quarter_start": TimeFeatureSpec(
        name="is_quarter_start",
        function=add_is_quarter_start,
        output_columns=("is_quarter_start",),
        valid_timeframes=_all_tfs,
        description="Quarter start flag",
    ),
    "is_quarter_end": TimeFeatureSpec(
        name="is_quarter_end",
        function=add_is_quarter_end,
        output_columns=("is_quarter_end",),
        valid_timeframes=_all_tfs,
        description="Quarter end flag",
    ),

    # ---------------------------------
    # Cyclic encoding
    # ---------------------------------
    "sin_hour": TimeFeatureSpec(
        name="sin_hour",
        function=add_sin_hour,
        output_columns=("sin_hour",),
        valid_timeframes=_intraday_tfs,
        description="Cyclic hour encoding sine",
    ),
    "cos_hour": TimeFeatureSpec(
        name="cos_hour",
        function=add_cos_hour,
        output_columns=("cos_hour",),
        valid_timeframes=_intraday_tfs,
        description="Cyclic hour encoding cosine",
    ),
    "sin_day_of_week": TimeFeatureSpec(
        name="sin_day_of_week",
        function=add_sin_day_of_week,
        output_columns=("sin_day_of_week",),
        valid_timeframes=_all_tfs,
        description="Cyclic weekday encoding sine",
    ),
    "cos_day_of_week": TimeFeatureSpec(
        name="cos_day_of_week",
        function=add_cos_day_of_week,
        output_columns=("cos_day_of_week",),
        valid_timeframes=_all_tfs,
        description="Cyclic weekday encoding cosine",
    ),
    "sin_month": TimeFeatureSpec(
        name="sin_month",
        function=add_sin_month,
        output_columns=("sin_month",),
        valid_timeframes=_all_tfs,
        description="Cyclic month encoding sine",
    ),
    "cos_month": TimeFeatureSpec(
        name="cos_month",
        function=add_cos_month,
        output_columns=("cos_month",),
        valid_timeframes=_all_tfs,
        description="Cyclic month encoding cosine",
    ),

    # ---------------------------------
    # Trading flags
    # ---------------------------------
    "market_open_flag": TimeFeatureSpec(
        name="market_open_flag",
        function=add_market_open_flag,
        output_columns=("market_open_flag",),
        valid_timeframes=_intraday_tfs,
        description="Market open candle flag",
    ),
    "market_close_flag": TimeFeatureSpec(
        name="market_close_flag",
        function=add_market_close_flag,
        output_columns=("market_close_flag",),
        valid_timeframes=_intraday_tfs,
        description="Market close candle flag",
    ),
    "is_new_day": TimeFeatureSpec(
        name="is_new_day",
        function=add_is_new_day,
        output_columns=("is_new_day",),
        valid_timeframes=_all_tfs,
        description="First candle of a new day",
    ),
}

# =============================================================================
# Helpers
# =============================================================================
def get_time_feature(name: str) -> TimeFeatureSpec:
    """
    Return specification of one time feature.
    """
    key = name.strip().lower()
    if key not in TIME_FEATURE_REGISTRY:
        raise KeyError(
            f"Unknown time feature: {name}. "
            f"Available: {sorted(TIME_FEATURE_REGISTRY.keys())}"
        )
    return TIME_FEATURE_REGISTRY[key]

# ---------------------------------------------------------
def validate_time_feature_timeframe(
    feature: str,
    timeframe: str,
) -> None:
    """
    Validate whether one time feature is allowed
    on the requested timeframe.
    """

    spec = get_time_feature(feature)

    tf = timeframe.upper()

    if tf not in spec.valid_timeframes:
        raise ValueError(
            f"Time feature '{feature}' "
            f"is not valid for timeframe '{tf}'. "
            f"Allowed timeframes: {spec.valid_timeframes}"
        )
    
# ---------------------------------------------------------
def validate_time_feature_live_old1(
    feature: str,
    timeframe: str,
) -> TimeFeatureSpec:
    """
    Validate that a time feature is executable in live mode.

    Time features are timestamp-derived and therefore use
    the same registered calculation function in batch and live.
    """
    return get_time_feature(feature)

# ---------------------------
def validate_time_feature_live(
    feature: str,
    timeframe: str,
) -> TimeFeatureSpec:
    """
    Validate that a time feature is executable in live mode
    and valid for the requested timeframe.
    """

    spec = get_time_feature(feature)
    tf = timeframe.upper()
    if tf not in spec.valid_timeframes:
        raise ValueError(
            f"Time feature '{feature}' "
            f"is not valid for live timeframe '{tf}'. "
            f"Allowed timeframes: {spec.valid_timeframes}"
        )
    return spec

# ---------------------------------------------------------
def validate_time_features(
    features: list[str],
    timeframe: str | None = None,
) -> FrozenSet[str]:
    """
    Validate requested features from config.
    """
    result = set()
    for feature in features:
        spec = get_time_feature(feature)

        if timeframe is not None:
            tf = timeframe.upper()
            if tf not in spec.valid_timeframes:
                raise ValueError(
                    f"Time feature '{feature}' "
                    f"is not valid for timeframe '{tf}'. "
                    f"Allowed: {spec.valid_timeframes}"
                )

        result.add(spec.name)

    return frozenset(result)

# ---------------------------------------------------------
def available_time_features() -> tuple[str, ...]:
    """
    Return all registered feature names.
    """
    return tuple(sorted(TIME_FEATURE_REGISTRY.keys()))

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    "TimeFeatureSpec",        # dataclass
    "TIME_FEATURE_TIMEFRAME_GROUP",  # dict
    "TIME_FEATURE_REGISTRY",         # dict
    "get_time_feature",                 # function
    "validate_time_feature_timeframe",  # function
    "validate_time_feature_live",       # function
    "validate_time_features",           # function
    "available_time_features",          # function
]

# ============================================================================= END