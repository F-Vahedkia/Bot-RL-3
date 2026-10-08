# f04_features/candle_window_builder.py (9)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from f03_data.mtf_dataset import MTFDataset
from f02_utils.functions.constants import _TF_MINUTES


# =============================================================================
# Constants
# =============================================================================

STANDARD_CANDLE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "spread",
)

SUPPORTED_PRICE_NORMALIZATION: frozenset[str] = frozenset(
    {"price_relative"}
)
SUPPORTED_PRICE_REFERENCE: frozenset[str] = frozenset(
    {"prev_close"}
)
SUPPORTED_VOLUME_NORMALIZATION: frozenset[str] = frozenset(
    {"relative"}
)

# =============================================================================
# Configuration contract
# =============================================================================

@dataclass(frozen=True, slots=True)
class CandleNormalizationConfig:
    enabled: bool = False
    price_method: str = "price_relative"
    price_reference: str = "prev_close"
    volume_method: str = "relative"
    volume_window: int = 20

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CandleNormalizationConfig":
        if config is None:
            raise ValueError("config is required")
        if not isinstance(config, Mapping):
            raise TypeError(
                f"config must be a mapping, got {type(config).__name__}"
            )

        candles_cfg = config.get("features", {}).get("candles", {}) or {}
        if not isinstance(candles_cfg, Mapping):
            raise TypeError("features.candles must be a mapping")

        normalization_cfg = candles_cfg.get("normalization", {}) or {}
        if not isinstance(normalization_cfg, Mapping):
            raise TypeError("features.candles.normalization must be a mapping")

        price_cfg = normalization_cfg.get("price", {}) or {}
        volume_cfg = normalization_cfg.get("volume", {}) or {}

        if not isinstance(price_cfg, Mapping):
            raise TypeError("features.candles.normalization.price must be a mapping")
        if not isinstance(volume_cfg, Mapping):
            raise TypeError("features.candles.normalization.volume must be a mapping")

        return cls(
            enabled=bool(normalization_cfg.get("enabled", False)),
            price_method=str(price_cfg.get("method", "price_relative")).strip().lower(),
            price_reference=str(price_cfg.get("reference", "prev_close")).strip().lower(),
            volume_method=str(volume_cfg.get("method", "relative")).strip().lower(),
            volume_window=_validate_positive_int(
                volume_cfg.get("window", 20),
                "features.candles.normalization.volume.window",
            ),
        )

    def validate(self) -> None:
        if self.price_method not in SUPPORTED_PRICE_NORMALIZATION:
            raise ValueError(
                f"Unsupported candle price normalization method "
                f"{self.price_method!r}. Supported methods: "
                f"{sorted(SUPPORTED_PRICE_NORMALIZATION)!r}"
            )

        if self.price_reference not in SUPPORTED_PRICE_REFERENCE:
            raise ValueError(
                f"Unsupported candle price reference "
                f"{self.price_reference!r}. Supported references: "
                f"{sorted(SUPPORTED_PRICE_REFERENCE)!r}"
            )

        if self.volume_method not in SUPPORTED_VOLUME_NORMALIZATION:
            raise ValueError(
                f"Unsupported candle volume normalization method "
                f"{self.volume_method!r}. Supported methods: "
                f"{sorted(SUPPORTED_VOLUME_NORMALIZATION)!r}"
            )

        _validate_positive_int(
            self.volume_window,
            "volume_window",
        )


# =============================================================================
# Generic validation / config helpers
# =============================================================================

def _validate_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{field_name} must be an integer, got {value!r}"
        ) from exc
    if result <= 0:
        raise ValueError(
            f"{field_name} must be > 0, got {result}"
        )
    return result


def _get_candles_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if config is None:
        raise ValueError("config is required")
    if not isinstance(config, Mapping):
        raise TypeError(
            f"config must be a mapping, got {type(config).__name__}"
        )

    features_cfg = config.get("features", {}) or {}
    if not isinstance(features_cfg, Mapping):
        raise TypeError("features must be a mapping")

    candles_cfg = features_cfg.get("candles", {}) or {}
    if not isinstance(candles_cfg, Mapping):
        raise TypeError("features.candles must be a mapping")

    return candles_cfg


def _resolve_symbol_value(
    mapping: Mapping[str, Any],
    symbol: str,
) -> Any:
    if symbol in mapping:
        return mapping[symbol]

    upper_symbol = str(symbol).upper()
    if upper_symbol in mapping:
        return mapping[upper_symbol]

    for key, value in mapping.items():
        if str(key).upper() == upper_symbol:
            return value

    return None


def resolve_required_candles(
    config: Mapping[str, Any],
    symbol: str,
) -> Dict[str, int]:
    """
    Return the configured observation Candle-window depth per timeframe.

    Source:
        features.candles.required[symbol]

    An absent/empty mapping means that no Candle columns are requested.
    The mapping preserves configuration insertion order.
    """
    candles_cfg = _get_candles_config(config)
    required_cfg = candles_cfg.get("required", {}) or {}
    if not isinstance(required_cfg, Mapping):
        raise TypeError("features.candles.required must be a mapping")

    symbol_cfg = _resolve_symbol_value(required_cfg, symbol)
    if symbol_cfg is None:
        return {}
    if not isinstance(symbol_cfg, Mapping):
        raise TypeError(
            f"features.candles.required[{symbol!r}] must be a mapping"
        )

    result: Dict[str, int] = {}
    for raw_tf, raw_count in symbol_cfg.items():
        tf = str(raw_tf).strip().upper()
        if not tf:
            raise ValueError("Candle timeframe cannot be empty")
        if tf not in _TF_MINUTES:
            raise ValueError(
                f"Unsupported candle timeframe {raw_tf!r} for symbol={symbol!r}"
            )

        count = _validate_positive_int(
            raw_count,
            f"features.candles.required[{symbol!r}][{tf!r}]",
        )
        result[tf] = count

    return result


def resolve_candle_names(
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    """
    Return Candle source/output names exactly in configuration order.

    Empty/missing names means that Candle observation is disabled by selection,
    even when Candle required-bar configuration exists.
    """
    candles_cfg = _get_candles_config(config)
    raw_names = candles_cfg.get("names", []) or []

    if isinstance(raw_names, (str, bytes)):
        raise TypeError("features.candles.names must be a sequence of strings")

    if not isinstance(raw_names, Sequence):
        raise TypeError("features.candles.names must be a sequence of strings")

    result: list[str] = []
    seen: set[str] = set()

    for raw_name in raw_names:
        name = str(raw_name).strip().lower()
        if not name:
            continue
        if name not in STANDARD_CANDLE_COLUMNS:
            raise ValueError(
                f"Unsupported candle column {raw_name!r}. Supported columns: "
                f"{list(STANDARD_CANDLE_COLUMNS)!r}"
            )
        if name not in seen:
            result.append(name)
            seen.add(name)

    return tuple(result)


# =============================================================================
# Candle normalization primitives
# =============================================================================

def normalize_price_relative(
    df: pd.DataFrame,
    *,
    reference: str = "prev_close",
) -> pd.DataFrame:
    """
    Normalize price-based Candle columns relative to the previous close.

    Formula:

        normalized = (value / previous_close) - 1

    The previous close is taken from the previous Candle in the same
    timeframe, not from the previous Observation timestamp.

    The input frame is never modified.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected pandas.DataFrame, got {type(df).__name__}"
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Candle normalization requires a DatetimeIndex")

    reference = str(reference).strip().lower()
    if reference not in SUPPORTED_PRICE_REFERENCE:
        raise ValueError(
            f"Unsupported price reference {reference!r}. Supported references: "
            f"{sorted(SUPPORTED_PRICE_REFERENCE)!r}"
        )

    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"Price normalization requires missing columns: {missing}"
        )

    result = df.copy()

    previous_close = pd.to_numeric(
        result["close"],
        errors="coerce",
    ).shift(1)
    previous_close = previous_close.replace(0, np.nan)

    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(
            result[column],
            errors="coerce",
        )
        result[column] = values.div(previous_close).sub(1.0)

    if "spread" in result.columns:
        spread = pd.to_numeric(
            result["spread"],
            errors="coerce",
        )
        result["spread"] = spread.div(previous_close)

    return result


def normalize_volume_relative(
    df: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.DataFrame:
    """
    Normalize volume against a trailing mean of previous volumes.

    Formula:

        volume_t / mean(volume_(t-window) ... volume_(t-1))

    The current volume is excluded from the denominator, so the operation is
    causal and identical in semantics for Batch and Live snapshots.

    At the beginning of the available history, fewer than ``window`` prior
    values may be available. Once at least one prior value exists, the mean of
    the available prior values is used. With no prior value, NaN is produced.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected pandas.DataFrame, got {type(df).__name__}"
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Candle normalization requires a DatetimeIndex")

    window = _validate_positive_int(window, "volume window")

    if "volume" not in df.columns:
        raise KeyError("Volume normalization requires column 'volume'")

    result = df.copy()

    volume = pd.to_numeric(
        result["volume"],
        errors="coerce",
    )

    previous_mean = (
        volume.shift(1)
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
        .replace(0, np.nan)
    )

    result["volume"] = volume.div(previous_mean)
    return result


def normalize_candle_frame(
    df: pd.DataFrame,
    *,
    normalization: CandleNormalizationConfig,
) -> pd.DataFrame:
    """
    Return a normalized/raw Candle frame according to configuration.

    When ``enabled`` is False, a copy of the original frame is returned.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected pandas.DataFrame, got {type(df).__name__}"
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Candle processing requires a DatetimeIndex")

    normalization.validate()

    if not normalization.enabled:
        return df.copy()

    result = normalize_price_relative(
        df,
        reference=normalization.price_reference,
    )

    result = normalize_volume_relative(
        result,
        window=normalization.volume_window,
    )

    return result


# =============================================================================
# Candle Window Builder
# =============================================================================

class CandleWindowBuilder:
    """
    Build the Candle-history portion of the final Observation.

    Contract:

        CandleWindowBuilder does NOT decide the Observation timeframe.

        The ObservationBuilder is responsible for creating the final
        observation_index according to the Observation Timeframe Resolution
        policy.

        For each timestamp T in the supplied observation_index and each
        requested candle timeframe TF, select the N most recent candles
        whose close time is <= T.

        lag0 = newest closed Candle

        lag1 = previous closed Candle

        ...

        lag(N-1) = oldest Candle in the requested window

    Input contract:

        observation_index is the authoritative Observation timeline.

        CandleWindowBuilder must never replace, resample, shrink, or extend
        this index.

    Output columns:

        <TF>_lag<k>_<name>

    The input MTFDataset is never mutated.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        if config is None:
            raise ValueError("config is required")
        if not isinstance(config, Mapping):
            raise TypeError(
                f"config must be a mapping, got {type(config).__name__}"
            )

        self.config = config
        self.normalization = CandleNormalizationConfig.from_config(config)
        self.normalization.validate()
        self._required_by_tf_cache: dict[str, Dict[str, int]] = {}
        self._names = resolve_candle_names(config)

    # -------------------------------------------------------------------------
    def required_for_symbol(self, symbol: str) -> Dict[str, int]:
        cache_key = str(symbol).upper()
        if cache_key not in self._required_by_tf_cache:
            self._required_by_tf_cache[cache_key] = resolve_required_candles(
                self.config,
                symbol,
            )
        return dict(self._required_by_tf_cache[cache_key])

    # -------------------------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    # -------------------------------------------------------------------------
    def build(
        self,
        dataset: MTFDataset,
        observation_index: pd.DatetimeIndex,
        *,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """
        Build Candle-window columns aligned exactly to ``observation_index``.

        If Candle names or required timeframe configuration is empty, an empty
        DataFrame with the supplied observation index is returned.

        Missing history is represented by NaN. A requested timeframe missing
        from the MTFDataset is also represented by its configured columns filled
        with NaN; no cross-timeframe merge is performed here.
        """
        self._validate_dataset(dataset)
        self._validate_observation_index(observation_index)

        resolved_symbol = symbol if symbol is not None else dataset.symbol
        if str(dataset.symbol).upper() != str(resolved_symbol).upper():
            raise ValueError(
                f"Dataset symbol {dataset.symbol!r} does not match requested "
                f"symbol {resolved_symbol!r}"
            )

        required_by_tf = self.required_for_symbol(resolved_symbol)

        if not required_by_tf or not self._names:
            return pd.DataFrame(index=observation_index.copy())

        result_parts: list[pd.DataFrame] = []

        # Config insertion order is intentionally preserved.
        for tf, window_size in required_by_tf.items():
            part = self._build_timeframe_window(
                dataset=dataset,
                timeframe=tf,
                window_size=window_size,
                observation_index=observation_index,
            )
            result_parts.append(part)

        if not result_parts:
            return pd.DataFrame(index=observation_index.copy())

        return pd.concat(result_parts, axis=1)

    # -------------------------------------------------------------------------
    def _build_timeframe_window(
        self,
        *,
        dataset: MTFDataset,
        timeframe: str,
        window_size: int,
        observation_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        tf = timeframe.upper()
        columns = [
            f"{tf}_lag{lag}_{name}"
            for lag in range(window_size)
            for name in self._names
        ]

        source = self._get_frame(dataset, tf)

        # Preserve the public observation shape even when the requested TF is
        # unavailable in the current snapshot.
        if source is None or source.empty:
            return pd.DataFrame(
                np.nan,
                index=observation_index,
                columns=columns,
                dtype=float,
            )

        source = source.sort_index().copy()
        self._validate_source_index_compatibility(
            source.index,
            observation_index,
            timeframe=tf,
        )
        source = self._prepare_source_frame(source)

        normalized = normalize_candle_frame(
            source,
            normalization=self.normalization,
        )

        # Candle index contains OPEN timestamps. A Candle is visible at T only
        # after its close timestamp has reached T.
        candle_close_times = normalized.index + pd.Timedelta(
            minutes=_TF_MINUTES[tf]
        )

        observation_values = observation_index.to_numpy()
        close_values = candle_close_times.to_numpy()

        positions = np.searchsorted(
            close_values,
            observation_values,
            side="right",
        ) - 1

        output = pd.DataFrame(
            np.nan,
            index=observation_index,
            columns=columns,
            dtype=float,
        )

        numeric_cache: dict[str, np.ndarray] = {}
        for name in self._names:
            numeric_cache[name] = pd.to_numeric(
                normalized[name],
                errors="coerce",
            ).to_numpy(dtype=float)

        for lag in range(window_size):
            source_positions = positions - lag
            valid = source_positions >= 0
            if not np.any(valid):
                continue

            # Guard against malformed/non-monotonic close-time data even though
            # the normal Data-layer contract should already exclude it.
            valid &= source_positions < len(normalized)
            if not np.any(valid):
                continue

            target_rows = np.flatnonzero(valid)
            source_rows = source_positions[valid]

            for name in self._names:
                column_name = f"{tf}_lag{lag}_{name}"
                output.loc[output.index[target_rows], column_name] = (
                    numeric_cache[name][source_rows]
                )

        return output

    # -------------------------------------------------------------------------
    @staticmethod
    def _prepare_source_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Return a clean, non-mutating source frame for Candle processing."""
        result = df.copy()

        # Source contract from f03_data is already expected to be sorted and
        # duplicate-free. These operations only make the boundary defensive.
        if not result.index.is_monotonic_increasing:
            result = result.sort_index()
        if result.index.has_duplicates:
            result = result[~result.index.duplicated(keep="last")]

        return result

    # -------------------------------------------------------------------------
    @staticmethod
    def _get_frame(
        dataset: MTFDataset,
        timeframe: str,
    ) -> pd.DataFrame | None:
        try:
            return dataset.get(timeframe)
        except KeyError:
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_dataset(dataset: MTFDataset) -> None:
        if dataset is None:
            raise ValueError("dataset is required")
        if not isinstance(dataset, MTFDataset):
            raise TypeError(
                f"Expected MTFDataset, got {type(dataset).__name__}"
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_observation_index(
        observation_index: pd.DatetimeIndex,
    ) -> None:
        if observation_index is None:
            raise ValueError("observation_index is required")
        if not isinstance(observation_index, pd.DatetimeIndex):
            raise TypeError(
                "observation_index must be a pandas.DatetimeIndex"
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_source_index_compatibility(
        source_index: pd.DatetimeIndex,
        observation_index: pd.DatetimeIndex,
        *,
        timeframe: str,
    ) -> None:
        source_tz = source_index.tz
        observation_tz = observation_index.tz

        if (source_tz is None) != (observation_tz is None):
            raise ValueError(
                f"Timezone awareness mismatch for candle timeframe {timeframe}: "
                f"source_tz={source_tz!r}, observation_tz={observation_tz!r}"
            )

        if source_tz is not None and str(source_tz) != str(observation_tz):
            raise ValueError(
                f"Timezone mismatch for candle timeframe {timeframe}: "
                f"source_tz={source_tz!r}, observation_tz={observation_tz!r}"
            )


# =============================================================================
# Functional API
# =============================================================================

def build_candle_window(
    dataset: MTFDataset,
    observation_index: pd.DatetimeIndex,
    config: Mapping[str, Any],
    *,
    symbol: str | None = None,
) -> pd.DataFrame:
    """
    Functional wrapper around CandleWindowBuilder.build().

    This is the preferred boundary function for ObservationBuilder so the
    current ObservationBuilder API does not need to expose a new object type.
    """
    builder = CandleWindowBuilder(config)
    return builder.build(
        dataset,
        observation_index,
        symbol=symbol,
    )


__all__ = [
    "STANDARD_CANDLE_COLUMNS",
    "CandleNormalizationConfig",
    "resolve_required_candles",
    "resolve_candle_names",
    "normalize_price_relative",
    "normalize_volume_relative",
    "normalize_candle_frame",
    "CandleWindowBuilder",
    "build_candle_window",
]
# ============================================================================= END
