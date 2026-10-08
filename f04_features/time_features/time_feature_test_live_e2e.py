# f04_features/time_features/time_feature_test_live_e2e.py
# Date reviewed:
#    1405/05/27-15:56 --> run result is OK --> freeze
#
# Run:
#    python -m f04_features.time_features.time_feature_test_live_e2e

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
from f04_features.time_features.time_feature_engine import (
    add_time_features_to_df,
    add_time_features_to_live_df,
)
from f04_features.time_features.time_feature_registry import (
    get_time_feature,
    validate_time_features,
    validate_time_feature_timeframe,
)
from f02_utils.config_completer import config_completer
from f02_utils.config_path_funcs import full_file_path

# =============================================================================
# Configuration
# =============================================================================
SYMBOL = "BITCOIN"
MAX_LIVE_ROWS = 5_000

# =============================================================================
# Load time-feature configuration
# ============================================================================= 1
def load_time_feature_config(cfg) -> list[str]:
    features = (
        cfg["features"]
        ["time_features"]
        ["symbols"]
        [SYMBOL]
    )
    if not isinstance(features, list):
        raise TypeError(
            f"{SYMBOL} time_features configuration must be a list."
        )
    return features

# =============================================================================
# Discover data files
# ============================================================================= 2
def discover_data_files(
    cfg,
    timeframes: list[str],
) -> dict[str, Path]:

    raw_dir_path = Path(cfg["paths"]["raw_dir"])
    files: dict[str, Path] = {}
    for tf in timeframes:
        parquet_path = full_file_path(raw_dir_path, SYMBOL, tf, "parquet")
        if parquet_path.exists():
            files[tf] = parquet_path
            continue
        csv_path = Path(str(parquet_path).replace(".parquet", ".csv"))
        if csv_path.exists():
            files[tf] = csv_path

    if not files:
        raise FileNotFoundError(
            f"No {SYMBOL}_*.parquet or {SYMBOL}_*.csv files "
            f"found in {raw_dir_path}"
        )
    return dict(sorted(files.items()))

# =============================================================================
# Load real OHLCVS dataframe
# ============================================================================= 3
def load_dataframe(path: Path) -> pd.DataFrame:

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)

    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(
            path,
            index_col=0,
        )
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "time"
    else:
        raise ValueError(f"Unsupported data format: {path}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"DataFrame index is not DatetimeIndex:\n{path}")

    if df.index.tz is None:
        raise ValueError(f"DataFrame index is timezone-naive:\n{path}")

    if str(df.index.tz) != "UTC":
        raise ValueError(
            f"DataFrame index is not UTC:\n"
            f"{path}\n"
            f"Detected timezone: {df.index.tz}"
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"DataFrame index is not monotonically increasing:\n{path}")

    if df.index.has_duplicates:
        raise ValueError(f"DataFrame contains duplicate timestamps:\n{path}")

    return df

# =============================================================================
# Validate registry / timeframe contract
# ============================================================================= 4
def validate_feature_contract(
    features: list[str],
    timeframe: str,
) -> None:

    validate_time_features(features)
    for feature_name in features:
        spec = get_time_feature(feature_name)
        validate_time_feature_timeframe(
            feature_name,
            timeframe,
        )
        if not spec.output_columns:
            raise AssertionError(
                f"Feature '{feature_name}' has no output columns."
            )

# =============================================================================
# Validate resulting dataframe
# ============================================================================= 5
def validate_result_columns(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    features: list[str],
) -> None:

    if len(df_before) != len(df_after):
        raise AssertionError(
            "Time-feature processing changed row count."
        )

    if not df_before.index.equals(df_after.index):
        raise AssertionError(
            "Time-feature processing changed dataframe index."
        )

    for feature_name in features:
        spec = get_time_feature(feature_name)
        for column in spec.output_columns:
            if column not in df_after.columns:
                raise AssertionError(
                    f"Missing output column '{column}' "
                    f"for feature '{feature_name}'."
                )

# =============================================================================
# Batch reference
# ============================================================================= 6
def build_batch_reference(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    return add_time_features_to_df(
        df=df,
        features=features,
    )

# =============================================================================
# Live simulation
# ============================================================================= 7
def build_live_result(
    df: pd.DataFrame,
    features: list[str],
    timeframe: str,
) -> pd.DataFrame:

    live_rows: list[pd.DataFrame] = []
    previous_timestamp: pd.Timestamp | None = None
    for timestamp in df.index:

        # Preserve original column dtypes.
        one_row = df.loc[[timestamp]].copy()

        result = add_time_features_to_live_df(
            df=one_row,
            features=features,
            timeframe=timeframe,
            previous_timestamp=previous_timestamp,
        )
        live_rows.append(result)
        previous_timestamp = timestamp

    if not live_rows:
        return df.copy()

    result = pd.concat(
        live_rows,
        axis=0,
    )

    # Restore original dtypes for original columns.
    for column in df.columns:
        result[column] = result[column].astype(
            df[column].dtype,
            copy=False,
        )

    return result

# =============================================================================
# Compare Batch vs Live
# ============================================================================= 8
def compare_batch_and_live(
    batch_df: pd.DataFrame,
    live_df: pd.DataFrame,
    original_df: pd.DataFrame,
    features: list[str],
) -> None:

    if not batch_df.index.equals(live_df.index):
        raise AssertionError("Batch and Live indexes are different.")

    if len(batch_df) != len(live_df):
        raise AssertionError("Batch and Live row counts are different.")

    for feature_name in features:
        spec = get_time_feature(feature_name)

        for column in spec.output_columns:

            if column not in batch_df.columns:
                raise AssertionError(
                    f"Batch output is missing '{column}'."
                )

            if column not in live_df.columns:
                raise AssertionError(
                    f"Live output is missing '{column}'."
                )

            batch_values = batch_df[column]
            live_values = live_df[column]

            if not batch_values.equals(live_values):
                mismatch = batch_values != live_values
                first_mismatch = mismatch.idxmax()
                raise AssertionError(
                    f"Batch/Live mismatch detected.\n"
                    f"Feature : {feature_name}\n"
                    f"Column  : {column}\n"
                    f"Timestamp: {first_mismatch}\n"
                    f"Batch   : {batch_df.loc[first_mismatch, column]}\n"
                    f"Live    : {live_df.loc[first_mismatch, column]}"
                )

    # Original OHLCVS columns must remain unchanged.
    for column in original_df.columns:
        print(column)
        if column not in batch_df.columns:
            raise AssertionError(
                f"Batch result lost original column '{column}'."
            )

        if column not in live_df.columns:
            raise AssertionError(
                f"Live result lost original column '{column}'."
            )

        batch_same = batch_df[column].equals(original_df[column])
        live_same = live_df[column].equals(original_df[column])

        if not batch_same or not live_same:
            print("\n===== ORIGINAL COLUMN DEBUG =====")
            print(f"Column: {column}")

            print("\nOriginal:")
            print(original_df[column].head())
            print("dtype:", original_df[column].dtype)

            print("\nBatch:")
            print(batch_df[column].head())
            print("dtype:", batch_df[column].dtype)

            print("\nLive:")
            print(live_df[column].head())
            print("dtype:", live_df[column].dtype)

            print("\nBatch equals original:", batch_same)
            print("Live equals original :", live_same)

            print(
                "Batch values equal:",
                original_df[column].to_numpy().tolist()
                == batch_df[column].to_numpy().tolist()
            )

            print(
                "Live values equal:",
                original_df[column].to_numpy().tolist()
                == live_df[column].to_numpy().tolist()
            )

            raise AssertionError(
                f"Original column '{column}' comparison failed."
            )
# =============================================================================
# Test one timeframe
# ============================================================================= 9
def test_timeframe(
    path: Path,
    timeframe: str,
    features: list[str],
) -> None:

    print("\n" + "-" * 80)
    print(f"Testing LIVE: {SYMBOL} / {timeframe}")
    print(f"File: {path.name}")
    print("-" * 80)

    df = load_dataframe(path)

    validate_feature_contract(
        features,
        timeframe,
    )
    print(f"Rows available : {len(df):,}")

    # -----------------------------------------------------
    # Use a real contiguous portion of the actual market data.
    # -----------------------------------------------------
    if len(df) > MAX_LIVE_ROWS:
        live_source = df.iloc[:MAX_LIVE_ROWS].copy()
    else:
        live_source = df.copy()
    print(f"Rows simulated : {len(live_source):,}")
    print(f"First timestamp : {live_source.index[0]}")
    print(f"Last timestamp  : {live_source.index[-1]}")

    # -----------------------------------------------------
    # Batch reference
    # -----------------------------------------------------
    batch_result = build_batch_reference(
        df=live_source,
        features=features,
    )
    # -----------------------------------------------------
    # Live simulation
    # -----------------------------------------------------
    live_result = build_live_result(
        df=live_source,
        features=features,
        timeframe=timeframe,
    )

    # -----------------------------------------------------
    # Structural validation
    # -----------------------------------------------------
    validate_result_columns(
        df_before=live_source,
        df_after=batch_result,
        features=features,
    )

    validate_result_columns(
        df_before=live_source,
        df_after=live_result,
        features=features,
    )

    # -----------------------------------------------------
    # Critical validation:
    # Live must produce exactly the same values as Batch.
    # -----------------------------------------------------
    compare_batch_and_live(
        batch_df=batch_result,
        live_df=live_result,
        original_df=live_source,
        features=features,
    )

    print(f"Columns before : {len(live_source.columns)}")
    print(f"Columns after  : {len(live_result.columns)}")

    print("\nGenerated live columns:")

    new_columns = [
        column
        for column in live_result.columns
        if column not in live_source.columns
    ]
    for column in new_columns:
        print(f"  + {column}")

    print(
        f"\nPASS: LIVE {SYMBOL}/{timeframe}"
        f" — Batch == Live"
    )

# =============================================================================
# Main
# ============================================================================= 10
def main() -> int:

    print("=" * 80)
    print("TIME FEATURES LIVE END-TO-END TEST")
    print("=" * 80)

    # -----------------------------------------------------
    # 1. Load completed configuration
    # -----------------------------------------------------
    cfg = config_completer(enable_env_override=True)

    features = load_time_feature_config(cfg)

    print(f"\nSymbol: {SYMBOL}")
    print(f"Configured features: {len(features)}")
    print(
        "Features:",
        ", ".join(features),
    )

    # -----------------------------------------------------
    # 2. Discover configured timeframes
    # -----------------------------------------------------
    timeframes = cfg["__timeframes_dict"][SYMBOL]
    data_files = discover_data_files(
        cfg,
        timeframes,
    )
    print("\nDiscovered data files:")

    for timeframe, path in data_files.items():
        print(
            f"  {timeframe:<5} -> {path.name}"
        )

    # -----------------------------------------------------
    # 3. Execute Live simulation for every discovered timeframe
    # -----------------------------------------------------
    tested = 0
    for timeframe, path in data_files.items():
        test_timeframe(
            path=path,
            timeframe=timeframe,
            features=features,
        )
        tested += 1

    # -----------------------------------------------------
    # 4. Final result
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print(
        "LIVE END-TO-END TEST PASSED — "
        f"{tested} timeframe(s) tested."
    )
    print(
        "Batch output and simulated Live output "
        "are identical for all tested time features."
    )
    print("=" * 80)

    return 0

# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    sys.exit(main())
