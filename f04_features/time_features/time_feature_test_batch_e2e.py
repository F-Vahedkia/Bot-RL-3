# f04_features/time_features/time_feature_test_batch_e2e.py

# Date reviewed:
#    1405/05/27-08:30 --> run result is OK.
#
# Run:
#    python -m f04_features.time_features.time_feature_test_batch_e2e


# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations
import sys
import pandas as pd
import logging

from f03_data.data_handler_G import BuildParams, DataHandler
from f03_data.mtf_dataset import MTFDataset
from f04_features.time_features.time_feature_engine import apply_time_features
from f04_features.time_features.time_feature_registry import get_time_feature
from f02_utils.config_completer import config_completer

# -------------------- Logger for this module ---------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


logging.basicConfig(
    level=getattr(logging, "DEBUG", logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(filename)-28s | %(lineno)-4d : %(funcName)-24s | %(message)s",
    datefmt="%H:%M:%S",
)

# =============================================================================
# Load time-feature configuration
# =============================================================================

def load_time_feature_config(cfg, symbol) -> list[str]:
    features = (
        cfg["features"]
        ["time_features"]
        ["symbols"]
        [symbol]
    )

    if not isinstance(features, list):
        raise TypeError(
            "BITCOIN time_features configuration must be a list."
        )

    return features


# =============================================================================
# Validate DataHandler output
# =============================================================================

def validate_datahandler_dataset(
    dataset: MTFDataset,
    *,
    symbol: str,
    expected_timeframes: list[str],
) -> None:
    if not isinstance(dataset, MTFDataset):
        raise TypeError(
            "DataHandler.build() must return MTFDataset, "
            f"got {type(dataset).__name__}."
        )

    if dataset.symbol != symbol:
        raise AssertionError(
            f"Dataset symbol mismatch: "
            f"expected={symbol}, got={dataset.symbol}"
        )

    if not dataset.frames:
        raise AssertionError(
            "DataHandler.build() returned an empty MTFDataset."
        )

    actual_timeframes = set(dataset.frames.keys())

    missing_timeframes = [
        tf.upper()
        for tf in expected_timeframes
        if tf.upper() not in actual_timeframes
    ]

    if missing_timeframes:
        raise AssertionError(
            "DataHandler output is missing expected timeframes: "
            f"{missing_timeframes}"
        )

    # Every downstream DataFrame must already satisfy the UTC contract.
    for timeframe, df in dataset.frames.items():

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"MTFDataset frame {timeframe} is not a DataFrame: "
                f"{type(df).__name__}"
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                f"MTFDataset frame {timeframe} does not have "
                "a DatetimeIndex."
            )

        if df.index.tz is None:
            raise AssertionError(
                f"MTFDataset frame {timeframe} has a "
                "timezone-naive index."
            )

        if str(df.index.tz) != "UTC":
            raise AssertionError(
                f"MTFDataset frame {timeframe} is not UTC. "
                f"Detected timezone: {df.index.tz}"
            )


# =============================================================================
# Validate generated columns
# =============================================================================

def validate_output_columns(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    requested_features: list[str],
) -> None:

    for feature_name in requested_features:
        spec = get_time_feature(feature_name)

        for column in spec.output_columns:
            if column not in df_after.columns:
                raise AssertionError(
                    f"Missing output column '{column}' "
                    f"for time feature '{feature_name}'."
                )

    if len(df_before) != len(df_after):
        raise AssertionError(
            "Time-feature engine changed dataframe row count."
        )

    if not df_before.index.equals(df_after.index):
        raise AssertionError(
            "Time-feature engine changed dataframe index."
        )

    # The input dataset supplied by DataHandler is required to be UTC.
    if df_after.index.tz is None:
        raise AssertionError(
            "Time-feature engine produced a timezone-naive index."
        )

    if str(df_after.index.tz) != "UTC":
        raise AssertionError(
            "Time-feature engine changed the dataframe timezone "
            f"from UTC to {df_after.index.tz}."
        )


# =============================================================================
# Main End-to-End Test
# =============================================================================

def main() -> int:

    # -------------------------------------------------------------------------
    # 0. Print header
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("TIME FEATURES END-TO-END BATCH TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Load configuration
    # -------------------------------------------------------------------------
    cfg = config_completer(enable_env_override=True)

    symbols = list(cfg["__warmups_dicts"].keys())
    SYMBOL = symbols[0]
    print(f"Selected symbol = {SYMBOL}")

    requested_features = load_time_feature_config(cfg, SYMBOL)

    print(f"\nSymbol: {SYMBOL}")
    print(f"Configured features: {len(requested_features)}")
    print(
        "Features:",
        ", ".join(requested_features),
    )

    # -------------------------------------------------------------------------
    # 2. Build the official batch dataset through DataHandler
    # -------------------------------------------------------------------------
    base_tf = cfg["__base_tfs_dict"][SYMBOL]
    timeframes = list(cfg["__timeframes_dict"][SYMBOL])

    print("\nData source:")
    print("  DataHandler.build()")

    print("\nRequested timeframes:")
    print("  " + ", ".join(timeframes))

    handler = DataHandler(
        cfg=cfg,
        symbol=SYMBOL,
    )

    params = BuildParams(
        symbol=SYMBOL,
        base_tf=base_tf,
        timeframes=timeframes,
        load_format=handler.save_format,
        mode="number",
        start_lastrows=None,
        end_lastrows=None,
    )

    dataset = handler.build(params)

    # -------------------------------------------------------------------------
    # 3. Validate official DataHandler output
    # -------------------------------------------------------------------------
    validate_datahandler_dataset(
        dataset,
        symbol=SYMBOL,
        expected_timeframes=timeframes,
    )

    print("\nDataHandler output:")
    print(f"  Type       : {type(dataset).__name__}")
    print(f"  Symbol     : {dataset.symbol}")
    print(f"  Base TF    : {dataset.base_tf}")
    print(
        "  Timeframes : "
        + ", ".join(dataset.frames.keys())
    )

    # -------------------------------------------------------------------------
    # 4. Test every timeframe returned by DataHandler
    # -------------------------------------------------------------------------
    tested = 0

    for timeframe, df in dataset.frames.items():

        print("\n" + "-" * 80)
        print(f"Testing {SYMBOL} / {timeframe}")
        print("-" * 80)

        # ---------------------------------------------------------------------
        # Keep the exact DataHandler output as the test input.
        # ---------------------------------------------------------------------
        print(f"Rows before: {len(df):,}")
        print(f"Columns before: {list(df.columns)}")
        print(f"Index timezone: {df.index.tz}")

        df_before = df.copy()

        # ---------------------------------------------------------------------
        # Build a one-timeframe MTFDataset for the Time Feature Engine.
        #
        # The source DataFrame itself is still the official DataHandler output.
        # No raw file is opened here.
        # ---------------------------------------------------------------------
        input_dataset = MTFDataset(
            symbol=SYMBOL,
            base_tf=timeframe,
        )

        input_dataset.add(
            timeframe,
            df_before.copy(),
        )

        # ---------------------------------------------------------------------
        # Execute the actual Time Feature Engine
        # ---------------------------------------------------------------------
        result = apply_time_features(
            dataset=input_dataset,
            time_feature_config={
                SYMBOL: requested_features,
            },
        )

        # ---------------------------------------------------------------------
        # Retrieve result
        # ---------------------------------------------------------------------
        result_df = result.get(timeframe)

        if result_df is None:
            raise AssertionError(
                f"Time Feature Engine returned no result "
                f"for timeframe {timeframe}."
            )

        # ---------------------------------------------------------------------
        # Validate
        # ---------------------------------------------------------------------
        validate_output_columns(
            df_before=df_before,
            df_after=result_df,
            requested_features=requested_features,
        )

        print(f"Rows after: {len(result_df):,}")
        print(f"Columns after: {len(result_df.columns)}")

        print("\nGenerated time-feature columns:")

        new_columns = [
            column
            for column in result_df.columns
            if column not in df_before.columns
        ]

        for column in new_columns:
            print(f"  + {column}")

        print(f"\nPASS: {SYMBOL}/{timeframe}")

        tested += 1

    # -------------------------------------------------------------------------
    # 5. Final result
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(
        f"END-TO-END TEST PASSED — "
        f"{tested} timeframe(s) tested."
    )
    print("=" * 80)

    return 0


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())

