# f04_features/data_flow_test.py
#

# Integration test for:
#     DataHandler -> MTFDataset -> FeaturePipeline
#
# The test verifies the current Time Features integration contract:
#     MTFDataset
#         -> FeaturePipeline
#             -> Time Features on dataset.base_tf
#             -> FeatureEngine
#             -> FeatureStoreV2
#             -> ObservationBuilder
#
# Time Features are intentionally tested at the Pipeline / Store dataset
# boundary, not as ObservationBuilder graph nodes. The current FeatureGraph
# owns Indicator Feature Strings; ObservationBuilder resolves those graph
# nodes and is not the owner of Time Feature specification.
#
# Run:
#     pytest -v -s f04_features/data_flow_test.py
#
# Or:
#     python -m f04_features.data_flow_test

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import logging
import pytest
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Sequence

from f03_data.data_handler_G import BuildParams, DataHandler
from f03_data.mtf_dataset import MTFDataset

from f04_features.feature_C_registry_1 import get_indicator
from f04_features.feature_C_engine_6 import FeatureEngine
from f04_features.feature_B_store import FeatureStoreV2
from f04_features.feature_B_graph import FeatureGraph
from f04_features.observation_B_builder import ObservationBuilder
from f04_features.feature_pipeline import FeaturePipeline
from f04_features.time_features.time_feature_engine import add_time_features_to_df
from f04_features.time_features.time_feature_registry import TIME_FEATURE_REGISTRY

from f02_utils.config_completer import config_completer

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration helpers
# =============================================================================

def _get_symbols(cfg: Dict[str, Any]) -> List[str]:
    """Return actual configured symbols from the completed configuration."""
    symbols = cfg.get("__symbols")
    if symbols:
        return [
            str(symbol).upper()
            for symbol in symbols
            if str(symbol).upper() != "COMMON"
        ]

    configured = cfg.get("features", {}).get("symbols", {}) or {}
    symbols = [
        str(symbol).upper()
        for symbol in configured
        if str(symbol).upper() != "COMMON"
    ]
    if symbols:
        return symbols

    raise RuntimeError("No symbols found in completed configuration.")


def _get_feature_specs(cfg: Dict[str, Any], symbol: str) -> List[str]:
    """Return symbol-local Indicator Feature Strings, falling back to global."""
    symbol_cfg = (
        cfg.get("features", {})
        .get("symbols", {})
        .get(symbol, {})
        or {}
    )

    symbol_specs = symbol_cfg.get("indicators", []) or []
    if symbol_specs:
        return list(symbol_specs)

    global_specs = cfg.get("features", {}).get("indicators", []) or []
    if global_specs:
        return list(global_specs)

    return []


def _get_time_feature_names(
    cfg: Dict[str, Any],
    symbol: str,
) -> List[str]:
    """Return the configured Time Feature names for one symbol."""
    time_cfg = cfg.get("features", {}).get("time_features", {}) or {}
    if not bool(time_cfg.get("enabled", False)):
        return []

    symbols_cfg = time_cfg.get("symbols", {}) or {}
    requested = symbols_cfg.get(symbol, []) or []
    return [str(name) for name in requested]


def _time_feature_output_columns(feature_names: Sequence[str]) -> List[str]:
    """Resolve the exact DataFrame columns produced by the Time Feature registry."""
    columns: List[str] = []

    for feature_name in feature_names:
        spec = TIME_FEATURE_REGISTRY.get(feature_name)
        if spec is None:
            raise AssertionError(
                f"Unknown configured Time Feature {feature_name!r}."
            )

        for column in spec.output_columns:
            if column not in columns:
                columns.append(column)

    return columns


def _assert_time_feature_configuration(
    cfg: Dict[str, Any],
    symbols: Sequence[str],
) -> None:
    """Validate only the parts of config required by this integration test."""
    time_cfg = cfg.get("features", {}).get("time_features", {}) or {}

    if not bool(time_cfg.get("enabled", False)):
        raise AssertionError(
            "features.time_features.enabled must be true for this integration test."
        )

    symbols_cfg = time_cfg.get("symbols", {}) or {}

    for symbol in symbols:
        requested = symbols_cfg.get(symbol, []) or []
        if not requested:
            raise AssertionError(
                f"No Time Features configured for symbol={symbol}."
            )

        for feature_name in requested:
            if feature_name not in TIME_FEATURE_REGISTRY:
                raise AssertionError(
                    f"Configured Time Feature {feature_name!r} for {symbol} "
                    "is not present in TIME_FEATURE_REGISTRY."
                )


# =============================================================================
# Dataset helpers
# =============================================================================

def _build_real_dataset(cfg: Dict[str, Any], symbol: str) -> MTFDataset:
    """Build the real batch MTFDataset through DataHandler."""
    base_tf = str(cfg["__base_tfs_dict"][symbol]).upper()
    timeframes = [
        str(tf).upper()
        for tf in cfg["__timeframes_dict"][symbol]
    ]

    handler = DataHandler(
        cfg=cfg,
        symbol=symbol,
    )

    params = BuildParams(
        symbol=symbol,
        base_tf=base_tf,
        timeframes=timeframes,
    )

    return handler.build(params)


def _warmup_for_symbol(
    cfg: Dict[str, Any],
    symbol: str,
    timeframe: str,
) -> int:
    warmups = cfg.get("__warmups_dicts", {}) or {}
    symbol_warmups = warmups.get(symbol, {}) or {}

    try:
        return int(symbol_warmups.get(timeframe, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _timeframes_for_symbol(
    cfg: Dict[str, Any],
    symbol: str,
) -> List[str]:
    return [
        str(tf).upper()
        for tf in cfg["__timeframes_dict"][symbol]
    ]


def _synthetic_frame(
    timeframe: str,
    periods: int,
    start: pd.Timestamp,
) -> pd.DataFrame:
    """Build deterministic UTC-aware OHLCV data for Live integration tests."""
    from f02_utils.functions.constants import _TF_MINUTES

    minutes = int(_TF_MINUTES[timeframe])
    index = pd.date_range(
        start=start,
        periods=periods,
        freq=pd.Timedelta(minutes=minutes),
        tz="UTC" if start.tzinfo is None else None,
        name="timestamp",
    )

    close = [
        100.0 + (minutes / 100.0) + float(i)
        for i in range(periods)
    ]

    return pd.DataFrame(
        {
            "open": [value - 0.2 for value in close],
            "high": [value + 0.5 for value in close],
            "low": [value - 0.5 for value in close],
            "close": close,
            "volume": [1000.0 + i for i in range(periods)],
            "spread": [0.1 for _ in range(periods)],
        },
        index=index,
    )


def _synthetic_dataset(
    cfg: Dict[str, Any],
    symbol: str,
    *,
    start: pd.Timestamp,
    periods_extra: int = 0,
) -> MTFDataset:
    """Build an MTFDataset with enough history for configured indicators."""
    base_tf = str(cfg["__base_tfs_dict"][symbol]).upper()
    timeframes = _timeframes_for_symbol(cfg, symbol)

    dataset = MTFDataset(
        symbol=symbol,
        base_tf=base_tf,
    )

    for timeframe in timeframes:
        periods = max(
            128,
            _warmup_for_symbol(cfg, symbol, timeframe) + 64,
        ) + periods_extra

        dataset.add(
            timeframe,
            _synthetic_frame(
                timeframe=timeframe,
                periods=periods,
                start=start,
            ),
        )

    return dataset


# =============================================================================
# FeaturePipeline construction
# =============================================================================

def _build_pipeline(
    cfg: Dict[str, Any],
    symbol: str,
) -> FeaturePipeline:
    specs = _get_feature_specs(cfg, symbol)

    if not specs:
        raise AssertionError(
            f"No active Indicator Feature Strings found for symbol={symbol}."
        )

    feature_engine = FeatureEngine(cfg)
    feature_store = FeatureStoreV2(cfg)
    feature_graph = FeatureGraph(specs)
    observation_builder = ObservationBuilder(cfg)

    pipeline = FeaturePipeline(
        symbol=symbol,
        feature_engine=feature_engine,
        feature_store=feature_store,
        observation_builder=observation_builder,
        feature_graph=feature_graph,
        feature_specs=specs,
        config=cfg,
    )

    if pipeline.symbol != symbol:
        raise AssertionError(
            f"Pipeline symbol mismatch: expected={symbol}, got={pipeline.symbol}"
        )

    return pipeline


# =============================================================================
# Time Feature assertions
# =============================================================================

def _assert_base_time_features(
    dataset: MTFDataset,
    symbol: str,
    feature_names: Sequence[str],
    *,
    expected_source: pd.DataFrame,
) -> List[str]:
    """Validate exact Time Feature values on the base timeframe."""
    base_tf = str(dataset.base_tf).upper()
    actual = dataset.get(base_tf)

    if actual is None or actual.empty:
        raise AssertionError(
            f"Base timeframe frame is empty for symbol={symbol}, base_tf={base_tf}."
        )

    expected = add_time_features_to_df(
        df=expected_source,
        features=list(feature_names),
    )

    output_columns = _time_feature_output_columns(feature_names)
    missing = [
        column
        for column in output_columns
        if column not in actual.columns
    ]

    if missing:
        raise AssertionError(
            f"Missing Time Feature columns for {symbol}/{base_tf}: {missing}"
        )

    pd.testing.assert_frame_equal(
        actual[output_columns],
        expected[output_columns],
        check_dtype=True,
        check_column_type=True,
        check_names=True,
    )

    return output_columns


def _assert_time_features_only_on_base_tf(
    dataset: MTFDataset,
    output_columns: Sequence[str],
) -> None:
    """The current TimeFeatureEngine contract applies them only to base_tf."""
    base_tf = str(dataset.base_tf).upper()

    for timeframe in dataset.timeframes:
        frame = dataset.get(timeframe)
        if frame is None:
            continue

        unexpected = [
            column
            for column in output_columns
            if column in frame.columns
        ]

        if timeframe.upper() == base_tf:
            continue

        assert not unexpected, (
            f"Time Feature columns must not be added to non-base timeframe "
            f"{timeframe}: {unexpected}"
        )


def _expected_indicator_output_columns(
    specs: Sequence[str],
) -> List[str]:
    """
    Independently resolve the exact Observation columns expected from the
    configured Indicator Feature Strings.

    This helper intentionally does not use ObservationBuilder to determine
    the expected columns, so that a filtering/selection bug inside
    ObservationBuilder cannot make the test pass.
    """
    graph = FeatureGraph(list(specs))

    expected: List[str] = []
    seen: set[str] = set()

    for node in graph.execution_order():
        spec = get_indicator(node.name, mode="train")

        if spec is None:
            raise AssertionError(
                f"Indicator '{node.name}' from FeatureGraph is not present "
                "in the Feature Registry."
            )

        output_names = list(spec.output_names or [])

        if not output_names or len(output_names) == 1:
            columns = [node.canonical]
        else:
            suffix = node.canonical[len(node.name):]
            columns = [
                f"{output_name}{suffix}"
                for output_name in output_names
            ]

        for column in columns:
            if column not in seen:
                expected.append(column)
                seen.add(column)

    return expected


def _assert_indicator_observation_complete(
    observation: pd.DataFrame,
    specs: Sequence[str],
    *,
    mode: str,
) -> None:
    """
    Verify that Observation contains every expected Indicator output column,
    in exactly the same order.

    This is a completeness assertion, not merely a non-empty assertion.
    """
    expected_columns = _expected_indicator_output_columns(specs)
    actual_columns = list(observation.columns)

    missing = [
        column
        for column in expected_columns
        if column not in actual_columns
    ]

    unexpected = [
        column
        for column in actual_columns
        if column not in expected_columns
    ]

    assert not missing and not unexpected and actual_columns == expected_columns, (
        f"{mode} Observation columns are not complete or ordered correctly.\n"
        f"Expected count : {len(expected_columns)}\n"
        f"Actual count   : {len(actual_columns)}\n"
        f"Missing        : {missing}\n"
        f"Unexpected     : {unexpected}\n"
        f"Expected       : {expected_columns}\n"
        f"Actual         : {actual_columns}"
    )

    # Explicit freeze-contract check for the current 23-indicator / M1-M2 test.
    assert len(specs) == 46, (
        f"Expected 46 configured Indicator Feature Strings "
        f"(23 indicators × 2 timeframes), got {len(specs)}."
    )

    assert len(expected_columns) == 80, (
        f"Expected 80 Indicator output columns "
        f"for the current 23-indicator / M1-M2 configuration, "
        f"got {len(expected_columns)}."
    )

    assert len(actual_columns) == 80, (
        f"{mode} Observation must contain exactly 80 Indicator columns, "
        f"got {len(actual_columns)}."
    )

# =============================================================================
# Batch Integration
# =============================================================================

@pytest.fixture(scope="module")
def completed_config() -> Dict[str, Any]:
    cfg = config_completer()
    if not isinstance(cfg, dict):
        raise TypeError(
            f"config_completer() must return dict, got {type(cfg).__name__}"
        )
    return cfg


@pytest.fixture(scope="module")
def configured_symbols(
    completed_config: Dict[str, Any],
) -> List[str]:
    symbols = _get_symbols(completed_config)
    _assert_time_feature_configuration(completed_config, symbols)
    return symbols


def test_data_to_pipeline_batch_time_features(
    completed_config: Dict[str, Any],
    configured_symbols: List[str],
) -> None:
    """Real DataHandler output must reach Time Features through FeaturePipeline.run()."""

    for symbol in configured_symbols:
        feature_names = _get_time_feature_names(completed_config, symbol)
        specs = _get_feature_specs(completed_config, symbol)

        assert specs, f"No active Indicator Feature Strings for symbol={symbol}."
        assert feature_names, f"No Time Features configured for symbol={symbol}."

        # ====== temporary_1 ============================= start
        print("\nDEBUG SYMBOL =", symbol)
        print("DEBUG raw path =", completed_config.get("paths", {}).get("raw_dir"))

        raw_path = Path(r"E:\Bot-RL-3\f03_data\raw") / symbol / "M1.parquet"

        if raw_path.exists():
            direct_df = pd.read_parquet(raw_path)
            print("DEBUG source format = parquet")
        else:
            csv_path = raw_path.with_suffix(".csv")

            if csv_path.exists():
                direct_df = pd.read_csv(
                    csv_path,
                    index_col=0,
                )
                direct_df.index = pd.to_datetime(
                    direct_df.index,
                    utc=True,
                )
                direct_df.index.name = "time"
                raw_path = csv_path
                print("DEBUG source format = csv")
            else:
                raise FileNotFoundError(
                    f"Neither parquet nor csv file exists for:\n"
                    f"  parquet: {raw_path}\n"
                    f"  csv:     {csv_path}"
                )

        print("DEBUG direct path =", raw_path)
        print("DEBUG direct exists =", raw_path.exists())
        print("DEBUG direct index type =", type(direct_df.index))
        print("DEBUG direct index dtype =", direct_df.index.dtype)
        print("DEBUG direct index tz =", getattr(direct_df.index, "tz", None))
        print("DEBUG direct index name =", direct_df.index.name)
        print("DEBUG direct columns =", direct_df.columns.tolist())
        # ====== temporary_1 ============================= end


        dataset = _build_real_dataset(completed_config, symbol)
        base_tf = str(dataset.base_tf).upper()
        raw_base = dataset.get(base_tf).copy()

        pipeline = _build_pipeline(completed_config, symbol)
        result = pipeline.run(dataset=dataset, mode="train")


        # ====== temporary_2 ============================= start
        print("\n========== DEBUG BATCH ==========")

        print("feature_specs:")
        for spec in specs:
            print("   ", spec)

        print("\nFeatureGraph execution_order:")
        for node in pipeline.feature_graph.execution_order():
            print("   ", node)

        print("\nFeature dataset:")
        for tf in result["features"].timeframes:
            frame = result["features"].get(tf)
            print(
                tf,
                "shape=", frame.shape,
                "columns=", frame.columns.tolist(),
            )

        print("\nObservationBuilder:")
        print("   whitelist =", pipeline.observation_builder.whitelist)
        print("   blacklist =", pipeline.observation_builder.blacklist)
        print("   observation shape =", result["observation"].shape)
        print("   observation columns =", result["observation"].columns.tolist())

        print("=================================\n")
        # ====== temporary_2 ============================= end


        # ========== بلوک قدیمی
        # assert isinstance(result, dict)
        # assert isinstance(result.get("features"), MTFDataset)
        # assert isinstance(result.get("observation"), pd.DataFrame)
        # assert result.get("observation").shape[1] > 0

        # ========== ابتدای بلوک جدید
        assert isinstance(result, dict)
        assert isinstance(result.get("features"), MTFDataset)

        observation = result.get("observation")

        assert isinstance(observation, pd.DataFrame)
        # === temporary-3 ============================== start
        print("\n ===>>>>>>> ===>>>>> DEBUG BEFORE OBSERVATION ASSERT")
        print("symbol =", symbol)
        print("dataset.timeframes =", dataset.timeframes)

        for tf in dataset.timeframes:
            frame = dataset.get(tf)
            if frame is None:
                print(tf, "-> None")
            else:
                print(
                    tf,
                    "shape =", frame.shape,
                    "empty =", frame.empty,
                    "start =", frame.index[0] if not frame.empty else None,
                    "end =", frame.index[-1] if not frame.empty else None,
                )

        print("observation shape =", observation.shape)
        # === temporary-3 ============================== end
        assert not observation.empty
        assert observation.shape[1] > 0

        _assert_indicator_observation_complete(
            observation,
            specs,
            mode=f"BATCH/{symbol}",
        )
        # ========== انتهای بلوک جدید


        assert pipeline.dataset is dataset

        output_columns = _assert_base_time_features(
            dataset,
            symbol,
            feature_names,
            expected_source=raw_base,
        )

        _assert_time_features_only_on_base_tf(
            dataset,
            output_columns,
        )

        stored = result["features"]
        stored_base = stored.get(base_tf)

        assert stored_base is not None
        for column in output_columns:
            assert column in stored_base.columns, (
                f"Time Feature {column!r} did not survive FeatureStoreV2.build() "
                f"for {symbol}/{base_tf}."
            )


# =============================================================================
# Live Integration
# =============================================================================

def test_pipeline_live_time_features_state(
    completed_config: Dict[str, Any],
    configured_symbols: List[str],
) -> None:
    """FeaturePipeline.process_live() must apply Time Features and preserve live timestamp state."""

    # 2026-01-05 23:58 UTC is intentional: the base M1 frame crosses midnight,
    # allowing is_new_day to be exercised inside the live snapshot itself.
    start = pd.Timestamp("2026-01-05 23:58:00", tz="UTC")

    for symbol in configured_symbols:
        feature_names = _get_time_feature_names(completed_config, symbol)
        assert feature_names, f"No Time Features configured for symbol={symbol}."

        specs = _get_feature_specs(completed_config, symbol)
        assert specs, f"No active Indicator Feature Strings for symbol={symbol}."

        pipeline = _build_pipeline(completed_config, symbol)
        pipeline.reset_live_state()

        snapshot_1 = _synthetic_dataset(
            completed_config,
            symbol,
            start=start,
            periods_extra=0,
        )
        snapshot_1_base_before = snapshot_1.get(snapshot_1.base_tf).copy()

        # ========== بلوک قدیمی
        # observation_1 = pipeline.process_live(snapshot_1)
        # assert isinstance(observation_1, pd.DataFrame)
        # assert not observation_1.empty
        # ========== ابتدای بلوک جدید
        observation_1 = pipeline.process_live(snapshot_1)

        assert isinstance(observation_1, pd.DataFrame)
        assert not observation_1.empty

        _assert_indicator_observation_complete(
            observation_1,
            _get_feature_specs(completed_config, symbol),
            mode=f"LIVE-1/{symbol}",
        )
        # ========== انتهای بلوک جدید

        cols = _assert_base_time_features(
            snapshot_1,
            symbol,
            feature_names,
            expected_source=snapshot_1_base_before,
        )
        _assert_time_features_only_on_base_tf(snapshot_1, cols)

        snapshot_2 = _synthetic_dataset(
            completed_config,
            symbol,
            start=start,
            periods_extra=1,
        )
        snapshot_2_base_before = snapshot_2.get(snapshot_2.base_tf).copy()

        observation_2 = pipeline.process_live(snapshot_2)
        assert isinstance(observation_2, pd.DataFrame)
        assert not observation_2.empty

        _assert_indicator_observation_complete(
            observation_2,
            specs,
            mode=f"LIVE-2/{symbol}",
        )

        _assert_base_time_features(
            snapshot_2,
            symbol,
            feature_names,
            expected_source=snapshot_2_base_before,
        )
        _assert_time_features_only_on_base_tf(snapshot_2, cols)

        last_timestamp = snapshot_2.get(snapshot_2.base_tf).index[-1]
        assert pipeline._time_feature_previous_timestamp == last_timestamp

        # Reset must clear the Time Feature live state as well as indicator state.
        pipeline.reset_live_state()
        assert pipeline._time_feature_previous_timestamp is None
        assert pipeline.dataset is None
        assert pipeline.features is None
        assert pipeline.observation is None


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = config_completer()
    if not isinstance(cfg, dict):
        raise TypeError(
            f"config_completer() must return dict, got {type(cfg).__name__}"
        )

    symbols = _get_symbols(cfg)
    _assert_time_feature_configuration(cfg, symbols)

    print("\n========== DATA -> FEATURE INTEGRATION TEST ==========")
    print(f"Configured symbols : {symbols}")

    for symbol in symbols:
        specs = _get_feature_specs(cfg, symbol)
        time_features = _get_time_feature_names(cfg, symbol)

        print(f"Symbol              : {symbol}")
        print(f"Indicator specs      : {len(specs)}")
        print(f"Time Features        : {len(time_features)}")
        print(f"Time Feature columns : {_time_feature_output_columns(time_features)}")

    print("======================================================\n")
    print(
        "Run the pytest command shown at the top of this file for the full "
        "integration assertions."
    )


if __name__ == "__main__":
    main()
