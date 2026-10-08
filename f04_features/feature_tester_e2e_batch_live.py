# f04_features/feature_tester_e2e_batch_live.py
# Last reviewed at 1405/06/17-08:35
# Run:
#   pytest -v -s f04_features/feature_tester_e2e_batch_live.py

"""  DocString
Purpose:
  End-to-End verification of the REAL, configuration-driven Feature Layer.

  The tester verifies that every active symbol-local Feature String from the
  completed configuration can travel through the actual Feature execution
  path:

      Config
        -> FeatureEngine
        -> FeatureStoreV2
        -> FeatureGraph
        -> ObservationBuilder

  Both Batch and Live execution paths are exercised and compared.

Scope:
  - Verifies configured Feature String resolution and execution.
  - Verifies Batch output reaches ObservationBuilder.
  - Verifies Live first-snapshot execution.
  - Verifies Live incremental output is identical to an independent Batch
    calculation on the corresponding complete snapshot.
  - Verifies Live state reset produces results identical to Batch.
  - Verifies Batch and Live expose the same final feature columns.
  - Verifies the Registry -> Batch-function parameter contract before
    executing Batch calculations.

Test Data:
  - Feature Strings are taken from config_completer(); no Feature String is
    hard-coded in the tester.
  - The Data Layer is NOT executed.
  - MTFDataset and its OHLCV frames are constructed locally using deterministic
    synthetic data that conforms to the established Data -> Features input
    contract.

Observation Filtering:
  - The tester creates a private copy of the completed configuration.
  - Observation whitelist/blacklist filters are cleared in that private copy.
  - This test is intentionally focused on Feature calculation, propagation,
    and Batch/Live consistency.
  - Observation filtering policy is a separate concern and is therefore not
    evaluated here.
  - A zero-column observation is never accepted as a successful result.

Architecture Boundary:
  - This file is TEST CODE ONLY.
  - It does not implement, replace, or modify FeatureEngine, FeatureStoreV2,
    FeatureGraph, ObservationBuilder, Registry, or any other runtime component.
  - Runtime Feature logic is always executed from the actual project classes;
    this file only supplies test inputs and assertions and reports results.
"""

# =============================================================================
# Imports
# =============================================================================

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List

import inspect
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from f03_data.mtf_dataset import MTFDataset
from f04_features.feature_C_registry_1 import get_indicator

from f04_features.feature_B_graph import FeatureGraph
from f04_features.feature_C_engine_6 import FeatureEngine
from f04_features.feature_B_store import FeatureStoreV2
from f04_features.observation_B_builder import ObservationBuilder

from f02_utils.config_completer import config_completer
from f02_utils.functions.parser import parse_spec
from f02_utils.functions.constants import _TF_MINUTES

# =============================================================================
# Configuration
# ============================================================================= 1

CFG = config_completer()


# =============================================================================
# Configuration helpers
# ============================================================================= 2

def _configured_symbols(cfg: Dict[str, Any]) -> List[str]:
    symbols = cfg.get("__symbols")
    if symbols:
        return [str(x) for x in symbols if str(x).upper() != "COMMON"]

    symbols_cfg = cfg.get("features", {}).get("symbols", {}) or {}
    symbols = [str(x) for x in symbols_cfg if str(x).upper() != "COMMON"]
    if symbols:
        return symbols

    raise RuntimeError(
        "No configured symbols found in completed configuration."
    )


def _timeframes_for_symbol(
    cfg: Dict[str, Any],
    symbol: str,
) -> List[str]:
    tf_dict = cfg.get("__timeframes_dict", {}) or {}
    if symbol in tf_dict and tf_dict[symbol]:
        return [str(tf).upper() for tf in tf_dict[symbol]]

    base_dict = cfg.get("__base_tfs_dict", {}) or {}
    base_tf = base_dict.get(symbol)
    if base_tf:
        return [str(base_tf).upper()]

    raise RuntimeError(
        f"No derived timeframes available for symbol={symbol!r}."
    )


def _base_tf_for_symbol(
    cfg: Dict[str, Any],
    symbol: str,
) -> str:
    base_dict = cfg.get("__base_tfs_dict", {}) or {}
    try:
        return str(base_dict[symbol]).upper()
    except KeyError as exc:
        raise RuntimeError(
            f"No derived base_tf available for symbol={symbol!r}."
        ) from exc


def _warmup_for_symbol(
    cfg: Dict[str, Any],
    symbol: str,
    tf: str,
) -> int:
    warmups = cfg.get("__all_required_bars", {}) or {}
    symbol_warmups = warmups.get(symbol, {}) or {}

    try:
        return int(symbol_warmups.get(tf, 0) or 0)
    except (TypeError, ValueError):
        return 0


# =============================================================================
# Private Observation test configuration
# ============================================================================= 3

def _resolved_feature_specs_for_symbol(
    cfg: Dict[str, Any],
    symbol: str,
) -> List[str]:
    symbol_cfg = (
        cfg.get("features", {})
        .get("symbols", {})
        .get(symbol, {})
        or {}
    )

    specs = symbol_cfg.get("indicators", []) or []

    return [str(x) for x in specs]


def _test_cfg_without_observation_filters(
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Keep the REAL completed config for FeatureEngine, while preventing
    Observation whitelist/blacklist settings from hiding calculated columns.

    The tester is specifically validating Feature String execution and
    propagation, not the separate whitelist/blacklist policy.
    """
    test_cfg = deepcopy(cfg)

    # Current/legacy locations that may feed ObservationBuilder.
    env_cfg = test_cfg.setdefault("env", {})
    env_cfg["features_whitelist"] = []
    env_cfg["features_blacklist"] = []

    features_cfg = test_cfg.setdefault("features", {})

    # Current location used by newer ObservationBuilder revisions.
    observation_cfg = features_cfg.get("observation")
    if isinstance(observation_cfg, dict):
        observation_cfg["features_whitelist"] = []
        observation_cfg["features_blacklist"] = []

    # Legacy/direct feature-level locations.
    features_cfg["features_whitelist"] = []
    features_cfg["features_blacklist"] = []

    return test_cfg


TEST_CFG = _test_cfg_without_observation_filters(CFG)


# =========================================================
def _prepare_batch_kwargs(
    spec,
    parsed_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Adapt Registry-canonical parameter names to the actual batch-function
    parameter names, using Registry aliases only.
    """
    accepted = set(inspect.signature(spec.fn).parameters)
    result: Dict[str, Any] = {}

    for key, value in parsed_kwargs.items():
        if key in accepted:
            result[key] = value
            continue

        parameter = spec.get_parameter(key)
        if parameter is None:
            result[key] = value
            continue

        alias = next(
            (name for name in parameter.aliases if name in accepted),
            None,
        )

        if alias is None:
            result[key] = value
        else:
            result[alias] = value

    return result


def _validate_batch_contract(specs: List[str]) -> None:
    for raw in specs:
        parsed = parse_spec(raw)

        spec = get_indicator(parsed.name, mode="train")
        assert spec is not None, (
            f"Batch registry entry not found for: {raw}"
        )

        kwargs = _prepare_batch_kwargs(spec, parsed.kwargs)

        try:
            inspect.signature(spec.fn).bind_partial(
                object(),
                **kwargs,
            )
        except TypeError as exc:
            raise AssertionError(
                f"Batch parameter contract mismatch for {raw!r}\n"
                f"Function: {spec.fn.__name__}\n"
                f"Parsed kwargs: {parsed.kwargs!r}\n"
                f"Prepared kwargs: {kwargs!r}"
            ) from exc


# =============================================================================
# Synthetic Data Layer-compatible MTFDataset
# ============================================================================= 4

def _frame(
    tf: str,
    periods: int,
    start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build a UTC-aware raw frame.

    Index timestamps are candle OPEN times, matching the established
    Data -> Features contract.
    """
    from f02_utils.functions.constants import _TF_MINUTES

    minutes = _TF_MINUTES[tf]

    index = pd.date_range(
        start=start,
        periods=periods,
        freq=pd.Timedelta(minutes=minutes),
        tz="UTC" if start.tz is None else None,
        name="timestamp",
    )

    tf_offset = float(minutes) / 100.0
    close = [100.0 + tf_offset + float(i) for i in range(periods)]

    return pd.DataFrame(
        {
            "open": [x - 0.2 for x in close],
            "high": [x + 0.5 for x in close],
            "low": [x - 0.5 for x in close],
            "close": close,
            "volume": [1000.0 + i for i in range(periods)],
            "spread": [0.1 for _ in range(periods)],
        },
        index=index,
    )


def _dataset(
    cfg: Dict[str, Any],
    symbol: str,
    periods_extra: int = 0,
) -> MTFDataset:
    
    base_tf = _base_tf_for_symbol(cfg, symbol)
    timeframes = _timeframes_for_symbol(cfg, symbol)

    warmups = cfg["__all_required_bars"][symbol]

    warmup_span_minutes = max(
        warmups[tf] * _TF_MINUTES[tf]
        for tf in timeframes
    )

    base_test_candles = max(
        16,
        max(warmups.values()) + 8,
    )

    test_span_minutes = (
        warmup_span_minutes
        + base_test_candles * _TF_MINUTES[base_tf]
    )

    if periods_extra:
        test_span_minutes += periods_extra * _TF_MINUTES[base_tf]

    periods_by_tf: Dict[str, int] = {
        tf: int(
            np.ceil(test_span_minutes / _TF_MINUTES[tf])
        )
        for tf in timeframes
    }

    start = pd.Timestamp(
        "2026-01-05 00:00:00",
        tz="UTC",
    )

    dataset = MTFDataset(
        symbol=symbol,
        base_tf=base_tf,
    )

    for tf in timeframes:
        dataset.add(
            tf,
            _frame(
                tf=tf,
                periods=periods_by_tf[tf],
                start=start,
            ),
        )

    return dataset


# =============================================================================
# Feature-layer helpers
# ============================================================================= 5

def _components(
    cfg: Dict[str, Any],
    specs: List[str],
):
    engine = FeatureEngine(cfg)
    store = FeatureStoreV2(cfg)
    graph = FeatureGraph(specs)
    builder = ObservationBuilder(cfg)

    return engine, store, graph, builder


def _batch_observation(
    cfg: Dict[str, Any],
    symbol: str,
    specs: List[str],
    dataset: MTFDataset,
) -> pd.DataFrame:
    del symbol  # retained in signature for readable test diagnostics

    engine, store, graph, builder = _components(cfg, specs)

    features = engine.execute(
        dataset=dataset,
        specs=specs,
        mode="train",
    )

    print("\n========== ENGINE BATCH OUTPUT ==========")
    print(type(features))
    if hasattr(features, "keys"):
        print("feature keys:", list(features.keys()))
    else:
        print(features)
    print("=========================================\n")

    stored = store.build(
        dataset,
        features,
    )

    print("\n========== STORE FRAMES DEBUG ==========")
    for tf, df in stored.frames.items():
        print(
            tf,
            "shape=", df.shape,
            "index_len=", len(df.index),
            "columns=",
            list(df.columns)
        )
    print("========================================\n")

    print("\n========== FEATURE STORE OUTPUT ==========")
    print(type(stored))
    if hasattr(stored, "__dict__"):
        print(stored.__dict__.keys())
    print("==========================================\n")
    print("\n========== FEATURE GRAPH ==========")
    print(graph)
    print("==================================\n")

    # builder.drop_na_head = False    # new- for debug only
    observation = builder.build(
        stored,
        graph,
    )

    print("\n========== OBSERVATION ==========")
    print(observation)
    print("shape:", observation.shape)
    print("columns:", list(observation.columns))
    print("=================================\n")


    return observation


def _live_observation(
    cfg: Dict[str, Any],
    engine: FeatureEngine,
    store: FeatureStoreV2,
    graph: FeatureGraph,
    builder: ObservationBuilder,
    dataset: MTFDataset,
    specs: List[str],
) -> pd.DataFrame:
    features = engine.execute(
        dataset=dataset,
        specs=specs,
        mode="live",
    )

    print("\n========== ENGINE LIVE OUTPUT ==========")
    print(type(features))
    if hasattr(features, "keys"):
        print("feature keys:", list(features.keys()))
    print("========================================\n")

    stored = store.build(
        dataset,
        features,
    )

    return builder.build(
        stored,
        graph,
    )


def _assert_real_observation_old1(
    observation: pd.DataFrame,
    *,
    symbol: str,
    mode: str,
    specs: List[str],
) -> None:
    assert isinstance(observation, pd.DataFrame)

    assert observation.shape[1] > 0, (
        f"{mode} produced ZERO feature columns for symbol={symbol}. "
        f"Configured specs={specs!r}"
    )

    assert not observation.empty, (
        f"{mode} produced an empty observation for symbol={symbol}. "
        f"columns={list(observation.columns)!r}"
    )


def _assert_real_observation(
    observation: pd.DataFrame,
    *,
    symbol: str,
    mode: str,
    specs: List[str],
) -> None:
    assert isinstance(observation, pd.DataFrame)

    assert observation.shape[1] > 0, (
        f"{mode} produced ZERO feature columns for symbol={symbol}. "
        f"Configured specs={specs!r}"
    )

    # MACD warmup may leave NaNs in some output columns.
    # Observation is valid if at least one feature column has real values.
    assert not observation.dropna(how="all").empty, (
        f"{mode} produced no real feature rows for symbol={symbol}. "
        f"columns={list(observation.columns)!r}"
    )


# =============================================================================
# Configuration sanity
# ============================================================================= 6

def test_config_completer_returns_dict():
    assert isinstance(CFG, dict)


def test_configured_symbols_have_feature_lists():
    symbols = _configured_symbols(CFG)
    assert symbols, "No testable symbols found in completed configuration."

    for symbol in symbols:
        specs = _resolved_feature_specs_for_symbol(CFG, symbol)
        print(specs)

        print(
            f"SYMBOL={symbol} | "
            f"configured_feature_specs={len(specs)} | "
            f"specs={specs}"
        )


# =============================================================================
# Batch
# ============================================================================= 7

@pytest.mark.parametrize("symbol", _configured_symbols(CFG))
def test_configured_features_batch_end_to_end(symbol: str):
    """
    Every active symbol-local Feature String from the real config must:
      Config -> FeatureEngine(batch) -> FeatureStore -> ObservationBuilder
    and expose at least one real feature column.
    """
    specs = _resolved_feature_specs_for_symbol(CFG, symbol)

    if not specs:
        pytest.skip(
            f"No active symbol-local Feature Strings for {symbol}."
        )

    dataset = _dataset(CFG, symbol)

    _validate_batch_contract(specs)

    observation = _batch_observation(
        TEST_CFG,
        symbol,
        specs,
        dataset,
    )

    _assert_real_observation(
        observation,
        symbol=symbol,
        mode="BATCH",
        specs=specs,
    )

    print(
        f"BATCH | symbol={symbol} | specs={len(specs)} "
        f"| columns={list(observation.columns)} "
        f"| shape={observation.shape}"
    )


# =============================================================================
# Live - first snapshot
# ============================================================================= 8

@pytest.mark.parametrize("symbol", _configured_symbols(CFG))
def test_configured_features_live_first_snapshot(symbol: str):
    """
    The first Live snapshot must calculate and expose the same configured
    Feature columns that the graph resolves in Batch.
    """
    specs = _resolved_feature_specs_for_symbol(CFG, symbol)

    if not specs:
        pytest.skip(
            f"No active symbol-local Feature Strings for {symbol}."
        )

    dataset = _dataset(CFG, symbol)
    engine, store, graph, builder = _components(TEST_CFG, specs)

    observation = _live_observation(
        TEST_CFG,
        engine,
        store,
        graph,
        builder,
        dataset,
        specs,
    )

    _assert_real_observation(
        observation,
        symbol=symbol,
        mode="LIVE-1",
        specs=specs,
    )

    print(
        f"LIVE-1 | symbol={symbol} | specs={len(specs)} "
        f"| columns={list(observation.columns)} "
        f"| shape={observation.shape}"
    )


# =============================================================================
# Live incremental == Batch
# ============================================================================= 9

@pytest.mark.parametrize("symbol", _configured_symbols(CFG))
def test_configured_features_live_incremental_matches_batch(
    symbol: str,
):
    """
    Live snapshot #1 -> Live snapshot #2 must equal an independent Batch
    calculation on the complete snapshot #2.
    """
    specs = _resolved_feature_specs_for_symbol(CFG, symbol)

    if not specs:
        pytest.skip(
            f"No active symbol-local Feature Strings for {symbol}."
        )

    live_engine, live_store, live_graph, live_builder = _components(
        TEST_CFG,
        specs,
    )

    snapshot_1 = _dataset(CFG, symbol, periods_extra=0)
    snapshot_2 = _dataset(CFG, symbol, periods_extra=1)

    first_live = _live_observation(
        TEST_CFG,
        live_engine,
        live_store,
        live_graph,
        live_builder,
        snapshot_1,
        specs,
    )

    _assert_real_observation(
        first_live,
        symbol=symbol,
        mode="LIVE-1",
        specs=specs,
    )

    live_final = _live_observation(
        TEST_CFG,
        live_engine,
        live_store,
        live_graph,
        live_builder,
        snapshot_2,
        specs,
    )

    _assert_real_observation(
        live_final,
        symbol=symbol,
        mode="LIVE-2",
        specs=specs,
    )

    batch_final = _batch_observation(
        TEST_CFG,
        symbol,
        specs,
        _dataset(CFG, symbol, periods_extra=1),
    )

    _assert_real_observation(
        batch_final,
        symbol=symbol,
        mode="BATCH",
        specs=specs,
    )

    assert_frame_equal(
        live_final,
        batch_final,
        check_dtype=True,
        check_column_type=True,
    )

    print(
        f"LIVE==BATCH | symbol={symbol} | specs={len(specs)} "
        f"| columns={list(live_final.columns)} "
        f"| shape={live_final.shape}"
    )


# =============================================================================
# Live reset
# ============================================================================= 10

@pytest.mark.parametrize("symbol", _configured_symbols(CFG))
def test_configured_features_live_reset_matches_batch(symbol: str):
    """
    Resetting FeatureEngine Live state must start a fresh calculation whose
    complete-snapshot result equals Batch.
    """
    specs = _resolved_feature_specs_for_symbol(CFG, symbol)

    if not specs:
        pytest.skip(
            f"No active symbol-local Feature Strings for {symbol}."
        )

    engine, store, graph, builder = _components(TEST_CFG, specs)

    _live_observation(
        TEST_CFG,
        engine,
        store,
        graph,
        builder,
        _dataset(CFG, symbol, periods_extra=0),
        specs,
    )

    engine.reset_live_state()

    restarted_live = _live_observation(
        TEST_CFG,
        engine,
        store,
        graph,
        builder,
        _dataset(CFG, symbol, periods_extra=1),
        specs,
    )

    _assert_real_observation(
        restarted_live,
        symbol=symbol,
        mode="LIVE-RESET",
        specs=specs,
    )

    expected_batch = _batch_observation(
        TEST_CFG,
        symbol,
        specs,
        _dataset(CFG, symbol, periods_extra=1),
    )

    _assert_real_observation(
        expected_batch,
        symbol=symbol,
        mode="BATCH",
        specs=specs,
    )

    assert_frame_equal(
        restarted_live,
        expected_batch,
        check_dtype=True,
        check_column_type=True,
    )


# =============================================================================
# Column-set equality
# ============================================================================= 11

@pytest.mark.parametrize("symbol", _configured_symbols(CFG))
def test_configured_features_batch_live_column_sets_match(symbol: str):
    """
    Batch and Live must expose exactly the same non-empty feature columns,
    in the same order.
    """
    specs = _resolved_feature_specs_for_symbol(CFG, symbol)

    if not specs:
        pytest.skip(
            f"No active symbol-local Feature Strings for {symbol}."
        )

    batch = _batch_observation(
        TEST_CFG,
        symbol,
        specs,
        _dataset(CFG, symbol),
    )

    _assert_real_observation(
        batch,
        symbol=symbol,
        mode="BATCH",
        specs=specs,
    )

    engine, store, graph, builder = _components(TEST_CFG, specs)

    live = _live_observation(
        TEST_CFG,
        engine,
        store,
        graph,
        builder,
        _dataset(CFG, symbol),
        specs,
    )

    _assert_real_observation(
        live,
        symbol=symbol,
        mode="LIVE",
        specs=specs,
    )

    assert list(live.columns) == list(batch.columns)


# =============================================================================
# Main
# ============================================================================= 12

if __name__ == "__main__":
    symbols = _configured_symbols(CFG)

    print("Configured test symbols:", symbols)

    for symbol in symbols:
        specs = _resolved_feature_specs_for_symbol(CFG, symbol)
        print(f"{symbol}: {specs}")

# ============================================================================= END
