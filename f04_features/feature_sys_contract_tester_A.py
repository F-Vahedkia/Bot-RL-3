# f04_features/feature_sys_contract_tester_A.py
# Date reviewed:
#    1405/05/24-15:08 --> run result is OK for 59 tests

# Run:
#   pytest -v -s f04_features/feature_sys_contract_tester_A.py

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import pandas as pd
import pytest

from f03_data.mtf_dataset import MTFDataset
from f04_features.feature_sys_contract import (
    FEATURE_SYSTEM_CONTRACT,
    FEATURE_SYSTEM_CONTRACT_VERSION,
    TRAIN_MODES,
    INCREMENTAL_MODES,
    ALL_MODES,
    FeatureStage,
    DatasetContract,
    FeatureExecutionContract,
    ObservationContract,
    FeatureSystemContract,
    build_dataset_contract,
    build_execution_contract,
    summarize_system_contract,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_frame() -> pd.DataFrame:
    index = pd.date_range(
        "2026-01-01 10:00:00",
        periods=2,
        freq="min",
        tz="UTC",
        name="timestamp",
    )
    return pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.0, 2.0],
            "volume": [100, 101],
        },
        index=index,
    )


def _make_dataset(
    symbol: str = "XAUUSD",
    base_tf: str = "M1",
    timeframes: tuple[str, ...] = ("M1", "H1"),
) -> MTFDataset:
    ds = MTFDataset(symbol=symbol, base_tf=base_tf)
    for tf in timeframes:
        ds.add(tf, _make_frame())
    return ds


# =============================================================================
# Version / modes
# =============================================================================

def test_contract_version() -> None:
    assert FEATURE_SYSTEM_CONTRACT_VERSION == "1"
    assert FEATURE_SYSTEM_CONTRACT.version == FEATURE_SYSTEM_CONTRACT_VERSION


def test_mode_sets() -> None:
    assert TRAIN_MODES == frozenset({"train", "optimize"})
    assert INCREMENTAL_MODES == frozenset(
        {"live", "paper", "shadow", "backtest", "replay", "eval"}
    )
    assert ALL_MODES == TRAIN_MODES | INCREMENTAL_MODES


def test_mode_sets_are_disjoint() -> None:
    assert not (TRAIN_MODES & INCREMENTAL_MODES)


# =============================================================================
# FeatureStage
# =============================================================================

def test_feature_stage_values() -> None:
    assert FeatureStage.RAW.value == "raw"
    assert FeatureStage.FEATURES.value == "features"
    assert FeatureStage.STORED.value == "stored"
    assert FeatureStage.OBSERVATION.value == "observation"


def test_system_stage_order() -> None:
    expected = (
        FeatureStage.RAW,
        FeatureStage.FEATURES,
        FeatureStage.STORED,
        FeatureStage.OBSERVATION,
    )
    assert FEATURE_SYSTEM_CONTRACT.stages == expected
    assert FEATURE_SYSTEM_CONTRACT.validate_stage_order() == expected


# =============================================================================
# DatasetContract
# =============================================================================

def test_dataset_contract_from_dataset() -> None:
    contract = DatasetContract.from_dataset(_make_dataset())
    assert contract.symbol == "XAUUSD"
    assert contract.base_tf == "M1"
    assert contract.timeframes == ("M1", "H1")


def test_dataset_contract_normalizes_timeframes() -> None:
    contract = DatasetContract.from_dataset(
        _make_dataset(
            base_tf="m1",
            timeframes=("m1", "h1"),
        )
    )
    assert contract.base_tf == "M1"
    assert contract.timeframes == ("M1", "H1")


def test_build_dataset_contract() -> None:
    ds = _make_dataset()
    contract = build_dataset_contract(ds)
    assert isinstance(contract, DatasetContract)
    assert contract.symbol == "XAUUSD"
    assert contract.base_tf == "M1"
    assert contract.timeframes == ("M1", "H1")


def test_dataset_contract_none() -> None:
    with pytest.raises(ValueError, match="dataset is required"):
        DatasetContract.from_dataset(None)


def test_dataset_contract_wrong_type() -> None:
    with pytest.raises(TypeError, match="Expected MTFDataset"):
        DatasetContract.from_dataset("bad")


def test_dataset_contract_missing_symbol() -> None:
    ds = _make_dataset()
    ds.symbol = ""
    with pytest.raises(ValueError, match="MTFDataset.symbol is required"):
        DatasetContract.from_dataset(ds)


def test_dataset_contract_missing_base_tf() -> None:
    ds = _make_dataset()
    ds.base_tf = ""
    with pytest.raises(ValueError, match="MTFDataset.base_tf is required"):
        DatasetContract.from_dataset(ds)


def test_dataset_contract_empty_timeframes() -> None:
    ds = MTFDataset(symbol="XAUUSD", base_tf="M1")
    with pytest.raises(
        ValueError,
        match="MTFDataset must contain at least one timeframe",
    ):
        DatasetContract.from_dataset(ds)


def test_dataset_contract_base_tf_not_present() -> None:
    ds = MTFDataset(symbol="XAUUSD", base_tf="H1")
    ds.add("M1", _make_frame())
    with pytest.raises(
        ValueError,
        match="Base timeframe .* is not present",
    ):
        DatasetContract.from_dataset(ds)


def test_dataset_contract_validate_matching_dataset() -> None:
    contract = DatasetContract.from_dataset(_make_dataset())
    contract.validate_dataset(_make_dataset())


def test_dataset_contract_symbol_mismatch() -> None:
    contract = DatasetContract.from_dataset(_make_dataset(symbol="XAUUSD"))
    with pytest.raises(ValueError, match="Dataset symbol mismatch"):
        contract.validate_dataset(_make_dataset(symbol="EURUSD"))


def test_dataset_contract_base_tf_mismatch() -> None:
    contract = DatasetContract.from_dataset(_make_dataset(base_tf="M1"))
    other = MTFDataset(symbol="XAUUSD", base_tf="H1")
    other.add("H1", _make_frame())
    with pytest.raises(ValueError, match="Base timeframe mismatch"):
        contract.validate_dataset(other)


def test_dataset_contract_timeframe_mismatch() -> None:
    contract = DatasetContract.from_dataset(
        _make_dataset(timeframes=("M1", "H1"))
    )
    other = _make_dataset(timeframes=("M1", "M5"))
    with pytest.raises(ValueError, match="Dataset timeframe mismatch"):
        contract.validate_dataset(other)


# =============================================================================
# FeatureExecutionContract
# =============================================================================

def test_feature_execution_contract_normalization() -> None:
    contract = FeatureExecutionContract(
        mode="LIVE",
        symbol="XAUUSD",
        base_tf="m1",
        timeframes=("m1", "h1"),
        specs=["sma(10)@m1", "rsi(14)@h1"],
    )
    assert contract.mode == "live"
    assert contract.symbol == "XAUUSD"
    assert contract.base_tf == "M1"
    assert contract.timeframes == ("M1", "H1")
    assert contract.specs == ("sma(10)@m1", "rsi(14)@h1")


@pytest.mark.parametrize("mode", sorted(ALL_MODES))
def test_feature_execution_contract_accepts_all_modes(mode: str) -> None:
    contract = FeatureExecutionContract(
        mode=mode,
        symbol="XAUUSD",
        base_tf="M1",
        timeframes=("M1",),
        specs=(),
    )
    assert contract.mode == mode


def test_feature_execution_contract_invalid_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported Feature mode"):
        FeatureExecutionContract(
            mode="invalid",
            symbol="XAUUSD",
            base_tf="M1",
            timeframes=("M1",),
            specs=(),
        )


def test_feature_execution_contract_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol is required"):
        FeatureExecutionContract(
            mode="train",
            symbol="",
            base_tf="M1",
            timeframes=("M1",),
            specs=(),
        )


def test_feature_execution_contract_empty_base_tf() -> None:
    with pytest.raises(ValueError, match="base_tf is required"):
        FeatureExecutionContract(
            mode="train",
            symbol="XAUUSD",
            base_tf="",
            timeframes=("M1",),
            specs=(),
        )


def test_feature_execution_contract_empty_timeframes() -> None:
    with pytest.raises(ValueError, match="At least one timeframe is required"):
        FeatureExecutionContract(
            mode="train",
            symbol="XAUUSD",
            base_tf="M1",
            timeframes=(),
            specs=(),
        )


def test_build_execution_contract() -> None:
    contract = build_execution_contract(
        dataset=_make_dataset(),
        specs=["sma(10)@M1"],
        mode="TRAIN",
    )
    assert isinstance(contract, FeatureExecutionContract)
    assert contract.mode == "train"
    assert contract.symbol == "XAUUSD"
    assert contract.base_tf == "M1"
    assert contract.timeframes == ("M1", "H1")
    assert contract.specs == ("sma(10)@M1",)


# =============================================================================
# ObservationContract
# =============================================================================

def test_observation_contract() -> None:
    contract = ObservationContract(
        symbol="XAUUSD",
        base_tf="m1",
        columns=["f1", "f2", "f3"],
        row_count=100,
    )
    assert contract.symbol == "XAUUSD"
    assert contract.base_tf == "M1"
    assert contract.columns == ("f1", "f2", "f3")
    assert contract.row_count == 100
    assert contract.dtype == "float64"
    assert contract.feature_count == 3


def test_observation_contract_zero_rows() -> None:
    contract = ObservationContract(
        symbol="XAUUSD",
        base_tf="M1",
        columns=(),
        row_count=0,
    )
    assert contract.row_count == 0
    assert contract.feature_count == 0


def test_observation_contract_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol is required"):
        ObservationContract(
            symbol="",
            base_tf="M1",
            columns=(),
            row_count=0,
        )


def test_observation_contract_empty_base_tf() -> None:
    with pytest.raises(ValueError, match="base_tf is required"):
        ObservationContract(
            symbol="XAUUSD",
            base_tf="",
            columns=(),
            row_count=0,
        )


def test_observation_contract_negative_rows() -> None:
    with pytest.raises(ValueError, match="row_count must be >= 0"):
        ObservationContract(
            symbol="XAUUSD",
            base_tf="M1",
            columns=(),
            row_count=-1,
        )


# =============================================================================
# FeatureSystemContract
# =============================================================================

def test_system_contract_ownership() -> None:
    contract = FEATURE_SYSTEM_CONTRACT
    assert contract.pipeline_owner == "FeaturePipeline"
    assert contract.engine_owner == "FeatureEngine"
    assert contract.cache_owner == "FeatureEngine"
    assert contract.store_owner == "FeatureStoreV2"
    assert contract.graph_owner == "FeatureGraph"
    assert contract.observation_owner == "ObservationBuilder"
    assert contract.registry_owner == "FeatureRegistry"


def test_system_contract_boundaries() -> None:
    contract = FEATURE_SYSTEM_CONTRACT
    assert contract.input_type == "MTFDataset"
    assert contract.feature_output_type == "MTFDataset"
    assert contract.observation_output_type == "pandas.DataFrame"


def test_system_contract_mode_sets() -> None:
    contract = FEATURE_SYSTEM_CONTRACT
    assert contract.modes == ALL_MODES
    assert contract.batch_modes == TRAIN_MODES
    assert contract.incremental_modes == INCREMENTAL_MODES


@pytest.mark.parametrize("mode", sorted(TRAIN_MODES))
def test_system_contract_batch_mode(mode: str) -> None:
    assert FEATURE_SYSTEM_CONTRACT.is_batch_mode(mode)
    assert not FEATURE_SYSTEM_CONTRACT.is_incremental_mode(mode)


@pytest.mark.parametrize("mode", sorted(INCREMENTAL_MODES))
def test_system_contract_incremental_mode(mode: str) -> None:
    assert FEATURE_SYSTEM_CONTRACT.is_incremental_mode(mode)
    assert not FEATURE_SYSTEM_CONTRACT.is_batch_mode(mode)


@pytest.mark.parametrize("mode", ["TRAIN", "Optimize", "LIVE", "Paper"])
def test_system_contract_validate_mode_case_normalization(mode: str) -> None:
    assert FEATURE_SYSTEM_CONTRACT.validate_mode(mode) == mode.lower()


def test_system_contract_validate_mode_invalid() -> None:
    with pytest.raises(ValueError, match="Unsupported mode"):
        FEATURE_SYSTEM_CONTRACT.validate_mode("invalid")


def test_custom_system_contract_rejects_batch_mode_outside_modes() -> None:
    with pytest.raises(ValueError, match="batch_modes must be a subset"):
        FeatureSystemContract(
            version="test",
            modes=frozenset({"a"}),
            batch_modes=frozenset({"b"}),
            incremental_modes=frozenset(),
        )


def test_custom_system_contract_rejects_incremental_mode_outside_modes() -> None:
    with pytest.raises(
        ValueError,
        match="incremental_modes must be a subset",
    ):
        FeatureSystemContract(
            version="test",
            modes=frozenset({"a"}),
            batch_modes=frozenset(),
            incremental_modes=frozenset({"b"}),
        )


def test_custom_system_contract_rejects_overlapping_modes() -> None:
    with pytest.raises(
        ValueError,
        match="batch_modes and incremental_modes must be disjoint",
    ):
        FeatureSystemContract(
            version="test",
            modes=frozenset({"a"}),
            batch_modes=frozenset({"a"}),
            incremental_modes=frozenset({"a"}),
        )


def test_system_contract_rejects_empty_version() -> None:
    with pytest.raises(ValueError, match="Contract version is required"):
        FeatureSystemContract(version="")


def test_system_contract_is_frozen() -> None:
    with pytest.raises((AttributeError, TypeError)):
        FEATURE_SYSTEM_CONTRACT.version = "2"


# =============================================================================
# Summary
# =============================================================================

def test_summarize_system_contract() -> None:
    summary = summarize_system_contract()

    assert summary["version"] == FEATURE_SYSTEM_CONTRACT_VERSION
    assert set(summary["modes"]) == set(ALL_MODES)
    assert set(summary["batch_modes"]) == set(TRAIN_MODES)
    assert set(summary["incremental_modes"]) == set(INCREMENTAL_MODES)

    assert summary["pipeline_owner"] == "FeaturePipeline"
    assert summary["engine_owner"] == "FeatureEngine"
    assert summary["cache_owner"] == "FeatureEngine"
    assert summary["store_owner"] == "FeatureStoreV2"
    assert summary["graph_owner"] == "FeatureGraph"
    assert summary["observation_owner"] == "ObservationBuilder"
    assert summary["registry_owner"] == "FeatureRegistry"

    assert summary["input_type"] == "MTFDataset"
    assert summary["feature_output_type"] == "MTFDataset"
    assert summary["observation_output_type"] == "pandas.DataFrame"

    assert summary["stages"] == [
        "raw",
        "features",
        "stored",
        "observation",
    ]
