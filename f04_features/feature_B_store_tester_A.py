# f04_features/feature_B_store_tester_A.py
# Date reviewed:
#    1405/05/24-15:04 ==> run result is OK for 22 tests

# Run:
#   pytest -v -s f04_features/feature_B_store_tester_A.py

# Tests the frozen FeatureStoreV2 contract:
#   - align_to_base()
#   - FeatureStoreV2.build()
#   - FeatureStoreV2.extract_metadata()
#   - FeatureStoreV2.save()
#   - functional build_feature_store()
#
# This tester intentionally does NOT test legacy APIs or old aligned-dataset
# ownership assumptions.

from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import pytest

from f03_data.mtf_dataset import MTFDataset
from f04_features.feature_B_store import (
    FeatureStoreV2,
    align_to_base,
    build_feature_store,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_df(
    index: list[str],
    *,
    close_start: float = 1.0,
    feature_values: list[float] | None = None,
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index, tz="UTC", name="timestamp")

    n = len(idx)
    data = {
        "open": [close_start + i for i in range(n)],
        "high": [close_start + i + 0.5 for i in range(n)],
        "low": [close_start + i - 0.5 for i in range(n)],
        "close": [close_start + i for i in range(n)],
        "volume": [100 + i for i in range(n)],
    }

    if feature_values is not None:
        data["feature_x"] = feature_values

    return pd.DataFrame(data, index=idx)


def _make_dataset(
    symbol: str = "XAUUSD",
    *,
    include_m5: bool = False,
    include_feature: bool = False,
    feature_values: list[float] | None = None,
) -> MTFDataset:
    ds = MTFDataset(
        symbol=symbol,
        base_tf="M1",
    )

    m1 = _make_df(
        [
            "2026-01-01 10:00:00",
            "2026-01-01 10:01:00",
            "2026-01-01 10:02:00",
        ],
        feature_values=feature_values if include_feature else None,
    )
    ds.add("M1", m1)

    if include_m5:
        m5 = _make_df(
            [
                "2026-01-01 10:00:00",
                "2026-01-01 10:05:00",
            ],
            close_start=10.0,
        )
        ds.add("M5", m5)

    return ds


def _copy_frames(dataset: MTFDataset) -> dict[str, pd.DataFrame]:
    return {
        tf: dataset.get(tf).copy(deep=True)
        for tf in dataset.timeframes
    }


# =============================================================================
# align_to_base()
# =============================================================================

def test_align_to_base_returns_base_index() -> None:
    ds = _make_dataset(include_m5=True)

    aligned = align_to_base(ds)

    assert isinstance(aligned, pd.DataFrame)
    assert list(aligned.index) == list(ds.get("M1").index)
    assert aligned.index.name == ds.get("M1").index.name


def test_align_to_base_no_lookahead() -> None:
    ds = _make_dataset(include_m5=True)

    m5 = _make_df(
        [
            "2026-01-01 10:05:00",
        ],
        close_start=500.0,
    )
    m5["m5_marker"] = 500.0
    ds.replace("M5", m5)

    ds.replace(
        "M1",
        _make_df(
            [
                "2026-01-01 10:00:00",
                "2026-01-01 10:01:00",
                "2026-01-01 10:02:00",
                "2026-01-01 10:05:00",
                "2026-01-01 10:06:00",
                "2026-01-01 10:09:00",
                "2026-01-01 10:10:00",
            ],
        ),
    )

    aligned = align_to_base(ds)

    # M5 candle opens at 10:05 and closes at 10:10.
    # Therefore it must not be visible before 10:10.
    assert pd.isna(
        aligned.loc[
            pd.Timestamp("2026-01-01 10:05:00", tz="UTC"),
            "m5_marker",
        ]
    )

    # It becomes visible exactly when the M5 candle closes.
    assert (
        aligned.loc[
            pd.Timestamp("2026-01-01 10:10:00", tz="UTC"),
            "m5_marker",
        ]
        == 500.0
    )


def test_align_to_base_does_not_mutate_dataset() -> None:
    ds = _make_dataset(include_m5=True)
    before = _copy_frames(ds)

    aligned = align_to_base(ds)

    assert isinstance(aligned, pd.DataFrame)

    for tf in ds.timeframes:
        pd.testing.assert_frame_equal(ds.get(tf), before[tf])


def test_align_to_base_selected_timeframes() -> None:
    ds = _make_dataset(include_m5=True)

    aligned = align_to_base(
        ds,
        include_timeframes=["M1"],
    )

    assert list(aligned.columns) == list(ds.get("M1").columns)


# =============================================================================
# FeatureStoreV2._validate_datasets()
# =============================================================================

def test_build_rejects_none_inputs() -> None:
    store = FeatureStoreV2(config={})
    valid = _make_dataset()

    with pytest.raises(ValueError, match="dataset is required"):
        store.build(None, valid)

    with pytest.raises(ValueError, match="features is required"):
        store.build(valid, None)


def test_build_rejects_wrong_input_types() -> None:
    store = FeatureStoreV2(config={})
    valid = _make_dataset()

    with pytest.raises(TypeError, match="Expected dataset to be MTFDataset"):
        store.build("bad", valid)

    with pytest.raises(TypeError, match="Expected features to be MTFDataset"):
        store.build(valid, "bad")


def test_build_rejects_symbol_mismatch() -> None:
    store = FeatureStoreV2(config={})

    raw = _make_dataset(symbol="XAUUSD")
    features = _make_dataset(symbol="EURUSD", include_feature=True, feature_values=[7, 8, 9])

    with pytest.raises(ValueError, match="Dataset symbol mismatch"):
        store.build(raw, features)


def test_build_rejects_timeframe_mismatch() -> None:
    store = FeatureStoreV2(config={})

    raw = _make_dataset(include_m5=False)
    features = _make_dataset(include_m5=True, include_feature=True, feature_values=[7, 8, 9])

    with pytest.raises(ValueError, match="Timeframe mismatch"):
        store.build(raw, features)


# =============================================================================
# FeatureStoreV2.build()
# =============================================================================

def test_build_merges_new_feature_columns() -> None:
    store = FeatureStoreV2(config={})

    raw = _make_dataset()
    features = _make_dataset(
        include_feature=True,
        feature_values=[10.0, 20.0, 30.0],
    )

    result = store.build(raw, features)

    assert isinstance(result, MTFDataset)
    assert result.symbol == raw.symbol
    assert result.base_tf == raw.base_tf
    assert "feature_x" in result.get("M1").columns
    assert result.get("M1")["feature_x"].tolist() == [10.0, 20.0, 30.0]


def test_build_does_not_mutate_source_datasets() -> None:
    store = FeatureStoreV2(config={})

    raw = _make_dataset()
    features = _make_dataset(
        include_feature=True,
        feature_values=[10.0, 20.0, 30.0],
    )

    raw_before = _copy_frames(raw)
    features_before = _copy_frames(features)

    result = store.build(raw, features)

    assert "feature_x" not in raw.get("M1").columns
    assert "feature_x" in result.get("M1").columns

    for tf in raw.timeframes:
        pd.testing.assert_frame_equal(raw.get(tf), raw_before[tf])

    for tf in features.timeframes:
        pd.testing.assert_frame_equal(features.get(tf), features_before[tf])


def test_build_preserves_equal_existing_column() -> None:
    store = FeatureStoreV2(config={})

    raw = _make_dataset(
        include_feature=True,
        feature_values=[10.0, 20.0, 30.0],
    )
    features = _make_dataset(
        include_feature=True,
        feature_values=[10.0, 20.0, 30.0],
    )

    result = store.build(raw, features)

    assert result.get("M1")["feature_x"].tolist() == [10.0, 20.0, 30.0]


def test_build_rejects_differing_column_collision() -> None:
    store = FeatureStoreV2(config={})

    raw = _make_dataset(
        include_feature=True,
        feature_values=[10.0, 20.0, 30.0],
    )
    features = _make_dataset(
        include_feature=True,
        feature_values=[11.0, 20.0, 30.0],
    )

    with pytest.raises(ValueError, match="FeatureStore column collision"):
        store.build(raw, features)


def test_build_returns_independent_dataset() -> None:
    store = FeatureStoreV2(config={})

    raw = _make_dataset()
    features = _make_dataset(
        include_feature=True,
        feature_values=[10.0, 20.0, 30.0],
    )

    result = store.build(raw, features)

    assert result is not raw

    result.get("M1").loc[
        pd.Timestamp("2026-01-01 10:00:00", tz="UTC"),
        "close",
    ] = 999.0

    assert (
        raw.get("M1").loc[
            pd.Timestamp("2026-01-01 10:00:00", tz="UTC"),
            "close",
        ]
        != 999.0
    )


# =============================================================================
# extract_metadata()
# =============================================================================

def test_extract_metadata_schema() -> None:
    store = FeatureStoreV2(config={})
    ds = _make_dataset(
        include_feature=True,
        feature_values=[10.0, 20.0, 30.0],
    )

    metadata = store.extract_metadata(ds)

    assert set(metadata) == {"M1"}
    frame = metadata["M1"]

    expected_columns = {
        "timeframe",
        "column",
        "dtype",
        "first_valid_ts",
        "nan_count",
        "coverage_ratio",
    }

    assert set(frame.columns) == expected_columns
    assert len(frame) == len(ds.get("M1").columns)


def test_extract_metadata_nan_and_coverage() -> None:
    store = FeatureStoreV2(config={})

    ds = MTFDataset(symbol="XAUUSD", base_tf="M1")
    df = _make_df(
        [
            "2026-01-01 10:00:00",
            "2026-01-01 10:01:00",
            "2026-01-01 10:02:00",
        ],
    )
    df["feature_x"] = [float("nan"), 2.0, 3.0]
    ds.add("M1", df)

    metadata = store.extract_metadata(ds)["M1"]
    row = metadata.loc[metadata["column"] == "feature_x"].iloc[0]

    assert row["nan_count"] == 1
    assert row["coverage_ratio"] == pytest.approx(2 / 3)


def test_extract_metadata_empty_dataframe() -> None:
    store = FeatureStoreV2(config={})

    ds = MTFDataset(symbol="XAUUSD", base_tf="M1")
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
    )
    ds.add("M1", empty)

    metadata = store.extract_metadata(ds)["M1"]

    assert list(metadata.columns) == [
        "timeframe",
        "column",
        "dtype",
        "first_valid_ts",
        "nan_count",
        "coverage_ratio",
    ]
    assert len(metadata) == 5
    assert set(metadata["column"]) == {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    assert metadata["first_valid_ts"].isna().all()
    assert (metadata["nan_count"] == 0).all()
    assert (metadata["coverage_ratio"] == 0.0).all()


# =============================================================================
# save()
# =============================================================================

@pytest.mark.parametrize("fmt, suffix", [("csv", ".csv"), ("parquet", ".parquet")])
def test_save_persists_each_timeframe(
    tmp_path: Path,
    fmt: str,
    suffix: str,
) -> None:
    store = FeatureStoreV2(config={})

    ds = _make_dataset(include_m5=True)
    metadata = store.extract_metadata(ds)

    paths = store.save(
        dataset=ds,
        metadata=metadata,
        out_dir=tmp_path,
        name="features",
        fmt=fmt,
    )

    assert set(paths) == {"M1", "M5"}

    for tf in ["M1", "M5"]:
        tf_dir = tmp_path / tf

        assert (tf_dir / f"features{suffix}").exists()
        assert (tf_dir / "features.meta.csv").exists()
        assert (tf_dir / "features.meta.json").exists()

        with (tf_dir / "features.meta.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        assert isinstance(payload, list)


def test_save_rejects_unsupported_format(tmp_path: Path) -> None:
    store = FeatureStoreV2(config={})
    ds = _make_dataset()
    metadata = store.extract_metadata(ds)

    with pytest.raises(ValueError, match="Unsupported format"):
        store.save(
            dataset=ds,
            metadata=metadata,
            out_dir=tmp_path,
            name="features",
            fmt="json",
        )


def test_save_requires_name(tmp_path: Path) -> None:
    store = FeatureStoreV2(config={})
    ds = _make_dataset()
    metadata = store.extract_metadata(ds)

    with pytest.raises(ValueError, match="name is required"):
        store.save(
            dataset=ds,
            metadata=metadata,
            out_dir=tmp_path,
            name="",
        )


# =============================================================================
# Functional API
# =============================================================================

def test_build_feature_store_functional_api(tmp_path: Path) -> None:
    raw = _make_dataset()
    features = _make_dataset(
        include_feature=True,
        feature_values=[100.0, 200.0, 300.0],
    )

    paths = build_feature_store(
        dataset=raw,
        features=features,
        out_dir=tmp_path,
        name="features",
        fmt="csv",
        config={},
    )

    assert "M1" in paths
    assert (tmp_path / "M1" / "features.csv").exists()
    assert (tmp_path / "M1" / "features.meta.csv").exists()
    assert (tmp_path / "M1" / "features.meta.json").exists()


# =============================================================================
# Contract regression
# =============================================================================

def test_store_does_not_create_aligned_attribute() -> None:
    store = FeatureStoreV2(config={})
    raw = _make_dataset()
    features = _make_dataset(
        include_feature=True,
        feature_values=[1.0, 2.0, 3.0],
    )

    result = store.build(raw, features)

    # The current contract keeps alignment as a pure utility and does not
    # attach an "aligned" dataset to MTFDataset.
    assert not hasattr(result, "aligned") or result.aligned is None

