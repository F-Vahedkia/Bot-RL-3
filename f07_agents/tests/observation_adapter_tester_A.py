# f07_agents/tests/observation_adapter_tester_A.py (t10)
#
# Run: pytest -v -s f07_agents/tests/observation_adapter_tester_A.py

# Goal:
#   Validate the official boundary between f04_features Observation and
#   f07_agents AgentObservation.
#
# This tester does not instantiate FeaturePipeline. It validates the adapter
# contract using DataFrames shaped exactly like the final Observation output.

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from f07_agents.contracts import AgentObservation
from f07_agents.observation_adapter import FeatureObservationAdapter


# =============================================================================
# Constants / Helpers
# =============================================================================
TIMESTAMP_1 = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
TIMESTAMP_2 = pd.Timestamp("2026-01-01 12:01:00", tz="UTC")
TIMESTAMP_3 = pd.Timestamp("2026-01-01 12:02:00", tz="UTC")

FEATURES = (
    'sma(column="close",period=10)@M1',
    'rsi(column="close",period=14)@M1',
    'time_hour@M1',
)


def _observation_df(
    *,
    index=None,
    values=None,
    columns=FEATURES,
):
    if index is None:
        index = pd.DatetimeIndex(
            [TIMESTAMP_1, TIMESTAMP_2, TIMESTAMP_3],
            name="timestamp",
        )
    else:
        index = pd.DatetimeIndex(index, name="timestamp")

    if values is None:
        values = [
            [10.0, 50.0, 12.0],
            [11.0, 51.0, 12.0],
            [12.0, 52.0, 12.0],
        ]

    return pd.DataFrame(values, index=index, columns=list(columns))


def _adapter(symbol="XAUUSD", base_tf="M1"):
    return FeatureObservationAdapter(
        symbol=symbol,
        base_tf=base_tf,
    )


# =============================================================================
# Constructor
# =============================================================================

def test_constructor_normalizes_symbol_and_base_tf():
    adapter = _adapter(symbol=" XAU USD ", base_tf=" m1 ")
    assert adapter.symbol == "XAUUSD"
    assert adapter.base_tf == "M1"


def test_constructor_rejects_empty_symbol():
    with pytest.raises(ValueError, match="symbol is required"):
        _adapter(symbol="   ")


def test_constructor_rejects_empty_base_tf():
    with pytest.raises(ValueError, match="base_tf is required"):
        _adapter(base_tf="   ")


# =============================================================================
# Observation DataFrame validation
# =============================================================================

def test_build_requires_observation():
    with pytest.raises(ValueError, match="observation is required"):
        _adapter().build(None)


def test_build_rejects_non_dataframe():
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        _adapter().build([[1.0, 2.0, 3.0]])


def test_build_rejects_empty_dataframe():
    df = pd.DataFrame(
        columns=list(FEATURES),
        index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
    )
    with pytest.raises(ValueError, match="must not be empty"):
        _adapter().build(df)


def test_build_requires_datetime_index():
    df = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        index=["2026-01-01 12:00:00"],
        columns=list(FEATURES),
    )
    with pytest.raises(TypeError, match="DatetimeIndex"):
        _adapter().build(df)


def test_build_requires_timezone_aware_index():
    idx = pd.DatetimeIndex(
        ["2026-01-01 12:00:00"],
        name="timestamp",
    )
    df = pd.DataFrame([[1.0, 2.0, 3.0]], index=idx, columns=list(FEATURES))
    with pytest.raises(ValueError, match="timezone-aware"):
        _adapter().build(df)


def test_build_requires_at_least_one_feature_column():
    df = pd.DataFrame(
        index=pd.DatetimeIndex(
            [TIMESTAMP_1],
            name="timestamp",
        )
    )
    with pytest.raises(ValueError, match="at least one feature column"):
        _adapter().build(df)


# =============================================================================
# build() -> latest AgentObservation
# =============================================================================

def test_build_returns_agent_observation():
    result = _adapter().build(_observation_df())
    assert isinstance(result, AgentObservation)


def test_build_uses_latest_timestamp():
    result = _adapter().build(_observation_df())
    assert result.timestamp == TIMESTAMP_3.to_pydatetime()


def test_build_preserves_symbol_and_base_tf():
    result = _adapter(symbol="XAUUSD", base_tf="M1").build(_observation_df())
    assert result.symbol == "XAUUSD"
    assert result.base_tf == "M1"


def test_build_extracts_latest_row_only():
    result = _adapter().build(_observation_df())
    assert result.values == (12.0, 52.0, 12.0)


def test_build_preserves_feature_column_order():
    result = _adapter().build(_observation_df())
    assert result.feature_names == FEATURES


def test_build_converts_values_to_float():
    df = _observation_df(
        values=[
            [10, 50, 12],
            [11, 51, 12],
            [12, 52, 12],
        ]
    )
    result = _adapter().build(df)
    assert result.values == (12.0, 52.0, 12.0)
    assert all(isinstance(value, float) for value in result.values)


def test_build_rejects_non_numeric_latest_row():
    df = _observation_df(
        values=[
            [10.0, 50.0, 12.0],
            [11.0, 51.0, 12.0],
            ["bad", 52.0, 12.0],
        ]
    )
    with pytest.raises(TypeError, match="numeric"):
        _adapter().build(df)


def test_build_rejects_nan_in_latest_row():
    df = _observation_df(
        values=[
            [10.0, 50.0, 12.0],
            [11.0, 51.0, 12.0],
            [np.nan, 52.0, 12.0],
        ]
    )
    with pytest.raises(ValueError, match="non-finite"):
        _adapter().build(df)


def test_build_rejects_infinite_in_latest_row():
    df = _observation_df(
        values=[
            [10.0, 50.0, 12.0],
            [11.0, 51.0, 12.0],
            [np.inf, 52.0, 12.0],
        ]
    )
    with pytest.raises(ValueError, match="non-finite"):
        _adapter().build(df)


def test_build_does_not_require_unique_index():
    idx = pd.DatetimeIndex(
        [TIMESTAMP_1, TIMESTAMP_2, TIMESTAMP_2],
        name="timestamp",
    )
    df = _observation_df(index=idx)
    result = _adapter().build(df)
    assert result.timestamp == TIMESTAMP_2.to_pydatetime()
    assert result.values == (12.0, 52.0, 12.0)


def test_build_does_not_mutate_dataframe():
    df = _observation_df()
    before = df.copy(deep=True)
    _adapter().build(df)
    pd.testing.assert_frame_equal(df, before)


# =============================================================================
# build_at()
# =============================================================================

def test_build_at_selects_requested_timestamp():
    result = _adapter().build_at(
        _observation_df(),
        timestamp=TIMESTAMP_2,
    )
    assert result.timestamp == TIMESTAMP_2.to_pydatetime()
    assert result.values == (11.0, 51.0, 12.0)


def test_build_at_accepts_python_datetime():
    timestamp = datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)
    result = _adapter().build_at(
        _observation_df(),
        timestamp=timestamp,
    )
    assert result.values == (11.0, 51.0, 12.0)


def test_build_at_rejects_naive_timestamp():
    naive = pd.Timestamp("2026-01-01 12:01:00")
    with pytest.raises(ValueError, match="timezone-aware"):
        _adapter().build_at(_observation_df(), timestamp=naive)


def test_build_at_rejects_missing_timestamp():
    with pytest.raises(KeyError, match="not found"):
        _adapter().build_at(
            _observation_df(),
            timestamp=pd.Timestamp("2026-01-01 12:03:00", tz="UTC"),
        )


def test_build_at_rejects_duplicate_index():
    idx = pd.DatetimeIndex(
        [TIMESTAMP_1, TIMESTAMP_2, TIMESTAMP_2],
        name="timestamp",
    )
    df = _observation_df(index=idx)
    with pytest.raises(ValueError, match="index must be unique"):
        _adapter().build_at(df, timestamp=TIMESTAMP_2)


def test_build_at_rejects_non_finite_selected_row():
    df = _observation_df(
        values=[
            [10.0, 50.0, 12.0],
            [11.0, np.nan, 12.0],
            [12.0, 52.0, 12.0],
        ]
    )
    with pytest.raises(ValueError, match="non-finite"):
        _adapter().build_at(df, timestamp=TIMESTAMP_2)


# =============================================================================
# build_sequence()
# =============================================================================

def test_build_sequence_returns_all_rows_in_order():
    result = _adapter().build_sequence(_observation_df())
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert [item.timestamp for item in result] == [
        TIMESTAMP_1.to_pydatetime(),
        TIMESTAMP_2.to_pydatetime(),
        TIMESTAMP_3.to_pydatetime(),
    ]


def test_build_sequence_preserves_values_and_feature_names():
    result = _adapter().build_sequence(_observation_df())
    assert result[0].values == (10.0, 50.0, 12.0)
    assert result[1].values == (11.0, 51.0, 12.0)
    assert result[2].values == (12.0, 52.0, 12.0)
    assert result[0].feature_names == FEATURES
    assert all(item.feature_names == FEATURES for item in result)


def test_build_sequence_rejects_duplicate_index():
    idx = pd.DatetimeIndex(
        [TIMESTAMP_1, TIMESTAMP_2, TIMESTAMP_2],
        name="timestamp",
    )
    df = _observation_df(index=idx)
    with pytest.raises(ValueError, match="index must be unique"):
        _adapter().build_sequence(df)


def test_build_sequence_rejects_non_finite_row():
    df = _observation_df(
        values=[
            [10.0, 50.0, 12.0],
            [11.0, 51.0, np.inf],
            [12.0, 52.0, 12.0],
        ]
    )
    with pytest.raises(ValueError, match="non-finite"):
        _adapter().build_sequence(df)


def test_build_sequence_does_not_mutate_dataframe():
    df = _observation_df()
    before = df.copy(deep=True)
    _adapter().build_sequence(df)
    pd.testing.assert_frame_equal(df, before)


# =============================================================================
# from_pipeline_output()
# =============================================================================

def test_from_pipeline_output_matches_build():
    df = _observation_df()
    direct = _adapter(symbol="EURUSD", base_tf="M5").build(df)
    via_factory = FeatureObservationAdapter.from_pipeline_output(
        observation=df,
        symbol="EURUSD",
        base_tf="M5",
    )
    assert via_factory == direct


def test_from_pipeline_output_preserves_contract_boundary():
    result = FeatureObservationAdapter.from_pipeline_output(
        observation=_observation_df(),
        symbol="GBPUSD",
        base_tf="H1",
    )
    assert result.symbol == "GBPUSD"
    assert result.base_tf == "H1"
    assert result.timestamp == TIMESTAMP_3.to_pydatetime()
    assert result.values == (12.0, 52.0, 12.0)
    assert result.feature_names == FEATURES


# =============================================================================
# Contract integration checks
# =============================================================================

def test_output_is_ready_for_symbol_agent_contract():
    result = _adapter().build(_observation_df())
    assert result.symbol == "XAUUSD"
    assert result.base_tf == "M1"
    assert result.timestamp.tzinfo is not None
    assert len(result.values) == len(result.feature_names)
    assert all(np.isfinite(value) for value in result.values)


def test_symbol_isolation_is_preserved():
    xau = _adapter(symbol="XAUUSD").build(_observation_df())
    eur = _adapter(symbol="EURUSD").build(_observation_df())
    assert xau.symbol == "XAUUSD"
    assert eur.symbol == "EURUSD"
    assert xau.values == eur.values
    assert xau.feature_names == eur.feature_names


# =============================================================================
# END
# =============================================================================
