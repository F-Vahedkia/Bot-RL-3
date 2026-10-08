# f03_data/tests_data_handler/test_read_raw_df.py

# Run: pytest f03_data/tests_data_handler/test_read_raw_df.py -v

import pandas as pd
import pytest
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)
from f03_data.data_handler_G import _read_raw_df


# =============================================================================
# Sample Data
# =============================================================================

@pytest.fixture
def sample_df():
    index = pd.date_range(
        "2026-01-01 00:00:00",
        periods=5,
        freq="1h",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "open":  [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
            "high":  [1.1010, 1.1020, 1.1030, 1.1040, 1.1050],
            "low":   [1.0990, 1.1000, 1.1010, 1.1020, 1.1030],
            "close": [1.1005, 1.1015, 1.1025, 1.1035, 1.1045],
            "volume": [100, 110, 120, 130, 140],
        },
        index=index,
    )


# =============================================================================
# 1. Parquet + DatetimeIndex
# =============================================================================

def test_read_raw_df_parquet_datetime_index(
    tmp_path,
    sample_df,
):
    path = tmp_path / "test.parquet"

    sample_df.to_parquet(path)

    result = _read_raw_df(
        path=path,
        mode="number",
        broker_timezone="UTC",
    )

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is not None
    assert str(result.index.tz) == "UTC"

    assert len(result) == 5

    assert list(result.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


# =============================================================================
# 2. Parquet + naive DatetimeIndex
# =============================================================================

def test_read_raw_df_parquet_naive_datetime_index(
    tmp_path,
    sample_df,
):
    path = tmp_path / "test.parquet"

    naive_df = sample_df.copy()
    naive_df.index = naive_df.index.tz_localize(None)

    naive_df.to_parquet(path)

    result = _read_raw_df(
        path=path,
        mode="number",
        broker_timezone="UTC",
    )

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is not None
    assert str(result.index.tz) == "UTC"

    assert len(result) == 5


# =============================================================================
# 3. CSV + time column
# =============================================================================

def test_read_raw_df_csv_time_column(
    tmp_path,
    sample_df,
):
    path = tmp_path / "test.csv"

    csv_df = sample_df.copy()

    # CSV باید time را به صورت column ذخیره کند
    csv_df.index.name = "time"
    csv_df.to_csv(path)

    result = _read_raw_df(
        path=path,
        mode="number",
        broker_timezone="UTC",
    )

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is not None
    assert str(result.index.tz) == "UTC"

    assert len(result) == 5


# =============================================================================
# 4. Column selection
# =============================================================================

def test_read_raw_df_columns(
    tmp_path,
    sample_df,
):
    path = tmp_path / "test.parquet"

    sample_df.to_parquet(path)

    result = _read_raw_df(
        path=path,
        mode="number",
        columns=["open", "high", "close"],
        broker_timezone="UTC",
    )

    assert list(result.columns) == [
        "open",
        "high",
        "close",
    ]

    assert len(result) == 5


# =============================================================================
# 5. number mode
# =============================================================================

def test_read_raw_df_number_mode(
    tmp_path,
    sample_df,
):
    path = tmp_path / "test.parquet"

    sample_df.to_parquet(path)

    result = _read_raw_df(
        path=path,
        mode="number",
        start_lastrows=4,
        end_lastrows=1,
        broker_timezone="UTC",
    )

    # از 5 ردیف:
    # start_lastrows=4  -> از ردیف index=1
    # end_lastrows=1    -> ردیف آخر حذف می‌شود
    # بنابراین 3 ردیف
    assert len(result) == 3

    assert result.index[0] == sample_df.index[1]
    assert result.index[-1] == sample_df.index[3]


# =============================================================================
# 6. number mode - all rows
# =============================================================================

def test_read_raw_df_number_mode_all_rows(
    tmp_path,
    sample_df,
):
    path = tmp_path / "test.parquet"

    sample_df.to_parquet(path)

    result = _read_raw_df(
        path=path,
        mode="number",
        start_lastrows=None,
        end_lastrows=None,
        broker_timezone="UTC",
    )

    assert len(result) == 5


# =============================================================================
# 7. number mode - zero rows
# =============================================================================

def test_read_raw_df_number_mode_zero_rows(
    tmp_path,
    sample_df,
):
    path = tmp_path / "test.parquet"

    sample_df.to_parquet(path)

    result = _read_raw_df(
        path=path,
        mode="number",
        start_lastrows=0,
        end_lastrows=0,
        broker_timezone="UTC",
    )

    assert result.empty


# =============================================================================
# 8. Negative row count must fail
# =============================================================================

def test_read_raw_df_negative_rows(
    tmp_path,
    sample_df,
):
    path = tmp_path / "test.parquet"

    sample_df.to_parquet(path)

    with pytest.raises(ValueError):
        _read_raw_df(
            path=path,
            mode="number",
            start_lastrows=-1,
            end_lastrows=0,
            broker_timezone="UTC",
        )


# =============================================================================
# 9. Invalid mode must fail
# =============================================================================

def test_read_raw_df_invalid_mode(
    tmp_path,
    sample_df,
):
    path = tmp_path / "test.parquet"

    sample_df.to_parquet(path)

    with pytest.raises(ValueError):
        _read_raw_df(
            path=path,
            mode="invalid",
            broker_timezone="UTC",
        )


# =============================================================================
# 10. Time mode
# =============================================================================

def test_read_raw_df_time_mode(
    tmp_path,
    sample_df,
):
    path = tmp_path / "test.parquet"

    sample_df.to_parquet(path)

    result = _read_raw_df(
        path=path,
        mode="time",
        start_time="2026-01-01 01:00:00+00:00",
        end_time="2026-01-01 04:00:00+00:00",
        broker_timezone="UTC",
    )

    # end_time is exclusive
    assert len(result) == 3

    assert result.index[0] == pd.Timestamp(
        "2026-01-01 01:00:00",
        tz="UTC",
    )

    assert result.index[-1] == pd.Timestamp(
        "2026-01-01 03:00:00",
        tz="UTC",
    )


# =============================================================================
# 11. Non-datetime index and no time column
# =============================================================================

def test_read_raw_df_invalid_index(
    tmp_path,
):
    path = tmp_path / "invalid.parquet"

    df = pd.DataFrame(
        {
            "open": [1.1, 1.2],
            "high": [1.2, 1.3],
            "low": [1.0, 1.1],
            "close": [1.15, 1.25],
            "volume": [100, 200],
        },
        index=[0, 1],
    )

    df.to_parquet(path)

    with pytest.raises(
        ValueError,
        match="DataFrame index is not DatetimeIndex and no 'time' column found",
    ):
        _read_raw_df(
            path=path,
            mode="number",
            broker_timezone="UTC",
        )