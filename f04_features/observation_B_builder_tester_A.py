# f04_features/observation_B_builder_tester_A.py
# Date reviewed:
#    1405/05/24:15:09 --> run result is OK for 23 tests

# Run:
#   pytest -v -s f04_features/observation_B_builder_tester_A.py

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f03_data.mtf_dataset import MTFDataset
from f04_features.feature_B_graph import FeatureGraph
from f04_features.observation_B_builder import ObservationBuilder

# ------------------------------------------------------------------- 1
def _frame(times, start=1.0):
    idx = pd.DatetimeIndex(times, tz="UTC", name="timestamp")
    n = len(idx)
    return pd.DataFrame({
        "open": [start+i for i in range(n)],
        "high": [start+i+0.5 for i in range(n)],
        "low": [start+i-0.5 for i in range(n)],
        "close": [start+i for i in range(n)],
        "volume": [100+i for i in range(n)],
    }, index=idx)

# ------------------------------------------------------------------- 2
def _ds(features=None):
    ds = MTFDataset(symbol="XAUUSD", base_tf="M1")
    df = _frame([
        "2026-01-01 10:00:00","2026-01-01 10:01:00",
        "2026-01-01 10:02:00","2026-01-01 10:03:00",
    ])
    for k, v in (features or {}).items():
        df[k] = v
    ds.add("M1", df)
    return ds

# ------------------------------------------------------------------- 3
def _cfg(shift=0, drop=False, whitelist=None, blacklist=None):
    return {
        "features": {
            "observation": {
                "shift_features_by": shift,
                "drop_na_head": drop,
                "features_whitelist": whitelist or [],
                "features_blacklist": blacklist or [],
            },
        },
    }

# ------------------------------------------------------------------- new-1
def _cfg_candles(
    *,
    base_mode="union_min",
    base_timeframe=None,
    required=None,
    names=None,
):
    return {
        "features": {
            "observation": {
                "base_timeframe": {
                    "mode": base_mode,
                    "timeframe": base_timeframe,
                },
            },
            "candles": {
                "required": {
                    "XAUUSD": required or {},
                },
                "names": names or [],
            },
        },
    }

# ------------------------------------------------------------------- 4
def test_builder_requires_config():
    with pytest.raises(ValueError, match="config is required"):
        ObservationBuilder(None)

# ------------------------------------------------------------------- 5
def test_builder_rejects_non_dict_config():
    with pytest.raises(TypeError, match="config must be dict"):
        ObservationBuilder([])

# ------------------------------------------------------------------- 6
def test_negative_shift_is_rejected():
    with pytest.raises(ValueError, match="shift_features_by must be >= 0"):
        ObservationBuilder(_cfg(shift=-1))

# ------------------------------------------------------------------- 7
def test_build_input_validation():
    b = ObservationBuilder(_cfg())
    g = FeatureGraph(['sma(column="close",period=10)@M1'])
    ds = _ds()
    with pytest.raises(ValueError, match="dataset is required"):
        b.build(None, g)
    with pytest.raises(ValueError, match="graph is required"):
        b.build(ds, None)
    with pytest.raises(TypeError, match="Expected dataset to be MTFDataset"):
        b.build("bad", g)
    with pytest.raises(TypeError, match="Expected graph to be FeatureGraph"):
        b.build(ds, "bad")

# ------------------------------------------------------------------- 8
def test_empty_dataset_rejected():
    b = ObservationBuilder(_cfg())
    ds = MTFDataset(symbol="XAUUSD", base_tf="M1")
    g = FeatureGraph(['sma(column="close",period=10)@M1'])
    with pytest.raises(ValueError, match="contains no timeframe frames"):
        b.build(ds, g)

# ------------------------------------------------------------------- 9
def test_feature_columns_single_output():
    b = ObservationBuilder(_cfg())
    g = FeatureGraph(['sma(column="close",period=10)@M1'])
    assert b.feature_columns(g) == ['sma(column="close",period=10)@M1']

# ------------------------------------------------------------------- 10
def test_feature_columns_multi_output():
    b = ObservationBuilder(_cfg())
    g = FeatureGraph(['macd(fast=12,slow=26,signal=9)@M1'])
    assert b.feature_columns(g) == [
        'macd(fast=12,slow=26,signal=9)@M1',
        'macd_signal(fast=12,slow=26,signal=9)@M1',
        'macd_hist(fast=12,slow=26,signal=9)@M1',
    ]

# ------------------------------------------------------------------- 11
# def test_unknown_registry_indicator_rejected():
#     b = ObservationBuilder(_cfg())
#     class FakeNode:
#         name = "does_not_exist"
#         canonical = "does_not_exist@M1"
#     class FakeGraph:
#         def execution_order(self):
#             return [FakeNode()]
#     with pytest.raises(ValueError, match="not present in Registry"):
#         b.feature_columns(FakeGraph())

# ------------------------------------------------------------------- 12
def test_build_selects_only_graph_features():
    spec = 'sma(column="close",period=10)@M1'
    ds = _ds({spec: [10.,20.,30.,40.], "unrelated":[9.,9.,9.,9.]})
    result = ObservationBuilder(_cfg()).build(ds, FeatureGraph([spec]))
    assert list(result.columns) == [spec]
    assert result[spec].tolist() == [10.,20.,30.,40.]

# ------------------------------------------------------------------- 13
def test_missing_feature_column_rejected():
    spec = 'sma(column="close",period=10)@M1'
    with pytest.raises(KeyError, match="required by FeatureGraph"):
        ObservationBuilder(_cfg()).build(_ds(), FeatureGraph([spec]))

# ------------------------------------------------------------------- 14
def test_multi_output_build():
    spec = 'macd(fast=12,slow=26,signal=9)@M1'
    ds = _ds({
        spec:[1.,2.,3.,4.],
        'macd_signal(fast=12,slow=26,signal=9)@M1':[.1,.2,.3,.4],
        'macd_hist(fast=12,slow=26,signal=9)@M1':[.9,1.8,2.7,3.6],
    })
    result = ObservationBuilder(_cfg()).build(ds, FeatureGraph([spec]))
    assert list(result.columns) == [
        spec,
        'macd_signal(fast=12,slow=26,signal=9)@M1',
        'macd_hist(fast=12,slow=26,signal=9)@M1',
    ]

# ------------------------------------------------------------------- 15
def test_build_does_not_mutate_dataset():
    spec = 'sma(column="close",period=10)@M1'
    ds = _ds({spec:[1.,2.,3.,4.]})
    before = ds.get("M1").copy(deep=True)
    ObservationBuilder(_cfg()).build(ds, FeatureGraph([spec]))
    pd.testing.assert_frame_equal(ds.get("M1"), before)

# ------------------------------------------------------------------- 16
def test_whitelist_filters():
    s1 = 'sma(column="close",period=10)@M1'
    s2 = 'rsi(column="close",period=14)@M1'
    ds = _ds({
        s1:[1.,2.,3.,4.],
        'rsi(column="close",period=14)@M1':[50.,51.,52.,53.],
    })
    b = ObservationBuilder(_cfg(whitelist=["rsi"]))
    out = b.build(ds, FeatureGraph([s1,s2]))
    assert list(out.columns) == ['rsi(column="close",period=14)@M1']

# ------------------------------------------------------------------- 17
def test_blacklist_filters():
    s1 = 'sma(column="close",period=10)@M1'
    s2 = 'rsi(column="close",period=14)@M1'
    ds = _ds({
        s1:[1.,2.,3.,4.],
        'rsi(column="close",method="ema",period=14)@M1':[50.,51.,52.,53.],
    })
    b = ObservationBuilder(_cfg(blacklist=["rsi"]))
    out = b.build(ds, FeatureGraph([s1,s2]))
    assert list(out.columns) == [s1]

# ------------------------------------------------------------------- 18
def test_glob_filter():
    spec = 'sma(column="close",period=10)@M1'
    ds = _ds({spec:[1.,2.,3.,4.]})
    b = ObservationBuilder(_cfg(whitelist=["*period=10*"]))
    assert list(b.build(ds, FeatureGraph([spec])).columns) == [spec]

# ------------------------------------------------------------------- 19
def test_shift():
    spec = 'sma(column="close",period=10)@M1'
    ds = _ds({spec:[10.,20.,30.,40.]})
    out = ObservationBuilder(_cfg(shift=1)).build(ds, FeatureGraph([spec]))
    assert pd.isna(out[spec].iloc[0])
    assert out[spec].iloc[1:].tolist() == [10.,20.,30.]

# ------------------------------------------------------------------- 20
def test_drop_leading_warmup_only():
    spec = 'sma(column="close",period=10)@M1'
    ds = _ds({spec:[np.nan,np.nan,30.,np.nan]})
    out = ObservationBuilder(_cfg(drop=True)).build(ds, FeatureGraph([spec]))
    assert len(out) == 2
    assert out[spec].iloc[0] == 30.0
    assert pd.isna(out[spec].iloc[1])

# ------------------------------------------------------------------- 21
def test_drop_na_false_preserves_leading_nan():
    spec = 'sma(column="close",period=10)@M1'
    ds = _ds({spec:[np.nan,np.nan,30.,40.]})
    out = ObservationBuilder(_cfg(drop=False)).build(ds, FeatureGraph([spec]))
    assert len(out) == 4
    assert out[spec].isna().iloc[:2].all()

# ------------------------------------------------------------------- 22
def test_feature_order_follows_graph():
    s1 = 'rsi(column="close",period=14)@M1'
    s2 = 'sma(column="close",period=10)@M1'
    ds = _ds({
        'rsi(column="close",period=14)@M1':[50.,51.,52.,53.],
        s2:[1.,2.,3.,4.],
    })
    out = ObservationBuilder(_cfg()).build(ds, FeatureGraph([s1,s2]))
    assert list(out.columns) == [
        s1, s2
    ]

# ------------------------------------------------------------------- 23
def test_build_numpy_float64():
    spec = 'sma(column="close",period=10)@M1'
    ds = _ds({spec:[1.,2.,3.,4.]})
    arr = ObservationBuilder(_cfg()).build_numpy(ds, FeatureGraph([spec]))
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert arr.shape == (4,1)
    np.testing.assert_allclose(arr[:,0], [1.,2.,3.,4.])

# ------------------------------------------------------------------- 24
def test_build_numpy_non_numeric_rejected():
    spec = 'sma(column="close",period=10)@M1'
    ds = _ds({spec:["a","b","c","d"]})
    with pytest.raises(TypeError, match="cannot be converted to float"):
        ObservationBuilder(_cfg()).build_numpy(ds, FeatureGraph([spec]))

# ------------------------------------------------------------------- 25
def test_no_aligned_attribute_required():
    spec = 'sma(column="close",period=10)@M1'
    ds = _ds({spec:[1.,2.,3.,4.]})
    assert not hasattr(ds, "aligned") or ds.aligned is None
    out = ObservationBuilder(_cfg()).build(ds, FeatureGraph([spec]))
    assert list(out.columns) == [spec]

# ------------------------------------------------------------------- 26
def test_higher_timeframe_alignment():
    ds = MTFDataset(symbol="XAUUSD", base_tf="M1")
    m1 = _frame([
    "2026-01-01 10:00:00",
    "2026-01-01 10:01:00",
    "2026-01-01 11:04:00",
    "2026-01-01 11:05:00",
    ])
    m1['sma(column="close",period=10)@M1'] = [1.,2.,3.,4.]
    h1 = _frame(["2026-01-01 10:05:00"], start=100.)
    h1['rsi(column="close",period=14)@H1'] = [55.]
    ds.add("M1", m1)
    ds.add("H1", h1)
    g = FeatureGraph([
        'sma(column="close",period=10)@M1',
        'rsi(column="close",period=14)@H1',
    ])
    out = ObservationBuilder(_cfg()).build(ds, g)
    assert list(out.columns) == [
        'sma(column="close",period=10)@M1',
        'rsi(column="close",period=14)@H1',
    ]
    # assert pd.isna(out.iloc[1, 1])
    # assert pd.isna(out.iloc[2, 1])
    assert pd.isna(
        out.loc[
            pd.Timestamp("2026-01-01 11:04:00", tz="UTC"),
            'rsi(column="close",period=14)@H1',
        ]
    ) is False

    assert (
        out.loc[
            pd.Timestamp("2026-01-01 11:04:00", tz="UTC"),
            'rsi(column="close",period=14)@H1',
        ]
        == 55.0
    )

    assert (
        out.loc[
            pd.Timestamp("2026-01-01 11:05:00", tz="UTC"),
            'rsi(column="close",period=14)@H1',
        ]
        == 55.0
    )

# ------------------------------------------------------------------- 27
def test_graph_contract_integration():
    spec = 'macd(fast=12,slow=26,signal=9)@H1'
    g = FeatureGraph([spec])
    node = g.all_nodes()[0]
    assert node.name == "macd"
    assert node.timeframe == "H1"
    assert node.canonical == spec
    assert ObservationBuilder(_cfg()).feature_columns(g) == [
        spec,
        'macd_signal(fast=12,slow=26,signal=9)@H1',
        'macd_hist(fast=12,slow=26,signal=9)@H1',
    ]

# ------------------------------------------------------------------- new-2
def test_observation_includes_candle_window():
    """
    تست اتصال CandleWindowBuilder
    """
    ds = MTFDataset(symbol="XAUUSD", base_tf="M1")

    m1 = _frame([
        "2026-01-01 10:00:00",
        "2026-01-01 10:01:00",
        "2026-01-01 10:02:00",
        "2026-01-01 10:03:00",
        "2026-01-01 10:04:00",
    ])

    spec = 'sma(column="close",period=10)@M1'
    m1[spec] = [1., 2., 3., 4., 5.]

    ds.add("M1", m1)

    cfg = _cfg_candles(
        required={"M1": 2},
        names=["close"],
    )

    out = ObservationBuilder(cfg).build(
        ds,
        FeatureGraph([spec]),
    )

    assert list(out.columns) == [
        spec,
        "M1_lag0_close",
        "M1_lag1_close",
    ]

# ------------------------------------------------------------------- new-3
def test_candle_window_uses_last_closed_candle_only():
    """
    تست مهم close_time <= T
    این تست دقیقاً قرارداد lag0 را کنترل می‌کند.
    """
    ds = MTFDataset(symbol="XAUUSD", base_tf="M5")

    m1 = _frame([
        "2026-01-01 10:00:00",
        "2026-01-01 10:01:00",
        "2026-01-01 10:02:00",
        "2026-01-01 10:03:00",
        "2026-01-01 10:04:00",
        "2026-01-01 10:05:00",
        "2026-01-01 10:06:00",
        "2026-01-01 10:07:00",
        "2026-01-01 10:08:00",
        "2026-01-01 10:09:00",
        "2026-01-01 10:10:00",
    ])

    spec = 'sma(column="close",period=10)@M1'
    m1[spec] = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]

    ds.add("M1", m1)

    cfg = _cfg_candles(
        base_mode="fixed",
        base_timeframe="M1",
        required={"M5": 2},
        names=["close"],
    )

    m5 = _frame(
        [
            "2026-01-01 10:00:00",
            "2026-01-01 10:05:00",
        ],
        start=100.0,
    )
    ds.add("M5", m5)

    out = ObservationBuilder(cfg).build(
        ds,
        FeatureGraph([spec]),
    )

    # M5 candle at 10:00 closes at 10:05.
    assert pd.isna(
        out.loc[
            pd.Timestamp("2026-01-01 10:04:00", tz="UTC"),
            "M5_lag0_close",
        ]
    )

    assert (
        out.loc[
            pd.Timestamp("2026-01-01 10:05:00", tz="UTC"),
            "M5_lag0_close",
        ]
        == 100.0
    )

    assert pd.isna(
        out.loc[
            pd.Timestamp("2026-01-01 10:05:00", tz="UTC"),
            "M5_lag1_close",
        ]
    )

    # At 10:10 the second M5 candle is closed, so:
    # lag0 -> 10:05 candle
    # lag1 -> 10:00 candle
    assert (
        out.loc[
            pd.Timestamp("2026-01-01 10:10:00", tz="UTC"),
            "M5_lag0_close",
        ]
        == 101.0
    )

    assert (
        out.loc[
            pd.Timestamp("2026-01-01 10:10:00", tz="UTC"),
            "M5_lag1_close",
        ]
        == 100.0
    )

# ------------------------------------------------------------------- new-4
def test_union_min_observation_index_does_not_use_dataset_base_tf():
    """
    تست مالکیت Observation Index
    این تست جلوی برگشتن ناخواسته به dataset.base_tf را می‌گیرد.
    """
    ds = MTFDataset(symbol="XAUUSD", base_tf="M5")

    m1 = _frame([
        "2026-01-01 10:00:00",
        "2026-01-01 10:01:00",
        "2026-01-01 10:02:00",
        "2026-01-01 10:03:00",
    ])

    spec = 'sma(column="close",period=10)@M1'
    m1[spec] = [1., 2., 3., 4.]

    ds.add("M1", m1)

    cfg = _cfg_candles(
        base_mode="union_min",
        required={"M1": 1},
        names=["close"],
    )

    out = ObservationBuilder(cfg).build(
        ds,
        FeatureGraph([spec]),
    )

    assert out.index.equals(m1.index)
    assert len(out) == 4

# ------------------------------------------------------------------- new-5
def test_candle_observation_does_not_mutate_dataset():
    """
    تست عدم تغییر Dataset
    """
    ds = MTFDataset(symbol="XAUUSD", base_tf="M5")

    m1 = _frame([
        "2026-01-01 10:00:00",
        "2026-01-01 10:01:00",
        "2026-01-01 10:02:00",
    ])

    spec = 'sma(column="close",period=10)@M1'
    m1[spec] = [1., 2., 3.]

    ds.add("M1", m1)

    before = {
        tf: frame.copy(deep=True)
        for tf, frame in ds.frames.items()
    }

    cfg = _cfg_candles(
        required={"M1": 2},
        names=["close"],
    )

    ObservationBuilder(cfg).build(
        ds,
        FeatureGraph([spec]),
    )

    for tf, frame_before in before.items():
        pd.testing.assert_frame_equal(
            ds.get(tf),
            frame_before,
        )

# ============================================================================= END
