# f04_features/feature_pipeline_tester_A.py
# Date reviewed:
#    1405/05/24:15:07  --> run result is OK for 17 tests

# Run: pytest -v -s f04_features/feature_pipeline_tester_A.py

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from f03_data.mtf_dataset import MTFDataset
from f04_features.feature_pipeline import FeaturePipeline


SYMBOL = "XAUUSD"


def frame():
    idx = pd.date_range(
        "2026-01-01 10:00:00",
        periods=3,
        freq="min",
        tz="UTC",
        name="timestamp",
    )
    return pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.0, 2.0, 3.0],
            "volume": [100, 101, 102],
        },
        index=idx,
    )


def ds(symbol=SYMBOL):
    d = MTFDataset(symbol=symbol, base_tf="M1")
    d.add("M1", frame())
    return d


def pipe(config=None, specs=None, symbol=SYMBOL):
    e = MagicMock()
    s = MagicMock()
    b = MagicMock()
    g = MagicMock()

    p = FeaturePipeline(
        symbol,
        e,
        s,
        b,
        g,
        specs or ["sma(10)@M1"],
        config or {"__base_tfs_dict": {symbol: "M1"}},
    )

    return p, e, s, b, g


def test_init_and_accessors():
    p, e, s, b, g = pipe()

    assert p.engine is e
    assert p.store is s
    assert p.graph is g
    assert p.dataset is None
    assert p.features is None
    assert p.observation is None


def test_base_tf_and_repr():
    p, *_ = pipe(
        config={"__base_tfs_dict": {SYMBOL: "H1"}},
        specs=["a", "b"],
    )

    assert p.base_tf == "H1"
    assert repr(p) == (
        "FeaturePipeline(symbol='XAUUSD', base_tf='H1', specs=2)"
    )

    q, *_ = pipe(config={"__base_tfs_dict": {}})

    with pytest.raises(
        RuntimeError,
        match="No base_tf found for symbol",
    ):
        _ = q.base_tf


def test_reset_live_state():
    p, e, *_ = pipe()

    p._dataset = ds()
    p._features = ds()
    p._observation = pd.DataFrame({"x": [1]})

    p.reset_live_state()

    e.reset_live_state.assert_called_once()
    assert p.dataset is None
    assert p.features is None
    assert p.observation is None


def test_run_validation():
    p, *_ = pipe()

    with pytest.raises(ValueError, match="dataset is None"):
        p.run(None)

    with pytest.raises(TypeError, match="Expected MTFDataset"):
        p.run("bad")

    with pytest.raises(
        ValueError,
        match="does not match pipeline symbol",
    ):
        p.run(ds("EURUSD"))


def test_run_full_flow():
    p, e, s, b, g = pipe()

    raw = ds()
    features = ds()
    stored = ds()

    obs = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    meta = {"M1": pd.DataFrame()}

    e.execute.return_value = features
    s.build.return_value = stored
    s.extract_metadata.return_value = meta
    b.build.return_value = obs

    r = p.run(raw, mode="train")

    e.execute.assert_called_once_with(
        dataset=raw,
        specs=p.feature_specs,
        mode="train",
    )

    s.build.assert_called_once_with(
        raw,
        features,
    )

    s.extract_metadata.assert_called_once_with(
        stored,
    )

    b.build.assert_called_once_with(
        stored,
        g,
    )

    assert r == {
        "dataset": raw,
        "features": stored,
        "metadata": meta,
        "observation": obs,
    }

    assert p.dataset is raw
    assert p.features is stored
    assert p.observation is obs


def test_run_engine_none_and_save_validation():
    p, e, s, b, _ = pipe()

    e.execute.return_value = None

    with pytest.raises(
        RuntimeError,
        match="FeatureEngine returned None",
    ):
        p.run(ds())

    e.execute.return_value = ds()
    s.build.return_value = ds()
    s.extract_metadata.return_value = {}
    b.build.return_value = pd.DataFrame(
        {"x": [1, 2, 3]}
    )

    with pytest.raises(
        ValueError,
        match="save_dir must be provided",
    ):
        p.run(
            ds(),
            save_features=True,
        )


def test_run_save_delegates(monkeypatch):
    p, e, s, b, _ = pipe()

    e.execute.return_value = ds()
    s.build.return_value = ds()
    s.extract_metadata.return_value = {}
    b.build.return_value = pd.DataFrame(
        {"x": [1, 2, 3]}
    )

    spy = MagicMock()
    monkeypatch.setattr(
        p,
        "save_feature_store",
        spy,
    )

    p.run(
        ds(),
        save_features=True,
        save_dir="out",
        save_name="abc",
    )

    spy.assert_called_once_with(
        p.features,
        out_dir="out",
        name="abc",
    )


def test_process_live_flow():
    p, e, s, b, g = pipe()

    raw = ds()
    features = ds()
    stored = ds()
    obs = pd.DataFrame({"x": [1, 2, 3]})

    # New FeaturePipeline live flow uses FeatureEngine.execute().
    e.execute.return_value = features

    # FeatureStoreV2.build() returns the stored Feature Dataset.
    s.build.return_value = stored

    # ObservationBuilder builds the final Observation.
    b.build.return_value = obs

    assert p.process_live(raw) is obs

    e.execute.assert_called_once_with(
        dataset=raw,
        specs=p.feature_specs,
        mode="live",
    )
    s.build.assert_called_once_with(
        dataset=raw,
        features=features,
    )
    b.build.assert_called_once_with(
        stored,
        g,
    )


def test_process_live_validation_and_none():
    p, e, s, b, _ = pipe()

    with pytest.raises(ValueError, match="dataset is None"):
        p.process_live(None)

    with pytest.raises(TypeError, match="Expected MTFDataset"):
        p.process_live("bad")

    with pytest.raises(ValueError, match="does not match pipeline symbol"):
        p.process_live(ds("EURUSD"))

    # New live flow uses FeatureEngine.execute().
    e.execute.return_value = None

    assert p.process_live(ds()) is None

    s.build.assert_not_called()
    b.build.assert_not_called()


def test_build_observation():
    p, _, _, b, g = pipe()

    dataset = ds()
    expected = pd.DataFrame({"x": [1, 2, 3]})
    b.build.return_value = expected

    assert p.build_observation(dataset) is expected

    b.build.assert_called_once()
    called_dataset, called_graph = b.build.call_args.args

    assert called_dataset is dataset
    assert called_graph is g


def test_build_feature_store():
    p, _, s, _, _ = pipe()

    raw = ds()
    feat = ds()
    result = ds()

    s.build.return_value = result

    assert p.build_feature_store(
        raw,
        feat,
    ) is result

    s.build.assert_called_once_with(
        raw,
        feat,
    )

    with pytest.raises(
        ValueError,
        match="raw_dataset is None",
    ):
        p.build_feature_store(
            None,
            feat,
        )

    with pytest.raises(
        ValueError,
        match="feature_dataset is None",
    ):
        p.build_feature_store(
            raw,
            None,
        )

    with pytest.raises(
        TypeError,
        match="Expected raw_dataset to be MTFDataset",
    ):
        p.build_feature_store(
            "bad",
            feat,
        )

    with pytest.raises(
        TypeError,
        match="Expected feature_dataset to be MTFDataset",
    ):
        p.build_feature_store(
            raw,
            "bad",
        )

    with pytest.raises(
        ValueError,
        match="does not match pipeline symbol",
    ):
        p.build_feature_store(
            ds("EURUSD"),
            feat,
        )

    s.build.return_value = None

    with pytest.raises(
        RuntimeError,
        match="FeatureStore returned None",
    ):
        p.build_feature_store(
            raw,
            feat,
        )

    s.build.return_value = "bad"

    with pytest.raises(
        TypeError,
        match="FeatureStore must return MTFDataset",
    ):
        p.build_feature_store(
            raw,
            feat,
        )


def test_save_feature_store():
    p, _, s, _, _ = pipe()

    d = ds()
    meta = {"M1": pd.DataFrame()}

    s.extract_metadata.return_value = meta
    s.save.return_value = {"ok": 1}

    assert (
        p.save_feature_store(
            d,
            out_dir="out",
            name="f",
        )
        == {"ok": 1}
    )

    s.extract_metadata.assert_called_once_with(d)

    s.save.assert_called_once_with(
        dataset=d,
        metadata=meta,
        out_dir="out",
        name="f",
        fmt="parquet",
    )

    with pytest.raises(
        ValueError,
        match="dataset is None",
    ):
        p.save_feature_store(
            None,
            out_dir="o",
            name="f",
        )

    with pytest.raises(
        TypeError,
        match="Expected dataset to be MTFDataset",
    ):
        p.save_feature_store(
            "bad",
            out_dir="o",
            name="f",
        )

    with pytest.raises(
        ValueError,
        match="does not match pipeline symbol",
    ):
        p.save_feature_store(
            ds("EURUSD"),
            out_dir="o",
            name="f",
        )


def test_export():
    p, *_ = pipe()

    with pytest.raises(
        RuntimeError,
        match="Dataset is empty. Run the pipeline first",
    ):
        p.export(
            "out",
            "f",
        )

    p._dataset = ds()
    p._features = ds()

    p.build_feature_store = MagicMock(
        return_value=ds()
    )
    p.save_feature_store = MagicMock(
        return_value={"ok": 1}
    )

    result = p.export(
        "out",
        "f",
        "csv",
    )

    assert result == {"ok": 1}

    p.build_feature_store.assert_called_once_with(
        raw_dataset=p._dataset,
        feature_dataset=p._features,
    )

    p.save_feature_store.assert_called_once_with(
        p.build_feature_store.return_value,
        out_dir="out",
        name="f",
        fmt="csv",
    )


def test_build_numpy_observation():
    p, _, _, b, g = pipe()

    dataset = ds()

    arr = np.asarray(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    b.build_numpy.return_value = arr

    assert p.build_numpy_observation(dataset) is arr

    b.build_numpy.assert_called_once()
    called_dataset, called_graph = b.build_numpy.call_args.args

    assert called_dataset is dataset
    assert called_graph is g


def test_rebuild_graph():
    p, *_ = pipe(
        specs=['sma(column="close",period=10)@M1']
    )

    new_specs = [
        'ema(column="close",period=20)@M1',
        'rsi(column="close",period=14)@M1',
    ]

    p.rebuild_graph(new_specs)

    assert p.feature_specs == new_specs
    assert p.feature_graph is not None
    assert p.graph is p.feature_graph

    old_specs = list(p.feature_specs)

    p.rebuild_graph()

    assert p.feature_specs == old_specs
    assert p.graph is p.feature_graph


def test_reload_config():
    p, *_ = pipe()

    with pytest.raises(
        ValueError,
        match="config is required",
    ):
        p.reload_config(None)

    with pytest.raises(
        TypeError,
        match="config must be dict",
    ):
        p.reload_config([])

    cfg = {
        "features": {
            "live_specs": [
                'ema(column="close",period=20)@M1'
            ]
        },
        "__base_tfs_dict": {
            SYMBOL: "M1"
        },
    }

    old_e = p.engine
    old_b = p.observation_builder

    p.reload_config(cfg)

    assert p.config is cfg
    assert p.feature_specs == [
        'ema(column="close",period=20)@M1'
    ]

    assert p.feature_graph is not None
    assert p.engine is not old_e
    assert p.observation_builder is not old_b


def test_info():
    p, *_ = pipe(
        specs=["a", "b"]
    )

    p.feature_graph.all_nodes = MagicMock(
        return_value=[
            1,
            2,
            3,
        ]
    )

    assert p.info() == {
        "symbol": SYMBOL,
        "base_tf": "M1",
        "spec_count": 2,
        "specs": [
            "a",
            "b",
        ],
        "timeframe_count": 3,
    }