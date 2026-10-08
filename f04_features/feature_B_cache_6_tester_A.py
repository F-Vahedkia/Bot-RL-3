# f04_features/feature_B_cache_6_tester_A.py
# Run: pytest -v -s f04_features/feature_B_cache_6_tester_A.py

""" Date Reviewed:
    1405/05/24-15:02 ==> run result is OK for 19 tests
"""
# ===================================================================
# Imports
# ===================================================================
from __future__ import annotations
import pandas as pd
import pytest

from f03_data.mtf_dataset import MTFDataset
from f04_features.feature_B_cache_6 import (
    _stable_hash,
    _frame_signature,
    ExecutionContract,
    FeatureCache,
    build_cache_key,
    cached_compute,
)
from f02_utils.functions.parser import parse_spec, ParsedSpec

# ===================================================================
# _stable_hash
# ===================================================================
def test_stable_hash_equal():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert _stable_hash(a) == _stable_hash(b)

def test_stable_hash_not_equal():
    a = {"x": 1}
    b = {"x": 2}
    assert _stable_hash(a) != _stable_hash(b)

# ===================================================================
# ExecutionContract
# ===================================================================
def test_contract_signature_equal():
    c1 = ExecutionContract(
        engine_version="1",
        resolver_version="1",
        config_version="1",
        registry_version="1",
    )
    c2 = ExecutionContract(
        engine_version="1",
        resolver_version="1",
        config_version="1",
        registry_version="1",
    )
    assert c1.signature() == c2.signature()

def test_contract_signature_changed():
    c1 = ExecutionContract("1","1","1","1")
    c2 = ExecutionContract("2","1","1","1")
    assert c1.signature() != c2.signature()

# ===================================================================
# _df_signature
# ===================================================================
def test_df_signature():
    df = pd.DataFrame({
        "close": [1, 2, 3],
        "volume": [5, 6, 7],
    })
    sig = _frame_signature(df)
    assert sig["shape"] == (3, 2)
    assert sig["columns"] == ["close", "volume"]
    assert "dtypes" in sig

# ===================================================================
# FeatureCache
# ===================================================================
def test_cache_set_get():
    cache = FeatureCache(max_size=5)
    cache.set("a", 123)
    assert cache.get("a") == 123

def test_cache_missing():
    cache = FeatureCache()
    assert cache.get("missing") is None

def test_cache_clear():
    cache = FeatureCache()
    cache.set("x", 1)
    cache.set("y", 2)
    cache.clear()
    assert cache.get("x") is None
    assert cache.get("y") is None

def test_cache_lru_eviction():
    cache = FeatureCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    # a becomes most recently used
    assert cache.get("a") == 1
    cache.set("c", 3)
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None

def test_cache_overwrite_keeps_value():
    cache = FeatureCache()
    cache.set("k", 1)
    cache.set("k", 999)
    # طبق پیاده‌سازی فعلی مقدار جایگزین نمی‌شود
    assert cache.get("k") == 999

# ===================================================================
# build_cache_key
# ===================================================================
def test_build_cache_key_same():
    df = pd.DataFrame({"close": [1, 2, 3]})
    contract = ExecutionContract("1", "1", "1", "1")
    dataset = MTFDataset(
        symbol="XAUUSD",
        base_tf="M1",
    )
    dataset.add("M1", df)
    k1 = build_cache_key(
        dataset=dataset,
        spec="sma(10)@M1",
        mode="train",
        contract=contract,
        tf="M1",
    )
    k2 = build_cache_key(
        dataset=dataset,
        spec="sma(10)@M1",
        mode="train",
        contract=contract,
        tf="M1",
    )
    assert k1 == k2


def test_build_cache_key_different_tf():
    df = pd.DataFrame({"close": [1, 2, 3]})
    contract = ExecutionContract("1", "1", "1", "1")
    dataset = MTFDataset(
        symbol="XAUUSD",
        base_tf="M1",
    )
    dataset.add("M1", df)
    dataset.add("M5", df.copy())
    k1 = build_cache_key(
        dataset=dataset,
        spec="sma(10)@M1",
        mode="train",
        contract=contract,
        tf="M1",
    )
    k2 = build_cache_key(
        dataset=dataset,
        spec="sma(10)@M1",
        mode="train",
        contract=contract,
        tf="M5",
    )
    assert k1 != k2


def test_build_cache_key_extra_changes():
    df = pd.DataFrame({"close": [1, 2, 3]})
    contract = ExecutionContract("1", "1", "1", "1")
    dataset = MTFDataset(
        symbol="XAUUSD",
        base_tf="M1",
    )
    dataset.add("M1", df)
    k1 = build_cache_key(
        dataset=dataset,
        spec="sma",
        mode="train",
        contract=contract,
        extra={"a": 1},
    )
    k2 = build_cache_key(
        dataset=dataset,
        spec="sma",
        mode="train",
        contract=contract,
        extra={"a": 2},
    )
    assert k1 != k2


# ===================================================================
# cached_compute()
# ===================================================================
class DummyEngine:
    def __init__(self):
        self.calls = 0

    def execute(self, dataset, specs, mode):
        self.calls += 1
        out = dataset.copy()
        for tf in out.frames:
            out.frames[tf]["dummy_feature"] = self.calls
        return out


def build_dataset():
    df = pd.DataFrame({
        "close": [1.0, 2.0, 3.0],
        "open": [1.0, 2.0, 3.0],
        "high": [1.0, 2.0, 3.0],
        "low": [1.0, 2.0, 3.0],
        "volume": [10, 11, 12],
    })
    ds = MTFDataset(
        symbol="XAUUSD",
        base_tf="M1",
    )
    ds.add("M1", df)
    return ds


def build_contract():
    return ExecutionContract("1", "1", "1", "1")


def test_cached_compute_first_call():
    cache = FeatureCache()
    engine = DummyEngine()
    dataset = build_dataset()
    result = cached_compute(
        cache=cache,
        dataset=dataset,
        ps=parse_spec("sma(close, 10)@M1"),
        mode="train",
        contract=build_contract(),
        compute_fn=lambda: engine.execute(dataset, ["dummy"], "train"),
    )
    assert engine.calls == 1
    assert "dummy_feature" in result["M1"].columns


def test_cached_compute_second_call_hits_cache():
    cache = FeatureCache()
    engine = DummyEngine()
    dataset = build_dataset()
    result1 = cached_compute(
        cache=cache,
        dataset=dataset,
        ps=parse_spec("sma(close, 10)@M1"),
        mode="train",
        contract=build_contract(),
        compute_fn=lambda: engine.execute(dataset, ["dummy"], "train"),
    )
    result2 = cached_compute(
        cache=cache,
        dataset=dataset,
        ps=parse_spec("sma(close, 10)@M1"),
        mode="train",
        contract=build_contract(),
        compute_fn=lambda: engine.execute(dataset, ["dummy"], "train"),
    )
    assert engine.calls == 1
    assert result1["M1"].equals(result2["M1"])


def test_cached_compute_spec_change():
    cache = FeatureCache()
    engine = DummyEngine()
    dataset = build_dataset()
    cached_compute(
        cache=cache,
        dataset=dataset,
        ps=parse_spec("sma(close, 10)@M1"),
        mode="train",
        contract=build_contract(),
        compute_fn=lambda: engine.execute(dataset, ["dummy"], "train"),
    )
    cached_compute(
        cache=cache,
        dataset=dataset,
        ps=parse_spec("ema(close, 10)@M1"),
        mode="train",
        contract=build_contract(),
        compute_fn=lambda: engine.execute(dataset, ["dummy"], "train"),
    )
    assert engine.calls == 2


def test_cached_compute_mode_change():
    cache = FeatureCache()
    engine = DummyEngine()
    dataset = build_dataset()
    cached_compute(
        cache=cache,
        dataset=dataset,
        ps=parse_spec("sma(close, 10)@M1"),
        mode="train",
        contract=build_contract(),
        compute_fn=lambda: engine.execute(dataset, ["dummy"], "train"),
    )
    cached_compute(
        cache=cache,
        dataset=dataset,
        ps=parse_spec("sma(close, 10)@M1"),
        mode="live",
        contract=build_contract(),
        compute_fn=lambda: engine.execute(dataset, ["dummy"], "train"),
    )
    assert engine.calls == 2


def test_cached_compute_contract_change():
    cache = FeatureCache()
    engine = DummyEngine()
    dataset = build_dataset()
    contract1 = ExecutionContract("1", "1", "1", "1")
    contract2 = ExecutionContract("2", "1", "1", "1")
    cached_compute(
        cache=cache,
        dataset=dataset,
        ps=parse_spec("sma(close, 10)@M1"),
        mode="train",
        contract=contract1,
        compute_fn=lambda: engine.execute(dataset, ["dummy"], "train"),
    )
    cached_compute(
        cache=cache,
        dataset=dataset,
        ps=parse_spec("sma(close, 10)@M1"),
        mode="train",
        contract=contract2,
        compute_fn=lambda: engine.execute(dataset, ["dummy"], "train"),
    )
    assert engine.calls == 2


def test_cached_compute_dataset_change():
    cache = FeatureCache()
    engine = DummyEngine()
    ds1 = build_dataset()
    ds2 = build_dataset()
    ds2["M1"].loc[3] = [4.0, 4.0, 4.0, 4.0, 13]
    cached_compute(
        cache=cache,
        dataset=ds1,
        ps=parse_spec("sma(close, 10)@M1"),
        mode="train",
        contract=build_contract(),
        compute_fn=lambda: engine.execute(ds1, ["dummy"], "train"),
    )
    cached_compute(
        cache=cache,
        dataset=ds2,
        ps=parse_spec("sma(close, 10)@M1"),
        mode="train",
        contract=build_contract(),
        compute_fn=lambda: engine.execute(ds2, ["dummy"], "train"),
    )
    assert engine.calls == 2

# =================================================================== END
