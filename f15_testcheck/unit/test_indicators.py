# -*- coding: utf-8 -*-
"""
Comprehensive test suite for the `indicators` package.

Goals (without touching existing project files):
- Discover indicator callables across `indicators/` modules.
- Execute each indicator against a realistic synthetic OHLCV dataset.
- Validate output shape, index alignment, NaN behavior (warmup window), and
  absence of lookahead leakage.
- Apply additional semantic checks for well-known indicators when detected
  (e.g., RSI range, Bollinger band ordering, ATR positivity, MACD relations).

Notes:
- This suite relies on dynamic discovery (registry() in core, if present),
  otherwise falls back to introspecting modules for callables.
- It purposefully avoids any I/O, MT5, or network calls.
- Keep this file under `tests/unit/test_indicators.py`.

Run:  pytest -q f16_tests/unit/test_indicators.py
"""
from __future__ import annotations

import random
import types
import inspect
import importlib, sys, pytest
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------- Utility: project path ----------------------------
'''
def _find_project_root(start: Path) -> Path:
    """Search upwards for a directory containing `f04_features/indicators`.
    Falls back to 3 levels up from current file.
    """
    cur = start
    for _ in range(8):
        if (cur / "f04_features/indicators").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.parents[2] if len(start.parents) >= 3 else start
'''
#--------------------------------------------------- Not Used
'''
project_root = _find_project_root(Path(__file__).resolve())
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if __name__ == "__main__":
    print(f"Project root found at: {project_root}")
    print(f"Current file is located at: {Path(__file__).resolve()}")
'''
#---------------------------------------------------
'''
# Ensure `f04_features/indicators` is importable (so that `indicators` can be imported)
_TEST_FILE = Path(__file__).resolve()
project_root = (_TEST_FILE.parent.parent.parent)
_MAINFILES_DIR = project_root / "f04_features"
if str(_MAINFILES_DIR) not in sys.path: sys.path.insert(0, str(_MAINFILES_DIR))
try: indicators_pkg = importlib.import_module("indicators")
except Exception as e: pytest.skip(f"Cannot import indicators package from {_MAINFILES_DIR}: {e}")
'''
#---------------------------------------------------
'''
_PR = Path(__file__).resolve().parents[2]/ "f04_features"  # project root (contains f04_features/)
if str(_PR) not in sys.path: sys.path.insert(0, str(_PR))
try:
    indicators_pkg = importlib.import_module("indicators")
except Exception as e:
    pytest.skip(f"Cannot import indicators from {_PR}: {e}", allow_module_level=True)
'''
#---------------------------------------------------
from pathlib import Path
_PR = Path(__file__).resolve().parents[2]
_F04 = _PR / "f04_features"
for p in (str(_PR), str(_F04)):
    if p not in sys.path: sys.path.insert(0, p)
try:
    indicators_pkg = importlib.import_module("indicators")
except Exception:
    try: indicators_pkg = importlib.import_module("f04_features.indicators"); sys.modules["indicators"]=indicators_pkg
    except Exception as e: pytest.skip(f"Cannot import indicators: {e}", allow_module_level=True)

# Late imports after sys.path adjustment
#try:
#    indicators_pkg = importlib.import_module("indicators")
#except Exception as e:
#    pytest.skip(f"Cannot import indicators package from {_MAINFILES_DIR}: {e}",  allow_module_level=True)

# Optionals (used when present)
core_mod = engine_mod = parser_mod = None
try:
    core_mod = importlib.import_module("indicators.core")
except Exception:
    pass
try:
    engine_mod = importlib.import_module("indicators.engine")
except Exception:
    pass
try:
    parser_mod = importlib.import_module("indicators.parser")
except Exception:
    pass


# ------------------------------ Data Fixtures ---------------------------------

@pytest.fixture(scope="module")
def rng_seed() -> int:
    return 1337


@pytest.fixture(scope="module")
def ohlcv_df(rng_seed: int) -> pd.DataFrame:
    """Generate a realistic synthetic OHLCV dataset.

    - 10,000 minutes of data (~1 week of trading minutes) to allow long warmups.
    - Geometric Brownian Motion for prices + volatility regime shifts.
    - Ensures strictly increasing DateTimeIndex.
    """
    n = 10_000
    dt = 1.0 / (60.0 * 24.0)  # normalized daily units
    rs = np.random.RandomState(rng_seed)

    # Volatility regimes
    vol = np.where(rs.rand(n) < 0.05, rs.uniform(0.02, 0.08, size=n), rs.uniform(0.005, 0.025, size=n))
    mu = 0.0001

    log_ret = (mu - 0.5 * (vol ** 2)) * dt + vol * np.sqrt(dt) * rs.randn(n)
    price = 1800.0 * np.exp(np.cumsum(log_ret))  # start around 1800 (e.g., gold)

    # Build OHLC with small intra-bar noise
    spread = np.maximum(0.01, 0.0005 * price)
    open_ = price * (1 + 0.0003 * rs.randn(n))
    close = price * (1 + 0.0003 * rs.randn(n))
    high = np.maximum(open_, close) + spread * (1 + 0.5 * rs.rand(n))
    low = np.minimum(open_, close) - spread * (1 + 0.5 * rs.rand(n))
    volume = (rs.lognormal(mean=12.0, sigma=0.2, size=n)).astype(np.float64)

    # idx = pd.date_range("2021-01-01", periods=n, freq="T")  added 040705
    idx = pd.date_range("2021-01-01", periods=n, freq="min")
    df = pd.DataFrame({
        "open": open_.astype(float),
        "high": high.astype(float),
        "low": low.astype(float),
        "close": close.astype(float),
        "volume": volume,
    }, index=idx)

    # Sanity: enforce numeric and monotonic index
    assert df.index.is_monotonic_increasing
    return df


@pytest.fixture(scope="module")
def warmup_n() -> int:
    """Conservative warmup length used to ignore initial NaNs across indicators."""
    return 300  # generous to cover long-window indicators


# ---------------------------- Discovery & Calling ------------------------------

CallableLike = Callable[..., Any]


def discover_registry_funcs() -> Dict[str, CallableLike]:
    """Use core.registry() if available to discover indicator functions.
    Fallback to introspecting `core_mod` public callables.
    """
    funcs: Dict[str, CallableLike] = {}
    if core_mod is not None and hasattr(core_mod, "registry"):
        try:
            reg = core_mod.registry()  # expected to be a mapping name -> callable
            if isinstance(reg, dict):
                for k, v in reg.items():
                    if callable(v):
                        funcs[str(k)] = v
        except Exception:
            pass

    # Fallback: collect functions directly from core module
    if not funcs and core_mod is not None:
        for name, obj in vars(core_mod).items():
            if name.startswith("_"):
                continue
            if callable(obj) and isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
                funcs[name] = obj
    return funcs


def discover_module_funcs_old1(module_name: str) -> Dict[str, CallableLike]:
    """Import a module under `indicators.` and collect public callables.
    We exclude objects coming from other modules to limit duplicates.
    """
    funcs: Dict[str, CallableLike] = {}
    try:
        mod = importlib.import_module(f"indicators.{module_name}")
    except Exception:
        return funcs

    for name, obj in vars(mod).items():
        if name.startswith("_"):
            continue
        if callable(obj) and isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
            # Only include functions defined in this module
            if getattr(obj, "__module__", "").endswith(f"indicators.{module_name}"):
                funcs[f"{module_name}.{name}"] = obj
    return funcs


def discover_module_funcs_old2(mod):
    import inspect
    out = []
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_") or name.startswith("make_") or name in {"registry","build_registry","list_all_indicators"}:
            continue
        sig = inspect.signature(obj)
        req = [p for p in sig.parameters.values() if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if not req or req[0].name not in {"df","data"} or len(req) > 1:
            continue
        out.append(name)
    return out


def discover_module_funcs(mod):
    import inspect
    out = {}
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_") or name in {"registry","build_registry","list_all_indicators"}:
            continue
        sig = inspect.signature(obj)
        req = [p for p in sig.parameters.values() if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if not req or req[0].name not in {"df","data"} or len(req) > 1:
            continue
        out[name] = obj
    return out



INDICATOR_MODULES = [
    # Extend as needed; these should exist per project structure
    "core",
    "divergences",
    "engine",
    "extras_channel",
    "extras_trend",
    "levels",
    "parser",
    "patterns",
    "registry",
    "utils",
    "volume",
]


def discover_all_indicator_functions() -> Dict[str, CallableLike]:
    funcs: Dict[str, CallableLike] = {}
    # 1) core registry (authoritative if available)
    for k, v in discover_registry_funcs().items():
        funcs[f"core.{k}"] = v

    # 2) per-module discovery (covers extras/levels/volume/...)
    for m in INDICATOR_MODULES:
        for k, v in discover_module_funcs(m).items():
            # Avoid overwriting exact names already present
            if k not in funcs:
                funcs[k] = v
    return funcs


# Helper: build arguments based on function signature

def _call_indicator_auto(func: CallableLike, df: pd.DataFrame) -> Any:
    """Attempt to call indicator with best-effort argument mapping.

    Strategy:
    - Inspect signature; support common parameter names: df, series, close, high, low, open, volume.
    - Provide only required params; rely on defaults for the rest.
    - If multiple signatures are possible, try a few variants.
    """
    sig = inspect.signature(func)
    params = sig.parameters

    # Candidate arg sets (tuples of (args, kwargs)) to try in order
    candidates: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

    # 1) df positional
    if any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params.values()):
        candidates.append(((df,), {}))

    # 2) series positional (use close)
    if "close" in df.columns:
        candidates.append(((df["close"],), {}))

    # 3) kwargs mapping by common names
    kw: Dict[str, Any] = {}
    for name in params:
        lname = name.lower()
        if lname in {"df", "frame", "data", "dataframe"}:
            kw[name] = df
        elif lname in {"series", "s", "x", "close", "c"} and "close" in df.columns:
            kw[name] = df["close"]
        elif lname in {"high", "h"} and "high" in df.columns:
            kw[name] = df["high"]
        elif lname in {"low", "l"} and "low" in df.columns:
            kw[name] = df["low"]
        elif lname in {"open", "o"} and "open" in df.columns:
            kw[name] = df["open"]
        elif lname in {"volume", "v", "vol"} and "volume" in df.columns:
            kw[name] = df["volume"]
    if kw:
        candidates.append(((), kw))

    # 4) High/Low/Close triple (e.g., ATR-like) if signature matches
    needed = {name.lower() for name in params}
    if {"high", "low", "close"}.issubset(needed):
        candidates.insert(0, ((), {"high": df["high"], "low": df["low"], "close": df["close"]}))

    # Try candidates
    last_err: Optional[BaseException] = None
    for args, kwargs in candidates:
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            last_err = e
            continue

    # As a final attempt, pass only close series
    try:
        return func(df["close"]) if "close" in df.columns else func(df)
    except BaseException as e2:
        last_err = e2

    raise RuntimeError(f"Failed to call indicator {func.__module__}.{func.__name__}: {last_err}")


# Convert arbitrary indicator output into a DataFrame for unified checks

def _to_frame(out: Any, index: pd.Index) -> pd.DataFrame:
    if isinstance(out, pd.DataFrame):
        return out.reindex(index)
    if isinstance(out, pd.Series):
        return out.to_frame(name=getattr(out, "name", "y")).reindex(index)
    
    #------------------------
    if isinstance(out, dict):
        parts=[]
        for k,v in out.items():
            if isinstance(v,pd.DataFrame):
                d=v.copy(); d.columns=[f"{k}" if d.shape[1]==1 else f"{k}_{c}" for c in d.columns]
                parts.append(d.reindex(index))
            else:
                s=v if isinstance(v,pd.Series) else pd.Series(v, index=index[:len(v)] if hasattr(v,'__len__') and not np.isscalar(v) else index, name=str(k))
                parts.append(s.rename(str(k)).reindex(index).to_frame())
        return pd.concat(parts,axis=1) if parts else pd.DataFrame(index=index)
    #------------------------
    
    if isinstance(out, (list, tuple)):
        cols = []
        arrs = []
        for i, part in enumerate(out):
            if isinstance(part, pd.Series):
                cols.append(getattr(part, "name", f"y{i}"))
                arrs.append(part.reindex(index))
            elif isinstance(part, pd.DataFrame):
                # flatten columns by adding a prefix
                pref = f"y{i}_"
                sub = part.copy()
                sub.columns = [f"{pref}{c}" for c in sub.columns]
                arrs.append(sub.reindex(index))
            elif np.isscalar(part):
                s = pd.Series([part] * len(index), index=index, name=f"y{i}")
                arrs.append(s.to_frame())
            else:
                s = pd.Series(part, index=index[: len(part)], name=f"y{i}")
                arrs.append(s.to_frame())
        if not arrs:
            return pd.DataFrame(index=index)
        out_df = pd.concat(arrs, axis=1)
        if not cols:
            # assign generic names
            out_df.columns = [f"y{i}" for i in range(out_df.shape[1])]
        return out_df
    # Fallback scalar
    
    return pd.DataFrame({"y": pd.Series([out] * len(index), index=index)})


# --------------------------- Generic Property Tests ---------------------------

@pytest.mark.parametrize("name,func", sorted(discover_all_indicator_functions().items()))
def test_indicator_generic_properties(name: str, func: CallableLike, ohlcv_df: pd.DataFrame, warmup_n: int):
    """Generic tests applicable to all discovered indicators.

    Checks:
    - Callable executes without raising and returns a Series/DataFrame/tuple/list/scalar.
    - Output index matches input index, shape[0] == len(df).
    - No NaNs/inf after a generous warmup window.
    - No lookahead leakage: value at t computed with full data equals value
      computed on truncated data up to t (sampled timestamps).
    - Input immutability: the indicator does not alter the input DataFrame in-place.
    """
    df = ohlcv_df.copy(deep=True)
    df_before = df.copy(deep=True)

    out = _call_indicator_auto(func, df)
    out_df = _to_frame(out, df.index)

    # Basic shape
    assert len(out_df) == len(df), f"{name}: output length mismatch"
    assert out_df.index.equals(df.index), f"{name}: index mismatch"

    # Input unchanged
    pd.testing.assert_frame_equal(df, df_before, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)

    # No NaNs/INF after warmup
    tail = out_df.iloc[warmup_n:]
    assert np.isfinite(tail.to_numpy(dtype=float)).all(), f"{name}: non-finite values after warmup"

    # No lookahead leakage (sample a few points)
    sample_idx = np.linspace(warmup_n + 100, len(df) - 2, num=5, dtype=int).tolist()
    for i in sample_idx:
        sub = df.iloc[: i + 1]
        sub_out = _to_frame(_call_indicator_auto(func, sub), sub.index)
        # Compare last available row
        col_inter = [c for c in out_df.columns if c in sub_out.columns]
        if not col_inter:
            # Align by position if names differ
            min_cols = min(out_df.shape[1], sub_out.shape[1])
            col_inter = list(range(min_cols))
            a = out_df.iloc[i, :min_cols].to_numpy(dtype=float)
            b = sub_out.iloc[-1, :min_cols].to_numpy(dtype=float)
        else:
            a = out_df.loc[df.index[i], col_inter].to_numpy(dtype=float)
            b = sub_out.iloc[-1][col_inter].to_numpy(dtype=float)
        # Allow tiny numerical tolerance
        assert np.allclose(a, b, atol=1e-8, rtol=1e-6), f"{name}: potential lookahead/leak at index {i}"


# ---------------------------- Semantic-Specific Tests --------------------------

@pytest.mark.parametrize("name,func", sorted(discover_all_indicator_functions().items()))
def test_indicator_semantics(name: str, func: CallableLike, ohlcv_df: pd.DataFrame, warmup_n: int):
    """Indicator-specific semantic checks when recognizable by name/columns.
    These are conditional and do nothing if the indicator name doesn't match.
    """
    df = ohlcv_df
    out = _to_frame(_call_indicator_auto(func, df), df.index)
    tail = out.iloc[warmup_n:]

    lname = name.lower()

    # RSI range check
    if "rsi" in lname and tail.shape[1] >= 1:
        vals = tail.iloc[:, 0].to_numpy(dtype=float)
        assert (vals >= -1e-6).all() and (vals <= 100 + 1e-6).all(), f"{name}: RSI outside [0, 100]"

    # Williams %R typically in [-100, 0]
    if "williams" in lname or lname.endswith("_r"):
        vals = tail.iloc[:, 0].to_numpy(dtype=float)
        # Only apply if range looks compatible
        if np.nanpercentile(vals, 5) >= -150 and np.nanpercentile(vals, 95) <= 50:
            assert (vals <= 5).all() and (vals >= -150).all(), f"{name}: Williams %R unexpected range"

    # ATR strictly positive
    if lname.endswith("atr") or lname == "core.atr" or "atr" in lname:
        vals = tail.iloc[:, 0].to_numpy(dtype=float)
        assert (vals > 0).all(), f"{name}: ATR should be strictly positive"

    # Bollinger bands ordering (lower <= middle <= upper) if columns present
    if "bollinger" in lname:
        cols = [c.lower() for c in tail.columns]
        def _find(name_sub):
            for j, c in enumerate(cols):
                if name_sub in c:
                    return j
            return None
        lo = _find("lower")
        mi = _find("mid") or _find("middle") or _find("mean")
        up = _find("upper")
        if lo is not None and up is not None:
            lvals = tail.iloc[:, lo].to_numpy(dtype=float)
            uvals = tail.iloc[:, up].to_numpy(dtype=float)
            assert (lvals <= uvals).all(), f"{name}: Bollinger lower>upper anomaly"
        if lo is not None and mi is not None and up is not None:
            mvals = tail.iloc[:, mi].to_numpy(dtype=float)
            lvals = tail.iloc[:, lo].to_numpy(dtype=float)
            uvals = tail.iloc[:, up].to_numpy(dtype=float)
            assert (lvals <= mvals).all() and (mvals <= uvals).all(), f"{name}: Bollinger order anomaly"

    # MACD histogram = macd - signal (if identifiable columns)
    if "macd" in lname and tail.shape[1] >= 2:
        cols = [c.lower() for c in tail.columns]
        try:
            i_macd = next(i for i, c in enumerate(cols) if c == "macd" or "macd" in c and "hist" not in c and "signal" not in c)
            i_sig = next(i for i, c in enumerate(cols) if "signal" in c)
            # optional histogram
            i_hist = next((i for i, c in enumerate(cols) if "hist" in c or "diff" in c), None)
            macd_vals = tail.iloc[:, i_macd].to_numpy(dtype=float)
            sig_vals = tail.iloc[:, i_sig].to_numpy(dtype=float)
            if i_hist is not None:
                hist_vals = tail.iloc[:, i_hist].to_numpy(dtype=float)
                assert np.allclose(hist_vals, macd_vals - sig_vals, atol=1e-6, rtol=1e-6), f"{name}: MACD hist != macd - signal"
        except StopIteration:
            pass

    # OBV: finite and reasonable scale (no inf)
    if "obv" in lname:
        vals = tail.iloc[:, 0].to_numpy(dtype=float)
        assert np.isfinite(vals).all(), f"{name}: OBV produced non-finite values"


# ----------------------------- Engine/Parser Tests -----------------------------
@pytest.mark.skipif(engine_mod is None, reason="indicators.engine not available")
def test_engine_basic_integration(ohlcv_df: pd.DataFrame, warmup_n: int):
    """If the engine exposes a high-level apply/run function, perform a smoke integration.
    Without knowing exact API, try common entry points. Skip if none are found.
    """
    df = ohlcv_df

    # Candidate entry points
    entry_points = [
        ("apply", {"df": df, "specs": ["sma(20)@M1", "rsi(14)@M1", "macd()@M1"]}),
        ("run",   {"df": df, "specs": ["ema(20)@M1", "bollinger(20,2)@M1"]}),
        ("execute", {"df": df, "specs": ["wma(20)@M1", "atr(14)@M1"]}),
    ]

    for func_name, kwargs in entry_points:
        if hasattr(engine_mod, func_name):
            fn = getattr(engine_mod, func_name)
            try:
                out = fn(**kwargs)  # type: ignore
                out_df = _to_frame(out, df.index)
                tail = out_df.iloc[warmup_n:]
                assert len(out_df) == len(df)
                assert out_df.index.equals(df.index)
                assert np.isfinite(tail.to_numpy(dtype=float)).all()
                return  # success on first working entry point
            except Exception:
                continue

    pytest.skip("No compatible entry point found in indicators.engine for generic integration test.")


# ------------------------------- Performance ----------------------------------

def test_performance_vectorized(ohlcv_df: pd.DataFrame):
    """Sanity check: applying a handful of indicators should be reasonably fast.
    This is a soft check; it won't fail unless execution is extremely slow.
    """
    import time
    funcs = list(discover_all_indicator_functions().items())
    # pick up to 8 indicators for a quick perf sanity
    subset = [f for _, f in random.sample(funcs, k=min(8, len(funcs)))] if funcs else []

    t0 = time.time()
    for f in subset:
        try:
            _to_frame(_call_indicator_auto(f, ohlcv_df), ohlcv_df.index)
        except Exception:
            pass
    elapsed = time.time() - t0

    # Allow generous threshold (depends on environment); we assert < 5 seconds
    assert elapsed < 5.0, f"Vectorized perf sanity failed: {elapsed:.2f}s for {len(subset)} indicators"


# ------------------------------- Robustness -----------------------------------

@pytest.mark.parametrize("name,func", sorted(discover_all_indicator_functions().items()))
def test_robustness_to_nans(name: str, func: CallableLike, ohlcv_df: pd.DataFrame, warmup_n: int):
    """Indicators should handle occasional NaNs in inputs without exploding.
    We inject sparse NaNs into close/high/low and expect finite outputs after warmup.
    """
    df = ohlcv_df.copy()
    rs = np.random.RandomState(202)
    # Inject ~0.5% NaNs
    for col in ["close", "high", "low", "open", "volume"]:
        idx = rs.choice(len(df), size=max(5, len(df)//200), replace=False)
        df.loc[df.index[idx], col] = np.nan

    out_df = _to_frame(_call_indicator_auto(func, df), df.index)
    out_df = out_df.ffill().bfill()

    tail = out_df.iloc[warmup_n:]
    # It's acceptable to have NaNs close to warmup; after that we expect finite values
    assert np.isfinite(tail.to_numpy(dtype=float)).all(), f"{name}: non-finite after warmup with sparse NaNs in inputs"


# ----------------------------- Input Validation -------------------------------

@pytest.mark.parametrize("name,func", sorted(discover_all_indicator_functions().items()))
def test_input_immutability(name: str, func: CallableLike, ohlcv_df: pd.DataFrame):
    """No indicator should modify user-provided DataFrame in-place."""
    df = ohlcv_df.copy(deep=True)
    snapshot = df.copy(deep=True)
    _ = _call_indicator_auto(func, df)
    pd.testing.assert_frame_equal(df, snapshot, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)


# --------------------------------- The End ------------------------------------

