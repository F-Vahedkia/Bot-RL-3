# -*- coding: utf-8 -*-
"""
f15_testcheck/diagnostics/test_trace_adv_triplet.py
⚠️ تست دیباگی «مسیر کامل» فقط برای سه فیچر: rsi_zone, sr_overlap_score, round_levels
- هدف: لاگِ مرحله‌به‌مرحله از: Test → Engine → Registry/Adapter → Indicator و برگشت
- بدون دست‌کاری دائمی هسته: فقط monkeypatch در زمان اجرا
- پیام‌های runtime انگلیسی هستند؛ کامنت‌ها فارسی

روش اجرا:
python -m pytest -s f15_testcheck/diagnostics/test_trace_adv_triplet.py
"""

import os, sys, time, json, pathlib, tempfile
import yaml
import pandas as pd
import pytest

# ---------- مسیر پروژه ----------
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ایمپورت‌های هسته (فقط استفاده؛ تغییری در کد اصلی داده نمی‌شود)
from f04_features.OLD.feature_engine import run_specs_v2 as _engine_run
import f04_features.OLD.feature_registry as _reg
import f04_features.indicators.extras_trend as _xtrend   # برای rsi_zone
import f04_features.indicators.levels as _levels         # برای sr_overlap_score
import f04_features.indicators.utils as _utils           # برای round_levels

# ---------- تنظیمات ----------
CONFIG_PATH = ROOT / "f01_config" / "config.yaml"
SYMBOL      = os.environ.get("BOT_SYMBOL", "XAUUSD")
BASE_TF     = "M5"                         # فقط M5 برای جلوگیری از timeout
TIMEFRAMES  = [BASE_TF]
TAIL_N      = int(os.environ.get("BOT_TAIL", "15000"))   # محدودسازی سطرها برای جلوگیری از timeout

TARGETS = ["rsi_zone", "sr_overlap_score", "round_levels"]  # فقط همین سه فیچر

# ---------- توابع کمکی I/O ----------
def _make_spec(name: str, df: pd.DataFrame) -> str:
    """
    برای sr_overlap_score / round_levels، anchor/step را از داده می‌سازد
    تا آداپترهای per_bar بدون حدس ثابت کار کنند.
    """
    if name == "sr_overlap_score" or name == "round_levels":
        close = df["close"].astype("float32")
        price_min = float(close.min())
        price_max = float(close.max())
        if price_max <= price_min:
            # داده خراب است، تست را بی‌خیال می‌شویم
            pytest.skip("degenerate price series for advanced SR tests")

        # یک grid ساده ~۱۰۰ سطح بین min/max (بدون عدد ثابت از بیرون)
        levels_count = 100
        step = (price_max - price_min) / float(levels_count)
        anchor = price_min

        if name == "sr_overlap_score":
            return (
                f"sr_overlap_score("
                f"anchor={anchor}, step={step}, n=10, tol_pct=0.05"
                f")@{BASE_TF}"
            )

        # round_levels
        return (
            f"round_levels("
            f"anchor={anchor}, step={step}, half=False"
            f")@{BASE_TF}"
        )

    # برای rsi_zone همان spec ساده کافی است
    return f"{name}@{BASE_TF}"

def _load_cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _processed_file(pdir: pathlib.Path, symbol: str, tf: str) -> pathlib.Path:
    p1 = pdir / symbol / f"{tf}.parquet"
    p2 = pdir / symbol / f"{tf}.csv"
    if p1.exists(): return p1
    if p2.exists(): return p2
    raise FileNotFoundError(f"Processed not found: {symbol}@{tf} under {pdir}")

def _read_ohlc(path: pathlib.Path, tf: str) -> pd.DataFrame:
    """
    خواندن دادهٔ واقعی و نرمال‌سازی ستون‌ها به open/high/low/close/volume
    (نسخهٔ سبک برای تست؛ پوشش پیشوند TF و tick_volume)
    """
    df = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)
    # ایندکس زمانی
    time_cols = [c for c in df.columns if str(c).lower() in ("time","datetime","date","timestamp")]
    if time_cols:
        df = df.set_index(time_cols[0])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    tf_l = tf.lower()
    cols_lc = {str(c).lower(): c for c in df.columns}
    def pick(*cands):
        for cand in cands:
            if cand in cols_lc: return cols_lc[cand]
        return None
    src = {
        "open":  pick(f"{tf_l}_open","open","o"),
        "high":  pick(f"{tf_l}_high","high","h"),
        "low":   pick(f"{tf_l}_low","low","l"),
        "close": pick(f"{tf_l}_close","close","c"),
        "volume":pick(f"{tf_l}_tick_volume", f"{tf_l}_volume", "tick_volume","volume","vol"),
    }
    # جست‌وجوی شامل پیشوند
    for k,v in list(src.items()):
        if v is None:
            found = [c for c in df.columns if (f"{tf_l}_" in str(c).lower() and k in str(c).lower())]
            if found: src[k] = found[0]
    for req in ("high","low","close"):
        assert src[req] is not None, f"Missing required column for {tf}: {req}"
    df = df.rename(columns={v:k for k,v in src.items() if v is not None})
    if "volume" not in df.columns:
        df["volume"] = 0
    keep = [c for c in ("open","high","low","close","volume") if c in df.columns]
    return df[keep].sort_index()

# ---------- ابزار لاگ‌گذاری موقت ----------
def _wrap(name, fn):
    """یک رپر ساده برای لاگ ورودی/خروجی هر تابع اندیکاتور/آداپتر."""
    def _inner(*args, **kwargs):
        t0 = time.time()
        print(f"[TRACE] ENTER {name} args={len(args)} kwargs={list(kwargs.keys())}", flush=True)
        try:
            out = fn(*args, **kwargs)
            dur = time.time() - t0
            # خلاصهٔ خروجی
            summary = ""
            if isinstance(out, pd.DataFrame):
                summary = f"DataFrame shape={out.shape} cols={list(out.columns)[:6]}"
            elif isinstance(out, pd.Series):
                summary = f"Series len={len(out)} name={out.name}"
            elif isinstance(out, dict):
                k = list(out.keys())[:6]
                summary = f"dict keys={k}"
            else:
                summary = f"type={type(out).__name__}"
            print(f"[TRACE] EXIT  {name} ok=1 time={dur:.3f}s {summary}", flush=True)
            return out
        except Exception as e:
            dur = time.time() - t0
            print(f"[TRACE] EXIT  {name} ok=0 time={dur:.3f}s err={type(e).__name__}: {e}", flush=True)
            raise
    return _inner

def _install_tracers():
    """
    نصب رپرها روی مسیر سه فیچر هدف:
      - Registry ADV entry
      - Indicator functions
    """
    # rsi_zone (مستقیماً در _ADV به تابع ثبت شده)
    if hasattr(_xtrend, "rsi_zone"):
        _xtrend.rsi_zone = _wrap("indicator.rsi_zone", _xtrend.rsi_zone)
    if hasattr(_reg, "_ADV") and "rsi_zone" in _reg._ADV:
        _reg._ADV["rsi_zone"] = _wrap("adapter.rsi_zone", _reg._ADV["rsi_zone"])

    # sr_overlap_score
    if hasattr(_levels, "sr_overlap_score"):
        _levels.sr_overlap_score = _wrap("indicator.sr_overlap_score", _levels.sr_overlap_score)
    if hasattr(_reg, "_adv_sr_overlap_score"):
        _reg._adv_sr_overlap_score = _wrap("adapter._adv_sr_overlap_score", _reg._adv_sr_overlap_score)
    if hasattr(_reg, "_ADV") and "sr_overlap_score" in _reg._ADV:
        _reg._ADV["sr_overlap_score"] = _wrap("registry.sr_overlap_score", _reg._ADV["sr_overlap_score"])

    # round_levels
    if hasattr(_utils, "round_levels"):
        _utils.round_levels = _wrap("indicator.round_levels", _utils.round_levels)
    if hasattr(_reg, "_adv_round_levels"):
        _reg._adv_round_levels = _wrap("adapter._adv_round_levels", _reg._adv_round_levels)
    if hasattr(_reg, "_ADV") and "round_levels" in _reg._ADV:
        _reg._ADV["round_levels"] = _wrap("registry.round_levels", _reg._ADV["round_levels"])

# ---------- تست اصلی (فقط سه فیچر هدف) ----------
def test_trace_three_advanced_features():
    # ۱) لود داده و محدودسازی سطرها
    cfg  = _load_cfg()
    processed_dir = (ROOT / cfg["paths"]["processed_dir"]).resolve()
    df_m5 = _read_ohlc(_processed_file(processed_dir, SYMBOL, BASE_TF), BASE_TF)
    if TAIL_N > 0:
        df_m5 = df_m5.tail(TAIL_N)
    print(f"[INFO] DF@{BASE_TF} shape={df_m5.shape}", flush=True)

    # ۲) نصب لاگ‌رپرها
    _install_tracers()

    # ۳) اجرای تک‌به‌تک هر spec (فقط سه فیچر هدف)
    fails = []
    for name in TARGETS:
        spec = _make_spec(name, df_m5)
        t0 = time.time()
        try:
            print(f"[RUN ] {spec}", flush=True)
            pre = set(df_m5.columns)
            out = _engine_run(df=df_m5.copy(), specs=[spec], base_tf=BASE_TF)
            added = list(set(out.columns) - pre)
            # بررسی: ستون جدید و نرمال بودن (نه تماماً NaN در انتهای بازه)
            ok_added = len(added) > 0
            ok_nan   = True
            if ok_added:
                seg = out[added].tail(min(200, len(out)))
                ok_nan = not seg.isna().all().all()
            if not (ok_added and ok_nan):
                raise AssertionError(f"added={len(added)} ok_nan={ok_nan} sample={added[:6]}")
            print(f"[PASS] {spec} ({time.time()-t0:.3f}s) +{len(added)}", flush=True)
        except Exception as e:
            msg = f"{spec} -> {type(e).__name__}: {e}"
            print(f"[FAIL] {msg}", flush=True)
            fails.append(msg)

    # ۴) گزارش نهایی
    print("\n=== TRACE SUMMARY ===", flush=True)
    print(f"Total: {len(TARGETS)} | PASS: {len(TARGETS)-len(fails)} | FAIL: {len(fails)}", flush=True)
    for m in fails:
        print(" -", m, flush=True)

    # اگر حتی یکی Fail شد، تست را قرمز کن تا مسیر در لاگ مشخص باشد
    assert not fails, f"Unexpected failures: {len(fails)}"
