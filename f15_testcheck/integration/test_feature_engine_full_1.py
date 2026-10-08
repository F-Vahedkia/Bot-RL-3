# -*- coding: utf-8 -*-
# created at 1404/08/18
# ============================   NOT USED   ==============================
"""
f15_testcheck/integration/test_feature_engine_full_1.py
تست جامع سطح-Engine برای بررسی صحت لوله‌کشی feature_registry → feature_engine

این فایل فقط برای تست است و هیچ کد هسته‌ای در آن قرار ندارد.
هدف: اطمینان از اینکه تمام فیچرهای رجیستری با موفقیت تا Engine اجرا می‌شوند.
"""

import os, time, pathlib, yaml
import numpy as np
import pandas as pd
import pytest
import multiprocessing as mp

# دسترسی به ریشهٔ پروژه (…/ <root>)
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


from f04_features.OLD.feature_registry import list_all_indicators
from f04_features.OLD.feature_engine import run_specs_v2

# ---------------------- پیکربندی عمومی ----------------------
CONFIG_PATH = pathlib.Path("f01_config/config.yaml")
SYMBOL      = os.environ.get("BOT_SYMBOL", "XAUUSD")
BASE_TF     = "M5"
EXTRA_TF    = "D1"
TIMEFRAMES  = [BASE_TF, EXTRA_TF]

# ---------------------- توابع کمکی (فقط تست) ----------------------
def _load_cfg() -> dict:
    """لود config.yaml پروژه (مسیرهای داده و …)."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _processed_file(pdir: pathlib.Path, symbol: str, tf: str) -> pathlib.Path:
    """یافتن فایل پردازش‌شدهٔ واقعی (Parquet یا CSV) برای نماد/TF."""
    p1 = pdir / symbol / f"{tf}.parquet"
    p2 = pdir / symbol / f"{tf}.csv"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Processed not found: {symbol}@{tf} under {pdir}")

def _read_ohlc(path: pathlib.Path, tf: str) -> pd.DataFrame:
    """
    خواندن دادهٔ واقعی (CSV/Parquet) و نرمال‌سازی ستون‌ها.
    اگر ستون‌ها با پیشوند TF (مثل M5_open) ذخیره شده باشند، به open/high/low/close/volume نگاشت می‌شوند.
    """
    df = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)

    # ایندکس زمانی
    if "time" in df.columns:
        df = df.set_index("time")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    # حذف پیشوند TF و نگاشت tick_volume → volume
    tf_l = tf.lower()
    cols_lc = {c.lower(): c for c in df.columns}
    wanted = {
        f"{tf_l}_open": "open",   "open": "open",
        f"{tf_l}_high": "high",   "high": "high",
        f"{tf_l}_low": "low",     "low": "low",
        f"{tf_l}_close": "close", "close": "close",
        f"{tf_l}_tick_volume": "volume", "tick_volume": "volume", "volume": "volume",
    }
    ren = {}
    for k_lc, std in wanted.items():
        if k_lc in cols_lc:
            ren[cols_lc[k_lc]] = std
    if ren:
        df = df.rename(columns=ren)

    # اعتبارسنجی حداقلی
    for c in ["open", "high", "low", "close", "volume"]:
        assert c in df.columns, f"Missing required column: {c}"

    return df.sort_index()

def _probe_spec_proc(df_base, spec, base_tf, q):
    """
    اجرای ایزولهٔ یک spec در پروسس جدا؛ نتیجهٔ PASS/FAIL را در Queue می‌گذارد.
    """
    import pandas as _pd
    from f04_features.OLD.feature_engine import run_specs_v2 as _run
    out = _run(df=df_base.copy(), specs=[spec], base_tf=base_tf)
    q.put(isinstance(out, _pd.DataFrame) and len(out) == len(df_base))

# --------------------------- تست اصلی ---------------------------
@pytest.mark.integration
def test_feature_engine_full():
    """
    پروب همهٔ فیچرها: برای هر کلید رجیستری و هر TF (M5/D1)،
    spec‌ها به‌صورت تک‌به‌تک به Engine ارسال می‌شوند؛ RUN/PASS/FAIL چاپ می‌شود.
    """
    cfg  = _load_cfg()
    pdir = pathlib.Path(cfg["paths"]["processed_dir"])

    # دادهٔ پایه (بر مبنای BASE_TF). وجود EXTRA_TF نیز بررسی می‌شود.
    df_base = _read_ohlc(_processed_file(pdir, SYMBOL, BASE_TF), BASE_TF)
    _       = _read_ohlc(_processed_file(pdir, SYMBOL, EXTRA_TF), EXTRA_TF)

    # دریافت کلیدهای رجیستری (بدون حدس؛ سازگار با امضاهای مختلف)
    try:
        reg = list_all_indicators(include_legacy=False)
    except TypeError:
        reg = list_all_indicators()
    keys = list(reg.keys()) if isinstance(reg, dict) else list(reg)
    assert len(keys) > 0, "هیچ فیچری در رجیستری یافت نشد."

    total = 0
    fails = []
    t_all = time.time()


    for tf in TIMEFRAMES:
        for k in keys:
            total += 1
            spec = f"{k}@{tf}"
            t0 = time.time()
            try:

                print(f"[RUN ] {spec}")
                q = mp.Queue()
                p = mp.Process(target=_probe_spec_proc, args=(df_base, spec, BASE_TF, q))
                p.start()
                p.join(60)  # 60s timeout
                if p.is_alive():
                    p.terminate(); p.join()
                    raise TimeoutError("stuck >60s")
                ok = (not q.empty()) and q.get_nowait()
                if not ok:
                    raise RuntimeError("engine returned invalid output")
                print(f"[PASS] {spec} ({time.time()-t0:.3f}s)")

            except Exception as e:
                msg = f"{spec} -> {e}"
                print(f"[FAIL] {msg}")
                fails.append(msg)

    dur = time.time() - t_all
    pass_count = total - len(fails)
    ratio = pass_count / max(1, total)

    print("\n=== Probe Summary ===")
    print(f"Total specs: {total} | PASS: {pass_count} | FAIL: {len(fails)} | Pass ratio: {ratio:.2%} | Time: {dur:.1f}s")
    if fails:
        print("\nFailed list (first 100):")
        for m in fails[:100]:
            print(" -", m)

    # معیار حداقلی: حداقل 95% باید PASS باشند
    assert ratio >= 0.95, f"Too many failures: {len(fails)} of {total}"
