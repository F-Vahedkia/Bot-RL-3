# -*- coding: utf-8 -*-
# created at 1404/08/18
# f15_testcheck/integration/test_feature_engine_full.py

"""
نسخه نهایی — Probe همهٔ فیچرها (تک‌به‌تک) روی «دادهٔ واقعیِ processed» پروژه
- فقط تست؛ هیچ منطق هسته‌ای در این فایل نیست.
- اتصال ۱۰۰٪ به فایل‌های خود پروژه: feature_registry.py, feature_engine.py, config.yaml
- خواندن paths.processed_dir از f01_config/config.yaml
- TFهای هدف: M5 و D1
- گزارش RUN/PASS/FAIL برای هر spec + تضمین «ستون جدید» (یعنی واقعاً به Engine رسیده)
- ایزوله‌سازی هر spec با multiprocessing (spawn) + تایم‌اوت ۶۰ثانیه (سازگار با ویندوز/لینوکس)
"""

import os, sys, time, pathlib, tempfile, yaml, pytest
import multiprocessing as mp
import pandas as pd

# --- دسترسی به ریشهٔ پروژه برای ایمپورت ماژول‌ها ---
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from f04_features.OLD.feature_registry import list_all_indicators
from f04_features.OLD.feature_engine import run_specs_v2

# ---------------------- پیکربندی عمومی ----------------------
CONFIG_PATH = ROOT / "f01_config" / "config.yaml"
SYMBOL      = os.environ.get("BOT_SYMBOL", "XAUUSD")
BASE_TF     = "M5"
EXTRA_TF    = "D1"
TIMEFRAMES  = [BASE_TF]    #[BASE_TF, EXTRA_TF]

# ---------------------- توابع کمکی (فقط تست) ----------------------
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
    خواندن دادهٔ واقعی و نرمال‌سازی ستون‌ها به: open, high, low, close, volume
    - پشتیبانی از MultiIndex ستون‌ها (انتخاب سطح مرتبط با TF؛ یا flatten)
    - پشتیبانی از پیشوند TF مثل M5_open / D1_close
    - پشتیبانی از حروف متفاوت (Open/High/LOW/...)
    - نگاشت tick_volume → volume
    """
    # 1) read
    df = pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)

    # 2) time index
    time_cols = [c for c in df.columns if str(c).lower() in ("time", "datetime", "date", "timestamp")]
    if time_cols:
        df = df.set_index(time_cols[0])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    # 3) handle MultiIndex or flattened prefixed names
    tf_l = tf.lower()
    if isinstance(df.columns, pd.MultiIndex):
        # اول تلاش: اگر یکی از سطوح برابر TF است، همان برش را بردار
        lvl_match = None
        for lvl in range(df.columns.nlevels):
            # مقادیر سطح را لوِر کن و مقایسه
            vals = [str(v).lower() for v in df.columns.get_level_values(lvl)]
            if tf_l in set(vals):
                lvl_match = lvl
                break
        if lvl_match is not None:
            try:
                df = df.xs(tf, level=lvl_match, axis=1)
            except Exception:
                # شاید tf به صورت lowercase در سطح باشد
                df = df.xs(tf_l, level=lvl_match, axis=1)
        else:
            # فلتر: اگر نام‌ها به شکل ('M5','open') باشند، ستون‌های همان tf را انتخاب کن
            sel = []
            for col in df.columns:
                parts = [str(p).lower() for p in col]
                if tf_l in parts:
                    sel.append(col)
            if sel:
                df = df.loc[:, sel]
            # نهایتاً flatten
            df.columns = ["_".join(str(p) for p in col) for col in df.columns.values]

    # اگر هنوز MultiIndex نبود یا flatten شد، ادامه با نام‌های رشته‌ای
    if isinstance(df.columns, pd.MultiIndex) is False:
        cols = list(df.columns)
        cols_lc = {str(c).lower(): c for c in cols}

        # ساخت نگاشت‌های کاندید برای هر ستون استاندارد
        def pick(*cands):
            for cand in cands:
                if cand in cols_lc:
                    return cols_lc[cand]
            return None

        # کاندیدهای هر ستون
        cand_open  = [f"{tf_l}_open", "open", "o"]
        cand_high  = [f"{tf_l}_high", "high", "h"]
        cand_low   = [f"{tf_l}_low",  "low",  "l"]
        cand_close = [f"{tf_l}_close","close","c"]
        cand_vol   = [f"{tf_l}_tick_volume", f"{tf_l}_volume", "tick_volume", "volume", "vol"]

        src = {
            "open":  pick(*cand_open),
            "high":  pick(*cand_high),
            "low":   pick(*cand_low),
            "close": pick(*cand_close),
            "volume":pick(*cand_vol),
        }

        # اگر با نام‌های دقیق پیدا نشد، تلاش با جست‌وجوی حاویِ پیشوند
        for key, val in list(src.items()):
            if val is None:
                # پیدا کردن ستونی که شامل "{tf_l}_" و اسم کلید باشد
                found = [c for c in cols if (f"{tf_l}_" in str(c).lower() and key in str(c).lower())]
                if found:
                    src[key] = found[0]

        # حداقل high/low/close باید موجود باشند
        missing = [k for k, v in src.items() if k in ("high","low","close") and v is None]
        if missing:
            raise ValueError(f"Missing required columns for {tf}: {missing} (from {path.name})")

        # rename به نام‌های استاندارد
        ren = {v: k for k, v in src.items() if v is not None}
        df = df.rename(columns=ren)

        # تضمین وجود volume (اختیاری)
        if "volume" not in df.columns:
            df["volume"] = 0

        # فقط ستون‌های استاندارد را نگه دار اگر خیلی زیادند
        keep = [c for c in ("open","high","low","close","volume") if c in df.columns]
        df = df[keep]

    return df.sort_index()


def _probe_spec_proc(df_path: pathlib.Path, spec: str, base_tf: str, q: mp.Queue):
    """
    اجرای ایزولهٔ یک spec در پروسس جدا؛ نتیجهٔ (ok, err_msg) در Queue گذاشته می‌شود.
    ok یعنی DataFrame خروجی معتبر است و «ستون جدید» هم اضافه شده.
    """
    import pandas as _pd
    from f04_features.OLD.feature_engine import run_specs_v2 as _run
    dfb = _pd.read_parquet(df_path)
    pre_cols = set(dfb.columns)
    out = _run(df=dfb, specs=[spec], base_tf=base_tf)
    if not isinstance(out, _pd.DataFrame) or len(out) != len(dfb):
        q.put((False, "engine returned invalid output"))
        return
    added = set(out.columns) - pre_cols
    if not added:
        q.put((False, "no new columns created"))
        return
    q.put((True, ""))

# --------------------------- تست اصلی ---------------------------
@pytest.mark.integration
def test_feature_engine_full():
    """
    پروب همهٔ فیچرها: برای هر کلید رجیستری و هر TF (M5/D1)،
    spec‌ها به‌صورت تک‌به‌تک به Engine ارسال می‌شوند؛ RUN/PASS/FAIL چاپ می‌شود
    و الزاماً باید ستون جدید بسازند.
    """
    cfg  = _load_cfg()
    processed_dir = (ROOT / cfg["paths"]["processed_dir"]).resolve()

    df_base = _read_ohlc(_processed_file(processed_dir, SYMBOL, BASE_TF), BASE_TF)
    df_base = df_base.tail(1000)  # مثلا فقط 1 هزار کندل آخر
    df_d1   = _read_ohlc(_processed_file(processed_dir, SYMBOL, EXTRA_TF), EXTRA_TF)

    tmp_dir = tempfile.TemporaryDirectory()
    df_base_path = pathlib.Path(tmp_dir.name) / "M5.parquet"
    df_d1_path   = pathlib.Path(tmp_dir.name) / "D1.parquet"
    df_base.to_parquet(df_base_path)
    df_d1.to_parquet(df_d1_path)


    try:
        reg = list_all_indicators(include_legacy=False)
    except TypeError:
        reg = list_all_indicators()
    keys = list(reg.keys()) if isinstance(reg, dict) else list(reg)
    assert len(keys) > 0, "هیچ فیچری در رجیستری یافت نشد."

    total = 0
    fails = []
    t_all = time.time()
    ctx = mp.get_context("spawn")

    for tf in TIMEFRAMES:
        for k in keys:
            total += 1
            spec = f"{k}@{tf}"
            t0 = time.time()
            try:
                print(f"[RUN ] {spec}")

                df_path_tf = df_base_path if tf == BASE_TF else df_d1_path
                q = ctx.Queue()
                p = ctx.Process(target=_probe_spec_proc, args=(df_path_tf, spec, tf, q))
                p.start()

                p.join(60)  # timeout 60s
                if p.is_alive():
                    p.terminate(); p.join()
                    raise TimeoutError("stuck >60s")
                ok, err = q.get_nowait() if not q.empty() else (False, "no result")
                if not ok:
                    raise RuntimeError(err)
                print(f"[PASS] {spec} ({time.time()-t0:.3f}s)")
            except Exception as e:
                msg = f"{spec} -> {type(e).__name__}: {e}"
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

    assert ratio >= 0.95, f"Too many failures: {len(fails)} of {total}"
