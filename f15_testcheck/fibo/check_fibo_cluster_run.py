# -*- coding: utf-8 -*-
# f15_scripts/check_fibo_cluster_run.py
# Status in (Bor-RT-2): Completed
r"""
اسکریپت اسموک‌ران واقعی خوشه‌های فیبوناچی — Bot-RL-3

هدف:
    - خواندن دادهٔ واقعی OHLCV برای نماد/تایم‌فریم‌های درخواستی (از فولدر داده‌های پروژه).
    - تولید سطوح فیبوناچی برای هر تایم‌فریم با استفاده از تابع fibo_levels (levels.py).
    - اجرای fib_cluster_cfg (کانفیگ-محور و در صورت فعال بودن، با tol_pct انطباقی مبتنی بر ADR).
    - گزارش Top-N خوشه‌ها با نمایش اطلاعات کلیدی.

نکات:
    - از ConfigLoader برای خواندن config.yaml استفاده می‌کند؛ اما پارامترهای CLI ارجحیت دارند.
    - مسیرهای پیش‌فرض داده‌ها بر اساس ساختار رایج پروژه انتخاب شده‌اند:
        data/processed/<SYMBOL>/<TF>.parquet  ← اولویت بالاتر
        data/raw/<SYMBOL>/<TF>.parquet
        data/raw/<SYMBOL>/<TF>.csv
      در صورت تمایل می‌توانید با --data-root و --data-layout شخصی‌سازی کنید.
    - برای tol_pct انطباقی، در صورت فعال بودن adaptive_tol در config:
        ref_price از آخرین close تایم‌فریم پایه برداشته می‌شود،
        adr_value از compute_adr(base_df) استخراج می‌شود.
    - این اسکریپت صرفاً یک اسموک‌ران است؛ برای بک‌تست و اجرای زنده، مسیرهای اختصاصی خود پروژه را دنبال کنید.

اجرا از ریشه:
    python .\f15_scripts\check_fibo_cluster_run.py --symbols XAUUSD --timeframes M15 H1 H4 D1 --base-tf H1 --top-n 12
اجرا از مسیر متفاوت:
    python .\f15_scripts\check_fibo_cluster_run.py --symbols XAUUSD --timeframes M15 H1 H4 D1 --data-root . --data-layout processed-first
اجرا روی چند نماد:
    python .\f15_scripts\check_fibo_cluster_run.py --symbols XAUUSD EURUSD GBPUSD --timeframes M15 H1 H4 D1

"""

import argparse
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np

# فارسی: اضافه کردن روت پروژه به sys.path برای اطمینان از ایمپورت پکیج‌ها
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# لودر کانفیگ پروژه
from f02_utils.config_loader import ConfigLoader
cfg = ConfigLoader().get_all()
import f04_features.indicators.levels as lv
from f02_utils.config_operations import _deep_get

# -----------------------------------------------------------------------------
# تنظیم لاگینگ سبک برای این اسکریپت (پیام‌ها باید انگلیسی باشند)
# -----------------------------------------------------------------------------
logger = logging.getLogger("check_fibo_cluster_run")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[CLUSTER] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# داده‌ساخت‌ها و کمکی‌ها
# -----------------------------------------------------------------------------

@dataclass
class RunParams:
    """پارامترهای اجرای اسموک‌ران (Resolved از CLI/Config)"""
    symbols: List[str]
    timeframes: List[str]
    base_tf: str
    k_fractal: int
    top_n: int
    data_root: Path
    data_layout: str  # "processed-first" | "raw-first"
    adr_window: int
    tz: str


# تبدیل wide→long برای چند لگ آخر از هر نسبت
def _levels_wide_to_long_multi(levels_df: pd.DataFrame, ratios_map, max_legs: int) -> pd.DataFrame:
    """
    ورودی:
        levels_df: دیتافریم خروجی fibo_levels (ستون‌های fib_236, fib_382, ...)
        ratios_map: لیست جفت (نام‌ستون، مقدار نسبت)
        max_legs: حداکثر n مقدار اخیر غیر-NaN از هر ستون
    خروجی:
        DataFrame با ستون‌های ['price','ratio'] شامل چند ردیف برای هر نسبت
    """
    rows = []
    for col, r in ratios_map:
        if col not in levels_df.columns:
            continue
        s = levels_df[col].dropna()
        if s.empty:
            continue
        # n مقدار آخرِ معتبر
        tail = s.tail(int(max_legs))
        for val in tail.tolist():
            rows.append({"price": float(val), "ratio": float(r)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["price","ratio"])


def resolve_run_params_from_config_and_cli(args: argparse.Namespace) -> RunParams:
    """
    خواندن config.yaml و ترکیب با آرگومان‌های CLI برای ساخت پارامترهای اجرا.
    اولویت: CLI > Config > Defaults
    """
    try:
        cfg_all = ConfigLoader().get_all()
    except Exception as e:
        logger.info("Could not load config.yaml: %s", e)
        cfg_all = {}

    # نمادها و تایم‌فریم‌ها
    cfg_symbols = _deep_get(cfg_all, "data.symbols", []) or _deep_get(cfg_all, "symbols", [])
    cfg_timeframes = _deep_get(cfg_all, "data.timeframes", []) or _deep_get(cfg_all, "timeframes", [])
    cfg_base_tf = _deep_get(cfg_all, "data.base_timeframe", None) or "H1"

    # k فرکتال (عرض پنجره‌ی سوئینگ)
    cfg_k = int(_deep_get(cfg_all, "features.fibonacci.swing_window_bars", 2) or 2)

    # سایر پیش‌فرض‌ها
    data_root = Path(args.data_root).resolve() if args.data_root else Path(".").resolve()
    data_layout = args.data_layout or "processed-first"
    adr_window = int(args.adr_window or 14)
    tz = str(args.tz or "UTC")

    # CLI Overrides
    symbols = args.symbols if args.symbols else (cfg_symbols if cfg_symbols else ["XAUUSD"])
    tfs = args.timeframes if args.timeframes else (cfg_timeframes if cfg_timeframes else ["M15", "H1", "H4", "D1"])
    base_tf = args.base_tf if args.base_tf else cfg_base_tf

    top_n = int(args.top_n or 10)
    k_fractal = int(args.k or cfg_k)

    return RunParams(
        symbols=symbols,
        timeframes=tfs,
        base_tf=base_tf,
        k_fractal=k_fractal,
        top_n=top_n,
        data_root=data_root,
        data_layout=data_layout,
        adr_window=adr_window,
        tz=tz,
    )


def _candidate_paths(data_root: Path, symbol: str, tf: str, data_layout: str) -> List[Path]:
    """
    تولید لیست مسیرهای کاندید برای فایل‌های داده.
    """
    raw_root  = Path(_deep_get(cfg, "paths.raw_dir"      ,   "f03_data/raw"))         # ← کلید دقیق کانفیگ شما
    proc_root = Path(_deep_get(cfg, "paths.processed_dir",   "f03_data/processed"))  # ← کلید دقیق کانفیگ شما

    # اگر relative بودند، نسبت به data_root حل‌شان کن:
    if not raw_root.is_absolute():
        raw_root = (data_root / raw_root).resolve()
    if not proc_root.is_absolute():
        proc_root = (data_root / proc_root).resolve()
    
    processed = [proc_root / symbol / f"{tf}.parquet", proc_root / symbol / f"{tf}.csv"]
    raw       = [raw_root  / symbol / f"{tf}.parquet", raw_root  / symbol / f"{tf}.csv"]
    return (processed + raw) if data_layout == "processed-first" else (raw + processed)


def _read_ohlcv(path: Path) -> Optional[pd.DataFrame]:
    """خواندن دیتافریم OHLCV از روی فایل parquet/csv با تبدیل ستون‌های نامتعارف در صورت نیاز"""
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            return None
        # نرمال‌سازی ستون‌ها: lower-case و اطمینان از وجود ستون‌های کلیدی
        df.columns = [str(c).strip() for c in df.columns]
        lower_map = {c: c.lower() for c in df.columns}
        df.rename(columns=lower_map, inplace=True)
        needed = {"time", "open", "high", "low", "close"}
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df.set_index("time", inplace=True, drop=True)
        elif df.index.name and "time" in str(df.index.name).lower():
            # فرض بر این‌که index همان time است
            pass
        else:
            # اگر زمان نداریم، نمی‌توانیم ادامه دهیم
            return None
        if not needed.issubset(set(df.columns) | {"time"}):
            # برخی فایل‌ها ممکن است volume نداشته باشند؛ الزامی نیست
            missing = needed - (set(df.columns) | {"time"})
            if missing:
                return None
        # مرتب‌سازی زمانی و حذف ردیف‌های NaT
        df = df[df.index.notna()].sort_index()
        return df
    except Exception:
        return None


def load_ohlcv_for_symbol(
    symbol: str,
    tfs: Sequence[str],
    data_root: Path,
    data_layout: str) -> Dict[str, pd.DataFrame]:
    """
    تلاش برای خواندن دادهٔ OHLCV هر تایم‌فریم از مسیرهای متعارف پروژه.
    در صورت عدم موفقیت برای یک TF، آن TF حذف می‌شود.
    """
    out: Dict[str, pd.DataFrame] = {}
    for tf in tfs:
        found = None
        for p in _candidate_paths(data_root, symbol, tf, data_layout):
            if p.exists():
                df = _read_ohlcv(p)
                if df is not None and not df.empty:
                    found = df
                    logger.info("Loaded %s %s: %s rows from %s", symbol, tf, len(df), p)
                    break
        if found is None:
            logger.info("Could not load %s %s from known locations.", symbol, tf)
        else:
            out[tf] = found
    return out


def build_fibo_levels_per_tf(df: pd.DataFrame, k: int) -> Optional[pd.DataFrame]:
    #     این تابع فرض می‌کند ستون‌های 'close','high','low' در df موجود باشد.
    """
    تولید سطوح فیبو برای یک تایم‌فریم با استفاده از levels.fibo_levels.
    خروجی levels.fibo_levels در این پروژه یک dict است؛ اینجا آن را به DataFrame قابل‌استفاده تبدیل می‌کنیم.
    """
    out = lv.fibo_levels(
        close=df["close"],
        high=df["high"],
        low=df["low"],
        k=int(k),
    )

    # حالت رایج: dict → ممکن است کلید 'levels' یا ساختارهای دیگر داشته باشد
    if isinstance(out, dict):
        # اگر خودِ dict حاوی DataFrame آماده باشد
        if "levels" in out and hasattr(out["levels"], "empty"):
            return out["levels"]

        # اگر mapping ساده از ستون‌ها باشد، تبدیل مستقیم
        try:
            df_levels = pd.DataFrame(out)
            return df_levels if not df_levels.empty else None
        except Exception:
            pass

        # لاگ تشخیصی حداقلی در صورت ساختار نامعمول
        logger.info("Unexpected fibo_levels dict keys: %s", list(out.keys()))
        return None

    # اگر خودِ خروجی DataFrame بود
    if hasattr(out, "empty"):
        return out if not out.empty else None

    # نوع ناشناخته
    logger.info("Unexpected fibo_levels return type: %s", type(out))
    return None


def compute_adr_value(df: pd.DataFrame, window: int, tz: str) -> Optional[float]:
    """
    محاسبه ADR با استفاده از levels.compute_adr و برگرداندن مقدار آخرین روز.
    """
    try:
        adr_series = lv.compute_adr(df=df, window=int(window), tz=str(tz))
        if adr_series is None or len(adr_series) == 0:
            return None
        return float(adr_series.iloc[-1])
    except Exception as e:
        logger.info("ADR computation failed: %s", e)
        return None


# -----------------------------------------------------------------------------
# Helpers: MA slope & SR levels (simple and robust)
# -----------------------------------------------------------------------------
'''
این دو تابع از اینجا پاک شده اند
compute_ma_slope_series(df: pd.DataFrame, window: int) -> Optional[pd.Series]:
extract_sr_levels_from_fractals(df: pd.DataFrame, k: int = 2, lookback: int = 1500, max_levels: int = 30) -> Optional[List[float]]:
'''

# -----------------------------------------------------------------------------
# Scoring enhancer: تقویت امتیاز خوشه با MA و SR (بدون دست‌زدن به هسته)
# -----------------------------------------------------------------------------

def enhance_cluster_scores(
    cluster_df: pd.DataFrame,
    ma_slope: Optional[pd.Series],
    sr_levels: Optional[List[float]],
    cfg_all: dict,
    ref_time: Optional[pd.Timestamp],
    adr_value: Optional[float],
) -> pd.DataFrame:
    """
    ورودی:
        cluster_df: خروجی fib_cluster_cfg (ستون score را دارد)
        ma_slope: سری شیب MA (یا None)
        sr_levels: لیست سطوح SR (یا None)
        cfg_all: تمام کانفیگ (برای خواندن وزن‌ها/تلورانس)
        ref_time: زمان مرجع برای نمونه‌برداری شیب MA
        adr_value: برای نرمال‌سازی شیب (اگر None باشد از قدر مطلق خودش استفاده می‌کنیم)
    خروجی:
        DataFrame با ستون score به‌روزشده (ستون‌های کمکی trend_score و sr_score هم اضافه می‌شود)
    """
    if cluster_df is None or cluster_df.empty:
        return cluster_df.copy()

    out = cluster_df.copy()

    # خواندن وزن‌ها و تلورانس از کانفیگ (اگر نبود، پیش‌فرض فعلی)
    w_trend = float(_deep_get(cfg_all, "features.fibonacci.cluster.w_trend", 10.0))
    w_sr    = float(_deep_get(cfg_all, "features.fibonacci.cluster.w_sr", 10.0))
    sr_tol  = float(_deep_get(cfg_all, "features.fibonacci.cluster.sr_tol_pct", 0.05))

    # --- 1) Trend score از MA slope ---
    trend_scalar = 0.0
    if ma_slope is not None and isinstance(ma_slope, pd.Series) and len(ma_slope) > 0:
        try:
            if ref_time is not None and ref_time in ma_slope.index:
                slope_val = float(ma_slope.loc[ref_time])
            elif ref_time is not None:
                slope_val = float(ma_slope.loc[:ref_time].iloc[-1])
            else:
                slope_val = float(ma_slope.iloc[-1])
            # نرمال‌سازی ساده: اگر ADR داریم، نسبت به آن؛ وگرنه tanh روی مقدار خام
            if adr_value and adr_value > 0:
                trend_scalar = float(np.tanh(slope_val / adr_value))
            else:
                trend_scalar = float(np.tanh(slope_val))
        except Exception:
            trend_scalar = 0.0

    out["trend_score"] = trend_scalar  # یک اسکالر روی تمام ردیف‌ها اعمال می‌شود

    # --- 2) SR overlap score برای هر خوشه ---
    def _sr_score(row):
        # اگر لیست SR نداریم، صفر
        if not sr_levels:
            return 0.0
        price = float(row["price_mean"]) if "price_mean" in row else None
        if price is None:
            return 0.0
        try:
            # levels.sr_overlap_score(price, sr_levels, tol_pct) → float
            return float(lv.sr_overlap_score(price=price, sr_levels=sr_levels, tol_pct=sr_tol))
        except Exception:
            return 0.0

    out["sr_score"] = out.apply(_sr_score, axis=1)

    # --- 3) ترکیب با وزن‌ها (جمع شونده روی score پایه) ---
    if "score" not in out.columns:
        out["score"] = 0.0
    out["score"] = out["score"] + w_trend * out["trend_score"] + w_sr * out["sr_score"]

    logger.info("Enhanced scores applied (w_trend=%.2f, w_sr=%.2f, sr_tol=%.3f).", w_trend, w_sr, sr_tol)
    return out


# -----------------------------------------------------------------------------
# اجرای اسموک‌ران برای یک نماد
# -----------------------------------------------------------------------------

def run_for_symbol(params: RunParams, symbol: str) -> None:
    """
    اجرای کامل اسموک‌ران برای یک نماد:
      - بارگذاری داده‌های چند TF
      - ساخت سطوح فیبو هر TF
      - اجرای fib_cluster_cfg با tol_pct انطباقی در صورت فعال بودن
      - چاپ Top-N نتایج
    """
    logger.info("=== Symbol: %s ===", symbol)

    # استفاده از هسته‌ی جدید (pipeline) به‌صورت import موضعی تا وابستگی بیرونی نداشته باشیم
    from f04_features.indicators import fibo_pipeline as fp
    import f04_features.indicators.levels as lv  # برای لاگِ ساختار wide و آمار ستون‌ها
    from f04_features.indicators.extras_trend import ma_slope as func_ma_slope

    # 1) بارگذاری داده‌ها برای تایم‌فریم‌های خواسته‌شده
    data_by_tf = load_ohlcv_for_symbol(symbol, params.timeframes, params.data_root, params.data_layout)
    if not data_by_tf:
        logger.info("No data found for symbol %s. Skipping.", symbol)
        return

    # 2) تعیین ref_time و ref_price از روی تایم‌فریم پایه
    if params.base_tf not in data_by_tf:
        logger.info("Base timeframe %s not loaded for %s. Using any available TF as base.", params.base_tf, symbol)
        base_tf = next(iter(data_by_tf.keys()))
    else:
        base_tf = params.base_tf

    base_df = data_by_tf[base_tf]
    if base_df.empty:
        logger.info("Base timeframe %s has no rows. Skipping symbol %s.", base_tf, symbol)
        return

    ref_time = base_df.index[-1]
    ref_price = float(base_df["close"].iloc[-1])
    logger.info("Base TF=%s ref_time=%s ref_price=%.5f", base_tf, str(ref_time), ref_price)

    # 3) ساخت سطوح فیبو برای هر TF
    tf_levels: Dict[str, pd.DataFrame] = {}
    #----------- start
    # فارسی: تعداد لگ‌های منتخب از کانفیگ (پیش‌فرض: 3)
    try:
        cfg_all = ConfigLoader().get_all()
    except Exception:
        cfg_all = {}
    max_legs = int(_deep_get(cfg_all, "features.fibonacci.leg_selection.max_legs_per_tf", 3))
    #----------- end
    for tf, df in data_by_tf.items():
        try:
            # استفاده از هسته: خروجی long (price, ratio) برای چند لگ اخیر
            lv_long = fp.build_fibo_levels_per_tf_multi(
                df=df,
                k=int(params.k_fractal),
                max_legs_per_tf=max_legs,
                cfg_all=cfg_all,
            )
            if lv_long is not None and not lv_long.empty:
                # تبدیل wide → long برای خوشه‌سازی (آخرین مقدار غیر-NaN هر نسبت)

                # تکراری است logger.info("Built fibo levels: TF=%s rows=%d", tf, len(levels_df))

                # قبل از ساخت rows:
                ratios_cfg = _deep_get(cfg_all, "features.fibonacci.retracement_ratios", [0.236,0.382,0.5,0.618,0.786])
                # اگر نام ستون‌ها مثل fib_236, fib_382, ... هستند:
                ratios_map = [(f"fib_{str(r).replace('0.','').ljust(3,'0')}", float(r)) for r in ratios_cfg]

                # فارسی: استخراج چند لگ اخیر از هر نسبت و ساخت فرم long
                tf_levels[tf] = lv_long  # حالا خروجی long را مستقیم می‌گذاریم

                # (اختیاری) لاگ تشخیصی کوتاه
                logger.info("TF=%s → multi-leg rows=%d", tf, 0 if tf_levels[tf].empty else len(tf_levels[tf]))

                # ------------- start
                # برای سازگاری با لاگ‌های قبلی، یک DataFrame wide فقط برای گزارش می‌سازیم
                levels_df = pd.DataFrame(lv.fibo_levels(close=df["close"], high=df["high"], low=df["low"], k=int(params.k_fractal)))
                logger.info("Built fibo levels: TF=%s rows=%d", tf, len(levels_df))
                # فارسی: آمار non-NaN و آخرین اندیس معتبر برای ستون‌های فیبو
                for c in ["fib_236", "fib_382", "fib_500", "fib_618", "fib_786"]:
                    if c in levels_df.columns:
                        s = levels_df[c]
                        nn = int(s.notna().sum())
                        last_idx = s.last_valid_index()
                        logger.info("Stats[%s %s] nonNaN=%d last_valid=%s", tf, c, nn, str(last_idx))
                # ------------- end
                logger.info("Columns[%s]: %s", tf, list(levels_df.columns))
                logger.info("Head[%s]:\n%s", tf, levels_df.head(3).to_string(index=False))

            else:
                logger.info("Empty levels for TF=%s", tf)
        except Exception as e:
            logger.info("Failed to build levels for TF=%s: %s", tf, e)

    if not tf_levels:
        logger.info("No fibo levels built for symbol %s. Skipping.", symbol)
        return

    # 4) محاسبه ADR برای tol_pct انطباقی (در صورت فعال بودن در کانفیگ)
    adr_value = fp.compute_adr_value(base_df, params.adr_window, params.tz)
    if adr_value is not None:
        logger.info("ADR(%d) for base TF %s: %.5f", params.adr_window, base_tf, adr_value)
    else:
        logger.info("ADR not available; will fall back to fixed tol_pct if adaptive is enabled.")

    # ============================================= start new block
    # --- Inject MA slope & SR levels into clustering (confluence) ---
    # فارسی: پارامترها را از کانفیگ بخوان (اختیاری؛ اگر نبود پیش‌فرض می‌گیریم)
    try:
        cfg_all = ConfigLoader().get_all()
    except Exception:
        cfg_all = {}

    ma_window = int(_deep_get(cfg_all, "features.levels.ma.window", 50) or 50)  # پیش‌فرض: 50
    sr_k = int(_deep_get(cfg_all, "features.levels.sr.k", 2) or 2)              # پیش‌فرض: 2 (فراکتال کلاسیک)
    sr_lookback = int(_deep_get(cfg_all, "features.levels.sr.lookback", 1500) or 1500)
    sr_max = int(_deep_get(cfg_all, "features.levels.sr.max_levels", 30) or 30)

    # محاسبه‌ی شیب MA روی base_df
    # ma_slope = fp.compute_ma_slope_series(base_df, ma_window)
    ma_slope = func_ma_slope(base_df, 
                            price_col = "close",
                            window = ma_window,
                            method = "sma",
                            norm = "none")

    # استخراج سطوح SR از فراکتال‌ها روی base_df
    sr_levels = fp.extract_sr_levels_from_fractals(base_df, k=sr_k, lookback=sr_lookback, max_levels=sr_max)

    if ma_slope is None:
        logger.info("MA slope not available; clustering will run without ma_slope.")
    else:
        logger.info("MA slope available with window=%d.", ma_window)

    if not sr_levels:
        logger.info("SR levels not available; clustering will run without sr_levels.")
    else:
        logger.info("SR levels extracted: %d levels.", len(sr_levels))

    # ============================================= end new block

    # 5) اجرای خوشه‌سازی
    try:
        result = fp.run_fibo_cluster(
            symbol=symbol,
            tf_dfs=data_by_tf,        # همان دیکشنری TF→DataFrame که بالاتر ساختی
            base_tf=base_tf,
            k_fractal=params.k_fractal,
            max_legs_per_tf=max_legs,
            tz=params.tz,
        )
        cluster_df = result.clusters
    except Exception as e:
        logger.info("fib_cluster_cfg failed: %s", e)
        return

    if cluster_df is None or cluster_df.empty:
        logger.info("No clusters produced for %s.", symbol)
        return
    """
    #========/ start
    # فارسی: تقویت امتیازها با MA و SR (بدون دست‌زدن به هسته)
    cluster_df = fp.enhance_cluster_scores(
        cluster_df=cluster_df,
        ma_slope=ma_slope,
        sr_levels=sr_levels,
        cfg_all=cfg_all,        # همان کانفیگی که قبلاً بالا در تابع خواندی
        ref_time=ref_time,
        adr_value=adr_value,
    )
    نکته‌ها:
    اگر MA یا SR در دسترس نباشند، تابع به‌صورت ایمن صفر را لحاظ می‌کند و فقط score پایه می‌ماند.
    وزن‌ها/تلورانس از همان کانفیگ فعلی شما خوانده می‌شوند؛ نیازی به افزودن کلید جدید نیست.
    ستون‌های trend_score و sr_score برای دیباگ اضافه می‌شوند و می‌توانی ببینی چه‌قدر اثر گذاشته‌اند.
    """
    #========/ end
    # (removed) Avoid double-counting: keep scores from fib_cluster_cfg only.
    logger.info("Cluster scoring kept from fib_cluster_cfg (no re-weighting).")

    # Light smoke logs (non-breaking)
    logger.info("clusters rows: %d", len(cluster_df))
    if hasattr(result, "abc_projections"):
        ap = result.abc_projections or []
        logger.info("abc_projections: %d", len(ap))
    if hasattr(result, "order_planner"):
        op = result.order_planner or {}
        logger.info("order_planner keys: %s", list(op.keys()))


    # 6) نمایش Top-N خوشه‌ها
    # تلاش برای مرتب‌سازی بر حسب 'score' در صورت وجود؛ در غیر اینصورت بر اساس 'price_mean'
    sort_cols = [c for c in ("score", "price_mean") if c in cluster_df.columns]
    if sort_cols:
        cluster_df = cluster_df.sort_values(by=sort_cols, ascending=[False] + [True]*(len(sort_cols)-1))

    top_n = min(params.top_n, len(cluster_df))
    logger.info("Top-%d clusters for %s:", top_n, symbol)

    try:
        # ستون‌های کلیدی که معمولاً انتظار می‌رود وجود داشته باشند
        # فارسی: ستون‌های کلیدی را اگر وجود دارند، نشان بده
        cols_pref = [c for c in ["price_mean", "price_min", "price_max", "score", "trend_score", "sr_score", "members", "tfs", "ratios"] if c in cluster_df.columns]
        preview = cluster_df[cols_pref].head(top_n) if cols_pref else cluster_df.head(top_n)
        # چاپ خوانا
        logger.info("\n%s", preview.to_string(index=False))
    except Exception:
        # اگر به هر دلیلی انتخاب ستون‌ها شکست خورد، کل DataFrame را نشان بده
        logger.info("\n%s", cluster_df.head(top_n).to_string(index=False))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data smoke run for Fibonacci clusters (config-driven).")
    parser.add_argument("--symbols", nargs="+", help="Symbols to run (e.g., XAUUSD EURUSD).")
    parser.add_argument("--timeframes", nargs="+", help="Timeframes to include (e.g., M15 H1 H4 D1).")
    parser.add_argument("--base-tf", help="Base timeframe for ref_time/ADR (default: from config or H1).")
    parser.add_argument("--k", type=int, help="Fractal window size for swings (default: from config or 2).")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top clusters to show (default: 10).")
    parser.add_argument("--data-root", help="Project root to resolve data paths (default: current directory).")
    parser.add_argument("--data-layout", choices=["processed-first", "raw-first"], default="processed-first",
                        help="File search order for data locations (default: processed-first).")
    parser.add_argument("--adr-window", type=int, default=14, help="Window for ADR computation (default: 14).")
    parser.add_argument("--tz", type=str, default="UTC", help="Timezone used by ADR computation (default: UTC).")
    args = parser.parse_args()

    params = resolve_run_params_from_config_and_cli(args)

    logger.info("Symbols: %s", params.symbols)
    logger.info("Timeframes: %s (base=%s)", params.timeframes, params.base_tf)
    logger.info("Fractal k: %d | Top-N: %d", params.k_fractal, params.top_n)
    logger.info("Data root: %s | Layout: %s", str(params.data_root), params.data_layout)

    for sym in params.symbols:
        run_for_symbol(params, sym)

    logger.info("Cluster smoke run finished.")


if __name__ == "__main__":
    main()
