# -*- coding: utf-8 -*-
# f04_features/pipelines/fibo_pipeline.py
# Status in (Bot-RL-3): Completed

"""
Pipeline فیبوناچی (قابل‌مصرف در هستهٔ ربات)
- ساخت سطوح فیبو per-TF (multi-leg → long)
- محاسبه ADR/MA-slope/SR
- اجرای خوشه‌ها (config-driven) و تقویت امتیاز با MA/SR
- یک API سطح‌بالا برای گرفتن خروجی نهایی DataFrame

قوانین پروژه:
- پیام‌های ترمینال/لاگ: انگلیسی
- توضیحات/کامنت‌ها: فارسی
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Sequence
import logging
import pandas as pd
import numpy as np

# توجه: ماژول‌های هسته (همین‌جا مصرف می‌شوند)
import f04_features.indicators.levels as lv
from f04_features.indicators.fibonacci import fib_cluster_cfg, select_legs_from_swings, levels_from_legs
from f04_features.indicators.utils import detect_swings
from f04_features.indicators.core import rsi as rsi_core, atr as atr_core
from f04_features.indicators.extras_trend import ma_slope_multistep as func_ma_slope
from f02_utils.config_loader import ConfigLoader  # :contentReference[oaicite:0]{index=0}
from f02_utils.config_operations import _deep_get

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[FIBO-PIPE] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# ابزار کمکی عمومی
# -----------------------------------------------------------------------------
def _ratios_map_from_cfg(cfg_all: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    استخراج نقشهٔ ستون→نسبت بر اساس levels.fibo_levels فعلی.
    - در پروژهٔ فعلی ستون‌ها fib_236..fib_786 هستند (levels.py). :contentReference[oaicite:1]{index=1}
    - اگر در آینده نسبت جدیدی به کانفیگ اضافه شد (مثلاً 0.886) اینجا می‌توانیم map را گسترش دهیم.
    """
    # در وضعیت فعلی تابع levels.fibo_levels ستون‌های ثابت می‌سازد:
    base = [( "fib_236", 0.236),
            ( "fib_382", 0.382),
            ( "fib_500", 0.500),
            ( "fib_618", 0.618),
            ( "fib_786", 0.786)]
    # اگر config نسبت‌های رتریس اضافه داشت اما levels.py ستونش را نمی‌سازد، نادیده گرفته می‌شود (سازگار).
    # می‌توان در آینده levels.fibo_levels را همگام کرد.
    return base


# -----------------------------------------------------------------------------
# ۱) ساخت سطوح فیبو per-TF (multi-leg → long)
# -----------------------------------------------------------------------------
def build_fibo_levels_per_tf_multi(
    df: pd.DataFrame,
    k: int,
    max_legs_per_tf: int = 3,
    cfg_all: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    ورودی:
      df: دیتافریم OHLCV تایم‌فریم
      k: پارامتر فراکتال/سوئینگ برای levels.fibo_levels
      max_legs_per_tf: تعداد n مقدار اخیرِ غیر-NaN از هر نسبت (multi-leg)
      cfg_all: برای همخوانی با کانفیگ (مثلاً استخراج نسبت‌ها)

    خروجی:
      DataFrame با ستون‌های ['price','ratio'] (چند ردیف برای هر نسبت)

    توضیح:
      - از lv.fibo_levels استفاده می‌کنیم (در پروژه دیکشنریِ ستون‌ها می‌دهد). :contentReference[oaicite:2]{index=2}
      - سپس چند مقدار آخر هر ستون را به فرمت long تبدیل می‌کنیم.
    """
    if df is None or df.empty or not {"close","high","low"}.issubset(df.columns):
        return None

    # ------------------------------------------------------------------------- Added 040618 start
    # --- مسیر ۱: ساخت سطوح از «لگ‌ها» (در صورت موفقیت) ---
    try:
        # پارامترهای انتخاب لگ از config
        _max_legs = int(_deep_get(cfg_all or {}, "features.fibonacci.leg_selection.max_legs_per_tf", max_legs_per_tf))
        _prom_min = _deep_get(cfg_all or {}, "features.fibonacci.leg_selection.prominence_min", None)
        # سوئینگ‌ها از سری قیمت بسته؛ حداقل فاصلهٔ قله/کف = k
        swings = detect_swings(price=df["close"].astype(float),
                               prominence=_prom_min, min_distance=int(k),
                               atr=None, atr_mult=None, tf=None)
        legs = select_legs_from_swings(swings,
                                       max_legs=_max_legs,
                                       min_prominence=_prom_min,
                                       min_length_pct=None,
                                       max_age_bars=None)
        if legs:
            # نگاشت نسبت‌ها از config و استخراج لیست عددی
            ratios_map = _ratios_map_from_cfg(cfg_all or {})
            _ratios = [r for _, r in ratios_map] or None
            leg_levels = levels_from_legs(legs, ratios=_ratios)  # → DataFrame ['price','ratio']
            if leg_levels is not None and not leg_levels.empty:
                logger.info("[FIBO-PIPE] Leg-path built %d rows.", len(leg_levels))
                return leg_levels[["price", "ratio"]]
    except Exception as _e:
        logger.info("Leg-based levels path failed; falling back. %s", _e)
    # ------------------------------------------------------------------------- Added 040618 end
    
    # --- مسیر ۲ (فالبک امن): منطق فعلی بر پایهٔ lv.fibo_levels ---
    out = lv.fibo_levels(close=df["close"], high=df["high"], low=df["low"], k=int(k))
    if isinstance(out, dict):
        levels_df = pd.DataFrame(out)
    elif hasattr(out, "empty"):
        levels_df = out
    else:
        logger.info("Unexpected fibo_levels output type: %s", type(out))
        return None

    if levels_df is None or levels_df.empty:
        return None

    # wide → long (آخرین n مقدار معتبر هر نسبت)
    ratios_map = _ratios_map_from_cfg(cfg_all or {})
    rows: List[Dict[str, float]] = []
    for col, r in ratios_map:
        if col not in levels_df.columns:
            continue
        s = pd.to_numeric(levels_df[col], errors="coerce").dropna()
        if s.empty:
            continue
        tail = s.tail(int(max_legs_per_tf))
        rows += [{"price": float(v), "ratio": float(r)} for v in tail.tolist()]

    #return pd.DataFrame(rows, columns=["price", "ratio"]) if rows else None
    out_df = pd.DataFrame(rows, columns=["price", "ratio"]) if rows else None
    if out_df is not None:
        logger.info("[FIBO-PIPE] Fallback fibo_levels-path built %d rows.", len(out_df))
    else:
        logger.info("[FIBO-PIPE] Fallback fibo_levels-path returned empty.")
    return out_df

# -----------------------------------------------------------------------------
# ۲) ADR / MA-slope / SR
# -----------------------------------------------------------------------------
def compute_adr_value(df: pd.DataFrame, window: int, tz: str) -> Optional[float]:
    """
    محاسبه ADR از lv.compute_adr و برگرداندن مقدار آخر (برای نرمال‌سازی tol/Trend).
    """
    try:
        adr_series = lv.compute_adr(df=df, window=int(window), tz=str(tz))  # :contentReference[oaicite:3]{index=3}
        if adr_series is None or len(adr_series) == 0:
            return None
        return float(adr_series.iloc[-1])
    except Exception as e:
        logger.info("ADR computation failed: %s", e)
        return None


def extract_sr_levels_from_fractals(
            df: pd.DataFrame, k: int = 2,
            lookback: int = 1500,
            max_levels: int = 30) -> Optional[List[float]]:
    """
    استخراج سطوح SR از روی فراکتال‌ها؛ اگر چیزی برنگشت از پیوت‌های کلاسیک fallback می‌گیریم.
    """
    try:
        seg = df.iloc[-int(lookback):] if len(df) > lookback else df
        if not {"high","low"}.issubset(seg.columns):
            return None

        fp = lv.fractal_points(high=seg["high"], low=seg["low"], k=int(k))  # :contentReference[oaicite:4]{index=4}
        sr_vals: List[float] = []

        # حالت‌های رایج: dict/DataFrame/tuple of Series (نسخه فعلی tuple برمی‌گرداند)
        if isinstance(fp, dict):
            for key in ("highs", "lows", "H", "L"):
                if key in fp:
                    vals = pd.Series(fp[key])
                    tail = pd.to_numeric(vals, errors="coerce").dropna().tail(int(max_levels))
                    sr_vals.extend([float(x) for x in tail])
        elif hasattr(fp, "columns"):
            for key in [c for c in fp.columns if "high" in c.lower() or "low" in c.lower()]:
                tail = pd.to_numeric(fp[key], errors="coerce").dropna().tail(int(max_levels))
                sr_vals.extend([float(x) for x in tail])

        # Fallback: پیوت‌های کلاسیک — توجه اینکه در پروژه tuple از Series بازمی‌گرداند. :contentReference[oaicite:5]{index=5}
        sr_vals = sorted(set(sr_vals))
        if not sr_vals:
            piv = lv.pivots_classic(high=seg["high"], low=seg["low"], close=seg["close"])
            if isinstance(piv, tuple):
                try:
                    cand = [float(pd.to_numeric(s, errors="coerce").dropna().iloc[-1])
                            for s in piv if hasattr(s, "iloc") and not pd.to_numeric(s, errors="coerce").dropna().empty]
                except Exception:
                    cand = []
                sr_vals = sorted(set(cand))
            elif isinstance(piv, dict):
                cand = [piv.get(k) for k in ("P","S1","S2","S3","R1","R2","R3")]
                cand = [float(x) for x in cand if x is not None]
                sr_vals = sorted(set(cand))

        return sr_vals if sr_vals else None

    except Exception as e:
        logger.info("SR extraction failed: %s", e)
        return None


# -----------------------------------------------------------------------------
# ۳) تقویت امتیاز خوشه با MA و SR (بدون دست‌زدن به هسته)
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
    ترکیب امتیاز پایهٔ خروجی خوشه با مؤلفه‌های Trend و SR (وزن‌ها از کانفیگ).
    """
    if cluster_df is None or cluster_df.empty:
        return cluster_df.copy()

    out = cluster_df.copy()
    w_trend = float(_deep_get(cfg_all, "features.fibonacci.cluster.w_trend", 10.0))
    w_sr    = float(_deep_get(cfg_all, "features.fibonacci.cluster.w_sr", 10.0))
    sr_tol  = float(_deep_get(cfg_all, "features.fibonacci.cluster.sr_tol_pct", 0.05))

    # Trend scalar از MA slope
    trend_scalar = 0.0
    if ma_slope is not None and isinstance(ma_slope, pd.Series) and len(ma_slope) > 0:
        try:
            if ref_time is not None and ref_time in ma_slope.index:
                slope_val = float(ma_slope.loc[ref_time])
            elif ref_time is not None:
                slope_val = float(ma_slope.loc[:ref_time].iloc[-1])
            else:
                slope_val = float(ma_slope.iloc[-1])
            trend_scalar = float(np.tanh((slope_val / adr_value) if (adr_value and adr_value > 0) else slope_val))
        except Exception:
            trend_scalar = 0.0

    out["trend_score"] = trend_scalar  # اسکالر

    # SR overlap per-row
    def _sr_score(row):
        if not sr_levels:
            return 0.0
        price = float(row.get("price_mean", np.nan))
        if not np.isfinite(price):
            return 0.0
        try:
            return float(lv.sr_overlap_score(price=price, sr_levels=sr_levels, tol_pct=sr_tol))  # :contentReference[oaicite:6]{index=6}
        except Exception:
            return 0.0

    out["sr_score"] = out.apply(_sr_score, axis=1)
    if "score" not in out.columns:
        out["score"] = 0.0

    out["score"] = out["score"] + w_trend * out["trend_score"] + w_sr * out["sr_score"]
    logger.info("Enhanced scores applied (w_trend=%.2f, w_sr=%.2f, sr_tol=%.3f).", w_trend, w_sr, sr_tol)
    return out


# -----------------------------------------------------------------------------
# ۴) API سطح‌بالا: اجرای کامل خوشه بر اساس Config
# -----------------------------------------------------------------------------
@dataclass
class FiboRunResult:
    """خروجی سطح‌بالای خوشهٔ فیبوناچی برای مصرف مستقیم در ربات/RL."""
    symbol: str
    base_tf: str
    ref_time: Optional[pd.Timestamp]
    ref_price: Optional[float]
    clusters: pd.DataFrame
    abc_projections: Optional[List[Dict[str, Any]]] = None
    
def run_fibo_cluster(
    symbol: str,
    tf_dfs: Dict[str, pd.DataFrame],
    base_tf: str,
    k_fractal: int = 2,
    max_legs_per_tf: int = 3,
    tz: str = "UTC",
    swings_for_abc: Optional[pd.DataFrame] = None) -> FiboRunResult:

    """
    اجرای سرراست خوشهٔ فیبوناچی:
      - ساخت سطوح فیبو برای هر TF
      - گرفتن ref_time/ref_price از TF پایه
      - محاسبه ADR/MA/SR
      - صدا زدن خوشه (config-driven) و تقویت امتیاز
    """
    cfg_all = ConfigLoader().get_all()  # :contentReference[oaicite:7]{index=7}

    # ۱) build tf_levels (long)
    tf_levels: Dict[str, pd.DataFrame] = {}
    for tf, df in tf_dfs.items():
        lv_df = build_fibo_levels_per_tf_multi(
            df=df, k=int(k_fractal),
            max_legs_per_tf=int(_deep_get(cfg_all, "features.fibonacci.leg_selection.max_legs_per_tf", max_legs_per_tf)),
            cfg_all=cfg_all,
        )
        if lv_df is not None and not lv_df.empty:
            tf_levels[tf] = lv_df

    if base_tf not in tf_dfs:
        raise ValueError(f"Base TF '{base_tf}' not in provided dataframes.")

    base_df = tf_dfs[base_tf]
    if base_df is None or base_df.empty:
        raise ValueError(f"Base TF '{base_tf}' dataframe is empty.")

    # ۲) مرجع زمانی/قیمت
    ref_time = base_df.index[-1] if len(base_df.index) > 0 else None
    ref_price = float(base_df["close"].iloc[-1]) if "close" in base_df.columns and len(base_df) else None

    # مقداردهی امن برای جلوگیری از UnboundLocalError در صورت استثناء
    adr_value = None
    ma_slope  = None
    sr_levels = None

    # ۳) ADR / MA / SR
    adr_window = int(_deep_get(cfg_all, "features.levels.adr.window", 14))
    try:
        adr_value = compute_adr_value(base_df, window=adr_window, tz=str(tz))
    except Exception as e:
        logger.info("ADR compute failed: %s", e)
        adr_value = None    
    
    ma_window = int(_deep_get(cfg_all, "features.levels.ma.window", 50))
    try:
        ma_slope = func_ma_slope(base_df, 
                                price_col = "close",
                                window = ma_window,
                                method = "sma",
                                norm = "none")
    except Exception as e:
        logger.info("MA slope compute failed: %s", e)
        ma_slope = None

    sr_k = int(_deep_get(cfg_all, "features.levels.sr.k", 2))
    sr_lookback = int(_deep_get(cfg_all, "features.levels.sr.lookback", 1500))
    sr_max = int(_deep_get(cfg_all, "features.levels.sr.max_levels", 30))

    try:
        sr_levels = extract_sr_levels_from_fractals(base_df, k=sr_k, lookback=sr_lookback, max_levels=sr_max)
    except Exception as e:
        logger.info("SR levels compute failed: %s", e)
        sr_levels = None

    if ma_slope is None:
        logger.info("MA slope not available; clustering will run without ma_slope.")
    else:
        logger.info("MA slope available with window=%d.", ma_window)

    if not sr_levels:
        logger.info("SR levels not available; clustering will run without sr_levels.")
    else:
        logger.info("SR levels extracted: %d levels.", len(sr_levels))

    # ۴) خوشه (با رَپر کانفیگ‌محور موجود در fibonacci.py) :contentReference[oaicite:8]{index=8}
    #    این رَپر tol_pct/weights/... را از config.yaml می‌خواند.
    
    # ۴-b) محاسبهٔ RSI-zone score به‌صورت سری زمانی (بر مبنای config فعلی)
    #rsi_cfg   = _deep_get(cfg_all, ["fibonacci", "rsi"]) or {}
    rsi_cfg   = _deep_get(cfg_all, "fibonacci.rsi") or {}
    rsi_len   = int(rsi_cfg.get("length", 14))
    rsi_ob    = float(rsi_cfg.get("overbought", 70.0))
    rsi_os    = float(rsi_cfg.get("oversold", 30.0))
    base_df   = tf_dfs[base_tf]
    rsi_vals  = rsi_core(base_df["close"], rsi_len)
    mid       = (rsi_os + rsi_ob) / 2.0
    span2     = (rsi_ob - rsi_os) / 2.0
    rsi_zone_score = (1.0 - (rsi_vals - mid).abs() / span2).clip(lower=0.0, upper=1.0)
    # هم‌راستا با منطق موجود در fibonacci.py: نزدیک OB/OS سقف امتیاز را نگه داریم
    rsi_zone_score = rsi_zone_score.where(~(rsi_vals >= rsi_ob), 1.0)
    rsi_zone_score = rsi_zone_score.where(~(rsi_vals <= rsi_os), 1.0)    
    
    cluster_df = fib_cluster_cfg(
        tf_levels=tf_levels,
        
        ma_slope=ma_slope,
        rsi_zone_score=rsi_zone_score,
        sr_levels=sr_levels,
        ref_time=ref_time,      
        #adr_value=adr_value,
        #cfg_all=cfg_all,        
        # Adaptive tol (اگر در رَپر/کانفیگ فعال است)
        # ref_price=ref_price,
        # atr_value=None,
        # Confluence hooks     
    )
    '''
    # ۶) Order Planner (اختیاری): SL/TP ساده بر اساس ATR از config
    op_cfg    = _deep_get(cfg_all, ["fibonacci", "order_planner"]) or {}
    sl_mult   = float(op_cfg.get("sl_atr_mult", 0.0))
    tp_mult   = float(op_cfg.get("tp_atr_mult", 0.0))
    partial_tp= op_cfg.get("partial_take_profits", {})

    atr_val   = base_df["ATR"].iloc[-1] if "ATR" in base_df.columns else None
    sl_price  = (ref_price - sl_mult * atr_val) if (atr_val is not None and sl_mult > 0) else None
    tp_price  = (ref_price + tp_mult * atr_val) if (atr_val is not None and tp_mult > 0) else None
    '''
    # ------------------------------------------------------------------------- Added 040618 start
    # ۶) Order Planner (اختیاری): SL/TP و Partial-TP بر اساس ATR و config
    #op_cfg     = _deep_get(cfg_all, ["fibonacci", "order_planner"]) or {}
    op_cfg     = _deep_get(cfg_all, "fibonacci.order_planner") or {}
    use_atr    = bool(op_cfg.get("use_atr_for_sl_tp", True))
    sl_mult    = float(op_cfg.get("sl_atr_mult", 0.0))
    tp_mult    = float(op_cfg.get("tp_atr_mult", 0.0))
    pconf      = op_cfg.get("partial_take_profits", {}) or {}
    p_enabled  = bool(pconf.get("enabled", False))
    p_levels   = list(pconf.get("levels_r_multiple", []) or [])
    p_portions = list(pconf.get("portions", []) or [])

    order_plan = None
    if use_atr and ("high" in base_df.columns and "low" in base_df.columns and "close" in base_df.columns):
        atr_len   = int(op_cfg.get("atr_length", 14))
        atr_vals  = atr_core(base_df["high"], base_df["low"], base_df["close"], n=atr_len)
        atr_val   = float(atr_vals.iloc[-1]) if len(atr_vals) else None
        if atr_val and sl_mult > 0:
            sl_long   = ref_price - sl_mult * atr_val
            sl_short  = ref_price + sl_mult * atr_val
            # TP پایه (یک‌تایی) اگر tp_mult > 0 تعیین شود
            tp_long   = (ref_price + tp_mult * atr_val) if tp_mult > 0 else None
            tp_short  = (ref_price - tp_mult * atr_val) if tp_mult > 0 else None
            # Partial-TP بر اساس R = |Entry - SL|
            r_long    = abs(ref_price - sl_long)
            r_short   = abs(sl_short - ref_price)
            ptp_long  = [ref_price + m * r_long  for m in p_levels] if p_enabled and r_long  > 0 else []
            ptp_short = [ref_price - m * r_short for m in p_levels] if p_enabled and r_short > 0 else []
            order_plan = {
                "atr_len": atr_len,
                "atr": atr_val,
                "sl_mult": sl_mult,
                "tp_mult": tp_mult,
                "long":  {"sl": sl_long,  "tp": tp_long,  "partial_levels": ptp_long,  "partial_portions": p_portions},
                "short": {"sl": sl_short, "tp": tp_short, "partial_levels": ptp_short, "partial_portions": p_portions},
            }
    # ------------------------------------------------------------------------- Added 040618 end
    
    res = FiboRunResult(
        symbol=symbol,
        base_tf=base_tf,
        ref_time=ref_time,
        ref_price=ref_price,
        clusters=cluster_df if cluster_df is not None else pd.DataFrame(),
    )
    '''
    # سنجاق خروجی Order Planner (اختیاری و غیرشکننده) 
    try:
        if order_plan is not None:
            setattr(res, "order_planner", order_plan)
    except Exception:
        pass
    return res
    '''
    # اگر سوئینگ‌ها داده شده باشد، خروجی ABC را ضمیمه کن (Non-breaking)
    try:
        if (swings_for_abc is not None) and (len(swings_for_abc)>=3):
            res = attach_abc_projections_to_result(res, swings_df=swings_for_abc, cfg_all=cfg_all)
    except Exception:
        pass    
    return res

# ============================================================================
# ABC Projections (multi-ratio) — non-breaking helpers
# ============================================================================
def get_abc_projections_from_swings(
    swings_df,
    ratios=None, #(1.0, 1.127, 1.236, 1.272, 1.414, 1.5, 1.618, 2.0),
    cfg_all=None,
):
    """
    تولید Projection چندنسبتی با fallback و فیلتر tol از config
    محاسبهٔ Projection برای الگوی ABC با نسبت‌های متعدد.
    - ورودی: دیتافریم سوئینگ‌ها (همانی که detect_ab_equal_cd مصرف می‌کند)
    - خروجی: لیستی از دیکشنری‌ها با کلیدهای استاندارد ABC + D_pred/err_pct برای هر نسبت
    - بدون وابستگی به خوشه/امتیازدهی؛ فقط محاسبهٔ هندسی
    """
    from f04_features.indicators.patterns import detect_ab_equal_cd, abc_projection_adapter_from_abcd

    # ========== گام 1) تلاش برای استخراج ABC/D واقعی از روی آخرین 4 سوئینگ
    
    abcd = detect_ab_equal_cd(swings_df)
    base = abc_projection_adapter_from_abcd(abcd) if abcd else None
    if not base:
        #Fallback: آخرین سه سوئینگِ معتبر با تناوب L/H/L یا H/L/H را به ABC تبدیل کن
        # انتظار ورودی: ستون‌های 'price' و 'kind' با مقادیر {'L','H'}
        df = swings_df.dropna(subset=["price", "kind"]).copy()
        if len(df) < 3:
            return []
        # 3 نقطهٔ آخر با تناوب معتبر
        tail = df.iloc[-6:]  # کمی بیشتر برای یافتن تناوب
        seq = []
        prev = None
        for ts, row in tail.iterrows():
            k = str(row["kind"])
            if prev is None or (k != prev):
                seq.append((float(row["price"]), k))
                prev = k
        if len(seq) < 3:
            return []
        A,B,C = seq[-3][0], seq[-2][0], seq[-1][0]
        base = {"A": A, "B": B, "C": C, "D_real": None, "length_AB": abs(B - A)}


    A = base["A"]; B = base["B"]; C = base["C"]
    D_real = base.get("D_real", None)
    length_AB = base.get("length_AB", None)
    if length_AB is None:
        length_AB = abs(B - A)

    # ========== جهت حرکتِ CD را مطابق جهتِ AB تعریف می‌کنیم (ساده و قابل‌گسترش)
    cd_dir = 1.0 if (B - A) >= 0 else -1.0

    # ========== نسبت‌ها را از ورودی بگیر؛ اگر None بود و cfg_all داریم، از config بخوان
    _ratios = ratios
    if _ratios is None and cfg_all is not None:
        _ratios = _deep_get(cfg_all, "fibonacci.extension_ratios")
    if not _ratios:
        # fallback ایمن در صورت نبود config
        _ratios = (1.0, 1.127, 1.236, 1.272, 1.414, 1.5, 1.618, 2.0)

    out = []
    for r in _ratios:
        # D_pred = C + r * (|AB|) در جهت cd_dir
        d_pred = C + cd_dir * (r * length_AB)
        err = None
        if D_real is not None:
            # خطای نسبی نسبت به طول AB (سازگار با اسکیمای موجود)
            err = abs((D_real - C) - (d_pred - C)) / length_AB if length_AB else None

        out.append({
            "A": A, "B": B, "C": C,
            "D_pred": float(d_pred),
            "D_real": D_real if D_real is not None else None,
            "length_AB": float(length_AB),
            "proj_ratio": float(r),
            "err_pct": float(err) if err is not None else None,
        })
    
    # ========== (اختیاری) فیلتر کردن بر اساس tol_pct از config (اگر موجود بود)
    tol_pct = None
    if cfg_all is not None:
        tol_pct = _deep_get(cfg_all, "fibonacci.cluster.tol_pct")
    if tol_pct is not None:
        out = [d for d in out if (d["err_pct"] is None) or (d["err_pct"] <= float(tol_pct))]    
    
    return out


def attach_abc_projections_to_result(result: "FiboRunResult", swings_df, cfg_all=None, ratios=None) -> "FiboRunResult":
    """
    Optional: محاسبهٔ ABC Projection و سنجاق به result. API اصلی را نمی‌شکند.
    """
    try:
        proj = get_abc_projections_from_swings(swings_df, ratios=ratios, cfg_all=cfg_all)
    except Exception:
        proj = []
    result.abc_projections = proj
    return result
    '''
    # نمونهٔ استفاده (خارج از run_fibo_cluster):
    res = run_fibo_cluster(symbol, tf_dfs, base_tf, k_fractal, max_legs_per_tf, tz)
    res = attach_abc_projections_to_result(res, swings_df, cfg_all=cfg_all)  # swings_df همان دیتافریم سوئینگ‌ها
    '''


def compose_fibo_with_abc_full(result, swings_df, ratios=None, cfg_all=None):
    """
    هِلپر غیرشکنندهٔ سطح‌بالا:
    - خروجی اصلی پایپلاین را دست‌نخورده برمی‌گرداند
    - در کنار آن، لیست کامل Projectionهای ABC (multi-ratio) را ضمیمه می‌کند
    """
    try:
        proj = get_abc_projections_from_swings(swings_df, ratios=ratios, cfg_all=cfg_all)
    except Exception:
        proj = []
    return {"result": result, "abc_projections": proj}









from f04_features.indicators.utils import (
    nearest_level_distance as _nearest_level_distance, # Used in: _fibo_features_full_adapter, _adv_round_levels
    compute_atr                                        # Used in: _fibo_features_full_adapter
)
# --- fibo_features_full ------------------------------------------------------ Func.1
def _fibo_features_full_adapter(*, symbol, tf_dfs, base_tf, atr_len: int = 14, **_) -> Dict[str, pd.Series]:
    """
    خروجی per-bar هم‌تراز با base_tf از خوشه‌های فیبوناچی:
    - فاصله‌ها (abs/signed/level)
    - نرمال‌سازی فاصله با ATR
    - نسبت فیبو و پرچم extension
    - نرمال‌سازی امتیاز خوشه (score به [0,1])
    - عبور دادن sr_score و trend_score (اگر در کلستر موجود باشند)
    """
    # 1) اجرای پایپلاین فیبو و دریافت کلسترها
    res = run_fibo_cluster(symbol=symbol, tf_dfs=tf_dfs, base_tf=str(base_tf))
    clusters = res.clusters if (res is not None and isinstance(res.clusters, pd.DataFrame)) else pd.DataFrame()

    # 2) سری‌های پایه از TF مبنا
    base = tf_dfs[str(base_tf)]
    close = base["close"]
    high  = base.get("high", None)
    low   = base.get("low", None)

    # 3) اگر کلستری نیست، ستون‌ها را NaN برگردان
    if clusters is None or clusters.empty:
        nan = close.astype("float32") * np.nan
        return {
            f"fibc_nearest_abs@{base_tf}": nan,
            f"fibc_nearest_signed@{base_tf}": nan,
            f"fibc_nearest_level@{base_tf}": nan,
            f"fibc_nearest_abs_atr@{base_tf}": nan,
            f"fibc_nearest_ratio@{base_tf}": nan,
            f"fibc_is_extension@{base_tf}": nan,
            f"fibc_score_norm@{base_tf}": nan,
            f"fibc_sr_score@{base_tf}": nan,
            f"fibc_trend_score@{base_tf}": nan,
        }

    # 4) لیست سطوح خوشه (price_mean اگر موجود؛ در غیراینصورت price)
    level_col = "price_mean" if "price_mean" in clusters.columns else ("price" if "price" in clusters.columns else None)
    if level_col is None:
        nan = close.astype("float32") * np.nan
        return {
            f"fibc_nearest_abs@{base_tf}": nan,
            f"fibc_nearest_signed@{base_tf}": nan,
            f"fibc_nearest_level@{base_tf}": nan,
            f"fibc_nearest_abs_atr@{base_tf}": nan,
            f"fibc_nearest_ratio@{base_tf}": nan,
            f"fibc_is_extension@{base_tf}": nan,
            f"fibc_score_norm@{base_tf}": nan,
            f"fibc_sr_score@{base_tf}": nan,
            f"fibc_trend_score@{base_tf}": nan,
        }
    levels = clusters[level_col].astype("float32").tolist()
    #logger.info("FIBO clusters columns: %s", list(clusters.columns))   # temp_logger

    # 5) nearest level metrics per-bar
    def _nearest_tuple(p: float):
        d = _nearest_level_distance(float(p), levels)
        # اندیس نزدیک‌ترین کلستر
        j = int(np.argmin([abs(float(p) - float(lv)) for lv in levels])) if levels else -1
        if j < 0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        row = clusters.iloc[j]
        
        # compute ratio from 'ratio' or nearest-to-1.0 from 'ratios'
        score = float(row.get("score", np.nan))
        if pd.notna(row.get("ratio", np.nan)):
            r = float(row["ratio"])
        else:
            rats = row.get("ratios", None)
            r = (float(min(rats, key=lambda x: abs(float(x) - 1.0)))
                if isinstance(rats, (list, tuple)) and len(rats) else float("nan"))
        is_ext = 1.0 if (np.isfinite(r) and r > 1.0) else 0.0    # is_extension
        srsc = float(row.get("sr_score", np.nan))
        trsc = float(row.get("trend_score", np.nan))
        tolp = float(row.get("tol_pct", np.nan))
        prfd = float(row.get("prefer_ratio_dist", np.nan))

        return (float(d["abs"]), float(d["signed"]), float(d["nearest_level"]),
                r, score, is_ext, srsc, trsc, tolp, prfd)

    vals = close.apply(_nearest_tuple)
    abs_ser    = vals.apply(lambda t: t[0]).astype("float32")
    signed_ser = vals.apply(lambda t: t[1]).astype("float32")
    level_ser  = vals.apply(lambda t: t[2]).astype("float32")
    ratio_ser  = vals.apply(lambda t: t[3]).astype("float32")
    score_raw  = vals.apply(lambda t: t[4]).astype("float32")
    is_ext_ser = vals.apply(lambda t: t[5]).astype("float32")
    srsc_ser   = vals.apply(lambda t: t[6]).astype("float32")
    trsc_ser   = vals.apply(lambda t: t[7]).astype("float32")
    tolpct_ser = vals.apply(lambda t: t[8]).astype("float32")
    prf_ser    = vals.apply(lambda t: t[9]).astype("float32")


    #logger.info("sr_score/trend_score NaN ratio: %.3f / %.3f", 
    #            float(srsc_ser.isna().mean()), float(trsc_ser.isna().mean()))   # temp_logger

    # 6) نرمال‌سازی score به [0,1] بر اساس کلسترهای موجود
    if "score" in clusters.columns and clusters["score"].notna().any():
        sc = clusters["score"].astype("float32")
        mn, mx = float(np.nanmin(sc)), float(np.nanmax(sc))
        scale = (mx - mn) if np.isfinite(mx - mn) and (mx - mn) > 0 else np.nan
        def _norm(v):
            return (float(v) - mn) / scale if np.isfinite(scale) else np.nan
        score_norm = score_raw.apply(_norm).astype("float32")
    else:
        score_norm = score_raw * np.nan

    # 7) ATR-based normalization (اختیاری، اگر high/low موجود باشد)
    if (high is not None) and (low is not None):
        atrv = compute_atr(base[["high","low","close"]], window=int(atr_len))
        abs_atr = (abs_ser / atrv.replace(0, np.nan)).astype("float32")
    else:
        abs_atr = abs_ser * np.nan

    # 8) عبور دادن sr_score / trend_score در صورت وجود (ثابت بر کل سری، بر اساس نزدیک‌ترین کلستر هر بار)
    # توجه: چون برای هر بار «نزدیک‌ترین کلستر» متفاوت می‌شود، استفادهٔ مستقیم از clusters برای این دو ستون قابل اتکا نیست،
    # مگر این‌که این مقادیر را در همان حلقه استخراج کنیم. (در صورت نیاز بعداً اضافه می‌کنیم.)
    
    conf_sr_ser = (srsc_ser > 0).astype("float32")  # 1=confluence present, 0=absent
    ratio_dev1 = (ratio_ser - 1.0).abs().astype("float32")
    valid_mask = (
        abs_ser.notna()
        & signed_ser.notna()
        & level_ser.notna()
        & ratio_ser.notna()
        & score_norm.notna()
    ).astype("float32")

    out = {
        f"fibc_nearest_abs@{base_tf}": abs_ser,
        f"fibc_nearest_signed@{base_tf}": signed_ser,
        f"fibc_nearest_level@{base_tf}": level_ser,
        f"fibc_nearest_abs_atr@{base_tf}": abs_atr,
        f"fibc_nearest_ratio@{base_tf}": ratio_ser,
        f"fibc_is_extension@{base_tf}": is_ext_ser,
        f"fibc_score_norm@{base_tf}": score_norm,
        f"fibc_sr_score@{base_tf}": srsc_ser,
        f"fibc_trend_score@{base_tf}": trsc_ser,
        f"fibc_tol_pct@{base_tf}": tolpct_ser,
        f"fibc_prefer_ratio_dist@{base_tf}": prf_ser,
        f"fibc_sr_confluence@{base_tf}": conf_sr_ser,
        f"fibc_ratio_dev1@{base_tf}": ratio_dev1,
        f"fibc_valid@{base_tf}": valid_mask,
    }
    return out


