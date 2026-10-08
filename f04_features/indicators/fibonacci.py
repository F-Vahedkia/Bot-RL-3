# -*- coding: utf-8 -*-
# f04_features/indicators/fibonacci.py
# Status in (Bot-RL-3): Completed

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Sequence, Iterable
import logging
import numpy as np
import pandas as pd

from f04_features.indicators.core import rsi as rsi_core
from f04_features.indicators.levels import sr_overlap_score
# from f02_utils.config_loader import ConfigLoader
from f02_utils.config_operations import _deep_get
from f04_features.indicators.utils import (
    # levels_from_recent_legs,  # هِلپر «n لگ اخیر»
    compute_atr,              # برای last_leg_levels
    detect_swings,            # برای last_leg_levels
    # get_ohlc_view,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# -----------------------------------------------------------------------------
# نسبت‌های پیش‌فرض فیبوناچی
# ----------------------------------------------------------------------------- OK
cfg_all = ConfigLoader().get_all()

if cfg_all is not None:
    # _deep_get فقط dot-notation می‌پذیرد
    DEFAULT_RETR_RATIOS = _deep_get(cfg_all, "features.fibonacci.retracement_ratios")
    DEFAULT_EXT_RATIOS  = _deep_get(cfg_all, "features.fibonacci.extension_ratios")

# ----------------------------------------------------------------------------- func-01
# کمکی: ساخت سطوح فیبو برای یک لگ (low→high یا high→low)
# ----------------------------------------------------------------------------- in progress
def _fib_levels_for_leg(
    old_price: float,
    new_price: float, 
    ratios: Sequence[float] = DEFAULT_RETR_RATIOS
) -> pd.DataFrame:
 
    if new_price == old_price:
        raise ValueError("Invalid leg: new_price == old_price")
    leg_up = new_price > old_price
    levels = []
    for r in ratios:
        price = (new_price - r * (new_price - old_price))
        levels.append({"ratio": float(r), "price": float(price), "leg_up": bool(leg_up)})
    
    df = pd.DataFrame(levels)
    return df.sort_values("price").reset_index(drop=True)


# ----------------------------------------------------------------------------- func-02
# انتخاب آخرین لگ معتبر از روی swings (آخرین H در کنار L یا برعکس)
# -----------------------------------------------------------------------------
def _last_valid_leg(swings: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp, float, float]:
    """
    English: Find the most recent swing leg (H with a preceding L, or L with a preceding H).
    Persian: آخرین لگ معتبر را بر اساس جفت H/L مجاور برمی‌گرداند.
    """
    if swings is None or swings.empty:
        raise ValueError("swings is empty")

    # انتظار داریم ستون‌های: index=time(UTC), columns: ['price','kind' in {'H','L'}]
    s = swings.sort_index()
    kinds = s["kind"].values
    prices = s["price"].values
    times = s.index.values

    # از انتها عقب می‌رویم تا به جفت H..L یا L..H برسیم
    for i in range(len(s) - 1, 0, -1):
        k2 = kinds[i]
        k1 = kinds[i - 1]
        if {k1, k2} == {"H", "L"}:
            t1, p1 = pd.to_datetime(times[i - 1]), float(prices[i - 1])
            t2, p2 = pd.to_datetime(times[i]), float(prices[i])
            # جهت لگ: از t1→t2
            return t1, t2, p1, p2
    # اگر پیدا نشد، از دو آخرین نقطه استفاده می‌کنیم:
    t1, p1 = pd.to_datetime(times[-2]), float(prices[-2])
    t2, p2 = pd.to_datetime(times[-1]), float(prices[-1])
    return t1, t2, p1, p2


# ----------------------------------------------------------------------------- func-03 (Not Used)
# ???
# -----------------------------------------------------------------------------
''' از ATR و swings برای «آخرین لگ معتبر» سطوح رتریسمنت می‌سازد؛ کاملاً فیبو-محور است. '''
def last_leg_levels(
    ohlc_df: pd.DataFrame,
    prominence: float | None = None,
    min_distance: int = 5,
    atr_mult: float | None = 1.0,
    ratios: Iterable[float] = DEFAULT_RETR_RATIOS,
) -> pd.DataFrame:
    """ساخت سطوح رتریسمنت برای «آخرین لگ معتبر» از روی سوئینگ‌های بسته."""
    close = ohlc_df["close"].astype(float)
    atr = compute_atr(ohlc_df, window=14)
    swings = detect_swings(
        close, prominence=prominence, min_distance=min_distance,
        atr=atr, atr_mult=atr_mult, tf=None
    )
    if swings is None or swings.empty or len(swings) < 2:
        return pd.DataFrame(columns=["ratio", "price", "leg_up"])
    s = swings.sort_index()
    p1, p2 = float(s["price"].iloc[-2]), float(s["price"].iloc[-1])
    leg_up = p2 > p1
    low, high = (p1, p2) if leg_up else (p2, p1)

    rows = []
    rng = high - low
    for r in ratios:
        # قیمت رتریسمنت استاندارد: در لگ صعودی از بالا رو به پایین، در لگ نزولی بالعکس
        price = (high - r * rng) if leg_up else (low + r * rng)
        rows.append({"ratio": float(r), "price": float(price), "leg_up": bool(leg_up)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- func-04,05
# Advanced leg selection (filters + cap) and levels builder
# -----------------------------------------------------------------------------
'''
نکته‌ها:
1- این پچ رفتار هیچ تابع موجودی را نمی‌شکند؛
    فقط ابزار انتخاب لگ/تبدیل سطوح را «داخل همین ماژول» اضافه می‌کند.
2- مصرفش در وایرینگ: به‌جای ساخت سطوح از N لگ خام،
    اول select_legs_from_swings(...) را صدا بزن،
    سپس نتیجه را با levels_from_legs(...) به ورودی fib_cluster بده.

(مثال مصرف در وایرینگ فعلی‌ات نزدیک به همین منطق است:
ساخت لگ‌ها و تبدیل به levels
؛ فقط اینجا فیلترها و سقف تعداد، داخل ماژول شده‌اند.
'''
def select_legs_from_swings(
    swings: pd.DataFrame,
    *,
    max_legs: int = 3,
    min_prominence: float | None = None,   # حداقل برجستگی لگ بر حسب «قیمت»
    min_length_pct: float | None = None,   # حداقل طول لگ نسبت به قیمت مرجع (مثلاً 0.002 = 0.2%)
    max_age_bars: int | None = None,       # حداکثر فاصلهٔ لگ تا انتهای سری (بر حسب «ردیف/بار»)
    body_vs_wick: str = "auto",            # "auto" | "body" | "wick" (در این نسخه آموزشی، فقط auto)
) -> List[dict]:
    """
توضیح آموزشی (فارسی):
    این تابع از جدول سوئینگ‌ها (ایندکس زمانی، ستون‌های price/kind∈{H,L}) لگ‌های مجاور H↔L می‌سازد،
    سپس با فیلترهای سبک آن‌ها را محدود می‌کند و «تا سقف max_legs» برمی‌گرداند (جدید→قدیم).

    English:
      Build adjacent H↔L legs from swings DataFrame, apply light filters, and cap to `max_legs`.
    """
    if swings is None or swings.empty:
        return []

    s = swings.sort_index()
    kinds = s["kind"].values
    prices = s["price"].astype(float).values
    times = s.index

    legs: List[dict] = []
    # از انتها به ابتدا: به‌دنبال جفت‌های مجاور L..H یا H..L
    for i in range(len(s) - 1, 0, -1):
        k2, k1 = kinds[i], kinds[i - 1]
        if {k1, k2} != {"H", "L"}:
            continue

        t1, p1 = pd.to_datetime(times[i - 1], utc=True), float(prices[i - 1])
        t2, p2 = pd.to_datetime(times[i], utc=True), float(prices[i])
        leg_up = (p2 > p1)
        low, high = (p1, p2) if leg_up else (p2, p1)
        extent = abs(high - low)

        # فیلتر 1: حداقل برجستگی مطلق
        if min_prominence is not None and extent < float(min_prominence):
            continue

        # فیلتر 2: حداقل طول نسبی نسبت به قیمت مرجع (قیمت انتهایی لگ)
        ref_px = max(min(p1, p2), 1e-9)  # محافظت از تقسیم بر صفر
        if min_length_pct is not None and (extent / ref_px) < float(min_length_pct):
            continue

        # فیلتر 3: سن لگ بر حسب بار (اختیاری)
        if max_age_bars is not None:
            age_bars = (len(s) - 1) - i  # فاصلهٔ لگ از آخرین سطر
            if age_bars > int(max_age_bars):
                continue

        legs.append({
            "from_ts": t1, "to_ts": t2,
            "low": low, "high": high,
            "leg_up": bool(leg_up),
            "extent": extent,
        })
        if len(legs) >= int(max_legs):
            break

    return legs


def levels_from_legs(
    legs: List[dict],
    ratios: Sequence[float] = DEFAULT_RETR_RATIOS
    ) -> pd.DataFrame:
    # ۱) نسبت‌ها: اگر خالی/None بود، از مقدار پیش‌فرض استفاده کن
    ratios = ratios or DEFAULT_RETR_RATIOS

    out_rows: List[dict] = []
    for lg in legs or []:
        # ۲) استخراج ایمن و فیلتر لگ‌های نامعتبر
        try:
            low  = float(lg.get("low"))
            high = float(lg.get("high"))
            if not (low < high or high < low):  # برابر یا NaN → رد
                continue
        except Exception:
            continue
        leg_up = bool(lg.get("leg_up", high >= low))

        retr = _fib_levels_for_leg(low, high, ratios=ratios)
        for _, r in retr.iterrows():
            out_rows.append({
                "ratio": float(r["ratio"]),
                "price": float(r["price"]),
                "leg_up": bool(leg_up),
                # اطلاعات اختیاری برای دیباگ (اگر نبود، None):
                "from_ts": lg.get("from_ts"),
                "to_ts":   lg.get("to_ts"),
            })

    if not out_rows:
        return pd.DataFrame(columns=["ratio", "price", "leg_up"])
    return pd.DataFrame(out_rows).sort_values("price").reset_index(drop=True)

# ----------------------------------------------------------------------------- func-06
# Golden Zone (نسخهٔ کامل‌تر)
# -----------------------------------------------------------------------------
def golden_zone(
    swings: pd.DataFrame,
    ratios: Tuple[float, float] = (0.382, 0.618),
    extra_ratios: Sequence[float] = DEFAULT_RETR_RATIOS,
) -> pd.DataFrame:
    """
    English:
      Build Golden Zone from the last valid leg. Returns a one-row DataFrame:
      ['zone_low','zone_high','center','width','width_pct','leg_up','from_ts','to_ts','nearest_ratio'].
    Persian:
      ساخت ناحیهٔ گلدن‌زون از آخرین لگ معتبر؛ خروجی یک ردیف با ویژگی‌های ناحیه.
    """
    # توضیح: swings باید ایندکس زمانی UTC داشته باشد.
    t1, t2, p1, p2 = _last_valid_leg(swings)
    leg_up = p2 > p1
    low, high = (p1, p2) if leg_up else (p2, p1)

    # سطوح فیبو برای لگ
    retr = _fib_levels_for_leg(low, high, ratios=extra_ratios)
    r_low, r_high = float(min(ratios)), float(max(ratios))
    # استخراج قیمت‌های محدودهٔ گلدن‌زون
    gz_low  = float(retr.loc[(retr["ratio"] - r_low).abs().idxmin(), "price"])
    gz_high = float(retr.loc[(retr["ratio"] - r_high).abs().idxmin(), "price"])
    zone_low, zone_high = (gz_low, gz_high) if gz_low <= gz_high else (gz_high, gz_low)
    center = 0.5 * (zone_low + zone_high)
    width = zone_high - zone_low
    width_pct = 100.0 * (width / center) if center != 0 else np.nan

    # نزدیک‌ترین نسبت به مرکز (برای گزارش)
    retr["dist"] = (retr["price"] - center).abs()
    nearest_idx = retr["dist"].idxmin()
    nearest_ratio = float(retr.loc[nearest_idx, "ratio"])

    out = pd.DataFrame(
        [{
            "zone_low": zone_low,
            "zone_high": zone_high,
            "center": center,
            "width": width,
            "width_pct": width_pct,
            "leg_up": bool(leg_up),
            "from_ts": pd.to_datetime(t1, utc=True),
            "to_ts": pd.to_datetime(t2, utc=True),
            "nearest_ratio": nearest_ratio,
        }]
    ).set_index("to_ts")
    return out


# ----------------------------------------------------------------------------- func-07
# Cluster/Confluence بین چند تایم‌فریم
# -----------------------------------------------------------------------------
def fib_cluster(
    tf_levels: Dict[str, pd.DataFrame],
    tol_pct: float = 0.08,
    prefer_ratio: float = 0.618,
    tf_weights: Optional[Dict[str, float]] = None,
    # --- افزوده‌های وزن‌دهی اختیاری ---
    ma_slope: Optional[pd.Series] = None,         # سری شیب MA (مثلاً از M5/H1)
    rsi_zone_score: Optional[pd.Series] = None,   # سری امتیاز RSI (-1,0,1 → [0..1])
    sr_levels: Optional[Sequence[float]] = None,  # سطوح S/R برای همپوشانی
    ref_time: Optional[pd.Timestamp] = None,      # زمان مرجع برای خواندن سری‌ها
    w_trend: float = 10.0,                        # وزن مؤلفهٔ ترند
    w_rsi: float = 10.0,                          # وزن مؤلفهٔ RSI
    w_sr: float = 10.0,                           # وزن مؤلفهٔ S/R
    sr_tol_pct: float = 0.05,                     # تلورانس نسبی برای همپوشانی S/R
) -> pd.DataFrame:
    """
    خوشه‌بندی سطوح فیبو در چند تایم‌فریم + امتیاز Confluence با وزن‌دهی اختیاری MA/RSI/SR.

    ورودی:
      tf_levels: {TF → DataFrame['ratio','price','leg_up'(bool)]}
      tol_pct: آستانهٔ نزدیکی برای خوشه‌بندی (٪ نسبت به قیمت پایه)
      prefer_ratio: نسبت مرجح (نزدیک‌تر به این نسبت امتیاز بهتر)
      tf_weights: وزن پیش‌فرض هر TF (اگر None → استفاده از نگاشت داخلی)
      ma_slope/rsi_zone_score: سری‌های زمینه‌ای (در صورت ارائه، وزن‌دهی فعال می‌شود)
      sr_levels: لیست سطوح S/R (برای امتیاز همپوشانی قیمت خوشه)
      ref_time: زمان مرجع جهت خواندن مقدار آخر ≤ ref_time از سری‌ها
      w_*: وزن هر مؤلفه در امتیاز نهایی
    خروجی:
      DataFrame مرتب بر اساس 'score' با ستون‌های:
      ['price_mean','price_min','price_max','members','tfs','ratios','score']
    """
    # --- گردآوری سطوح ---
    rows = []
    for tf, df in tf_levels.items():
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({
                "tf": tf,
                "ratio": float(r["ratio"]),
                "price": float(r["price"]),
                "leg_up": bool(r.get("leg_up", True)),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "price_mean","price_min","price_max","members","tfs","ratios","score"
        ])

    all_levels = pd.DataFrame(rows).sort_values("price").reset_index(drop=True)

    # --- وزن‌های TF (پیش‌فرض) ---
    if tf_weights is None:
        tf_weights = {"M1": 1, "M5": 1.2, "M15": 1.4, "M30": 1.6, "H1": 1.8, "H4": 2.0, "D1": 2.3, "W1": 2.6, "MN1": 3.0}

    # --- کمکی: مقدار سری در ref_time یا آخرین مقدار ---
    def _at_or_before(s: Optional[pd.Series], ts: Optional[pd.Timestamp]) -> float:
        if s is None or len(s) == 0:
            return np.nan
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        if ts is None:
            return float(s.iloc[-1])
        ts = pd.to_datetime(ts, utc=True)
        s2 = s.loc[:ts]
        if len(s2) == 0:
            return np.nan
        return float(s2.iloc[-1])

    trend_val = _at_or_before(ma_slope, ref_time)            # ممکن است NaN باشد
    rsi_val   = _at_or_before(rsi_zone_score, ref_time)      # -1/0/1 در پیاده‌سازی پیشنهادی

    # نرمال‌سازی سادهٔ مؤلفه‌ها به [0..1]
    trend_score = float(np.tanh(abs(trend_val))) if not np.isnan(trend_val) else 0.0
    rsi_score   = float((rsi_val + 1.0) / 2.0) if not np.isnan(rsi_val) else 0.5  # -1→0, 0→0.5, 1→1

    clusters: List[Dict] = []
    used = np.zeros(len(all_levels), dtype=bool)

    i = 0
    while i < len(all_levels):
        if used[i]:
            i += 1
            continue

        base_price = all_levels.loc[i, "price"]
        tol_abs = base_price * (tol_pct / 100.0)
        members_idx = [i]
        used[i] = True

        j = i + 1
        while j < len(all_levels):
            if used[j]:
                j += 1
                continue
            if abs(all_levels.loc[j, "price"] - base_price) <= tol_abs:
                members_idx.append(j)
                used[j] = True
                j += 1
            else:
                break

        members = all_levels.loc[members_idx].copy()
        price_mean = float(members["price"].mean())
        price_min  = float(members["price"].min())
        price_max  = float(members["price"].max())
        tfs = list(members["tf"].unique())
        ratios = list(sorted(set([float(x) for x in members["ratio"].tolist()])))

        # --- امتیاز پایه (مانند نسخهٔ قبلی) ---
        tf_score = sum(tf_weights.get(tf, 1.0) for tf in tfs)
        ratio_closeness = min([abs(r - prefer_ratio) for r in ratios]) if ratios else 1.0
        ratio_score = 1.0 - min(1.0, ratio_closeness)  # نزدیک‌تر به 0.618 بهتر (۰..۱)
        tightness = price_max - price_min
        tight_score = 1.0 - float(tightness / (base_price * 0.01 + 1e-9))  # نرمال در ۱٪ قیمت
        tight_score = max(0.0, min(1.0, tight_score))

        base_score = 50.0 * (tf_score / (len(tf_levels) + 1e-9)) + 30.0 * ratio_score + 20.0 * tight_score

        # --- مؤلفهٔ SR (در صورت وجود sr_levels) ---
        sr_score = 0.0
        if sr_levels:
            sr_score = sr_overlap_score(price_mean, sr_levels, tol_pct=sr_tol_pct)  # ۰..۱

        # --- امتیاز نهایی با وزن‌دهی اختیاری ---
        score = base_score + (w_trend * trend_score) + (w_rsi * rsi_score) + (w_sr * sr_score)

        clusters.append({
            "price_mean": price_mean,
            "price_min": price_min,
            "price_max": price_max,
            "members": len(members_idx),
            "tfs": tfs,
            "ratios": ratios,
            "score": float(score),
            "sr_score": float(sr_score),
            "trend_score": float(trend_score),
        })
        i += 1

    out = pd.DataFrame(clusters).sort_values(["score","price_mean"], ascending=[False, True]).reset_index(drop=True)
    return out

'''
نحوهٔ مصرف (مثال)
اگر خواستی از طریق Engine صدا بزنی، چون fib_cluster به‌صورت «تجمیعی» است، آن را این‌طور Spec بده
 (بدون DF ورودی؛ Engine تلاش دوم را بدون df فراخوانی می‌کند):

fib_cluster(tf={'H1': df_h1_levels,'H4': df_h4_levels}, 
                tol_pct=0.1,
                ref_time='2025-08-29T08:13:00Z',
                ma_slope=ma_slope_series,
                rsi_zone_score=rsi_series,
                sr_levels=[...] )

(برای اجرای واقعی، معمولاً این فراخوانی را در کد پایتون می‌کنی نه CLI
؛ چون لازم است tf_levels و Series ها را پاس بدهی.)
'''
# ----------------------------------------------------------------------------- func-08
# سطوح Extension / تارگت + برآورد RR ساده
# -----------------------------------------------------------------------------
def fib_ext_targets(
    entry_price: float,
    leg_low: float,
    leg_high: float,
    side: str,
    ext_ratios: Sequence[float] = DEFAULT_EXT_RATIOS,
    sl_atr: Optional[float] = None,
    sl_atr_mult: float = 1.5,
) -> pd.DataFrame:
    """
    English:
      Build extension targets (e.g., 1.272/1.618/2.0) and a simple RR estimate.
      'side' in {'long','short'}. If sl_atr supplied, SL distance = sl_atr_mult * sl_atr.
    Persian:
      تولید تارگت‌های Extension و برآورد سادهٔ RR با SL مبتنی بر ATR.
    """
    if side not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'")
    leg_up = leg_high > leg_low
    rng = (leg_high - leg_low)
    if rng <= 0:
        raise ValueError("Invalid leg (high <= low)")

    # محاسبهٔ تارگت‌ها
    targets: List[Dict] = []
    for r in ext_ratios:
        if side == "long":
            price = leg_high + r * rng
            rr_den = (sl_atr_mult * sl_atr) if sl_atr else abs(entry_price - (leg_low if leg_up else leg_high))
            rr_num = abs(price - entry_price)
        else:
            price = leg_low - r * rng
            rr_den = (sl_atr_mult * sl_atr) if sl_atr else abs(entry_price - (leg_high if leg_up else leg_low))
            rr_num = abs(entry_price - price)
        rr = (rr_num / rr_den) if rr_den and rr_den > 0 else np.nan
        targets.append({"ratio": float(r), "price": float(price), "RR": float(rr)})

    df = pd.DataFrame(targets).sort_values("price").reset_index(drop=True)
    return df

'''
# === [PATCH] f04_features/indicators/fibonacci.py :: Config-aware wrappers === pach_1
# توجه: این بخش را در ابتدای فایل (پس از importهای موجود) اضافه کنید یا نزدیک بخش ثابت‌ها.
# هدف: افزودن خوانش اختیاری پارامترها از config.yaml «بدون تغییر» منطق توابع موجود.
#  - هیچ رفتار قبلی را نمی‌شکنیم؛ فقط توابع wrapper جدید می‌سازیم که از کانفیگ بخوانند.
#  - اگر کلیدهای کانفیگ موجود نباشند، به مقادیر پیش‌فرض فعلی برمی‌گردیم.
#  - پیام‌های اجرایی (log) انگلیسی هستند؛ توضیحات فارسی.

# ──────────────────────────────────────────────────────────────────────────────
# راهنمای ساختار کلیدهای کانفیگ (اختیاری):
# features:
#   fibonacci:
#     retracement_ratios: [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]
#     extension_ratios:   [1.272, 1.618, 2.0, 2.618, 3.618]
#     sl_atr_mult:        1.5
#     golden_zone:
#       ratios: [0.382, 0.618]
#     cluster:
#       tol_pct:       0.08
#       prefer_ratio:  0.618
#       tf_weights:    {"M15":0.5, "H1":1.0, "H4":1.5, "D1":2.0}
#       w_trend:       10.0
#       w_rsi:         10.0
#       w_sr:          10.0
#       sr_tol_pct:    0.05
#
# نکته: اگر هر کلیدی موجود نباشد، به پیش‌فرض‌های فعلی ماژول برمی‌گردیم.
# ──────────────────────────────────────────────────────────────────────────────
'''

# ----------------------------------------------------------------------------- func-09
#
# -----------------------------------------------------------------------------
def _load_fibo_cfg() -> Dict[str, Any]:
    """
    بارگذاری کل کانفیگ و استخراج زیربخش فیبوناچی.
    اگر لودر/فایل موجود نباشد یا خطایی رخ دهد، دیکشنری خالی بازمی‌گردد.
    """
    try:
        cfg_all: Dict[str, Any] = ConfigLoader().get_all()
    except Exception as e:
        # پیام انگلیسی برای log
        logger.debug("Failed to load config for Fibonacci wrappers: %s", e)
        return {}
    fib = _deep_get(cfg_all, "features.fibonacci", {}) or {}
    if not isinstance(fib, dict):
        fib = {}
    return fib


# ----------------------------------------------------------------------------- func-10 (Not Used)
# Wrapper 1: golden_zone_cfg  → خواندن پارامترها از config و فراخوانی golden_zone
# -----------------------------------------------------------------------------
def golden_zone_cfg(
    swings,  # pd.DataFrame
    ratios: Optional[Tuple[float, float]] = None,
    extra_ratios: Optional[Sequence[float]] = None,
):
    """
    نسخهٔ کانفیگ‌محور برای golden_zone:
      - اگر ratios/extra_ratios داده نشود، از config.yaml خوانده می‌شود.
      - در صورت نبود کلیدها در کانفیگ، از پیش‌فرض‌های اصلی ماژول استفاده می‌گردد.

    ورودی‌ها:
        swings: دیتافریم سوئینگ‌ها
        ratios: نسبت‌های بازگشتی اصلی زون طلایی (اختیاری؛ اگر None → از کانفیگ/پیش‌فرض)
        extra_ratios: سایر نسبت‌های رتریسمنت برای محاسبات تکمیلی (اختیاری)

    خروجی:
        pd.DataFrame مطابق خروجی تابع اصلی golden_zone
    """
    fib_cfg = _load_fibo_cfg()

    # نسبت‌های زون طلایی: اولویت با آرگومان تابع → سپس کانفیگ → سپس پیش‌فرض فعلی
    gz_ratios = ratios
    if gz_ratios is None:
        cfg_gz = fib_cfg.get("golden_zone", {}) if isinstance(fib_cfg, dict) else {}
        if isinstance(cfg_gz, dict):
            cfg_r = cfg_gz.get("ratios")
            if isinstance(cfg_r, (list, tuple)) and len(cfg_r) == 2:
                gz_ratios = (float(cfg_r[0]), float(cfg_r[1]))

    if gz_ratios is None:
        # fallback به پیش‌فرض تابع اصلی (0.382, 0.618)
        gz_ratios = (0.382, 0.618)

    # نسبت‌های رتریسمنت تکمیلی
    extra = extra_ratios
    if extra is None:
        cfg_extra = fib_cfg.get("retracement_ratios")
        if isinstance(cfg_extra, (list, tuple)) and cfg_extra:
            extra = tuple(float(x) for x in cfg_extra)

    if extra is None:
        # fallback به ثابت‌های فعلی ماژول (DEFAULT_RETR_RATIOS)
        try:
            extra = DEFAULT_RETR_RATIOS  # noqa: F821  (در همین فایل تعریف شده)
        except NameError:
            extra = (0.236, 0.382, 0.5, 0.618, 0.786)

    logger.info("Golden zone (config-driven) → ratios=%s, extra=%s", gz_ratios, extra)
    return golden_zone(swings=swings, ratios=gz_ratios, extra_ratios=extra)  # noqa: F821


# ----------------------------------------------------------------------------- func-11,12
# Adaptive tolerance (ATR/ADR) + config-aware wrapper for fib_cluster
# -----------------------------------------------------------------------------
'''
نحوهٔ مصرف در وایرینگ (فقط جایگزینی یک خط):
در check_wiring_fib_cluster.py هر جا fib_cluster(...) را مستقیم صدا می‌زنی،
برای حالت کانفیگ-محور کافی است به‌جای آن fibonacci.fib_cluster_cfg(...) را صدا بزنی تا
tol به‌شکل انطباقی/فیکس محاسبه و پاس شود (باقی پارامترها همانند قبل):
'''
def _adaptive_tol_pct_from_df(
    df: pd.DataFrame,
    *,
    mode: str = "FIXED",      # "FIXED" | "ATR" | "ADR"
    fixed_pct: float = 0.08,  # درصد (مثلاً 0.08 = 0.08%)
    atr_mult: float = 1.0,
    adr_mult: float = 1.0,
    ref_price: Optional[float] = None,
) -> float:
    """
    توضیح آموزشی (فارسی):
      این تابع تلورانس نسبی خوشه‌بندی را بر اساس نوسان محاسبه می‌کند.
      اگر ستون موردنیاز (ATR/ADR) در df نباشد، به مقدار ثابت (fixed_pct) برمی‌گردیم.

    English:
      Compute clustering tolerance (% of price) using ATR/ADR adaptively.
      Falls back to fixed_pct if needed columns are missing.
    """
    try:
        mode_u = str(mode or "FIXED").upper()
        # ref price: آخرین close اگر مقدار ندادیم
        if ref_price is None:
            if "close" in df.columns and len(df["close"]) > 0:
                ref_price = float(df["close"].iloc[-1])
            else:
                return float(fixed_pct)

        ref_price = max(float(ref_price), 1e-9)

        if mode_u == "ATR":
            if "ATR" in df.columns and df["ATR"].dropna().size:
                atr_val = float(df["ATR"].dropna().iloc[-1])
                tol = (atr_val / ref_price) * 1.000 * float(atr_mult)
                return float(tol)
            return float(fixed_pct)

        if mode_u == "ADR":
            # ADR = Average Daily Range (به واحد قیمت)
            for col in ("ADR", "adr", "__ADR", "__adr"):
                if col in df.columns and df[col].dropna().size:
                    adr_val = float(df[col].dropna().iloc[-1])
                    tol = (adr_val / ref_price) * 1.000 * float(adr_mult)
                    return float(tol)
            return float(fixed_pct)

        # FIXED
        return float(fixed_pct)
    except Exception:
        # ایمنی در برابر هر خطای پیش‌بینی‌نشده
        return float(fixed_pct)


def fib_cluster_cfg(
    tf_levels: Dict[str, pd.DataFrame],
    *,
    # می‌توانی tol_pct را override کنی؛ اگر None باشد از config/df محاسبه می‌شود
    tol_pct: Optional[float] = None,
    prefer_ratio: Optional[float] = None,
    tf_weights: Optional[Dict[str, float]] = None,
    ma_slope: Optional[pd.Series] = None,
    rsi_zone_score: Optional[pd.Series] = None,
    sr_levels: Optional[Sequence[float]] = None,
    ref_time: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    توضیح آموزشی (فارسی):
      این wrapper پارامترهای خوشه‌بندی را از config.yaml می‌خواند و در صورت
      خالی بودن tol_pct، آن را به‌صورت انطباقی از داده محاسبه می‌کند (ATR/ADR→٪).
      اگر ATR/ADR در دسترس نباشد، با تلورانس ثابت کار می‌کند.

    English:
      Config-aware wrapper for fib_cluster. Computes adaptive tol_pct (ATR/ADR)
      with safe fallback to fixed percentage.
    """
    fib_cfg = _load_fibo_cfg() if '_load_fibo_cfg' in globals() else {}
    clus_cfg = fib_cfg.get("cluster", {}) if isinstance(fib_cfg, dict) else {}

    # خواندن پارامترها از config (در نبود، به پیش‌فرض‌ها می‌افتد)
    prefer_ratio_ = float(clus_cfg.get("prefer_ratio", 0.618)) if prefer_ratio is None else float(prefer_ratio)
    tf_weights_   = clus_cfg.get("tf_weights", None) if tf_weights is None else tf_weights
    w_trend_      = float(clus_cfg.get("w_trend", 10.0))
    w_rsi_        = float(clus_cfg.get("w_rsi", 10.0))
    w_sr_         = float(clus_cfg.get("w_sr", 10.0))
    sr_tol_pct_   = float(clus_cfg.get("sr_tol_pct", 0.05))

    # tol: اگر آرگومان ندادی، از کانفیگ/داده محاسبه می‌کنیم
    if tol_pct is None:
        mode      = str(clus_cfg.get("tol_mode", "FIXED")).upper()
        fixed_pct = float(clus_cfg.get("tol_pct", 0.08))     # درصد
        atr_mult  = float(clus_cfg.get("atr_mult", 1.0))
        adr_mult  = float(clus_cfg.get("adr_mult", 1.0))

        # برای محاسبه tol به یک df مرجع نیاز داریم؛ بهترین گزینه TF با بیشترین اعضای سطح
        ref_df: Optional[pd.DataFrame] = None
        for tf, df in (tf_levels or {}).items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                # انتظار داریم این df ها فقط «سطوح» باشند و ستون close نداشته باشند؛
                # بنابراین اگر close نداشتند، tol را بر اساس FIXED می‌گذاریم.
                ref_df = df
                break

        # اگر df سطوح close ندارد، تلاش کن یک df کانتکست بیرونی پاس دهی (در wiring)
        # اینجا به‌صورت ایمن اگر ref_df مناسب نبود، به fixed می‌افتیم:
        if (ref_df is None) or ("close" not in ref_df.columns):
            tol_pct_ = fixed_pct
        else:
            tol_pct_ = _adaptive_tol_pct_from_df(
                ref_df, mode=mode, fixed_pct=fixed_pct,
                atr_mult=atr_mult, adr_mult=adr_mult,
                ref_price=None
            )
    else:
        tol_pct_ = float(tol_pct)

    # فراخوانی هستهٔ اصلی
    #========start======================================================================
    # [PATCH: reporting metadata] افزودن دو ستون گزارشی به خروجی fib_cluster_cfg
    # - tol_pct: همان تلورانس نهایی که در این تابع resolve/محاسبه شده (tol_pct_)
    # - prefer_ratio_dist: کمینه فاصله |ratio - prefer_ratio_| از روی ستون 'ratios' خروجی هسته
    #   * این منطق صرفاً گزارشی است و تغییری در محاسبات هسته ایجاد نمی‌کند.
    #   * پیام‌های runtime باید انگلیسی باشند (در صورت نیاز به log).
    #========end========================================================================

    result_df = fib_cluster(
        tf_levels=tf_levels,
        tol_pct=tol_pct_,
        prefer_ratio=prefer_ratio_,
        tf_weights=tf_weights_,
        ma_slope=ma_slope,
        rsi_zone_score=rsi_zone_score,
        sr_levels=sr_levels,
        ref_time=ref_time,
        w_trend=w_trend_,
        w_rsi=w_rsi_,
        w_sr=w_sr_,
        sr_tol_pct=sr_tol_pct_,
    )

    #========start======================================================================
    # در صورت وجود خروجی، ستون‌های tol_pct و prefer_ratio_dist را اضافه کن
    #========end========================================================================
    if result_df is not None and not result_df.empty:
        #======start==============================================================
        # tol_pct: تلورانس نهایی که در همین تابع resolve شده است
        #======end================================================================
        try:
            result_df = result_df.copy()
            result_df["tol_pct"] = float(tol_pct_)
        except Exception:
            # Runtime message in English (silent fallback)
            # print("[FIB] Failed to attach tol_pct to cluster output.")
            pass

        #======start==============================================================
        # prefer_ratio_dist: کمینه فاصله نسبت‌های هر ردیف تا prefer_ratio_
        #   - ستون 'ratios' در خروجی هسته موجود است (لیست نسبت‌ها)
        #======end================================================================
        try:
            _pr = float(prefer_ratio_)

            def _min_pref_dist(ratios):
                #-----start------------------------------------------------------
                # محافظت در برابر ورودی‌های غیرلیست/لیست خالی/مقادیر غیرعددی
                #-----end--------------------------------------------------------
                try:
                    if not isinstance(ratios, (list, tuple)) or len(ratios) == 0:
                        return float("nan")
                    return min(abs(float(r) - _pr) for r in ratios)
                except Exception:
                    return float("nan")

            result_df["prefer_ratio_dist"] = result_df["ratios"].apply(_min_pref_dist)
        except Exception:
            # Runtime message in English (silent fallback)
            # print("[FIB] Failed to compute prefer_ratio_dist; column set to NaN.")
            result_df["prefer_ratio_dist"] = float("nan")

    return result_df


# ────────────────────────────────────────────────────────────────────────────── Pach_2
# Adaptive tol_pct for fib_cluster (Option 1)
# - اگر در کانفیگ فعال باشد و ورودی لازم را داشته باشیم، tol_pct بر اساس ADR یا ATR محاسبه می‌شود.
# - اگر ورودی کافی نبود، به مقدار ثابت (config/پیش‌فرض) برمی‌گردیم.
# - هیچ تغییری در تابع اصلی fib_cluster نمی‌دهیم؛ فقط رَپر fib_cluster_cfg را هوشمند می‌کنیم.
# ──────────────────────────────────────────────────────────────────────────────

# ----------------------------------------------------------------------------- func-13,14 (Not Used)
#
# -----------------------------------------------------------------------------
# این تابع فقط در فایل registry.py فراخوانی شده است
def _infer_ref_price_from_tf_levels(
    tf_levels: Dict[str, "pd.DataFrame"],  # type: ignore[name-defined]
    ref_time: Optional["pd.Timestamp"] = None  # type: ignore[name-defined]
) -> Optional[float]:
    """
    تلاش ابتدایی برای حدس ref_price از روی tf_levels:
      - tf_levels معمولاً خروجی سطوح است، نه قیمت. بنابراین اغلب ref_price از اینجا قابل‌استخراج نیست.
      - این تابع برای سازگاری آینده گذاشته شده؛ اگر دیتافریم‌ها ستونی مانند 'close' داشته باشند، از آخرین مقدار استفاده می‌کند.
      - در غیر این صورت None برمی‌گرداند و رَپر از مقدار ثابت tol_pct استفاده می‌کند.

    توجه: این حدس‌زدن اختیاری است و اگر به نتیجه نرسیم، رفتار قبلی حفظ می‌شود.
    """
    try:
        for _, df in tf_levels.items():
            if hasattr(df, "columns"):
                cols = [c.lower() for c in df.columns]
                if "close" in cols:
                    # اگر ref_time داریم، نزدیک‌ترین مقدار قبل یا در ref_time را برداشت می‌کنیم
                    if hasattr(df, "index") and df.index.size > 0 and ref_time is not None:
                        # فیلتر تا ref_time
                        subset = df.loc[df.index <= ref_time]
                        if subset.shape[0] > 0:
                            return float(subset["close"].iloc[-1])
                    # در غیر این صورت آخرین مقدار
                    return float(df["close"].iloc[-1])
    except Exception as e:
        logger.debug("Could not infer ref price from tf_levels: %s", e)
    return None

# این تابع فقط در فایل registry.py فراخوانی شده است
def _compute_adaptive_tol_pct(
    ref_price: Optional[float],
    vol_value: Optional[float],
    k: float,
    min_pct: float,
    max_pct: float
) -> Optional[float]:
    """
    محاسبه tol_pct انطباقی:
      tol_pct = clip( (vol_value / ref_price) * k , [min_pct, max_pct] )

    اگر هرکدام از ref_price یا vol_value نداشتیم، None برمی‌گردانیم.
    """
    if ref_price is None or vol_value is None:
        return None
    try:
        raw = (float(vol_value) / float(ref_price)) * float(k)
        # محدودسازی در بازهٔ [min_pct, max_pct]
        return max(min(raw, max_pct), min_pct)
    except Exception as e:
        logger.debug("Adaptive tol_pct computation failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────── added 040612

# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# ====== ساختار پارامترهای پیکربندی فیبو ======
@dataclass
class FiboParams:
    retracement_ratios: List[float]
    extension_ratios: List[float]
    prefer_ratio: float
    tol_mode: str
    tol_fixed_pct: float
    tol_atr_mult: float
    tol_adr_mult: float
    leg_selection: Dict[str, Any]
    w_base: float
    w_sr: float
    w_ma: float
    w_rsi: float
    rsi_len: int
    rsi_ob: float
    rsi_os: float


# ----------------------------------------------------------------------------- func-15,16
#
# -----------------------------------------------------------------------------
def _merge_overrides(base: Dict[str, Any], symbol: str, tf: str) -> Dict[str, Any]:
    """ترکیب تنظیمات پایه با override های per-TF و per-symbol (اولویت با نماد، بعد TF)."""
    cfg = dict(base)
    sym_ovr = (base.get("symbol_overrides") or {}).get(symbol) or {}
    tf_ovr = (base.get("tf_overrides") or {}).get(tf) or {}
    # نماد بر TF اولویت دارد؛ سپس روی cfg اعمال می‌کنیم
    def deep_update(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deep_update(dst[k], v)
            else:
                dst[k] = v
    deep_update(cfg, tf_ovr)
    deep_update(cfg, sym_ovr)
    return cfg


def _load_fibo_params(global_cfg: Dict[str, Any], symbol: str, tf: str) -> FiboParams:
    """خواندن پارامترهای فیبو از config با درنظرگرفتن overrideهای per-symbol و per-TF."""
    fibo_cfg = global_cfg.get("features", {}).get("fibonacci", {}) or {}
    fibo_cfg = _merge_overrides(fibo_cfg, symbol, tf)

    ratios_r = fibo_cfg.get("retracement_ratios", [0.236, 0.382, 0.5, 0.618, 0.786, 0.886])
    ratios_e = fibo_cfg.get("extension_ratios", [1.272, 1.618, 2.0, 2.618, 3.618])
    prefer = fibo_cfg.get("prefer_ratio", 0.618)

    tol_cfg = fibo_cfg.get("tol", {}) or {}
    tol_mode = str(tol_cfg.get("mode", "ADR")).upper()
    tol_fixed = float(tol_cfg.get("fixed_pct", 0.0025))
    tol_atr = float(tol_cfg.get("atr_mult", 1.0))
    tol_adr = float(tol_cfg.get("adr_mult", 1.0))

    leg_sel = fibo_cfg.get("leg_selection", {}) or {}
    w_cfg = fibo_cfg.get("confluence_weights", {}) or {}
    rsi_cfg = fibo_cfg.get("rsi", {}) or {}

    return FiboParams(
        retracement_ratios=ratios_r,
        extension_ratios=ratios_e,
        prefer_ratio=prefer,
        tol_mode=tol_mode,
        tol_fixed_pct=tol_fixed,
        tol_atr_mult=tol_atr,
        tol_adr_mult=tol_adr,
        leg_selection=leg_sel,
        w_base=float(w_cfg.get("base", 1.0)),
        w_sr=float(w_cfg.get("sr_overlap", 0.5)),
        w_ma=float(w_cfg.get("ma_slope", 0.25)),
        w_rsi=float(w_cfg.get("rsi_zone", 0.25)),
        rsi_len=int(rsi_cfg.get("length", 14)),
        rsi_ob=float(rsi_cfg.get("overbought", 70)),
        rsi_os=float(rsi_cfg.get("oversold", 30)),
    )



# ----------------------------------------------------------------------------- func-17,18 (Not Used)
#
# -----------------------------------------------------------------------------
# ====== ابزارهای کمکی RSI و tol انطباقی ======
# این تابع فقط در فایل registry.py فراخوانی شده است
def _adaptive_tol_pct(df: pd.DataFrame, params: FiboParams, adr_col: str = "ADR") -> float:
    """محاسبهٔ tol درصدی بر اساس حالت انتخاب شده: FIXED/ATR/ADR."""
    if params.tol_mode == "FIXED":
        return params.tol_fixed_pct
    if params.tol_mode == "ATR":
        atr = df.get("ATR")
        if atr is None or atr.dropna().empty:
            logger.warning("ATR not found for adaptive tolerance; fallback to fixed.")
            return params.tol_fixed_pct
        px = df["close"]
        tol = (atr.iloc[-1] / px.iloc[-1]) * params.tol_atr_mult
        return float(tol)
    # ADR (پیش‌فرض)
    adr = df.get(adr_col)
    if adr is None or adr.dropna().empty:
        logger.warning("ADR not found for adaptive tolerance; fallback to fixed.")
        return params.tol_fixed_pct
    px = df["close"]
    tol = (adr.iloc[-1] / px.iloc[-1]) * params.tol_adr_mult
    return float(tol)


# ====== امتیازدهی RSI zone برای خوشه‌ها ======
# این تابع فقط در فایل registry.py فراخوانی شده است
def _rsi_zone_score(df: pd.DataFrame, params: FiboParams) -> float:
    """
    امتیاز RSI zone: اگر RSI نزدیک overbought/oversold باشد، بازگشتی/ادامه‌دار بودن را تقویت/تضعیف می‌کند.
    خروجی در [0..1] نرمالیزه می‌شود.
    """
    rsi = rsi_core(df["close"], params.rsi_len).iloc[-1]
    if rsi >= params.rsi_ob:
        return 1.0  # Near overbought
    if rsi <= params.rsi_os:
        return 1.0  # Near oversold
    # خطی بین OS..OB
    span = params.rsi_ob - params.rsi_os
    if span <= 0:
        return 0.5
    return 1.0 - abs((rsi - (params.rsi_os + span/2)) / (span/2))

# ----------------------------------------------------------------------------- func-19
#
# -----------------------------------------------------------------------------
def fib_ext_targets_cfg(last_leg: Tuple[float, float],
                        global_cfg: Dict[str, Any],
                        symbol: str,
                        tf: str) -> pd.DataFrame:
    """
    تولید تارگت‌های Extension با نسبت‌های پیکربندی‌شده و خروجی مناسب Order Planner.
    """
    params = _load_fibo_params(global_cfg, symbol, tf)
    lo, hi = last_leg
    targets = []
    for r in params.extension_ratios:
        # مثال ساده: تارگت‌های ادامه روند صعودی
        tgt = hi + r * (hi - lo)
        targets.append({"tf": tf, "ratio": r, "target_price": tgt})
    df = pd.DataFrame(targets)
    logger.info(f"[FIB] Generated {len(df)} extension targets for {symbol}/{tf}.")
    return df


# ----------------------------------------------------------------------------- Main-20
# Main Body
# -----------------------------------------------------------------------------
__all__ = ["fib_cluster_cfg", "fib_ext_targets_cfg"]



# =====================================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# ===================================================================================== 05/02/13
""" Func Names                           Used in Functions: ...
                                    1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
1  _fib_levels_for_leg             --  --  --  --  ok  ok  --  --  --  --  --  --  --  --  --  --  --  --  --  --
2  _last_valid_leg                 --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --  --
3 
4  select_legs_from_swings         --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- USED in feature_registry, fibo_pipeline
5  levels_from_legs                --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- USED in feature_registry, fibo_pipeline
6  golden_zone                     --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --
7  fib_cluster                     --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --
8  fib_ext_targets                 --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- USED in feature_registry
9  _load_fibo_cfg                  --  --  --  --  --  --  --  --  --  ok  --  ok  --  --  --  --  --  --  --  --
10 golden_zone_cfg                 --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- (Not Used)
11 _adaptive_tol_pct_from_df       --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --
12 fib_cluster_cfg                 --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok USED in fibo_pipeline
13 _infer_ref_price_from_tf_levels --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- (Not Used)
14 _compute_adaptive_tol_pct       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- (Not Used)
15 _merge_overrides                --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --
16 _load_fibo_params               --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
17 _adaptive_tol_pct               --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- (Not Used)
18 _rsi_zone_score                 --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- (Not Used)
19 fib_ext_targets_cfg             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok USED in feature_registry
20 Main Body                       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
------[]
"""
