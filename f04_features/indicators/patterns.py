# -*- coding: utf-8 -*-
# f04_features/indicators/patterns.py
# Status in (Bot-RL-3): Reviewed at 1405/01/12
"""
===============================================================================
🕯 الگوهای کندلی — نسخهٔ نهایی و کامنت‌گذاری‌شده (FINAL)
-------------------------------------------------------------------------------
- هدف: تولید فلگ‌های کندلی (int8) برای بهره‌برداری در Engine/Registry
- اصول:
  • خروجی‌ها همگی int8 هستند (کاهش حافظه و سازگاری با بقیهٔ پروژه)
  • ضد-لوک‌اِهد: در الگوهای چندکندلی با shift() رعایت شده
  • نرمال‌سازی مبتنی بر ATR/TR برای پایداری در رِژیم‌های نوسانی مختلف
  • نام‌گذاری ستون‌ها با پیشوند pat_* (سازگار با Engine/Registry)
- نکتهٔ مهم: هر الگو «یک نسخه» دارد و آستانه‌ها Adaptive هستند؛
  این یعنی بدون انفجار ستون‌ها، برای self-optimize فقط ضرایب کم‌تعدادی (k) تیون می‌شوند.
===============================================================================
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
# پیام‌های runtime اگر لازم شد: انگلیسی نگه می‌داریم (فعلاً لاگ خاموش است)

from f04_features.indicators.utils import compute_atr

# =============================================================================
# [General helpers] هِلپرهای عمومی برای بدنه/رنج/شدُو
# =============================================================================
def _apply_scale(kwargs: dict, rules: list[tuple[str, float, float, float]]) -> dict:
    """مشتق‌گیری پارامترها از scale_k در صورت نبودِ خودِ پارامترها؛ بدون افزودن کلید جدید به کانفیگ."""
    sk = kwargs.get("scale_k", None)
    if sk is None:
        return kwargs
    out = dict(kwargs)
    for name, factor, lo, hi in rules:
        if name not in out or out[name] is None:
            val = factor * sk
            if lo is not None:  val = max(lo, val)
            if hi is not None:  val = min(hi, val)
            out[name] = val
    return out

def _fmtf(x: float, nd: int = 2) -> str:
    """نمایش عدد اعشاری به‌صورت کوتاه (trim zeros)، برای نام ستون‌ها."""
    s = f"{float(x):.{nd}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"

def _body(open_: pd.Series, close: pd.Series) -> pd.Series:
    """بدنهٔ signed: مثبت (سبز) وقتی close>open و منفی (قرمز) وقتی close<open."""
    return (close - open_).astype("float32")

def _abs_body(open_: pd.Series, close: pd.Series) -> pd.Series:
    """بدنهٔ مطلق (قدر مطلق اختلاف close و open)."""
    return (close - open_).abs().astype("float32")

def _range(high: pd.Series, low: pd.Series) -> pd.Series:
    """رنج کندل (high-low)."""
    return (high - low).astype("float32")

def _body_wicks(open_, high, low, close) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    محاسبهٔ بدنه و شدوهای بالا/پایین برای هر کندل.
    - upper: فاصلهٔ high تا max(open, close)
    - lower: فاصلهٔ min(open, close) تا low
    """
    body_abs = (close - open_).abs().astype("float32")
    upper = (high - close.where(close > open_, open_)).astype("float32")
    lower = (close.where(close < open_, open_) - low).astype("float32")
    return body_abs, upper, lower

# =============================================================================
# [Basic patterns] الگوهای پایه‌ای (با بهبودهای پایدار)
# =============================================================================
def engulfing_flags(open_, high, low, close) -> Tuple[pd.Series, pd.Series]:
    """
    Engulfing ساده (بدون فیلتر ATR):
    - bull: کندل فعلی صعودی و بدنه‌اش بدنهٔ قبلی را می‌بلعد
    - bear: کندل فعلی نزولی و بدنه‌اش بدنهٔ قبلی را می‌بلعد
    نکتهٔ ضد-لوک‌اِهد: فقط از داده‌های کندل قبلی با shift(1) استفاده می‌شود.
    """
    body = close - open_
    bull = (body.shift(1) < 0) & (close > open_.shift(1)) & (open_ < close.shift(1))
    bear = (body.shift(1) > 0) & (close < open_.shift(1)) & (open_ > close.shift(1))
    return bull.fillna(False).astype("int8"), bear.fillna(False).astype("int8")

def doji_flag(open_, close,
              atr: pd.Series | None = None,
              atr_ratio_thresh: float = 0.1,
              range_ratio_thresh: float = 0.2) -> pd.Series:
    """
    Doji: بدنهٔ بسیار کوچک.
    - معیار 1 (پیشنهادی): |body| / ATR <= atr_ratio_thresh  (ATR بیرونی ترجیح دارد)
    - معیار 2 (کمکی):    |body| / range <= range_ratio_thresh
    دلیل: استفاده از ATR باعث می‌شود در رِژیم‌های کم‌/پرنوسان، کیفیت پایدار بماند.
    """
    body_abs = _abs_body(open_, close)
    # اگر ATR داده نشد، از متوسط بدنهٔ 20 تایی به‌عنوان fallback استفاده می‌کنیم
    fallback = (close - open_).abs().rolling(20, min_periods=5).mean()
    atrv = atr if atr is not None else fallback
    rng_candle = (close - open_).abs().replace(0, np.nan)  # می‌توانست high-low باشد؛ ساده نگه داشتیم

    cond1 = (body_abs / atrv.replace(0, np.nan) <= atr_ratio_thresh).fillna(False)
    cond2 = (body_abs / rng_candle <= range_ratio_thresh).fillna(False)
    return (cond1 & cond2).astype("int8")

def pinbar_flags(open_, high, low, close, ratio: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """
    Pinbar تقریبی:
    - شدوی بلند در یک سمت (>= ratio * body) و شدوی سمت مقابل کوچک (<= body)
    - bull: شدوی پایین بلند؛ bear: شدوی بالا بلند
    """
    body_abs, upper, lower = _body_wicks(open_, high, low, close)
    bull = (lower >= ratio * body_abs) & (upper <= body_abs)
    bear = (upper >= ratio * body_abs) & (lower <= body_abs)
    return bull.fillna(False).astype("int8"), bear.fillna(False).astype("int8")

# =============================================================================
# [Advanced patterns] الگوهای پیشرفته (Adaptive + Anti-Lookahead)
# =============================================================================
def hammer_shooting_flags(open_, high, low, close,
                          min_body_frac: float = 0.0,
                          wick_ratio: float = 2.0,
                          opp_wick_k: float = 1.25) -> Tuple[pd.Series, pd.Series]:
    """
    Hammer (bull) / Shooting Star (bear)
    - شدوی اصلی >= wick_ratio * body
    - شدوی مخالف <= opp_wick_k * body  (نرم‌تر از «<= body» برای افزایش ریکال)
    - بدنه حداقل min_body_frac از رنج باشد (اختیاری)
    نکته: اگر ریکال پایین شد، opp_wick_k کمی افزایش یابد (1.25 → 1.5).
    """
    body_abs, upper, lower = _body_wicks(open_, high, low, close)
    rng = _range(high, low).replace(0, np.nan)
    body_ok = ((body_abs / rng) >= min_body_frac).fillna(False)

    bull = (lower >= wick_ratio * body_abs) & (upper <= opp_wick_k * body_abs) & body_ok
    bear = (upper >= wick_ratio * body_abs) & (lower <= opp_wick_k * body_abs) & body_ok
    return bull.fillna(False).astype("int8"), bear.fillna(False).astype("int8")

def harami_flags(open_, high, low, close) -> Tuple[pd.Series, pd.Series]:
    """
    Harami (دو کندلی):
    - بدنهٔ کندل جاری داخل بدنهٔ کندل قبلی است.
    - جهت براساس رنگ دو کندل تعیین می‌شود (bull/bear).
    ضد-لوک‌اِهد: فقط از دادهٔ کندل قبلی با shift(1) استفاده می‌شود.
    """
    body = close - open_
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    prev_low = prev_open.where(prev_open < prev_close, prev_close)
    prev_high = prev_open.where(prev_open > prev_close, prev_close)
    cur_low = open_.where(open_ < close, close)
    cur_high = open_.where(open_ > close, close)
    inside = (cur_low >= prev_low) & (cur_high <= prev_high)
    bull = (body > 0) & ((prev_close - prev_open) < 0) & inside
    bear = (body < 0) & ((prev_close - prev_open) > 0) & inside
    return bull.fillna(False).astype("int8"), bear.fillna(False).astype("int8")

def inside_outside_flags(open_, high, low, close,
                         min_range_k_atr: float = 0.0,
                         atr_win: int = 14) -> Tuple[pd.Series, pd.Series]:
    """
    Inside/Outside با قید «حداقل رنج نسبت به ATR» (برای کاهش نویز):
    - inside: رنج کندل جاری داخل رنج کندل قبل
    - outside: رنج کندل جاری رنج کندل قبل را می‌بلعد
    - اگر min_range_k_atr>0: فقط در صورتی فلگ بده که (high-low) >= k*ATR
    """
    prev_high = high.shift(1)
    prev_low  = low.shift(1)
    inside = (high <= prev_high) & (low >= prev_low)
    outside = (high >= prev_high) & (low <= prev_low)

    if min_range_k_atr > 0.0:
        _df  = pd.DataFrame({"high": high, "low": low, "close": close})
        atrv = compute_atr(_df, atr_win, method="classic").replace(0, np.nan) # 0 -> NaN
        rng  = (high - low).astype("float32")
        ok   = (rng >= (min_range_k_atr * atrv)).fillna(False)
        inside  = (inside  & ok)
        outside = (outside & ok)

    return inside.astype("int8"), outside.astype("int8")

def marubozu_flags(open_, high, low, close, wick_frac: float = 0.1) -> Tuple[pd.Series, pd.Series]:
    """
    Marubozu:
    - بدنهٔ قوی با شدوهای بسیار کوچک نسبت به رنج ( (upper+lower)/range <= wick_frac )
    """
    body_abs, upper, lower = _body_wicks(open_, high, low, close)
    rng = _range(high, low).replace(0, np.nan)
    small_wicks = ((upper + lower) / rng <= wick_frac).fillna(False)
    bull = (close > open_) & small_wicks
    bear = (close < open_) & small_wicks
    return bull.fillna(False).astype("int8"), bear.fillna(False).astype("int8")

def tweezer_flags(high, low,
                  tol_frac: float | None = 0.001,
                  tol_k: float | None = None,
                  tol_mode: str = "atr_price",
                  atr_win: int = 14,
                  close: pd.Series | None = None) -> Tuple[pd.Series, pd.Series]:
    """
    Tweezer Top/Bottom (دو کندل با سقف/کف تقریباً برابر)
    دو حالت:
      1) ثابت: tol_frac مقدار دارد → تلورانس نسبی ثابت (ساده و سریع)
      2) Adaptive: tol_frac=None و tol_k مقدار دارد →
         tol_mode="atr_price": tol = k * ATR / mid_price
         (مقیاس‌پذیر با ولتیلیتی و سطح قیمت، self-optimize پسند)
    """
    prev_high = high.shift(1)
    prev_low  = low.shift(1)

    if tol_frac is None and tol_k is not None and tol_mode == "atr_price":
        _close = (close if close is not None else low)
        _df    = pd.DataFrame({"high": high, "low": low, "close": _close})
        atrv   = compute_atr(_df, atr_win, method="classic").replace(0, np.nan) # 0 -> NaN
        mid_price = ((high + low) / 2.0).replace(0, np.nan)
        tol = (tol_k * (atrv / mid_price)).astype("float32")
        top    = ((high - prev_high).abs() <= (tol * prev_high.abs())).fillna(False)
        bottom = ((low  - prev_low ).abs() <= (tol * prev_low.abs())).fillna(False)
    else:
        tol = (tol_frac if tol_frac is not None else 0.001)
        top    = ((high - prev_high).abs() <= (tol * prev_high.abs())).fillna(False)
        bottom = ((low  - prev_low ).abs() <= (tol * prev_low.abs())).fillna(False)

    return top.astype("int8"), bottom.astype("int8")

def three_soldiers_crows_flags(open_, close,
                               atr_ref: pd.Series | None = None,
                               min_body_atr: float = 0.2) -> Tuple[pd.Series, pd.Series]:
    """
    Three White Soldiers / Three Black Crows (سه کندل متوالی قوی در یک جهت)
    - معیار قدرت: |body| / ATR >= min_body_atr
    - ضد-لوک‌اِهد: برای سه‌تایی‌ها از shift استفاده شده (بدون نشت آینده)
    """
    body = _body(open_, close)
    if atr_ref is None:
        # fallback: اگر ATR بیرونی پاس نشده، از میانگین قدر مطلق بدنه‌ها استفاده می‌کنیم
        atr_ref = body.abs().rolling(20, min_periods=5).mean().replace(0, np.nan)
    z = (body.abs() / atr_ref.replace(0, np.nan))
    strong_up = (body > 0) & (z >= min_body_atr)
    strong_dn = (body < 0) & (z >= min_body_atr)
    bull = (strong_up & strong_up.shift(1) & strong_up.shift(2)).fillna(False)
    bear = (strong_dn & strong_dn.shift(1) & strong_dn.shift(2)).fillna(False)
    return bull.astype("int8"), bear.astype("int8")

def morning_evening_star_flags(open_, high, low, close,
                               small_body_atr: float = 0.3,
                               atr_win: int = 14) -> Tuple[pd.Series, pd.Series]:
    """
    Morning/Evening Star (سه‌ کندلی):
    - کندل 1 قوی، کندل 2 بدنهٔ کوچک (نسبت به ATR)، کندل 3 بازگشت قوی و عبور از نیمهٔ کندل 1
    - الزام گپ لحاظ نشده (سازگاری با FX)
    """
    _df  = pd.DataFrame({"high": high, "low": low, "close": close})
    atrv = compute_atr(_df, atr_win, method="classic").replace(0, np.nan) # 0 -> NaN
    body = _body(open_, close)
    b1 = body.shift(2)  # کندل 1 (قدیمی‌ترین در سه‌تایی)
    b2 = body.shift(1)  # کندل 2
    b3 = body           # کندل 3 (جدیدترین)
    small2 = (b2.abs() / atrv <= small_body_atr)
    morning = (b1 < 0) & small2 & (b3 > 0) & (close > (open_.shift(2) + close.shift(2)) / 2.0)
    evening = (b1 > 0) & small2 & (b3 < 0) & (close < (open_.shift(2) + close.shift(2)) / 2.0)
    return morning.fillna(False).astype("int8"), evening.fillna(False).astype("int8")

def piercing_darkcloud_flags(open_, close,
                             min_body_ratio: float = 0.2) -> Tuple[pd.Series, pd.Series]:
    """
    Piercing / Dark Cloud (دو کندلی):
    - علاوه بر نفوذ تا نیمهٔ بدنهٔ قبلی، بدنه‌های دو کندل حداقل آستانهٔ نسبتی داشته باشند
    - دلیل: حذف سیگنال‌های ضعیف در رنج‌های خیلی کوچک/پرنوسان
    """
    o1, c1 = open_.shift(1), close.shift(1)
    o2, c2 = open_, close
    mid1 = (o1 + c1) / 2.0

    body1 = (c1 - o1).abs()
    body2 = (c2 - o2).abs()
    # fallback ساده برای مقیاس: میانگین بدنه‌های اخیر
    body_mean20 = (close - open_).abs().rolling(20, min_periods=5).mean().replace(0, np.nan)
    b1_ok = (body1 / body_mean20 >= min_body_ratio)
    b2_ok = (body2 / body_mean20 >= min_body_ratio)

    piercing  = (c1 < o1) & (c2 > o2) & (c2 >= mid1) & (o2 < c1) & b1_ok & b2_ok
    darkcloud = (c1 > o1) & (c2 < o2) & (c2 <= mid1) & (o2 > c1) & b1_ok & b2_ok
    return piercing.fillna(False).astype("int8"), darkcloud.fillna(False).astype("int8")

def belt_hold_flags(open_, high, low, close,
                    wick_frac: float = 0.1) -> Tuple[pd.Series, pd.Series]:
    """
    Belt Hold (bull/bear):
    - باز شدن نزدیک کف/سقف (شدوی متناظر بسیار کوچک)
    - بسته‌شدن در جهت روند کندل
    """
    body_abs, up, lo = _body_wicks(open_, high, low, close)
    rng = _range(high, low).replace(0, np.nan)
    near_low_open  = ((open_ - low)  / rng <= wick_frac).fillna(False)
    near_high_open = ((high - open_) / rng <= wick_frac).fillna(False)
    bull = near_low_open & (close > open_)
    bear = near_high_open & (close < open_)
    return bull.astype("int8"), bear.astype("int8")

# =============================================================================
# [Registry] API سازگار با Engine — هر کلید یک «سازندهٔ ستون‌ها» برمی‌گرداند
# =============================================================================
def registry() -> Dict[str, callable]:
    """
    خروجی: دیکشنری از نام Spec → سازنده (callable)
    هر سازنده: dict[str -> Series["float32"]] برمی‌گرداند.
    """

    # ---------- Basics ---------- 1~3
    def make_engulf(df, **_):
        """Engulfing (bull/bear)"""
        b, s = engulfing_flags(df["open"], df["high"], df["low"], df["close"])
        return {"pat_engulf_bull": b.astype("float32"),
                "pat_engulf_bear": s.astype("float32")}

    def make_doji(df,
                  atr_ratio_thresh: float = 0.1,
                  range_ratio_thresh: float = 0.2,
                  atr_win: int = 14,
                  **_):
        """Doji با قیدهای ATR و نسبت به رنج"""
        atrv = compute_atr(df, atr_win, method="classic")
        flag = doji_flag(df["open"], df["close"], atr=atrv,
                         atr_ratio_thresh=atr_ratio_thresh,
                         range_ratio_thresh=range_ratio_thresh)
        return {f"pat_doji_{atr_ratio_thresh}_{range_ratio_thresh}": flag.astype("float32")}

    def make_pin(df, ratio: float = 2.0, **kwargs):
        kwargs = _apply_scale(kwargs, rules=[("ratio", 2.50, 1.10, 10.0)])
        """Pinbar (bull/bear)"""
        b, s = pinbar_flags(df["open"], df["high"], df["low"], df["close"], ratio)
        r = _fmtf(ratio, 2)   # r: ratio
        return {f"pat_pin_bull_{r}": b.astype("float32"),
                f"pat_pin_bear_{r}": s.astype("float32")}
    
    # ---------- Advanceds ---------- 4~12
    def make_hammer_star(df,
                         min_body_frac: float = 0.0,
                         wick_ratio: float = 2.0,
                         opp_wick_k: float = 1.25,
                         **kwargs):
        kwargs = _apply_scale(kwargs, rules=[("wick_ratio", 2.50, 0.50, 10.0)])
        """Hammer/Shooting با پارامترهای adaptive ساده"""
        b, s = hammer_shooting_flags(df["open"], df["high"], df["low"], df["close"],
                                     min_body_frac, wick_ratio, opp_wick_k)
        w = _fmtf(wick_ratio, 2)   # w: wick_ratio
        return {f"pat_hammer_bull_{w}": b.astype("float32"), 
                f"pat_shoot_bear_{w}": s.astype("float32")}

    def make_harami(df, **_):
        """Harami (bull/bear)"""
        b, s = harami_flags(df["open"], df["high"], df["low"], df["close"])
        return {"pat_harami_bull": b.astype("float32"),
                "pat_harami_bear": s.astype("float32")}

    def make_inside_outside(df, min_range_k_atr: float = 0.0, atr_win: int = 14, **kwargs):
        """Inside/Outside با قید حداقل رنج نسبتی به ATR"""
        kwargs = _apply_scale(kwargs, rules=[("min_range_k_atr", 0.50, 0.05, 10.0)])
        inside, outside = inside_outside_flags(df["open"], df["high"], df["low"], df["close"],
                                               min_range_k_atr=min_range_k_atr, atr_win=atr_win)
        return {"pat_inside": inside.astype("float32"),
                "pat_outside": outside.astype("float32")}

    def make_marubozu(df, wick_frac: float = 0.1, **_):
        """Marubozu (bull/bear)"""
        b, s = marubozu_flags(df["open"], df["high"], df["low"], df["close"], wick_frac)
        wf = _fmtf(wick_frac, 2)   # wf: wick_frac
        return {f"pat_marubozu_bull_{wf}": b.astype("float32"),
                f"pat_marubozu_bear_{wf}": s.astype("float32")}
    
    def make_tweezer(df, tol_frac: float | None = 0.001,
                    tol_k: float | None = None,
                    tol_mode: str = "atr_price",
                    atr_win: int = 14, **kwargs):
        # اگر فقط scale_k داده شده باشد و tol_k نیامده باشد، از آن مشتق می‌کنیم
        kwargs = _apply_scale(kwargs, rules=[("tol_k", 1.00, 0.05, 10.0)])
        """
        Tweezer (Adaptive/Static)
        - اگر tol_frac=None → adaptive با tol_k
        - در نام ستون، برای شفافیت برچسب k یا tol_frac درج می‌شود
        """
        top, bottom = tweezer_flags(df["high"], df["low"],
                                    tol_frac=tol_frac, tol_k=tol_k,
                                    tol_mode=tol_mode, atr_win=atr_win,
                                    close=df["close"])
        # اگر tol_k باشد با پیشوند k و اعشار کنترل‌شده؛ در غیر این صورت tol_frac با اعشار کنترل‌شده
        label = ('k' + _fmtf(tol_k, 2)) if tol_frac is None else _fmtf(tol_frac, 3)        
        return {f"pat_tweezer_top_{label}": top.astype("float32"),
                f"pat_tweezer_bot_{label}": bottom.astype("float32")}

    def make_soldiers_crows(df, min_body_atr: float = 0.2, atr_win: int = 14, **kwargs):
        """Three Soldiers / Three Crows (ATR-based)"""
        kwargs = _apply_scale(kwargs, rules=[("min_body_atr", 0.25, 0.02, 10.0)])
        atrv = compute_atr(df, atr_win, method="classic")
        b, s = three_soldiers_crows_flags(df["open"], df["close"],
                                          atr_ref=atrv, min_body_atr=min_body_atr)
        return {f"pat_3soldiers_{min_body_atr}": b.astype("float32"),
                f"pat_3crows_{min_body_atr}": s.astype("float32")}

    def make_morn_even(df, small_body_atr: float = 0.3, atr_win: int = 14, **kwargs):
        """Morning/Evening Star (ATR-based)"""
        kwargs = _apply_scale(kwargs, rules=[("small_body_atr", 0.30, 0.02, 10.0)])
        m, e = morning_evening_star_flags(df["open"], df["high"], df["low"], df["close"],
                                          small_body_atr=small_body_atr, atr_win=atr_win)
        sb = _fmtf(small_body_atr, 2)   # sb: small_body_atr
        return {f"pat_morning_{sb}": m.astype("float32"),
                f"pat_evening_{sb}": e.astype("float32")}

    def make_piercing_dark(df, min_body_ratio: float = 0.2, **_):
        """Piercing/Dark Cloud با حداقل بدنهٔ نسبتی (پایداری بهتر)"""
        p, d = piercing_darkcloud_flags(df["open"], df["close"], min_body_ratio=min_body_ratio)
        return {f"pat_piercing_{min_body_ratio}": p.astype("float32"),
                f"pat_darkcloud_{min_body_ratio}": d.astype("float32")}

    def make_belt(df, wick_frac: float = 0.1, **kwargs):
        # نرخ بالا ⇒ frac کمتر؛ مشتق اولیه از scale_k
        """Belt Hold (bull/bear)"""
        kwargs = _apply_scale(kwargs, rules=[("wick_frac", 0.10, 0.005, 0.50)])
        b, s = belt_hold_flags(df["open"], df["high"], df["low"], df["close"], wick_frac)
        wf = _fmtf(wick_frac, 2)   # wf: wick_frac
        return {f"pat_belt_bull_{wf}": b.astype("float32"),
                f"pat_belt_bear_{wf}": s.astype("float32")}
    
    # دیکشنری نهایی:
    return {
        "pat_engulf": make_engulf,
        "pat_doji": make_doji,
        "pat_pin": make_pin,
        "pat_hammer_star": make_hammer_star,
        "pat_harami": make_harami,
        "pat_inside_outside": make_inside_outside,
        "pat_marubozu": make_marubozu,
        "pat_tweezer": make_tweezer,
        "pat_3soldiers_crows": make_soldiers_crows,
        "pat_morning_evening": make_morn_even,
        "pat_piercing_dark": make_piercing_dark,
        "pat_belt": make_belt,
    }

def detect_ab_equal_cd(swings: pd.DataFrame, ratio_tol: float = 0.05) -> Optional[Dict[str, Any]]:
    """
    English:
      Detect a simple AB=CD pattern from the last four consecutive swing points.
      If 'bar' column is missing, it will be created based on time order.
    Persian:
      تشخیص سادهٔ الگوی AB=CD از چهار نقطهٔ متوالی آخر سوئینگ.
      اگر ستون 'bar' وجود نداشته باشد، با ترتیب زمانی ساخته می‌شود.

    Args:
        swings: DataFrame با ایندکس زمانی UTC و ستون‌های حداقل ['price','kind'] (kind ∈ {'H','L'})
        ratio_tol: تلورانس نسبی برای برابری طول AB و CD (پیش‌فرض 0.05 = 5%)

    Returns:
        dict شامل اطلاعات الگو در صورت موفقیت؛ در غیر این صورت None.
    """
    if swings is None or len(swings) < 4:
        return None

    # نرمال‌سازی حداقلی اسکیمای ورودی
    s = swings.copy()
    # اطمینان از وجود ستون‌های پایه
    if "price" not in s.columns or "kind" not in s.columns:
        logger.warning("[ABCD] swings lacks required columns ['price','kind']")
        return None

    # مرتب‌سازی زمانی و ساخت bar در صورت نبود
    s = s.sort_index()
    if "bar" not in s.columns:
        s = s.reset_index().rename(columns={"index": "time"})  # اگر نام دیگری باشد، pandas خودش index می‌سازد
        if "time" not in s.columns:
            # fallback: بساز از ایندکس فعلی
            s["time"] = pd.to_datetime(s.index, utc=True)
        s["time"] = pd.to_datetime(s["time"], utc=True, errors="coerce")
        s["bar"] = np.arange(len(s), dtype=int)
        s = s.set_index("bar")
    else:
        s = s.sort_values("bar").reset_index(drop=False).set_index("bar")

    # باید حداقل 4 نقطه داشته باشیم
    if len(s) < 4:
        return None

    # آخرین چهار نقطه (A,B,C,D)
    A = s.iloc[-4]
    B = s.iloc[-3]
    C = s.iloc[-2]
    D = s.iloc[-1]

    # طول‌ها
    ab = abs(float(B["price"]) - float(A["price"]))
    cd = abs(float(D["price"]) - float(C["price"]))
    if ab <= 0:
        return None

    err = abs(cd / ab - 1.0)
    if err <= float(ratio_tol):
        out = {
            "pattern": "AB=CD",
            "error": float(err),
            "points": {
                "A": {"price": float(A["price"]), "kind": str(A["kind"])},
                "B": {"price": float(B["price"]), "kind": str(B["kind"])},
                "C": {"price": float(C["price"]), "kind": str(C["kind"])},
                "D": {"price": float(D["price"]), "kind": str(D["kind"])},
            },
        }
        logger.info("[ABCD] pattern detected with error=%.4f", err)
        return out

    logger.info("[ABCD] no pattern within tolerance (err=%.4f, tol=%.4f)", err, ratio_tol)
    return None

def abc_projection_adapter_from_abcd(abcd: dict):
    if not abcd: 
        return None
    A,B,C,D = (abcd["points"][k]["price"] for k in ("A","B","C","D"))
    length_AB = abs(B - A)
    proj_ratio = (abs(D - C) / length_AB) if length_AB else None
    return {"A":A,"B":B,"C":C,"D_real":D,"length_AB":length_AB,"proj_ratio":proj_ratio,"err_pct":abcd.get("error")}


# =====================================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# ===================================================================================== 05/02/09
""" Func Names                           Used in Functions: ...
                                1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21
------[General helpers]
1  _apply_scale                --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
2  _fmtf                       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
3  _body                       --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  ok  --  --  --  --  --
4  _abs_body                   --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --
5  _range                      --  --  --  --  --  --  --  --  --  ok  --  --  ok  --  --  --  --  ok  --  --  --
6  _body_wicks                 --  --  --  --  --  --  --  --  ok  ok  --  --  ok  --  --  --  --  ok  --  --  --
------[Basic patterns]
7  engulfing_flags             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
8  doji_flag                   --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
9  pinbar_flags                --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
------[Advanced patterns]
10 hammer_shooting_flags       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
11 harami_flags                --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
12 inside_outside_flags        --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
13 marubozu_flags              --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
14 tweezer_flags               --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
15 three_soldiers_crows_flags  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
16 morning_evening_star_flags  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
17 piercing_darkcloud_flags    --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
18 belt_hold_flags             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
------[Registry]
19 registry                    --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- USED in feature_registry

20 detect_ab_equal_cd          --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- USED in fibo_pipeline
21 abc_projection_adapter_from_abcd    --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- USED in fibo_pipeline
"""
