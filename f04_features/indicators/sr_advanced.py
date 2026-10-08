# -*- coding: utf-8 -*-
# f04_features/indicators/sr_advanced.py
# Status in (Bot-RL-3): ==>> In progress

from __future__ import annotations
import numpy as np
import pandas as pd
# pd.set_option('future.no_silent_downcasting', True)
import logging
from numba import njit
from typing import Dict
from .core import atr as atr_core

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#============================================================================== MainPart-1
# Fair Value Gap FVG (3-bar)
# تشخیص + lifecycle + انتشار + امتیاز
"""
تشخیص Fair Value Gap (FVG) بدون نگاه به آینده (look-ahead) و قابل‌استفاده در Engine/Registry

قواعد:
- تعریف کلاسیک 3-کندله:
  * FVG صعودی (bull): low[i] > high[i - lookback]  (به‌طور پیش‌فرض lookback=2)
  * FVG نزولی (bear): high[i] < low[i - lookback]
- آستانهٔ حداقل اندازهٔ گپ نسبت به ATR برای حذف نویزهای کوچک
- ضد لوک‌اِهد: فلگ‌ها با shift(+1) اعمال می‌شوند.
- خروجی استاندارد: سری‌های فلگ و محدودهٔ زون‌ها برای مصرف در Engine/Registry

نکات:
- همهٔ خروجی‌ها per-bar و float32/int8 هستند.
- در این گام، «زون‌های زنده/پرشدن زون/عمر زون» به‌صورت ساده ارائه می‌شود (ایجاد اولیهٔ زون).
"""
#==============================================================================

#-------------------------------------------------------------------- ok
def detect_fvg_legacy(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    *,
    lookback: int = 2,
    atr_window: int = 14,
    min_size_pct_of_atr: float = 0.50,
    use_middle_filter: bool = True,
    debug_mode: bool = False,
) -> Dict[str, pd.Series]:
    """
    تشخیص FVG به‌صورت سری‌های پرچم و بازهٔ قیمتی زون.

    پارامترها:
    - lookback: فاصلهٔ مرجع 3-کندله (به‌طور کلاسیک 2 → مقایسهٔ کندل i با i-2)
    - atr_window: طول ATR برای سنجش حداقل اندازهٔ گپ
    - min_size_pct_of_atr: حداقل نسبت اندازهٔ گپ به ATR (مثلاً 0.5 یعنی 50٪ ATR)

    خروجی:
    - fvg_up:   پرچم FVG صعودی (int8) با shift(+1) برای ضد-لوک‌اِهد
    - fvg_down: پرچم FVG نزولی (int8) با shift(+1)
    - fvg_top:  سقف زون FVG (float32) — فقط روی کندلِ ایجاد مقدار دارد، در غیر این‌صورت NaN
    - fvg_bottom: کف زون FVG (float32) — فقط روی کندلِ ایجاد مقدار دارد، در غیر این‌صورت NaN
    """
    # --- checks ---------------------------------------------------- ok
    if not (len(open_) == len(high) == len(low) == len(close)):
        raise ValueError("Input series must have equal length.")

    lookback = int(lookback)
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    # --- ATR برای سنجش حداقل اندازهٔ معتبر ------------------------- ok
    atrv = atr_core(high, low, close, n=int(atr_window)).astype("float32")

    # --- شرایط خام FVG (سه-کندلی) ---------------------------------- ok
    # Bullish: low[i] > high[i - lookback]
    bull_raw = (low > high.shift(lookback))
    if debug_mode: print(f"== 1 ==> len(bull_raw)={len(bull_raw)}")   ### for debug

    # Bearish: high[i] < low[i - lookback]
    bear_raw = (high < low.shift(lookback))

    # --- اندازهٔ گپ (فقط مثبت تعریف شود) --------------------------- ok
    # برای bull: فاصلهٔ low[i] تا high[i-lookback]
    bull_gap = (low - high.shift(lookback)).clip(lower=0.0)
    if debug_mode:  print(f"== 2 ==> len(bull_gap)={len(bull_gap)}")   ### for debug

    # برای bear: فاصلهٔ low[i-lookback] تا high[i]
    bear_gap = (low.shift(lookback) - high).clip(lower=0.0)

    # --- آستانهٔ اندازه برحسب ATR ---------------------------------- ok
    thr = (atrv * float(min_size_pct_of_atr)).astype("float32")
    if debug_mode:  print(f"== 3 ==> len(thr)={len(thr)}")             ### for debug

    bull_ok = bull_raw & (bull_gap >= thr)
    bear_ok = bear_raw & (bear_gap >= thr)
    if debug_mode:  print(f"== 4 ==> len(bull_ok)={len(bull_ok)}")     ### for debug

    ################################################################# Added 05/01/13
    if debug_mode:
        mid_body = []     ### for debug
        filter_mask = []  ### for debug

    if use_middle_filter:
        # --- Middle Candle Body Expansion Filter -------------------
        # طبق تعریف استاندارد fvg، کندل میانی عبارت است از کندل قبل از کندل جاری
        # بنابراین برای بدنه کندل میانی کافی است که از شیفت 1 استفاده کنیم
        mid_body = (close.shift(1) - open_.shift(1)).abs().astype("float32")
        if debug_mode:  print(f"== 5 ==> len(mid_body)={len(mid_body)}")   ### for debug

        # اعمال فیلتر
        filter_mask = (mid_body >= thr)
        if debug_mode:  print(f"== 6 ==> len(filter_mask)={len(filter_mask)}")   ### for debug
        bull_ok &= filter_mask
        bear_ok &= filter_mask
    ################################################################# Added 05/01/13
    if debug_mode:  print(f"== 7 ==> len(bull_ok)={len(bull_ok)}")   # for debug
    
    # --- محدودهٔ زون ----------------------------------------------- ok
    # bull: کف زون = high[i-lookback]، سقف زون = low[i]
    fvg_top_bull = low                 .where(bull_ok).astype("float32")
    fvg_bot_bull = high.shift(lookback).where(bull_ok).astype("float32")

    # bear: سقف زون = low[i-lookback]، کف زون = high[i]
    fvg_top_bear = low.shift(lookback).where(bear_ok).astype("float32")
    fvg_bot_bear = high               .where(bear_ok).astype("float32")

    # --- پرچم‌های نهایی با ضد-لوک‌اِهد ----------------------------- ok
    fvg_up   = bull_ok.astype("int8")
    fvg_down = bear_ok.astype("int8")

    # --- تجمیع زون‌ها (در کندل ایجاد؛ در غیر اینصورت NaN) ---------- ok
    # اگر هر دو رخ دهد (نادر)، اولویت‌بندی ساده: bear بر bull یا برعکس.
    # اینجا bull را مقدم می‌گیریم مگر bear هم‌زمان باشد؛ می‌توان سیاست را در آینده تنظیم کرد.
    fvg_top    = fvg_top_bull.where(bull_ok, fvg_top_bear)
    fvg_bottom = fvg_bot_bull.where(bull_ok, fvg_bot_bear)

    # --- فقط روی کندل ایجاد مقدار می‌خواهیم؛ (در بقیهٔ کندل‌ها NaN) --- ok
    fvg_top    = fvg_top   .where(bull_ok | bear_ok)
    fvg_bottom = fvg_bottom.where(bull_ok | bear_ok)

    if debug_mode:
        return {
            "open" : open_,                           # for debug
            "high" : high ,                           # for debug
            "low"  : low  ,                           # for debug
            "close": close,                           # for debug
            "bull_raw": bull_raw,                     # for debug
            "bear_raw": bear_raw,                     # for debug
            "bull_gap": bull_gap.astype("float32"),   # for debug
            "bear_gap": bear_gap.astype("float32"),   # for debug
            "thr": thr.astype("float32"),             # for debug
            "mid_body": mid_body.astype("float32"),   # for debug
            "filter_mask": filter_mask.astype(bool),  # for debug
            "bull_ok": bull_ok,                       # for debug
            "bear_ok": bear_ok,                       # for debug

            "fvg_up":         fvg_up.astype("int8"   ).shift(1),
            "fvg_dwnn":     fvg_down.astype("int8"   ).shift(1),
            "fvg_top":       fvg_top.astype("float32").shift(1),
            "fvg_bottom": fvg_bottom.astype("float32").shift(1),
        }
    else:
        return {
            "fvg_up":         fvg_up.astype("int8"   ).shift(1),
            "fvg_down":     fvg_down.astype("int8"   ).shift(1),
            "fvg_top":       fvg_top.astype("float32").shift(1),
            "fvg_bottom": fvg_bottom.astype("float32").shift(1),
        }

#-------------------------------------------------------------------- ok
def detect_fvg_optimized(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    *,
    lookback: int = 2,
    atr_window: int = 14,
    min_size_pct_of_atr: float = 0.50,
    use_middle_filter: bool = True,
    debug_mode: bool = False,
) -> Dict[str, pd.Series]:
    """
    تشخیص FVG به‌صورت سری‌های پرچم و بازهٔ قیمتی زون.

    پارامترها:
    - lookback: فاصلهٔ مرجع 3-کندله (به‌طور کلاسیک 2 → مقایسهٔ کندل i با i-2)
    - atr_window: طول ATR برای سنجش حداقل اندازهٔ گپ
    - min_size_pct_of_atr: حداقل نسبت اندازهٔ گپ به ATR (مثلاً 0.5 یعنی 50٪ ATR)

    خروجی:
    - fvg_up:   پرچم FVG صعودی (int8) با shift(+1) برای ضد-لوک‌اِهد
    - fvg_down: پرچم FVG نزولی (int8) با shift(+1)
    - fvg_top:  سقف زون FVG (float32) — فقط روی کندلِ ایجاد مقدار دارد، در غیر این‌صورت NaN
    - fvg_bottom: کف زون FVG (float32) — فقط روی کندلِ ایجاد مقدار دارد، در غیر این‌صورت NaN
    """
    # 1. تبدیل به آرایه numpy برای سرعت حداکثری
    o, h, l, c = open_.values, high.values, low.values, close.values
    
    # --- checks ----------------------------------------------------
    if not (len(o) == len(h) == len(l) == len(c)):
        raise ValueError("Input series must have equal length.")

    lookback = int(lookback)
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    # --- ATR برای سنجش حداقل اندازهٔ معتبر -------------------------
    atrv = atr_core(high, low, close, n=int(atr_window)).values.astype("float32")
    
    length = len(o)
    if debug_mode: print(f"=====> len(o)={len(o)}")   ### for debug

    thr = atrv * float(min_size_pct_of_atr)
    
    # استفاده از View برای جلوگیری از کپی (جایگزین shift)
    # کندل وسط = i - 1
    # i-lookback index:
    idx = lookback
    
    # 2. محاسبات منطقی با آرایه های خام
    bull_raw = (l[idx:length] > h[0:length-idx])
    bear_raw = (h[idx:length] < l[0:length-idx])
    if debug_mode: print(f"== 1 ==> len(bull_raw)={len(bull_raw)}")   ### for debug

    bull_gap = np.maximum(0, l[idx:] - h[:-idx])
    bear_gap = np.maximum(0, l[:-idx] - h[idx:])
    if debug_mode: print(f"== 2 ==> len(bull_gap)={len(bull_gap)}")   ### for debug

    bull_ok = bull_raw & (bull_gap >= thr[idx:])
    bear_ok = bear_raw & (bear_gap >= thr[idx:])
    if debug_mode: 
        print(f"== 3 ==> len(thr)={len(thr)}")           ### for debug
        print(f"== 4 ==> len(bull_ok)={len(bull_ok)}")   ### for debug

    #################################################################
    if debug_mode: 
        mid_body_f    = np.zeros(length, dtype="float32")      ### for debug
        filter_mask_f = np.zeros(length, dtype="bool")         ### for debug

    if use_middle_filter:
        mid_idx = lookback - 1
    # old:
        # mid_body = np.abs(c[mid_idx:-1] - o[mid_idx:-1])
        # اعمال فیلتر بر روی نتیجه (با توجه به طولِ هم‌تراز)
    # new:    
        mid_body = np.abs(
            c[mid_idx: mid_idx + len(bull_ok)] - o[mid_idx: mid_idx + len(bull_ok)]
        )
        if debug_mode: print(f"== 5 ==> len(mid_body)={len(mid_body)}")         ### for debug

        filter_mask = (mid_body >= thr[idx:])
        if debug_mode: print(f"== 6 ==> len(filter_mask)={len(filter_mask)}")   ### for debug
        bull_ok &= filter_mask
        bear_ok &= filter_mask
    #################################################################
    if debug_mode: print(f"== 7 ==> len(bull_ok)={len(bull_ok)}")                 ### for debug

    # 3. ساخت خروجی ها (ترکیب با NaN)
    # خروجی‌ها را با طول اصلی و پر از NaN می‌سازیم
    fvg_up     = np.zeros(length, dtype="int8")
    fvg_down   = np.zeros(length, dtype="int8")
    fvg_top    = np.full(length, np.nan, dtype="float32")
    fvg_bottom = np.full(length, np.nan, dtype="float32")
    
    # 4. نگاشت داده‌ها به موقعیت اصلی (offset)
    # توجه: نتیجه در موقعیت `idx` (همان i) ثبت می‌شود
    fvg_up  [idx:] = bull_ok.astype("int8")
    fvg_down[idx:] = bear_ok.astype("int8")
    
    top_slice = fvg_top   [idx:]
    bot_slice = fvg_bottom[idx:]

    top_slice[bull_ok] = l[idx:][bull_ok]
    top_slice[bear_ok] = l[:-idx][bear_ok]

    bot_slice[bull_ok] = h[:-idx][bull_ok]
    bot_slice[bear_ok] = h[idx:][bear_ok]


    # 5. اعمال shift(1) برای ضد-لوک‌اِهد
    # در numpy، shift(1) یعنی کل آرایه را یک واحد جابجا کنیم
    def shift_arr(arr):
        res = np.empty_like(arr)
        res[0] = np.nan if arr.dtype.kind == 'f' else 0
        res[1:] = arr[:-1]
        return res

    if debug_mode:
        bull_raw_full = np.zeros(length, dtype="int8")
        bear_raw_full = np.zeros(length, dtype="int8")
        bull_raw_full[idx:length] = bull_raw.astype("int8")
        bear_raw_full[idx:length] = bear_raw.astype("int8")

        bull_gap_full = np.zeros(length, dtype="float32")
        bear_gap_full = np.zeros(length, dtype="float32")
        bull_gap_full[idx:] = bull_gap.astype("float32")
        bear_gap_full[idx:] = bear_gap.astype("float32")

        bull_ok_full = np.zeros(length, dtype="bool")
        bear_ok_full = np.zeros(length, dtype="bool")
        bull_ok_full[idx:] = bull_ok.astype("bool")
        bear_ok_full[idx:] = bear_ok.astype("bool")

        mid_body_f[idx:] = mid_body.astype("float32")
        filter_mask_f[idx:] = filter_mask.astype(bool)

        return {
            "open" : open_ ,                                                                # for debug
            "high" : high  ,                                                                # for debug
            "low"  : low   ,                                                                # for debug
            "close": close ,                                                                # for debug
            "bull_raw":    pd.Series(bull_raw_full.astype(bool     ), index=open_.index),   # for debug
            "bear_raw":    pd.Series(bear_raw_full.astype(bool     ), index=open_.index),   # for debug
            "bull_gap":    pd.Series(bull_gap_full.astype("float32"), index=open_.index),   # for debug
            "bear_gap":    pd.Series(bear_gap_full.astype("float32"), index=open_.index),   # for debug
            "thr":         pd.Series(thr          .astype("float32"), index=open_.index),   # for debug
            "mid_body":    pd.Series(mid_body_f   .astype("float32"), index=open_.index),   # for debug
            "filter_mask": pd.Series(filter_mask_f.astype(bool     ), index=open_.index),   # for debug
            "bull_ok":     pd.Series(bull_ok_full .astype(bool     ), index=open_.index),   # for debug
            "bear_ok":     pd.Series(bear_ok_full .astype(bool     ), index=open_.index),   # for debug
            
            "fvg_up":     pd.Series(shift_arr(fvg_up    ), index=open_.index, dtype="int8"   ),
            "fvg_down":   pd.Series(shift_arr(fvg_down  ), index=open_.index, dtype="int8"   ),
            "fvg_top":    pd.Series(shift_arr(fvg_top   ), index=open_.index, dtype="float32"),
            "fvg_bottom": pd.Series(shift_arr(fvg_bottom), index=open_.index, dtype="float32"),
        }
    else:
        return {
            "fvg_up":     pd.Series(shift_arr(fvg_up    ), index=open_.index, dtype="int8"   ), # Flag 1 for bullish
            "fvg_down":   pd.Series(shift_arr(fvg_down  ), index=open_.index, dtype="int8"   ), # Flag 1 for bearish
            "fvg_top":    pd.Series(shift_arr(fvg_top   ), index=open_.index, dtype="float32"),
            "fvg_bottom": pd.Series(shift_arr(fvg_bottom), index=open_.index, dtype="float32"),            
        }

#-------------------------------------------------------------------- ok
# موارد زیر باید در فایلی بنام confluence باشند. در اینجا فقط باید فیچر بسازیم
# distance_to_fvg
# dist = min(|price - fvg_top| , |price - fvg_bottom|)
# normalized_distance = dist / ATR

def make_fvg(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    """
    Used from function: detect_fvg_optimized()
    Factory رجیستری: دریافت DataFrame با ستون‌های 'open','high','low','close'
    و برگرداندن خروجی استاندارد FVG (برای Engine/Registry).

    cfg شامل پارامترهای detect_fvg است:
      - lookback: int
      - atr_window: int
      - min_size_pct_of_atr: float
    """

    # ============================================ ok
    # Step-0: Config Values & Sanity Checks
    # ============================================
    # --- Used in Step-3 ------------------------
    # سطر زیر بیان میکند که هر اِف وی جی چند کندل فرصت دارد تا لمس شود
    N = int(cfg.get("max_bars_alive", 4))
    # --- sanity check ---
    if N < 1:
        raise ValueError(f"max_bars_alive must be >= 1, got {N}")
    if N > 50:
        # یا عددی که برای سیستم خودت مناسب می‌دانی
        raise ValueError(f"max_bars_alive too large for production: {N}")

    # --- Used in Step-4 (subpart-5) ------------
    w_size  = float(cfg.get("w_size" , 0.5))
    w_age   = float(cfg.get("w_age"  , 0.3))
    w_touch = float(cfg.get("w_touch", 0.2))
    # --- sanity check ---
    if any(w < 0 for w in (w_size, w_age, w_touch)):
        raise ValueError("FVG weights (w_size, w_age, w_touch) must be non-negative.")

    w_sum = w_size + w_age + w_touch
    if w_sum == 0:
        raise ValueError("Sum of FVG weights must be > 0.")
    # نرمال‌سازی
    w_size  /= w_sum
    w_age   /= w_sum
    w_touch /= w_sum

    # --- اعتبارسنجی ورودی ------------------------------------------ ok
    needed = ("open", "high", "low", "close")
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
    
    # ============================================ ok
    #  Step-1: Basic Output of FVG
    # ============================================
    out = detect_fvg_optimized(
        df["open"], df["high"], df["low"], df["close"],
        lookback=int(cfg.get("lookback", 2)),
        atr_window=int(cfg.get("atr_window", 14)),
        min_size_pct_of_atr=float(cfg.get("min_size_pct_of_atr", 0.50)),
        use_middle_filter=True,
    )
    # assert output structure
    for c in ("fvg_top", "fvg_bottom", "fvg_up", "fvg_down"):
        if c not in out:
            raise KeyError(f"detect_fvg_optimized missing '{c}'")
    if not out.index.equals(df.index):
        out = out.reindex(df.index)
        
    # ============================================ ok
    # Step-2: One-Bar Lifecycle (without look-ahead)
    # ============================================
    # نکته: یک شیفت 1 کندلی در تمام خروجی های توابع تشخیص اِف وی جی قبلاً اعمال شده است

    # یافتن کف و سقف زون اِف وی جی
    up  = out["fvg_up"].astype(bool)
    dn  = out["fvg_down"].astype(bool)
    top = out["fvg_top"]
    bot = out["fvg_bottom"]

    # تشخیص اینکه اصلاً زونی در کندل قبلی وجود دارد یا نه
    # این زون در صورت وجود در کندل جاری ذخیره شده است
    prev_exists = top.notna() & bot.notna()
    # تشخیص تولد زون: زون در کندل قبلی وجود دارد ولی در دو کندل قبلی وجود نداشته است
    born = prev_exists & ~(prev_exists.shift(1).astype(bool).fillna(False))

    # The union of the two sets "filled" and "expired" is equal to the set "touched".
    # The intersection of the two sets "filled" and "expired" is the empty set.
    touched_next = ((        # دو سطر زیر لمس زون توسط کندل 4 را بررسی میکنند
        ((df["low" ] <= top) & up ) |  # این سطر چک میکند که در حالت صعودی کف کندل 4 به زیر سقف زون آمده
        ((df["high"] >= bot) & dn )    # این سطر چک میکند که در حالت نزولی سقف کندل 4 به بالای کف زون آمده
    ) &
        prev_exists          # این سطر بررسی میکند که آیا در کندل 4 زون اِف وی جی مربوط به کندلهای 1و2و3 ثبت شده است؟
    )

    filled_next = ((         # دو سطر زیر پوشش کامل زون توسط کندل 4 را بررسی میکنند
        ((df["low" ] <= bot) & up ) | # این سطر چک میکند که در حالت صعودی کف کندل 4 به کف زون رسیده
        ((df["high"] >= top) & dn )   # این سطر چک میکند که در حالت نزولی سقف کندل 4 به سقف زون رسیده
    ) &
        prev_exists          # این سطر بررسی میکند که آیا در کندل 4 زون اِف وی جی مربوط به کندلهای 1و2و3 ثبت شده است؟
    )
    
    expired_next = (
        prev_exists &        # این سطر بررسی میکند که آیا در کندل 4 زون اِف وی جی مربوط به کندلهای 1و2و3 ثبت شده است؟
        touched_next &       # 
        (~filled_next)       # این سطر بررسی میکند که آیا زون کندلهای 1و2و3 در این کندل 4 پُر نشده است؟
    )
    
    out.update({             # قبل از خروجی نهایی، مقادیر درست/نادرست به مقادیر 1/0 تبدیل میشوند
        "fvg_born"        :         born.astype("int8"),
        "fvg_touched_next": touched_next.astype("int8"), 
        "fvg_filled_next" :  filled_next.astype("int8"),
        "fvg_expired_next": expired_next.astype("int8"),
    })

    # ============================================ ok
    # Step-3: Multi-Bar Lifecycle
    # ============================================
    # جمع آوری زونهای اِن کندل گذشته
    tops = [out["fvg_top"]   .shift(k) for k in range(N)]  # = range(0,N)
    bots = [out["fvg_bottom"].shift(k) for k in range(N)]
    ups  = [out["fvg_up"]    .shift(k) for k in range(N)]
    dns  = [out["fvg_down"]  .shift(k) for k in range(N)]

    # وجود زون در پنجرهٔ 0...اِن-1 -----------------------------------
    # بررسی این که آیا در این پنجره زمانی، زونی وجود داشته است یا نه؟
    """
    در زیر یک لیست اِن تایی از سری ها داریم سری اول دارای شیفت 1 کندلی، سری دوم دارای شیفت 2 کندلی و ... هستند
    هر سری دارای مقادیر درست/نادرست است
    مثلاً سومین سری دارای مقادیر زیر است:
    out["fvg_top"].shift(3).notna & out["fvg_bottom"].shift(3).notna
    """
    has_zone_cols = [(t.notna() & b.notna()) for t, b in zip(tops, bots)]
    # این خط می‌گوید: «آیا در اِن کندل قبلی، در هیچ‌کدام از آن‌ها زونی وجود داشته؟»
    # has_zone = pd.concat(has_zone_cols, axis=1).any(axis=1) # هزینه محاسباتی زیاد است. با سطرهای زیر جایگزین میشود
    has_zone = pd.Series(                                                    # سطر کم هزینه و جایگزین
        np.column_stack([c.to_numpy() for c in has_zone_cols]).any(axis=1),  # سطر کم هزینه و جایگزین
        index=df.index                                                       # سطر کم هزینه و جایگزین
    )                                                                        # سطر کم هزینه و جایگزین

    # تاچ کندل اخیر با هر کدام از زون‌های پنجرهٔ 0...اِن-1 ----------
    touched_list = [((df["low"] <= t) & (df["high"] >= b)) for t, b in zip(tops, bots)]
    # touched_any = pd.concat(touched_list, axis=1).any(axis=1)  # هزینه محاسباتی زیاد است. با سطرهای زیر جایگزین میشود
    touched_any = pd.Series(                                                 # سطر کم هزینه و جایگزین
        np.column_stack([c.to_numpy() for c in touched_list]).any(axis=1),   # سطر کم هزینه و جایگزین
        index=df.index,                                                      # سطر کم هزینه و جایگزین
    )                                                                        # سطر کم هزینه و جایگزین
    
    # پُرشدن هر کدام از زون‌های پنجرهٔ 0...اِن-1 با کندل اخیر -------
    filled_list_up = [((u.astype(bool)) & (df["low"] <= b)) for u, b in zip(ups, bots)]
    filled_list_dn = [((d.astype(bool)) & (df["high"] >= t)) for d, t in zip(dns, tops)]
    # filled_any = (                                        # هزینه محاسباتی زیاد است. با سطرهای زیر جایگزین میشود
    #     pd.concat(filled_list_up, axis=1).any(axis=1) |
    #     pd.concat(filled_list_dn, axis=1).any(axis=1)     )
    filled_up_any = np.column_stack([c.to_numpy() for c in filled_list_up]).any(axis=1)  # سطر کم هزینه و جایگزین
    filled_dn_any = np.column_stack([c.to_numpy() for c in filled_list_dn]).any(axis=1)  # سطر کم هزینه و جایگزین
    filled_any = filled_up_any | filled_dn_any                                           # سطر کم هزینه و جایگزین
    filled_any = pd.Series(filled_any, index=df.index)                                   # سطر کم هزینه و جایگزین

    # --- Expire (فقط قدیمی‌ترین زون) -------------------------------
    t_old, b_old = tops[-1], bots[-1]
    exist_old = t_old.notna() & b_old.notna()
    # touch_old = (df["low"] <= t_old) & (df["high"] >= b_old)   # old , deleted
    # expired_now = exist_old & (~touch_old)                     # old , deleted
    expired_now = exist_old                                  # new

    # --- Alive (حداقل یک زون در پنجره باشد که پُـر نشده باشد) ----
    # --- Preparing input data ------------------
    high_ = df["high"].to_numpy()
    low_  = df["low"].to_numpy()
    tops_mat = np.column_stack([t.to_numpy() for t in tops])
    bots_mat = np.column_stack([b.to_numpy() for b in bots])
    ups_mat  = np.column_stack([u.to_numpy() for u in ups])
    dns_mat  = np.column_stack([d.to_numpy() for d in dns])

    # --- نسخه اولیه ----------------------------
    def compute_alive_orig(low, high, tops, bots, ups, dns):
        alive_list = []
        for t, b, u, d in zip(tops, bots, ups, dns):
            zone_exists = t.notna() & b.notna()
            zone_filled = (
                ((low  <= b) & u) |
                ((high >= t) & d)
            ) & zone_exists
            zone_alive = zone_exists & (~zone_filled)
            alive_list.append(zone_alive)

        alive_now = pd.Series(
            np.column_stack([a.to_numpy() for a in alive_list]).any(axis=1),
            index=low.index
        )
        return alive_now
    
    # --- نسخه سریعتر ---------------------------
    def compute_alive_numpy(low, high, tops_, bots_, ups_, dns_):
        low_  = low.to_numpy()[:, None]
        high_ = high.to_numpy()[:, None]
        zone_exists = ~np.isnan(tops_) & ~np.isnan(bots_)
        zone_filled = (
            (ups_ & (low_  <= bots_)) |
            (dns_ & (high_ >= tops_))
        )
        zone_alive = zone_exists & (~zone_filled)
        alive_now = pd.Series(zone_alive.any(axis=1), index=low.index)
        return alive_now

    # --- نسخه فوق سریع -------------------------
    @njit
    def compute_alive_ultra(low_, high_, tops_, bots_, ups_, dns_):
        T, N = tops_.shape
        alive = np.zeros(T, dtype=np.bool_)
        for i in range(T):
            alive_flag = False
            for j in range(N):
                t = tops_[i, j]
                b = bots_[i, j]
                if np.isnan(t) or np.isnan(b):
                    continue

                filled = False
                if ups_[i, j]:
                    if low_[i] <= b:
                        filled = True
                elif dns_[i, j]:
                    if high_[i] >= t:
                        filled = True

                if not filled:
                    alive_flag = True
                    break

            alive[i] = alive_flag
        return alive

    # --- Alive result --------------------------
    # alive_now = compute_alive_orig(df["low"], df["high"], tops, bots, ups, dns)
    # alive_now = compute_alive_numpy(df["low"], df["high"], tops_mat, bots_mat, ups_mat, dns_mat)
    alive_now = pd.Series(
        compute_alive_ultra(low_, high_, tops_mat, bots_mat, ups_mat, dns_mat),
        index = df.index
    )

    out.update({
        "fvg_touched_any": touched_any.astype("int8"),   # تعیین میکند که آیا هیچکدام از زونهای داخل پنجره لمس شده اند
        "fvg_filled_any":   filled_any.astype("int8"),   # تعیین میکند که آیا هیچکدام از زونهای داخل پنجره پـُر شده اند
        "fvg_expired_now": expired_now.astype("int8"),   # تعیین میکند که آیا قدیمی ترین زون منقضی شده است
        "fvg_alive_window":  alive_now.astype("int8"),   # تعیین میکند که آیا حداقل یک زون در پنجره هست که پـُر نشده باشد
    })

    # ============================================ ok
    # Step‑4: Propagate Zone to Engine
    # Propagate the zone onto the target timeframe for easy consumption in the Engine
    # ============================================

    # Selectin the newest/nearest active zone in range 0..N-1 ------- subpart-1
    # ستون 0: shift(0),..., ستون آخر: shift(N-1)
    # cand_top = pd.concat(tops, axis=1).astype("float32")  # هزینه محاسباتی زیاد است. با سطرهای زیر جایگزین میشود
    # cand_bot = pd.concat(bots, axis=1).astype("float32")  # هزینه محاسباتی زیاد است. با سطرهای زیر جایگزین میشود
    
    cand_top = pd.DataFrame(                                # سطر کم هزینه و جایگزین
        np.column_stack([s.to_numpy() for s in tops]),
        index=df.index,
        dtype="float32"
    )

    cand_bot = pd.DataFrame(                                # سطر کم هزینه و جایگزین
        np.column_stack([s.to_numpy() for s in bots]),
        index=df.index,
        dtype="float32"
    )

    # انتخاب اولین مقدار غیر نن (معتبر) از چپ -------------
    sel_top = cand_top.bfill(axis=1).iloc[:, 0]
    sel_bot = cand_bot.bfill(axis=1).iloc[:, 0]
    # انتخاب فقط زونهایی که واقعاً فعال هستند -------------
    active_top = sel_top.where(alive_now)
    active_bot = sel_bot.where(alive_now)


    # --- Zone size & Normalization by ATR -------------------------- subpart-2
    atrv = atr_core(df["high"], df["low"], df["close"], n=int(cfg.get("atr_window", 14))).astype("float32")
    gap_size = (active_top - active_bot).abs().astype("float32")
    size_norm = (gap_size / (atrv.clip(lower=1e-8))).astype("float32")

    # Calc. zone age (dist from newest selected zone) --------------- subpart-3
    # در بخش قبل، cand_top و sel_top ساخته شده‌اند
    # سن زون برابر است با شماره ستونی که sel_top از آن آمده (0..N-1)
    
    # cand_top = cand_top.copy()  # چون تاانتهای این تابع دیگراز این دیتافریم استفاده نمیشود،این سطررا کامنت میکنم
    cand_top.columns = np.arange(cand_top.shape[1])  # changing column names to 0,1,...,N-1
    #===== deleted ==================== start
    # cmp = cand_top.eq(sel_top, axis=0).to_numpy()  # in which column of cand_top, the value is equal to sel_top
    #                                                # cmp will be a DataFrame contains TRUE/FALSE
    #                                                # در هر ردیف، ستونی که (درست) است همان ستونی است که زون انتخاب شده از آن آمده
    # any_true = cmp.any(axis=1)  # تعیین میکند که هر ردیف اصلاً زون معتبری داشت یا نه
    #                             # any_true: will be a one-dimensional vector
    # """ for below line:
    # for i in rows(any_ture):
    #     if any_true[i] == True:
    #         idx[i] = argmax_row[i] + 1
    #     else:
    #         idx[i] = NaN
    # """
    # idx = np.where(any_true, cmp.argmax(axis=1) + 1, np.nan)  # اگر هیچ True نبود → NaN
    #===== deleted ==================== end
    #===== added ====================== start
    cmp = cand_top.eq(sel_top, axis=0).to_numpy()
    rev_cmp = cmp[:, ::-1]                     # reverse columns
    any_true = rev_cmp.any(axis=1)
    idx = np.where(any_true, N - rev_cmp.argmax(axis=1), np.nan)
    #===== added ====================== end
    age_bars_est = pd.Series(idx, index=df.index, dtype="float32").where(active_top.notna())


    # --- Current touch & counting touches in window=N -------------- subpart-4
    touch_now = (
        (df["low"] <= active_top) & (df["high"] >= active_bot) &
        active_top.notna() & active_bot.notna()
    ).astype("int8")
    # به‌عنوان تقریب عملیاتی: مجموع لمس‌ها در پنجرهٔ ثابت N (برای زون فعال)
    touch_count = (
        touch_now
        .rolling(N, min_periods=1).sum().where(active_top.notna()).fillna(0).astype("float32")
    )

    
    # --- امتیاز زون: ترکیب اندازه/قدمت/تعداد لمس (بدون look-ahead) --- subpart-5

    # Simple Normilization of age/touch -----------------------------
    age_score = 1.0 / (1.0 + age_bars_est)           # هر چقدر که زون جدیدتر باشد امتیازش بیشتر است
    touch_score = (touch_count / float(max(1, N)))   # لمس‌های بیشتر (در N) → امتیاز بالاتر
    # Calc. fvg_score by weighted mean-------------------------------
    score = (w_size  *   size_norm.fillna(0.0) +   # size_norm  : comes form subpart-2
             w_age   *   age_score.fillna(0.0) +   # age_score  : comes form subpart-5
             w_touch * touch_score.fillna(0.0)     # touch_score: comes form subpart-5
    ).astype("float32")

    out.update({
        "fvg_active_top":    active_top.astype("float32"),      # from subpart-1
        "fvg_active_bottom": active_bot.astype("float32"),      # from subpart-1
        "fvg_gap_size":           gap_size.fillna(0).astype("float32"),   # from subpart-2
        "fvg_size_norm":         size_norm.fillna(0).astype("float32"),   # from subpart-2
        "fvg_age_bars_est":   age_bars_est.fillna(0).astype("float32"),        # from subpart-3
        "fvg_touch_now":      touch_now.astype("int8"),       # from subpart-4
        "fvg_touch_count":  touch_count.astype("float32"),    # from subpart-4
        "fvg_score":              score.astype("float32"),         # from subpart-5
    })
    return out


#============================================================================== MainPart-2
# Supply/Demand (SD) (Base → Impulse → Return)
#==============================================================================

def detect_sd(
    open_, high, low, close,
    *,
    base_len: int = 3,
    atr_window: int = 14,           # atr = atr_core(h,l,c,atr_window)
    base_atr_max: float = 0.6,      # base_atr_max   <   ratio = ((high-low)/atr)
    impulse_atr_min: float = 1.2,   #                    ratio = ((high-low)/atr)   <   impulse_atr_min
):
    """
    Production-grade Supply/Demand Zone Detector
    -------------------------------------------
    Detects institutional-quality SD zones using a strictly vectorized,
    no-lookahead, ATR-normalized base/impulse model.
    """
    
    # ================================= ok
    # 1) ATR + Range + Safety
    # =================================
    # هدف: اندازه‌گیری قدرت حرکت قیمت نسبت به نوسان بازار
    
    atr = atr_core(high, low, close, n=int(atr_window)).astype("float32")
    rng = (high - low).astype("float32")
    atr_safe = atr.clip(lower=1e-8)
    atr_safe[atr.isna()] = np.nan
    ratio = (rng / atr_safe).astype("float32")  # displacement measure


    # ================================= ok
    # 2) Base Detection (Low Volatility)
    # =================================
    # هدف: اگر تعداد اِل کندل پشت سر هم کم‌نوسان باشند آنگاه یک بیس ساخته شده است
    # بیس: ناحیه‌ای است که بازار در حال جمع کردن سفارشات است (بیس = ناحیه تجمیع)
    
    base_mask = ratio <= float(base_atr_max)        # bool
    L = int(base_len)
    base_count = (
        base_mask
        .rolling(L, min_periods=L)
        .sum()
    )
    base_complete = (base_count == L)


    # ================================= ok
    # 3) Impulse Detection (Strong Displacement)
    #    + Body-Dominance Filter (very important)
    # =================================
    # هدف: تشخیص حرکت انفجاری (ایمپالس)

    body = (close - open_).astype("float32")
    body_size = body.abs()

    # wick-heavy impulses removed
    body_ratio = body_size / rng.clip(lower=1e-8)
    strong_body = body_ratio >= 0.55       # safe threshold

    impulse_raw = ratio >= float(impulse_atr_min)
    impulse = impulse_raw & strong_body


    # ================================= ok
    # 4) Zone Birth
    # =================================
    # base must finish on previous candle
    base_prev = base_complete.shift(1, fill_value=False).astype(bool)
    born = (base_prev & impulse).astype("int8")


    # ================================= ok
    # 5) Direction (Supply / Demand)
    # =================================
    impulse_up = impulse & (body > 0)
    impulse_dn = impulse & (body < 0)

    sd_direction = np.select(
        [impulse_up, impulse_dn],
        [1, -1],                  # +1 تقاضا = demand ,    ,  عرضه = -1 supply
        default=0,
    ).astype("int8")
    sd_direction = (sd_direction * born).astype("int8")


    # ================================= ok
    # 6) Zone Boundaries
    # =================================
    # در اینجا همانطور که بیس شیفت خورده، باید مقادیر سقف و کف هم شیفت بخورند
    base_high = (
        high.shift(1)
        .rolling(L, min_periods=L)
        .max()
        .astype("float32")
    )

    base_low = (
        low.shift(1)
        .rolling(L, min_periods=L)
        .min()
        .astype("float32")
    )

    sd_top    = base_high.where(born == 1).astype("float32")
    sd_bottom = base_low .where(born == 1).astype("float32")
    

    # ================================= ok
    # 7) Output
    # =================================
    return {
        # "open": open_,         # for debug
        # "high": high,          # for debug
        # "low": low,            # for debug
        # "close": close,        # for debug

        # "atr": atr,               # for debug part-1
        # "rng": rng,               # for debug part-1
        # "atr_safe": atr_safe,     # for debug part-1
        # "ratio": ratio,           # for debug part-1

        # "base_mask": base_mask,          # for debug part-2
        # "base_count": base_count,        # for debug part-2
        # "base_complete": base_complete,  # for debug part-2

        # "body": body,                 # for debug part-3
        # "body_size": body_size,       # for debug part-3
        # "body_ratio": body_ratio,     # for debug part-3
        # "strong_body": strong_body,   # for debug part-3
        # "impulse_raw": impulse_raw,   # for debug part-3
        # "impulse": impulse,           # for debug part-3
        
        # "body": body,                   # for debug- from part-3
        # "impulse": impulse,             # for debug- from part-3
        # "impulse_up": impulse_up,       # for debug part-4
        # "impulse_dn": impulse_dn,       # for debug part-4
        # "sd_direction": sd_direction,   # for debug part-4

        # "base_complete": base_complete,  # for debug- from part-2
        # "impulse": impulse,              # for debug- from part-3
        # "base_prev": base_prev,          # for debug part-5

        #------------------------------
        "sd_born": born,                # it's type is astype("int8")
        "sd_top": sd_top,               # it's type is astype("float32")
        "sd_bottom": sd_bottom,         # it's type is astype("float32")
        "sd_direction": sd_direction,   # +1 demand, -1 supply, 0 none
    }

#--- Numba‑Accelerated -------------------------------------------------------- start
@njit(cache=True)
def _detect_sd_numba_core(
    open_, high, low, close,
    atr,
    base_len,
    base_atr_max,
    impulse_atr_min
):
    n = len(close)

    born = np.zeros(n, dtype=np.int8)
    top = np.full(n, np.nan, dtype=np.float32)
    bottom = np.full(n, np.nan, dtype=np.float32)
    direction = np.zeros(n, dtype=np.int8)

    ratio = np.empty(n, dtype=np.float32)
    base_mask = np.zeros(n, dtype=np.int8)

    # -----------------------------
    # ratio + base mask
    # -----------------------------
    for i in range(n):
        rng = high[i] - low[i]

        atr_safe = atr[i]
        if np.isnan(atr_safe):
            ratio[i] = np.nan
            continue

        if atr_safe < 1e-8:
            atr_safe = 1e-8

        ratio[i] = rng / atr_safe

        if ratio[i] <= base_atr_max:
            base_mask[i] = 1


    # -----------------------------
    # main detection loop
    # -----------------------------
    for i in range(base_len + 2, n):

        # check base window
        base_ok = True
        for j in range(i - base_len - 1, i - 1):
            if base_mask[j] == 0:
                base_ok = False
                break

        if not base_ok:
            continue


        # impulse check
        rng = high[i] - low[i]
        if rng <= 0:
            continue

        r = ratio[i]
        if np.isnan(r) or r < impulse_atr_min:
            continue


        body = close[i] - open_[i]
        body_ratio = abs(body) / rng

        if body_ratio < 0.55:
            continue


        # direction
        if body > 0:
            direction[i] = 1
        else:
            direction[i] = -1


        born[i] = 1


        # compute base boundaries
        hi = high[i-1]
        lo = low[i-1]

        for j in range(i - base_len - 1, i - 1):
            if high[j] > hi:
                hi = high[j]
            if low[j] < lo:
                lo = low[j]

        top[i] = hi
        bottom[i] = lo


    return born, top, bottom, direction

def detect_sd_numba(
    open_, high, low, close,
    *,
    base_len=3,
    atr_window=14,
    base_atr_max=0.6,
    impulse_atr_min=1.2
):
    """
    Numba-accelerated Supply/Demand detector.
    Suitable for multi-million candle datasets.
    """

    atr = atr_core(high, low, close, n=int(atr_window)).astype("float32")

    o = open_.values.astype(np.float32)
    h = high.values.astype(np.float32)
    l = low.values.astype(np.float32)
    c = close.values.astype(np.float32)
    a = atr.values.astype(np.float32)

    born, top, bottom, direction = _detect_sd_numba_core(
        o, h, l, c, a,
        int(base_len),
        float(base_atr_max),
        float(impulse_atr_min)
    )

    idx = close.index

    return {
        "sd_born": pd.Series(born, index=idx, dtype="int8"),
        "sd_top": pd.Series(top, index=idx, dtype="float32"),
        "sd_bottom": pd.Series(bottom, index=idx, dtype="float32"),
        "sd_direction": pd.Series(direction, index=idx, dtype="int8"),
    }

#--- Numba‑Accelerated -------------------------------------------------------- end

def make_sd(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    needed = ("open","high","low","close")
    for c in needed:
        if c not in df.columns: raise KeyError(f"Missing column: {c}")
    out = detect_sd(df["open"], df["high"], df["low"], df["close"],
                    base_len=int(cfg.get("base_len",3)),
                    atr_window=int(cfg.get("atr_window",14)),
                    base_atr_max=float(cfg.get("base_atr_max",0.6)),
                    impulse_atr_min=float(cfg.get("impulse_atr_min",1.2)))
    # lifecycle چند-کندلی
    N = int(cfg.get("max_bars_alive", 3))
    tops = [out["sd_top"].shift(k) for k in range(1, N+1)]
    bots = [out["sd_bottom"].shift(k) for k in range(1, N+1)]
    has_zone = pd.concat([(t.notna() & b.notna()) for t,b in zip(tops,bots)], axis=1).any(axis=1)
    hit_list = [((df["low"]<=t) & (df["high"]>=b)) for t,b in zip(tops,bots)]
    hit_any = pd.concat(hit_list, axis=1).any(axis=1).astype("int8")
    expired_now = ((tops[-1].notna() & bots[-1].notna()) & (~pd.concat(hit_list,axis=1).any(axis=1))).astype("int8")
    alive_now = (has_zone & (~hit_any.astype(bool))).astype("int8")
    # انتشار زون فعال
    cand_top = pd.concat(tops, axis=1).astype("float32")
    cand_bot = pd.concat(bots, axis=1).astype("float32")
    sel_top = cand_top.bfill(axis=1).iloc[:,0]
    sel_bot = cand_bot.bfill(axis=1).iloc[:,0]
    active_top = sel_top.where(alive_now.astype(bool))
    active_bot = sel_bot.where(alive_now.astype(bool))
    # اندازه/سن/لمس و امتیاز
    atrv = atr_core(df["high"], df["low"], df["close"], n=int(cfg.get("atr_window",14))).astype("float32")
    size = (active_top - active_bot).abs().astype("float32")
    size_norm = (size / atrv.replace(0,np.nan)).astype("float32")
    
    cand_top = cand_top.copy()
    cand_top.columns = np.arange(cand_top.shape[1])
    cmp = cand_top.eq(sel_top, axis=0)
    any_true = cmp.any(axis=1).to_numpy()
    idx = np.where(any_true, cmp.to_numpy().argmax(axis=1) + 1, np.nan)
    age_bars = pd.Series(idx, index=df.index, dtype="float32").where(active_top.notna())
    
    touch_now = ((df["low"]<=active_top) & (df["high"]>=active_bot) & active_top.notna() & active_bot.notna()).astype("int8")
    touch_cnt = touch_now.rolling(N, min_periods=1).sum().where(active_top.notna()).fillna(0).astype("float32")
    w_size=float(cfg.get("w_size",0.5))
    w_age=float(cfg.get("w_age",0.3))
    w_touch=float(cfg.get("w_touch",0.2))
    age_score = 1.0/(1.0 + age_bars.replace(0,np.nan))
    touch_score = (touch_cnt/float(max(1,N)))
    score = (w_size*size_norm.fillna(0)+w_age*age_score.fillna(0)+w_touch*touch_score.fillna(0)).astype("float32")
    out.update({
        "sd_alive_n": alive_now.astype("int8"),
        "sd_filled_window": hit_any.astype("int8"),
        "sd_expired_now": expired_now.astype("int8"),
        "sd_active_top": active_top.astype("float32"),
        "sd_active_bottom": active_bot.astype("float32"),
        "sd_size_norm": size_norm.fillna(0).astype("float32"),
        "sd_age_bars_est": age_bars.fillna(0).astype("float32"),
        "sd_touch_now": touch_now.astype("int8"),
        "sd_touch_count": touch_cnt.astype("float32"),
        "sd_score": score,
    })
    return out


#============================================================================== MainPart-3
# Order Block (OB)
# تشخیص + lifecycle + انتشار + امتیاز
#==============================================================================

def detect_ob(open_, high, low, close, *,
              atr_window:int=14,
              body_atr_min:float=0.6,     # حداقل بدنهٔ کندل کاندید نسبت به ATR
              wick_ratio_max:float=0.6,   # حداکثر نسبت ویک به بدنه برای کاندید
              bos_lookback:int=5          # حداکثر فاصلهٔ شکست ساختار بعد از کاندید
             ) -> Dict[str, pd.Series]:
    """
    تشخیص اولیهٔ Order Block:
    - کندل کاندید با بدنهٔ کافی (نسبت به ATR) و ویک‌های محدود.
    - سپس Break of Structure (BOS): قیمت در bos_lookback کندل بعدی، سقف/کف اخیر را می‌شکند.
    - ضد لوک‌اِهد: پرچم تولد با shift(+1) اعمال می‌شود.
    """
    atrv = atr_core(high, low, close, n=int(atr_window)).astype("float32")
    body = (close - open_).astype("float32")
    up = (body > 0)
    dn = (body < 0)
    body_abs = body.abs()
    rng = (high - low).astype("float32")
    wick_sum = (rng - body_abs).clip(lower=0.0)

    # کاندیدهای معتبر: بدنه کافی و ویک محدود
    valid = (body_abs / atrv.replace(0, np.nan) >= float(body_atr_min)) & \
            ((wick_sum / body_abs.replace(0, np.nan)) <= float(wick_ratio_max))

    # شکست ساختار: اگر کندل‌های بعدی، سقف/کف آخرِ قبل از کاندید را بشکنند
    recent_high = high.shift(1).rolling(int(bos_lookback), min_periods=1).max()
    recent_low  = low.shift(1).rolling(int(bos_lookback),  min_periods=1).min()
    bos_up = (close > recent_high)   # شکست به بالا
    bos_dn = (close < recent_low)    # شکست به پایین

    # تولد OB پس از BOS: 
    # - برای OB صعودی معمولاً آخرین کندل نزولی معتبر قبل از BOS
    # - برای OB نزولی آخرین کندل صعودی معتبر قبل از BOS
    # تقریب عملیاتی: در همان کندل BOS، تولد OB با مرزهای کندلِ مخالفِ معتبر قبل از آن
    # برای ضد لوک‌اِهد، پرچم روی کندل بعدی اعمال می‌شود.
    ob_bull_cand = (dn & valid)  # کندل نزولیِ معتبر (پتانسیل OB صعودی)
    ob_bear_cand = (up & valid)  # کندل صعودیِ معتبر (پتانسیل OB نزولی)

    # اندیس آخرین کاندید معتبر قبل از هر کندل
    last_bull_idx = (
        ob_bull_cand.astype("boolean")
        .where(ob_bull_cand, pd.NA)
        .ffill()
        .notna()
    )
    last_bear_idx = (
        ob_bear_cand.astype("boolean")
        .where(ob_bear_cand, pd.NA)
        .ffill()
        .notna()
    )

    born_bull = bos_up.astype("int8").shift(1).fillna(0).astype("int8") & last_bull_idx.astype("int8")
    born_bear = bos_dn.astype("int8").shift(1).fillna(0).astype("int8") & last_bear_idx.astype("int8")

    # مرزهای زون: بدنهٔ کندل کاندید (close/open)؛ به‌صورت تقریبی
    ob_top_bull = open_.where(ob_bull_cand).ffill().where(born_bull.astype(bool))
    ob_bot_bull = close.where(ob_bull_cand).ffill().where(born_bull.astype(bool))

    ob_top_bear = close.where(ob_bear_cand).ffill().where(born_bear.astype(bool))
    ob_bot_bear = open_.where(ob_bear_cand).ffill().where(born_bear.astype(bool))

    # تجمیع یک زون: اگر هر دو رخ دهد، اولویت با ساختارهای همسو با BOS است
    ob_top = ob_top_bull.where(born_bull.astype(bool), ob_top_bear)
    ob_bot = ob_bot_bull.where(born_bull.astype(bool), ob_bot_bear)

    # پرچم تولد نهایی (بدون همپوشانی): 
    born = ((born_bull.astype(bool) | born_bear.astype(bool))).astype("int8")
    return {
        "ob_born": born,
        "ob_top": ob_top.astype("float32"),
        "ob_bottom": ob_bot.astype("float32"),
    }


def make_ob(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    """
    ساخت خروجی Order Block برای Engine/Registry:
    - lifecycle چند-کندلی (alive/filled/expired)
    - انتشار زون فعال (active_top/bottom)
    - اندازه/سن/تعداد لمس و امتیاز نهایی
    پارامترها:
      atr_window, body_atr_min, wick_ratio_max, bos_lookback,
      max_bars_alive, w_size, w_age, w_touch
    """
    need = ("open","high","low","close")
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    out = detect_ob(
        df["open"], df["high"], df["low"], df["close"],
        atr_window=int(cfg.get("atr_window", 14)),
        body_atr_min=float(cfg.get("body_atr_min", 0.6)),
        wick_ratio_max=float(cfg.get("wick_ratio_max", 0.6)),
        bos_lookback=int(cfg.get("bos_lookback", 5)),
    )

    # lifecycle چند-کندلی (هم‌راستا با FVG/SD)
    N = int(cfg.get("max_bars_alive", 3))
    tops = [out["ob_top"].shift(k) for k in range(1, N+1)]
    bots = [out["ob_bottom"].shift(k) for k in range(1, N+1)]
    has_zone = pd.concat([(t.notna() & b.notna()) for t,b in zip(tops,bots)], axis=1).any(axis=1)
    hit_list = [((df["low"]<=t) & (df["high"]>=b)) for t,b in zip(tops,bots)]
    hit_any = pd.concat(hit_list, axis=1).any(axis=1).astype("int8")
    expired_now = ((tops[-1].notna() & bots[-1].notna()) & (~pd.concat(hit_list,axis=1).any(axis=1))).astype("int8")
    alive_now = (has_zone & (~hit_any.astype(bool))).astype("int8")

    # انتشار زون فعال
    cand_top = pd.concat(tops, axis=1).astype("float32")
    cand_bot = pd.concat(bots, axis=1).astype("float32")
    sel_top = cand_top.bfill(axis=1).iloc[:,0]
    sel_bot = cand_bot.bfill(axis=1).iloc[:,0]
    active_top = sel_top.where(alive_now.astype(bool))
    active_bot = sel_bot.where(alive_now.astype(bool))

    # اندازه/سن/لمس و امتیاز (هم‌وزن با FVG/SD برای یکنواختی)
    atrv = atr_core(df["high"], df["low"], df["close"], n=int(cfg.get("atr_window",14))).astype("float32")
    size = (active_top - active_bot).abs().astype("float32")
    size_norm = (size / atrv.replace(0, np.nan)).astype("float32")
    
    cand_top = cand_top.copy()
    cand_top.columns = np.arange(cand_top.shape[1])
    cmp = cand_top.eq(sel_top, axis=0)
    any_true = cmp.any(axis=1).to_numpy()
    idx = np.where(any_true, cmp.to_numpy().argmax(axis=1) + 1, np.nan)
    age_bars = pd.Series(idx, index=df.index, dtype="float32").where(active_top.notna())
    
    touch_now = ((df["low"]<=active_top) & (df["high"]>=active_bot) & active_top.notna() & active_bot.notna()).astype("int8")
    touch_cnt = touch_now.rolling(N, min_periods=1).sum().where(active_top.notna()).fillna(0).astype("float32")

    w_size=float(cfg.get("w_size",0.5))
    w_age=float(cfg.get("w_age",0.3))
    w_touch=float(cfg.get("w_touch",0.2))
    age_score = 1.0/(1.0 + age_bars.replace(0,np.nan))
    touch_score = (touch_cnt/float(max(1,N)))
    score = (w_size*size_norm.fillna(0)+w_age*age_score.fillna(0)+w_touch*touch_score.fillna(0)).astype("float32")

    out.update({
        "ob_alive_n": alive_now.astype("int8"),
        "ob_filled_window": hit_any.astype("int8"),
        "ob_expired_now": expired_now.astype("int8"),
        "ob_active_top": active_top.astype("float32"),
        "ob_active_bottom": active_bot.astype("float32"),
        "ob_size_norm": size_norm.fillna(0).astype("float32"),
        "ob_age_bars_est": age_bars.fillna(0).astype("float32"),
        "ob_touch_now": touch_now.astype("int8"),
        "ob_touch_count": touch_cnt.astype("float32"),
        "ob_score": score,
    })
    return out


#============================================================================== MainPart-4
# Liquidity Sweep (EQH/EQL grab + close back inside)
# تشخیص + lifecycle + انتشار + امتیاز
#==============================================================================

def detect_liq_sweep(open_, high, low, close, *,
                     lookback:int=5,            # پنجرهٔ مرجع برای سقف/کف اخیر
                     atr_window:int=14,
                     min_tail_atr:float=0.5     # حداقل طول ویک نسبت به ATR
                     ) -> Dict[str, pd.Series]:
    """
    تشخیص اولیهٔ Liquidity Sweep:
    - Bearish sweep: high > recent_high  و  close < recent_high  و  ویک بالایی بلند
    - Bullish sweep: low  < recent_low   و  close > recent_low   و  ویک پایینی بلند
    زونِ پایش:
      - برای bearish: [recent_high , high]
      - برای bullish: [low , recent_low]
    ضدّ لوک‌اِهد: پرچم «تولد» با shift(+1) اعمال می‌شود.
    """
    atrv = atr_core(high, low, close, n=int(atr_window)).astype("float32")

    # سقف/کف اخیر (تا کندل قبل)
    recent_high = high.shift(1).rolling(int(lookback), min_periods=1).max().astype("float32")
    recent_low  = low.shift(1).rolling(int(lookback),  min_periods=1).min().astype("float32")

    # ویک‌ها
    upper_tail = (high - np.maximum(open_, close)).clip(lower=0.0).astype("float32")
    lower_tail = (np.minimum(open_, close) - low).clip(lower=0.0).astype("float32")

    # شرط طول ویک نسبت به ATR
    tail_ok_up = (upper_tail / atrv.replace(0, np.nan) >= float(min_tail_atr))
    tail_ok_dn = (lower_tail / atrv.replace(0, np.nan) >= float(min_tail_atr))

    # الگوی سوئیپ
    bearish_raw = (high > recent_high) & (close < recent_high) & tail_ok_up
    bullish_raw = (low  < recent_low)  & (close > recent_low)  & tail_ok_dn

    # زون‌های سوئیپ (روی همان کندل رخداد)
    ls_top_bear = high.where(bearish_raw).astype("float32")
    ls_bot_bear = recent_high.where(bearish_raw).astype("float32")

    ls_top_bull = recent_low.where(bullish_raw).astype("float32")
    ls_bot_bull = low.where(bullish_raw).astype("float32")

    # پرچم تولد با ضدّ لوک‌اِهد
    born_bear = bearish_raw.astype("int8").shift(1).fillna(0).astype("int8")
    born_bull = bullish_raw.astype("int8").shift(1).fillna(0).astype("int8")

    # تجمیع به یک زون (اگر هر دو نادر اتفاق افتاد، اولویت ساده: bear سپس bull)
    ls_top = ls_top_bear.where(bearish_raw, ls_top_bull)
    ls_bot = ls_bot_bear.where(bearish_raw, ls_bot_bull)
    born   = ((bearish_raw | bullish_raw).astype("int8").shift(1).fillna(0)).astype("int8")

    return {
        "ls_born": born,
        "ls_top":  ls_top.astype("float32"),
        "ls_bottom": ls_bot.astype("float32"),
    }


def make_liq_sweep(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    """
    ساخت خروجی Liquidity Sweep برای Engine/Registry:
    - lifecycle چند-کندلی (alive/filled/expired)
    - انتشار زون فعال (active_top/bottom)
    - اندازه/سن/تعداد لمس و امتیاز نهایی (سازگار با FVG/SD/OB)
    پارامترها: lookback, atr_window, min_tail_atr, max_bars_alive, w_size/w_age/w_touch
    """
    need = ("open","high","low","close")
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    out = detect_liq_sweep(
        df["open"], df["high"], df["low"], df["close"],
        lookback=int(cfg.get("lookback", 5)),
        atr_window=int(cfg.get("atr_window", 14)),
        min_tail_atr=float(cfg.get("min_tail_atr", 0.5)),
    )

    # lifecycle چند-کندلی (هم‌راستا با بقیه)
    N = int(cfg.get("max_bars_alive", 3))
    tops = [out["ls_top"].shift(k) for k in range(1, N+1)]
    bots = [out["ls_bottom"].shift(k) for k in range(1, N+1)]
    has_zone = pd.concat([(t.notna() & b.notna()) for t,b in zip(tops,bots)], axis=1).any(axis=1)
    hit_list = [((df["low"]<=t) & (df["high"]>=b)) for t,b in zip(tops,bots)]
    hit_any = pd.concat(hit_list, axis=1).any(axis=1).astype("int8")
    expired_now = ((tops[-1].notna() & bots[-1].notna()) & (~pd.concat(hit_list,axis=1).any(axis=1))).astype("int8")
    alive_now = (has_zone & (~hit_any.astype(bool))).astype("int8")

    # انتشار زون فعال
    cand_top = pd.concat(tops, axis=1).astype("float32")
    cand_bot = pd.concat(bots, axis=1).astype("float32")
    sel_top = cand_top.bfill(axis=1).iloc[:,0]
    sel_bot = cand_bot.bfill(axis=1).iloc[:,0]
    active_top = sel_top.where(alive_now.astype(bool))
    active_bot = sel_bot.where(alive_now.astype(bool))

    # اندازه/سن/لمس و امتیاز
    atrv = atr_core(df["high"], df["low"], df["close"], n=int(cfg.get("atr_window",14))).astype("float32")
    size = (active_top - active_bot).abs().astype("float32")
    size_norm = (size / atrv.replace(0, np.nan)).astype("float32")

    cand_top = cand_top.copy()
    cand_top.columns = np.arange(cand_top.shape[1])
    cmp = cand_top.eq(sel_top, axis=0)
    any_true = cmp.any(axis=1).to_numpy()
    idx = np.where(any_true, cmp.to_numpy().argmax(axis=1) + 1, np.nan)
    age_bars = pd.Series(idx, index=df.index, dtype="float32").where(active_top.notna())

    touch_now = ((df["low"]<=active_top) & (df["high"]>=active_bot) & active_top.notna() & active_bot.notna()).astype("int8")
    touch_cnt = touch_now.rolling(N, min_periods=1).sum().where(active_top.notna()).fillna(0).astype("float32")

    w_size=float(cfg.get("w_size",0.5))
    w_age=float(cfg.get("w_age",0.3))
    w_touch=float(cfg.get("w_touch",0.2))
    age_score = 1.0/(1.0 + age_bars.replace(0, np.nan))
    touch_score = (touch_cnt/float(max(1,N)))
    score = (w_size*size_norm.fillna(0)+w_age*age_score.fillna(0)+w_touch*touch_score.fillna(0)).astype("float32")

    out.update({
        "ls_alive_n": alive_now.astype("int8"),
        "ls_filled_window": hit_any.astype("int8"),
        "ls_expired_now": expired_now.astype("int8"),
        "ls_active_top": active_top.astype("float32"),
        "ls_active_bottom": active_bot.astype("float32"),
        "ls_size_norm": size_norm.fillna(0).astype("float32"),
        "ls_age_bars_est": age_bars.fillna(0).astype("float32"),
        "ls_touch_now": touch_now.astype("int8"),
        "ls_touch_count": touch_cnt.astype("float32"),
        "ls_score": score,
    })
    return out


#============================================================================== MainPart-5
# Breaker / Flip Zone (Invalidated OB → Reverse Zone)
#==============================================================================

def detect_breaker_flip(open_, high, low, close, *,
                        atr_window:int=14,
                        ob_body_atr_min:float=0.6,
                        ob_wick_ratio_max:float=0.6,
                        ob_bos_lookback:int=5,
                        lookback:int=5) -> Dict[str, pd.Series]:
    """
    تشخیص اولیهٔ Breaker/Flip:
    - ابتدا OBها را استخراج می‌کنیم (همین فایل: detect_ob)؛
    - اگر در 1..lookback کندلِ بعدی، قیمت «کامل» از بالای TOP (یا از زیر BOTTOM) OB عبور کند،
      همان کندلِ عبور، Breaker/Flip می‌سازد.
    - زون Breaker بدنهٔ همان کندلِ عبور است (body-top/body-bottom).
    - ضدّ لوک‌اِهد: پرچم تولد با shift(+1) اعمال می‌شود.
    """
    # OBهای اخیر
    ob = detect_ob(open_, high, low, close,
                   atr_window=atr_window,
                   body_atr_min=ob_body_atr_min,
                   wick_ratio_max=ob_wick_ratio_max,
                   bos_lookback=ob_bos_lookback)
    ob_top = ob["ob_top"]
    ob_bot = ob["ob_bottom"]

    # عبور از زون‌های OB در پنجرهٔ 1..L
    L = int(lookback)
    tops = [ob_top.shift(k) for k in range(1, L+1)]
    bots = [ob_bot.shift(k) for k in range(1, L+1)]

    inv_up_cols = [ (close > t) & t.notna() for t in tops ]   # عبور به بالا
    inv_dn_cols = [ (close < b) & b.notna() for b in bots ]   # عبور به پایین
    inv_up = pd.concat(inv_up_cols, axis=1).any(axis=1)
    inv_dn = pd.concat(inv_dn_cols, axis=1).any(axis=1)

    # زون Breaker = بدنهٔ کندل عبور
    body_top = np.maximum(open_, close).astype("float32")
    body_bot = np.minimum(open_, close).astype("float32")

    bf_top_up = body_top.where(inv_up)
    bf_bot_up = body_bot.where(inv_up)
    bf_top_dn = body_top.where(inv_dn)
    bf_bot_dn = body_bot.where(inv_dn)

    # تجمیع به یک زون (اولویت با inv_up)
    bf_top = bf_top_up.where(inv_up, bf_top_dn)
    bf_bot = bf_bot_up.where(inv_up, bf_bot_dn)

    born = ((inv_up | inv_dn).astype("int8").shift(1).fillna(0)).astype("int8")
    return {
        "bf_born": born,
        "bf_top":  bf_top.astype("float32"),
        "bf_bottom": bf_bot.astype("float32"),
    }


def make_breaker_flip(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    """
    Breaker/Flip برای Engine/Registry:
    - lifecycle چند-کندلی (alive/filled/expired)
    - انتشار زون فعال (active_top/bottom)
    - اندازه/قدمت/تعداد لمس و امتیاز نهایی
    پارامترها:
      atr_window, ob_body_atr_min, ob_wick_ratio_max, ob_bos_lookback, lookback,
      max_bars_alive, w_size, w_age, w_touch
    """
    need = ("open","high","low","close")
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    out = detect_breaker_flip(
        df["open"], df["high"], df["low"], df["close"],
        atr_window=int(cfg.get("atr_window", 14)),
        ob_body_atr_min=float(cfg.get("ob_body_atr_min", 0.6)),
        ob_wick_ratio_max=float(cfg.get("ob_wick_ratio_max", 0.6)),
        ob_bos_lookback=int(cfg.get("ob_bos_lookback", 5)),
        lookback=int(cfg.get("lookback", 5)),
    )

    # lifecycle چند-کندلی (هم‌راستا با FVG/SD/OB)
    N = int(cfg.get("max_bars_alive", 3))
    tops = [out["bf_top"].shift(k) for k in range(1, N+1)]
    bots = [out["bf_bottom"].shift(k) for k in range(1, N+1)]
    has_zone = pd.concat([(t.notna() & b.notna()) for t,b in zip(tops,bots)], axis=1).any(axis=1)
    hit_list = [((df["low"]<=t) & (df["high"]>=b)) for t,b in zip(tops,bots)]
    hit_any = pd.concat(hit_list, axis=1).any(axis=1).astype("int8")
    expired_now = ((tops[-1].notna() & bots[-1].notna()) & (~pd.concat(hit_list,axis=1).any(axis=1))).astype("int8")
    alive_now = (has_zone & (~hit_any.astype(bool))).astype("int8")

    # انتشار زون فعال
    cand_top = pd.concat(tops, axis=1).astype("float32")
    cand_bot = pd.concat(bots, axis=1).astype("float32")
    sel_top = cand_top.bfill(axis=1).iloc[:,0]
    sel_bot = cand_bot.bfill(axis=1).iloc[:,0]
    active_top = sel_top.where(alive_now.astype(bool))
    active_bot = sel_bot.where(alive_now.astype(bool))

    # اندازه/سن/لمس و امتیاز
    atrv = atr_core(df["high"], df["low"], df["close"], n=int(cfg.get("atr_window",14))).astype("float32")
    size = (active_top - active_bot).abs().astype("float32")
    size_norm = (size / atrv.replace(0, np.nan)).astype("float32")

    cand_top = cand_top.copy()
    cand_top.columns = np.arange(cand_top.shape[1])
    cmp = cand_top.eq(sel_top, axis=0)
    any_true = cmp.any(axis=1).to_numpy()
    idx = np.where(any_true, cmp.to_numpy().argmax(axis=1) + 1, np.nan)
    age_bars = pd.Series(idx, index=df.index, dtype="float32").where(active_top.notna())

    touch_now = ((df["low"]<=active_top) & (df["high"]>=active_bot) & active_top.notna() & active_bot.notna()).astype("int8")
    touch_cnt = touch_now.rolling(N, min_periods=1).sum().where(active_top.notna()).fillna(0).astype("float32")

    w_size=float(cfg.get("w_size",0.5))
    w_age=float(cfg.get("w_age",0.3))
    w_touch=float(cfg.get("w_touch",0.2))
    age_score = 1.0/(1.0 + age_bars.replace(0, np.nan))
    touch_score = (touch_cnt/float(max(1,N)))
    score = (w_size*size_norm.fillna(0)+w_age*age_score.fillna(0)+w_touch*touch_score.fillna(0)).astype("float32")

    out.update({
        "bf_alive_n": alive_now.astype("int8"),
        "bf_filled_window": hit_any.astype("int8"),
        "bf_expired_now": expired_now.astype("int8"),
        "bf_active_top": active_top.astype("float32"),
        "bf_active_bottom": active_bot.astype("float32"),
        "bf_size_norm": size_norm.fillna(0).astype("float32"),
        "bf_age_bars_est": age_bars.fillna(0).astype("float32"),
        "bf_touch_now": touch_now.astype("int8"),
        "bf_touch_count": touch_cnt.astype("float32"),
        "bf_score": score,
    })
    return out


#============================================================================== MainPart-6
# SR Fusion (FVG + SD + OB + BF + LS)
# Fair Value Gap, Supply/Demand, Order Block, Breaker/Flip, Liquidity Sweep
#==============================================================================

def make_sr_fusion(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    """
    فیوژن 5 گانهٔ SR با کنترل پایداری:
      - EMA smoothing + age decay                 EMA زمانی روی امتیازها + decay برحسب سن زون
      - hysteresis (enter/exit)                   Hysteresis با آستانه‌ی ورود/خروج (enter/exit)
      - optional MTF confirmation (confirm_col)   MTF confirmation اختیاری با ستون/فلگ ورودی (confirm_col)
      - tie-break (تازگی/فعال‌بودن)               Tie-break & conflict (برنده = بالاترین امتیاز؛ در برابری: زون تازه‌تر/Active)
      - cooldown پس از filled/expired             Cooldown پس از filled/expired (خاموش‌کردن سهم مؤلفه طی cooldown_bars)
      - min-confidence                            Min-confidence (زیر حداقل، خروجی صفر)
    پارامترها (نمونهٔ پیش‌فرض):
      ema_span=5, age_norm=5, enter_th=0.55, exit_th=0.35,
      cooldown_bars=3, min_conf=0.30, confirm_col=None
      max_bars_alive=3, atr_window=14, w_size=0.5, w_age=0.3, w_touch=0.2
    چند نکته:
    - برای MTF، اگر ستونی برای تایید داری (مثلاً از TF بالاتر)، 
      نامش را با confirm_col بده تا فقط در همان نقاط، فیوژن فعال شود.
    - هرجا خواستی وزن‌دهی/آستانه‌ها را تغییر بدهی، پارامترها در cfg هستند.
    
    """
    # 1) محاسبهٔ مؤلفه‌ها (با همان سازوکار داخلیِ همین فایل)
    fvg = make_fvg(df, **cfg)
    sd  = make_sd(df,  **cfg)
    ob  = make_ob(df,  **cfg)
    bf  = make_breaker_flip(df, **cfg)
    ls  = make_liq_sweep(df, **cfg)

    # 2) استخراج امتیاز و متادیتا
    comps = [
        ("fvg", fvg, "fvg_score", "fvg_active_top", "fvg_active_bottom", "fvg_age_bars_est", "fvg_alive_n", "fvg_filled_window", "fvg_expired_now"),
        ("sd",  sd,  "sd_score",  "sd_active_top",  "sd_active_bottom",  "sd_age_bars_est",  "sd_alive_n",  "sd_filled_window",  "sd_expired_now"),
        ("ob",  ob,  "ob_score",  "ob_active_top",  "ob_active_bottom",  "ob_age_bars_est",  "ob_alive_n",  "ob_filled_window",  "ob_expired_now"),
        ("bf",  bf,  "bf_score",  "bf_active_top",  "bf_active_bottom",  "bf_age_bars_est",  "bf_alive_n",  "bf_filled_window",  "bf_expired_now"),
        ("ls",  ls,  "ls_score",  "ls_active_top",  "ls_active_bottom",  "ls_age_bars_est",  "ls_alive_n",  "ls_filled_window",  "ls_expired_now"),
    ]

    ema_span   = int(cfg.get("ema_span", 5))
    age_norm   = float(cfg.get("age_norm", 5.0))
    enter_th   = float(cfg.get("enter_th", 0.55))
    exit_th    = float(cfg.get("exit_th", 0.35))
    cooldown_k = int(cfg.get("cooldown_bars", 3))
    min_conf   = float(cfg.get("min_conf", 0.30))
    confirm_col = cfg.get("confirm_col", None)  # نام ستونی در df که True/1 = تایید MTF

    eff_scores = {}
    tops_map, bots_map, ages_map, alive_map = {}, {}, {}, {}
    cool_map = {}  # rolling filled/expired

    for key, dct, s_score, s_top, s_bot, s_age, s_alive, s_filled, s_expired in comps:
        sc  = dct.get(s_score,  pd.Series(0.0, index=df.index, dtype="float32")).astype("float32")
        top = dct.get(s_top,    pd.Series(np.nan, index=df.index, dtype="float32")).astype("float32")
        bot = dct.get(s_bot,    pd.Series(np.nan, index=df.index, dtype="float32")).astype("float32")
        age = dct.get(s_age,    pd.Series(0.0, index=df.index, dtype="float32")).astype("float32")
        liv = dct.get(s_alive,  pd.Series(0, index=df.index, dtype="int8")).astype("int8")
        fil = dct.get(s_filled, pd.Series(0, index=df.index, dtype="int8")).astype("int8")
        exp = dct.get(s_expired,pd.Series(0, index=df.index, dtype="int8")).astype("int8")

        # EMA smoothing + age decay + alive mask
        sc_ema = sc.ewm(span=ema_span, adjust=False, min_periods=1).mean().astype("float32")
        decay  = (1.0 / (1.0 + (age / max(1e-6, age_norm)))).astype("float32")
        alive_mask = liv.astype(bool)
        sc_eff = (sc_ema * decay).where(alive_mask, 0.0)

        # cooldown: اگر اخیرأ filled/expired شده، سهم صفر
        cool = (fil.astype(bool) | exp.astype(bool)).rolling(cooldown_k, min_periods=1).max().fillna(0).astype("int8")
        sc_eff = sc_eff.where(~cool.astype(bool), 0.0)

        eff_scores[key] = sc_eff.astype("float32")
        tops_map[key] = top
        bots_map[key] = bot
        ages_map[key] = age
        alive_map[key] = liv
        cool_map[key] = cool

    # 3) MTF confirmation (اختیاری)
    if confirm_col is not None and confirm_col in df.columns:
        conf = df[confirm_col].astype("int8").astype(bool)
        for k in list(eff_scores.keys()):
            eff_scores[k] = eff_scores[k].where(conf, 0.0)

    # 4) انتخاب برنده و tie-break (تازگی/فعال بودن)
    eff_df = pd.DataFrame(eff_scores).astype("float32")
    sr_score_smooth = eff_df.max(axis=1).astype("float32")
    winner = eff_df.idxmax(axis=1).astype("string")

    # tie-break: اگر چند مؤلفه تقریباً برابرند (در محدودهٔ 0.02)، تازه‌تر/active ترجیح داده شود
    eps = float(cfg.get("tie_eps", 0.02))
    for i in range(len(winner)):
        candidates = [k for k in eff_df.columns if abs(eff_df.iloc[i][k] - sr_score_smooth.iloc[i]) <= eps]
        if len(candidates) > 1:
            # تازه‌تر را انتخاب کن: سن کمتر
            sel = min(candidates, key=lambda k: (ages_map[k].iloc[i] if not np.isnan(ages_map[k].iloc[i]) else 1e9))
            # اگر غیر فعال بود، سعی کن فعال‌ترین را بگیری
            if not bool(alive_map[sel].iloc[i]):
                actives = [k for k in candidates if bool(alive_map[k].iloc[i])]
                if actives:
                    sel = min(actives, key=lambda k: (ages_map[k].iloc[i] if not np.isnan(ages_map[k].iloc[i]) else 1e9))
            winner.iloc[i] = sel

    # 5) فعال‌سازی hysteresis (enter/exit) + min-confidence
    sr_state = pd.Series(0, index=df.index, dtype="int8")
    sr_score = pd.Series(0.0, index=df.index, dtype="float32")
    sr_top   = pd.Series(np.nan, index=df.index, dtype="float32")
    sr_bot   = pd.Series(np.nan, index=df.index, dtype="float32")

    for i in range(len(df)):
        s = sr_score_smooth.iloc[i]
        if sr_state.iloc[i-1] if i>0 else 0:
            # حالت on → بمان مگر زیر exit یا زیر min_conf
            if (s < exit_th) or (s < min_conf):
                sr_state.iloc[i] = 0
            else:
                sr_state.iloc[i] = 1
        else:
            # حالت off → فقط اگر عبور از enter و min_conf
            if (s >= enter_th) and (s >= min_conf):
                sr_state.iloc[i] = 1
            else:
                sr_state.iloc[i] = 0

        if sr_state.iloc[i]:
            w = winner.iloc[i]
            sr_score.iloc[i] = s
            sr_top.iloc[i]   = tops_map[w].iloc[i]
            sr_bot.iloc[i]   = bots_map[w].iloc[i]
        # else: صفر/NaN باقی می‌ماند

    return {
        "sr_score_raw": sr_score_smooth.astype("float32"),
        "sr_score": sr_score.astype("float32"),
        "sr_active_top": sr_top.astype("float32"),
        "sr_active_bottom": sr_bot.astype("float32"),
        "sr_on": sr_state.astype("int8"),
        "sr_source": winner.astype("string"),
    }

