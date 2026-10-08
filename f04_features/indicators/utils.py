# -*- coding: utf-8 -*-
# f04_features/indicators/utils.py
# Status in (Bot-RL-3): Reviewed at 1404/09/28

"""
- ابزارهای کمکی مشترک برای اندیکاتورها
- کشف تایم‌فریم‌ها از نام ستون‌ها
- نگهبان NaN/Inf و سبک کردن dtype
- zscore، true_range
"""
# =============================================================================
# Imports & Logger
# ============================================================================= (review:040924)
from __future__ import annotations
import re
import logging
import numpy as np
import pandas as pd
from numba import njit
from dataclasses import dataclass, field
# from datetime import datetime
from typing import Sequence, Optional, Dict, Tuple, List, Iterable, Any

from f04_features.indicators.zigzag import zigzag_wrapper as zigzag
# # وزن‌دهی — نام ستون‌های قابل‌قبول (اولین موجود انتخاب می‌شود)
# DEFAULT_MA_SLOPE_CANDIDATES: List[str] = [
#     "__ma_slope@M5", "__ma_slope@H1", "__ma_slope@H4"
# ]
# DEFAULT_RSI_SCORE_CANDIDATES: List[str] = [
#     "__rsi_zone@H1__rsi_zone_score", "__rsi_zone@H4__rsi_zone_score"
# ]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

""" --------------------------------------------------------------------------- OK C1 (review:040924)
کشف تایم‌فریم‌ها از روی نام ستون‌ها
"""
@dataclass
class TFView:
    tf: str
    cols: Dict[str, str]  # mapping: standard_name -> df_column_name
                          #        :        "open" -> "H1_open"

_TF_REGEX = re.compile(r"^(?P<tf>[A-Z0-9]+)_(?P<field>open|high|low|close|volume|spread)$", re.IGNORECASE)


""" --------------------------------------------------------------------------- OK C2 (review:040924) (For Test Only)
FiboTestConfig: پیکربندی تستِ مستقل از scripts  (Test-only, not runtime defaults)
-----------------------------------------------------------------------------
توضیح آموزشی (فارسی):
این کلاس فقط برای سناریوهای تست/وایرینگ سبک استفاده می‌شود تا وابستگی به f15_scripts
حذف شود. اگر فولدر scripts پاک شود، این کلاس هنوز داخل هسته باقی می‌ماند.
- مقادیر پیش‌فرض مطابق بلوک «پیکربندی تست» قبلی هستند.
- بعداً اگر کلیدهای متناظر از قبل در config.yaml موجود باشند، می‌توانیم همین کلاس
  را از روی کانفیگ پُر کنیم (طبق قانون کانفیگ شما: اگر کلید موجود بود، تکراری نسازیم).
- پیام‌های اجرایی در این کلاس نداریم؛ فقط داده و هِلپرهای سبک.

"""
@dataclass
class FiboTestConfig:
    # مسیر دیتاست پردازش‌شده (در صورت نیاز برای سناریوی تست)
    DATA_FILE: str = r"f03_data/processed/XAUUSD/H1.parquet"

    # تایم‌فریم‌ها و پنجرهٔ برش دادهٔ اخیر
    TFS: List[str] = field(default_factory=lambda: 
                            ["M1", "M5", "M30", "H1", "H4", "D1", "W1"])
    TAILS: Dict[str, int] = field(default_factory=lambda: 
                            {"M1": 1000, "M5": 1000, "M30": 1000, "H1": 500, "H4": 500, "D1": 500, "W1":200})
    N_LEGS: int = 5                # تعداد لگ‌های اخیر برای هر TF

    # پارامترهای خوشه‌بندی فیبو (درصدها به واحد percent هستند)
    TOL_PCT: float = 0.20          # پنجرهٔ همگرایی خوشه‌ها (٪)
    PREFER_RATIO: float = 0.618    # نسبت مرجح

    # پارامترهای سطوح رُند S/R
    SR_STEP: float = 10.0          # گام سطوح رُند (مثلاً XAUUSD≈10)
    SR_COUNT: int = 25             # تعداد سطوح رُند حول قیمت آخر
    SR_TOL_PCT: float = 0.05       # تلورانس نسبی برای همپوشانی با S/R

    # وزن‌دهی مؤلفه‌های کانفلوئنس
    W_TREND: float = 10.0          # وزن ترند (MA slope)
    W_RSI: float = 10.0            # وزن RSI zone
    W_SR: float = 10.0             # وزن همپوشانی S/R

    # ------------------------ هِلپرهای مصرف ------------------------
    def sr_levels(self, ref_price: float) -> List[float]:
        """
        توضیح آموزشی (فارسی):
          بر اساس قیمت مرجع، سطوح رُند را به‌صورت متقارن می‌سازد تا
          به fib_cluster / fib_cluster_cfg پاس بدهیم.
        """
        return round_levels(anchor=ref_price, step=self.SR_STEP, count=self.SR_COUNT)

    def to_cluster_kwargs(self) -> Dict[str, Any]:
        """
        توضیح آموزشی (فارسی):
          پارامترهای مرتبط با خوشه‌بندی را در قالب یک دیکشنری آماده می‌کند
          تا به wrapper یا خود fib_cluster پاس داده شوند.
        """
        return {
            "tol_pct": self.TOL_PCT,
            "prefer_ratio": self.PREFER_RATIO,
            # وزن‌ها و تلورانس SR معمولاً در امضای wrapper مصرف می‌شوند:
            "w_trend": self.W_TREND,
            "w_rsi": self.W_RSI,
            "w_sr": self.W_SR,
            "sr_tol_pct": self.SR_TOL_PCT,
        }


""" --------------------------------------------------------------------------- OK Func3 (review:040924, 050214)
هِلپر عمومی S/R: تولید سطوح رُندِ متقارن پیرامون ref
"""
def round_levels(anchor: float, step: float, count: int = 10) -> List[float]:
    """
    تولید یک «شبکهٔ سطوح رُند» حول مقدار anchor با فاصلهٔ step.
    مثال: round_levels(1945.3, 10, n=5) → [1895, 1905, ..., 1995]

    پارامترها:
      anchor: لنگر قیمتی (مثلاً آخرین قیمت)
      step: فاصلهٔ شبکه (مثلاً 10.0 برای طلا، یا 0.5 …)
      n: چند سطح به بالا/پایین (دوطرفه)

    خروجی: لیست سطوح رُند (کوچک به بزرگ)
    """
    if step <= 0:
        raise ValueError("step must be positive")

    base = np.floor(anchor / step) * step  # کف رند نزدیک
    levels = [base + k * step for k in range(-count, count + 1)]
    return sorted(levels)


""" --------------------------------------------------------------------------- OK Func4 (review:040924, 050214)
استخراج نمای استانداردِ OHLC برای TF خواسته‌شده از روی ستون‌های پیشونددار.
خروجی: DataFrame با ستون‌های ['open','high','low','close','volume','spread'] (هر کدام که موجود باشد)
"""
def get_ohlc_view(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    # استخراج ستونها متناسب با 6 ستون استاندارد
    cols = {}
    for k in ["open", "high", "low", "close", "volume", "spread"]:
        c = f"{tf}_{k}"
        if c in df.columns:
            cols[k] = df[c]
    
    # حذف سطرهایی که تمام ستونهای آنها NaN باشد
    out = pd.DataFrame(cols).dropna(how="all")
    if out.empty:
        raise ValueError(f"OHLC for TF={tf} not found")

    # اگر tick_volume داریم، یک alias به نام volume هم می‌سازیم
    if "tick_volume" in out.columns and "volume" not in out.columns:
        out.rename(columns={"tick_volume": "volume"}, inplace=True)

    out.index = pd.to_datetime(out.index, utc=True)
    out.sort_index(inplace=True)
    return out


""" --------------------------------------------------------------------------- OK Func5 (review:040924, 050214)
یک هلپر عمومی برای انتخاب اولین ستون موجود از چند نامِ کاندید.
اولین ستونی که در دیتافریم موجود است را برمی‌گرداند؛ در غیر این‌صورت None.
"""
def pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[pd.Series]:

    for c in candidates:
        if c in df.columns:
            s = df[c]
            if s is not None and len(s) > 0:
                return s
    return None


""" --------------------------------------------------------------------------- OK Func6 (040924)
این تابع از روی نام ستون‌های دیتافریم می فهمد چه «تایم‌فریم»‌هایی داخل داده وجود دارد
 و برای هر تایم‌فریم، نگاشتی از ستون‌های واقعی به اسامی استاندارد OHLC می سازد.
مثال:
 {
  "M30": TFView(tf="M30", cols={"open":"M30_open", "high":"M30_high", "low":"M30_low", "close": "M30_close",
                                # اگر ستون‌های دیگری مثل volume/spread هم داشتیم، اینجا اضافه می‌شدند
                               }
                ),
  "H1":  TFView(tf="H1",  cols={"open":"H1_open", "high":"H1_high", "low":"H1_low", "close": "H1_close",
                               }
                )
}
 """
def detect_timeframes(df: pd.DataFrame) -> Dict[str, TFView]:
    buckets: Dict[str, Dict[str, str]] = {}
    for col in df.columns:
        m = _TF_REGEX.match(col)
        if not m:
            continue
        tf = m.group("tf").upper()
        field = m.group("field").lower()
        buckets.setdefault(tf, {})[field] = col
    return {tf: TFView(tf=tf, cols=mapping) for tf, mapping in buckets.items()}


""" --------------------------------------------------------------------------- OK Func7 (review:040924)
برش یک TF با استانداردسازی نام ستون‌ها
استخراج ستون‌های یک تایم‌فریم و بازسازی نام‌های استاندارد OHLC و volume
- df: دیتافریم اصلی با ستون‌های پیشونددار TF
- view: نگاشت استاندارد نام ستون‌ها به نام واقعی دیتافریم

خروجی:
    DataFrame با ستون‌های ['open','high','low','close','volume','spread'] (هر کدام موجود باشد)
"""
def slice_tf(df: pd.DataFrame, view: TFView) -> pd.DataFrame:

    cols = list(view.cols.values())       # فقط ستون‌های واقعی دیتافریم
    rename_map = {c: k_std for k_std, c in view.cols.items()}  # نگاشت مستقیم به نام استاندارد

    sdf = df[cols].rename(columns=rename_map).copy()
    return sdf


""" --------------------------------------------------------------------------- OK Func8 (review:040924)
نگهبان NaN و dtype سبک
"""
def nan_guard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_integer_dtype(df[c]):
            # Nullable Int64 را دست‌نخورده می‌گذاریم
            pass
    return df


""" --------------------------------------------------------------------------- OK Func9 (review:040924)
z-Score ساده یا همان نرمال سازی
آموزشی:
rolling یک پنجرهٔ متحرک روی سری می‌سازد و 
min_periods مشخص می‌کند که چند داده معتبر لازم است تا آماره محاسبه شود
"""
def zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or window
    # above line is equal to: 
    # mp = window if min_periods is None else min_periods
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std()
    return ((s - mean) / std.replace(0, np.nan)).astype("float32")


""" --------------------------------------------------------------------------- OK Func10 (New 050208)
True Range (برای ATR و ...)
"""
def true_range_pandas(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:    
    """
    World-class True Range (TR)
    مطابق استاندارد Wilder's ATR
    بدون هیچ مقدار ساختگی / سازگار 100% با TA-Lib
    """

    h = high.astype("float64")
    l = low.astype("float64")
    c = close.astype("float64")

    # prevClose
    prev = c.shift(1)

    # سه کاندید TR
    tr1 = (h - l).abs()
    tr2 = (h - prev).abs()
    tr3 = (l - prev).abs()

    # True Range = max of three
    out = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # اولین مقدار TR = high-low (طبق استاندارد جهانی)
    if len(out) > 0:
        out.iloc[0] = tr1.iloc[0]

    return out.rename("true_range")


@njit(cache=True)
def _true_range_njit(high, low, close):
    length = high.size
    tr = np.empty(length)
    tr[0] = high[0] - low[0]  # first bar: standard TR definition

    for i in range(1, length):
        h_l = high[i] - low[i]
        h_pc = abs(high[i] - close[i - 1])
        l_pc = abs(low[i] - close[i - 1])
        tr[i] = max(h_l, h_pc, l_pc)
    return tr

# --- WRAPPERs ---
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    if len(high) < 500_000:
        return true_range_pandas(high=high, low=low, close=close)
    idx = close.index
    high = high.to_numpy(np.float64)
    low = low.to_numpy(np.float64)
    close = close.to_numpy(np.float64)
    out = _true_range_njit(high, low, close)
    return pd.Series(out, index=idx, dtype="float64", name="true_range")

""" --------------------------------------------------------------------------- OK Func11 (New 050208)
Exponential Moving Average (EMA)  ==>  alpha = 2/(n+1)
"""
def _ema_numpy(s: pd.Series, n: int, min_periods: int = None) -> pd.Series:
    """
    Ultra-fast EMA compatible with TA-Lib / TradingView
    - Initialization = SMA(n)
    - No look-ahead
    - Supports min_periods
    - MUCH faster than pandas.ewm on multi-million rows
    """
    if s is None:
        logger.warning("ema_fast(): input is None")
        return pd.Series(dtype="float64", name=f"ema_{n}")

    arr = s.to_numpy(dtype=np.float64)
    length = len(arr)

    if length == 0:
        return pd.Series(dtype="float64", index=s.index, name=f"ema_{n}")

    out = np.full(length, np.nan, dtype=np.float64)

    alpha = 2.0 / (n + 1)

    # ------------ Case 1: data shorter than min_periods ------------
    if length < min_periods:
        return pd.Series(out, index=s.index, dtype="float64", name=f"ema_{n}")

    # ------------ Case 2: data >= min_periods ------------

    # ---- warm-up sum for SMA initialization ----
    if length >= n:
        sma_init = np.sum(arr[:n]) / n
        ema_prev = sma_init
        start = n - 1
        out[start] = sma_init
    else:
        # When length < n but >= min_periods
        # Warm-up SMA with available data (partial window)
        sma_init = np.sum(arr[:length]) / length
        ema_prev = sma_init
        start = length - 1
        out[start] = sma_init

    # ---- recursive EMA ----
    for i in range(start + 1, length):
        ema_prev = arr[i] * alpha + ema_prev * (1 - alpha)
        out[i] = ema_prev

    # ---- min_periods handling ----
    # We fill valid values from (min_periods - 1) onward
    if start > (min_periods - 1):
        # warm-up from min_periods-1 to start-1 with SMA(min_window)
        # like pandas (progressive warm-up)
        for t in range(min_periods - 1, start):
            k = t + 1
            window_mean = np.sum(arr[:k]) / k
            out[t] = window_mean

    # done
    return pd.Series(out, index=s.index, dtype="float64", name=f"ema_{n}")

@njit(cache=True)
def _ema_njit(tr, window, min_periods):
    n = len(tr)
    out = np.empty(n, dtype=np.float64)
    alpha = 2.0 / (window + 1)

    # --- fill all with NaN initially ---
    for i in range(n):
        out[i] = np.nan

    if n < min_periods:
        return out

    # --------------------------------------------------
    # 1) Progressive warm-up (partial SMA)
    # --------------------------------------------------
    s = 0.0
    limit = window if n >= window else n

    for i in range(limit):
        s += tr[i]

        if i >= min_periods - 1:
            out[i] = s / (i + 1)

    # --------------------------------------------------
    # 2) If we have full window → switch to real EMA
    # --------------------------------------------------
    if n >= window:
        e = s / window
        out[window - 1] = e

        for i in range(window, n):
            e = (tr[i] * alpha) + (e * (1.0 - alpha))
            out[i] = e

    return out

# --- Wrapper ---
def ema(s: pd.Series, n: int, min_periods: int = None) -> pd.Series:

    # --- normalize min_periods ---
    if min_periods is None:
        min_periods = n
    if min_periods < 1:
        min_periods = 1
    if min_periods > n:
        min_periods = n

    if len(s) < 750_000:   # بصورت تجربی و با سعی و خطا بدست آمده است
        return _ema_numpy(s=s, n=n, min_periods=min_periods)
    else:
        idx = s.index
        s = s.to_numpy(np.float64)
        out = _ema_njit(tr=s, window=n, min_periods=min_periods)
        return pd.Series(out, index=idx, dtype="float64", name=f"ema_{n}")


""" --------------------------------------------------------------------------- OK Func12 (New 050208)
Compute ATR (ATR)
"""

@njit(cache=True)
def _rma_wilder(tr, window):     # ==> alpha = 1/n
    n = tr.size
    out = np.empty(n, dtype=np.float64)
    alpha = 1.0 / window

    # warmup: SMA initial
    s = 0.0
    for i in range(window):
        s += tr[i]
    r = s / window
    out[window - 1] = r

    # recursive Wilder RMA
    for i in range(window, n):
        r = (tr[i] * alpha) + (r * (1.0 - alpha))
        out[i] = r

    # leading values before window-1 = NaN
    for i in range(window - 1):
        out[i] = np.nan
    return out


@njit(cache=True)
def _sma_classic(tr, window, min_periods=None):           # by loops
    if min_periods is None:
        min_periods = window
    if min_periods < 1:
        min_periods = 1
    if min_periods > window:
        min_periods = window

    n = len(tr)
    out = np.empty(n, dtype=np.float64)
    s = 0.0

    for i in range(window - 1):
        s += tr[i]
        if i >= min_periods - 1:
            out[i] = s / (i + 1)
        else:
            out[i] = np.nan

    s += tr[window - 1]
    out[window - 1] = s / window

    # sliding window SMA
    for i in range(window, n):
        s += tr[i]
        s -= tr[i - window]
        out[i] = s / window

    return out

def _sma_classic_vecnumpy(tr, window, min_periods=None):  # NOT USED, But: NOT DELETE
    tr = np.asarray(tr, dtype=np.float64)
    n = tr.size

    if min_periods is None:
        min_periods = window

    min_periods = max(1, min(min_periods, window))

    out = np.full(n, np.nan, dtype=np.float64)

    csum = np.cumsum(tr)

    # progressive means (for min_periods < window)
    idx = np.arange(min_periods - 1, min(window, n))
    if idx.size > 0:
        out[idx] = csum[idx] / (idx + 1)

    # full window SMA
    if n >= window:
        out[window - 1:] = (csum[window - 1:] - np.concatenate(([0.0], csum[:-window]))) / window

    return out


# --- Main Function ---
def compute_atr(
        df: pd.DataFrame,
        window: int = 14,
        method: str = "wilder",
        min_periods: int = None,  # if None: min_periods = window
    ) -> pd.Series:
    """
    Wilder/EMA/Classic ATR. Returns a pandas Series aligned with df.index (hybrid interface).

    Parameters
    ----------
    df : ['high', 'low', 'close'] pandas Dataframe
    window : int, ATR window
    method : str, one of ['classic', 'wilder', 'ema']
        "classic": میانگین سادهٔ TR (SMA) با min_periods نیم‌پنجره (مطابق نسخهٔ خودت)
        "wilder" : هموارسازی وایلدر با α = 1/window
        "ema"    : هموارسازی نمایی رایج با α = 2/(window+1)
    min_periods  : حداقل دیتاهای لازم برای شروع محاسبات در پنجره
                   اگر تعریف نشود، برابر با طول پنجره در نظر گرفته میشود

    Returns
    -------
    ATR : pd.Series float64, aligned with df.index
    """
    # --- نگهبان‌های ورودی ---
    # t1 = datetime.now() ######################========############
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_periods is None:
        min_periods = window
    if not {"high", "low", "close"}.issubset(set(df.columns)):
        raise ValueError("DF must contain columns: high, low, close")
    # t2 = datetime.now() ######################========############

    high  = df["high"] .to_numpy(dtype=np.float64)
    low   = df["low"]  .to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    # t3 = datetime.now() ######################========############


    # --- True Range ---
    tr = _true_range_njit(high, low, close)
    # t4 = datetime.now() ######################========############

    # --- نرمال‌سازی روش ---
    m = (method or "wilder").strip().lower()
    # t5 = datetime.now() ######################========############

    if m == "wilder":
        out = _rma_wilder(tr, window)
    elif m == "classic":
        out = _sma_classic(tr, window, min_periods)
        # out = _sma_classic_vecnumpy(tr, window, min_periods)  # NOT DELETE
    elif m == "ema":
        out = ema(tr, window, min_periods)
    else:
        raise ValueError(f"Invalid ATR mode: {method}. Use wilder/classic/ema.")
    # t6 = datetime.now() ######################========############
    # print(f"compute_atr: Time t1 to t2: {round((t2 - t1).total_seconds(), 4)} seconds")
    # print(f"compute_atr: Time t2 to t3: {round((t3 - t2).total_seconds(), 4)} seconds")
    # print(f"compute_atr: Time t3 to t4: {round((t4 - t3).total_seconds(), 4)} seconds")
    # print(f"compute_atr: Time t4 to t5: {round((t5 - t4).total_seconds(), 4)} seconds")
    # print(f"compute_atr: Time t5 to t6: {round((t6 - t5).total_seconds(), 4)} seconds")
    return pd.Series(out.astype(np.float64), index=df.index, name="atr_hybrid")


""" --------------------------------------------------------------------------- OK Func13
Swing Detection (H/L)
English:
    Local swing detection without SciPy:
    - A point is swing-high if it's the maximum in a ±min_distance window.
    - A point is swing-low  if it's the minimum in a ±min_distance window.
    - Optional 'prominence' and 'atr_mult * ATR' filters to drop weak swings.
Persian:
    تشخیص قله/کف محلی بدون SciPy:
    - بیشینه/کمینه در پنجرهٔ ±min_distance
    - فیلتر اختیاری بر اساس prominence و همچنین آستانهٔ ATR (atr_mult * ATR)

# نکته: price باید ایندکس زمانی UTC و مرتب داشته باشد.

ورودی‌ها:
    price: Series اندیس‌گذاری‌شده بر حسب زمان (UTC)
    prominence: حداقل برجستگی نسبت به لبه‌های پنجره (اختیاری)
    min_distance: نصفِ اندازهٔ پنجره به دو طرف
    atr: سری ATR هم‌تراز (اختیاری)
    atr_mult: اگر داده شود، آستانهٔ حذف سوئینگ‌های ضعیف = atr_mult * ATR
    tf: نام تایم‌فریم برای متادیتا (اختیاری)

خروجی:
    DataFrame با ایندکس زمانی، ستون‌ها: ['price','kind','atr','tf']
    kind ∈ {'H','L'}

"""
def detect_swings_old1(   # بر اساس (یک) سری است. نه بر اساس سریهای (سقف وکف)
    price: pd.Series,
    prominence: Optional[float] = None,   # معنی: برجستگی، امتیاز، برتری- حداقل برجستگی یک کندل نسبت به دو کندل مجاورش
    min_distance: int = 5,                # نصف پهنای پنجره لغزان
    atr: Optional[pd.Series] = None,
    atr_mult: Optional[float] = None,
    tf: Optional[str] = None,
    ) -> pd.DataFrame:

    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas Series indexed by time")

    # تضمین ترتیب زمانی سریِ ورودی (پایدار): اگر مرتب نیست، یک‌بار پایدار مرتب می‌کنیم
    if not price.index.is_monotonic_increasing:
        price = price.sort_index(kind="stable")
        #kind="stable" تضمین می‌کند که در صورت برابری ایندکس‌ها، ترتیب قبلی داده‌ها حفظ شود

    idx = price.index # یک اندکس است و معمولاً در چنین پروژه ای انتظار میرود که از نوع DatatimeIndex باشد
    n = len(price)
    if n < (2 * min_distance + 1):
        logger.debug("detect_swings: insufficient length (n=%d, min_distance=%d)", n, min_distance)
        return pd.DataFrame(columns=["price", "kind", "atr", "tf"])

    highs: list[Tuple[pd.Timestamp, float]] = []
    lows:  list[Tuple[pd.Timestamp, float]] = []

    # پنجره‌ی لغزان برای اکسترمم محلی
    for i in range(min_distance, n - min_distance):
        p = price.iloc[i]  # مقدار عضو آی-‌ام سری را بر اساس موقعیت عددی، و مستقل از نام اندکس، برمی‌گرداند. 
        left = price.iloc[i - min_distance : i]
        right = price.iloc[i + 1 : i + 1 + min_distance]

        is_high = p >= left.max() and p >= right.max()
        is_low  = p <= left.min() and p <= right.min()

        if not (is_high or is_low):
            continue

        prom_left  = abs(p - price.iloc[i-1])  # prom_left  = abs(p - left.iloc[-1])
        prom_right = abs(p - price.iloc[i+1])  # prom_right = abs(p - right.iloc[0])        

        # فیلتر ساده پرومیننس: فاصله از نزدیک‌ترین همسایهٔ طرفین
        if (prominence is not None) and (prominence > 0):
            if ((prom_left < prominence) or (prom_right < prominence)):
                continue

        # فیلتر مبتنی بر ATR (اگر دادهٔ ATR و ضریب atr_mult داده شده باشد)
        atr_here = float(atr.iloc[i]) if (atr is not None and pd.notna(atr.iloc[i])) else np.nan
        if (atr is not None) and (atr_mult is not None) and (atr_mult > 0) and pd.notna(atr_here):
            local_prom = max(prom_left, prom_right)
            if local_prom < atr_mult * atr_here:
                continue

        # اگر p توسط هر دو فیلتر قبول شود:
        ts = idx[i]  # ts: TimeStamp (of this swing)
        if is_high:
            highs.append((ts, float(p)))
        if is_low:
            lows.append((ts, float(p)))

    # خروجی یکدست بصورت یک لیست از دیکشنری ها
    rows: List[Dict] = []
    
    for ts, val in highs:
        atr_value = float(atr.loc[ts]) if (atr is not None and ts in atr.index) else np.nan
        rows.append({"ts": ts, "price": val, "kind": "H", "atr": atr_value, "tf": tf})
    for ts, val in lows:
        atr_value = float(atr.loc[ts]) if (atr is not None and ts in atr.index) else np.nan
        rows.append({"ts": ts, "price": val, "kind": "L", "atr": atr_value, "tf": tf})

    swings = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

    # تعریف و تنظیم اندکس دیتافریم نهایی
    if not swings.empty:
        swings.set_index("ts", inplace=True)                   # ستون ts=timestamp را به اندکس دیتافریم تبدیل میکند
        swings.index = pd.to_datetime(swings.index, utc=True)
    else:
        logger.debug("detect_swings: no swings detected")
    
    return swings

def detect_swings_new1(   # براساس زیگزاگ است. اما با همان پارامترهای تابع قدیمی
    high: pd.Series,
    low: pd.Series,
    prominence: Optional[float] = None,
    min_distance: int = 5, 
    atr: Optional[pd.Series] = None,
    atr_mult: Optional[float] = None,
    tf: Optional[str] = None,
) -> pd.DataFrame:
    """
    تشخیص Swing High / Swing Low بر اساس سری‌های high و low.
    سازگار با الگوریتم zigzag موجود در پروژه.
    خروجی:
        index = timestamp
        price
        kind = {"H","L"}
        atr
        tf
    """

    # --------------- Validation ---------------
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series):
        raise TypeError("high and low must be pandas Series indexed by time")

    if not high.index.equals(low.index):
        raise ValueError("high and low must have the same index")

    if not high.index.is_monotonic_increasing:
        high = high.sort_index(kind="stable")
        low = low.sort_index(kind="stable")

    n = len(high)
    if n < (2 * min_distance + 1):
        return pd.DataFrame(columns=["price", "kind", "atr", "tf"])

    # --------------- Zigzag پارامترهای ---------------
    depth = int(min_distance)
    backstep = depth
    deviation = prominence or 0.0

    # --------------- اجرای ZigZag اصلی پروژه ---------------
    zz = zigzag(
        high=high,
        low=low,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        final_check = True,
    )

    # --------------- استخراج نقاط Pivot ---------------
    piv = zz[zz["state"] != 0].copy()
    if piv.empty:
        return pd.DataFrame(columns=["price", "kind", "atr", "tf"])

    # state: +1 = high swing , -1 = low swing
    piv["kind"] = np.where(piv["state"] == 1, "H", "L")
    piv["price"] = np.where(piv["state"] == 1, piv["high_zz"], piv["low_zz"])

    # --------------- ATR فیلترینگ ---------------
    if atr is not None:
        piv["atr"] = atr.reindex(piv.index).astype(float)
    else:
        piv["atr"] = np.nan

    if atr is not None and atr_mult is not None and atr_mult > 0:
        prev_price = high.reindex(piv.index).shift(1)
        diff = (piv["price"] - prev_price).abs()
        piv = piv[diff >= piv["atr"] * atr_mult]

    # --------------- افزودن tf ---------------
    piv["tf"] = tf

    # خروجی نهایی
    out = piv[["price", "kind", "atr", "tf"]].copy()
    out.index = pd.to_datetime(out.index, utc=True)

    return out

""" -------------------------------------
Base: استخراج پیوت های قطعی و قابل اعتماد => تنها نسخهٔ مناسب فیچر.
"""
def detect_swings_new20(   # براساس زیگزاگ است با پارامترهای جدید برای تابع
    high: pd.Series,
    low: pd.Series,
    *,
    depth: int,
    deviation: float = 0.0,
    backstep: Optional[int] = None,
    atr: Optional[pd.Series] = None,
    atr_mult: Optional[float] = None,
    tf: Optional[str] = None,
) -> pd.DataFrame:
    """
    این نسخه همان base / feature-ready / pivot-based swing detector است
    که باید در ساخت فیبوناچی و تمام فیچرهای training-backtest-live استفاده شود
    -------------------------
    World-class swing extractor built on top of zigzag().

    Inputs:
        high, low : pd.Series (same index, monotonic)
        depth     : zigzag depth
        deviation : zigzag deviation
        backstep  : zigzag backstep (default = depth)
        atr       : optional ATR series
        atr_mult  : ATR filter multiplier
        tf        : timeframe label

    Output DataFrame:
        index : timestamp (UTC)
        price : swing price (high or low)
        kind  : "H" | "L"
        atr   : ATR value at swing (if provided)
        tf    : timeframe tag
    """

    # ---------- Validation ----------
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series):
        raise TypeError("high and low must be pandas Series")

    if not high.index.equals(low.index):
        raise ValueError("high and low must share the same index")

    if depth <= 0:
        raise ValueError("depth must be positive")

    if backstep is None:
        backstep = depth

    if not high.index.is_monotonic_increasing:
        high = high.sort_index(kind="stable")
        low = low.sort_index(kind="stable")

    if len(high) < depth * 2 + 1:
        return pd.DataFrame(columns=["price", "kind", "atr", "tf"])

    # ---------- ZigZag Core ----------
    zz = zigzag(
        high=high,
        low=low,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
    )

    piv = zz.loc[zz["state"] != 0, ["state", "high_zz", "low_zz"]].copy()
    if piv.empty:
        return pd.DataFrame(columns=["price", "kind", "atr", "tf"])

    # ---------- Swing Extraction ----------
    piv["kind"] = np.where(piv["state"] == 1, "H", "L")
    piv["price"] = np.where(piv["state"] == 1, piv["high_zz"], piv["low_zz"])

    # ---------- ATR ----------
    if atr is not None:
        piv["atr"] = atr.reindex(piv.index).astype(float)
    else:
        piv["atr"] = np.nan

    if atr is not None and atr_mult and atr_mult > 0:
        prev = piv["price"].shift(1)
        piv = piv[(piv["price"] - prev).abs() >= piv["atr"] * atr_mult]

    # ---------- Final Output ----------
    out = piv[["price", "kind", "atr"]].copy()
    out["tf"] = tf
    out.index = pd.to_datetime(out.index, utc=True)

    return out

def detect_swings_new21(   # محاسبه ATR در همین تابع انجام میشود.
    df: pd.DataFrame,
    *,
    depth: int,
    deviation: float = 0.0,
    backstep: Optional[int] = None,
    # atr: Optional[pd.Series] = None,
    atr_mult: Optional[float] = None,
    tf: Optional[str] = None,
) -> pd.DataFrame:
    """
    این نسخه همان base / feature-ready / pivot-based swing detector است
    که باید در ساخت فیبوناچی و تمام فیچرهای training-backtest-live استفاده شود
    -------------------------
    World-class swing extractor built on top of zigzag().

    Inputs:
        high, low : pd.Series (same index, monotonic)
        depth     : zigzag depth
        deviation : zigzag deviation
        backstep  : zigzag backstep (default = depth)
                    atr       : optional ATR series ( => Deleted)
        atr_mult  : ATR filter multiplier
        tf        : timeframe label

    Output DataFrame:
        index : timestamp (UTC)
        price : swing price (high or low)
        kind  : "H" | "L"
                    atr   : ATR value at swing (if provided) ( => Deleted)
        tf    : timeframe tag
    """

    # ---------- Validation ----------
    high = df["high"]
    low  = df["low"]

    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series):
        raise TypeError("high and low must be pandas Series")

    if not high.index.equals(low.index):
        raise ValueError("high and low must share the same index")

    if depth <= 0:
        raise ValueError("depth must be positive")

    if backstep is None:
        backstep = depth

    if not high.index.is_monotonic_increasing:
        high = high.sort_index(kind="stable")
        low = low.sort_index(kind="stable")

    if len(high) < depth * 2 + 1:
        return pd.DataFrame(columns=["price", "kind", "atr", "tf"])


    atr_window = depth
    method = "wilder"
    min_periods = atr_window
    atr = compute_atr(df=df, window=atr_window, method=method, min_periods=min_periods)

    # ---------- ZigZag Core ----------
    zz = zigzag(
        high=high,
        low=low,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
    )

    piv = zz.loc[zz["state"] != 0, ["state", "high_zz", "low_zz"]].copy()
    if piv.empty:
        return pd.DataFrame(columns=["price", "kind", "atr", "tf"])

    # ---------- Swing Extraction ----------
    piv["kind"] = np.where(piv["state"] == 1, "H", "L")
    piv["price"] = np.where(piv["state"] == 1, piv["high_zz"], piv["low_zz"])

    # ---------- ATR ----------
    if atr is not None:
        piv["atr"] = atr.reindex(piv.index).astype(float)
    else:
        piv["atr"] = np.nan

    if atr is not None and atr_mult and atr_mult > 0:
        prev = piv["price"].shift(1)
        piv = piv[(piv["price"] - prev).abs() >= piv["atr"] * atr_mult]

    # ---------- Final Output ----------
    out = piv[["price", "kind", "atr"]].copy()
    out["tf"] = tf
    out.index = pd.to_datetime(out.index, utc=True)

    return out

def detect_swings(   # tf حذف شد- همراه با تغییرات دیگر
    df: pd.DataFrame,
    *,
    depth: int,
    deviation: float = 0.0,
    backstep: Optional[int] = None,
) -> pd.DataFrame:
    """
    این نسخه همان base / feature-ready / pivot-based swing detector است
    که باید در ساخت فیبوناچی و تمام فیچرهای training-backtest-live استفاده شود
    -------------------------
    World-class swing extractor built on top of zigzag().

    Inputs:
        df        : pd.DataFrame (same index, monotonic)
        depth     : zigzag depth
        deviation : zigzag deviation
        backstep  : zigzag backstep (default = depth)

    Output DataFrame:
        index : timestamp (UTC)
        price : swing price (high or low)
        kind  : "H" | "L"
        atr   : atr values
    """

    # ---------- Validation ----------
    # --- df ---
    if not df.index.is_monotonic_increasing:
        df = df.sort_index(kind="stable")
    
    # --- hlc ---
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")
    
    # --- depth ---
    if not isinstance(depth, int) or depth <= 0:
        raise ValueError("depth must be a positive integer")

    # --- backstep ---
    if backstep is None:
        backstep = depth
    if not isinstance(backstep, int) or backstep <= 0:
        raise ValueError("backstep must be a positive integer")
    
    # --- len ---
    high = df["high"]
    low  = df["low"]
    if len(high) < depth * 2 + 1:
        return pd.DataFrame(columns=["price", "kind", "atr"])

    # --- atr parameters ---
    atr_window = depth
    method = "wilder"
    min_periods = atr_window
    atr = compute_atr(df=df, window=atr_window, method=method, min_periods=min_periods)

    # ---------- ZigZag Core ----------
    zz = zigzag(
        high=high,
        low=low,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
    )

    piv = zz.loc[zz["state"] != 0, ["state", "high_zz", "low_zz"]].copy()
    if piv.empty:
        return pd.DataFrame(columns=["price", "kind", "atr"])

    # ---------- Swing Extraction ----------
    piv["kind"] = np.where(piv["state"] == 1, "H", "L")
    piv["price"] = np.where(piv["state"] == 1, piv["high_zz"], piv["low_zz"])

    # ---------- ATR ----------
    if atr is not None:
        piv["atr"] = atr.reindex(piv.index).astype(float)
    else:
        piv["atr"] = np.nan

    # ---------- Final Output ----------
    out = piv[["price", "kind", "atr"]].copy()
    out.index = pd.to_datetime(out.index, utc=True)

    return out



""" --------------------------------------------------------------------------- OK Func14
Z-Score distance
English: Return (x - mu) / sigma with small epsilon for stability.
Persian: نرمال‌سازی فاصله با زی‌اسکور.
"""
def zscore_distance(x: float, mu: float, sigma: float, eps: float = 1e-12) -> float:

    s = abs(sigma) if sigma is not None else 0.0
    return float((x - mu) / (s + eps))


""" --------------------------------------------------------------------------- OK Func15
Nearest level distance
English: Find nearest level to 'price' and return distances (signed/abs) and the level.
Persian: نزدیک‌ترین سطح قیمتی به price را برمی‌گرداند.
"""
def nearest_level_distance(price: float, levels: Sequence[float]) -> Dict[str, float]:

    # 1) گارد ایمن‌تر برای توالی خالی
    if levels is None or len(levels) == 0:
        return {"nearest_level": float("nan"), "signed": float("nan"), "abs": float("nan")}    

    # 2) حذف مقادیر غیرمتناهی/NaN از levels
    clean = [lv for lv in levels if np.isfinite(lv)]
    if len(clean) == 0:
        return {"nearest_level": float("nan"), "signed": float("nan"), "abs": float("nan")}

    # 3) محاسبهٔ فاصله‌ها
    diffs = [price - lv for lv in clean]
    j = np.argmin([abs(d) for d in diffs])  # تبدیل به int لازم نیست
    return {"nearest_level": float(clean[j]), "signed": float(diffs[j]), "abs": float(abs(diffs[j]))}


""" --------------------------------------------------------------------------- OK Func16
ساخت سطوح فیبوی رتریسمنت برای «n لگ اخیر» از روی سوئینگ‌های بسته.
منظور از سوئینگ بسته، سوئینگی است که بعد از آن یک سوئینگ مخالف شکل گرفته است و دیگر قابل تغییر نیست
ورودی:
    - ohlc_df: DataFrame با ستون‌های open/high/low/close (ایندکس UTC مرتب)
    - n_legs: تعداد لگ‌های اخیر (پیش‌فرض 10)
    - ratios: نسبت‌های فیبو (اگر None → [0.236,0.382,0.5,0.618,0.786])
    - prominence/min_distance/atr_mult: پارامترهای فیلتر سوئینگ (برای حذف نویز)

خروجی:
    DataFrame ستون‌ها: ['ratio','price','leg_up','leg_idx']
    - leg_idx: شمارهٔ لگ از انتها (1 = آخرین لگ، 2 = یکی قبل‌تر، ...)
"""
def levels_from_recent_legs(
    ohlc_df: pd.DataFrame,
    n_legs: int = 10,
    ratios: Optional[Iterable[float]] = None,
    prominence: Optional[float] = None,         # معنی: برجستگی، امتیاز، برتری
    min_distance: int = 5,
    atr_mult: Optional[float] = 1.0,
    ) -> pd.DataFrame:

    # 1) اگر ratios ندادیم، لیست پیش‌فرض ساخته می‌شود. --------------
    if ratios is None:
        ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

    # 2) محاسبهٔ ATR برای فیلتر سوئینگ (در صورت نیاز) ---------------
    atr = None
    try:
        atr = compute_atr(ohlc_df, window=14, method="wilder")
    except Exception:
        pass

    # 3) محاسبه سوئینگ ها -------------------------------------------
    close = ohlc_df["close"].astype(float)
    try:
        swings = detect_swings(
            close,
            prominence=prominence,
            min_distance=min_distance,
            atr=atr,
            atr_mult=atr_mult,
            tf=None,
        )
    except Exception as ex:
        # اگر detect_swings در دسترس نبود
        return pd.DataFrame(columns=["ratio", "price", "leg_up", "leg_idx"])

    # 4) اگر سوئینگ کمتر از ۲ نقطه بود، خروجی خالی است --------------
    if swings is None or swings.empty or len(swings) < 2:
        return pd.DataFrame(columns=["ratio", "price", "leg_up", "leg_idx"])

    # 5) سورت نمودن با ایندکس و تبدیل لیست قیمت‌های سوئینگ به آرایه -
    s = swings.sort_index()
    prices = s["price"].astype(float).to_numpy()

    rows: List[dict] = []
    # از آخرین نقطه شروع می‌کنیم: (i-1 → i) یک لگ
    # i: اندیس آخرین سوئینگ، i-1: سوئینگ قبلی
    last_i = len(prices) - 1
    max_legs = min(n_legs, last_i)  # به تعداد جفت‌ها می‌تونیم لگ بسازیم

    # 6) از آخرین نقطه شروع می‌کند؛ هر جفت متوالی یک لگ می‌شود (p1→p2).
    for k in range(0, max_legs):
        i = last_i - k     # سوئینگ جدیدتر
        j = i - 1          # سوئینگ قبل از آن. یعنی قدیمی تر
        if j < 0:
            break
        p1, p2 = prices[j], prices[i]   # p1: old_price,   p2: new_price
        # 7) جهت لگ (leg_up) تعیین می‌شود، سپس low و high انتخاب می‌شود.
        leg_up = p2 > p1
        low, high = (p1, p2) if leg_up else (p2, p1)

        rng = high - low   # it means "range"
        if rng <= 0:
            continue
        # 8) برای هر نسبت فیبو، قیمت رتریسمنت محاسبه و در لیست rows ذخیره می‌شود.
        for r in ratios:
            # قیمت رتریسمنتِ لگ (استاندارد)
            price = (high - r * rng) if leg_up else (low + r * rng)
            rows.append({
                "ratio": float(r),
                "price": float(price),
                "leg_up": bool(leg_up),
                "leg_idx": int(k+1),   # آخرین لگ یا اخیرترین لگ (جدیدترین) دارای ایندکس 1 خواهد بود
            })
    # 9) دیتافریم نهایی ساخته و بر اساس price و leg_idx مرتب می‌شود.
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["price", "leg_idx"]).reset_index(drop=True)
    return out


# ===================================================================================== جدول زیر مانده
# تست پوشش کد (برای توسعه‌دهندگان) 
# =====================================================================================
""" Func Names                           Used in Functions: ...
                              1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
1  TFView                    --  --  --  --  --  ok  ok  --  --  --  --  --  --  --  --  --
2  FiboTestConfig            --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- (Not Used, Used in Test Files: check_wiring_fib_cluster.py)
3  round_levels              --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --  --
4  get_ohlc_view             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- Used in feature_engine.py
5  pick_first_existing       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- (Not Used)
6  detect_timeframes         --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- (Not Used)
7  slice_tf                  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- (Not Used)
8  nan_guard                 --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- Used in feature_engine.py
9  zscore                    --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- (Not Used)
10 true_range (wrapper)      --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  IMPORTANT +++
11 ema        (wrapper)      --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  IMPORTANT +++
12 compute_atr               --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  IMPORTANT +++
13 detect_swings             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  خیلی زیاد از این تابع استفاده شده
14 zscore_distance           --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- (Not Used)
15 nearest_level_distance    --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- Used in feature_registry.py
16 levels_from_recent_legs   --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- (Not Used, Used in Test Files: check_wiring_fib_cluster.py)
"""