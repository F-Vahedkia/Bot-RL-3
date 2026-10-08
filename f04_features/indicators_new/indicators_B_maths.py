
# این فایل همان فایل core.py است. منتها برای حالت live

import numpy as np
from numba.experimental import jitclass
from numba import njit, int32, float64, types, boolean
from numba.typed import List

# ============================================================================
# Helper function
# ============================================================================
EPS = 1e-9

@njit
def _safe_div(a, b):
    """Safe division that returns 0 when abs(b) < EPS (vectorized)"""
    return np.where(np.abs(b) < EPS, 0.0, a / b)

"""
Core classes:
-------------
SMA, WMA, EMA, ROC, RSI, TR, ATR, MACD, Bollinger, Keltner, Stochastic,
CCI, MFI, OBV, WilliamsR, ParabolicSARS, HeikinAshi

Extra trends classes:
---------------------
Supertrend, Aroon, DEMA, TEMA, KAMA, HMA,
"""
#  State()
# ============================================================================
# Core Indicator States  (PART-1)
# ============================================================================

# =================================================================== 1. SMAState
sma_spec = [
    ('n', int32),
    ('min_periods', int32),
    ('buf', float64[:]),
    ('sum', float64),
    ('idx', int32),
    ('count', int32),
]

@jitclass(sma_spec)
class SMAState:
    def __init__(self, n, min_periods=-1):
        self.n = n
        self.min_periods = n if min_periods == -1 else max(1, min(min_periods, n))
        self.buf = np.zeros(n, dtype=np.float64)
        self.sum = 0.0
        self.idx = 0
        self.count = 0

    def update(self, x):
        x = float(x)
        
        if self.count == self.n:
            # حذف قدیمی‌ترین مقدار
            self.sum -= self.buf[self.idx]
        else:
            self.count += 1
        
        # اضافه کردن مقدار جدید
        self.buf[self.idx] = x
        self.sum += x
        self.idx = (self.idx + 1) % self.n
        
        if self.count < self.min_periods:
            return np.nan
        return self.sum / self.count

    def reset(self):
        self.buf[:] = 0.0
        self.sum = 0.0
        self.idx = 0
        self.count = 0


# =================================================================== 2. WMAState
wma_spec = [
    ('n', int32),
    ('min_periods', int32),
    ('buf', float64[:]),
    ('weights_full', float64[:]),
    ('idx', int32),
    ('count', int32),
]

@jitclass(wma_spec)
class WMAState:
    def __init__(self, n, min_periods=-1):
        self.n = n
        self.min_periods = n if min_periods == -1 else max(1, min(min_periods, n))
        self.buf = np.zeros(n, dtype=np.float64)
        self.weights_full = np.arange(1, n + 1, dtype=np.float64)
        self.weights_full /= self.weights_full.sum()
        self.idx = 0
        self.count = 0

    def update(self, x):
        self.buf[self.idx] = float(x)
        self.idx = (self.idx + 1) % self.n
        
        if self.count < self.n:
            self.count += 1
        
        if self.count < self.min_periods:
            return np.nan
        
        # محاسبه weighted average
        result = 0.0
        weight_sum = 0.0
        
        for i in range(self.count):
            # خواندن از buffer به ترتیب قدیمی‌ترین تا جدیدترین
            buf_idx = (self.idx - self.count + i) % self.n
            weight = self.weights_full[self.n - self.count + i]
            result += self.buf[buf_idx] * weight
            weight_sum += weight
        
        return result / weight_sum

    def reset(self):
        self.buf[:] = 0.0
        self.idx = 0
        self.count = 0


# =================================================================== 3. EMAState
ema_spec = [
    ('n', int32),
    ('min_periods', int32),
    ('alpha', float64),
    ('value', float64),
    ('count', int32),
    ('initialized', types.boolean),
]

@jitclass(ema_spec)
class EMAState:
    def __init__(self, n, min_periods=-1):
        self.n = n
        self.min_periods = n if min_periods == -1 else max(1, min(min_periods, n))
        self.alpha = 2.0 / (n + 1.0)
        self.value = 0.0
        self.count = 0
        self.initialized = False

    def update(self, x):
        x = float(x)
        self.count += 1
        
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        
        if self.count < self.min_periods:
            return np.nan
        return self.value

    def reset(self):
        self.value = 0.0
        self.count = 0
        self.initialized = False


# =================================================================== 4. ROCState
roc_spec = [
    ('n', int32),
    ('buf', float64[:]),
    ('idx', int32),
    ('count', int32),
]

@jitclass(roc_spec)
class ROCState:
    def __init__(self, n):
        self.n = n
        self.buf = np.zeros(n + 1, dtype=np.float64)
        self.idx = 0
        self.count = 0

    def update(self, x):
        x = float(x)
        self.buf[self.idx] = x
        self.idx = (self.idx + 1) % (self.n + 1)
        
        if self.count < self.n + 1:
            self.count += 1
        
        if self.count <= self.n:
            return np.nan
        
        # قدیمی‌ترین مقدار
        old_idx = (self.idx - self.n - 1) % (self.n + 1)
        old_val = self.buf[old_idx]
        
        return (x - old_val) / old_val * 100.0

    def reset(self):
        self.buf[:] = 0.0
        self.idx = 0
        self.count = 0


# =================================================================== 5. RSIState
rsi_spec = [
    ('n', int32),
    ('method', int32),  # 0=ema, 1=wilders
    ('alpha', float64),
    ('prev', float64),
    ('gains', float64[:]),
    ('losses', float64[:]),
    ('avg_gain', float64),
    ('avg_loss', float64),
    ('idx', int32),
    ('count', int32),
    ('initialized', types.boolean),
    ('avg_initialized', types.boolean),
]

@jitclass(rsi_spec)
class RSIState:
    """
    method: Literal["ema", "wilders"] → method: int32 (0=ema, 1=wilders)
    prev=None → prev=0.0 + flag initialized
    avg_gain=None, avg_loss=None → مقادیر عددی + flag avg_initialized
    deque → آرایه NumPy با circular indexing

    نحوه استفاده:
    rsi = RSIState(n=14, method=0)  # 0=ema, 1=wilders
    result = rsi.update(100.5)
    """
    
    def __init__(self, n, method=0):  # 0=ema, 1=wilders
        self.n = n
        self.method = method
        self.prev = 0.0
        self.gains = np.zeros(n, dtype=np.float64)
        self.losses = np.zeros(n, dtype=np.float64)
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.idx = 0
        self.count = 0
        self.initialized = False
        self.avg_initialized = False

        # محاسبه alpha بر اساس method
        if method == 1:  # wilders
            self.alpha = 1.0 / n
        else:  # ema
            self.alpha = 2.0 / (n + 1)

    def update(self, close):
        close = float(close)
        
        if not self.initialized:
            self.prev = close
            self.initialized = True
            return np.nan

        d = close - self.prev
        gain = max(d, 0.0)
        loss = max(-d, 0.0)

        self.gains[self.idx] = gain
        self.losses[self.idx] = loss
        self.idx = (self.idx + 1) % self.n
        
        if self.count < self.n:
            self.count += 1

        if self.count < self.n:
            self.prev = close
            return np.nan

        if not self.avg_initialized:
            self.avg_gain = np.mean(self.gains)
            self.avg_loss = np.mean(self.losses)
            self.avg_initialized = True
        else:
            self.avg_gain = (1 - self.alpha) * self.avg_gain + self.alpha * gain
            self.avg_loss = (1 - self.alpha) * self.avg_loss + self.alpha * loss

        self.prev = close
        
        # if self.avg_loss == 0.0:
        #     rs = 0.0
        # else:
        #     rs = self.avg_gain / self.avg_loss
        if self.avg_loss == 0.0:
            if self.avg_gain == 0.0:
                return 50.0
            return 100.0
        rs = self.avg_gain / self.avg_loss

        return 100.0 - (100.0 / (1.0 + rs))

    def reset(self):
        self.prev = 0.0
        self.gains[:] = 0.0
        self.losses[:] = 0.0
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.idx = 0
        self.count = 0
        self.initialized = False
        self.avg_initialized = False


# =================================================================== 6. TrueRangeState
tr_spec = [
    ('prev_close', float64),
    ('initialized', types.boolean),
]

@jitclass(tr_spec)
class TrueRangeState:
    """
    prev_close=None → prev_close=0.0 + flag initialized
    """
    
    def __init__(self):
        self.prev_close = 0.0
        self.initialized = False

    def update(self, h, l, c):
        h, l, c = float(h), float(l), float(c)
        
        if not self.initialized:
            tr = h - l
            self.initialized = True
        else:
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        
        self.prev_close = c
        return tr

    def reset(self):
        self.prev_close = 0.0
        self.initialized = False


# =================================================================== 7. ATRState
atr_spec = [
    ('n', int32),
    ('method', int32),  # 0=classic, 1=wilder, 2=ema
    ('min_periods', int32),
    ('alpha', float64),
    ('tr', TrueRangeState.class_type.instance_type),
    ('values', float64[:]),
    ('atr', float64),
    ('idx', int32),
    ('count', int32),
    ('atr_initialized', types.boolean),
]

@jitclass(atr_spec)
class ATRState:
    """
    method: Literal → method: int32 (0=classic, 1=wilder, 2=ema)
    min_periods: Optional[int] → min_periods: int32 با مقدار پیش‌فرض -1
    atr=None → atr=0.0 + flag atr_initialized
    deque → آرایه NumPy
    استفاده از TrueRangeState به‌عنوان nested jitclass

    نحوه استفاده:
    atr = ATRState(n=14, method=1, min_periods=14)  # 0=classic, 1=wilder, 2=ema
    result = atr.update(high, low, close)
    """
    
    def __init__(self, n, method=1, min_periods=-1):  # 0=classic, 1=wilder, 2=ema
        self.n = n
        self.method = method
        self.min_periods = n if min_periods == -1 else min_periods
        self.tr = TrueRangeState()
        self.values = np.zeros(n, dtype=np.float64)
        self.atr = 0.0
        self.idx = 0
        self.count = 0
        self.atr_initialized = False

        # محاسبه alpha بر اساس method
        if method == 1:  # wilder
            self.alpha = 1.0 / n
        elif method == 2:  # ema
            self.alpha = 2.0 / (n + 1)
        else:  # classic
            self.alpha = 0.0

    def update(self, h, l, c):
        tr = self.tr.update(h, l, c)
        self.values[self.idx] = tr
        self.idx = (self.idx + 1) % self.n
        
        if self.count < self.n:
            self.count += 1

        if self.count < self.min_periods:
            return np.nan

        if self.method == 0:  # classic
            # میانگین ساده همیشه
            return np.mean(self.values[:self.count])
        else:
            # wilder یا ema
            if not self.atr_initialized:
                self.atr = np.mean(self.values[:self.count])
                self.atr_initialized = True
            else:
                self.atr = (1 - self.alpha) * self.atr + self.alpha * tr
            return self.atr

    def reset(self):
        self.tr.reset()
        self.values[:] = 0.0
        self.atr = 0.0
        self.idx = 0
        self.count = 0
        self.atr_initialized = False


# =================================================================== 8. MACDState
macd_spec = [
    ('fast', EMAState.class_type.instance_type),
    ('slow', EMAState.class_type.instance_type),
    ('sig', EMAState.class_type.instance_type),
]

@jitclass(macd_spec)
class MACDState:

    """
    استفاده از EMAState به‌عنوان nested jitclass

    نحوه استفاده:

    macd = MACDState(fast=12, slow=26, signal=9)
    macd_line, signal_line, histogram = macd.update(close)

    نکته:
    EMA مربوط به signal نباید NaNهای اولیه MACD را دریافت کند.
    بنابراین تا زمانی که slow EMA معتبر نشده است، signal EMA
    به‌روزرسانی نمی‌شود.
    """

    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = EMAState(fast, -1)
        self.slow = EMAState(slow, -1)
        self.sig = EMAState(signal, -1)

    def update(self, close):
        f = self.fast.update(close)
        s = self.slow.update(close)

        # تا زمانی که slow EMA معتبر نشده،
        # MACD و Signal و Histogram معتبر نیستند.
        if np.isnan(s):
            return np.nan, np.nan, np.nan

        macd = f - s

        # فقط MACD معتبر وارد Signal EMA می‌شود.
        sig = self.sig.update(macd)

        # Signal EMA خودش تا تکمیل min_periods مقدار NaN می‌دهد.
        if np.isnan(sig):
            return macd, np.nan, np.nan

        hist = macd - sig

        return macd, sig, hist

    def reset(self):
        self.fast.reset()
        self.slow.reset()
        self.sig.reset()


# =================================================================== 9. BollingerState
bollinger_spec = [
    ('n', int32),
    ('k', float64),
    ('min_periods', int32),
    ('buf', float64[:]),
    ('idx', int32),
    ('count', int32),
]

@jitclass(bollinger_spec)
class BollingerState:
    """
    deque(maxlen=n) → آرایه NumPy با circular indexing
    min_periods: Optional[int] → min_periods: int32 با مقدار پیش‌فرض -1

    نحوه استفاده:
    bb = BollingerState(n=20, k=2.0, min_periods=20)
    upper, mid, lower, w, p = bb.update(close)
    """
    
    def __init__(self, n, k, min_periods=-1):
        self.n = n
        self.k = k
        self.min_periods = n if min_periods == -1 else min_periods
        self.buf = np.zeros(n, dtype=np.float64)
        self.idx = 0
        self.count = 0

    def update(self, close):
        close = float(close)
        self.buf[self.idx] = close
        self.idx = (self.idx + 1) % self.n
        
        if self.count < self.n:
            self.count += 1

        if self.count < self.min_periods:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        
        arr = self.buf[:self.count]
        mid = np.mean(arr)
        std = np.std(arr)

        upper = mid + self.k * std
        lower = mid - self.k * std
                
        w = upper - lower
        p = (close - lower) / w if w != 0.0 else np.nan

        return upper, mid, lower, w, p

    def reset(self):
        self.buf[:] = 0.0
        self.idx = 0
        self.count = 0


# =================================================================== 10. KeltnerState
keltner_spec = [
    ('ema', EMAState.class_type.instance_type),
    ('atr', ATRState.class_type.instance_type),
    ('m', float64),
]

@jitclass(keltner_spec)
class KeltnerState:
    """
    استفاده از EMAState و ATRState به‌عنوان nested jitclass
    min_periods: Optional[int] → min_periods: int32 با مقدار پیش‌فرض -1

    نحوه استفاده:
    kc = KeltnerState(n=20, m=2.0, min_periods=20)
    upper, mid, lower, w, p = kc.update(high, low, close)
    """
    
    def __init__(self, n, m, min_periods=-1):
        self.ema = EMAState(n, min_periods)
        self.atr = ATRState(n, 1, min_periods)
        self.m = m

    def update(self, h, l, c):
        mid = self.ema.update(c)
        atr = self.atr.update(h, l, c)
        if np.isnan(atr):
            return np.nan, np.nan, np.nan, np.nan, np.nan
        
        upper = mid + self.m * atr
        lower = mid - self.m * atr

        w = upper - lower
        p = (c - lower) / w if w != 0.0 else np.nan

        return upper, mid, lower, w, p

    def reset(self):
        self.ema.reset()
        self.atr.reset()


# =================================================================== 11. StochasticState
stochastic_spec = [
    ('k_period', int32),
    ('d_period', int32),
    ('smooth_k', int32),
    ('method', int32),  # 0=sma, 1=ema
    ('min_periods', int32),
    ('highs', float64[:]),
    ('lows', float64[:]),
    ('idx', int32),
    ('count', int32),
    ('has_k_smoother', types.boolean),
    ('k_smoother_sma', SMAState.class_type.instance_type),
    ('k_smoother_ema', EMAState.class_type.instance_type),
    ('d_smoother_sma', SMAState.class_type.instance_type),
    ('d_smoother_ema', EMAState.class_type.instance_type),
]

@jitclass(stochastic_spec)
class StochasticState:
    """
    method: Literal["sma", "ema"] → method: int32 (0=sma, 1=ema)
    deque برای highs/lows → آرایه‌های NumPy با circular indexing
    min_periods: Optional[int] → min_periods: int32 با مقدار پیش‌فرض -1
    به دلیل محدودیت Numba در Optional nested jitclass، هر دو smoother (SMA و EMA) ایجاد می‌شوند اما فقط یکی استفاده می‌شود (دیگری dummy است)
    flag has_k_smoother برای مدیریت smooth_k > 1

    نحوه استفاده:
    stoch = StochasticState(k_period=14, d_period=3, smooth_k=3, method=0, min_periods=14)  # 0=sma, 1=ema
    k, d = stoch.update(high, low, close)
    """
    
    def __init__(self, k_period=14, d_period=3, smooth_k=3, method=0, min_periods=-1):
        """
        method: 0=sma, 1=ema
        """
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        self.method = method
        self.min_periods = k_period if min_periods == -1 else min_periods
        
        # بافرهای high/low
        self.highs = np.zeros(k_period, dtype=np.float64)
        self.lows = np.zeros(k_period, dtype=np.float64)
        self.idx = 0
        self.count = 0
        
        # هموارساز %K
        self.has_k_smoother = smooth_k > 1
        if self.has_k_smoother:
            if method == 0:  # sma
                self.k_smoother_sma = SMAState(smooth_k, 1)
                self.k_smoother_ema = EMAState(1       , 1)  # dummy
            else:  # ema
                self.k_smoother_sma = SMAState(1       , 1)  # dummy
                self.k_smoother_ema = EMAState(smooth_k, 1)
        else:
            self.k_smoother_sma = SMAState(1, 1)  # dummy
            self.k_smoother_ema = EMAState(1, 1)  # dummy
        
        # هموارساز %D
        if method == 0:  # sma
            self.d_smoother_sma = SMAState(d_period, 1)
            self.d_smoother_ema = EMAState(1       , 1)  # dummy
        else:  # ema
            self.d_smoother_sma = SMAState(1       , 1)  # dummy
            self.d_smoother_ema = EMAState(d_period, 1)

    def update(self, h, l, c):
        h, l, c = float(h), float(l), float(c)
        
        self.highs[self.idx] = h
        self.lows[self.idx] = l
        self.idx = (self.idx + 1) % self.k_period
        
        if self.count < self.k_period:
            self.count += 1

        if self.count < self.min_periods:
            return np.nan, np.nan

        # محاسبه %K خام
        arr_h = self.highs[:self.count]
        arr_l = self.lows[:self.count]
        hh = np.max(arr_h)
        ll = np.min(arr_l)
        
        if hh - ll == 0.0:
            raw_k = 0.0
        else:
            raw_k = 100.0 * (c - ll) / (hh - ll)
        
        # هموارسازی %K
        if self.has_k_smoother:
            if self.method == 0:  # sma
                k = self.k_smoother_sma.update(raw_k)
            else:  # ema
                k = self.k_smoother_ema.update(raw_k)
        else:
            k = raw_k
        
        # محاسبه %D
        if self.method == 0:  # sma
            d = self.d_smoother_sma.update(k)
        else:  # ema
            d = self.d_smoother_ema.update(k)
        
        return k, d

    def reset(self):
        self.highs[:] = 0.0
        self.lows[:] = 0.0
        self.idx = 0
        self.count = 0
        if self.has_k_smoother:
            if self.method == 0:
                self.k_smoother_sma.reset()
            else:
                self.k_smoother_ema.reset()
        if self.method == 0:
            self.d_smoother_sma.reset()
        else:
            self.d_smoother_ema.reset()


# =================================================================== 12. CCIState
cci_spec = [
    ('n', int32),
    ('min_periods', int32),
    ('tp', float64[:]),
    ('idx', int32),
    ('count', int32),
]

@jitclass(cci_spec)
class CCIState:
    """
    deque(maxlen=n) → آرایه NumPy با circular indexing
    min_periods: Optional[int] → min_periods: int32 با مقدار پیش‌فرض -1
    EPS به‌عنوان ثابت global تعریف شد

    نحوه استفاده:
    cci = CCIState(n=20, min_periods=20)
    result = cci.update(high, low, close)
    """
    
    def __init__(self, n, min_periods=-1):
        self.n = n
        self.min_periods = n if min_periods == -1 else min_periods
        self.tp = np.zeros(n, dtype=np.float64)
        self.idx = 0
        self.count = 0

    def update(self, h, l, c):
        tp = (float(h) + float(l) + float(c)) / 3.0
        self.tp[self.idx] = tp
        self.idx = (self.idx + 1) % self.n
        
        if self.count < self.n:
            self.count += 1

        if self.count < self.min_periods:
            return np.nan

        arr = self.tp[:self.count]
        ma = np.mean(arr)
        md = np.mean(np.abs(arr - ma))
        return (tp - ma) / (0.015 * md + EPS)
    
    def reset(self):
        self.tp[:] = 0.0
        self.idx = 0
        self.count = 0


# =================================================================== 13. MFIState
mfi_spec = [
    ('n', int32),
    ('min_periods', int32),
    ('prev_tp', float64),
    ('has_prev_tp', types.boolean),
    ('pos', float64[:]),
    ('neg', float64[:]),
    ('idx', int32),
    ('count', int32),
]

@jitclass(mfi_spec)
class MFIState:
    """
    prev_tp: Optional[float] → prev_tp: float64 + has_prev_tp: boolean
    deque(maxlen=n) برای pos/neg → آرایه‌های NumPy با circular indexing
    min_periods: Optional[int] → min_periods: int32 با مقدار پیش‌فرض -1    
    
    نحوه استفاده:
    mfi = MFIState(n=14, min_periods=14)
    result = mfi.update(high, low, close, volume)
    """
    
    def __init__(self, n, min_periods=-1):
        self.n = n
        self.min_periods = n if min_periods == -1 else min_periods
        self.prev_tp = 0.0
        self.has_prev_tp = False
        self.pos = np.zeros(n, dtype=np.float64)
        self.neg = np.zeros(n, dtype=np.float64)
        self.idx = 0
        self.count = 0

    def update(self, h, l, c, v):
        tp = (float(h) + float(l) + float(c)) / 3.0
        mf = tp * float(v)

        if not self.has_prev_tp:
            self.prev_tp = tp
            self.has_prev_tp = True
            return np.nan

        if tp > self.prev_tp:
            self.pos[self.idx] = mf
            self.neg[self.idx] = 0.0
        elif tp < self.prev_tp:
            self.pos[self.idx] = 0.0
            self.neg[self.idx] = mf
        else:
            self.pos[self.idx] = 0.0
            self.neg[self.idx] = 0.0

        self.idx = (self.idx + 1) % self.n
        if self.count < self.n:
            self.count += 1

        self.prev_tp = tp

        if self.count < self.min_periods:
            return np.nan

        sum_pos = np.sum(self.pos[:self.count])
        sum_neg = np.sum(self.neg[:self.count])

        # Formula: 100.0 * sum_pos / (sum_pos + sum_neg)
        denom = sum_pos + sum_neg
        if np.abs(denom) < EPS:
            return 50.0  # هر دو صفرند
        return 100.0 * sum_pos / denom

    def reset(self):
        self.prev_tp = 0.0
        self.has_prev_tp = False
        self.pos[:] = 0.0
        self.neg[:] = 0.0
        self.idx = 0
        self.count = 0


# =================================================================== 14. OBVState
obv_spec = [
    ('prev', float64),
    ('has_prev', types.boolean),
    ('value', float64),
]

@jitclass(obv_spec)
class OBVState:
    """
    prev: Optional[float] → prev: float64 + has_prev: boolean
    هیچ deque‌ای وجود نداشت، فقط مدیریت None

    نحوه استفاده:
    obv = OBVState()
    result = obv.update(close, volume)
    """
    
    def __init__(self):
        self.prev = 0.0
        self.has_prev = False
        self.value = 0.0

    def update(self, c, v):
        c = float(c)
        v = float(v)

        if not self.has_prev:
            self.prev = c
            self.has_prev = True
            return 0.0

        if c > self.prev:
            self.value += v
        elif c < self.prev:
            self.value -= v

        self.prev = c
        return self.value

    def reset(self):
        self.prev = 0.0
        self.has_prev = False
        self.value = 0.0


# =================================================================== 15. WilliamsRState
williamsr_spec = [
    ('n', int32),
    ('min_periods', int32),
    ('highs', float64[:]),
    ('lows', float64[:]),
    ('idx', int32),
    ('count', int32),
]

@jitclass(williamsr_spec)
class WilliamsRState:
    """
    deque(maxlen=n) برای highs/lows → آرایه‌های NumPy با circular indexing
    min_periods: Optional[int] → min_periods: int32 با مقدار پیش‌فرض -1

    نحوه استفاده:
    wr = WilliamsRState(n=14, min_periods=14)
    result = wr.update(high, low, close)
    """
    
    def __init__(self, n, min_periods=-1):
        self.n = n
        self.min_periods = n if min_periods == -1 else min_periods
        self.highs = np.zeros(n, dtype=np.float64)
        self.lows = np.zeros(n, dtype=np.float64)
        self.idx = 0
        self.count = 0

    def update(self, h, l, c):
        self.highs[self.idx] = float(h)
        self.lows[self.idx] = float(l)
        c = float(c)
        
        self.idx = (self.idx + 1) % self.n
        if self.count < self.n:
            self.count += 1

        if self.count < self.min_periods:
            return np.nan

        if self.count < self.n:
            hh = np.max(self.highs[:self.count])
            ll = np.min(self.lows[:self.count])
        else:
            hh = np.max(self.highs)
            ll = np.min(self.lows)
            
        denominator = hh - ll
        if abs(denominator) < EPS:
            return np.nan
        
        return -100.0 * (hh - c) / denominator

    def reset(self):
        self.highs[:] = 0.0
        self.lows[:] = 0.0
        self.idx = 0
        self.count = 0


# =================================================================== 16. ParabolicSARState
psar_spec = [
    ('af_start', float64),
    ('af_step', float64),
    ('af_max', float64),
    ('initialized', types.boolean),
    ('sar', float64),
    ('ep', float64),
    ('af', float64),
    ('uptrend', types.boolean),
]

@jitclass(psar_spec)
class ParabolicSARState:
    """
    هیچ deque‌ای وجود نداشت
    تمام فیلدهای state به صورت صریح در spec تعریف شدند
    initialized: boolean برای مدیریت اولین update

    نحوه استفاده:
    psar = ParabolicSARState(af_start=0.02, af_step=0.02, af_max=0.2)
    result = psar.update(high, low)
    """
    
    def __init__(self, af_start=0.02, af_step=0.02, af_max=0.2):
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max
        self.initialized = False
        self.sar = 0.0
        self.ep = 0.0
        self.af = 0.0
        self.uptrend = True

    def update(self, h, l):
        h, l = float(h), float(l)

        if not self.initialized:
            self.sar = l
            self.ep = h
            self.af = self.af_start
            self.uptrend = True
            self.initialized = True
            return np.nan

        self.sar = self.sar + self.af * (self.ep - self.sar)

        if self.uptrend:
            if h > self.ep:
                self.ep = h
                self.af = min(self.af + self.af_step, self.af_max)
            if l < self.sar:
                self.uptrend = False
                self.sar = self.ep
                self.ep = l
                self.af = self.af_start
        else:
            if l < self.ep:
                self.ep = l
                self.af = min(self.af + self.af_step, self.af_max)
            if h > self.sar:
                self.uptrend = True
                self.sar = self.ep
                self.ep = h
                self.af = self.af_start

        return self.sar

    def reset(self):
        self.initialized = False
        self.sar = 0.0
        self.ep = 0.0
        self.af = 0.0
        self.uptrend = True


# =================================================================== 17. HeikinAshiState
heikinashi_spec = [
    ('prev_open', float64),
    ('prev_close', float64),
    ('has_prev', types.boolean),
]

@jitclass(heikinashi_spec)
class HeikinAshiState:
    """
    prev_open/prev_close: Optional[float] → float64 + has_prev: boolean
    هیچ deque‌ای وجود نداشت

    نحوه استفاده:
    ha = HeikinAshiState()
    ha_open, ha_high, ha_low, ha_close = ha.update(open, high, low, close)
    """
    
    def __init__(self):
        self.prev_open = 0.0
        self.prev_close = 0.0
        self.has_prev = False

    def update(self, o, h, l, c):
        o, h, l, c = float(o), float(h), float(l), float(c)
        ha_close = (o + h + l + c) / 4.0

        if not self.has_prev:
            ha_open = (o + c) / 2.0
            self.has_prev = True
        else:
            ha_open = (self.prev_open + self.prev_close) / 2.0

        ha_high = max(h, ha_open, ha_close)
        ha_low = min(l, ha_open, ha_close)

        self.prev_open = ha_open
        self.prev_close = ha_close

        return ha_open, ha_high, ha_low, ha_close

    def reset(self):
        self.prev_open = 0.0
        self.prev_close = 0.0
        self.has_prev = False


# =============================================================================
# Extra Trends Indicator States (PART-2)
# =============================================================================
# =================================================================== 18. SupertrendState
supertrend_spec = [
    ('period', int32),
    ('multiplier', float64),
    ('min_periods', int32),
    ('atr_state', ATRState.class_type.instance_type),
    ('basic_upper', float64),
    ('basic_lower', float64),
    ('final_upper', float64),
    ('final_lower', float64),
    ('direction', int32),
    ('supertrend', float64),
    ('count', int32),
]

@jitclass(supertrend_spec)
class SupertrendState:
    """ State for Supertrend indicator (ATR-based trend following).
    
    نحوه استفاده:
    # Classic ATR
        st = SupertrendState(period=10, multiplier=3.0, atr_method=0, min_periods=10)

    # Wilder ATR (default)
        st = SupertrendState(period=10, multiplier=3.0, atr_method=1, min_periods=-1)

    # EMA ATR
        st = SupertrendState(period=10, multiplier=3.0, atr_method=2, min_periods=10)

    # Update
        supertrend_value, direction = st.update(high, low, close)
    """
    
    def __init__(self, period=10, multiplier=3.0, atr_method=0, min_periods=-1):
        """
        Args:
            period: ATR period
            multiplier: Band multiplier
            atr_method: ATR calculation method (0=classic, 1=wilder, 2=ema)
            min_periods: Minimum periods before returning valid values (default: period)
        """
        self.period = period
        self.multiplier = multiplier
        self.min_periods = period if min_periods == -1 else min_periods
        
        # ATR state
        self.atr_state = ATRState(period, atr_method, -1)
        
        # Band state
        self.basic_upper = 0.0
        self.basic_lower = 0.0
        self.final_upper = 0.0
        self.final_lower = 0.0
        
        # Trend state
        self.direction = 1  # 1 = uptrend, -1 = downtrend
        self.supertrend = 0.0
        
        # Warm-up counter
        self.count = 0
    
    def update(self, high, low, close):
        """
        Update Supertrend state incrementally.
        
        Returns:
            (supertrend, direction): Supertrend value and direction (1 or -1)
        """
        high, low, close = float(high), float(low), float(close)
        self.count += 1
        
        # Update ATR
        atr = self.atr_state.update(high, low, close)
        
        # Warm-up period
        if self.count < self.min_periods or np.isnan(atr):
            return np.nan, self.direction
        
        # Calculate basic bands
        hl_avg = (high + low) / 2.0
        self.basic_upper = hl_avg + self.multiplier * atr
        self.basic_lower = hl_avg - self.multiplier * atr
        
        # Calculate final bands
        if self.final_upper == 0.0:
            self.final_upper = self.basic_upper
        else:
            self.final_upper = (self.basic_upper 
                               if self.basic_upper < self.final_upper or close > self.final_upper 
                               else self.final_upper)
        
        if self.final_lower == 0.0:
            self.final_lower = self.basic_lower
        else:
            self.final_lower = (self.basic_lower 
                               if self.basic_lower > self.final_lower or close < self.final_lower 
                               else self.final_lower)
        
        # Determine direction and supertrend
        if self.supertrend == 0.0:
            # First valid value
            self.direction = 1
            self.supertrend = self.final_lower
        else:
            if self.direction == 1:
                if close <= self.final_lower:
                    self.direction = -1
                    self.supertrend = self.final_upper
                else:
                    self.supertrend = self.final_lower
            else:  # direction == -1
                if close >= self.final_upper:
                    self.direction = 1
                    self.supertrend = self.final_lower
                else:
                    self.supertrend = self.final_upper
        
        return self.supertrend, self.direction
    
    def reset(self):
        """Reset state for new sequence."""
        self.atr_state.reset()
        self.basic_upper = 0.0
        self.basic_lower = 0.0
        self.final_upper = 0.0
        self.final_lower = 0.0
        self.direction = 1
        self.supertrend = 0.0
        self.count = 0


# =================================================================== 19. AroonState
aroon_spec = [
    ('period', int32),
    ('min_periods', int32),
    ('highs', float64[:]),
    ('lows', float64[:]),
    ('head', int32),
    ('count', int32),
]

@jitclass(aroon_spec)
class AroonState:
    """ Aroon Indicator State - Production-grade stateful implementation
    
    Aroon Up: ((period - bars_since_high) / period) * 100
    Aroon Down: ((period - bars_since_low) / period) * 100
    Aroon Oscillator: Aroon Up - Aroon Down
    
    Parameters:
    -----------
    period : int
        Lookback period (معمولاً 25)
    min_periods : int
        حداقل تعداد داده برای شروع محاسبه (پیش‌فرض: -1 = period)
    
    Returns:
    --------
    tuple: (aroon_up, aroon_down, oscillator) یا (np.nan, np.nan, np.nan)

    نحوه استفاده:
    --------------
    # Default (min_periods = period)
        aroon = AroonState(period=25, min_periods=-1)

    # Custom min_periods
        aroon = AroonState(period=25, min_periods=10)

    # Update
        aroon_up, aroon_down, oscillator = aroon.update(high, low)
    """
    
    def __init__(self, period=25, min_periods=-1):
        if period < 1:
            raise ValueError("period must be >= 1")
        
        self.period = period
        if min_periods == -1:
            self.min_periods = period
        else:
            self.min_periods = max(1, min(min_periods, period))
        
        # Rolling buffers برای high و low (circular arrays)
        self.highs = np.zeros(period, dtype=np.float64)
        self.lows = np.zeros(period, dtype=np.float64)
        
        # Circular buffer management
        self.head = 0
        self.count = 0
    
    def update(self, high, low):
        """
        Update state با یک کندل جدید
        
        Parameters:
        -----------
        high : float
            قیمت بالای کندل
        low : float
            قیمت پایین کندل
        
        Returns:
        --------
        tuple: (aroon_up, aroon_down, oscillator)
        """
        high = float(high)
        low = float(low)
        
        # اضافه کردن به circular buffers
        self.highs[self.head] = high
        self.lows[self.head] = low
        
        self.head = (self.head + 1) % self.period
        self.count += 1
        
        current_len = min(self.count, self.period)
        
        # بررسی min_periods
        if self.count < self.min_periods:
            return (np.nan, np.nan, np.nan)

        # ============================================بلوک قدیمی حذف شده
        # # پیدا کردن آخرین موقعیت highest high
        # max_high = self.highs[0]
        # bars_since_high = current_len - 1
        # for i in range(current_len):
        #     if self.highs[i] >= max_high:
        #         max_high = self.highs[i]
        #         bars_since_high = current_len - 1 - i
        
        # # پیدا کردن آخرین موقعیت lowest low
        # min_low = self.lows[0]
        # bars_since_low = current_len - 1
        # for i in range(current_len):
        #     if self.lows[i] <= min_low:
        #         min_low = self.lows[i]
        #         bars_since_low = current_len - 1 - i


        # ============================================بلوک جدید- شروع
        # ترتیب منطقی زمانی:
        # قبل از پر شدن buffer، داده‌ها از index 0 شروع می‌شوند.
        # بعد از پر شدن، self.head به قدیمی‌ترین داده اشاره می‌کند.
        start = 0 if self.count < self.period else self.head

        # پیدا کردن آخرین موقعیت highest high
        max_high = self.highs[start]
        bars_since_high = current_len - 1

        for i in range(current_len):
            idx = (start + i) % self.period

            if self.highs[idx] >= max_high:
                max_high = self.highs[idx]
                bars_since_high = current_len - 1 - i

        # پیدا کردن آخرین موقعیت lowest low
        min_low = self.lows[start]
        bars_since_low = current_len - 1

        for i in range(current_len):
            idx = (start + i) % self.period

            if self.lows[idx] <= min_low:
                min_low = self.lows[idx]
                bars_since_low = current_len - 1 - i
        # ============================================بلوک جدید- پایان

        # محاسبه Aroon Up و Aroon Down
        aroon_up = ((self.period - bars_since_high) / self.period) * 100.0
        aroon_down = ((self.period - bars_since_low) / self.period) * 100.0
        
        # محاسبه Oscillator
        oscillator = aroon_up - aroon_down
        
        return (aroon_up, aroon_down, oscillator)
    
    def reset(self):
        """Reset state به حالت اولیه"""
        self.highs[:] = 0.0
        self.lows[:] = 0.0
        self.head = 0
        self.count = 0


# =================================================================== 20. DEMAState
dema_spec = [
    ('n', int32),
    ('min_periods', int32),
    ('ema1_state', EMAState.class_type.instance_type),
    ('ema2_state', EMAState.class_type.instance_type),
    ('count', int32),
]

@jitclass(dema_spec)
class DEMAState:
    """ Double Exponential Moving Average - Production-grade stateful implementation
    
    DEMA = 2*EMA(price) - EMA(EMA(price))
    
    Parameters:
    -----------
    n : int
        Period برای EMA
    min_periods : int
        حداقل تعداد داده برای شروع محاسبه (پیش‌فرض: -1 = n)
    
    Returns:
    --------
    float: DEMA value یا np.nan

    نحوه استفاده:
    --------------
    dema = DEMAState(n=20, min_periods=-1)  # Default: min_periods = n
    dema = DEMAState(n=20, min_periods=10)
    dema_value = dema.update(close_price)
    """
    
    def __init__(self, n, min_periods=-1):
        if n < 1:
            raise ValueError("n must be >= 1")
        
        self.n = n
        if min_periods == -1:
            self.min_periods = n
        else:
            self.min_periods = max(1, min(min_periods, n))
        
        # Two nested EMAs
        self.ema1_state = EMAState(n, 1)  # First EMA
        self.ema2_state = EMAState(n, 1)  # EMA of EMA
        
        self.count = 0
    
    def update(self, x):
        """
        Update با مقدار جدید و برگرداندن DEMA
        
        Parameters:
        -----------
        x : float
            مقدار ورودی جدید
        
        Returns:
        --------
        float: DEMA value
        """
        x = float(x)
        self.count += 1
        
        # First EMA
        ema1 = self.ema1_state.update(x)
        
        # Second EMA (EMA of first EMA)
        ema2 = self.ema2_state.update(ema1)
        
        # DEMA = 2*EMA1 - EMA2
        dema = 2.0 * ema1 - ema2
        
        # Return NaN until min_periods reached
        if self.count < self.min_periods:
            return np.nan
        
        return dema
    
    def reset(self):
        """Reset state به حالت اولیه"""
        self.ema1_state = EMAState(self.n, 1)
        self.ema2_state = EMAState(self.n, 1)
        self.count = 0


# =================================================================== 21. TEMAState
tema_spec = [
    ('n', int32),
    ('min_periods', int32),
    ('ema1_state', EMAState.class_type.instance_type),
    ('ema2_state', EMAState.class_type.instance_type),
    ('ema3_state', EMAState.class_type.instance_type),
    ('count', int32),
]

@jitclass(tema_spec)
class TEMAState:
    """ Triple Exponential Moving Average - Production-grade stateful implementation
    
    TEMA = 3*EMA(price) - 3*EMA(EMA(price)) + EMA(EMA(EMA(price)))
    
    Parameters:
    -----------
    n : int
        Period برای EMA
    min_periods : int
        حداقل تعداد داده برای شروع محاسبه (پیش‌فرض: -1 = n)
    
    Returns:
    --------
    float: TEMA value یا np.nan

    نحوه استفاده:
    --------------
    tema = TEMAState(n=20, min_periods=-1)  # Default: min_periods = n
    tema = TEMAState(n=20, min_periods=10)
    tema_value = tema.update(close_price)
    """
    
    def __init__(self, n, min_periods=-1):
        if n < 1:
            raise ValueError("n must be >= 1")
        
        self.n = n
        if min_periods == -1:
            self.min_periods = n
        else:
            self.min_periods = max(1, min(min_periods, n))
        
        # Three nested EMAs
        self.ema1_state = EMAState(n, 1)  # First EMA
        self.ema2_state = EMAState(n, 1)  # EMA of EMA
        self.ema3_state = EMAState(n, 1)  # EMA of EMA of EMA
        
        self.count = 0
    
    def update(self, x):
        """
        Update با مقدار جدید و برگرداندن TEMA
        
        Parameters:
        -----------
        x : float
            مقدار ورودی جدید
        
        Returns:
        --------
        float: TEMA value
        """
        x = float(x)
        self.count += 1
        
        # First EMA
        ema1 = self.ema1_state.update(x)
        
        # Second EMA (EMA of first EMA)
        ema2 = self.ema2_state.update(ema1)
        
        # Third EMA (EMA of second EMA)
        ema3 = self.ema3_state.update(ema2)
        
        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        tema = 3.0 * ema1 - 3.0 * ema2 + ema3
        
        if self.count < self.min_periods:
            return np.nan
        
        return tema
    
    def reset(self):
        """Reset state به حالت اولیه"""
        self.ema1_state = EMAState(self.n, 1)
        self.ema2_state = EMAState(self.n, 1)
        self.ema3_state = EMAState(self.n, 1)
        self.count = 0


# =================================================================== 22. KAMAState
kama_spec = [
    ('n', int32),
    ('fast_span', int32),
    ('slow_span', int32),
    ('min_periods', int32),
    ('buffer', float64[:]),
    ('buffer_idx', int32),
    ('buffer_full', boolean),
    ('kama', float64),
    ('kama_initialized', boolean),
    ('count', int32),
]

@jitclass(kama_spec)
class KAMAState:
    """ Kaufman's Adaptive Moving Average - Production-grade stateful implementation
    
    KAMA adapts to market volatility using efficiency ratio (ER).
    
    ER = |Change| / Volatility
    SC = [ER × (fastest - slowest) + slowest]²
    KAMA = KAMA_prev + SC × (Price - KAMA_prev)
    
    Parameters:
    -----------
    n : int
        Period برای محاسبه Efficiency Ratio (پیش‌فرض: 10)
    fast_span : int
        Fast EMA span (پیش‌فرض: 2)
    slow_span : int
        Slow EMA span (پیش‌فرض: 30)
    min_periods : int
        حداقل تعداد داده برای شروع محاسبه (پیش‌فرض: -1 = n)
    
    Returns:
    --------
    float: KAMA value یا np.nan

    نحوه استفاده:
    --------------
    kama = KAMAState(n=10, fast_span=2, slow_span=30, min_periods=-1)
    kama_value = kama.update(close_price)
    """
    
    def __init__(self, n=10, fast_span=2, slow_span=30, min_periods=-1):
        if n < 1:
            raise ValueError("n must be >= 1")
        if fast_span < 1:
            raise ValueError("fast_span must be >= 1")
        if slow_span < 1:
            raise ValueError("slow_span must be >= 1")
        
        self.n = n
        self.fast_span = fast_span
        self.slow_span = slow_span
        
        if min_periods == -1:
            self.min_periods = n
        else:
            self.min_periods = max(1, min(min_periods, n))
        
        # Circular buffer for price history
        self.buffer = np.zeros(n, dtype=np.float64)
        self.buffer_idx = 0
        self.buffer_full = False
        
        self.kama = np.nan
        self.kama_initialized = False
        self.count = 0
    
    def update(self, x):
        """
        Update با مقدار جدید و برگرداندن KAMA
        
        Parameters:
        -----------
        x : float
            مقدار ورودی جدید (معمولاً close price)
        
        Returns:
        --------
        float: KAMA value
        """
        x = float(x)
        self.count += 1
        
        # Add to buffer
        self.buffer[self.buffer_idx] = x
        self.buffer_idx = (self.buffer_idx + 1) % self.n
        
        if self.buffer_idx == 0:
            self.buffer_full = True
        
        # Need at least n values
        if not self.buffer_full and self.buffer_idx < self.n:
            return np.nan
        
        # Calculate Efficiency Ratio (ER)
        # ER = |Change| / Volatility
        # Change = |price_now - price_n_periods_ago|
        # Volatility = sum of |price[i] - price[i-1]| over n periods
        
        oldest_idx = self.buffer_idx  # Points to oldest value in circular buffer
        newest_idx = (self.buffer_idx - 1) % self.n  # Points to newest value
        
        change = abs(x - self.buffer[oldest_idx])
        
        # Calculate volatility (sum of absolute price changes)
        volatility = 0.0
        for i in range(self.n - 1):
            curr_idx = (oldest_idx + i + 1) % self.n
            prev_idx = (oldest_idx + i) % self.n
            volatility += abs(self.buffer[curr_idx] - self.buffer[prev_idx])
        
        # Avoid division by zero
        if volatility < EPS:
            er = 0.0
        else:
            er = change / volatility
        
        # Calculate Smoothing Constant (SC)
        # SC = [ER × (fastest - slowest) + slowest]²
        # fastest = 2/(fast_span + 1)
        # slowest = 2/(slow_span + 1)
        fastest = 2.0 / (self.fast_span + 1.0)
        slowest = 2.0 / (self.slow_span + 1.0)
        
        sc = er * (fastest - slowest) + slowest
        sc = sc * sc  # Square it
        
        # Initialize KAMA with first valid price
        if not self.kama_initialized:
            self.kama = x
            self.kama_initialized = True
        else:
            # KAMA = KAMA_prev + SC × (Price - KAMA_prev)
            self.kama = self.kama + sc * (x - self.kama)
        
        # Return NaN until min_periods reached
        if self.count < self.min_periods:
            return np.nan
        
        return self.kama
    
    def reset(self):
        """Reset state به حالت اولیه"""
        self.buffer = np.zeros(self.n, dtype=np.float64)
        self.buffer_idx = 0
        self.buffer_full = False
        self.kama = np.nan
        self.kama_initialized = False
        self.count = 0


# =================================================================== 23. HMAState
hma_spec = [
    ('n', int32),
    ('min_periods', int32),
    ('half_period', int32),
    ('sqrt_period', int32),
    ('wma_half_state', WMAState.class_type.instance_type),
    ('wma_full_state', WMAState.class_type.instance_type),
    ('wma_sqrt_state', WMAState.class_type.instance_type),
    ('count', int32),
]

@jitclass(hma_spec)
class HMAState:
    """ Hull Moving Average - Production-grade stateful implementation
    
    HMA = WMA(sqrt(n)) of [2*WMA(n/2) - WMA(n)]
    
    Parameters:
    -----------
    n : int
        Period برای HMA
    min_periods : int
        حداقل تعداد داده برای شروع محاسبه (پیش‌فرض: -1 = n)
    
    Returns:
    --------
    float: HMA value یا np.nan

    نحوه استفاده:
    --------------
    hma = HMAState(n=20, min_periods=-1)  # Default: min_periods = n
    hma = HMAState(n=20, min_periods=10)
    hma_value = hma.update(close_price)
    """
    
    def __init__(self, n, min_periods=-1):
        if n < 1:
            raise ValueError("n must be >= 1")
        
        self.n = n
        if min_periods == -1:
            self.min_periods = n
        else:
            self.min_periods = max(1, min(min_periods, n))
        
        self.half_period = n // 2
        self.sqrt_period = int(np.sqrt(float(n)))
        
        # Three WMA states
        self.wma_half_state = WMAState(self.half_period, 1)
        self.wma_full_state = WMAState(n               , 1)
        self.wma_sqrt_state = WMAState(self.sqrt_period, 1)
        
        self.count = 0
    
    def update(self, x):
        """
        Update با مقدار جدید و برگرداندن HMA
        
        Parameters:
        -----------
        x : float
            مقدار ورودی جدید
        
        Returns:
        --------
        float: HMA value
        """
        x = float(x)
        self.count += 1
        
        # WMA with half period
        wma_half = self.wma_half_state.update(x)
        
        # WMA with full period
        wma_full = self.wma_full_state.update(x)
        
        # Raw HMA = 2*WMA(n/2) - WMA(n)
        raw_hma = 2.0 * wma_half - wma_full
        
        # Final HMA = WMA(sqrt(n)) of raw_hma
        hma = self.wma_sqrt_state.update(raw_hma)
        
        if self.count < self.min_periods:
            return np.nan
        
        return hma
    
    def reset(self):
        """Reset state به حالت اولیه"""
        self.wma_half_state = WMAState(self.half_period, 1)
        self.wma_full_state = WMAState(self.n          , 1)
        self.wma_sqrt_state = WMAState(self.sqrt_period, 1)
        self.count = 0

# ===================================================================
