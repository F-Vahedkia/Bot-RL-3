# Last revewed at 1405/06/17

"""
indicators_B_batch.py
===================
Adapter layer: wraps scalar-based indicator classes from indicators_class.py
into DataFrame-oriented functions compatible with feature_registry.py

Each function signature: fn(df: pd.DataFrame, **kwargs) -> pd.DataFrame | pd.Series
"""
import numpy as np
import pandas as pd
from numba import njit

from typing import Literal, Optional
from .indicators_B_maths import (
    SMAState       , WMAState       , EMAState       , ROCState,
    RSIState       , TrueRangeState , ATRState       , MACDState,
    BollingerState , KeltnerState   , StochasticState, CCIState,
    MFIState       , OBVState       , WilliamsRState , ParabolicSARState,
    HeikinAshiState, SupertrendState, AroonState     , DEMAState,
    TEMAState      , KAMAState      , HMAState
)

# ============================================================================
# 1. SMA - Simple Moving Average
# ============================================================================
@njit
def sma_batch(data, period, min_periods=-1):
    """
    محاسبه SMA به صورت batch با استفاده از SMAState
    
    Parameters:
    -----------
    data : np.ndarray
        آرایه داده‌های ورودی
    period : int
        دوره SMA
    min_periods : int
        حداقل تعداد داده برای محاسبه (پیش‌فرض: period)
    
    Returns:
    --------
    np.ndarray
        آرایه نتایج SMA
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    
    state = SMAState(period, min_periods)
    
    for i in range(n):
        result[i] = state.update(data[i])
    
    return result


def sma_batch_df(df, column,
                 period, min_periods=-1,
                 result_col='sma', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    column : str
        نام ستون برای محاسبه
    period : int
        دوره SMA
    min_periods : int
        حداقل تعداد داده
    result_col : str, optional
        نام ستون نتیجه (پیش‌فرض: f'sma_{period}')
    
    Returns:
    --------
    pd.Series
        سری نتایج SMA
    """
    data = df[column].to_numpy(dtype=np.float64)
    result = sma_batch(data, period, min_periods)
       
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
         return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)       


# ============================================================================
# 2. WMA - Weighted Moving Average
# ============================================================================
@njit
def wma_batch(data, period, min_periods=-1):
    """
    محاسبه WMA به صورت batch با استفاده از WMAState
    
    Parameters:
    -----------
    data : np.ndarray
        آرایه داده‌های ورودی
    period : int
        دوره WMA
    min_periods : int
        حداقل تعداد داده برای محاسبه (پیش‌فرض: period)
    
    Returns:
    --------
    np.ndarray
        آرایه نتایج WMA
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    
    state = WMAState(period, min_periods)
    
    for i in range(n):
        result[i] = state.update(data[i])
    
    return result


def wma_batch_df(df, column,
                 period, min_periods=-1,
                 result_col='wma', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    column : str
        نام ستون برای محاسبه
    period : int
        دوره WMA
    min_periods : int
        حداقل تعداد داده
    result_col : str, optional
        نام ستون نتیجه (پیش‌فرض: f'wma_{period}')
    
    Returns:
    --------
    pd.Series
        سری نتایج WMA
    """
    data = df[column].to_numpy(dtype=np.float64)
    result = wma_batch(data, period, min_periods)
      
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)


# ============================================================================
# 3. EMA - Exponential Moving Average
# ============================================================================
@njit
def ema_batch(data, period, min_periods=-1):
    """
    محاسبه EMA به صورت batch با استفاده از EMAState
    
    Parameters:
    -----------
    data : np.ndarray
        آرایه داده‌های ورودی
    period : int
        دوره EMA
    min_periods : int
        حداقل تعداد داده برای محاسبه (پیش‌فرض: period)
    
    Returns:
    --------
    np.ndarray
        آرایه نتایج EMA
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    
    state = EMAState(period, min_periods)
    
    for i in range(n):
        result[i] = state.update(data[i])
    
    return result


def ema_batch_df(df, column,
                 period, min_periods=-1,
                 result_col='ema', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    column : str
        نام ستون برای محاسبه
    period : int
        دوره EMA
    min_periods : int
        حداقل تعداد داده
    result_col : str, optional
        نام ستون نتیجه (پیش‌فرض: f'ema_{period}')
    
    Returns:
    --------
    pd.Series
        سری نتایج EMA
    """
    data = df[column].to_numpy(dtype=np.float64)
    result = ema_batch(data, period, min_periods)
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
       

# ============================================================================
# 4. ROC - Rate of Change
# ============================================================================
@njit
def roc_batch(data, period):
    """
    محاسبه ROC (Rate of Change) به صورت batch با استفاده از ROCState
    
    Parameters:
    -----------
    data : np.ndarray
        آرایه داده‌های ورودی
    period : int
        دوره ROC
    
    Returns:
    --------
    np.ndarray
        آرایه نتایج ROC (درصد تغییرات)
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    
    state = ROCState(period)
    
    for i in range(n):
        result[i] = state.update(data[i])
    
    return result


def roc_batch_df(df, column,
                 period,
                 result_col='roc', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    column : str
        نام ستون برای محاسبه
    period : int
        دوره ROC
    result_col : str, optional
        نام ستون نتیجه (پیش‌فرض: f'roc_{period}')
    
    Returns:
    --------
    pd.Series
        سری نتایج ROC
    """
    data = df[column].to_numpy(dtype=np.float64)
    result = roc_batch(data, period)
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)


# ============================================================================
# 5. RSI - Relative Strength Index ("ema=0", wilder=1)
# ============================================================================
@njit
def rsi_batch(data, period, method=0):
    """
    محاسبه RSI (Relative Strength Index) به صورت batch با استفاده از RSIState
    
    Parameters:
    -----------
    data : np.ndarray
        آرایه داده‌های قیمت بسته شدن
    period : int
        دوره RSI
    method : int, optional
        روش محاسبه میانگین: 0=ema (پیش‌فرض), 1=wilders
    
    Returns:
    --------
    np.ndarray
        آرایه نتایج RSI (مقادیر بین 0 تا 100)
    """
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    
    state = RSIState(period, method)
    
    for i in range(n):
        result[i] = state.update(data[i])
    
    return result


def rsi_batch_df(df, column='close',
                 period=14, method='ema',
                 result_col='rsi', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    column : str, optional
        نام ستون قیمت (پیش‌فرض: 'close')
    period : int, optional
        دوره RSI (پیش‌فرض: 14)
    method : str, optional
        روش محاسبه: 'ema' یا 'wilders' (پیش‌فرض: 'ema')
    result_col : str, optional
        نام ستون نتیجه (پیش‌فرض: f'rsi_{period}_{method}')
    
    Returns:
    --------
    pd.Series
        سری نتایج RSI
    """
    # تبدیل method از string به int
    method_int = 1 if method.lower() == 'wilders' else 0
    
    data = df[column].to_numpy(dtype=np.float64)
    result = rsi_batch(data, period, method_int)
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}_{method}': result
        }, index=df.index)
    else:
         return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
       

# ============================================================================
# 6. True Range
# ============================================================================
@njit
def truerange_batch(high, low, close):
    """
    محاسبه True Range به صورت batch با استفاده از TrueRangeState
    
    Parameters:
    -----------
    high : np.ndarray
        آرایه قیمت‌های بالا
    low : np.ndarray
        آرایه قیمت‌های پایین
    close : np.ndarray
        آرایه قیمت‌های بسته شدن
    
    Returns:
    --------
    np.ndarray
        آرایه نتایج True Range
    """
    n = len(high)
    result = np.empty(n, dtype=np.float64)
    
    state = TrueRangeState()
    
    for i in range(n):
        result[i] = state.update(high[i], low[i], close[i])
    
    return result


def truerange_batch_df(df, high_column='high', low_column='low', close_column='close',
                       result_col='true_range', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    high_column : str, optional
        نام ستون قیمت بالا (پیش‌فرض: 'high')
    low_column : str, optional
        نام ستون قیمت پایین (پیش‌فرض: 'low')
    close_column : str, optional
        نام ستون قیمت بسته شدن (پیش‌فرض: 'close')
    result_col : str, optional
        نام ستون نتیجه (پیش‌فرض: 'true_range')
    
    Returns:
    --------
    pd.Series
        سری نتایج True Range
    """
    high = df[high_column].to_numpy(dtype=np.float64)
    low = df[low_column].to_numpy(dtype=np.float64)
    close = df[close_column].to_numpy(dtype=np.float64)
    
    result = truerange_batch(high, low, close)

    return pd.DataFrame({
        f'{result_col}': result
    }, index=df.index)


# ============================================================================
# 7. ATR - Average True Range (classic=0, "wilder=1", ema=2)
# ============================================================================
@njit
def atr_batch(high, low, close, n=14, method=1, min_periods=-1):
    """
    محاسبه ATR به صورت batch با استفاده از ATRState
    
    Parameters:
    -----------
    high : np.ndarray
        آرایه قیمت‌های بالا
    low : np.ndarray
        آرایه قیمت‌های پایین
    close : np.ndarray
        آرایه قیمت‌های بسته شدن
    n : int, optional
        دوره ATR (پیش‌فرض: 14)
    method : int, optional
        روش محاسبه: 0=classic, 1=wilder, 2=ema (پیش‌فرض: 1)
    min_periods : int, optional
        حداقل دوره برای محاسبه (پیش‌فرض: -1 که برابر n می‌شود)
    
    Returns:
    --------
    np.ndarray
        آرایه نتایج ATR
    """
    length = len(high)
    result = np.empty(length, dtype=np.float64)
    
    state = ATRState(n, method, min_periods)
    
    for i in range(length):
        result[i] = state.update(high[i], low[i], close[i])
    
    return result


def atr_batch_df(df, high_column='high', low_column='low', close_column='close', 
                 period=14, method='wilder', min_periods=-1, 
                 result_col='atr', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    period : int, optional
        دوره ATR (پیش‌فرض: 14)
    method : int, optional
        روش محاسبه: 0=classic, 1=wilder, 2=ema (پیش‌فرض: 1)
    min_periods : int, optional
        حداقل دوره برای محاسبه (پیش‌فرض: -1 که برابر n می‌شود)
    high_column : str, optional
        نام ستون قیمت بالا (پیش‌فرض: 'high')
    low_column : str, optional
        نام ستون قیمت پایین (پیش‌فرض: 'low')
    close_column : str, optional
        نام ستون قیمت بسته شدن (پیش‌فرض: 'close')
    result_col : str, optional
        نام ستون نتیجه (پیش‌فرض: 'atr')
    
    Returns:
    --------
    pd.Series
        سری نتایج ATR
    """
    high = df[high_column].to_numpy(dtype=np.float64)
    low = df[low_column].to_numpy(dtype=np.float64)
    close = df[close_column].to_numpy(dtype=np.float64)
    
    # تبدیل method از string به int
    method_map = {'classic': 0, 'wilder': 1, 'ema': 2}
    method_int = method_map.get(str(method).lower(), 1)  # default: 'wilder'
    if method_int is None:
        raise ValueError(f"Invalid method: {method}. Choose from {list(method_map.keys())}")
    
    result = atr_batch(high, low, close, n=period, method=method_int, min_periods=min_periods)
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}_{method}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)


# ============================================================================
# 8. MACD - Moving Average Convergence Divergence
# ============================================================================
@njit
def macd_batch(close, fast=12, slow=26, signal=9):
    """
    محاسبه MACD به صورت batch با استفاده از MACDState
    
    Parameters:
    -----------
    close : np.ndarray
        آرایه قیمت‌های بسته شدن
    fast : int, optional
        دوره EMA سریع (پیش‌فرض: 12)
    slow : int, optional
        دوره EMA کند (پیش‌فرض: 26)
    signal : int, optional
        دوره EMA سیگنال (پیش‌فرض: 9)
    
    Returns:
    --------
    tuple of np.ndarray
        (macd_line, signal_line, histogram)
    """
    length = len(close)
    macd_line = np.empty(length, dtype=np.float64)
    signal_line = np.empty(length, dtype=np.float64)
    histogram = np.empty(length, dtype=np.float64)
    
    state = MACDState(fast, slow, signal)
    
    for i in range(length):
        macd, sig, hist = state.update(close[i])
        macd_line[i] = macd
        signal_line[i] = sig
        histogram[i] = hist
    
    return macd_line, signal_line, histogram


def macd_batch_df(df, column='close',                     # close_col='close',
                  fast=12, slow=26, signal=9, 
                  macd_col='macd', signal_col='macd_signal',
                  hist_col='macd_hist', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    fast : int, optional
        دوره EMA سریع (پیش‌فرض: 12)
    slow : int, optional
        دوره EMA کند (پیش‌فرض: 26)
    signal : int, optional
        دوره EMA سیگنال (پیش‌فرض: 9)
    column : str, optional
        نام ستون قیمت بسته شدن (پیش‌فرض: 'close')
    macd_col : str, optional
        نام ستون MACD line (پیش‌فرض: 'macd')
    signal_col : str, optional
        نام ستون Signal line (پیش‌فرض: 'macd_signal')
    hist_col : str, optional
        نام ستون Histogram (پیش‌فرض: 'macd_hist')
    
    Returns:
    --------
    pd.DataFrame
        دیتافریم با سه ستون: MACD line, Signal line, Histogram
    """
    data = df[column].to_numpy(dtype=np.float64)
    
    macd_line, signal_line, histogram = macd_batch(data, fast=fast, slow=slow, signal=signal)

    # print("MACD valid:", np.sum(~np.isnan(macd_line)))
    # print("Signal valid:", np.sum(~np.isnan(signal_line)))
    # print("Hist valid:", np.sum(~np.isnan(histogram)))

    # print("MACD tail:")
    # print(macd_line[-10:])

    # print("Signal tail:")
    # print(signal_line[-10:])

    # print("Hist tail:")
    # print(histogram[-10:])

    if add_para_to_names:
        return pd.DataFrame({
            f'{macd_col}_{fast}_{slow}_{signal}': macd_line,
            f'{signal_col}_{fast}_{slow}_{signal}': signal_line,
            f'{hist_col}_{fast}_{slow}_{signal}': histogram
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{macd_col}': macd_line,
            f'{signal_col}': signal_line,
            f'{hist_col}': histogram
        }, index=df.index)


# ============================================================================
# 9. Bollinger Bands
# ============================================================================
@njit
def bollinger_batch(close, n=20, k=2.0, min_periods=-1):
    """
    محاسبه Bollinger Bands به صورت batch با استفاده از BollingerState
    
    Parameters:
    -----------
    close : np.ndarray
        آرایه قیمت‌های بسته شدن
    n : int, optional
        دوره محاسبه (پیش‌فرض: 20)
    k : float, optional
        ضریب انحراف معیار (پیش‌فرض: 2.0)
    min_periods : int, optional
        حداقل تعداد داده برای محاسبه (پیش‌فرض: -1 یعنی برابر n)
    
    Returns:
    --------
    tuple of np.ndarray
        (upper, middle, lower, width, percent)
        - upper: باند بالایی
        - middle: میانگین متحرک (SMA)
        - lower: باند پایینی
        - width: عرض باند (upper - lower)
        - percent: موقعیت قیمت در باند (0 تا 1)
    """
    length = len(close)
    upper = np.empty(length, dtype=np.float64)
    middle = np.empty(length, dtype=np.float64)
    lower = np.empty(length, dtype=np.float64)
    width = np.empty(length, dtype=np.float64)
    percent = np.empty(length, dtype=np.float64)
    
    state = BollingerState(n, k, min_periods)
    
    for i in range(length):
        u, m, l, w, p = state.update(close[i])
        upper[i] = u
        middle[i] = m
        lower[i] = l
        width[i] = w
        percent[i] = p
    
    return upper, middle, lower, width, percent


def bollinger_batch_df(df, column='close',
                       period=20, multiplier=2.0, min_periods=-1,
                       upper_col='bb_upper', middle_col='bb_middle', 
                       lower_col='bb_lower', width_col='bb_width', 
                       percent_col='bb_percent', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    period : int, optional
        دوره محاسبه (پیش‌فرض: 20)
    multiplier : float, optional
        ضریب انحراف معیار (پیش‌فرض: 2.0)
    min_periods : int, optional
        حداقل تعداد داده برای محاسبه (پیش‌فرض: -1 یعنی برابر n)
    close_col : str, optional
        نام ستون قیمت بسته شدن (پیش‌فرض: 'close')
    upper_col : str, optional
        نام ستون باند بالایی (پیش‌فرض: 'bb_upper')
    middle_col : str, optional
        نام ستون میانگین (پیش‌فرض: 'bb_middle')
    lower_col : str, optional
        نام ستون باند پایینی (پیش‌فرض: 'bb_lower')
    width_col : str, optional
        نام ستون عرض باند (پیش‌فرض: 'bb_width')
    percent_col : str, optional
        نام ستون درصد موقعیت (پیش‌فرض: 'bb_percent')
    
    Returns:
    --------
    pd.DataFrame
        دیتافریم با پنج ستون: upper, middle, lower, width, percent
    """
    data = df[column].to_numpy(dtype=np.float64)
    
    upper, middle, lower, width, percent = bollinger_batch(
        data, n=period, k=multiplier, min_periods=min_periods
    )
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{upper_col}_{period}_{multiplier}': upper,
            f'{middle_col}_{period}_{multiplier}': middle,
            f'{lower_col}_{period}_{multiplier}': lower,
            f'{width_col}_{period}_{multiplier}': width,
            f'{percent_col}_{period}_{multiplier}': percent
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{upper_col}': upper,
            f'{middle_col}': middle,
            f'{lower_col}': lower,
            f'{width_col}': width,
            f'{percent_col}': percent
        }, index=df.index)


# ============================================================================
# 10. Keltner Channel
# ============================================================================
@njit
def keltner_batch(high, low, close, n=20, m=2.0, min_periods=-1):
    """
    محاسبه Keltner Channel به صورت batch با استفاده از KeltnerState
    
    Parameters:
    -----------
    high : np.ndarray
        آرایه قیمت‌های بالا
    low : np.ndarray
        آرایه قیمت‌های پایین
    close : np.ndarray
        آرایه قیمت‌های بسته شدن
    n : int, optional
        دوره محاسبه (پیش‌فرض: 20)
    m : float, optional
        ضریب ATR (پیش‌فرض: 2.0)
    min_periods : int, optional
        حداقل تعداد داده برای محاسبه (پیش‌فرض: -1 یعنی برابر n)
    
    Returns:
    --------
    tuple of np.ndarray
        (upper, middle, lower, width, percent)
        - upper: کانال بالایی (EMA + m*ATR)
        - middle: میانگین متحرک نمایی (EMA)
        - lower: کانال پایینی (EMA - m*ATR)
        - width: عرض کانال (upper - lower)
        - percent: موقعیت قیمت در کانال
    """
    length = len(close)
    upper = np.empty(length, dtype=np.float64)
    middle = np.empty(length, dtype=np.float64)
    lower = np.empty(length, dtype=np.float64)
    width = np.empty(length, dtype=np.float64)
    percent = np.empty(length, dtype=np.float64)
    
    state = KeltnerState(n, m, min_periods)
    
    for i in range(length):
        u, mid, l, w, p = state.update(high[i], low[i], close[i])
        upper[i] = u
        middle[i] = mid
        lower[i] = l
        width[i] = w
        percent[i] = p
    
    return upper, middle, lower, width, percent


def keltner_batch_df(df, high_column='high', low_column='low', close_column='close',
                     period=20, multiplier=2.0, min_periods=-1,
                     upper_col='kc_upper', middle_col='kc_middle',
                     lower_col='kc_lower', width_col='kc_width',
                     percent_col='kc_percent', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    period : int, optional
        دوره محاسبه (پیش‌فرض: 20)
    multiplier : float, optional
        ضریب ATR (پیش‌فرض: 2.0)
    min_periods : int, optional
        حداقل تعداد داده برای محاسبه (پیش‌فرض: -1 یعنی برابر n)
    high_column : str, optional
        نام ستون قیمت بالا (پیش‌فرض: 'high')
    low_column : str, optional
        نام ستون قیمت پایین (پیش‌فرض: 'low')
    close_column : str, optional
        نام ستون قیمت بسته شدن (پیش‌فرض: 'close')
    upper_col : str, optional
        نام ستون کانال بالایی (پیش‌فرض: 'kc_upper')
    middle_col : str, optional
        نام ستون میانگین (پیش‌فرض: 'kc_middle')
    lower_col : str, optional
        نام ستون کانال پایینی (پیش‌فرض: 'kc_lower')
    width_col : str, optional
        نام ستون عرض کانال (پیش‌فرض: 'kc_width')
    percent_col : str, optional
        نام ستون درصد موقعیت (پیش‌فرض: 'kc_percent')
    
    Returns:
    --------
    pd.DataFrame
        دیتافریم با پنج ستون: upper, middle, lower, width, percent
    """
    high = df[high_column].to_numpy(dtype=np.float64)
    low = df[low_column].to_numpy(dtype=np.float64)
    close = df[close_column].to_numpy(dtype=np.float64)
    
    upper, middle, lower, width, percent = keltner_batch(
        high, low, close, n=period, m=multiplier, min_periods=min_periods
    )
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{upper_col}_{period}_{multiplier}': upper,
            f'{middle_col}_{period}_{multiplier}': middle,
            f'{lower_col}_{period}_{multiplier}': lower,
            f'{width_col}_{period}_{multiplier}': width,
            f'{percent_col}_{period}_{multiplier}': percent
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{upper_col}': upper,
            f'{middle_col}': middle,
            f'{lower_col}': lower,
            f'{width_col}': width,
            f'{percent_col}': percent
        }, index=df.index)


# ============================================================================
# 11. Stochastic Oscillator ("sma=0", ema=1)
# ============================================================================
@njit
def stochastic_batch(high, low, close, k_period=14, d_period=3, smooth_k=3, method=0, min_periods=-1):
    """
    محاسبه Stochastic Oscillator به صورت batch با استفاده از StochasticState
    
    Parameters:
    -----------
    high : np.ndarray
        آرایه قیمت‌های بالا
    low : np.ndarray
        آرایه قیمت‌های پایین
    close : np.ndarray
        آرایه قیمت‌های بسته شدن
    k_period : int, optional
        دوره محاسبه %K (پیش‌فرض: 14)
    d_period : int, optional
        دوره محاسبه %D (پیش‌فرض: 3)
    smooth_k : int, optional
        دوره هموارسازی %K (پیش‌فرض: 3)
    method : int, optional
        روش هموارسازی: 0=sma, 1=ema (پیش‌فرض: 0)
    min_periods : int, optional
        حداقل تعداد داده برای محاسبه (پیش‌فرض: -1 یعنی برابر k_period)
    
    Returns:
    --------
    tuple of np.ndarray
        (k, d)
        - k: %K (Fast Stochastic)
        - d: %D (Slow Stochastic - سیگنال)
    """
    length = len(close)
    k = np.empty(length, dtype=np.float64)
    d = np.empty(length, dtype=np.float64)
    
    state = StochasticState(k_period, d_period, smooth_k, method, min_periods)
    
    for i in range(length):
        k_val, d_val = state.update(high[i], low[i], close[i])
        k[i] = k_val
        d[i] = d_val
    
    return k, d


def stochastic_batch_df(df, high_column='high', low_column='low', close_column='close',
                       k_period=14, d_period=3, smooth_k=3, method='sma', min_periods=-1,
                       k_col='stoch_k', d_col='stoch_d', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    k_period : int, optional
        دوره محاسبه %K (پیش‌فرض: 14)
    d_period : int, optional
        دوره محاسبه %D (پیش‌فرض: 3)
    smooth_k : int, optional
        دوره هموارسازی %K (پیش‌فرض: 3)
    method : str, optional
        روش هموارسازی: 'sma' یا 'ema' (پیش‌فرض: 'sma')
    min_periods : int, optional
        حداقل تعداد داده برای محاسبه (پیش‌فرض: -1 یعنی برابر k_period)
    high_column : str, optional
        نام ستون قیمت بالا (پیش‌فرض: 'high')
    low_column : str, optional
        نام ستون قیمت پایین (پیش‌فرض: 'low')
    close_column : str, optional
        نام ستون قیمت بسته شدن (پیش‌فرض: 'close')
    k_col : str, optional
        نام ستون %K (پیش‌فرض: 'stoch_k')
    d_col : str, optional
        نام ستون %D (پیش‌فرض: 'stoch_d')
    
    Returns:
    --------
    pd.DataFrame
        دیتافریم با دو ستون: k, d
    """
    high = df[high_column].to_numpy(dtype=np.float64)
    low = df[low_column].to_numpy(dtype=np.float64)
    close = df[close_column].to_numpy(dtype=np.float64)
    
    # تبدیل method از string به int
    method_map = {'sma': 0, 'ema': 1}
    method_int = method_map.get(str(method).lower(), 0)  # default: 'sma'
    if method_int is None:
        raise ValueError(f"Invalid method: {method}. Choose from {list(method_map.keys())}")
        
    k, d = stochastic_batch(high, low, close, k_period=k_period, 
                           d_period=d_period, smooth_k=smooth_k, 
                           method=method_int, min_periods=min_periods)
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{k_col}_{k_period}_{d_period}_{smooth_k}_{method}': k,
            f'{d_col}_{k_period}_{d_period}_{smooth_k}_{method}': d
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{k_col}': k,
            f'{d_col}': d
        }, index=df.index)


# ============================================================================
# 12. CCI (Commodity Channel Index)
# ============================================================================
@njit
def cci_batch(high, low, close, n=20, min_periods=-1):
    """
    محاسبه Commodity Channel Index (CCI) به صورت batch با استفاده از CCIState
    
    CCI = (TP - SMA(TP)) / (0.015 × Mean Deviation)
    که TP = (High + Low + Close) / 3
    
    Parameters:
    -----------
    high : np.ndarray
        آرایه قیمت‌های بالا
    low : np.ndarray
        آرایه قیمت‌های پایین
    close : np.ndarray
        آرایه قیمت‌های بسته شدن
    n : int, optional
        دوره محاسبه CCI (پیش‌فرض: 20)
    min_periods : int, optional
        حداقل تعداد داده برای محاسبه (پیش‌فرض: -1 یعنی برابر n)
    
    Returns:
    --------
    np.ndarray
        آرایه مقادیر CCI
    """
    length = len(close)
    result = np.empty(length, dtype=np.float64)
    
    state = CCIState(n, min_periods)
    
    for i in range(length):
        result[i] = state.update(high[i], low[i], close[i])
    
    return result


def cci_batch_df(df, high_column='high', low_column='low', close_column='close',
                 period=20, min_periods=-1,
                 result_col='cci', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    period : int, optional
        دوره محاسبه CCI (پیش‌فرض: 20)
    min_periods : int, optional
        حداقل تعداد داده برای محاسبه (پیش‌فرض: -1 یعنی برابر n)
    high_column : str, optional
        نام ستون قیمت بالا (پیش‌فرض: 'high')
    low_column : str, optional
        نام ستون قیمت پایین (پیش‌فرض: 'low')
    close_column : str, optional
        نام ستون قیمت بسته شدن (پیش‌فرض: 'close')
    output_col : str, optional
        نام ستون خروجی (پیش‌فرض: 'cci')
    
    Returns:
    --------
    pd.Series
        سری مقادیر CCI
    """
    high = df[high_column].to_numpy(dtype=np.float64)
    low = df[low_column].to_numpy(dtype=np.float64)
    close = df[close_column].to_numpy(dtype=np.float64)
    
    result = cci_batch(high, low, close, n=period, min_periods=min_periods)

    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)


# ============================================================================
# 13. MFI (Money Flow Index)
# ============================================================================
@njit
def mfi_batch(high, low, close, volume, n=14, min_periods=-1):
    """
    محاسبه Money Flow Index (MFI) به صورت batch با استفاده از MFIState
    
    MFI = 100 - (100 / (1 + Money Flow Ratio))
    Money Flow Ratio = Positive Money Flow / Negative Money Flow
    Money Flow = Typical Price × Volume
    Typical Price = (High + Low + Close) / 3
    
    Parameters:
    -----------
    high : np.ndarray
        آرایه قیمت‌های بالا
    low : np.ndarray
        آرایه قیمت‌های پایین
    close : np.ndarray
        آرایه قیمت‌های بسته شدن
    volume : np.ndarray
        آرایه حجم معاملات
    n : int, optional
        دوره محاسبه MFI (پیش‌فرض: 14)
    min_periods : int, optional
        حداقل تعداد داده برای محاسبه (پیش‌فرض: -1 یعنی برابر n)
    
    Returns:
    --------
    np.ndarray
        آرایه مقادیر MFI (بین 0 تا 100)
    """
    length = len(close)
    result = np.empty(length, dtype=np.float64)
    
    state = MFIState(n, min_periods)
    
    for i in range(length):
        result[i] = state.update(high[i], low[i], close[i], volume[i])

    return result


def mfi_batch_df(df, high_column='high', low_column='low', close_column='close', volume_column='volume',
                 period=14, min_periods=-1,
                 result_col='mfi', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    period : int, optional
        دوره محاسبه MFI (پیش‌فرض: 14)
    min_periods : int, optional
        حداقل تعداد داده برای محاسبه (پیش‌فرض: -1 یعنی برابر n)
    high_column : str, optional
        نام ستون قیمت بالا (پیش‌فرض: 'high')
    low_column : str, optional
        نام ستون قیمت پایین (پیش‌فرض: 'low')
    close_column : str, optional
        نام ستون قیمت بسته شدن (پیش‌فرض: 'close')
    volume_column : str, optional
        نام ستون حجم معاملات (پیش‌فرض: 'volume')
    output_col : str, optional
        نام ستون خروجی (پیش‌فرض: 'mfi')
    
    Returns:
    --------
    pd.Series
        سری مقادیر MFI
    """
    high = df[high_column].to_numpy(dtype=np.float64)
    low = df[low_column].to_numpy(dtype=np.float64)
    close = df[close_column].to_numpy(dtype=np.float64)
    volume = df[volume_column].to_numpy(dtype=np.float64)

    result = mfi_batch(high, low, close, volume, n=period, min_periods=min_periods)

    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
         return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
       

# ============================================================================
# 14. OBV (On-Balance Volume)
# ============================================================================
@njit
def obv_batch(close, volume):
    """
    محاسبه On-Balance Volume (OBV) به صورت batch با استفاده از OBVState
    
    OBV یک اندیکاتور تجمعی است که حجم را بر اساس جهت حرکت قیمت اضافه یا کم می‌کند:
    - اگر close > close_قبلی: OBV += volume
    - اگر close < close_قبلی: OBV -= volume
    - اگر close == close_قبلی: OBV بدون تغییر
    
    Parameters:
    -----------
    close : np.ndarray
        آرایه قیمت‌های بسته شدن
    volume : np.ndarray
        آرایه حجم معاملات
    
    Returns:
    --------
    np.ndarray
        آرایه مقادیر OBV (تجمعی)
    """
    length = len(close)
    result = np.empty(length, dtype=np.float64)
    
    state = OBVState()
    
    for i in range(length):
        result[i] = state.update(close[i], volume[i])
    
    return result


def obv_batch_df(df, close_column='close', volume_column='volume',
                 result_col='obv', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    close_column : str, optional
        نام ستون قیمت بسته شدن (پیش‌فرض: 'close')
    volume_column : str, optional
        نام ستون حجم معاملات (پیش‌فرض: 'volume')
    output_col : str, optional
        نام ستون خروجی (پیش‌فرض: 'obv')
    
    Returns:
    --------
    pd.Series
        سری مقادیر OBV
    """
    close = df[close_column].to_numpy(dtype=np.float64)
    volume = df[volume_column].to_numpy(dtype=np.float64)
    
    result = obv_batch(close, volume)
    
    return pd.DataFrame({
        f'{result_col}': result
    }, index=df.index)


# ============================================================================
# 15. Williams %R
# ============================================================================
@njit
def williamsr_batch(high, low, close, n=14, min_periods=-1):
    """
    محاسبه Williams %R به صورت batch با استفاده از WilliamsRState
    
    Williams %R یک اسیلاتور مومنتوم است که موقعیت قیمت بسته شدن را نسبت به 
    محدوده high-low در n دوره اخیر اندازه‌گیری می‌کند:
    %R = -100 × (HH - Close) / (HH - LL)
    
    مقادیر بین 0 تا -100 است:
    - نزدیک به 0: overbought
    - نزدیک به -100: oversold
    
    Parameters:
    -----------
    high : np.ndarray
        آرایه قیمت‌های بالا
    low : np.ndarray
        آرایه قیمت‌های پایین
    close : np.ndarray
        آرایه قیمت‌های بسته شدن
    n : int, optional
        تعداد دوره‌های lookback (پیش‌فرض: 14)
    min_periods : int, optional
        حداقل تعداد داده برای محاسبه معتبر (پیش‌فرض: -1 یعنی برابر n)
    
    Returns:
    --------
    np.ndarray
        آرایه مقادیر Williams %R (بین 0 تا -100)
    """
    length = len(close)
    result = np.empty(length, dtype=np.float64)
    
    state = WilliamsRState(n, min_periods)
    
    for i in range(length):
        result[i] = state.update(high[i], low[i], close[i])
    
    return result


def williamsr_batch_df(df, high_column='high', low_column='low', close_column='close',
                       period=14, min_periods=-1,
                       result_col='williamsr', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    period : int, optional
        تعداد دوره‌های lookback (پیش‌فرض: 14)
    min_periods : int, optional
        حداقل تعداد داده برای محاسبه معتبر (پیش‌فرض: -1 یعنی برابر n)
    high_column : str, optional
        نام ستون قیمت بالا (پیش‌فرض: 'high')
    low_column : str, optional
        نام ستون قیمت پایین (پیش‌فرض: 'low')
    close_column : str, optional
        نام ستون قیمت بسته شدن (پیش‌فرض: 'close')
    output_col : str, optional
        نام ستون خروجی (پیش‌فرض: 'williamsr')
    
    Returns:
    --------
    pd.Series
        سری مقادیر Williams %R
    """
    high = df[high_column].to_numpy(dtype=np.float64)
    low = df[low_column].to_numpy(dtype=np.float64)
    close = df[close_column].to_numpy(dtype=np.float64)
    
    result = williamsr_batch(high, low, close, n=period, min_periods=min_periods)
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result,
        }, index=df.index)


# ============================================================================
# 16. Parabolic SAR
# ============================================================================
@njit
def parabolicsar_batch(high, low, af_start=0.02, af_step=0.02, af_max=0.2):
    """
    محاسبه Parabolic SAR به صورت batch با استفاده از ParabolicSARState
    
    Parabolic SAR (Stop and Reverse) یک اندیکاتور دنبال‌کننده روند است که
    نقاط توقف و معکوس شدن احتمالی را مشخص می‌کند. SAR به صورت پویا بر اساس
    قیمت و شتاب حرکت (acceleration factor) محاسبه می‌شود.
    
    فرمول:
    SAR(t+1) = SAR(t) + AF × (EP - SAR(t))
    
    جایی که:
    - AF: Acceleration Factor (از af_start شروع و با هر EP جدید af_step افزایش می‌یابد تا af_max)
    - EP: Extreme Point (بالاترین high در uptrend یا پایین‌ترین low در downtrend)
    
    معکوس شدن روند:
    - در uptrend: اگر low < SAR → تبدیل به downtrend
    - در downtrend: اگر high > SAR → تبدیل به uptrend
    
    Parameters:
    -----------
    high : np.ndarray
        آرایه قیمت‌های بالا
    low : np.ndarray
        آرایه قیمت‌های پایین
    af_start : float, optional
        مقدار اولیه Acceleration Factor (پیش‌فرض: 0.02)
    af_step : float, optional
        مقدار افزایش AF در هر EP جدید (پیش‌فرض: 0.02)
    af_max : float, optional
        حداکثر مقدار AF (پیش‌فرض: 0.2)
    
    Returns:
    --------
    np.ndarray
        آرایه مقادیر Parabolic SAR
    
    Notes:
    ------
    - اولین مقدار np.nan است (برای initialization)
    - SAR در uptrend زیر قیمت و در downtrend بالای قیمت قرار دارد
    - هنگام معکوس شدن روند، SAR به EP قبلی جهش می‌کند
    - AF پس از هر معکوس شدن به af_start ریست می‌شود
    """
    length = len(high)
    result = np.empty(length, dtype=np.float64)
    
    state = ParabolicSARState(af_start, af_step, af_max)
    
    for i in range(length):
        result[i] = state.update(high[i], low[i])
    
    return result


def parabolicsar_batch_df(df, high_column='high', low_column='low',
                  af_start=0.02, af_step=0.02, af_max=0.2,
                  result_col='psar', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    af_start : float, optional
        مقدار اولیه Acceleration Factor (پیش‌فرض: 0.02)
    af_step : float, optional
        مقدار افزایش AF در هر EP جدید (پیش‌فرض: 0.02)
    af_max : float, optional
        حداکثر مقدار AF (پیش‌فرض: 0.2)
    high_column : str, optional
        نام ستون قیمت بالا (پیش‌فرض: 'high')
    low_column : str, optional
        نام ستون قیمت پایین (پیش‌فرض: 'low')
    output_col : str, optional
        نام ستون خروجی (پیش‌فرض: 'psar')
    
    Returns:
    --------
    pd.Series
        سری مقادیر Parabolic SAR
    
    Examples:
    ---------
    >>> # استفاده پایه
    >>> df['psar'] = psar_batch_df(df)
    
    >>> # تنظیم پارامترهای AF
    >>> df['psar_fast'] = psar_batch_df(df, af_start=0.03, af_step=0.03, af_max=0.3)
    
    >>> # استفاده برای تشخیص روند
    >>> df['trend'] = np.where(df['close'] > df['psar'], 1, -1)
    """
    high = df[high_column].to_numpy(dtype=np.float64)
    low = df[low_column].to_numpy(dtype=np.float64)
    
    result = parabolicsar_batch(high, low, af_start=af_start, af_step=af_step, af_max=af_max)
    
    if add_para_to_names:    
        return pd.DataFrame({
            f'{result_col}_{af_start}_{af_step}_{af_max}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)


# ============================================================================
# 17. Heikin-Ashi
# ============================================================================
@njit
def heikinashi_batch(open_, high, low, close):
    """
    محاسبه Heikin-Ashi Candles به صورت batch با استفاده از HeikinAshiState
    
    Heikin-Ashi یک تکنیک نمایش کندل است که نویز بازار را کاهش داده و
    روندها را واضح‌تر نشان می‌دهد. کندل‌های Heikin-Ashi از میانگین قیمت‌های
    فعلی و قبلی محاسبه می‌شوند.
    
    فرمول‌ها:
    HA-Close = (Open + High + Low + Close) / 4
    HA-Open = (HA-Open(prev) + HA-Close(prev)) / 2
    HA-High = max(High, HA-Open, HA-Close)
    HA-Low = min(Low, HA-Open, HA-Close)
    
    برای اولین کندل:
    HA-Open = (Open + Close) / 2
    
    Parameters:
    -----------
    open_arr : np.ndarray
        آرایه قیمت‌های باز شدن
    high : np.ndarray
        آرایه قیمت‌های بالا
    low : np.ndarray
        آرایه قیمت‌های پایین
    close : np.ndarray
        آرایه قیمت‌های بسته شدن
    
    Returns:
    --------
    tuple of (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        چهار آرایه برای HA-Open, HA-High, HA-Low, HA-Close
    
    Notes:
    ------
    - کندل‌های Heikin-Ashi معمولاً روندها را صاف‌تر نشان می‌دهند
    - کندل‌های سبز متوالی نشان‌دهنده روند صعودی قوی
    - کندل‌های قرمز متوالی نشان‌دهنده روند نزولی قوی
    - کندل‌های با سایه کوچک نشان‌دهنده روند قوی‌تر
    - از HA برای تشخیص نقاط ورود/خروج و فیلتر کردن سیگنال‌های نویزی استفاده می‌شود
    
    Examples:
    ---------
    >>> ha_o, ha_h, ha_l, ha_c = heikinashi_batch(open, high, low, close)
    >>> # تشخیص روند صعودی قوی: کندل‌های سبز بدون سایه پایین
    >>> strong_uptrend = (ha_c > ha_o) & (ha_l == ha_o)
    """
    length = len(open_)
    ha_open = np.empty(length, dtype=np.float64)
    ha_high = np.empty(length, dtype=np.float64)
    ha_low = np.empty(length, dtype=np.float64)
    ha_close = np.empty(length, dtype=np.float64)
    
    state = HeikinAshiState()
    
    for i in range(length):
        o, h, l, c = state.update(open_[i], high[i], low[i], close[i])
        ha_open[i] = o
        ha_high[i] = h
        ha_low[i] = l
        ha_close[i] = c
    
    return ha_open, ha_high, ha_low, ha_close


def heikinashi_batch_df(df,
                        open_column='open', high_column='high', low_column='low', close_column='close',
                        result_prefix='ha', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    open_column : str, optional
        نام ستون قیمت باز شدن (پیش‌فرض: 'open')
    high_column : str, optional
        نام ستون قیمت بالا (پیش‌فرض: 'high')
    low_column : str, optional
        نام ستون قیمت پایین (پیش‌فرض: 'low')
    close_column : str, optional
        نام ستون قیمت بسته شدن (پیش‌فرض: 'close')
    output_prefix : str, optional
        پیشوند برای ستون‌های خروجی (پیش‌فرض: 'ha_')
    
    Returns:
    --------
    pd.DataFrame
        دیتافریم با چهار ستون: {prefix}open, {prefix}high, {prefix}low, {prefix}close
    
    Examples:
    ---------
    >>> # استفاده پایه
    >>> ha_df = heikinashi_batch_df(df)
    >>> # ستون‌ها: ha_open, ha_high, ha_low, ha_close
    
    >>> # تغییر پیشوند
    >>> ha_df = heikinashi_batch_df(df, output_prefix='heikin_')
    
    >>> # افزودن به دیتافریم اصلی
    >>> df[['ha_open', 'ha_high', 'ha_low', 'ha_close']] = heikinashi_batch_df(df)
    
    >>> # تشخیص کندل‌های صعودی/نزولی
    >>> ha_df = heikinashi_batch_df(df)
    >>> df['ha_bullish'] = ha_df['ha_close'] > ha_df['ha_open']
    >>> df['ha_bearish'] = ha_df['ha_close'] < ha_df['ha_open']
    
    >>> # تشخیص روند قوی (بدون سایه در جهت مخالف)
    >>> df['strong_uptrend'] = (ha_df['ha_close'] > ha_df['ha_open']) & \
    ...                         (ha_df['ha_low'] == ha_df['ha_open'])
    >>> df['strong_downtrend'] = (ha_df['ha_close'] < ha_df['ha_open']) & \
    ...                           (ha_df['ha_high'] == ha_df['ha_open'])
    """
    open_arr = df[open_column].to_numpy(dtype=np.float64)
    high = df[high_column].to_numpy(dtype=np.float64)
    low = df[low_column].to_numpy(dtype=np.float64)
    close = df[close_column].to_numpy(dtype=np.float64)
    
    ha_o, ha_h, ha_l, ha_c = heikinashi_batch(open_arr, high, low, close)
    
    return pd.DataFrame({
        f'{result_prefix}_open': ha_o,
        f'{result_prefix}_high': ha_h,
        f'{result_prefix}_low': ha_l,
        f'{result_prefix}_close': ha_c
    }, index=df.index)


# ============================================================================
# 18. Supertrend (classic=0, "wilder=1", ema=2)
# ============================================================================
@njit
def supertrend_batch(high, low, close, period=10, multiplier=3.0, 
                     atr_method=1, min_periods=-1):
    """
    محاسبه Supertrend به صورت batch با استفاده از SupertrendState
    
    Supertrend یک اندیکاتور trend-following است که بر اساس ATR عمل می‌کند.
    این اندیکاتور باندهای پویا ایجاد کرده و جهت روند را تشخیص می‌دهد.
    
    فرمول‌ها:
    Basic Upper Band = (High + Low) / 2 + multiplier × ATR
    Basic Lower Band = (High + Low) / 2 - multiplier × ATR
    
    Final Upper Band:
    - اگر Basic Upper < Final Upper(قبلی) یا Close > Final Upper(قبلی):
      Final Upper = Basic Upper
    - در غیر این صورت: Final Upper = Final Upper(قبلی)
    
    Final Lower Band:
    - اگر Basic Lower > Final Lower(قبلی) یا Close < Final Lower(قبلی):
      Final Lower = Basic Lower
    - در غیر این صورت: Final Lower = Final Lower(قبلی)
    
    تشخیص روند:
    - اگر Close <= Final Lower: تغییر به downtrend (direction = -1)
    - اگر Close >= Final Upper: تغییر به uptrend (direction = 1)
    - Supertrend = Final Lower در uptrend، Final Upper در downtrend
    
    Parameters:
    -----------
    high : np.ndarray
        آرایه قیمت‌های بالا
    low : np.ndarray
        آرایه قیمت‌های پایین
    close : np.ndarray
        آرایه قیمت‌های بسته شدن
    period : int, optional
        دوره ATR (پیش‌فرض: 10)
    multiplier : float, optional
        ضریب باند (پیش‌فرض: 3.0)
    atr_method : int, optional
        روش محاسبه ATR:
        - 0: classic (SMA of TR)
        - 1: wilder (پیش‌فرض، RMA)
        - 2: ema (EMA of TR)
    min_periods : int, optional
        حداقل تعداد داده برای مقدار معتبر
        (پیش‌فرض: -1 که به period تبدیل می‌شود)
    
    Returns:
    --------
    tuple of (np.ndarray, np.ndarray)
        - supertrend: مقادیر Supertrend (np.nan برای warm-up period)
        - direction: جهت روند (1 = uptrend, -1 = downtrend)
    
    Notes:
    ------
    - Supertrend یک اندیکاتور lagging است که برای تشخیص روند استفاده می‌شود
    - direction = 1 (uptrend): قیمت بالای Supertrend → سیگنال خرید
    - direction = -1 (downtrend): قیمت زیر Supertrend → سیگنال فروش
    - تغییر direction نشان‌دهنده reversal احتمالی است
    - multiplier بالاتر → باندهای وسیع‌تر → سیگنال‌های کمتر اما قوی‌تر
    - multiplier پایین‌تر → باندهای باریک‌تر → سیگنال‌های بیشتر اما نویزی‌تر
    - مقادیر رایج: period=10, multiplier=3.0 (پیش‌فرض)
    - برای بازارهای پرنوسان: multiplier بالاتر (3.5-4.0)
    - برای بازارهای آرام: multiplier پایین‌تر (2.0-2.5)
    
    Examples:
    ---------
    >>> # استفاده پایه با Wilder ATR (پیش‌فرض)
    >>> st, direction = supertrend_batch(high, low, close)
    
    >>> # با Classic ATR
    >>> st, direction = supertrend_batch(high, low, close, atr_method=0)
    
    >>> # با EMA ATR و multiplier بالاتر
    >>> st, direction = supertrend_batch(high, low, close, 
    ...                                  period=14, multiplier=4.0, atr_method=2)
    
    >>> # تشخیص سیگنال‌های خرید/فروش
    >>> st, direction = supertrend_batch(high, low, close)
    >>> buy_signal = (direction == 1) & (np.roll(direction, 1) == -1)  # تغییر به uptrend
    >>> sell_signal = (direction == -1) & (np.roll(direction, 1) == 1)  # تغییر به downtrend
    """
    length = len(high)
    supertrend = np.empty(length, dtype=np.float64)
    direction = np.empty(length, dtype=np.int32)
    
    state = SupertrendState(period, multiplier, atr_method, min_periods)
    
    for i in range(length):
        st_val, dir_val = state.update(high[i], low[i], close[i])
        supertrend[i] = st_val
        direction[i] = dir_val
    
    return supertrend, direction


def supertrend_batch_df(df, high_column='high', low_column='low', close_column='close',
                        period=10, multiplier=3.0, method='wilder', min_periods=-1,
                        super_col='supertrend', direction_col='st_direction', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    high_column : str, optional
        نام ستون قیمت بالا (پیش‌فرض: 'high')
    low_column : str, optional
        نام ستون قیمت پایین (پیش‌فرض: 'low')
    close_column : str, optional
        نام ستون قیمت بسته شدن (پیش‌فرض: 'close')
    period : int, optional
        دوره ATR (پیش‌فرض: 10)
    multiplier : float, optional
        ضریب باند (پیش‌فرض: 3.0)
    atr_method : int or str, optional
        روش محاسبه ATR:
        - 0 یا 'classic': SMA of TR
        - 1 یا 'wilder': RMA (پیش‌فرض)
        - 2 یا 'ema': EMA of TR
    min_periods : int, optional
        حداقل تعداد داده برای مقدار معتبر (پیش‌فرض: -1 → period)
    output_col : str, optional
        نام ستون خروجی Supertrend (پیش‌فرض: 'supertrend')
    direction_col : str, optional
        نام ستون خروجی جهت (پیش‌فرض: 'st_direction')
    
    Returns:
    --------
    pd.DataFrame
        دیتافریم با دو ستون: Supertrend و direction
    
    Examples:
    ---------
    >>> # استفاده پایه
    >>> st_df = supertrend_batch_df(df)
    >>> # ستون‌ها: supertrend, st_direction
    
    >>> # با پارامترهای سفارشی
    >>> st_df = supertrend_batch_df(df, period=14, multiplier=4.0, atr_method='ema')
    
    >>> # افزودن به دیتافریم اصلی
    >>> df[['supertrend', 'st_direction']] = supertrend_batch_df(df)
    
    >>> # تشخیص سیگنال‌های معاملاتی
    >>> st_df = supertrend_batch_df(df)
    >>> df['st'] = st_df['supertrend']
    >>> df['st_dir'] = st_df['st_direction']
    >>> 
    >>> # سیگنال خرید: تغییر از downtrend به uptrend
    >>> df['buy_signal'] = (df['st_dir'] == 1) & (df['st_dir'].shift(1) == -1)
    >>> 
    >>> # سیگنال فروش: تغییر از uptrend به downtrend
    >>> df['sell_signal'] = (df['st_dir'] == -1) & (df['st_dir'].shift(1) == 1)
    >>> 
    >>> # فاصله قیمت از Supertrend (برای تشخیص قدرت روند)
    >>> df['st_distance'] = (df['close'] - df['st']) / df['st'] * 100
    
    >>> # استراتژی ترکیبی با RSI
    >>> st_df = supertrend_batch_df(df, period=10, multiplier=3.0)
    >>> rsi_df = rsi_batch_df(df, n=14)
    >>> df['st_dir'] = st_df['st_direction']
    >>> df['rsi'] = rsi_df['rsi']
    >>> 
    >>> # خرید: Supertrend uptrend + RSI oversold
    >>> df['strong_buy'] = (df['st_dir'] == 1) & (df['rsi'] < 30)
    >>> 
    >>> # فروش: Supertrend downtrend + RSI overbought
    >>> df['strong_sell'] = (df['st_dir'] == -1) & (df['rsi'] > 70)
    
    >>> # مقایسه روش‌های مختلف ATR
    >>> st_classic = supertrend_batch_df(df, atr_method='classic', 
    ...                                  output_col='st_classic')
    >>> st_wilder = supertrend_batch_df(df, atr_method='wilder', 
    ...                                 output_col='st_wilder')
    >>> st_ema = supertrend_batch_df(df, atr_method='ema', 
    ...                              output_col='st_ema')
    """
    # تبدیل method از string به int
    if isinstance(method, str):
        method_map = {'classic': 0, 'wilder': 1, 'ema': 2}
        atr_method_int = method_map.get(str(method).lower(), 1)  # default: 'wilder'
    else:
        atr_method_int = int(method)  # اگر عدد بود، مستقیم استفاده کن

    high = df[high_column].to_numpy(dtype=np.float64)
    low = df[low_column].to_numpy(dtype=np.float64)
    close = df[close_column].to_numpy(dtype=np.float64)
    
    st, direction = supertrend_batch(high, low, close, period=period, 
                                     multiplier=multiplier, atr_method=atr_method_int,
                                     min_periods=min_periods)
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{super_col}_{period}_{multiplier}_{method}': st,
            f'{direction_col}_{period}_{multiplier}_{method}': direction
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{super_col}': st,
            f'{direction_col}': direction
        }, index=df.index)


# ============================================================================
# 19. Aroon
# ============================================================================
@njit
def aroon_batch(high, low, period=25, min_periods=-1):
    """
    محاسبه Aroon Indicator به صورت batch با استفاده از AroonState
    
    Aroon یک اندیکاتور روند است که قدرت و جهت روند را با اندازه‌گیری
    زمان از آخرین high/low تشخیص می‌دهد.
    
    فرمول‌ها:
    Aroon Up = ((period - bars_since_high) / period) × 100
    Aroon Down = ((period - bars_since_low) / period) × 100
    Aroon Oscillator = Aroon Up - Aroon Down
    
    تفسیر:
    - Aroon Up نزدیک 100: روند صعودی قوی (high اخیر)
    - Aroon Down نزدیک 100: روند نزولی قوی (low اخیر)
    - Aroon Up > 70 و Aroon Down < 30: روند صعودی
    - Aroon Down > 70 و Aroon Up < 30: روند نزولی
    - هر دو نزدیک 50: بازار خنثی/رنج
    - Oscillator > 0: روند صعودی غالب
    - Oscillator < 0: روند نزولی غالب
    
    Parameters:
    -----------
    high : np.ndarray
        آرایه قیمت‌های بالا
    low : np.ndarray
        آرایه قیمت‌های پایین
    period : int, optional
        دوره lookback (پیش‌فرض: 25)
    min_periods : int, optional
        حداقل تعداد داده برای مقدار معتبر
        (پیش‌فرض: -1 که به period تبدیل می‌شود)
    
    Returns:
    --------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        - aroon_up: مقادیر Aroon Up (0-100)
        - aroon_down: مقادیر Aroon Down (0-100)
        - oscillator: Aroon Oscillator (-100 تا +100)
    
    Notes:
    ------
    - Aroon توسط Tushar Chande در سال 1995 معرفی شد
    - نام Aroon از زبان سانسکریت به معنی "نور سپیده‌دم" است
    - برخلاف اسیلاتورهای دیگر، Aroon بر زمان تمرکز دارد نه قیمت
    - مقدار 100 = high/low در همین لحظه رخ داده
    - مقدار 0 = high/low در ابتدای period رخ داده
    - Aroon Up و Down می‌توانند همزمان بالا باشند (consolidation)
    - تقاطع Aroon Up و Down می‌تواند سیگنال تغییر روند باشد
    - دوره معمول: 25 (حدود یک ماه معاملاتی)
    - برای timeframe کوتاه‌تر: period=14
    - برای timeframe بلندتر: period=50
    
    Examples:
    ---------
    >>> # استفاده پایه
    >>> aroon_up, aroon_down, oscillator = aroon_batch(high, low)
    
    >>> # با period سفارشی
    >>> aroon_up, aroon_down, oscillator = aroon_batch(high, low, period=14)
    
    >>> # با min_periods کمتر برای سیگنال زودتر
    >>> aroon_up, aroon_down, oscillator = aroon_batch(high, low, 
    ...                                                 period=25, min_periods=10)
    
    >>> # تشخیص روند صعودی قوی
    >>> strong_uptrend = (aroon_up > 70) & (aroon_down < 30)
    
    >>> # تشخیص روند نزولی قوی
    >>> strong_downtrend = (aroon_down > 70) & (aroon_up < 30)
    
    >>> # تشخیص consolidation (هر دو بالا)
    >>> consolidation = (aroon_up > 70) & (aroon_down > 70)
    
    >>> # سیگنال خرید: تقاطع Aroon Up از پایین
    >>> buy_signal = (aroon_up > aroon_down) & (np.roll(aroon_up, 1) <= np.roll(aroon_down, 1))
    
    >>> # سیگنال فروش: تقاطع Aroon Down از پایین
    >>> sell_signal = (aroon_down > aroon_up) & (np.roll(aroon_down, 1) <= np.roll(aroon_up, 1))
    """
    length = len(high)
    aroon_up = np.empty(length, dtype=np.float64)
    aroon_down = np.empty(length, dtype=np.float64)
    oscillator = np.empty(length, dtype=np.float64)
    
    state = AroonState(period, min_periods)
    
    for i in range(length):
        up, down, osc = state.update(high[i], low[i])
        aroon_up[i] = up
        aroon_down[i] = down
        oscillator[i] = osc
    
    return aroon_up, aroon_down, oscillator


def aroon_batch_df(df, high_column='high', low_column='low',
                   period=25, min_periods=-1,
                   aroon_up_col='aroon_up', aroon_down_col='aroon_down',
                   aroon_osc_col='aroon_oscillator', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    high_column : str, optional
        نام ستون قیمت بالا (پیش‌فرض: 'high')
    low_column : str, optional
        نام ستون قیمت پایین (پیش‌فرض: 'low')
    period : int, optional
        دوره lookback (پیش‌فرض: 25)
    min_periods : int, optional
        حداقل تعداد داده برای مقدار معتبر (پیش‌فرض: -1 → period)
    aroon_up_col : str, optional
        نام ستون خروجی Aroon Up (پیش‌فرض: 'aroon_up')
    aroon_down_col : str, optional
        نام ستون خروجی Aroon Down (پیش‌فرض: 'aroon_down')
    oscillator_col : str, optional
        نام ستون خروجی Oscillator (پیش‌فرض: 'aroon_oscillator')
    
    Returns:
    --------
    pd.DataFrame
        دیتافریم با سه ستون: Aroon Up, Aroon Down, Oscillator
    
    Examples:
    ---------
    >>> # استفاده پایه
    >>> aroon_df = aroon_batch_df(df)
    >>> # ستون‌ها: aroon_up, aroon_down, aroon_oscillator
    
    >>> # با period سفارشی
    >>> aroon_df = aroon_batch_df(df, period=14)
    
    >>> # افزودن به دیتافریم اصلی
    >>> df[['aroon_up', 'aroon_down', 'aroon_osc']] = aroon_batch_df(df)
    
    >>> # تشخیص سیگنال‌های معاملاتی
    >>> aroon_df = aroon_batch_df(df, period=25)
    >>> df['aroon_up'] = aroon_df['aroon_up']
    >>> df['aroon_down'] = aroon_df['aroon_down']
    >>> df['aroon_osc'] = aroon_df['aroon_oscillator']
    >>> 
    >>> # روند صعودی قوی
    >>> df['strong_uptrend'] = (df['aroon_up'] > 70) & (df['aroon_down'] < 30)
    >>> 
    >>> # روند نزولی قوی
    >>> df['strong_downtrend'] = (df['aroon_down'] > 70) & (df['aroon_up'] < 30)
    >>> 
    >>> # بازار خنثی/رنج
    >>> df['neutral'] = (df['aroon_up'] < 50) & (df['aroon_down'] < 50)
    >>> 
    >>> # Consolidation (هر دو بالا)
    >>> df['consolidation'] = (df['aroon_up'] > 70) & (df['aroon_down'] > 70)
    
    >>> # سیگنال‌های تقاطع
    >>> aroon_df = aroon_batch_df(df)
    >>> df['aroon_up'] = aroon_df['aroon_up']
    >>> df['aroon_down'] = aroon_df['aroon_down']
    >>> 
    >>> # تقاطع صعودی (Aroon Up از پایین عبور می‌کند)
    >>> df['bullish_cross'] = (
    ...     (df['aroon_up'] > df['aroon_down']) & 
    ...     (df['aroon_up'].shift(1) <= df['aroon_down'].shift(1))
    ... )
    >>> 
    >>> # تقاطع نزولی (Aroon Down از پایین عبور می‌کند)
    >>> df['bearish_cross'] = (
    ...     (df['aroon_down'] > df['aroon_up']) & 
    ...     (df['aroon_down'].shift(1) <= df['aroon_up'].shift(1))
    ... )
    
    >>> # استراتژی ترکیبی با ADX
    >>> aroon_df = aroon_batch_df(df, period=25)
    >>> adx_df = adx_batch_df(df, n=14)  # فرض: تابع adx_batch_df موجود است
    >>> df['aroon_up'] = aroon_df['aroon_up']
    >>> df['aroon_down'] = aroon_df['aroon_down']
    >>> df['adx'] = adx_df['adx']
    >>> 
    >>> # خرید: Aroon صعودی + ADX قوی
    >>> df['strong_buy'] = (
    ...     (df['aroon_up'] > 70) & 
    ...     (df['aroon_down'] < 30) & 
    ...     (df['adx'] > 25)
    ... )
    >>> 
    >>> # فروش: Aroon نزولی + ADX قوی
    >>> df['strong_sell'] = (
    ...     (df['aroon_down'] > 70) & 
    ...     (df['aroon_up'] < 30) & 
    ...     (df['adx'] > 25)
    ... )
    
    >>> # استفاده از Oscillator برای تایید روند
    >>> aroon_df = aroon_batch_df(df)
    >>> df['aroon_osc'] = aroon_df['aroon_oscillator']
    >>> 
    >>> # روند صعودی: Oscillator مثبت و در حال افزایش
    >>> df['uptrend_confirmed'] = (
    ...     (df['aroon_osc'] > 0) & 
    ...     (df['aroon_osc'] > df['aroon_osc'].shift(1))
    ... )
    >>> 
    >>> # روند نزولی: Oscillator منفی و در حال کاهش
    >>> df['downtrend_confirmed'] = (
    ...     (df['aroon_osc'] < 0) & 
    ...     (df['aroon_osc'] < df['aroon_osc'].shift(1))
    ... )
    
    >>> # تشخیص تغییر روند زودهنگام
    >>> aroon_df = aroon_batch_df(df, period=25)
    >>> df['aroon_up'] = aroon_df['aroon_up']
    >>> df['aroon_down'] = aroon_df['aroon_down']
    >>> 
    >>> # شروع روند صعودی: Aroon Up بالای 50 و در حال افزایش
    >>> df['early_uptrend'] = (
    ...     (df['aroon_up'] > 50) & 
    ...     (df['aroon_up'] > df['aroon_up'].shift(1)) &
    ...     (df['aroon_down'] < 50)
    ... )
    >>> 
    >>> # شروع روند نزولی: Aroon Down بالای 50 و در حال افزایش
    >>> df['early_downtrend'] = (
    ...     (df['aroon_down'] > 50) & 
    ...     (df['aroon_down'] > df['aroon_down'].shift(1)) &
    ...     (df['aroon_up'] < 50)
    ... )
    
    >>> # مقایسه period های مختلف
    >>> aroon_short = aroon_batch_df(df, period=14, 
    ...                              aroon_up_col='aroon_up_14',
    ...                              aroon_down_col='aroon_down_14')
    >>> aroon_long = aroon_batch_df(df, period=50,
    ...                             aroon_up_col='aroon_up_50',
    ...                             aroon_down_col='aroon_down_50')
    >>> 
    >>> # سیگنال قوی: هر دو timeframe موافق
    >>> df['strong_signal'] = (
    ...     (aroon_short['aroon_up_14'] > 70) & 
    ...     (aroon_long['aroon_up_50'] > 70)
    ... )
    """
    high = df[high_column].to_numpy(dtype=np.float64)
    low = df[low_column].to_numpy(dtype=np.float64)
    
    aroon_up, aroon_down, oscillator = aroon_batch(high, low, 
                                                    period=period, 
                                                    min_periods=min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{aroon_up_col}_{period}': aroon_up,
            f'{aroon_down_col}_{period}': aroon_down,
            f'{aroon_osc_col}_{period}': oscillator
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{aroon_up_col}': aroon_up,
            f'{aroon_down_col}': aroon_down,
            f'{aroon_osc_col}': oscillator
        }, index=df.index)


# ============================================================================
# 20. DEMA (Double Exponential Moving Average)
# ============================================================================
@njit
def dema_batch(data, n=20, min_periods=-1):
    """
    محاسبه Double Exponential Moving Average به صورت batch
    
    DEMA = 2×EMA(price) - EMA(EMA(price))
    
    Parameters:
    -----------
    data : np.ndarray
        آرایه قیمت‌ها
    n : int, optional
        دوره EMA (پیش‌فرض: 20)
    min_periods : int, optional
        حداقل تعداد داده برای مقدار معتبر (پیش‌فرض: -1 → n)
    
    Returns:
    --------
    np.ndarray
        آرایه مقادیر DEMA
    
    Examples:
    ---------
    >>> dema = dema_batch(close, n=20)
    >>> dema = dema_batch(close, n=10, min_periods=5)
    """
    length = len(data)
    result = np.empty(length, dtype=np.float64)
    
    state = DEMAState(n, min_periods)
    
    for i in range(length):
        result[i] = state.update(data[i])
    
    return result


def dema_batch_df(df, column='close',
                  period=20, min_periods=-1,
                  result_col='dema', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    column : str, optional
        نام ستون قیمت (پیش‌فرض: 'close')
    n : int, optional
        دوره EMA (پیش‌فرض: 20)
    min_periods : int, optional
        حداقل تعداد داده (پیش‌فرض: -1 → n)
    dema_col : str, optional
        نام ستون خروجی (پیش‌فرض: 'dema')
    
    Returns:
    --------
    pd.DataFrame
        دیتافریم با ستون DEMA
    
    Examples:
    ---------
    >>> df['dema'] = dema_batch_df(df)
    >>> df['dema_10'] = dema_batch_df(df, n=10)
    >>> # تقاطع قیمت و DEMA
    >>> df['dema'] = dema_batch_df(df, n=20)
    >>> df['signal'] = np.where(df['close'] > df['dema'], 1, -1)
    """
    data = df[column].to_numpy(dtype=np.float64)
    result = dema_batch(data, n=period, min_periods=min_periods)
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)


# ============================================================================
# 21. TEMA (Triple Exponential Moving Average)
# ============================================================================
@njit
def tema_batch(data, n=20, min_periods=-1):
    """
    محاسبه Triple Exponential Moving Average به صورت batch
    
    TEMA = 3×EMA(price) - 3×EMA(EMA(price)) + EMA(EMA(EMA(price)))
    
    Parameters:
    -----------
    data : np.ndarray
        آرایه قیمت‌ها
    n : int, optional
        دوره EMA (پیش‌فرض: 20)
    min_periods : int, optional
        حداقل تعداد داده برای مقدار معتبر (پیش‌فرض: -1 → n)
    
    Returns:
    --------
    np.ndarray
        آرایه مقادیر TEMA
    
    Examples:
    ---------
    >>> tema = tema_batch(close, n=20)
    >>> tema = tema_batch(close, n=10, min_periods=5)
    """
    length = len(data)
    result = np.empty(length, dtype=np.float64)
    
    state = TEMAState(n, min_periods)
    
    for i in range(length):
        result[i] = state.update(data[i])
    
    return result


def tema_batch_df(df, column='close',
                  period=20, min_periods=-1,
                  result_col='tema', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    column : str, optional
        نام ستون قیمت (پیش‌فرض: 'close')
    n : int, optional
        دوره EMA (پیش‌فرض: 20)
    min_periods : int, optional
        حداقل تعداد داده (پیش‌فرض: -1 → n)
    tema_col : str, optional
        نام ستون خروجی (پیش‌فرض: 'tema')
    
    Returns:
    --------
    pd.DataFrame
        دیتافریم با ستون TEMA
    
    Examples:
    ---------
    >>> df['tema'] = tema_batch_df(df)
    >>> df['tema_10'] = tema_batch_df(df, n=10)
    >>> # مقایسه EMA, DEMA, TEMA
    >>> df['ema'] = ema_batch_df(df, n=20)
    >>> df['dema'] = dema_batch_df(df, n=20)
    >>> df['tema'] = tema_batch_df(df, n=20)
    """
    data = df[column].to_numpy(dtype=np.float64)
    result = tema_batch(data, n=period, min_periods=min_periods)
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)


# ============================================================================
# 22. KAMA (Kaufman’s Adaptive Moving Average)
# ============================================================================
@njit
def kama_batch(data, n=10, fast_span=2, slow_span=30, min_periods=-1):
    """
    محاسبه Kaufman's Adaptive Moving Average به صورت batch
    
    KAMA با استفاده از Efficiency Ratio به نوسانات بازار واکنش نشان می‌دهد
    
    Parameters:
    -----------
    data : np.ndarray
        آرایه قیمت‌ها
    n : int, optional
        دوره محاسبه Efficiency Ratio (پیش‌فرض: 10)
    fast_span : int, optional
        Fast EMA span (پیش‌فرض: 2)
    slow_span : int, optional
        Slow EMA span (پیش‌فرض: 30)
    min_periods : int, optional
        حداقل تعداد داده (پیش‌فرض: -1 → n)
    
    Returns:
    --------
    np.ndarray
        آرایه مقادیر KAMA
    
    Examples:
    ---------
    >>> kama = kama_batch(close)
    >>> kama = kama_batch(close, n=10, fast_span=2, slow_span=30)
    """
    length = len(data)
    result = np.empty(length, dtype=np.float64)
    
    state = KAMAState(n, fast_span, slow_span, min_periods)
    
    for i in range(length):
        result[i] = state.update(data[i])
    
    return result


def kama_batch_df(df, column='close',
                  period=10, fast_span=2, slow_span=30, min_periods=-1,
                  result_col='kama', add_para_to_names = False):
    """
    Wrapper برای استفاده با DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    column : str, optional
        نام ستون قیمت (پیش‌فرض: 'close')
    n : int, optional
        دوره ER (پیش‌فرض: 10)
    fast_span : int, optional
        Fast span (پیش‌فرض: 2)
    slow_span : int, optional
        Slow span (پیش‌فرض: 30)
    min_periods : int, optional
        حداقل تعداد داده (پیش‌فرض: -1 → n)
    kama_col : str, optional
        نام ستون خروجی (پیش‌فرض: 'kama')
    
    Returns:
    --------
    pd.DataFrame
        دیتافریم با ستون KAMA
    
    Examples:
    ---------
    >>> df['kama'] = kama_batch_df(df)
    >>> df['kama_fast'] = kama_batch_df(df, n=5, fast_span=2, slow_span=20)
    >>> # سیگنال تقاطع
    >>> df['kama'] = kama_batch_df(df)
    >>> df['signal'] = np.where(df['close'] > df['kama'], 1, -1)
    """
    data = df[column].to_numpy(dtype=np.float64)
    result = kama_batch(data, n=period, fast_span=fast_span, slow_span=slow_span, 
                       min_periods=min_periods)
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}_{fast_span}_{slow_span}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)


# ============================================================================
# 23. HMA (Hull Moving Average)
# ============================================================================
@njit
def hma_batch(data, n=20, min_periods=-1):
    """
    محاسبه Hull Moving Average برای آرایه کامل داده.
    
    HMA = WMA(sqrt(n)) of [2*WMA(n/2) - WMA(n)]
    
    Parameters:
    -----------
    data : np.ndarray
        آرایه قیمت‌ها
    n : int
        دوره HMA (پیش‌فرض: 20)
    min_periods : int
        حداقل داده برای محاسبه (پیش‌فرض: -1 = n)
    
    Returns:
    --------
    np.ndarray: آرایه مقادیر HMA
    """
    length = len(data)
    result = np.empty(length, dtype=np.float64)
    
    state = HMAState(n, min_periods)
    
    for i in range(length):
        result[i] = state.update(data[i])
    
    return result


def hma_batch_df(df, column='close',
                 period=20, min_periods=-1,
                 result_col='hma', add_para_to_names = False):
    """
    محاسبه Hull Moving Average برای DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم ورودی
    column : str
        نام ستون قیمت (پیش‌فرض: 'close')
    n : int
        دوره HMA (پیش‌فرض: 20)
    min_periods : int
        حداقل داده برای محاسبه (پیش‌فرض: -1 = n)
    result_col : str
        نام ستون خروجی (پیش‌فرض: 'HMA')
    
    Returns:
    --------
    pd.DataFrame: دیتافریم با ستون HMA اضافه شده
    
    مثال:
    ------
    # HMA استاندارد 20 دوره‌ای
    df = hma_batch_df(df, column='close', n=20)
    
    # HMA سریع 9 دوره‌ای
    df = hma_batch_df(df, column='close', n=9, result_col='HMA_9')
    
    # HMA با min_periods سفارشی
    df = hma_batch_df(df, column='close', n=20, min_periods=10)
    
    # استراتژی تقاطع HMA
    df = hma_batch_df(df, n=9, result_col='HMA_fast')
    df = hma_batch_df(df, n=21, result_col='HMA_slow')
    df['signal'] = np.where(df['HMA_fast'] > df['HMA_slow'], 1, -1)
    """
    
    data = df[column].to_numpy(dtype=np.float64)
    result = hma_batch(data, n=period, min_periods=min_periods)
    
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)


# ============================================================================
# Registry-compatible exports
# ============================================================================
__all__ = [
    "sma_batch",           # f'{result_col}'                       # f'{result_col}_{period}'
    "wma_batch",           # f'{result_col}'                       # f'{result_col}_{period}'
    "ema_batch",           # f'{result_col}'                       # f'{result_col}_{period}'
    "roc_batch",           # f'{result_col}'                       # f'{result_col}_{period}'
    "rsi_batch",           # f'{result_col}'                       # f'{result_col}_{period}_{method}'
    "true_range_batch",    # f'{result_col}'
    "atr_batch",           # f'{result_col}'                       # f'{result_col}_{n}_{method}'
    "macd_batch",          # f'{macd_col}', f'{signal_col}', f'{hist_col}'
                           # f'{macd_col}_{fast}_{slow}_{signal}', f'{signal_col}_{fast}_{slow}_{signal}', f'{hist_col}_{fast}_{slow}_{signal}'

    "bollinger_batch",     # f'{upper_col}', f'{middle_col}', f'{lower_col}', f'{width_col}', f'{percent_col}'
                           # f'{upper_col}_{n}_{k}', f'{middle_col}_{n}_{k}', f'{lower_col}_{n}_{k}', f'{width_col}_{n}_{k}', f'{percent_col}_{n}_{k}'

    "keltner_batch",       # f'{upper_col}', f'{middle_col}', f'{lower_col}', f'{width_col}', f'{percent_col}'
                           # f'{upper_col}_{n}_{m}', f'{middle_col}_{n}_{m}', f'{lower_col}_{n}_{m}', f'{width_col}_{n}_{m}', f'{percent_col}_{n}_{m}'

    "stochastic_batch",    # f'{k_col}', f'{d_col}'
                           # f'{k_col}_{k_period}_{d_period}_{smooth_k}_{method}', f'{d_col}_{k_period}_{d_period}_{smooth_k}_{method}'

    "cci_batch",           # f'{result_col}'                       # f'{result_col}_{n}'
    "mfi_batch",           # f'{result_col}'                       # f'{result_col}_{n}'
    "obv_batch",           # f'{result_col}'
    "williamsr_batch",     # f'{result_col}'                       # f'{result_col}_{n}'
    "parabolic_sar_batch", # f'{result_col}': result               # f'{result_col}_{af_start}_{af_step}_{af_max}'
    "heikin_ashi_batch",   # f'{result_prefix}_open', f'{result_prefix}_high', f'{result_prefix}_low', f'{result_prefix}_close'
    "supertrend_batch",    # f'{super_col}',  f'{direction_col}'
                           # f'{super_col}_{period}_{multiplier}_{atr_method}', f'{direction_col}_{period}_{multiplier}_{atr_method}'

    "aroon_batch",         # f'{aroon_up_col}', f'{aroon_down_col}', f'{aroon_osc_col}'
                           # f'{aroon_up_col}_{period}', f'{aroon_down_col}_{period}', f'{aroon_osc_col}_{period}'

    "dema_batch",          # f'{result_col}'                       # f'{result_col}_{n}'
    "tema_batch",          # f'{result_col}'                       # f'{result_col}_{n}'
    "kama_batch",          # f'{result_col}'                       # f'{result_col}_{n}_{fast_span}_{slow_span}'
    "hma_batch",           # f'{result_col}'                       # f'{result_col}'
]
