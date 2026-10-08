# Last revewed at 1405/06/17

"""
Classic Technical Indicators - Wrapper Classes
Stateful wrappers around indicators_maths state classes.
Each wrapper receives candle data and maintains internal state.
"""

from typing import Literal, Optional, Tuple
from .indicators_B_maths import (
    SMAState       , WMAState       , EMAState       , ROCState,
    RSIState       , TrueRangeState , ATRState       , MACDState,
    BollingerState , KeltnerState   , StochasticState, CCIState,
    MFIState       , OBVState       , WilliamsRState , ParabolicSARState,
    HeikinAshiState, SupertrendState, AroonState     , DEMAState,
    TEMAState      , KAMAState      , HMAState
)
# ============================================================================
# Core Indicator States (PART-1)
# ============================================================================

# =================================================================== 1. SMA
class SMA:
    """ Simple Moving Average wrapper """
    
    def __init__(self, period: int = 20, min_periods: int = None, **kwargs):
        self.period = period
        self.min_periods = min_periods
        self.state = SMAState(n=period, min_periods=min_periods if min_periods is not None else -1)

    def update(self, column: float) -> float:
        return self.state.update(column)

    def reset(self) -> None:
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 2. WMA
class WMA:
    """ Weighted Moving Average wrapper """
    
    def __init__(self, period: int = 20, min_periods: int = None, **kwargs):
        self.period = period
        self.min_periods = min_periods
        self.state = WMAState(n=period, min_periods=min_periods if min_periods is not None else -1)
    
    def update(self, column: float) -> float:
        return self.state.update(column)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 3. EMA
class EMA:
    """ Exponential Moving Average wrapper """
    
    def __init__(self, period: int = 20, min_periods: int = None, **kwargs):
        self.period = period
        self.min_periods = min_periods
        self.state = EMAState(n=period, min_periods=min_periods if min_periods is not None else -1)
    
    def update(self, column: float) -> float:
        return self.state.update(column)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 4. ROC
class ROC:
    """ Rate of Change wrapper """
    
    def __init__(self, period: int = 12, **kwargs):
        self.period = period
        self.state = ROCState(n=period)
    
    def update(self, column: float) -> float:
        return self.state.update(column)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 5. RSI
class RSI:
    """ Relative Strength Index wrapper """
    
    def __init__(self, period: int = 14, method: Literal["ema", "wilders"] = "ema", **kwargs):
        self.period = period
        self.method = method
        # تبدیل string method به int برای jitclass
        method_int = 0 if method == "ema" else 1
        self.state = RSIState(n=period, method=method_int)
    
    def update(self, column: float) -> float:
        return self.state.update(column)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 6. TrueRange
class TrueRange:
    """ True Range wrapper """
    
    def __init__(self, **kwargs):
        self.state = TrueRangeState()
    
    def update(self, high: float, low: float, close: float) -> float:
        return self.state.update(high, low, close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 7. ATR
class ATR:
    """ Average True Range wrapper """
    
    def __init__(self, period: int = 14, method: Literal["classic", "wilder", "ema"] = "wilder", **kwargs):
        self.period = period
        self.method = method
        method_map = {"classic": 0, "wilder": 1, "ema": 2}
        method_int = method_map[method]
        self.state = ATRState(n=period, method=method_int)
    
    def update(self, high: float, low: float, close: float) -> float:
        return self.state.update(high, low, close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 8. MACD
class MACD:
    """ Moving Average Convergence Divergence wrapper """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.state = MACDState(fast=fast, slow=slow, signal=signal)
    
    def update(self, close: float) -> Tuple[float, float, float]:
        """Returns (macd, signal, histogram)"""
        return self.state.update(close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 9. BollingerBands
class BollingerBands:
    """ Bollinger Bands wrapper """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, min_periods: Optional[int] = None, **kwargs):
        self.period = period
        self.std_dev = std_dev
        self.min_periods = min_periods
        
        # تبدیل min_periods به فرمت jitclass
        mp = -1 if min_periods is None else min_periods
        self.state = BollingerState(n=period, k=std_dev, min_periods=mp)
    
    def update(self, close: float) -> Tuple[float, float, float, float, float]:
        """Returns (upper, middle, lower, width, percent)"""
        return self.state.update(close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 10. KeltnerChannel
class KeltnerChannel:
    """ Keltner Channel wrapper """
    
    def __init__(
        self, 
        period: int = 20, 
        multiplier: float = 2.0,
        min_periods: Optional[int] = None,
        **kwargs,
    ):
        self.period = period
        self.multiplier = multiplier
        self.min_periods = min_periods
        self.state = KeltnerState(
            n=period, 
            m=multiplier, 
            min_periods=min_periods if min_periods is not None else -1
        )
    
    def update(self, high: float, low: float, close: float) -> Tuple[float, float, float, float, float]:
        """Returns (upper, middle, lower, width, percentile)"""
        return self.state.update(high, low, close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 11. Stochastic
class Stochastic:
    """ Stochastic Oscillator wrapper """
    
    def __init__(
        self, 
        k_period: int = 14, 
        d_period: int = 3,
        smooth_k: int = 3,
        method: Literal["sma", "ema"] = "sma",
        min_periods: Optional[int] = None,
        **kwargs,
    ):
        """
        Stochastic Oscillator
        
        Parameters:
        -----------
        k_period : int
            دوره محاسبه %K (پیش‌فرض: 14)
        d_period : int
            دوره هموارسازی %D (پیش‌فرض: 3)
        smooth_k : int
            دوره هموارسازی %K (پیش‌فرض: 3)
            - smooth_k=1 → Fast Stochastic
            - smooth_k=3 → Slow Stochastic
        method : {"sma", "ema"}
            روش هموارسازی (پیش‌فرض: "sma")
        min_periods : int, optional
            حداقل داده برای شروع محاسبه
        """
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        self.method = method
        self.min_periods = min_periods
        
        # تبدیل method از string به int برای jitclass
        method_map = {"sma": 0, "ema": 1}
        method_int = method_map[method]
        
        self.state = StochasticState(
            k_period=k_period,
            d_period=d_period,
            smooth_k=smooth_k,
            method=method_int,
            min_periods=min_periods if min_periods is not None else -1
        )
    
    def update(self, high: float, low: float, close: float) -> Tuple[float, float]:
        """
        Returns:
        --------
        k : float
            مقدار %K (هموارشده)
        d : float
            مقدار %D (سیگنال)
        """
        return self.state.update(high, low, close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 12. CCI
class CCI:
    """ Commodity Channel Index wrapper """
    
    def __init__(self, period: int = 20, min_periods: Optional[int] = None, **kwargs):
        self.period = period
        self.min_periods = min_periods
        self.state = CCIState(
            n=period, 
            min_periods=min_periods if min_periods is not None else -1
        )
    
    def update(self, high: float, low: float, close: float) -> float:
        return self.state.update(high, low, close)
    
    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 13. MFI
class MFI:
    """ Money Flow Index wrapper """
    
    def __init__(self, period: int = 14, min_periods: Optional[int] = None, **kwargs):
        self.period = period
        self.min_periods = min_periods
        self.state = MFIState(
            n=period, 
            min_periods=min_periods if min_periods is not None else -1
        )
    
    def update(self, high: float, low: float, close: float, volume: float) -> float:
        return self.state.update(high, low, close, volume)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 14. OBV
class OBV:
    """ On-Balance Volume wrapper """
    
    def __init__(self, **kwargs):
        self.state = OBVState()
    
    def update(self, close: float, volume: float) -> float:
        return self.state.update(close, volume)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 15. WilliamsR
class WilliamsR:
    """ Williams %R wrapper """
    
    def __init__(self, period: int = 14, min_periods: Optional[int] = None, **kwargs):
        self.period = period
        self.state = WilliamsRState(
            n=period,
            min_periods=min_periods if min_periods is not None else -1
        )
    
    def update(self, high: float, low: float, close: float) -> float:
        return self.state.update(high, low, close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


# =================================================================== 16. ParabolicSAR
class ParabolicSAR:
    """ Parabolic SAR wrapper 

    Parameters:
    ----------
    af_start : float, default 0.02
        Initial acceleration factor
    af_step : float, default 0.02
        Acceleration factor increment on each new extreme point
    af_max : float, default 0.2
        Maximum acceleration factor
    """
    
    def __init__(self, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2, **kwargs):
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max
        self.state = ParabolicSARState(af_start=af_start, af_step=af_step, af_max=af_max)
    
    def update(self, high: float, low: float) -> float:
        """
        Returns:
        -------
        float
            SAR value (NaN on first call)
        """
        return self.state.update(high, low)

    def reset(self):
        """Reset state for new sequence"""
        self.state.reset()


# =================================================================== 17. HeikinAshi
class HeikinAshi:
    """ Heikin-Ashi wrapper """
    
    def __init__(
        self,
        open_column='open',
        high_column='high',
        low_column='low',
        close_column='close',
        result_prefix='ha',
    ):
        self.state = HeikinAshiState()
    
    def update(self, open_: float, high: float, low: float, close: float, **kwargs) -> Tuple[float, float, float, float]:
        """
        Returns: (ha_open, ha_high, ha_low, ha_close)
        On first call, HA Open = (Open + Close) / 2
        """
        return self.state.update(open_, high, low, close)

    def reset(self):
        """Reset state for new sequence"""
        self.state.reset()


# =============================================================================
# Extra Trends Indicator States (PART-2)
# =============================================================================

# =================================================================== 18. Supertrend
class Supertrend:
    """ Supertrend indicator - ATR-based trend following system
    
    Supertrend uses Average True Range (ATR) to create dynamic support/resistance
    bands that adapt to market volatility. It provides clear trend direction and
    potential reversal signals.
    
    Parameters
    ----------
    period : int, default 10
        ATR calculation period
    multiplier : float, default 3.0
        ATR multiplier for band width (higher = wider bands, fewer signals)
    atr_method : {"classic", "wilder", "ema"}, default "wilder"
        ATR calculation method:
        - "classic": Simple moving average of True Range
        - "wilder": Wilder's smoothing (RMA) - recommended
        - "ema": Exponential moving average
    min_periods : int, optional
        Minimum periods before valid output (default: period)
    
    Attributes
    ----------
    state : SupertrendState
        Internal jitclass state for high-performance computation
    
    Methods
    -------
    update(high, low, close) -> Tuple[float, int]
        Update with new OHLC data and return (supertrend_value, direction)
    reset()
        Reset all internal state to initial values
    
    Returns
    -------
    tuple of (float, int)
        - supertrend_value : float
            Current supertrend level (support in uptrend, resistance in downtrend)
            Returns NaN during warm-up period
        - direction : int
            Trend direction: 1 (uptrend/bullish), -1 (downtrend/bearish)
    
    Examples
    --------
    Basic usage with default Wilder ATR:
    
    >>> st = Supertrend(period=10, multiplier=3.0)
    >>> for high, low, close in ohlc_data:
    ...     value, direction = st.update(high, low, close)
    ...     if not np.isnan(value):
    ...         trend = "UPTREND" if direction == 1 else "DOWNTREND"
    ...         print(f"{trend}: Supertrend at {value:.2f}")
    
    Trading signals (trend changes):
    
    >>> st = Supertrend(period=10, multiplier=3.0)
    >>> prev_direction = 0
    >>> for high, low, close in ohlc_data:
    ...     value, direction = st.update(high, low, close)
    ...     if direction != prev_direction and prev_direction != 0:
    ...         if direction == 1:
    ...             print(f"BUY SIGNAL at {close:.2f}, Supertrend: {value:.2f}")
    ...         else:
    ...             print(f"SELL SIGNAL at {close:.2f}, Supertrend: {value:.2f}")
    ...     prev_direction = direction
    
    Using different ATR methods:
    
    >>> # Classic ATR (simple MA)
    >>> st_classic = Supertrend(period=14, multiplier=2.5, atr_method="classic")
    >>> 
    >>> # Wilder ATR (recommended, default)
    >>> st_wilder = Supertrend(period=14, multiplier=2.5, atr_method="wilder")
    >>> 
    >>> # EMA-based ATR (more responsive)
    >>> st_ema = Supertrend(period=14, multiplier=2.5, atr_method="ema")
    
    Custom min_periods for faster signals:
    
    >>> # Start producing signals after 5 periods instead of 10
    >>> st = Supertrend(period=10, multiplier=3.0, min_periods=5)
    >>> value, direction = st.update(high, low, close)
    
    Reset for new data sequence:
    
    >>> st = Supertrend(period=10, multiplier=3.0)
    >>> # Process first symbol
    >>> for candle in symbol1_data:
    ...     value, direction = st.update(*candle)
    >>> 
    >>> # Reset and process second symbol
    >>> st.reset()
    >>> for candle in symbol2_data:
    ...     value, direction = st.update(*candle)
    
    Notes
    -----
    **Calculation:**
    
    1. Calculate ATR using specified method
    2. Compute basic bands:
       - Basic Upper Band = (High + Low) / 2 + multiplier × ATR
       - Basic Lower Band = (High + Low) / 2 - multiplier × ATR
    
    3. Compute final bands (with persistence logic):
       - Final Upper = min(Basic Upper, Previous Final Upper) if close > Previous Final Upper
       - Final Lower = max(Basic Lower, Previous Final Lower) if close < Previous Final Lower
    
    4. Determine direction:
       - Uptrend (1): Close > Final Lower Band
       - Downtrend (-1): Close < Final Upper Band
    
    5. Supertrend value:
       - In uptrend: Supertrend = Final Lower Band (support)
       - In downtrend: Supertrend = Final Upper Band (resistance)
    
    **Trading Interpretation:**
    
    - **Buy Signal**: Direction changes from -1 to 1 (price crosses above supertrend)
    - **Sell Signal**: Direction changes from 1 to -1 (price crosses below supertrend)
    - **Stop Loss**: Use supertrend value as dynamic stop loss level
    - **Trend Filter**: Only take trades in direction of supertrend
    
    **Parameter Selection:**
    
    - **Shorter period (7-10)**: More responsive, more signals, more whipsaws
    - **Longer period (14-20)**: Smoother, fewer signals, less noise
    - **Lower multiplier (2.0-2.5)**: Tighter bands, more signals
    - **Higher multiplier (3.0-4.0)**: Wider bands, fewer but stronger signals
    
    **ATR Method Comparison:**
    
    - **classic**: Simple average, equal weight to all periods
    - **wilder**: Wilder's smoothing, emphasizes recent data, industry standard
    - **ema**: Most responsive to recent volatility changes
    
    **Performance:**
    
    - Implemented with Numba jitclass for maximum speed
    - O(1) time complexity per update
    - Suitable for real-time streaming data
    - No lookback window required after initialization
    
    See Also
    --------
    ATR : Average True Range indicator
    ParabolicSAR : Alternative trend-following indicator
    """
    
    def __init__(
        self,
        period: int = 10,
        multiplier: float = 3.0,
        high_column: str = "high",
        low_column: str = "low",
        close_column: str = "close",
        method: Literal["classic", "wilder", "ema"] = "wilder",
        min_periods: Optional[int] = None,
    ):

        """
        Initialize Supertrend indicator
        
        Parameters
        ----------
        period : int, default 10
            ATR calculation period
        multiplier : float, default 3.0
            ATR multiplier for band width
        atr_method : {"classic", "wilder", "ema"}, default "wilder"
            ATR calculation method
        min_periods : int, optional
            Minimum periods before valid output (default: period)
        """
        self.period = period
        self.multiplier = multiplier
        self.high_column = high_column
        self.low_column = low_column
        self.close_column = close_column
        self.atr_method = method
        self.min_periods = min_periods if min_periods is not None else period
        
        # Map string method to integer for jitclass
        method_map = {"classic": 0, "wilder": 1, "ema": 2}
        atr_method_int = method_map.get(method, 1)  # Default to wilder
        
        # Initialize state with integer method
        self.state = SupertrendState(
            period=period,
            multiplier=multiplier,
            atr_method=atr_method_int,
            min_periods=self.min_periods
        )
    
    def update(self, high: float, low: float, close: float, **kwargs) -> Tuple[float, int]:
        """
        Update Supertrend with new candle data
        
        Parameters
        ----------
        high : float
            High price of current candle
        low : float
            Low price of current candle
        close : float
            Close price of current candle
        
        Returns
        -------
        tuple of (float, int)
            - supertrend_value : float
                Current supertrend level (NaN during warm-up)
            - direction : int
                Trend direction (1 = uptrend, -1 = downtrend)
        
        Examples
        --------
        >>> st = Supertrend(period=10, multiplier=3.0)
        >>> value, direction = st.update(110.5, 108.2, 109.8)
        >>> if not np.isnan(value):
        ...     print(f"Supertrend: {value:.2f}, Direction: {direction}")
        """
        return self.state.update(high, low, close)
    
    def reset(self):
        """
        Reset indicator state to initial values
        
        Use this when starting a new data sequence or switching symbols.
        All internal state (ATR, bands, direction) will be cleared.
        
        Examples
        --------
        >>> st = Supertrend(period=10, multiplier=3.0)
        >>> # Process first dataset
        >>> for candle in data1:
        ...     st.update(*candle)
        >>> 
        >>> # Reset for new dataset
        >>> st.reset()
        >>> for candle in data2:
        ...     st.update(*candle)
        """
        self.state.reset()


# =================================================================== 19. Aroon
class Aroon:
    """ Aroon Indicator - Identifies trend strength and potential reversals
    
    Aroon measures time elapsed since the highest high and lowest low within
    a lookback period. It helps identify trend strength, consolidation periods,
    and potential trend changes.
    
    Parameters
    ----------
    period : int, default 25
        Lookback period for finding highs/lows
    min_periods : int, optional
        Minimum periods before valid output (default: period)
    
    Returns
    -------
    tuple of (float, float, float)
        - aroon_up : float (0-100)
            Time since highest high: 100 = new high, 0 = high at start of period
        - aroon_down : float (0-100)
            Time since lowest low: 100 = new low, 0 = low at start of period
        - oscillator : float (-100 to +100)
            Aroon Up - Aroon Down: positive = uptrend, negative = downtrend
    
    Examples
    --------
    Basic usage:
    
    >>> aroon = Aroon(period=25)
    >>> for high, low in price_data:
    ...     up, down, osc = aroon.update(high, low)
    ...     if not np.isnan(up):
    ...         if osc > 50:
    ...             print(f"Strong uptrend: Oscillator = {osc:.1f}")
    ...         elif osc < -50:
    ...             print(f"Strong downtrend: Oscillator = {osc:.1f}")
    
    Trend identification:
    
    >>> aroon = Aroon(period=25)
    >>> for high, low in price_data:
    ...     up, down, osc = aroon.update(high, low)
    ...     if not np.isnan(up):
    ...         if up > 70 and down < 30:
    ...             print("Strong uptrend")
    ...         elif down > 70 and up < 30:
    ...             print("Strong downtrend")
    ...         elif up < 50 and down < 50:
    ...             print("Consolidation/No clear trend")
    
    Crossover signals:
    
    >>> aroon = Aroon(period=25)
    >>> prev_up, prev_down = 0, 0
    >>> for high, low in price_data:
    ...     up, down, osc = aroon.update(high, low)
    ...     if not np.isnan(up):
    ...         if prev_up < prev_down and up > down:
    ...             print("Bullish crossover - potential buy signal")
    ...         elif prev_up > prev_down and up < down:
    ...             print("Bearish crossover - potential sell signal")
    ...         prev_up, prev_down = up, down
    
    Faster signals with custom min_periods:
    
    >>> # Start producing signals after 10 periods instead of 25
    >>> aroon = Aroon(period=25, min_periods=10)
    >>> up, down, osc = aroon.update(high, low)
    
    Notes
    -----
    **Calculation:**
    
    - Aroon Up = ((period - bars_since_highest_high) / period) × 100
    - Aroon Down = ((period - bars_since_lowest_low) / period) × 100
    - Oscillator = Aroon Up - Aroon Down
    
    **Interpretation:**
    
    - **Aroon Up near 100**: Recent new high → strong uptrend
    - **Aroon Down near 100**: Recent new low → strong downtrend
    - **Both near 100**: High volatility, potential reversal zone
    - **Both below 50**: Consolidation, weak trend
    - **Oscillator > +50**: Strong uptrend
    - **Oscillator < -50**: Strong downtrend
    - **Oscillator near 0**: Neutral/ranging market
    
    **Trading Signals:**
    
    - **Buy**: Aroon Up crosses above Aroon Down
    - **Sell**: Aroon Down crosses above Aroon Up
    - **Trend Confirmation**: Aroon Up > 70 confirms uptrend
    - **Consolidation**: Both lines below 50 → avoid trend trades
    
    **Parameter Selection:**
    
    - **Shorter period (14-20)**: More sensitive, faster signals, more noise
    - **Standard period (25)**: Balanced, widely used
    - **Longer period (30-50)**: Smoother, fewer signals, stronger trends
    
    See Also
    --------
    ADX : Alternative trend strength indicator
    Supertrend : Trend-following indicator with dynamic stops
    """
    
    def __init__(self, period: int = 25, min_periods: Optional[int] = None, **kwargs):
        """
        Initialize Aroon indicator
        
        Parameters
        ----------
        period : int, default 25
            Lookback period for finding highs/lows
        min_periods : int, optional
            Minimum periods before valid output (default: period)
        """
        self.period = period
        self.min_periods = min_periods if min_periods is not None else -1
        
        self.state = AroonState(
            period=period, 
            min_periods=self.min_periods
        )
    
    def update(self, high: float, low: float) -> Tuple[float, float, float]:
        """
        Update Aroon with new price data
        
        Parameters
        ----------
        high : float
            High price of current candle
        low : float
            Low price of current candle
        
        Returns
        -------
        tuple of (float, float, float)
            - aroon_up : Aroon Up value (0-100, NaN during warm-up)
            - aroon_down : Aroon Down value (0-100, NaN during warm-up)
            - oscillator : Aroon Oscillator (-100 to +100, NaN during warm-up)
        
        Examples
        --------
        >>> aroon = Aroon(period=25)
        >>> up, down, osc = aroon.update(110.5, 108.2)
        >>> if not np.isnan(up):
        ...     print(f"Up: {up:.1f}, Down: {down:.1f}, Osc: {osc:.1f}")
        """
        return self.state.update(high, low)
    
    def reset(self):
        """
        Reset indicator state to initial values
        
        Clears all internal buffers and counters. Use when starting
        a new data sequence or switching symbols.
        
        Examples
        --------
        >>> aroon = Aroon(period=25)
        >>> # Process first dataset
        >>> for high, low in data1:
        ...     aroon.update(high, low)
        >>> 
        >>> # Reset for new dataset
        >>> aroon.reset()
        >>> for high, low in data2:
        ...     aroon.update(high, low)
        """
        self.state.reset()


# =================================================================== 20. DEMA
class DEMA:
    """
    Double Exponential Moving Average (DEMA) - Reduced lag trend indicator
    
    DEMA applies exponential smoothing twice to reduce lag compared to standard EMA.
    
    Parameters
    ----------
    period : int, default 20
        EMA period for both smoothing passes
    min_periods : int, optional
        Minimum periods before valid output (default: period)
    
    Returns
    -------
    float
        DEMA value (NaN during warm-up)
    
    Notes
    -----
    Formula: DEMA = 2 × EMA(price) - EMA(EMA(price))
    
    Interpretation:
    - Price > DEMA: Bullish trend
    - Price < DEMA: Bearish trend
    - Crossovers signal potential trend changes
    
    Advantages: ~40% less lag than standard EMA
    Limitations: More sensitive to whipsaws in ranging markets
    
    Examples
    --------
    >>> dema = DEMA(period=20)
    >>> value = dema.update(105.50)
    >>> if not np.isnan(value):
    ...     print(f"DEMA: {value:.2f}")
    """
    
    def __init__(self, period: int = 20, min_periods: Optional[int] = None, **kwargs):
        """
        Initialize DEMA indicator
        
        Parameters
        ----------
        period : int, default 20
            EMA period for both smoothing passes
        min_periods : int, optional
            Minimum periods before valid output (default: period)
        """
        self.period = period
        self.min_periods = min_periods if min_periods is not None else -1
        
        self.state = DEMAState(
            n=period,
            min_periods=self.min_periods
        )
    
    def update(self, value: float) -> float:
        """
        Update DEMA with new price value
        
        Parameters
        ----------
        value : float
            New price value (typically close price)
        
        Returns
        -------
        float
            DEMA value (NaN during warm-up period)
        """
        return self.state.update(value)
    
    def reset(self):
        """
        Reset indicator state to initial values
        """
        self.state.reset()


# =================================================================== 21. TEMA
class TEMA:
    """
    Triple Exponential Moving Average (TEMA) - Ultra-low lag trend indicator
    
    TEMA applies exponential smoothing three times with a special formula to minimize lag
    while maintaining smoothness.
    
    Parameters
    ----------
    period : int, default 20
        EMA period for all three smoothing passes
    min_periods : int, optional
        Minimum periods before valid output (default: period)
    
    Returns
    -------
    float
        TEMA value (NaN during warm-up)
    
    Notes
    -----
    Formula: TEMA = 3×EMA(price) - 3×EMA(EMA(price)) + EMA(EMA(EMA(price)))
    
    Interpretation:
    - Price > TEMA: Strong bullish trend
    - Price < TEMA: Strong bearish trend
    - Crossovers signal potential trend reversals
    
    Advantages:
    - ~60% less lag than standard EMA
    - Faster response to price changes
    - Smoother than simple price differencing
    
    Limitations:
    - More sensitive to noise and false signals
    - Requires more data for initialization
    - Can overshoot in volatile markets
    
    Examples
    --------
    >>> tema = TEMA(period=20)
    >>> value = tema.update(105.50)
    >>> if not np.isnan(value):
    ...     print(f"TEMA: {value:.2f}")
    
    >>> # Custom min_periods for earlier signals
    >>> tema_fast = TEMA(period=20, min_periods=10)
    >>> tema_fast.reset()  # Start fresh sequence
    """
    
    def __init__(self, period: int = 20, min_periods: Optional[int] = None, **kwargs):
        """
        Initialize TEMA indicator
        
        Parameters
        ----------
        period : int, default 20
            EMA period for all three smoothing passes
        min_periods : int, optional
            Minimum periods before valid output (default: period)
        """
        self.period = period
        self.min_periods = min_periods if min_periods is not None else -1
        
        self.state = TEMAState(
            n=period,
            min_periods=self.min_periods
        )
    
    def update(self, value: float) -> float:
        """
        Update TEMA with new price value
        
        Parameters
        ----------
        value : float
            New price value (typically close price)
        
        Returns
        -------
        float
            TEMA value (NaN during warm-up period)
        """
        return self.state.update(value)
    
    def reset(self):
        """
        Reset indicator state to initial values
        """
        self.state.reset()


# =================================================================== 22. KAMA
class KAMA:
    """
    Kaufman's Adaptive Moving Average (KAMA) - Volatility-adaptive trend indicator
    
    KAMA automatically adjusts its smoothing based on market efficiency, becoming faster
    in trending markets and slower in ranging markets.
    
    Parameters
    ----------
    period : int, default 10
        Period for calculating Efficiency Ratio
    fast_span : int, default 2
        Fast EMA span for trending markets
    slow_span : int, default 30
        Slow EMA span for ranging markets
    min_periods : int, optional
        Minimum periods before valid output (default: period)
    
    Returns
    -------
    float
        KAMA value (NaN during warm-up)
    
    Notes
    -----
    Formula:
    - ER = |Change| / Volatility
    - SC = [ER × (fastest - slowest) + slowest]²
    - KAMA = KAMA_prev + SC × (Price - KAMA_prev)
    
    Where:
    - Change = |price_now - price_n_periods_ago|
    - Volatility = sum of |price[i] - price[i-1]|
    - fastest = 2/(fast_span + 1)
    - slowest = 2/(slow_span + 1)
    
    Interpretation:
    - Price > KAMA: Bullish trend
    - Price < KAMA: Bearish trend
    - KAMA slope indicates trend strength
    - Flat KAMA suggests ranging market
    
    Advantages:
    - Adapts to market conditions automatically
    - Reduces whipsaws in sideways markets
    - Responsive during strong trends
    - Self-adjusting smoothing constant
    
    Limitations:
    - Requires sufficient history (period + 1 bars minimum)
    - Complex calculation vs simple moving averages
    - May lag at trend inception
    
    Examples
    --------
    >>> kama = KAMA(period=10, fast_span=2, slow_span=30)
    >>> value = kama.update(105.50)
    >>> if not np.isnan(value):
    ...     print(f"KAMA: {value:.2f}")
    
    >>> # More responsive settings
    >>> kama_fast = KAMA(period=5, fast_span=2, slow_span=20, min_periods=5)
    >>> kama_fast.reset()  # Start fresh sequence
    """
    
    def __init__(
        self, 
        period: int = 10, 
        fast_span: int = 2, 
        slow_span: int = 30, 
        min_periods: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize KAMA indicator
        
        Parameters
        ----------
        period : int, default 10
            Period for calculating Efficiency Ratio
        fast_span : int, default 2
            Fast EMA span for trending markets
        slow_span : int, default 30
            Slow EMA span for ranging markets
        min_periods : int, optional
            Minimum periods before valid output (default: period)
        """
        self.period = period
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.min_periods = min_periods if min_periods is not None else -1
        
        self.state = KAMAState(
            n=period,
            fast_span=fast_span,
            slow_span=slow_span,
            min_periods=self.min_periods
        )
    
    def update(self, value: float) -> float:
        """
        Update KAMA with new price value
        
        Parameters
        ----------
        value : float
            New price value (typically close price)
        
        Returns
        -------
        float
            KAMA value (NaN during warm-up period)
        """
        return self.state.update(value)
    
    def reset(self):
        """
        Reset indicator state to initial values
        """
        self.state.reset()


# =================================================================== 23. HMA
class HMA:
    """
    Hull Moving Average (HMA) - Ultra-responsive trend indicator with minimal lag
    
    HMA uses weighted moving averages in a nested formula to achieve both smoothness
    and responsiveness, significantly reducing lag compared to traditional moving averages.
    
    Parameters
    ----------
    period : int, default 20
        Base period for HMA calculation
    min_periods : int, optional
        Minimum periods before valid output (default: period)
    
    Returns
    -------
    float
        HMA value (NaN during warm-up)
    
    Notes
    -----
    Formula: HMA = WMA(√n) of [2×WMA(n/2) - WMA(n)]
    
    Calculation steps:
    1. Calculate WMA with half period: WMA(n/2)
    2. Calculate WMA with full period: WMA(n)
    3. Compute raw HMA: 2×WMA(n/2) - WMA(n)
    4. Apply final smoothing: WMA(√n) of raw HMA
    
    Interpretation:
    - Price > HMA: Bullish trend
    - Price < HMA: Bearish trend
    - HMA slope: Trend strength and direction
    - Crossovers: Potential trend reversals
    
    Advantages:
    - Extremely low lag (~70% less than SMA)
    - Smooth curve without excessive noise
    - Fast response to price changes
    - Excellent for trend following
    
    Limitations:
    - More complex calculation
    - Can produce false signals in choppy markets
    - Requires more data for initialization
    
    Examples
    --------
    >>> hma = HMA(period=20)
    >>> value = hma.update(105.50)
    >>> if not np.isnan(value):
    ...     print(f"HMA: {value:.2f}")
    
    >>> # Faster signals with lower min_periods
    >>> hma_fast = HMA(period=20, min_periods=10)
    >>> hma_fast.reset()  # Start fresh
    """
    
    def __init__(self, period: int = 20, min_periods: Optional[int] = None, **kwargs):
        """
        Initialize HMA indicator
        
        Parameters
        ----------
        period : int, default 20
            Base period for HMA calculation
        min_periods : int, optional
            Minimum periods before valid output (default: period)
        """
        self.period = period
        self.min_periods = min_periods if min_periods is not None else -1
        
        self.state = HMAState(
            n=period,
            min_periods=self.min_periods
        )
    
    def update(self, value: float) -> float:
        """
        Update HMA with new price value
        
        Parameters
        ----------
        value : float
            New price value (typically close price)
        
        Returns
        -------
        float
            HMA value (NaN during warm-up period)
        """
        return self.state.update(value)
    
    def reset(self):
        """
        Reset indicator state to initial values
        """
        self.state.reset()


# =================================================================== 
