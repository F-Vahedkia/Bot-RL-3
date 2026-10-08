# f04_features/indicators_new/feature_C_registry_1.py
# Run: python -m f04_features.feature_C_registry_1
""" Last reviewed:
    1405/04/31
    1405/05/18

Feature Registry - Single Source of Truth
Unified registry for all features/indicators across train, backtest, and live modes.

# در engine:
    spec = get_indicator("sma", mode="live")  # → SMA class
    spec = get_indicator("sma", mode="train")  # → sma_batch_df function

# لیست indicators:
    live_indicators = list_indicators(mode="live")
"""

# =============================================================================
# Imports
# =============================================================================
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union

# Layer 2 imports (batch functions for train/backtest)
from f04_features.indicators_new.indicators_B_batch import (
    sma_batch_df, wma_batch_df, ema_batch_df, roc_batch_df, rsi_batch_df, truerange_batch_df,
    atr_batch_df, macd_batch_df, bollinger_batch_df, keltner_batch_df,  stochastic_batch_df,
    cci_batch_df, mfi_batch_df, obv_batch_df, williamsr_batch_df, parabolicsar_batch_df,
    heikinashi_batch_df, supertrend_batch_df, aroon_batch_df, dema_batch_df, tema_batch_df,
    kama_batch_df, hma_batch_df,
)
# Layer 3 imports (stateful classes for live/backtest)
from f04_features.indicators_new.indicators_B_class import (
    SMA, WMA, EMA, ROC, RSI, TrueRange, ATR, MACD, BollingerBands,
    KeltnerChannel, Stochastic, CCI, MFI, OBV, WilliamsR, ParabolicSAR,
    HeikinAshi, Supertrend, Aroon, DEMA, TEMA, KAMA, HMA
)
# =============================================================================
# Modes
# =============================================================================
TRAIN_MODES = {"train", "optimize"}
INCREMENTAL_MODES = {"live", "paper", "shadow", "backtest", "replay", "eval"}
BATCH_MODES = TRAIN_MODES
ALL_MODES = TRAIN_MODES | INCREMENTAL_MODES

# =============================================================================
# Class_1
# =============================================================================
_NO_DEFAULT = object()

@dataclass(frozen=True)
class ParameterSpec:
    """
    Definition of one user-visible parameter.
    """
    name: str
    dtype: type | tuple[type, ...] = object
    default: Any = _NO_DEFAULT
    required: bool = False
    aliases: tuple[str, ...] = ()
    choices: Optional[Sequence[Any]] = None
    minimum: Any = None
    maximum: Any = None
    description: str = ""
    visible: bool = True

# =============================================================================
# Class_2
# =============================================================================
@dataclass
class IndicatorSpec:
    # ---------------- Identity ----------------
    name: str
    fn: Callable
    is_stateful: bool
    # ---------------- Runtime ----------------
    modes: Set[str] = field(default_factory=lambda: ALL_MODES.copy())
    required_cols: list[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    needs_tf_map: bool = False
    output_names: Optional[List[str]] = None
    description: str = ""
    # ---------------- Parser Metadata ----------------
    parameters: list[ParameterSpec] = field(default_factory=list)
    accepts_var_args: bool = False
    accepts_var_kwargs: bool = False
    allow_unknown_kwargs: bool = False
    # ---------------- Validation ----------------
    strict: bool = True
    version: str = "1"
    deprecated_aliases: dict[str, str] = field(default_factory=dict)
    # ------------------------------------------------
    warmup: Optional[int] = None
    # ---------------- Methods -----------------------
    def supports(self, mode: str) -> bool:
        return mode in self.modes
    
    def is_batch_mode(self, mode: str) -> bool:
        """آیا این mode نیاز به batch function دارد؟"""
        return mode in BATCH_MODES
    
    def is_incremental_mode(self, mode: str) -> bool:
        """آیا این mode نیاز به stateful class دارد؟"""
        return mode in INCREMENTAL_MODES

    @property
    def parameter_names(self):
        return [p.name for p in self.parameters]

    def get_parameter(self, name):
        for p in self.parameters:
            if p.name == name:
                return p
            if name in p.aliases:
                return p
        return None

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

# =============================================================================
# Shared Parameter Definitions
# =============================================================================
SMA_PARAMETERS = [
    ParameterSpec(
        name="column",
        aliases=("close_col", "price_col", "source", "src", "field", "price", "value"),
        dtype=str,
        required=True,
        choices=("open", "high", "low", "close"),
        description="Source price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        required=True,
        minimum=1,
        description="Moving average period",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
WMA_PARAMETERS = [
    ParameterSpec(
        name="column",
        aliases=("close_col", "price_col", "source", "src", "field", "price", "value"),
        dtype=str,
        required=True,
        choices=("open", "high", "low", "close"),
        description="Source price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        required=True,
        minimum=1,
        description="Moving average period",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
EMA_PARAMETERS = [
    ParameterSpec(
        name="column",
        aliases=("close_col", "price_col", "source", "src", "field", "price", "value"),
        dtype=str,
        required=True,
        choices=("open", "high", "low", "close"),
        description="Source price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        required=True,
        minimum=1,
        description="Moving average period",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
ROC_PARAMETERS = [
    ParameterSpec(
        name="column",
        aliases=("close_col", "price_col", "source", "src", "field", "price", "value"),
        dtype=str,
        required=True,
        choices=("open", "high", "low", "close"),
        description="Source price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        required=True,
        minimum=1,
        description="Rate of Change period",
    ),
]
RSI_PARAMETERS = [
    ParameterSpec(
        name="column",
        aliases=("close_col", "price_col", "source", "src", "field", "price", "value"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close"),
        description="Source price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=14,
        minimum=1,
        description="RSI period",
    ),
    ParameterSpec(
        name="method",
        aliases=("ma_method",),
        dtype=str,
        default="ema",
        choices=("ema", "sma"),
        description="Smoothing method",
    ),
]
TRUE_RANGE_PARAMETERS = [
    ParameterSpec(
        name="high_column",
        aliases=("high_col", "high"),
        dtype=str,
        default="high",
        choices=("high",),
        description="High price column",
    ),
    ParameterSpec(
        name="low_column",
        aliases=("low_col", "low"),
        dtype=str,
        default="low",
        choices=("low",),
        description="Low price column",
    ),
    ParameterSpec(
        name="close_column",
        aliases=("close_col", "close"),
        dtype=str,
        default="close",
        choices=("close",),
        description="Close price column",
    ),
]
ATR_PARAMETERS = [
    ParameterSpec(
        name="high_column",
        aliases=("high_col", "high"),
        dtype=str,
        default="high",
        choices=("open", "high", "low", "close", "volume"),
        description="High price column",
    ),
    ParameterSpec(
        name="low_column",
        aliases=("low_col", "low"),
        dtype=str,
        default="low",
        choices=("open", "high", "low", "close", "volume"),
        description="Low price column",
    ),
    ParameterSpec(
        name="close_column",
        aliases=("close_col", "close"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Close price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=14,
        minimum=1,
        description="ATR period",
    ),
    ParameterSpec(
        name="method",
        aliases=("atr_method",),
        dtype=str,
        default="wilder",
        choices=("wilder", "sma", "ema"),
        description="ATR smoothing method",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
MACD_PARAMETERS = [
    ParameterSpec(
        name="column",
        aliases=("close_col", "price_col", "source", "src", "field", "price", "value"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Close price column",
    ),
    ParameterSpec(
        name="fast",
        aliases=("fast_period", "fast_length", "fast_window", "fast_n"),
        dtype=int,
        default=12,
        minimum=1,
        description="Fast EMA period",
    ),
    ParameterSpec(
        name="slow",
        aliases=("slow_period", "slow_length", "slow_window", "slow_n"),
        dtype=int,
        default=26,
        minimum=1,
        description="Slow EMA period",
    ),
    ParameterSpec(
        name="signal",
        aliases=("signal_period", "signal_length", "signal_window", "signal_n"),
        dtype=int,
        default=9,
        minimum=1,
        description="Signal EMA period",
    ),
]
BOLLINGER_BANDS_PARAMETERS = [
    ParameterSpec(
        name="column",
        aliases=("close_col", "price_col", "source", "src", "field", "price", "value"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Close price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=20,
        minimum=1,
        description="Moving average period",
    ),
    ParameterSpec(
        name="multiplier",
        aliases=("m", "k", "std", "std_dev"),
        dtype=(int, float),
        default=2.0,
        minimum=0.0,
        description="Standard deviation multiplier",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
KELTNER_CHANNEL_PARAMETERS = [
    ParameterSpec(
        name="high_column",
        aliases=("high_col", "high"),
        dtype=str,
        default="high",
        choices=("open", "high", "low", "close", "volume"),
        description="High price column",
    ),
    ParameterSpec(
        name="low_column",
        aliases=("low_col", "low"),
        dtype=str,
        default="low",
        choices=("open", "high", "low", "close", "volume"),
        description="Low price column",
    ),
    ParameterSpec(
        name="close_column",
        aliases=("close_col", "close"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Close price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=20,
        minimum=1,
        description="Channel period",
    ),
    ParameterSpec(
        name="multiplier",
        aliases=("m",),
        dtype=(float, int),
        default=2.0,
        minimum=0.0,
        description="ATR multiplier",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
STOCHASTIC_PARAMETERS = [
    ParameterSpec(
        name="high_column",
        aliases=("high_col", "high"),
        dtype=str,
        default="high",
        choices=("open", "high", "low", "close", "volume"),
        description="High price column",
    ),
    ParameterSpec(
        name="low_column",
        aliases=("low_col", "low"),
        dtype=str,
        default="low",
        choices=("open", "high", "low", "close", "volume"),
        description="Low price column",
    ),
    ParameterSpec(
        name="close_column",
        aliases=("close_col", "close"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Close price column",
    ),
    ParameterSpec(
        name="k_period",
        aliases=("k", "k_length", "k_window"),
        dtype=int,
        default=14,
        minimum=1,
        description="%K period",
    ),
    ParameterSpec(
        name="d_period",
        aliases=("d", "d_length", "d_window"),
        dtype=int,
        default=3,
        minimum=1,
        description="%D period",
    ),
    ParameterSpec(
        name="smooth_k",
        aliases=("slowing", "smooth"),
        dtype=int,
        default=3,
        minimum=1,
        description="%K smoothing period",
    ),
    ParameterSpec(
        name="method",
        aliases=("ma_method",),
        dtype=str,
        default="sma",
        choices=("sma", "ema"),
        description="Smoothing method",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
CCI_PARAMETERS = [
    ParameterSpec(
        name="high_column",
        aliases=("high_col", "high"),
        dtype=str,
        default="high",
        choices=("open", "high", "low", "close", "volume"),
        description="High price column",
    ),
    ParameterSpec(
        name="low_column",
        aliases=("low_col", "low"),
        dtype=str,
        default="low",
        choices=("open", "high", "low", "close", "volume"),
        description="Low price column",
    ),
    ParameterSpec(
        name="close_column",
        aliases=("close_col", "close"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Close price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=20,
        minimum=1,
        description="CCI period",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
MFI_PARAMETERS = [
    ParameterSpec(
        name="high_column",
        aliases=("high_col", "high"),
        dtype=str,
        default="high",
        choices=("open", "high", "low", "close", "volume"),
        description="High price column",
    ),
    ParameterSpec(
        name="low_column",
        aliases=("low_col", "low"),
        dtype=str,
        default="low",
        choices=("open", "high", "low", "close", "volume"),
        description="Low price column",
    ),
    ParameterSpec(
        name="close_column",
        aliases=("close_col", "close"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Close price column",
    ),
    ParameterSpec(
        name="volume_column",
        aliases=("volume_col", "volume"),
        dtype=str,
        default="volume",
        choices=("open", "high", "low", "close", "volume"),
        description="Volume column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=14,
        minimum=1,
        description="MFI period",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
OBV_PARAMETERS = [
    ParameterSpec(
        name="close_column",
        aliases=("close_col", "close"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Close price column",
    ),
    ParameterSpec(
        name="volume_column",
        aliases=("volume_col", "volume"),
        dtype=str,
        default="volume",
        choices=("open", "high", "low", "close", "volume"),
        description="Volume column",
    ),
]
WILLIAMS_R_PARAMETERS = [
    ParameterSpec(
        name="high_column",
        aliases=("high_col", "high"),
        dtype=str,
        default="high",
        choices=("open", "high", "low", "close", "volume"),
        description="High price column",
    ),
    ParameterSpec(
        name="low_column",
        aliases=("low_col", "low"),
        dtype=str,
        default="low",
        choices=("open", "high", "low", "close", "volume"),
        description="Low price column",
    ),
    ParameterSpec(
        name="close_column",
        aliases=("close_col", "close"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Close price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=14,
        minimum=1,
        description="Williams %R period",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp"),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
PARABOLIC_SAR_PARAMETERS = [
    ParameterSpec(
        name="high_column",
        aliases=("high_col", "high"),
        dtype=str,
        default="high",
        choices=("open", "high", "low", "close", "volume"),
        description="High price column",
    ),
    ParameterSpec(
        name="low_column",
        aliases=("low_col", "low"),
        dtype=str,
        default="low",
        choices=("open", "high", "low", "close", "volume"),
        description="Low price column",
    ),
    ParameterSpec(
        name="af_start",  # acceleration factor _start
        aliases=("af_start"),
        dtype=(int, float),
        default=0.02,
        minimum=0.0,
        maximum=1.0,
        description="Initial acceleration factor",
    ),
    ParameterSpec(
        name="af_step",  # acceleration factor _step
        aliases=("af_step", "step"),
        dtype=(int, float),
        default=0.02,
        minimum=0.0,
        maximum=1.0,
        description="Acceleration factor increment",
    ),
    ParameterSpec(
        name="af_max",  # acceleration factor _max
        aliases=("af_max", "max_step"),
        dtype=(int, float),
        default=0.2,
        minimum=0.0,
        maximum=1.0,
        description="Maximum acceleration factor",
    ),
]
HEIKIN_ASHI_PARAMETERS = [
    ParameterSpec(
        name="open_column",
        aliases=("open_col", "open"),
        dtype=str,
        default="open",
        choices=("open", "high", "low", "close", "volume"),
        description="Open price column",
    ),
    ParameterSpec(
        name="high_column",
        aliases=("high_col", "high"),
        dtype=str,
        default="high",
        choices=("open", "high", "low", "close", "volume"),
        description="High price column",
    ),
    ParameterSpec(
        name="low_column",
        aliases=("low_col", "low"),
        dtype=str,
        default="low",
        choices=("open", "high", "low", "close", "volume"),
        description="Low price column",
    ),
    ParameterSpec(
        name="close_column",
        aliases=("close_col", "close"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Close price column",
    ),
    ParameterSpec(
        name="result_prefix",
        dtype=str,
        default="ha",
        description="Prefix for Heikin Ashi output columns",
    ),
]
SUPERTREND_PARAMETERS = [
    ParameterSpec(
        name="high_column",
        aliases=("high_col", "high"),
        dtype=str,
        default="high",
        choices=("open", "high", "low", "close", "volume"),
        description="High price column",
    ),
    ParameterSpec(
        name="low_column",
        aliases=("low_col", "low"),
        dtype=str,
        default="low",
        choices=("open", "high", "low", "close", "volume"),
        description="Low price column",
    ),
    ParameterSpec(
        name="close_column",
        aliases=("close_col", "close"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Close price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=10,
        minimum=1,
        description="ATR period",
    ),
    ParameterSpec(
        name="multiplier",
        aliases=("m", "factor"),
        dtype=(int, float),
        default=3.0,
        minimum=0.0,
        description="ATR multiplier",
    ),
    ParameterSpec(
        name="method",
        aliases=("atr_method",),
        dtype=str,
        default="wilder",
        choices=("wilder", "ema", "sma"),
        description="ATR smoothing method",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
AROON_PARAMETERS = [
    ParameterSpec(
        name="high_column",
        aliases=("high_col", "high"),
        dtype=str,
        default="high",
        choices=("open", "high", "low", "close", "volume"),
        description="High price column",
    ),
    ParameterSpec(
        name="low_column",
        aliases=("low_col", "low"),
        dtype=str,
        default="low",
        choices=("open", "high", "low", "close", "volume"),
        description="Low price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=25,
        minimum=1,
        description="Aroon period",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
DEMA_PARAMETERS = [
    ParameterSpec(
        name="column",
        aliases=("close_col", "price_col", "source", "src", "field", "price", "value"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Source price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=20,
        minimum=1,
        description="DEMA period",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
TEMA_PARAMETERS = [
    ParameterSpec(
        name="column",
        aliases=("close_col", "price_col", "source", "src", "field", "price", "value"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Source price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=20,
        minimum=1,
        description="TEMA period",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
KAMA_PARAMETERS = [
    ParameterSpec(
        name="column",
        aliases=("close_col", "price_col", "source", "src", "field", "price", "value"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Source price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=10,
        minimum=1,
        description="KAMA efficiency ratio period",
    ),
    ParameterSpec(
        name="fast_span",
        aliases=("fast",),
        dtype=int,
        default=2,
        minimum=1,
        description="Fast EMA span",
    ),
    ParameterSpec(
        name="slow_span",
        aliases=("slow",),
        dtype=int,
        default=30,
        minimum=2,
        description="Slow EMA span",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]
HMA_PARAMETERS = [
    ParameterSpec(
        name="column",
        aliases=("close_col", "price_col", "source", "src", "field", "price", "value"),
        dtype=str,
        default="close",
        choices=("open", "high", "low", "close", "volume"),
        description="Source price column",
    ),
    ParameterSpec(
        name="period",
        aliases=("n", "window", "length"),
        dtype=int,
        default=20,
        minimum=1,
        description="Hull Moving Average period",
    ),
    ParameterSpec(
        name="min_periods",
        aliases=("minp",),
        dtype=int,
        default=-1,
        description="Minimum number of observations",
    ),
]

# ============================================================================
# Layer 2: Batch Functions
# Used for: Train / Optimize
# ============================================================================
_BATCH_INDICATORS = {
    "sma": IndicatorSpec(
        name="sma",
        fn=sma_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        parameters=SMA_PARAMETERS,
        description="Simple Moving Average (batch)"
    ),
    "wma": IndicatorSpec(
        name="wma",
        fn=wma_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        parameters=WMA_PARAMETERS,
        description="Weighted Moving Average (batch)"
    ),
    "ema": IndicatorSpec(
        name="ema",
        fn=ema_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        parameters=EMA_PARAMETERS,
        description="Exponential Moving Average (batch)"
    ),
    "roc": IndicatorSpec(
        name="roc",
        fn=roc_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        parameters=ROC_PARAMETERS,
        description="Rate of Change (batch)"
    ),
    "rsi": IndicatorSpec(
        name="rsi",
        fn=rsi_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        parameters=RSI_PARAMETERS,
        description="Relative Strength Index (batch)"
    ),
    "true_range": IndicatorSpec(
        name="true_range",
        fn=truerange_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        parameters=TRUE_RANGE_PARAMETERS,
        description="True Range (batch)"
    ),
    "atr": IndicatorSpec(
        name="atr",
        fn=atr_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        parameters=ATR_PARAMETERS,
        description="Average True Range (batch)"
    ),
    "macd": IndicatorSpec(
        name="macd",
        fn=macd_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        output_names=["macd", "macd_signal", "macd_hist"],
        parameters=MACD_PARAMETERS,
        description="MACD (batch)"
    ),
    "bollinger_bands": IndicatorSpec(
        name="bollinger_bands",
        fn=bollinger_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        output_names=["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent"],
        parameters=BOLLINGER_BANDS_PARAMETERS,
        description="Bollinger Bands (batch)"
    ),
    "keltner_channel": IndicatorSpec(
        name="keltner_channel",
        fn=keltner_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        output_names=["kc_upper", "kc_middle", "kc_lower", "kc_width", "kc_percent"],
        parameters=KELTNER_CHANNEL_PARAMETERS,
        description="Keltner Channel (batch)"
    ),
    "stochastic": IndicatorSpec(
        name="stochastic",
        fn=stochastic_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        output_names=["stoch_k", "stoch_d"],
        parameters=STOCHASTIC_PARAMETERS,
        description="Stochastic Oscillator (batch)"
    ),
    "cci": IndicatorSpec(
        name="cci",
        fn=cci_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        parameters=CCI_PARAMETERS,
        description="Commodity Channel Index (batch)"
    ),
    "mfi": IndicatorSpec(
        name="mfi",
        fn=mfi_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close", "volume"],
        parameters=MFI_PARAMETERS,
        description="Money Flow Index (batch)"
    ),
    "obv": IndicatorSpec(
        name="obv",
        fn=obv_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close", "volume"],
        parameters=OBV_PARAMETERS,
        description="On-Balance Volume (batch)"
    ),
    "williams_r": IndicatorSpec(
        name="williams_r",
        fn=williamsr_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        parameters=WILLIAMS_R_PARAMETERS,
        description="Williams %R (batch)"
    ),
    "parabolic_sar": IndicatorSpec(
        name="parabolic_sar",
        fn=parabolicsar_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low"],
        parameters=PARABOLIC_SAR_PARAMETERS,
        description="Parabolic SAR (batch)"
    ),
    "heikin_ashi": IndicatorSpec(
        name="heikin_ashi",
        fn=heikinashi_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["open", "high", "low", "close"],
        output_names=["ha_open", "ha_high", "ha_low", "ha_close"],
        parameters=HEIKIN_ASHI_PARAMETERS,
        description="Heikin Ashi (batch)"
    ),
    "supertrend": IndicatorSpec(
        name="supertrend",
        fn=supertrend_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        output_names=["supertrend", "st_direction"],
        parameters=SUPERTREND_PARAMETERS,
        description="Supertrend (batch)"
    ),
    "aroon": IndicatorSpec(
        name="aroon",
        fn=aroon_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low"],
        output_names=["aroon_up", "aroon_down", "aroon_oscillator"],
        parameters=AROON_PARAMETERS,
        description="Aroon Oscillator (batch)"
    ),
    "dema": IndicatorSpec(
        name="dema",
        fn=dema_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        parameters=DEMA_PARAMETERS,
        description="Double Exponential Moving Average (batch)"
    ),
    "tema": IndicatorSpec(
        name="tema",
        fn=tema_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        parameters=TEMA_PARAMETERS,
        description="Triple Exponential Moving Average (batch)"
    ),
    "kama": IndicatorSpec(
        name="kama",
        fn=kama_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        parameters=KAMA_PARAMETERS,
        description="Kaufman Adaptive Moving Average (batch)"
    ),
    "hma": IndicatorSpec(
        name="hma",
        fn=hma_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        parameters=HMA_PARAMETERS,
        description="Hull Moving Average (batch)"
    ),
}

# ============================================================================
# Layer 3: Stateful Incremental Indicators
# Used for: live / paper / shadow / backtest / replay / eval
# ============================================================================
_LIVE_INDICATORS = {
    "sma": IndicatorSpec(
        name="sma",
        fn=SMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        parameters=SMA_PARAMETERS,
        description="Simple Moving Average (stateful)"
    ),
    "wma": IndicatorSpec(
        name="wma",
        fn=WMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        parameters=WMA_PARAMETERS,
        description="Weighted Moving Average (stateful)"
    ),
    "ema": IndicatorSpec(
        name="ema",
        fn=EMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        parameters=EMA_PARAMETERS,
        description="Exponential Moving Average (stateful)"
    ),
    "roc": IndicatorSpec(
        name="roc",
        fn=ROC,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        parameters=ROC_PARAMETERS,
        description="Rate of Change (stateful)"
    ),
    "rsi": IndicatorSpec(
        name="rsi",
        fn=RSI,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        parameters=RSI_PARAMETERS,
        description="Relative Strength Index (stateful)"
    ),
    "true_range": IndicatorSpec(
        name="true_range",
        fn=TrueRange,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        parameters=TRUE_RANGE_PARAMETERS,
        description="True Range (stateful)"
    ),
    "atr": IndicatorSpec(
        name="atr",
        fn=ATR,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        parameters=ATR_PARAMETERS,
        description="Average True Range (stateful)"
    ),
    "macd": IndicatorSpec(
        name="macd",
        fn=MACD,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        output_names=["macd", "macd_signal", "macd_hist"],
        parameters=MACD_PARAMETERS,
        description="MACD (stateful)"
    ),
    "bollinger_bands": IndicatorSpec(
        name="bollinger_bands",
        fn=BollingerBands,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        output_names=["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent"],
        parameters=BOLLINGER_BANDS_PARAMETERS,
        description="Bollinger Bands (stateful)"
    ),
    "keltner_channel": IndicatorSpec(
        name="keltner_channel",
        fn=KeltnerChannel,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        output_names=["kc_upper", "kc_middle", "kc_lower", "kc_width", "kc_percent"],
        parameters=KELTNER_CHANNEL_PARAMETERS,
        description="Keltner Channel (stateful)"
    ),
    "stochastic": IndicatorSpec(
        name="stochastic",
        fn=Stochastic,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        output_names=["stoch_k", "stoch_d"],
        parameters=STOCHASTIC_PARAMETERS,
        description="Stochastic Oscillator (stateful)"
    ),
    "cci": IndicatorSpec(
        name="cci",
        fn=CCI,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        parameters=CCI_PARAMETERS,
        description="Commodity Channel Index (stateful)"
    ),
    "mfi": IndicatorSpec(
        name="mfi",
        fn=MFI,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close", "volume"],
        parameters=MFI_PARAMETERS,
        description="Money Flow Index (stateful)"
    ),
    "obv": IndicatorSpec(
        name="obv",
        fn=OBV,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close", "volume"],
        parameters=OBV_PARAMETERS,
        description="On-Balance Volume (stateful)"
    ),
    "williams_r": IndicatorSpec(
        name="williams_r",
        fn=WilliamsR,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        parameters=WILLIAMS_R_PARAMETERS,
        description="Williams %R (stateful)"
    ),
    "parabolic_sar": IndicatorSpec(
        name="parabolic_sar",
        fn=ParabolicSAR,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low"],
        parameters=PARABOLIC_SAR_PARAMETERS,
        description="Parabolic SAR (stateful)"
    ),
    "heikin_ashi": IndicatorSpec(
        name="heikin_ashi",
        fn=HeikinAshi,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["open", "high", "low", "close"],
        output_names=["ha_open", "ha_high", "ha_low", "ha_close"],
        parameters=HEIKIN_ASHI_PARAMETERS,
        description="Heikin Ashi (stateful)"
    ),
    "supertrend": IndicatorSpec(
        name="supertrend",
        fn=Supertrend,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        output_names=["supertrend", "st_direction"],
        parameters=SUPERTREND_PARAMETERS,
        description="Supertrend (stateful)"
    ),
    "aroon": IndicatorSpec(
        name="aroon",
        fn=Aroon,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low"],
        output_names=["aroon_up", "aroon_down", "aroon_oscillator"],
        parameters=AROON_PARAMETERS,
        description="Aroon Oscillator (stateful)"
    ),
    "dema": IndicatorSpec(
        name="dema",
        fn=DEMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        parameters=DEMA_PARAMETERS,
        description="Double Exponential Moving Average (stateful)"
    ),
    "tema": IndicatorSpec(
        name="tema",
        fn=TEMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        parameters=TEMA_PARAMETERS,
        description="Triple Exponential Moving Average (stateful)"
    ),
    "kama": IndicatorSpec(
        name="kama",
        fn=KAMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        parameters=KAMA_PARAMETERS,
        description="Kaufman Adaptive Moving Average (stateful)"
    ),
    "hma": IndicatorSpec(
        name="hma",
        fn=HMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        parameters=HMA_PARAMETERS,
        description="Hull Moving Average (stateful)"
    ),
}

# =============================================================================
# Validate Registry
# ============================================================================= func_1 خوانده شد و فهمیده شد
def validate_registry() -> None:
    """
    این تابع عدم تقارن Batch و Live را فقط برای اندیکاتورهای مشترک در هر دو دیکشنری بررسی می‌کند.
    """
    shared = set(_BATCH_INDICATORS) & set(_LIVE_INDICATORS)
    for name in shared:
        batch_spec = _BATCH_INDICATORS[name]
        live_spec  = _LIVE_INDICATORS[name]

        # --- Validate spec name --------------------------
        if batch_spec.name != live_spec.name:
            raise ValueError(f"name mismatch for indicator '{name}'")
        
        # --- Validate required_cols ----------------------
        if batch_spec.required_cols != live_spec.required_cols:
            raise ValueError(f"required_cols mismatch for indicator '{name}'")

        # --- Validate parameters -------------------------
        if batch_spec.parameters != live_spec.parameters:
            raise ValueError(f"parameters mismatch for indicator '{name}'")

        # --- Validate output_names -----------------------
        if batch_spec.output_names != live_spec.output_names:
            raise ValueError(
                f"output_names mismatch for indicator '{name}': "
                f"batch={batch_spec.output_names!r}, "
                f"live={live_spec.output_names!r}"
            )

        # --- Validate is_stateful ------------------------
        if batch_spec.is_stateful is True:
            raise ValueError(f"batch indicator '{name}' must be non-stateful")

        if live_spec.is_stateful is False:
            raise ValueError(f"live indicator '{name}' must be stateful")
        
validate_registry()

# =============================================================================
# Registry Builder
# ============================================================================= func_2 خوانده شد و فهمیده شد
def build_registry() -> Dict[str, Dict[str, IndicatorSpec]]:
    """
    این تابع دو دیکشنری _BATCH_INDICATORS و _LIVE_INDICATORS را با هم تلفیق نموده و
    یک دیکشنری بصورت زیر خروجی میدهد
        REGISTRY
        |
        +-- rsi
        |    ├── batch
        |    └── live
        |
        +-- ema
        |    ├── batch
        |    └── live
        |
        +-- ...
    """
    names = sorted(set(_BATCH_INDICATORS) | set(_LIVE_INDICATORS))
    registry: Dict[str, Dict[str, IndicatorSpec]] = {}
    for name in names:
        entry: Dict[str, IndicatorSpec] = {}
        if name in _BATCH_INDICATORS:
            entry["batch"] = _BATCH_INDICATORS[name]
        if name in _LIVE_INDICATORS:
            entry["live"] = _LIVE_INDICATORS[name]
        registry[name] = entry

    return registry

# Global registry instance
REGISTRY = build_registry()

# =============================================================================
# Public API
# ============================================================================= func_3 خوانده شد و فهمیده شد
def get_indicator(name: str, mode: Optional[str] = None) -> Optional[IndicatorSpec]:
                                                     # ) -> Optional[Union[IndicatorSpec, Dict[str, IndicatorSpec]]]:

    """
    این تابع IndicatorSpec مربوط به یک اندیکاتور را با توجه به نام و مود آن برمیگرداند
    """
    entry = REGISTRY.get(name)

    if entry is None:
        return None
    if mode is None:
        raise ValueError("mode must be explicitly specified")
    if mode not in ALL_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {sorted(ALL_MODES)}")
    
    if mode in TRAIN_MODES:
        spec = entry.get("batch")
        return spec if spec and spec.supports(mode) else None
    elif mode in INCREMENTAL_MODES:
        spec = entry.get("live")
        return spec if spec and spec.supports(mode) else None
    # elif mode in HYBRID_MODES:
    #     batch_spec = entry.get("batch")
    #     live_spec  = entry.get("live")
    #     if batch_spec is None or live_spec is None:
    #         return None
    #     return {"batch": batch_spec, "live": live_spec}
    return None

# ============================================================================= func_4 خوانده شد و فهمیده شد
def list_indicators(mode: Optional[str] = None) -> list[str]:
    """
    این تابع بر اساس مود، لیست نام اندیکاتورها را برمیگرداند.
    در صورتیکه مود به آن داده نشود، لیست تمام اندیکاتورهای موجود در رجیستری را برمیگرداند
    """
    if mode is None:
        return sorted(REGISTRY.keys())

    if mode not in ALL_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {sorted(ALL_MODES)}")

    if mode in INCREMENTAL_MODES:
        return [
            name for name, entry in REGISTRY.items()
            if "live" in entry and entry["live"].supports(mode)
        ]
    elif mode in TRAIN_MODES:
        return [
            name for name, entry in REGISTRY.items()
            if "batch" in entry and entry["batch"].supports(mode)
        ]
    # elif mode in HYBRID_MODES:
    #     return sorted(REGISTRY.keys())

    return []

# ============================================================================= func_5 هنوز مانده است
# ???????? بنظر باید اصلاح شود یا اینکه اصلاً مورد نیاز نباشد
def get_required_columns(name: str, mode: str) -> list[str]:
    spec = get_indicator(name, mode)
    if spec is None:
        return []
    if isinstance(spec, dict):
        batch_spec = spec.get("batch")
        live_spec  = spec.get("live")
        if batch_spec is None or live_spec is None:
            return []
        return live_spec.required_cols
    return spec.required_cols

# ============================================================================= func_6 هنوز مانده است
# ???????? بنظر اصلاً مورد نیاز نباشد
def is_stateful(name: str, mode: str) -> bool:
    spec = get_indicator(name, mode)
    if spec is None:
        return False
    if isinstance(spec, dict):
        return True
    return spec.is_stateful

# ============================================================================= END



""" CONTRACT TEXT FOR feature_C_registry_1.py:

A) PUBLIC API MAP

**Class**

* `IndicatorSpec`

  * Methods:

    * `supports(mode: str) -> bool`
    * `is_batch_mode(mode: str) -> bool`
    * `is_incremental_mode(mode: str) -> bool`
    * `__call__(*args, **kwargs)`

**Functions**

* `validate_registry() -> None`

  * Validates consistency between batch/live indicator definitions
  * Raises `ValueError` on mismatch conditions

* `build_registry() -> Dict[str, Dict[str, IndicatorSpec]]`

  * Constructs unified registry with keys: `"batch"`, `"live"`

* `get_indicator(name: str, mode: Optional[str]) -> Optional[IndicatorSpec] | Dict[str, IndicatorSpec]`

  * Resolves indicator spec by mode (train vs incremental)
  * Enforces explicit mode requirement

* `list_indicators(mode: Optional[str]) -> list[str]`

  * Returns indicator names filtered by mode

* `get_required_columns(name: str, mode: str) -> list[str]`

  * Returns required OHLCV columns for indicator

* `is_stateful(name: str, mode: str) -> bool`

  * Returns whether indicator is stateful in given mode

**Entry Point / Module Behavior**

* On import:

  * `validate_registry()` is executed immediately
  * `REGISTRY = build_registry()` is created
  * Debug print executes:

    * `"===== REGISTRY DUMP ====="`
    * prints all indicators, batch/live function names and required cols

---

B) DEPENDENCY CONTRACTS

**Standard Library**

* `dataclasses.dataclass`, `dataclasses.field`
* `typing: Callable, Set, Optional, Dict, Union`

**Internal Imports (Stateful / Live Layer)**

* `f04_features.indicators_new.indicators_B_class`

  * SMA, WMA, EMA, ROC, RSI, TrueRange, ATR, MACD, BollingerBands,
    KeltnerChannel, Stochastic, CCI, MFI, OBV, WilliamsR,
    ParabolicSAR, HeikinAshi, Supertrend, Aroon, DEMA, TEMA, KAMA, HMA

**Internal Imports (Batch / Train Layer)**

* `f04_features.indicators_new.indicators_B_batch`

  * sma_batch_df, wma_batch_df, ema_batch_df, roc_batch_df, rsi_batch_df,
    truerange_batch_df, atr_batch_df, macd_batch_df, bollinger_batch_df,
    keltner_batch_df, stochastic_batch_df, cci_batch_df, mfi_batch_df,
    obv_batch_df, williamsr_batch_df, parabolicsar_batch_df,
    heikinashi_batch_df, supertrend_batch_df, aroon_batch_df,
    dema_batch_df, tema_batch_df, kama_batch_df, hma_batch_df

**Internal Dependency Contracts**

* Batch indicators MUST be stateless functions
* Live indicators MUST be stateful classes
* Both layers MUST share identical:

  * `name`
  * `required_cols`
* Registry validation enforces consistency and raises `ValueError` on mismatch
* No external API calls or IO dependencies

---

C) DATA CONTRACT

**Core Schema**

`IndicatorSpec`

* `name: str`
* `fn: Callable`
* `is_stateful: bool`
* `modes: Set[str]`
* `required_cols: list[str]`
* `needs_tf_map: bool`
* `description: str`

**Mode Sets**

* `TRAIN_MODES = {"train", "optimize"}`
* `INCREMENTAL_MODES = {"live", "paper", "shadow", "backtest", "replay", "eval"}`
* `BATCH_MODES = TRAIN_MODES`
* `ALL_MODES = TRAIN_MODES ∪ INCREMENTAL_MODES`

**Registry Structure**

* `REGISTRY: Dict[str, Dict[str, IndicatorSpec]]`

  * Key: indicator name (e.g., `"sma"`)
  * Value:

    * `"batch": IndicatorSpec` (train-mode function)
    * `"live": IndicatorSpec` (stateful class)

**Function I/O Contracts**

* `get_indicator`

  * Input: `(name: str, mode: str | None)`
  * Output:

    * `IndicatorSpec` OR `None`
    * (hybrid dict return is defined but disabled/commented)

* `list_indicators`

  * Input: `mode: str | None`
  * Output: `list[str]`

* `get_required_columns`

  * Input: `(name: str, mode: str)`
  * Output: `list[str]`
  * If unresolved → `[]`

* `is_stateful`

  * Input: `(name: str, mode: str)`
  * Output: `bool`

**Data Constraints**

* `required_cols` defines strict OHLCV schema dependency per indicator
* Mode must be explicitly provided in `get_indicator`
* Invalid mode → `ValueError`
* Missing indicator → `None`
* Registry must pass validation before use (batch/live symmetry enforced)

**Error Contracts**

* `ValueError`:

  * missing mode
  * invalid mode
  * registry mismatch (name/cols/statefulness inconsistencies)

"""

