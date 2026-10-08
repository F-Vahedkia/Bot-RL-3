# f02_utils/config_completer.py
# Date reviewed:
#    1405/05/24-17:25 --> run result is ?????????????????

# Run: python -m f02_utils.config_completer

# =============================================================================
#  Imports
# =============================================================================
from typing import Any, Dict, List, Literal, Optional, Union
from pathlib import Path
from f02_utils.config_loader import load_config
from f02_utils.functions.parse_warmups import get_warmup_from_config_allsyms
import logging
from f02_utils.functions.constants import normalize_and_sort_timeframe_dicts as normalize_dict
# =============================================================================
# Logger
# =============================================================================
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =============================================================================
# Merge common symbols
# =============================================================================
def _apply_common_symbol_config(cfg: Dict) -> Dict:
    """
    Apply features.symbols.COMMON settings to all symbols listed in
    COMMON.symbols.

    COMMON.candles are appended to each target symbol's candles.
    COMMON.indicators are appended to each target symbol's indicators.

    The COMMON section itself is not included as a trading symbol.
    """

    symbols_cfg = cfg.get("features", {}).get("symbols", {})

    if not isinstance(symbols_cfg, dict):
        raise ValueError(
            "config.features.symbols must be a dictionary."
        )

    common_cfg = symbols_cfg.get("COMMON")

    if common_cfg is None:
        return cfg

    if not isinstance(common_cfg, dict):
        raise ValueError(
            "config.features.symbols.COMMON must be a dictionary."
        )

    common_symbols = common_cfg.get("symbols", [])

    if not isinstance(common_symbols, list):
        raise ValueError(
            "config.features.symbols.COMMON.symbols must be a list."
        )

    common_candles = common_cfg.get("candles", [])
    common_indicators = common_cfg.get("indicators", [])

    if not isinstance(common_candles, list):
        raise ValueError(
            "config.features.symbols.COMMON.candles must be a list."
        )

    if not isinstance(common_indicators, list):
        raise ValueError(
            "config.features.symbols.COMMON.indicators must be a list."
        )

    for symbol in common_symbols:

        if symbol not in symbols_cfg:
            raise ValueError(
                f"Symbol '{symbol}' is listed in COMMON.symbols "
                f"but does not exist in config.features.symbols."
            )
        symbol_cfg = symbols_cfg[symbol]
        if not isinstance(symbol_cfg, dict):
            raise ValueError(
                f"config.features.symbols.{symbol} must be a dictionary."
            )
        # -------------------------------------------------
        # candles
        # -------------------------------------------------
        symbol_candles = symbol_cfg.setdefault("candles", [])
        if not isinstance(symbol_candles, list):
            raise ValueError(
                f"config.features.symbols.{symbol}.candles must be a list."
            )
        for candle in common_candles:
            if candle not in symbol_candles:
                symbol_candles.append(candle)

        # -------------------------------------------------
        # indicators
        # -------------------------------------------------
        symbol_indicators = symbol_cfg.setdefault("indicators", [])
        if not isinstance(symbol_indicators, list):
            raise ValueError(
                f"config.features.symbols.{symbol}.indicators must be a list."
            )
        for indicator in common_indicators:
            if indicator not in symbol_indicators:
                symbol_indicators.append(indicator)

    return cfg


def _apply_common_symbol_config_2(cfg: Dict) -> Dict:
    """
    Expand the COMMON configuration into the selected symbols.

    COMMON.symbols:
        List of symbols that should receive the common configuration.

    COMMON.candles:
        Candles added to each selected symbol.

    COMMON.indicators:
        Indicators added to each selected symbol.

    A symbol does NOT need to have candles or indicators beforehand.
    Missing candles/indicators are created automatically.

    The COMMON section itself is not treated as a trading symbol.
    """

    # ------------------------------------------------------------------
    # Locate features.symbols
    # ------------------------------------------------------------------
    features_cfg = cfg.get("features", {})

    if not isinstance(features_cfg, dict):
        raise ValueError(
            "config.features must be a dictionary."
        )

    symbols_cfg = features_cfg.get("symbols", {})

    if not isinstance(symbols_cfg, dict):
        raise ValueError(
            "config.features.symbols must be a dictionary."
        )

    # ------------------------------------------------------------------
    # COMMON
    # ------------------------------------------------------------------
    common_cfg = symbols_cfg.get("COMMON")

    # No COMMON section -> nothing to do
    if common_cfg is None:
        return cfg

    if not isinstance(common_cfg, dict):
        raise ValueError(
            "config.features.symbols.COMMON must be a dictionary."
        )

    # ------------------------------------------------------------------
    # COMMON.symbols
    # ------------------------------------------------------------------
    common_symbols = common_cfg.get("symbols", [])

    if common_symbols is None:
        common_symbols = []

    if not isinstance(common_symbols, list):
        raise ValueError(
            "config.features.symbols.COMMON.symbols must be a list."
        )

    # ------------------------------------------------------------------
    # COMMON.candles
    # ------------------------------------------------------------------
    common_candles = common_cfg.get("candles", [])

    if common_candles is None:
        common_candles = []

    if not isinstance(common_candles, list):
        raise ValueError(
            "config.features.symbols.COMMON.candles must be a list."
        )

    # ------------------------------------------------------------------
    # COMMON.indicators
    # ------------------------------------------------------------------
    common_indicators = common_cfg.get("indicators", [])

    if common_indicators is None:
        common_indicators = []

    if not isinstance(common_indicators, list):
        raise ValueError(
            "config.features.symbols.COMMON.indicators must be a list."
        )

    # ------------------------------------------------------------------
    # Apply COMMON to every selected symbol
    # ------------------------------------------------------------------
    for symbol in common_symbols:

        # --------------------------------------------------------------
        # The symbol must exist as a real symbol in the configuration.
        # --------------------------------------------------------------
        if symbol not in symbols_cfg:
            raise ValueError(
                f"Symbol '{symbol}' is listed in "
                f"config.features.symbols.COMMON.symbols "
                f"but does not exist in config.features.symbols."
            )

        symbol_cfg = symbols_cfg[symbol]

        if not isinstance(symbol_cfg, dict):
            raise ValueError(
                f"config.features.symbols.{symbol} must be a dictionary."
            )

        # ==============================================================
        # candles
        # ==============================================================

        # The symbol may have NO candles at all.
        # In that case create an empty list first.
        symbol_candles = symbol_cfg.get("candles")

        if symbol_candles is None:
            symbol_candles = []
            symbol_cfg["candles"] = symbol_candles

        elif not isinstance(symbol_candles, list):
            raise ValueError(
                f"config.features.symbols.{symbol}.candles "
                f"must be a list."
            )

        # Add COMMON candles.
        for candle in common_candles:
            if candle not in symbol_candles:
                symbol_candles.append(candle)

        # ==============================================================
        # indicators
        # ==============================================================

        # The symbol may have NO indicators at all.
        # In that case create an empty list first.
        symbol_indicators = symbol_cfg.get("indicators")

        if symbol_indicators is None:
            symbol_indicators = []
            symbol_cfg["indicators"] = symbol_indicators

        elif not isinstance(symbol_indicators, list):
            raise ValueError(
                f"config.features.symbols.{symbol}.indicators "
                f"must be a list."
            )

        # Add COMMON indicators.
        for indicator in common_indicators:
            if indicator not in symbol_indicators:
                symbol_indicators.append(indicator)

    # ------------------------------------------------------------------
    # Return the completed configuration
    # ------------------------------------------------------------------
    return cfg


def _apply_common_symbol_config_3(cfg: Dict) -> Dict:
    """
    Expand the COMMON indicator configuration into the selected symbols.

    COMMON.symbols:
        List of symbols that should receive the common configuration.

    COMMON.indicators:
        Indicators added to each selected symbol.

    A symbol does NOT need to have indicators beforehand.
    Missing indicators are created automatically.

    The COMMON section itself is not treated as a trading symbol.
    """

    # ------------------------------------------------------------------
    # Locate features.symbols
    # ------------------------------------------------------------------

    features_cfg = cfg.get("features", {})
    if not isinstance(features_cfg, dict):
        raise ValueError(
            "config.features must be a dictionary."
        )

    symbols_cfg = features_cfg.get("symbols", {})
    if not isinstance(symbols_cfg, dict):
        raise ValueError(
            "config.features.symbols must be a dictionary."
        )

    # ------------------------------------------------------------------
    # COMMON
    # ------------------------------------------------------------------

    common_cfg = symbols_cfg.get("COMMON")

    # No COMMON section -> nothing to do
    if common_cfg is None:
        return cfg

    if not isinstance(common_cfg, dict):
        raise ValueError(
            "config.features.symbols.COMMON must be a dictionary."
        )

    # ------------------------------------------------------------------
    # COMMON.symbols
    # ------------------------------------------------------------------

    common_symbols = common_cfg.get("symbols", [])

    if common_symbols is None:
        common_symbols = []

    if not isinstance(common_symbols, list):
        raise ValueError(
            "config.features.symbols.COMMON.symbols must be a list."
        )

    # ------------------------------------------------------------------
    # COMMON.indicators
    # ------------------------------------------------------------------

    common_indicators = common_cfg.get("indicators", [])

    if common_indicators is None:
        common_indicators = []

    if not isinstance(common_indicators, list):
        raise ValueError(
            "config.features.symbols.COMMON.indicators must be a list."
        )

    # ------------------------------------------------------------------
    # Apply COMMON indicators to every selected symbol
    # ------------------------------------------------------------------

    for symbol in common_symbols:

        # --------------------------------------------------------------
        # The symbol must exist as a real symbol in the configuration.
        # --------------------------------------------------------------

        if symbol not in symbols_cfg:
            raise ValueError(
                f"Symbol '{symbol}' is listed in "
                f"config.features.symbols.COMMON.symbols "
                f"but does not exist in config.features.symbols."
            )

        symbol_cfg = symbols_cfg[symbol]

        if not isinstance(symbol_cfg, dict):
            raise ValueError(
                f"config.features.symbols.{symbol} must be a dictionary."
            )

        # ==============================================================
        # indicators
        # ==============================================================

        # The symbol may have NO indicators at all.
        # In that case create an empty list first.

        symbol_indicators = symbol_cfg.get("indicators")

        if symbol_indicators is None:
            symbol_indicators = []
            symbol_cfg["indicators"] = symbol_indicators

        elif not isinstance(symbol_indicators, list):
            raise ValueError(
                f"config.features.symbols.{symbol}.indicators "
                f"must be a list."
            )

        # Add COMMON indicators.

        for indicator in common_indicators:
            if indicator not in symbol_indicators:
                symbol_indicators.append(indicator)

    # ------------------------------------------------------------------
    # Return the completed configuration
    # ------------------------------------------------------------------

    return cfg


# =============================================================================
# Read candles requirements
# =============================================================================

def _extract_candles_required_bars(cfg: Dict) -> Dict[str, Dict[str, int]]:
    """
    Extract raw-candle requirements from:

        config.features.candles.required

    Expected configuration structure:

        features:
          candles:
            required:
                XAUUSD:
                M1: 20
                M5: 4
                H1: 4
                EURUSD:
                M5: 5
                M10: 10
                BITCOIN:
                M1: 10
                H1: 20

    Returns
    -------
    Dict[str, Dict[str, int]]
        Required raw candle bars for each symbol and timeframe.
    """

    # ------------------------------------------------------------------
    # Locate features
    # ------------------------------------------------------------------

    features_cfg = cfg.get("features", {})

    if not isinstance(features_cfg, dict):
        raise ValueError(
            "config.features must be a dictionary."
        )

    # ------------------------------------------------------------------
    # Locate candles
    # ------------------------------------------------------------------

    candles_cfg = features_cfg.get("candles", {}).get("required", {})

    if candles_cfg is None:
        return {}

    if not isinstance(candles_cfg, dict):
        raise ValueError(
            "config.features.candles.required must be a dictionary."
        )

    # ------------------------------------------------------------------
    # Extract candles requirements
    # ------------------------------------------------------------------

    candles_required_bars: Dict[str, Dict[str, int]] = {}

    for symbol, tf_dict in candles_cfg.items():

        if not isinstance(tf_dict, dict):
            raise ValueError(
                f"config.features.candles.required.{symbol} "
                f"must be a dictionary."
            )

        symbol_bars: Dict[str, int] = {}

        for timeframe, bar_count in tf_dict.items():

            if not isinstance(bar_count, int) or isinstance(bar_count, bool):
                raise ValueError(
                    f"Required bar count for symbol '{symbol}', "
                    f"timeframe '{timeframe}' must be an integer."
                )

            if bar_count <= 0:
                raise ValueError(
                    f"Required bar count for symbol '{symbol}', "
                    f"timeframe '{timeframe}' must be > 0."
                )

            symbol_bars[str(timeframe).upper()] = bar_count

        if symbol_bars:
            candles_required_bars[symbol] = symbol_bars

    return candles_required_bars


# =============================================================================
# Merge results of two above function
# =============================================================================

def _merge_required_bars(
    warmups_dicts: Dict[str, Dict[str, int]],
    candles_required_bars: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    """
    Merge indicator warmup requirements and raw-candle requirements.

    For every symbol and timeframe:

        required_bars = max(warmup_bars, candle_bars)

    If a symbol/timeframe exists in only one of the two dictionaries,
    its value is copied directly to the result.

    Returns
    -------
    Dict[str, Dict[str, int]]
        Final required bars for each symbol and timeframe.
    """

    required_bars: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Merge indicator warmups
    # ------------------------------------------------------------------

    for symbol, tf_dict in warmups_dicts.items():

        required_bars[symbol] = {}

        for timeframe, bar_count in tf_dict.items():
            required_bars[symbol][timeframe] = bar_count

    # ------------------------------------------------------------------
    # Merge raw-candle requirements
    # ------------------------------------------------------------------

    for symbol, tf_dict in candles_required_bars.items():

        if symbol not in required_bars:
            required_bars[symbol] = {}

        for timeframe, bar_count in tf_dict.items():

            if timeframe in required_bars[symbol]:
                required_bars[symbol][timeframe] = max(
                    required_bars[symbol][timeframe],
                    bar_count,
                )
            else:
                required_bars[symbol][timeframe] = bar_count

    all_required_bars = required_bars
    return all_required_bars

# =============================================================================
# Main
# =============================================================================

def config_completer(path: Optional[Union[str, Path]] = None,
                    env_prefix: str = "BOT_",
                    enable_env_override: bool = True,
                    copy_: Literal["main", "shallow", "mutable-safe", "deep"] = "shallow"
                    ) -> Dict[str, Any]:

    """ Docstring:
    Load, complete, normalize, and derive the main configuration
    requirements used by the bot.

    The function:

    - applies COMMON indicator settings to selected symbols;
    - calculates indicator warm-up requirements;
    - extracts raw-candle requirements;
    - merges both into ``__all_required_bars`` using the maximum
    requirement for overlapping symbol/timeframe pairs;
    - derives ``__symbols``, ``__timeframes_dict``, and
    ``__base_tfs_dict`` from the final normalized and sorted requirements.

    Returns
    -------
    Dict[str, Any]
        Completed configuration containing the original settings and
        the derived internal configuration dictionaries.
    """

    """ Examples:
    Some Data structures:
    __warmups_dicts = {
        "XAUUSD" : {'M1': 14,  'M5':9 , 'H1':14, 'D1':12},
        "EURUSD" : {'M5': 12, 'M30':14, 'D1':5 },
        "BITCOIN": {'H2': 16,  'D1':9 },
    }
    __symbols = ["XAUUSD", "EURUSD", "BITCOIN"]

    __timeframes_dict = {
        "XAUUSD" : ['M1',  'M5', 'H1', 'D1],
        "EURUSD" : ['M5', 'M30', 'D1'],
        "BITCOIN": ['H2',  'D1'],
    }

    __base_tfs_dict = {
        "XAUUSD" : 'M1' ,
        "EURUSD" : 'M5' ,
        "BITCOIN": 'M12',
    }
    """
    cfg = load_config(
        path=path,
        env_prefix=env_prefix,
        enable_env_override=enable_env_override,
        copy_=copy_,
        )

    if cfg is None:
        raise ValueError("Message from f02_utils/config_completer: cfg is None."
                         "Therefore can't initialize robot.")

    # -----------------------------------------------------
    # Apply COMMON symbol configuration
    # -----------------------------------------------------
    cfg = _apply_common_symbol_config_3(cfg)

    # -----------------------------------------------------
    # 1) Warmup Dicts
    # -----------------------------------------------------
    warmups_dicts: Dict[str, Dict[str, int]] = get_warmup_from_config_allsyms(cfg)
    warmups_dicts.pop("COMMON", None)

    if warmups_dicts == {}:
        cfg["__warmups_dicts"] = {}
    else:
        warmups_dicts = normalize_dict(warmups_dicts)
        cfg["__warmups_dicts"] = warmups_dicts

    # -----------------------------------------------------
    # 2) candles_required_bars
    # -----------------------------------------------------
    candles_required_bars: Dict[str, Dict[str, int]] = _extract_candles_required_bars(cfg)
    if candles_required_bars == {}:
        cfg["__candles_required_bars"] = {}
    else:
        candles_required_bars = normalize_dict(candles_required_bars)
        cfg["__candles_required_bars"] = candles_required_bars

    # -----------------------------------------------------
    # 3) all_required_bars
    # -----------------------------------------------------
    all_required_bars: Dict[str, Dict[str, int]] = _merge_required_bars(warmups_dicts, candles_required_bars)
    if all_required_bars == {}:
        print("WARNING ! Message from f02_utils/config_completer: In config.features.symbols there is no any timeframe.")
        cfg["__all_required_bars"] = {}
        cfg["__symbols"] = []
        cfg["__timeframes_dict"] = {}
        cfg["__base_tfs_dict"] = {}
    else:
        # 1) --- all required bars
        cfg["__all_required_bars"] = normalize_dict(all_required_bars) # یک دیکشنری که حاوی چندین warmup_dict است

        # 2) --- symbols
        symbols: List[str] = list(all_required_bars.keys())

        cfg["__symbols"] = symbols

        # 3) --- timeframes
        timeframes_dict = {sym: list(warmup.keys()) for sym, warmup in all_required_bars.items()}
        cfg["__timeframes_dict"] = timeframes_dict

        # 4) --- base_tfs_dict
        base_tfs_dict = {sym: list(warmup.keys())[0] for sym, warmup in all_required_bars.items()}
        """
        # بلوک زیر، همان کار سطر بالا را انجام میدهد.
        # ولی نمادهایی که دیکشنری وارم-آپ انها تهی است را بطور خودکار به حساب نمی آورد.
        """
        # base_tfs_dict = {}
        # for sym, warmup in all_required_bars.items():
        #     if warmup.keys():
        #         base_tfs_dict[sym] = list(warmup.keys())[0]
        cfg["__base_tfs_dict"] = base_tfs_dict


    return cfg

# ===================================================================
# TESTER
# ===================================================================
def main():
    cfg = config_completer()

    print("\n", "="*3, "__warmups_dicts", "="* (60-len("__warmups_dicts")))
    print(cfg["__warmups_dicts"])

    print("\n", "="*3, "__candles_required_bars", "="* (60-len("__candles_required_bars")))
    print(cfg["__candles_required_bars"])

    print("\n", "="*3, "__all_required_bars", "="* (60-len("__all_required_bars")))
    print(cfg["__all_required_bars"])

    print("\n", "="*3, "__symbols", "="* (60-len("__symbols")))
    print(cfg["__symbols"])

    print("\n", "="*3, "__timeframes_dict", "="* (60-len("__timeframes_dict")))
    print(cfg["__timeframes_dict"])

    print("\n", "="*3, "__base_tfs_dict", "="* (60-len("__base_tfs_dict")))
    print(cfg["__base_tfs_dict"])

    for sym in cfg["__symbols"]:
        print("\n", "="*3, {sym} , "="* (60-len(sym)-4))
        specs = list(cfg["features"]["symbols"][sym]["indicators"])
        for s in specs:
            print(s)
        print("\n", (cfg["features"]["candles"][sym]))
    print("\n")

# ===================================================================
if __name__ == "__main__":
    raise SystemExit(main())
