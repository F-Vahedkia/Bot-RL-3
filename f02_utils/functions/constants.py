# f02_utils/functions/constants.py

from typing import Dict, List, Literal, Union
from datetime import datetime, timezone

# =======================================================================================
# نگاشت دقیقه هر تایم‌فریم (برای تبدیل lookback به بازهٔ زمانی امن) 
# =======================================================================================
_TF_MINUTES = {
    "M1" : 1,    "M2" : 2,     "M3" : 3,     "M4" : 4,    "M5" : 5,    "M6": 6,
    "M10": 10,   "M12": 12,    "M15": 15,    "M20": 20,   "M30": 30,
    "H1" : 60,   "H2" : 120,   "H3" : 180,   "H4" : 240,  "H6" : 360,  "H8": 480, "H12": 720,
    "D1" : 1440, "W1" : 10080, "MN1": 43200,
}

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)

# =======================================================================================
# نگاشت تایم‌فریم‌ها 
# =======================================================================================
_TF_MAP = {
    # دقیقه
    "M1" : "M1" , "1M" : "M1" , "1" : "M1" , "1m" : "M1" , "m1" : "M1" ,
    "M2" : "M2" , "2M" : "M2" , "2" : "M2" , "2m" : "M2" , "m2" : "M2" ,
    "M3" : "M3" , "3M" : "M3" , "3" : "M3" , "3m" : "M3" , "m3" : "M3" ,
    "M4" : "M4" , "4M" : "M4" , "4" : "M4" , "4m" : "M4" , "m4" : "M4" ,    
    "M5" : "M5" , "5M" : "M5" , "5" : "M5" , "5m" : "M5" , "m5" : "M5" ,
    "M6" : "M6" , "6M" : "M6" , "6" : "M6" , "6m" : "M6" , "m6" : "M6" ,

    "M10": "M10", "10M": "M10", "10": "M10", "10m": "M10", "m10": "M10",
    "M12": "M12", "12M": "M12", "12": "M12", "12m": "M12", "m12": "M12",
    "M15": "M15", "15M": "M15", "15": "M15", "15m": "M15", "m15": "M15",
    "M20": "M20", "20M": "M20", "20": "M20", "20m": "M20", "m20": "M20",
    "M30": "M30", "30M": "M30", "30": "M30", "30m": "M30", "m30": "M30",
    
    # ساعت
    "H1" : "H1" , "1H"  : "H1"  , "1h" : "H1" , "h1" : "H1" ,
    "H2" : "H2" , "2H"  : "H2"  , "2h" : "H2" , "h2" : "H2" ,
    "H3" : "H3" , "3H"  : "H3"  , "3h" : "H3" , "h3" : "H3" ,
    "H4" : "H4" , "4H"  : "H4"  , "4h" : "H4" , "h4" : "H4" ,
    "H6" : "H6" , "6H"  : "H6"  , "6h" : "H6" , "h6" : "H6" ,
    "H8" : "H8" , "8H"  : "H8"  , "8h" : "H8" , "h8" : "H8" ,
    "H12": "H12", "12H" : "H12" , "12h": "H12", "h12": "H12",
    
    # روز/هفته/ماه
    "D1" : "D1" , "1D"  : "D1"  , "1d" : "D1" , "d1" : "D1" , 
    "W1" : "W1" , "1W"  : "W1"  , "1w" : "W1" , "w1" : "W1" ,
    "MN1": "MN1", "1MN" : "MN1" , "1mn": "MN1", "mn1": "MN1",
}

# =======================================================================================

def mapping_tf(tf: str) -> str:
    if not isinstance(tf, str):
        raise TypeError("The input parameter must be 'str'.")
    tf_temp = tf.upper().replace(" ","")
    if tf_temp not in _TF_MAP:
        raise ValueError(f"Can not map the timeframe {tf} to standard form !")
    return _TF_MAP[tf_temp]
        
# =======================================================================================

def mapping_tfs(
    tfs: Union[List, Dict],
    mode: Literal["keys", "values", "both"] = "keys"
) -> Union[List, Dict]:
    """
    نگاشت تایم‌فریم‌ها در لیست یا دیکشنری به فرم استاندارد.

    Args:
        tfs: لیستی از رشته‌ها یا دیکشنری با کلیدها/مقدارهای رشته‌ای
        mode: مشخص میکند چه بخشی نگاشت شود ('keys', 'values', 'both')

    Returns:
        ساختاری مشابه ورودی با تایم‌فریم‌های استانداردشده
    """
    if isinstance(tfs, list):
        return [mapping_tf(item) for item in tfs]
    
    elif isinstance(tfs, dict):
        mode_lower = mode.lower()

        if mode_lower == "keys":      # نگاشت کلیدها
            return {mapping_tf(key): value for key, value in tfs.items()}
        
        elif mode_lower == "values":   # نگاشت مقدارها
            return {key: mapping_tf(value) for key, value in tfs.items()}
        
        elif mode_lower == "both":     # نگاشت هردو
            return {
                mapping_tf(key) if isinstance(key, str) else key: 
                mapping_tf(value) if isinstance(value, str) else value
                for key, value in tfs.items()
            }
        else:
            # این خطا هرگز با Literal اتفاق نمیافتد، اما برای ایمنی بیشتر نگهش دارید
            raise ValueError("mode must be 'keys', 'values', or 'both'")
    else:
        raise TypeError("The input parameter must be 'List' or 'Dict'.")
    
# =======================================================================================

def normalize_and_sort_timeframe_dicts(
    tf_dicts: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, int]]:
    """
    Normalize timeframe names and sort timeframes from smallest to largest.

    Input structure:
        Dict[symbol, Dict[timeframe, int]]
    Example:
        {
            "XAUUSD": {"1h": 4, "1M": 20, "5  m": 9},
            "EURUSD": {"10  m": 10, "m 5": 5},
        }
    Output:
        {
            "XAUUSD": {"M1": 20, "M5": 9, "H1": 4},
            "EURUSD": {"M5": 5, "M10": 10},
        }
    """
    if not isinstance(tf_dicts, dict):
        raise TypeError(
            "The input parameter must be a dictionary."
        )

    result: Dict[str, Dict[str, int]] = {}

    for symbol, symbol_tf_dict in tf_dicts.items():

        if not isinstance(symbol_tf_dict, dict):
            raise TypeError(
                f"Timeframe data for symbol '{symbol}' "
                f"must be a dictionary."
            )

        # --------------------------------------------------------------
        # Normalize timeframe names
        # --------------------------------------------------------------

        normalized_tf_dict = mapping_tfs(
            symbol_tf_dict,
            mode="keys",
        )

        # --------------------------------------------------------------
        # Prevent silent data loss after normalization
        # Example:
        # {"M1": 20, "1M": 30}
        # both become "M1"
        # --------------------------------------------------------------

        if len(normalized_tf_dict) != len(symbol_tf_dict):
            raise ValueError(
                f"Duplicate timeframe after normalization "
                f"for symbol '{symbol}'."
            )

        # --------------------------------------------------------------
        # Sort timeframes from smallest to largest
        # --------------------------------------------------------------

        sorted_tf_dict = dict(
            sorted(
                normalized_tf_dict.items(),
                key=lambda item: _TF_MINUTES[item[0]],
            )
        )

        result[symbol] = sorted_tf_dict

    return result

# ======================================================================================= END

