# f02_utils/parse_warmups.py

# =============================================================================
# Imports
# =============================================================================
import logging
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

# ------------------ Importing Internal Modules ---------------------
from f03_data.data_layer_functions import check_tfs
from f02_utils.functions.parser import parse_spec
from f02_utils.config_loader import load_config
from f02_utils.config_path_funcs import project_root
from f02_utils.functions.constants import mapping_tfs

# -------------------- Logger for this module -----------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =============================================================================
# تابع کمکی برای محاسبه warmup از روی لیست رشته‌های اندیکاتور
# =============================================================================
def _compute_warmup_from_specs(specs: List[str]) -> Dict[str, int]:
    """
    محاسبه warmup (حداقل تعداد کندل‌های مورد نیاز) برای هر تایم‌فریم.
    
    برای هر اندیکاتور، پارامتر دوره‌ای اصلی (period, n, fast, slow, k_period و ...) استخراج می‌شود.
    پارامترهایی مانند min_periods نادیده گرفته می‌شوند.
    """
    # مپ اندیکاتور به نام پارامتر دوره‌ای (در args یا kwargs)
    # همچنین در صورت وجود چند پارامتر (مثل MACD)، بزرگترین آنها را برمی‌گردانیم
    period_params = {
        'sma'            : ['period'],
        'wma'            : ['period'],
        'ema'            : ['period'],
        'roc'            : ['period'],
        'rsi'            : ['period'],
        'true_range'     : [        ], # بدون دوره
        'atr'            : ['period'],
        'macd'           : [        ], # در بخش استثنائات محاسبه میشود
        'bollinger_bands': ['period'],
        'keltner_channel': ['period'],
        'stochastic'     : [        ], # در بخش استثنائات محاسبه میشود
        'cci'            : ['period'],
        'mfi'            : ['period'],
        'obv'            : [        ],
        'williams_r'     : ['period'],
        'parabolic_sar'  : [        ], # بدون دوره
        'heikin_ashi'    : [        ], # بدون دوره
        'supertrend'     : ['period'],
        'aroon'          : ['period'],
        'dema'           : [        ], # در بخش استثنائات محاسبه میشود
        'tema'           : [        ], # در بخش استثنائات محاسبه میشود
        'kama'           : ['period'],
        'hma'            : [        ], # در بخش استثنائات محاسبه میشود
    }
    
    warmup_per_tf = {}
    
    for spec in specs:
        try:
            ps = parse_spec(spec)
            # pprint.pprint(vars(ps))
            tf = ps.timeframe
            name = ps.name.lower()
            # === محاسبه وارم آپ مورد نیاز بصورت استثناء ====================== start
            # --- MACD ------------------------------------
            if name == "macd":
                fast = ps.kwargs.get("fast", 12)
                slow = ps.kwargs.get("slow", 26)
                signal = ps.kwargs.get("signal", 9)

                period = max(slow, fast) + signal - 1

                if tf not in warmup_per_tf or period > warmup_per_tf[tf]:
                    warmup_per_tf[tf] = period

                continue
            # --- Stochastic ------------------------------
            elif name == "stochastic":
                k_period = ps.kwargs.get("k_period", 14)
                smooth_k = ps.kwargs.get("smooth_k", 3)
                d_period = ps.kwargs.get("d_period", 3)

                period = int(k_period) + int(smooth_k) + int(d_period) - 2

                if tf not in warmup_per_tf or period > warmup_per_tf[tf]:
                    warmup_per_tf[tf] = period

                continue
            # --- Dema ------------------------------------
            elif name == "dema":
                n = ps.kwargs.get("period", ps.kwargs.get("n", 20))

                period = 2 * n - 1

                if tf not in warmup_per_tf or period > warmup_per_tf[tf]:
                    warmup_per_tf[tf] = period

                continue
            # --- TEMA ------------------------------------
            elif name == "tema":
                n = ps.kwargs.get("period", ps.kwargs.get("n", 20))

                if len(ps.args) > 0 and isinstance(ps.args[0], (int, float)):
                    n = int(ps.args[0])

                period = 3 * int(n) - 2

                if tf not in warmup_per_tf or period > warmup_per_tf[tf]:
                    warmup_per_tf[tf] = period

                continue
            # --- HMA -------------------------------------
            elif name == "hma":
                n = ps.kwargs.get("period", ps.kwargs.get("n", 20))

                if len(ps.args) > 0 and isinstance(ps.args[0], (int, float)):
                    n = int(ps.args[0])

                period = int(n) + int(np.sqrt(int(n))) - 1

                if tf not in warmup_per_tf or period > warmup_per_tf[tf]:
                    warmup_per_tf[tf] = period

                continue
            # === محاسبه وارم آپ مورد نیاز بصورت استثناء ====================== end

            # یافتن پارامترهای دوره‌ای بر اساس مپ
            params_to_check = period_params.get(name, [])
            periods = []
            
            # استخراج از args (بر اساس ترتیب)
            # برای سادگی، اگر پارامتر period در args باشد (مثل sma(20))، همان را بگیر
            if 'period' in params_to_check and len(ps.args) > 0 and isinstance(ps.args[0], (int, float)):
                periods.append(int(ps.args[0]))
            # استخراج از kwargs
            for p in params_to_check:
                if p in ps.kwargs and isinstance(ps.kwargs[p], (int, float)):
                    periods.append(int(ps.kwargs[p]))
            
            # اگر اندیکاتور خاصی ناشناخته باشد، اولین عدد در args را امتحان کن
            if not periods and not params_to_check and len(ps.args) > 0:
                for arg in ps.args:
                    if isinstance(arg, (int, float)):
                        periods.append(int(arg))
                        break
            
            # محاسبه period نهایی
            if not periods:
                # اندیکاتورهای بدون دوره (مانند TR, OBV, ParabolicSAR, HeikinAshi)
                period = 1
            else:
                period = max(periods)  # برای MACD بزرگترین (slow) را می‌گیرد
            
            # به‌روزرسانی حداکثر warmup در تایم‌فریم
            if tf not in warmup_per_tf or period > warmup_per_tf[tf]:
                warmup_per_tf[tf] = period
                
        except Exception as e:
            logger.warning(f"Error parsing {spec}: {e}")
    
    return warmup_per_tf

# =============================================================================
# رپر تابع بالا برای فراخوانی در مصرف کنندگان
# =============================================================================
# ==> Address Used: config.features.symbols.indicators
def get_warmup_from_config_allsyms(cfg: Optional[Dict] = None) -> Dict[str, Dict[str, int]]:
    """
    - خواندن لیست رشته‌های اندیکاتور از کانفیگ و محاسبه وارم-آپ مورد نیاز برای هر تایم‌فریم.
    - این کار برای هر نماد بطور جداگانه انجام میشود. یعنی از حلقه روی نمادها استفاده میشود.
    - در نهایت نمادهایی که دیکشنری وارم-آپ آنها تهی است، در نتیجه نهایی قرار نمیگیرند.

    Parameters:
    -----------
    cfg : dict, optional
        دیکشنری کانفیگ. اگر None باشد، با load_config() بارگذاری می‌شود.
    
    Returns:
    --------
    dict
        دیکشنری با کلید نماد و مقدار یک دیکشنری warmup_dict
    final_dict = {
        "XAUUSD" : {"M1": 26, "M5": 20, "H1": 14},
        "EURUSD" : {"M10: 10, "H1": 12},
        "BITCOIN": {"M30: 14, "H4": 19},
        ...
        }
    """
    if cfg is None:
        cfg = load_config()
    
    final_dict: Dict[str, Dict[str, int]]= {}
    symbols = (cfg.get("features") or {}).get("symbols") or {} # دیکشنری نمادها را برمیگرداند

    for symbol in symbols.keys():
        specs = symbols[symbol].get("indicators") or []
        if not isinstance(specs, list):
            logger.warning(f"Indicators specs is not a list: {specs}")
            final_dict[symbol] = {}  # برگرداندن دیکشنری خالی به جای None برای جلوگیری از کرش
            continue

        warmup_dict = _compute_warmup_from_specs(specs)

        candles_list = symbols[symbol].get("candles") or []
        if candles_list:
            candles_std = mapping_tfs(candles_list, mode="values")
            for tf in candles_std:
                if tf not in warmup_dict:
                    warmup_dict[tf] = 0
                    logger.debug(f"Added {tf} with warmup=0 for {symbol} (from candles)")

        # مرتب‌سازی با check_tfs
        if warmup_dict:
            tfs = list(warmup_dict.keys())
            _, _, sorted_tfs = check_tfs(tfs[0], tfs)
            warmup_dict = {tf: warmup_dict[tf] for tf in sorted_tfs if tf in warmup_dict}
        final_dict[symbol] = warmup_dict

    # حذف نمادهایی که دیکشنری وارم-آپ آنها تهی است
    warmups_dicts = {sym: dic for sym, dic in final_dict.items() if dic != {}}
    
    # cfg["__warmups_dicts"] = final_dict
    return warmups_dicts


# =============================================================================
# TESTER: for "get_warmup_from_config_allsyms"
# =============================================================================
#
#                   تابع زیر منسوخ شده است
# def main():
#     path = Path(project_root() / "f01_config" / "config.yaml")
#     cfg = load_config((path), enable_env_override=True)
#     if cfg is None or cfg == {}:
#         return None

#     specs = (cfg.get("features") or {}).get("indicators" or {})

#     print("\n", "="*3, "specs", "="* (60-len("specs")))
#     print(specs)

#     warmups_dicts = get_warmup_from_config_allsyms()
    
#     print("\n", "="*3, "type(warmups_dicts)", "="* (60-len("type(warmups_dicts)")))
#     print(type(warmups_dicts))

#     print("\n", "="*3, "warmups_dicts", "="* (60-len("warmups_dicts")))
#     print(warmups_dicts)

# =============================================================================
# Run: python -m f02_utils.parse_warmups

# if __name__ == "__main__":
#     raise SystemExit(main())
