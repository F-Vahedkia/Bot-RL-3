# f02_utils/logging_utils
# last reviewed at: 1405/05/17

# ---------------------------------------------------------------------------------------
""" Using Rule:

from f02_utils.logging_utils import setup_logging

# اعمال روی console_handler
console_handler.addFilter(ModuleFilter(["f03_data", "f04_features"]))
console_handler.addFilter(FunctionFilter(["build", "_load_raw"]))
"""

# ---------------------------------------------------------------------------------------
import sys
import logging
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------------------
class ModuleFilter(logging.Filter):
    def __init__(self, names: list):
        super().__init__()
        self.names = names

    def filter(self, record: logging.LogRecord) -> bool:
        return any(record.name.startswith(name) for name in self.names)
    
# ---------------------------------------------------------------------------------------
class FunctionFilter(logging.Filter):
    def __init__(self, func_names: list):
        super().__init__()
        self.func_names = func_names

    def filter(self, record: logging.LogRecord) -> bool:
        return record.funcName in self.func_names
    
# ---------------------------------------------------------------------------------------
def setup_logging(cfg: Dict[str, Any]) -> None:
    """
    تنظیمات سیستم لاگینگ را بر اساس کانفیگ انجام می‌دهد.

    عملیات انجام‌شده:
        1. خواندن تنظیمات از بخش logging در فایل config.yaml (سطح، فرمت، مسیر فایل).
        2. پاک کردن هندلرهای قبلی و تنظیم مجدد.
        3. افزودن console_handler برای نمایش لاگ‌ها در ترمینال (با اعمال فیلترها).
        4. افزودن file_handler برای ذخیره‌ی همه‌ی لاگ‌ها در فایل (بدون فیلتر).
        5. تنظیم سطح لاگ برای کتابخانه‌های داخلی پروژه و کتابخانه‌های خارجی (urllib3, requests).
        6. ثبت پیام شروع در لاگ.

    پارامترها:
        cfg: دیکشنری کانفیگ (بخش logging از آن استخراج می‌شود)
    """

    # تنظیمات logging بر اساس config فایل
    log_cfg     = cfg.get("logging", {})
    log_level   = getattr(logging, log_cfg.get("level", "INFO").upper())
    log_format  = log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    log_datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_file    = log_cfg.get("file", "logs/Bot-RL-3.log")

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, log_datefmt))

    # ===== فیلتر بر اساس نام فایل ===== start
    # با فعال بودن این دو سطر، فقط لاگهای مربوط به این فایل در ترمینال نوشته میشود
    # اما تمام لاگها همچنان در فایل لاگ نوشته میشوند
    target_module = ["f03_data.mt5_data_loader_E"]  # نام فایلهای مورد نظر
    console_handler.addFilter(ModuleFilter(target_module))
    # ===== فیلتر بر اساس نام فایل ===== end

    # ===== فیلتر بر اساس نام تابع ===== start
    target_functions = ["run", "_fetch_candles"]  # نام توابع مورد نظر
    console_handler.addFilter(FunctionFilter(target_functions))
    # =================================== end

    root_logger.addHandler(console_handler)

    if log_cfg.get("file_enabled", True) and log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format, log_datefmt))
        root_logger.addHandler(file_handler)

    for lib in ["f03_data", "f04_features", "f02_utils"]:
        logging.getLogger(lib).setLevel(log_level)
        logging.getLogger(lib).propagate = True

    for lib in ["urllib3", "requests"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    logging.info("=" * 60)
    logging.info("Bot-RL-3 Starting - Listener-based, No Polling")
    logging.info("=" * 60)

# ---------------------------------------------------------------------------------------
