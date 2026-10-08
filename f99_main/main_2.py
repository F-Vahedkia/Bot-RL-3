#!/usr/bin/env python3
"""
Bot-RL-3: Main Orchestrator
ارکستراتور اصلی ربات معاملاتی با قابلیت توسعه ماژولار

# اجرای عادی ربات
    python main_2.py
# با مسیر config دلخواه
    python main_2.py -c my_config.yaml
# فقط دانلود داده تاریخی
    python main_2.py --download --symbols EURUSD XAUUSD --timeframes M5 H1 --lookback 5000
# بررسی سلامت
    python main_2.py --health
# با تنظیم لاگ DEBUG
    python main_2.py --log-level DEBUG
"""
# =============================================================================
#    IMPORTS
# =============================================================================
from __future__ import annotations

import sys
import signal
import time
import logging
import logging.config
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import argparse

# -------------------- Internal Imports -------------------
from f02_utils.config_loader import load_config
from f03_data.market_data_engine.market_data_engine_2 import MarketDataEngine
from f03_data.data_handler_F_2 import DataHandler
from f03_data.mt5_data_loader_E import MT5DataLoader_batch

# =============================================================================
#    LOGGER SETUP
# =============================================================================
def setup_logging(cfg: Dict[str, Any]) -> None:
    """تنظیمات logging بر اساس config فایل"""
    log_cfg = cfg.get("logging", {})
    
    # تنظیمات پیش‌فرض
    log_level = getattr(logging, log_cfg.get("level", "INFO").upper())
    log_format = log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    log_datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_file = log_cfg.get("file", "logs/Bot-RL-3.log")
    
    # ایجاد پوشه logs اگر وجود ندارد
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # تنظیم root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # حذف handlers قبلی
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, log_datefmt))
    root_logger.addHandler(console_handler)
    
    # File handler (اختیاری)
    if log_cfg.get("file_enabled", True) and log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format, log_datefmt))
        root_logger.addHandler(file_handler)
    
    # تنظیم loggers خاص
    logging.getLogger("f03_data").setLevel(log_level)
    logging.getLogger("f04_features").setLevel(log_level)
    logging.getLogger("f02_utils").setLevel(log_level)
    
    # کاهش noise از کتابخانه‌های خارجی
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    logging.info("=" * 60)
    logging.info("Bot-RL-3 Starting - Logging initialized")
    logging.info("=" * 60)


# =============================================================================
#    Main Orchestrator Class
# =============================================================================
class BotOrchestrator:
    """
    ارکستراتور اصلی ربات
    مسئولیت: هماهنگی تمام مؤلفه‌ها، مدیریت lifecycle، خطاپذیری
    """
    # ---------------------------------------------------------------
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Args:
            config_path: مسیر فایل config.yaml (اختیاری)
        """
        self.config_path = config_path
        self.cfg: Dict[str, Any] = {}
        self._running = False
        self._components: Dict[str, Any] = {}
        
        # مؤلفه‌های اصلی
        self.engine: Optional[MarketDataEngine] = None
        self.data_handler: Optional[DataHandler] = None
        self.downloader: Optional[MT5DataLoader_batch] = None
        
        self.logger = logging.getLogger(__name__)
    
    # ---------------------------------------------------------------
    def load_configuration(self) -> None:
        """بارگذاری پیکربندی از فایل"""
        try:
            self.cfg = load_config(self.config_path) if self.config_path else load_config()
            self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    # ---------------------------------------------------------------
    def _setup_signal_handlers(self) -> None:
        """تنظیم signal handlers برای graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    # ---------------------------------------------------------------
    def _init_components(self) -> None:
        """ایجاد و مقداردهی اولیه تمام مؤلفه‌ها"""
        
        # 1. Market Data Engine (با EventBus داخلی)
        self.engine = MarketDataEngine(cfg=self.cfg)
        self._components["engine"] = self.engine
        self.logger.info("MarketDataEngine initialized")
        
        # 2. DataHandler (با اتصال به EventBus)
        self.data_handler = DataHandler(
            cfg=self.cfg,
            event_bus=self.engine.get_event_bus()
        )
        self._components["data_handler"] = self.data_handler
        self.logger.info("DataHandler initialized and attached to EventBus")
        
        # 3. MT5DataLoader (برای دانلود داده‌های تاریخی - اختیاری)
        if self.cfg.get("data_loader", {}).get("enabled", False):
            self.downloader = MT5DataLoader_batch(
                cfg=self.cfg,
                event_bus=self.engine.get_event_bus()
            )
            self._components["downloader"] = self.downloader
            self.logger.info("MT5DataLoader initialized")
    
    # ---------------------------------------------------------------
    def start(self) -> None:
        """شروع تمام مؤلفه‌ها"""
        if self._running:
            self.logger.warning("Bot is already running")
            return
        
        self.logger.info("Starting Bot-RL-3...")
        
        try:
            # بارگذاری کانفیگ
            self.load_configuration()
            
            # تنظیم signal handlers
            self._setup_signal_handlers()
            
            # مقداردهی مؤلفه‌ها
            self._init_components()
            
            # دریافت تنظیمات از کانفیگ
            symbols = self.cfg.get("download_defaults", {}).get("symbols", ["EURUSD", "XAUUSD"])
            timeframes = self.cfg.get("download_defaults", {}).get("timeframes", ["M5", "H1"])
            poll_interval = self.cfg.get("executor", {}).get("poll_interval_sec", 2.0)
            
            # شروع Market Data Engine
            self.engine.start(
                symbols=symbols,
                timeframes=timeframes,
                poll_interval_sec=poll_interval
            )
            
            self._running = True
            self.logger.info(f"Bot-RL-3 started successfully with symbols={symbols}, timeframes={timeframes}")
            
        except Exception as e:
            self.logger.exception(f"Failed to start bot: {e}")
            self.shutdown()
            raise
    
    # ---------------------------------------------------------------
    def stop(self) -> None:
        """توقف graceful تمام مؤلفه‌ها"""
        self._running = False
        self.logger.info("Stopping Bot-RL-3...")
        
        # توقف به ترتیب معکوس
        if self.engine:
            try:
                self.engine.stop()
                self.logger.info("MarketDataEngine stopped")
            except Exception as e:
                self.logger.error(f"Error stopping engine: {e}")
        
        if self.data_handler:
            try:
                self.data_handler.stop_consuming()
                self.logger.info("DataHandler stopped")
            except Exception as e:
                self.logger.error(f"Error stopping data handler: {e}")
        
        self.logger.info("Bot-RL-3 stopped")
    
    # ---------------------------------------------------------------
    def shutdown(self) -> None:
        """Shutdown کامل با پاکسازی منابع"""
        self.logger.info("Initiating shutdown...")
        self.stop()
        self.logger.info("Shutdown complete")
    
    # ---------------------------------------------------------------
    def download_historical_data(self, 
                                  symbols: Optional[List[str]] = None,
                                  timeframes: Optional[List[str]] = None,
                                  lookback_bars: int = 1000) -> None:
        """دانلود داده‌های تاریخی (عملیات یکبار)"""
        if not self.downloader:
            self.logger.warning("Downloader not initialized. Set data_loader.enabled=true in config")
            return
        
        self.logger.info("Starting historical data download...")
        try:
            # اطمینان از اتصال
            if not self.downloader.conn.initialize():
                raise RuntimeError("MT5 connection failed")
            
            # ساخت و اجرای پلن
            plans = self.downloader.build_plan(
                symbols=symbols,
                timeframes=timeframes,
                lookback_bars=lookback_bars
            )
            results = self.downloader.run(plans)
            
            # گزارش نتایج
            for r in results:
                if "error" in r:
                    self.logger.error(f"Failed: {r['symbol']}/{r['timeframe']} - {r['error']}")
                else:
                    self.logger.info(f"Downloaded: {r['symbol']}/{r['timeframe']} - {r['rows_written']} rows")
            
            self.logger.info("Historical data download completed")
        except Exception as e:
            self.logger.exception(f"Download failed: {e}")
        finally:
            self.downloader.conn.shutdown()
    
    # ---------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        """گزارش سلامت سیستم"""
        status = {
            "running": self._running,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {}
        }
        
        if self.engine:
            status["components"]["engine"] = {
                "running": self.engine._running,
                "subscribers": self.engine.event_bus.subscriber_count()
            }
        
        if self.data_handler:
            status["components"]["data_handler"] = {
                "subscribed": self.data_handler._subscriber_id is not None
            }
        
        return status
    
    # ---------------------------------------------------------------
    def run(self) -> None:
        """حلقه اصلی اجرا (blocking)"""
        try:
            self.start()
            
            # حلقه اصلی - تا زمان دریافت signal ادامه می‌دهد
            while self._running:
                time.sleep(1)
                
                # در صورت نیاز: heartbeat یا health check دوره‌ای
                # if int(time.time()) % 60 == 0:
                #     self.logger.debug(f"Health: {self.health_check()}")
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.exception(f"Unexpected error in main loop: {e}")
        finally:
            self.shutdown()


# =============================================================================
#    Command Line Interface
# =============================================================================
def parse_args() -> argparse.Namespace:
    """پردازش پارامترهای خط فرمان"""
    parser = argparse.ArgumentParser(
        description="Bot-RL-3 - Trading Bot with MT5 Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to config.yaml file"
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download historical data and exit (without live trading)"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to download (e.g., EURUSD XAUUSD)"
    )
    
    parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        help="Timeframes to download (e.g., M5 H1)"
    )
    
    parser.add_argument(
        "--lookback",
        type=int,
        default=1000,
        help="Number of bars to download (default: 1000)"
    )
    
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check health status and exit"
    )
    
    return parser.parse_args()


# =============================================================================
#    Main Entry Point
# =============================================================================
def main() -> None:
    """نقطه ورود اصلی"""
    args = parse_args()
    
    # بارگذاری اولیه کانفیگ برای logging
    try:
        cfg = load_config(args.config) if args.config else load_config()
    except Exception:
        cfg = {}
    
    # راه‌اندازی logging
    setup_logging(cfg)
    
    # ایجاد orchestrator
    bot = BotOrchestrator(config_path=args.config)
    
    # حالت دانلود تاریخی
    if args.download:
        try:
            bot.load_configuration()
            bot._init_components()
            
            symbols = args.symbols or cfg.get("download_defaults", {}).get("symbols")
            timeframes = args.timeframes or cfg.get("download_defaults", {}).get("timeframes")
            
            bot.download_historical_data(
                symbols=symbols,
                timeframes=timeframes,
                lookback_bars=args.lookback
            )
        except Exception as e:
            logging.error(f"Download failed: {e}")
            sys.exit(1)
        return
    
    # حالت health check
    if args.health:
        try:
            bot.load_configuration()
            bot._init_components()
            health = bot.health_check()
            print(health)
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            sys.exit(1)
        return
    
    # حالت عادی: اجرای ربات
    bot.run()


if __name__ == "__main__":
    main()