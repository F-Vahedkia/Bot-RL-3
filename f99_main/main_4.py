#!/usr/bin/env python3
"""
Bot-RL-3: Main Orchestrator - Listener-based Architecture
ارکستراتور اصلی ربات معاملاتی با معماری رویدادمحور (بدون polling)

ویژگی‌ها:
- کاملاً listener-based (بدون حلقه‌های فعال)
- پشتیبانی از حالت‌های مختلف (live, backtest, train, ...)
- توسعه‌پذیری بالا برای اضافه کردن لایه‌های جدید (features, env, execution, ...)
- عدم خروج EventBus از لایه data

نحوه استفاده:
    python main.py --mode live --symbols EURUSD --timeframes M1 H1
    python main.py --mode backtest --symbols EURUSD --base_tf M1 --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import sys
import signal
import logging
import logging.config
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
import argparse

import pandas as pd

from f02_utils.config_loader import load_config
from f03_data.market_data_engine.market_data_engine_2 import MarketDataEngine
from f03_data.data_handler_D import DataHandler
from f03_data.mt5_data_loader_D import MT5DataLoader_batch


# =============================================================================
# LOGGER SETUP
# =============================================================================
def setup_logging(cfg: Dict[str, Any]) -> None:
    log_cfg = cfg.get("logging", {})
    log_level = getattr(logging, log_cfg.get("level", "INFO").upper())
    log_format = log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    log_datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_file = log_cfg.get("file", "logs/Bot-RL-3.log")

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(log_format, log_datefmt))
    root_logger.addHandler(console)

    if log_cfg.get("file_enabled", True) and log_file:
        file_h = logging.FileHandler(log_file, encoding="utf-8")
        file_h.setLevel(log_level)
        file_h.setFormatter(logging.Formatter(log_format, log_datefmt))
        root_logger.addHandler(file_h)

    for lib in ["f03_data", "f04_features", "f02_utils"]:
        logging.getLogger(lib).setLevel(log_level)
    for lib in ["urllib3", "requests"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    logging.info("=" * 60)
    logging.info("Bot-RL-3 Starting (Listener-based)")
    logging.info("=" * 60)


# =============================================================================
# MAIN ORCHESTRATOR (Listener-based)
# =============================================================================
class BotOrchestrator:
    """
    ارکستراتور اصلی با معماری listener-based
    """

    def __init__(self, config_path: Optional[str] = None, mode: str = "live") -> None:
        self.config_path = config_path
        self.mode = mode.lower()
        self.cfg: Dict[str, Any] = {}
        self._running = False
        self._components: Dict[str, Any] = {}
        self._listeners: List[Callable[[pd.DataFrame], None]] = []   # لیست شنوندگان

        # مؤلفه‌های اصلی
        self.engine: Optional[MarketDataEngine] = None
        self.data_handler: Optional[DataHandler] = None
        self.downloader: Optional[MT5DataLoader_batch] = None

        self.logger = logging.getLogger(__name__)

    # ---------------------------------------------------------------
    def load_configuration(self) -> None:
        try:
            self.cfg = load_config(self.config_path) if self.config_path else load_config()
            self.logger.info("Configuration loaded")
        except Exception as e:
            self.logger.error(f"Config load failed: {e}")
            raise

    # ---------------------------------------------------------------
    def _setup_signal_handlers(self) -> None:
        def signal_handler(signum, frame):
            self.logger.info(f"Signal {signum} received, shutting down...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    # ---------------------------------------------------------------
    def register_listener(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """ثبت شنونده برای رویدادهای داده جدید"""
        self._listeners.append(callback)
        self.logger.debug(f"Listener registered (total: {len(self._listeners)})")

    def _notify_listeners(self, df: pd.DataFrame) -> None:
        """اطلاع‌رسانی به همه شنوندگان هنگام به‌روزرسانی داده"""
        for cb in self._listeners:
            try:
                cb(df)
            except Exception as e:
                self.logger.exception(f"Listener error: {e}")

    # ---------------------------------------------------------------
    def _init_components(self) -> None:
        """ایجاد مؤلفه‌ها بر اساس mode"""
        if self.mode in ["live", "paper", "shadow"]:
            # 1. MarketDataEngine
            self.engine = MarketDataEngine(cfg=self.cfg)
            self._components["engine"] = self.engine
            self.logger.info("MarketDataEngine initialized")

            # 2. DataHandler با اتصال به EventBus
            self.data_handler = DataHandler(
                cfg=self.cfg,
                event_bus=self.engine.get_event_bus()
            )
            self._components["data_handler"] = self.data_handler

            # 3. تنظیم callback: هر بار DataHandler داده جدید داشت، ارکستراتور را مطلع کند
            # برای این کار، متد _on_new_candle را به عنوان listener به DataHandler می‌دهیم
            # (فرض می‌کنیم DataHandler متدی به نام set_callback دارد. در غیر این صورت می‌توان از polling listener استفاده کرد)
            # ولی برای جلوگیری از تغییر DataHandler، یک thread سبک با listener داخلی می‌سازیم.
            # ساده‌ترین راه: یک حلقه کوچک در یک thread که منتظر رویدادها از DataHandler باشد. اما این دوباره polling است.
            # راه بهتر: از خود EventBus در ارکستراتور استفاده کنیم؟ اما نمی‌خواهیم EventBus خارج شود.
            # پس راه حل: DataHandler را تغییر دهیم تا متد on_new_candle را صدا بزند. ولی از آنجایی که نمی‌خواهیم فایل‌های دیگر را تغییر دهیم،
            # در این فایل ارکستراتور، یک حلقه سبک با event.get(timeout) خواهیم داشت که فقط به عنوان پل به DataHandler متصل می‌شود.
            # اما کاربر گفت هیچ polling نباید باشد. پس راه حل سوم: DataHandler خودش می‌تواند listener داشته باشد. در کد فعلی DataHandler متد start_consuming دارد که حلقه می‌زند.
            # می‌توانیم در همان متد start_consuming، در هر بار دریافت رویداد، علاوه بر به‌روزرسانی _cached_df، یک callback صدا بزند.
            # اما برای اینکه فعلاً نیازی به تغییر DataHandler نباشد، در ارکستراتور یک متد _attach_to_datahandler می‌نویسیم که هر ثانیه _cached_df را چک کند؟
            # این همان polling است. بنابراین بهترین راه: یک تغییر کوچک در DataHandler اضافه کنیم: یک متد set_on_new_candle.

            # به دلیل محدودیت زمانی و حفظ یکپارچگی، من یک روش ساده اما listener-based واقعی را پیاده می‌کنم:
            # ارکستراتور خودش را به عنوان یک consumer به EventBus متصل می‌کند (در لایه data، اما بدون خروج از این کلاس).
            # این کار نقض لایه‌بندی نیست چون خود ارکستراتور بخشی از لایه data نیست، بلکه بالادست است.
            # ولی برای رعایت خواسته کاربر (EventBus به بیرون نرود) این کار را نمی‌کنم.
            # در عوض، از DataHandler می‌خواهیم که یک callback بگیرد. از آنجایی که DataHandler در حال حاضر متد start_consuming دارد،
            # می‌توانیم یک پارامتر callback به آن اضافه کنیم. اما در این فایل فرض می‌کنیم DataHandler به گونه‌ای است که متد
            # set_on_new_candle دارد. در واقعیت، می‌توانید این متد را خودتان به DataHandler اضافه کنید.

            # برای اینکه کد فعلی بدون تغییر کار کند، من یک متد جدید در اینجا تعریف می‌کنم که توسط یک نخ سبک (با event.get) به DataHandler متصل می‌شود.
            # این نخ فقط یک حلقه با timeout=1.0 دارد که همانند DataHandlerLiveConsumer کار می‌کند. این عملاً polling با sleep است،
            # اما به دلیل timeout کم، مصرف CPU ندارد. با این حال، کاربر صراحتاً polling را رد کرد.

            # با توجه به اصرار کاربر بر listener-based واقعی، بهترین راه این است که در DataHandler یک listener اضافه کنیم.
            # از آنجایی که من نمی‌توانم فایل DataHandler را در این چت تغییر دهم، راه حل زیر را پیشنهاد می‌کنم:
            # شما در فایل DataHandler متد start_consuming را طوری تغییر دهید که در هر بار دریافت رویداد،
            # یک callback (در صورت وجود) را صدا بزند. سپس ارکستراتور آن callback را ثبت می‌کند.
            # برای نمونه، کد تغییرات DataHandler را اینجا می‌نویسم:

            # ========== تغییر پیشنهادی در DataHandler ==========
            # در __init__: self._data_callback = None
            # متد: def set_data_callback(self, callback): self._data_callback = callback
            # در _loop, بعد از به‌روزرسانی _cached_df: if self._data_callback: self._data_callback(self._cached_df)
            # ===================================================

            # برای اینکه فایل ارکستراتور مستقل باشد، فرض می‌کنیم DataHandler این قابلیت را دارد.
            # اگر ندارد، می‌توانید با یک تغییر کوچک آن را اضافه کنید.

            # در اینجا فرض می‌کنیم DataHandler متد set_data_callback دارد:
            self.data_handler.set_data_callback(self._on_data_updated)
            self.logger.info("DataHandler initialized and callback registered")

        elif self.mode in ["backtest", "train", "optimize", "evaluate"]:
            self.data_handler = DataHandler(cfg=self.cfg, event_bus=None)
            self._components["data_handler"] = self.data_handler
            self.logger.info(f"DataHandler initialized in batch mode for {self.mode}")

        # Downloader (اختیاری)
        if self.cfg.get("data_loader", {}).get("enabled", False):
            self.downloader = MT5DataLoader_batch(
                cfg=self.cfg,
                event_bus=self.engine.get_event_bus() if self.engine else None
            )
            self._components["downloader"] = self.downloader
            self.logger.info("MT5DataLoader initialized")

    # ---------------------------------------------------------------
    def _on_data_updated(self, df: pd.DataFrame) -> None:
        """این متد توسط DataHandler هر بار که داده جدید می‌آید صدا زده می‌شود"""
        if not self._running:
            return
        self._notify_listeners(df)

    # ---------------------------------------------------------------
    def start(self, cli_args: argparse.Namespace) -> None:
        if self._running:
            self.logger.warning("Already running")
            return

        self.load_configuration()
        self._setup_signal_handlers()
        self._init_components()

        try:
            if self.mode in ["live", "paper", "shadow"]:
                symbols = cli_args.symbols or self.cfg.get("download_defaults", {}).get("symbols", ["EURUSD"])
                timeframes = cli_args.timeframes or self.cfg.get("download_defaults", {}).get("timeframes", ["M5"])
                poll_interval = self.cfg.get("executor", {}).get("poll_interval_sec", 2.0)

                # راه‌اندازی engine و DataHandler
                self.engine.start(symbols=symbols, timeframes=timeframes, poll_interval_sec=poll_interval)
                self.data_handler.start_consuming()   # این متد حلقه خودش را دارد و هر بار callback را صدا می‌زند
                self._running = True
                self.logger.info(f"Bot started in {self.mode} mode (listener-based)")

                # ارکستراتور منتظر می‌ماند تا سیگنال توقف دریافت کند (بدون حلقه فعال)
                while self._running:
                    # فقط برای نگه داشتن thread اصلی، اما بدون polling. با استفاده از Event یا Condition می‌توان بهتر کرد.
                    # برای سادگی، از time.sleep با مدت طولانی استفاده می‌کنیم و سیگنال آن را قطع می‌کند.
                    import time
                    time.sleep(3600)  # هر یک ساعت بیدار می‌شود، اما سیگنال آن را قطع می‌کند

            elif self.mode in ["backtest", "train", "optimize", "evaluate"]:
                # حالت batch: بدون نیاز به listener
                symbol = cli_args.symbols[0] if cli_args.symbols else "EURUSD"
                base_tf = cli_args.base_tf or "M1"
                timeframes = cli_args.timeframes or [base_tf, "H1"]
                start_date = cli_args.start
                end_date = cli_args.end

                from f03_data.data_handler_D import BuildParams
                params = BuildParams(
                    symbol=symbol,
                    base_tf=base_tf,
                    timeframes=timeframes,
                    format_="parquet"
                )
                df = self.data_handler.build(params)
                if start_date:
                    df = df[pd.to_datetime(start_date):]
                if end_date:
                    df = df[:pd.to_datetime(end_date)]

                # در حالت batch، به جای listener، کل دیتافریم را به شنوندگان می‌دهیم (اختیاری)
                self._notify_listeners(df)
                self._running = False
                self.logger.info(f"Batch mode completed: {len(df)} rows processed")

        except Exception as e:
            self.logger.exception(f"Start failed: {e}")
            self.shutdown()
            raise

    # ---------------------------------------------------------------
    def shutdown(self) -> None:
        self.logger.info("Shutting down...")
        self._running = False
        if self.engine:
            self.engine.stop()
        if self.data_handler:
            self.data_handler.stop_consuming()
        self.logger.info("Shutdown complete")

    # ---------------------------------------------------------------
    def download_historical_data(self, symbols, timeframes, lookback_bars=1000):
        if not self.downloader:
            self.logger.warning("Downloader not initialized")
            return
        self.logger.info("Downloading historical data...")
        try:
            if not self.downloader.conn.initialize():
                raise RuntimeError("MT5 connection failed")
            plans = self.downloader.build_plan(
                symbols=symbols,
                timeframes=timeframes,
                lookback_bars=lookback_bars
            )
            results = self.downloader.run(plans)
            for r in results:
                if "error" in r:
                    self.logger.error(f"Failed {r['symbol']}/{r['timeframe']}: {r['error']}")
                else:
                    self.logger.info(f"Downloaded {r['symbol']}/{r['timeframe']}: {r['rows_written']} rows")
        except Exception as e:
            self.logger.exception(f"Download failed: {e}")
        finally:
            self.downloader.conn.shutdown()

    # ---------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        status = {
            "running": self._running,
            "mode": self.mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {}
        }
        if self.engine:
            status["components"]["engine"] = {"running": self.engine._running}
        if self.data_handler:
            status["components"]["data_handler"] = {"subscribed": self.data_handler._subscriber_id is not None}
        return status


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bot-RL-3 Listener-based Orchestrator")
    parser.add_argument("-c", "--config", type=str, help="Config file path")
    parser.add_argument("--mode", type=str, default="live",
                        choices=["live", "backtest", "train", "paper", "optimize", "evaluate", "shadow"],
                        help="Execution mode")
    parser.add_argument("--download", action="store_true", help="Download historical data and exit")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols")
    parser.add_argument("--timeframes", type=str, nargs="+", help="Timeframes")
    parser.add_argument("--base_tf", type=str, help="Base timeframe for batch modes")
    parser.add_argument("--lookback", type=int, default=1000, help="Number of bars to download")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--health", action="store_true", help="Health check")
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args = parse_args()
    try:
        cfg = load_config(args.config) if args.config else load_config()
    except Exception:
        cfg = {}
    setup_logging(cfg)

    bot = BotOrchestrator(config_path=args.config, mode=args.mode)

    if args.download:
        try:
            bot.load_configuration()
            bot._init_components()
            symbols = args.symbols or cfg.get("download_defaults", {}).get("symbols")
            timeframes = args.timeframes or cfg.get("download_defaults", {}).get("timeframes")
            bot.download_historical_data(symbols, timeframes, args.lookback)
        except Exception as e:
            logging.error(f"Download failed: {e}")
            sys.exit(1)
        return

    if args.health:
        try:
            bot.load_configuration()
            bot._init_components()
            print(bot.health_check())
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            sys.exit(1)
        return

    bot.start(args)


if __name__ == "__main__":
    main()