#!/usr/bin/env python3
"""
Bot-RL-3: Ultimate Orchestrator - Listener-based, Modular, Production-Ready
ارکستراتور نهایی ربات معاملاتی با معماری رویدادمحور، کاملاً ماژولار و توسعه‌پذیر

قابلیت‌ها:
- پشتیبانی از حالت‌های live, paper, shadow, backtest, train, optimize, evaluate
- ارتباط با DataHandler از طریق callback (بدون polling)
- ثبت شنونده (listener) برای لایه‌های بالایی مانند features, risk, execution
- دانلود داده تاریخی، بررسی سلامت، graceful shutdown
- کاملاً سازگار با فایل‌های موجود (market_data_engine, data_handler_D, mt5_data_loader_D)

نحوه استفاده:
    python main.py --mode live --symbols EURUSD --timeframes M1 H1
    python main.py --mode backtest --symbols EURUSD --base_tf M1 --start 2024-01-01 --end 2024-12-31
    python main.py --download --symbols EURUSD --lookback 5000
"""

from __future__ import annotations

import sys
import signal
import time
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
# SECTION 1: LOGGING SETUP (تابع جداگانه برای لاگینگ)
# =============================================================================

def setup_logging(cfg: Dict[str, Any]) -> None:
    """Configure logging based on config file"""
    log_cfg = cfg.get("logging", {})
    log_level = getattr(logging, log_cfg.get("level", "INFO").upper())
    log_format = log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    log_datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_file = log_cfg.get("file", "logs/Bot-RL-3.log")

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(log_format, log_datefmt))
    root_logger.addHandler(console)

    if log_cfg.get("file_enabled", True) and log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter(log_format, log_datefmt))
        root_logger.addHandler(fh)

    # Set level for project modules
    for mod in ["f03_data", "f04_features", "f02_utils"]:
        logging.getLogger(mod).setLevel(log_level)
    for lib in ["urllib3", "requests"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    logging.info("=" * 60)
    logging.info("Bot-RL-3 Ultimate Orchestrator started")
    logging.info("=" * 60)


# =============================================================================
# SECTION 2: HELPER CLASSES (ماژولار)
# =============================================================================

class ModeNotSupportedError(Exception):
    """Raised when an invalid mode is provided"""
    pass


class ComponentManager:
    """مدیریت مؤلفه‌های ربات (برای ثبت و دسترسی آسان)"""
    def __init__(self):
        self._components: Dict[str, Any] = {}

    def register(self, name: str, component: Any) -> None:
        self._components[name] = component

    def get(self, name: str) -> Any:
        return self._components.get(name)

    def list_all(self) -> List[str]:
        return list(self._components.keys())


# =============================================================================
# SECTION 3: MAIN ORCHESTRATOR CLASS (کامل و نهایی)
# =============================================================================

class BotOrchestrator:
    """
    ارکستراتور نهایی - تمام قابلیت‌های قبلی حفظ شده و قابلیت listener اضافه شده است.
    """

    def __init__(self, config_path: Optional[str] = None, mode: str = "live") -> None:
        self.config_path = config_path
        self.mode = mode.lower()
        self.cfg: Dict[str, Any] = {}
        self._running = False
        self.component_mgr = ComponentManager()

        # Listener system (جدید)
        self._data_listeners: List[Callable[[pd.DataFrame], None]] = []

        # Core components
        self.engine: Optional[MarketDataEngine] = None
        self.data_handler: Optional[DataHandler] = None
        self.downloader: Optional[MT5DataLoader_batch] = None

        # For batch processing (used in backtest/train modes)
        self._batch_df: Optional[pd.DataFrame] = None
        self._batch_index: int = 0

        self.logger = logging.getLogger(__name__)

    # ---------------------------------------------------------------
    # Configuration and Signal Handling (دست نخورده)
    # ---------------------------------------------------------------
    def load_configuration(self) -> None:
        try:
            self.cfg = load_config(self.config_path) if self.config_path else load_config()
            self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise

    def _setup_signal_handlers(self) -> None:
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    # ---------------------------------------------------------------
    # Listener Management (جدید، بدون حذف چیزی)
    # ---------------------------------------------------------------
    def register_data_listener(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """Register a listener to be called every time new data arrives (live mode)"""
        self._data_listeners.append(callback)
        self.logger.debug(f"Data listener registered. Total: {len(self._data_listeners)}")

    def _notify_data_listeners(self, df: pd.DataFrame) -> None:
        """Notify all registered listeners with the updated dataframe"""
        for cb in self._data_listeners:
            try:
                cb(df)
            except Exception as e:
                self.logger.exception(f"Listener callback error: {e}")

    # ---------------------------------------------------------------
    # Component Initialization (تمام مؤلفه‌های قبلی + اضافه شدن listener به DataHandler)
    # ---------------------------------------------------------------
    def _init_components(self) -> None:
        """Initialize all components based on mode (live/batch)"""
        # ----------------------------------------------
        # Live/Paper/Shadow modes
        # ----------------------------------------------
        if self.mode in ["live", "paper", "shadow"]:
            # 1. MarketDataEngine
            self.engine = MarketDataEngine(cfg=self.cfg)
            self.component_mgr.register("engine", self.engine)
            self.logger.info("MarketDataEngine initialized")

            # 2. DataHandler with EventBus
            self.data_handler = DataHandler(
                cfg=self.cfg,
                event_bus=self.engine.get_event_bus()
            )
            self.component_mgr.register("data_handler", self.data_handler)

            # 3. Set callback on DataHandler to get notified on new candles
            #    This requires a small addition to DataHandler (set_data_callback method)
            #    If not available, we will fallback to a lightweight thread (commented)
            if hasattr(self.data_handler, 'set_data_callback'):
                self.data_handler.set_data_callback(self._on_data_updated)
                self.logger.info("DataHandler callback registered (listener mode)")
            else:
                self.logger.warning("DataHandler does not have set_data_callback. "
                                    "Please add it or use polling fallback.")
                # Fallback: start a consumer thread inside orchestrator (polling, but kept only if needed)
                self._start_fallback_consumer()

            self.logger.info("DataHandler initialized")

        # ----------------------------------------------
        # Batch modes (backtest, train, optimize, evaluate)
        # ----------------------------------------------
        elif self.mode in ["backtest", "train", "optimize", "evaluate"]:
            self.data_handler = DataHandler(cfg=self.cfg, event_bus=None)
            self.component_mgr.register("data_handler", self.data_handler)
            self.logger.info(f"DataHandler initialized in batch mode for {self.mode}")

        # ----------------------------------------------
        # Downloader (optional, for CLI download command)
        # ----------------------------------------------
        if self.cfg.get("data_loader", {}).get("enabled", False):
            self.downloader = MT5DataLoader_batch(
                cfg=self.cfg,
                event_bus=self.engine.get_event_bus() if self.engine else None
            )
            self.component_mgr.register("downloader", self.downloader)
            self.logger.info("MT5DataLoader initialized")

    def _start_fallback_consumer(self) -> None:
        """Fallback consumer thread in case DataHandler lacks callback support."""
        import threading
        def consumer_loop():
            sub_id = self.engine.event_bus.subscribe()
            q = self.engine.event_bus.get_queue(sub_id)
            self.logger.info("Fallback consumer thread started")
            while self._running:
                try:
                    event = q.get(timeout=1.0)
                    if event and event.get("event_type") == "NEW_CANDLE":
                        # Force DataHandler to update its cache
                        self.data_handler.on_new_candle(event["payload"])
                        # Notify listeners
                        if self.data_handler._cached_df is not None:
                            self._notify_data_listeners(self.data_handler._cached_df)
                except Exception as e:
                    self.logger.debug(f"Fallback consumer error: {e}")
        t = threading.Thread(target=consumer_loop, daemon=True)
        t.start()

    # ---------------------------------------------------------------
    # Data update handler (called by DataHandler callback)
    # ---------------------------------------------------------------
    def _on_data_updated(self, df: pd.DataFrame) -> None:
        """Called automatically by DataHandler when new data arrives."""
        if self._running:
            self._notify_data_listeners(df)

    # ---------------------------------------------------------------
    # Public start method (جایگزین start قبلی بدون حذف)
    # ---------------------------------------------------------------
    def start(self, cli_args: argparse.Namespace) -> None:
        """Start the orchestrator based on mode and CLI arguments."""
        if self._running:
            self.logger.warning("Bot is already running")
            return

        self.load_configuration()
        self._setup_signal_handlers()
        self._init_components()

        try:
            if self.mode in ["live", "paper", "shadow"]:
                self._start_live(cli_args)
            elif self.mode in ["backtest", "train", "optimize", "evaluate"]:
                self._start_batch(cli_args)
            else:
                raise ModeNotSupportedError(f"Mode {self.mode} not supported")

        except Exception as e:
            self.logger.exception(f"Failed to start bot: {e}")
            self.shutdown()
            raise

    # ---------------------------------------------------------------
    # Live mode implementation (listener-based, no polling)
    # ---------------------------------------------------------------
    def _start_live(self, args: argparse.Namespace) -> None:
        symbols = args.symbols or self.cfg.get("download_defaults", {}).get("symbols", ["EURUSD"])
        timeframes = args.timeframes or self.cfg.get("download_defaults", {}).get("timeframes", ["M5"])
        poll_interval = self.cfg.get("executor", {}).get("poll_interval_sec", 2.0)

        self.engine.start(symbols=symbols, timeframes=timeframes, poll_interval_sec=poll_interval)
        # If DataHandler has its own consumer thread, start it
        if hasattr(self.data_handler, 'start_consuming'):
            self.data_handler.start_consuming()
        self._running = True
        self.logger.info(f"Bot started in {self.mode} mode (listener-based)")

        # Keep main thread alive (wait for shutdown signal)
        while self._running:
            time.sleep(3600)  # Long sleep, wake up by signal

    # ---------------------------------------------------------------
    # Batch mode implementation (unchanged from original)
    # ---------------------------------------------------------------
    def _start_batch(self, args: argparse.Namespace) -> None:
        symbol = args.symbols[0] if args.symbols else "EURUSD"
        base_tf = args.base_tf or "M1"
        timeframes = args.timeframes or [base_tf, "H1"]
        start_date = args.start
        end_date = args.end

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

        # Notify listeners with the whole dataframe (for batch processing)
        self._notify_data_listeners(df)
        self._running = False
        self.logger.info(f"Batch mode completed: {len(df)} rows processed")

    # ---------------------------------------------------------------
    # Shutdown and cleanup (همانند قبل)
    # ---------------------------------------------------------------
    def shutdown(self) -> None:
        self.logger.info("Shutting down orchestrator...")
        self._running = False
        if self.engine:
            self.engine.stop()
        if self.data_handler and hasattr(self.data_handler, 'stop_consuming'):
            self.data_handler.stop_consuming()
        self.logger.info("Orchestrator shutdown complete")

    # ---------------------------------------------------------------
    # Download historical data (کاملاً دست نخورده)
    # ---------------------------------------------------------------
    def download_historical_data(self,
                                  symbols: Optional[List[str]] = None,
                                  timeframes: Optional[List[str]] = None,
                                  lookback_bars: int = 1000) -> None:
        if not self.downloader:
            self.logger.warning("Downloader not initialized. Enable data_loader in config.")
            return
        self.logger.info("Starting historical data download...")
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
    # Health check (دست نخورده)
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
            status["components"]["data_handler"] = {
                "subscribed": self.data_handler._subscriber_id is not None
            }
        return status


# =============================================================================
# SECTION 4: COMMAND LINE INTERFACE (جدا شده)
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bot-RL-3 Ultimate Orchestrator")
    parser.add_argument("-c", "--config", type=str, help="Path to config.yaml")
    parser.add_argument("--mode", type=str, default="live",
                        choices=["live", "backtest", "train", "paper", "optimize", "evaluate", "shadow"],
                        help="Execution mode")
    parser.add_argument("--download", action="store_true", help="Download historical data and exit")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to trade or download")
    parser.add_argument("--timeframes", type=str, nargs="+", help="Timeframes")
    parser.add_argument("--base_tf", type=str, help="Base timeframe for batch modes")
    parser.add_argument("--lookback", type=int, default=1000, help="Number of bars to download")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD) for batch modes")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD) for batch modes")
    parser.add_argument("--health", action="store_true", help="Check health and exit")
    return parser.parse_args()


# =============================================================================
# SECTION 5: MAIN ENTRY POINT
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
            bot.download_historical_data(symbols=symbols, timeframes=timeframes, lookback_bars=args.lookback)
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