"""
Run: python -m main_6_2_orch

Bot-RL-3: Main Orchestrator - Unified with Mode Support (Listener-based, No Polling)
ارکستراتور اصلی ربات معاملاتی با پشتیبانی از حالت‌های مختلف و معماری رویدادمحور بدون polling

نحوه استفاده:
    python main_3_orch.py --mode live --symbols EURUSD --timeframes M5
    python main_3_orch.py --mode backtest --symbols EURUSD --base_tf M1 --start 2024-01-01 --end 2024-12-31
    python main_3_orch.py --download --symbols EURUSD --lookback 5000
    python main_3_orch.py --health
"""
# Run: python -m main_6_2_orch --mode `train` --symbols XAUUSD_I --base_tf H1 --timeframes H1 H4 D1
# Run: python -m main_6_2_orch --mode train --symbols XAUUSD_I --base_tf H1 --timeframes H1 H4 D1

# Run: python -m main_6_2_orch --download --symbols XAUUSD_I --timeframes M1 M2 M4 M20 H1 H4

# main_6_2_orch.py
# =============================================================================
#    IMPORTS
# =============================================================================
from __future__ import annotations

import sys
import signal
import threading          # ✅ اضافه شد برای _stop_event
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import argparse
import pandas as pd

# -------------------- Internal Imports -------------------
from f02_utils.config_loader import load_config
from f03_data.market_data_engine.market_data_engine_2 import MarketDataEngine
from f03_data.data_handler_F_2 import DataHandler, BuildParams
from f03_data.mt5_data_loader_E import MT5DataLoader_batch

from f04_features.feature_B_bootstrap import build_feature_system
from f04_features.feature_B_graph import FeatureGraph
from f04_features.observation_B_builder import ObservationBuilder

from f02_utils.logging_utils import setup_logging

# =============================================================================
#    DATA SOURCE ABSTRACTION (توسعه‌پذیری)
# =============================================================================
class DataSource:
    """کلاس پایه برای تمام منابع داده"""
    def get_next_candle(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def has_next(self) -> bool:
        raise NotImplementedError

    def close(self):
        pass


class LiveDataSource(DataSource):
    """منبع داده زنده از طریق EventBus (5گانه data-provider)"""
    def __init__(self, engine: MarketDataEngine, data_handler: DataHandler):
        self.engine = engine
        self.data_handler = data_handler
        self._running = True

    def get_next_candle(self) -> Optional[Dict[str, Any]]:
        if self.data_handler._cached_df is not None and not self.data_handler._cached_df.empty:
            latest = self.data_handler._cached_df.iloc[-1].to_dict()
            return latest
        return None

    def has_next(self) -> bool:
        return self._running

    def close(self):
        self.engine.stop()
        self.data_handler.stop_consuming()


class BatchDataSource(DataSource):
    """منبع داده بچ (فایل‌های Parquet/CSV) برای بک‌تست و آموزش"""
    def __init__(self, data_handler: DataHandler, symbol: str, base_tf: str, timeframes: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.data_handler = data_handler
        self.symbol = symbol
        self.base_tf = base_tf
        self.timeframes = timeframes
        self.df = None
        self._index = 0
        self._load_data(start_date, end_date)

    def _load_data(self, start_date: Optional[str], end_date: Optional[str]):
        params = BuildParams(
            symbol=self.symbol,
            base_tf=self.base_tf,
            timeframes=self.timeframes,
            format_="parquet"
        )
        self.df = self.data_handler.build(params)
        if start_date:
            self.df = self.df[pd.to_datetime(start_date).tz_localize('UTC'):]
        if end_date:
            self.df = self.df[:pd.to_datetime(end_date).tz_localize('UTC')]
        self._index = 0
        # print(f" ====> shape of df is ===== {self.df.shape}")     # for debug
        # print(list(self.df.columns))                              # for debug

    def get_next_candle(self) -> Optional[Dict[str, Any]]:
        if self._index >= len(self.df):
            return None
        row = self.df.iloc[self._index]
        self._index += 1
        result = row.to_dict()
        result['time'] = row.name   # ایندکس را به عنوان ستون 'time' اضافه میکند
        return result

    def has_next(self) -> bool:
        return self._index < len(self.df)

    def close(self):
        pass


# =============================================================================
#    MAIN ORCHESTRATOR (Listener-based, No Polling)
# =============================================================================
class BotOrchestrator:
    """
    ارکستراتور اصلی ربات با پشتیبانی از حالت‌های مختلف و معماری رویدادمحور (بدون polling)
    """
    def __init__(self, config_path: Optional[str] = None, mode: str = "live") -> None:
        self.config_path = config_path
        self.mode = mode.lower()
        self.cfg: Dict[str, Any] = {}
        self._running = False
        self._components: Dict[str, Any] = {}

        # ✅ listener system (جدید)
        self._data_listeners: List[Callable[[pd.DataFrame], None]] = []
        self._stop_event = threading.Event()   # برای نگه‌داشتن نخ اصلی بدون polling

        # مؤلفه‌های اصلی
        self.engine: Optional[MarketDataEngine] = None
        self.data_handler: Optional[DataHandler] = None
        self.downloader: Optional[MT5DataLoader_batch] = None
        self.data_source: Optional[DataSource] = None

        self.logger = logging.getLogger(__name__)

        self.feature_engine = None
        self.feature_graph = None
        self.obs_builder = None
        self.feature_specs = []
    
    # ---------------------------------------------------------------
    def load_configuration(self) -> None:
        try:
            self.cfg = load_config(self.config_path) if self.config_path else load_config()
            self.logger.info("Configuration loaded successfully")
            
            # مقداردهی feature system
            feature_system = build_feature_system(self.config_path)
            self.feature_engine = feature_system.get_engine()
            self.feature_store = feature_system.get_store()
            self.feature_cache = feature_system.get_cache()
            
            # خواندن specs از config (پیش‌فرض خالی)
            self.feature_specs = self.cfg.get("features", {}).get("indicators", [])
            if not self.feature_specs:
                self.logger.warning("No feature specs found in config")
            self.feature_graph = FeatureGraph(self.feature_specs)
            self.obs_builder = ObservationBuilder(self.cfg)
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    # ---------------------------------------------------------------
    def _setup_signal_handlers(self) -> None:
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    # ---------------------------------------------------------------
    # ✅ Listener management (جدید)
    # ---------------------------------------------------------------
    def register_data_listener(self, callback: Callable[[pd.DataFrame], None]) -> None:
        """ثبت شنونده برای دریافت داده‌های جدید (لایه features, risk, ...)"""
        self._data_listeners.append(callback)
        self.logger.debug(f"Data listener registered. Total: {len(self._data_listeners)}")


    def _notify_data_listeners(self, df: pd.DataFrame) -> None:
        for cb in self._data_listeners:
            try:
                cb(df)
            except Exception as e:
                self.logger.exception(f"Listener callback error: {e}")


    def _on_data_updated(self, df: pd.DataFrame) -> None:
        if not self._running or self.feature_engine is None:
            return
        features_df = self.feature_engine.execute(df, self.feature_specs, mode=self.mode)
        obs_df = self.obs_builder.build(features_df, self.feature_graph)
        # حالا obs_df را به استراتژی بدهید (اختیاری)
        self.logger.debug(f"Observation ready: {obs_df.shape}")

    # ---------------------------------------------------------------
    def _init_components(self) -> None:
        """ایجاد مؤلفه‌ها بر اساس mode"""
        # برای حالت‌های زنده (live, paper, shadow) نیاز به MarketDataEngine داریم
        if self.mode in ["live", "paper", "shadow"]:
            self.engine = MarketDataEngine(cfg=self.cfg)
            self._components["engine"] = self.engine
            self.logger.info("MarketDataEngine initialized (live mode)")

            self.data_handler = DataHandler(
                cfg=self.cfg,
                event_bus=self.engine.get_event_bus()
            )
            self._components["data_handler"] = self.data_handler
            self.logger.info("DataHandler initialized and attached to EventBus")

            # ✅======== اتصال callback برای دریافت داده جدید (بدون polling)
            if hasattr(self.data_handler, 'set_data_callback'):
                self.data_handler.set_data_callback(self._on_data_updated)
                self.logger.info("DataHandler callback registered (listener mode)")
            else:
                self.logger.warning("DataHandler has no set_data_callback. Please add it to enable listener mode.")
            # ==========
            self.data_source = LiveDataSource(self.engine, self.data_handler)

        # برای حالت‌های بچ (backtest, train, optimize, evaluate)
        elif self.mode in ["train", "backtest", "optimize", "evaluate"]:
            # نیازی به MarketDataEngine نیست، مستقیماً از DataHandler بچ استفاده می‌کنیم
            self.data_handler = DataHandler(cfg=self.cfg, event_bus=None)
            self._components["data_handler"] = self.data_handler
            # بعداً در اجرا، پارامترهای build از خط فرمان گرفته می‌شود
            self.logger.info(f"DataHandler initialized in batch mode for {self.mode}")

        # برای دانلود (دستور جداگانه)
        # اگر در کانفیگ symbols و timeframes تعریف شده باشند، دانلودر را فعال کن
        dl_cfg = self.cfg.get("download_defaults", {})
        if dl_cfg.get("symbols") and dl_cfg.get("timeframes"):
            self.downloader = MT5DataLoader_batch(
                cfg=self.cfg,
            )
            self._components["downloader"] = self.downloader
            self.logger.info("MT5DataLoader initialized")

    # ---------------------------------------------------------------
    def start_live(self, symbols: List[str], timeframes: List[str], poll_interval: float = 2.0):
        """شروع حالت live/paper/shadow"""
        self.engine.start(symbols=symbols, timeframes=timeframes, poll_interval_sec=poll_interval)
        self.data_handler.start_consuming()  # شروع مصرف رویدادها (نخ خودش را دارد)
        self._running = True


    def start_batch(self, symbol: str, base_tf: str, timeframes: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None):
        self.data_source = BatchDataSource(self.data_handler, symbol, base_tf, timeframes, start_date, end_date)
        self._running = True
        self.logger.info(f"Batch mode started: {len(self.data_source.df)} rows")

    # ---------------------------------------------------------------
    # ❌ متد run_live_loop حذف شد (polling با time.sleep حذف گردید)
    # ---------------------------------------------------------------
    def run_batch_loop(self):
        if self.feature_engine is None:
            return
        all_candles = []
        while self._running and self.data_source.has_next():
            candle = self.data_source.get_next_candle()
            if candle:
                all_candles.append(candle)
            else:
                break
        if all_candles:
            df = pd.DataFrame(all_candles)
            # تنظیم اندیس زمانی از ستون 'time'
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time', drop=True)

            # ✅ (اختیاری) بررسی ثابت بودن H4_close در بازه‌های ۴ ساعته
            self.logger.info(f"Sample H4_close: {df['H4_close'].iloc[:20].tolist()}")

            features_df = self.feature_engine.execute(df, self.feature_specs, mode=self.mode)
            
            # ===== ذخیره‌سازی دیتافریم فیچرها ================================ start
            # این قسمت برای برنامه اصلی نیست و فقط برای بیرون کشیدن دیتافریم حاوی فیچرهه برای بررسی صحت محاسبات است
            if self.mode == "train" and features_df is not None and not features_df.empty:
                # ساخت نام فایل بر اساس نماد و تایم‌فریم‌ها
                symbol = getattr(self.data_source, 'symbol', 'unknown')
                base_tf = getattr(self.data_source, 'base_tf', 'unknown')
                tfs_str = '_'.join(getattr(self.data_source, 'timeframes', ['unknown']))
                
                # مسیر ذخیره در دایرکتوری فعلی (می‌توانی به دایرکتوری دلخواه تغییر دهی)
                path = self.cfg.get("paths", {}).get("feature_dir", ".")
                Path(path).mkdir(parents=True, exist_ok=True)
                save_path = f"{path}/features_{symbol}_{base_tf}_{tfs_str}.parquet"

                # ذخیره با فرمت parquet (سریع و فشرده)
                features_df.to_parquet(save_path, index=True)
                self.logger.info(f"✅ Features saved to: {save_path} (shape: {features_df.shape})")
                
                # اگر csv هم می‌خواهی، این خط را اضافه کن:
                features_df.to_csv(save_path.replace('.parquet', '.csv'), index=True)
            # ===== ذخیره‌سازی دیتافریم فیچرها ================================ end

            obs_df = self.obs_builder.build(features_df, self.feature_graph)
            # پردازش obs_df
        self._running = False

    # ---------------------------------------------------------------
    def start(self, cli_args: argparse.Namespace) -> None:
        """شروع بر اساس mode و پارامترهای خط فرمان"""
        if self._running:
            self.logger.warning("Bot is already running")
            return

        self.load_configuration()
        self._setup_signal_handlers()
        self._init_components()

        try:
            if self.mode in ["live", "paper", "shadow"]:
                
                symbols = cli_args.symbols or self.cfg.get("download_defaults", {}).get("symbols", ["EURUSD"])
                
                timeframes = cli_args.timeframes or self.cfg.get("download_defaults", {}).get("timeframes", ["M5"])
                
                poll_interval = self.cfg.get("executor", {}).get("poll_interval_sec", 2.0)
                self.start_live(symbols, timeframes, poll_interval)

                # ✅ به جای حلقه polling، نخ اصلی منتظر سیگنال می‌ماند (بدون مصرف CPU)
                self.logger.info("Entering listener-based wait (no polling). Press Ctrl+C to stop.")
                self._stop_event.wait()

            elif self.mode in ["train", "backtest", "optimize", "evaluate"]:

                symbol = (cli_args.symbols[0] if cli_args.symbols else "EURUSD")

                base_tf = cli_args.base_tf or "M5"
                timeframes = cli_args.timeframes or [base_tf, "H1"]

                start_date = cli_args.start
                end_date = cli_args.end
                
                self.start_batch(symbol, base_tf, timeframes, start_date, end_date)
                self.run_batch_loop()

            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

        except Exception as e:
            self.logger.exception(f"Failed to start bot: {e}")
            self.shutdown()
            raise

    # ---------------------------------------------------------------
    def stop(self) -> None:
        self._running = False
        self.logger.info("Stopping Bot-RL-3...")
        if self.engine:
            self.engine.stop()
        # if self.data_handler:
        #     self.data_handler.stop_consuming()
        if self.data_source:
            self.data_source.close()
        self._stop_event.set()   # آزاد کردن نخ اصلی
        self.logger.info("Bot-RL-3 stopped")

    
    def shutdown(self) -> None:
        self.logger.info("Initiating shutdown...")
        self.stop()
        self.logger.info("Shutdown complete")

    # ---------------------------------------------------------------
    def download_historical_data_old1(self,
                                  symbols: Optional[List[str]] = None,
                                  timeframes: Optional[List[str]] = None,
                                  lookback_bars: Optional [int] = None) -> None:
        if not self.downloader:
            self.logger.warning("Downloader not initialized.")
            return
        self.logger.info("Starting historical data download...")
        
        if lookback_bars is None:
            lookback_bars = self.cfg.get("download_defaults", {}).get("lookback_bars", 1000)

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
                    self.logger.error(f"Failed: {r['symbol']}/{r['timeframe']} - {r['error']}")
                else:
                    self.logger.info(f"Downloaded: {r['symbol']}/{r['timeframe']} - {r['rows_written']} rows")
            self.logger.info("Historical data download completed")
        except Exception as e:
            self.logger.exception(f"Download failed: {e}")
        finally:
            self.downloader.conn.shutdown()

    def download_historical_data_old2(self,
                                  symbols: Optional[List[str]] = None,
                                  timeframes: Optional[List[str]] = None,
                                  lookback_bars: Optional[int] = None) -> None:
        if not self.downloader:
            self.logger.warning("Downloader not initialized.")
            return
        
        # خواندن از کانفیگ اگر در خط فرمان داده نشده باشد
        dl_cfg = self.cfg.get("download_defaults", {})
        if not symbols:
            symbols = dl_cfg.get("symbols", [])
        if not timeframes:
            timeframes = dl_cfg.get("timeframes", [])
        if lookback_bars is None:
            lookback_bars = dl_cfg.get("lookback_bars", 1000)
        
        self.logger.info("Starting historical data download...")
        self.logger.info(f"Parameters: symbols={symbols}, timeframes={timeframes}, lookback_bars={lookback_bars}")
        
        try:
            if not self.downloader.conn.initialize():
                raise RuntimeError("MT5 connection failed")
            
            # ساخت پلن با استفاده از کانفیگ
            plans = self.downloader.build_plan(
                symbols=symbols,
                timeframes=timeframes,
                lookback_bars=lookback_bars,
                date_from=dl_cfg.get("date_from"),
                date_to=dl_cfg.get("date_to")
            )
            
            results = self.downloader.run(plans)
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

    def download_historical_data_old3(self,
                                  symbols: Optional[List[str]] = None,
                                  timeframes: Optional[List[str]] = None,
                                  lookback_bars: Optional[int] = None) -> None:
        if not self.downloader:
            self.logger.warning("Downloader not initialized.")
            return
        
        dl_cfg = self.cfg.get("download_defaults", {})
        if not symbols:
            symbols = dl_cfg.get("symbols", [])
        if not timeframes:
            timeframes = dl_cfg.get("timeframes", [])
        if lookback_bars is None:
            lookback_bars = dl_cfg.get("lookback_bars", 1000)
        
        self.logger.info("Starting historical data download...")
        self.logger.info(f"Parameters: symbols={symbols}, timeframes={timeframes}, lookback_bars={lookback_bars}")
        
        try:
            if not self.downloader.conn.initialize():
                raise RuntimeError("MT5 connection failed")
            
            # تبدیل تاریخ به فرمت ISO
            date_from = dl_cfg.get("date_from")
            if date_from and isinstance(date_from, str) and len(date_from) == 10:
                date_from = f"{date_from}T00:00:00Z"
            
            date_to = dl_cfg.get("date_to")
            if date_to and isinstance(date_to, str) and date_to.lower() == "now":
                date_to = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
            plans = self.downloader.build_plan(
                symbols=symbols,
                timeframes=timeframes,
                lookback_bars=lookback_bars,
                date_from=date_from,
                date_to=date_to
            )
            
            results = self.downloader.run(plans)
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

    def download_historical_data(self,
                                  symbols: Optional[List[str]] = None,
                                  timeframes: Optional[List[str]] = None,
                                  lookback_bars: Optional[int] = None) -> None:
        if not self.downloader:
            self.logger.warning("Downloader not initialized.")
            return
        
        dl_cfg = self.cfg.get("download_defaults", {})
        if not symbols:
            symbols = dl_cfg.get("symbols", [])
        if not timeframes:
            timeframes = dl_cfg.get("timeframes", [])
        if lookback_bars is None:
            lookback_bars = dl_cfg.get("lookback_bars", 1000)
        
        self.logger.info("Starting historical data download...")
        self.logger.info(f"Parameters: symbols={symbols}, timeframes={timeframes}, lookback_bars={lookback_bars}")
        
        try:
            if not self.downloader.conn.initialize():
                raise RuntimeError("MT5 connection failed")
            
            # ✅ تبدیل تاریخ به فرمت ISO (قبل از ارسال به build_plan)
            date_from = dl_cfg.get("date_from")
            if date_from and isinstance(date_from, str):
                if len(date_from) == 10:  # YYYY-MM-DD
                    date_from = f"{date_from}T00:00:00Z"
                elif date_from.lower() == "now":
                    date_from = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                # اگر فرمت دیگری داشت، همان را بفرست
            
            date_to = dl_cfg.get("date_to")
            if date_to and isinstance(date_to, str):
                if date_to.lower() == "now":
                    date_to = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                elif len(date_to) == 10:
                    date_to = f"{date_to}T23:59:59Z"
            
            self.logger.info(f"Converted date_from={date_from}, date_to={date_to}")
            
            plans = self.downloader.build_plan(
                symbols=symbols,
                timeframes=timeframes,
                lookback_bars=lookback_bars,
                date_from=date_from,
                date_to=date_to
            )
            
            results = self.downloader.run(plans)
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
        status = {
            "running": self._running,
            "mode": self.mode,
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


# =============================================================================
#    COMMAND LINE INTERFACE
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bot-RL-3 - Unified Trading Bot (Listener-based)")
    parser.add_argument("-c", "--config", type=str, help="Path to config.yaml")
    parser.add_argument("--mode", type=str, default="live",
                        choices=["live", "backtest", "train", "paper", "optimize", "evaluate", "shadow"],
                        help="Execution mode")
    parser.add_argument("--download", action="store_true", help="Download historical data and exit")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols for trading or download")
    parser.add_argument("--timeframes", type=str, nargs="+", help="Timeframes")
    parser.add_argument("--base_tf", type=str, help="Base timeframe for batch modes")
    parser.add_argument("--lookback", type=int, default=10234, help="Number of bars to download")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD) for batch modes")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD) for batch modes")
    parser.add_argument("--health", action="store_true", help="Health check and exit")
    return parser.parse_args()


# =============================================================================
#    MAIN ENTRY POINT
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
            symbols    = args.symbols    or cfg.get("download_defaults", {}).get("symbols")
            timeframes = args.timeframes or cfg.get("download_defaults", {}).get("timeframes")
            lookback   = args.lookback if hasattr(args, 'lookback') and args.lookback else None
            bot.download_historical_data(
                symbols=symbols,
                timeframes=timeframes,
                lookback_bars=lookback
            )
        except Exception as e:
            logging.error(f"Download failed: {e}")
            sys.exit(1)
        return

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

    # اجرای معمولی بر اساس mode
    bot.start(args)


if __name__ == "__main__":
    main()