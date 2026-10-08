# f99_main / main_6_4_orch_feat_comp.py
# Date Reviewed:
#     1405-05-22-22:00

"""
Main Orchestrator of Bot-RL-3
"""
def docstring():

    """
    نقش تا الان:
        ارکستراتور اصلی ربات معاملاتی که اجزای لایه‌های Data و Features
        و سپس مسیرهای بالاتر سیستم را بر اساس mode هماهنگ می‌کند.

    معماری Feature:
        برای هر Symbol یک مسیر مستقل ایجاد می‌شود:

            DataHandler(symbol)
                ↓
            MTFDataset(symbol)
                ↓
            FeaturePipeline(symbol)
                ↓
            FeatureEngine(symbol)
                ↓
            FeatureStore
                ↓
            ObservationBuilder
                ↓
            Symbol-Agent

        در سطح بالاتر:
            Symbol-Agentها
                ↓
            Meta-Agent

    حالت‌های اجرایی:
        live / paper / shadow
        train / backtest / optimize / evaluate
        download / health

    اصول مهم:
        - هر Symbol دارای DataHandler مستقل است.
        - هر Symbol دارای FeaturePipeline مستقل است.
        - هر FeaturePipeline از FeatureEngine مستقل همان Symbol استفاده می‌کند.
        - FeaturePipeline، MTFDataset را از لایه Data دریافت می‌کند و خودش مسئول دریافت داده از DataHandler یا MarketDataEngine نیست.
        - FeaturePipeline مسئول هماهنگ‌سازی FeatureEngine، FeatureStore و ObservationBuilder برای همان Symbol است.
        - در حالت Live، داده‌ها از MarketDataEngine و EventBus به DataHandler همان Symbol می‌رسند و سپس به FeaturePipeline همان Symbol منتقل می‌شوند.
        - mode و feature specifications در سطح orchestrator / pipeline تعیین شده و به FeatureEngine منتقل می‌شوند.
        - FeatureEngine مسئول اجرای محاسبات Feature و نگهداری stateهای incremental مربوط به Live است.
        - این فایل مسئول orchestration است و نباید منطق محاسبات اندیکاتورها یا ساخت Featureهای هر Symbol را در خود پیاده‌سازی کند.

    روش های اجرا:
    - MODE = live / paper / shadow
        python -m main_6_4_orch_feat_comp --mode MODE --symbols EURUSD --timeframes M5
        python -m main_6_4_orch_feat_comp --mode MODE --symbols EURUSD XAUUSD BITCOIN --timeframes M1 M5 H1
    
    - MODE = train  / backtest / optimize / evaluate
        python -m main_6_4_orch_feat_comp --mode train    --symbols XAUUSD --base_tf H1 --timeframes H1 H4 D1
        python -m main_6_4_orch_feat_comp --mode backtest --symbols EURUSD --base_tf M1 --timeframes M1 M5 H1 --start 2024-01-01 --end 2024-12-31
    
    -Download
        python -m main_6_4_orch_feat_comp --download --symbols EURUSD --timeframes M1 M5 --lookback 5000
    - Health check
        python -m main_6_4_orch_feat_comp --health

    اجرا همراه با آدرس دهی فایل کانفیگ:
        python -m main_6_4_orch_feat_comp --config [path to config.yaml] --mode live --symbols EURUSD --timeframes M5
        python -m main_6_4_orch_feat_comp --config [path to config.yaml] --mode live --symbols EURUSD XAUUSD BITCOIN --timeframes M1 M5 H1
    
    ===========================================================================
    برای آینده:
    -----------
    Main / Root Orchestrator
        │
        ├── Config
        ├── Data
        ├── Features
        ├── Observation
        ├── Machine Learning / Symbol-Agent
        ├── Meta-Agent
        ├── Risk Management
        ├── Order / Execution
        ├── Position Management
        ├── Broker / MT5
        └── Monitoring / Lifecycle
        
        نسخه‌ای که الان در اختیار داریم فقط Data و Feature و Observation را دارد.
        در فایل فعلی نیز خروجی Observation هنوز به listener ها تحویل می‌شود و
        بعد از آن subsystem های معاملاتی در این فایل هنوز وجود ندارند.
    """
    pass

# main_6_4_orchestrator_feature_compatible.py
# main_6_5.py ==> After deleting feature_bootstrap.py

# main_6_5.py
# =============================================================================
#    IMPORTS
# =============================================================================
from __future__ import annotations

import sys
import signal
import threading
import logging
import argparse
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable

# -------------------- Internal Imports -------------------
from f03_data.live_market_engine import MarketDataEngine
from f03_data.data_handler_G import DataHandler, BuildParams
from f03_data.mt5_data_loader_E import MT5DataLoader_batch
from f03_data.mtf_dataset import MTFDataset

from f04_features.feature_B_graph import FeatureGraph
from f04_features.observation_B_builder import ObservationBuilder
from f04_features.feature_C_engine_6 import FeatureEngine
from f04_features.feature_B_store import FeatureStoreV2
from f04_features.feature_pipeline import FeaturePipeline

from f02_utils.config_loader import load_config
from f02_utils.logging_utils import setup_logging

# =============================================================================
#    DATA SOURCE ABSTRACTION
# =============================================================================
class DataSource:
    """کلاس پایه برای تمام منابع داده"""
    def get_next_candle(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def has_next(self) -> bool:
        raise NotImplementedError

    def close(self):
        pass

# =============================================================================
#    LIVE DATA SOURCE
# =============================================================================
class LiveDataSource(DataSource):
    """منبع داده زنده از طریق EventBus (live_market_engine)"""
    def __init__(self, engine: MarketDataEngine, data_handler: DataHandler):
        self.engine = engine
        self.data_handler = data_handler
        self._running = True

    def get_next_candle(self) -> Optional[Dict[str, Any]]:
        dataset = self.data_handler.get_latest_dataset()
        if dataset is None:
            return None

        try:
            df = dataset.get(self.data_handler._base_tf)
        except KeyError:
            return None

        if df.empty:
            return None

        row = df.iloc[-1]
        latest = row.to_dict()
        latest["time"] = row.name
        return latest


    def has_next(self) -> bool:
        return self._running

    def close(self):
        self.engine.stop()
        self.data_handler.stop_consuming()

# =============================================================================
#    BATCH DATA SOURCE
# =============================================================================
class BatchDataSource(DataSource):
    """منبع داده بچ (فایل‌های Parquet/CSV) برای بک‌تست و آموزش"""
    def __init__(self,
                 data_handler: DataHandler,
                 symbol: str,
                 base_tf: str,
                 timeframes: List[str],
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None
                 ):
        self.data_handler = data_handler
        self.symbol = symbol
        self.base_tf = base_tf
        self.timeframes = timeframes
        self.df = None
        self._index = 0
        self.dataset = None
        self._load_data(start_date, end_date)


    def _load_data(self, start_date: Optional[str], end_date: Optional[str]):
        params = BuildParams(
            symbol=self.symbol,
            base_tf=self.base_tf,
            timeframes=self.timeframes,
            format_="parquet"
        )
        # self.df = self.data_handler.build(params)
        dataset = self.data_handler.build(params)
        self.dataset = dataset
        self.df = dataset.get(self.base_tf)
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
        self._data_listeners: List[Callable[[Any], None]] = []
        self._stop_event = threading.Event()   # برای نگه‌داشتن نخ اصلی بدون polling

        # مؤلفه‌های اصلی
        self.engine: Optional[MarketDataEngine] = None
        self.data_handlers: Dict[str, DataHandler] = {}
        self.feature_pipelines: Dict[str, FeaturePipeline] = {}
        self.feature_engines: Dict[str, Any] = {}
        self.feature_stores: Dict[str, Any] = {}
        self.data_sources: Dict[str, DataSource] = {}

        # Backward-compatible aliases for single-symbol paths.
        self.data_handler: Optional[DataHandler] = None
        self.data_source: Optional[DataSource] = None
        self.downloader: Optional[MT5DataLoader_batch] = None

        self.logger = logging.getLogger(__name__)

        self.feature_engine = None
        self.feature_store = None
        self.feature_graph = None
        self.obs_builder = None
        self.feature_specs: List[str] = []
        self.pipeline = None

        # ---------- batch streaming state ----------
        # self._batch_buffer = []  # حذف میشود. چون متد مصرف کننده اش کلاً حذف کامنت شده است.
        self._batch_df = None

    # ---------------------------------------------------------------
    def load_configuration(self) -> None:
        try:
            self.cfg = (
                load_config(self.config_path)
                if self.config_path
                else load_config()
            )
            self.logger.info("Configuration loaded successfully")

            self.feature_specs = list(
                self.cfg.get("features", {}).get("indicators", [])
            )

            if not self.feature_specs:
                self.logger.warning("No feature specs found in config")

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise


    def _build_feature_pipeline(self, symbol: str) -> FeaturePipeline:
        """Build one independent FeaturePipeline for one Symbol."""
        feature_engine = FeatureEngine(self.cfg)
        feature_store = FeatureStoreV2(self.cfg)
        feature_graph = FeatureGraph(self.feature_specs)
        observation_builder = ObservationBuilder(self.cfg)

        pipeline = FeaturePipeline(
            symbol=symbol,
            feature_engine=feature_engine,
            feature_store=feature_store,
            observation_builder=observation_builder,
            feature_graph=feature_graph,
            feature_specs=self.feature_specs,
            config=self.cfg,
        )

        self.feature_engines[symbol] = feature_engine
        self.feature_stores[symbol] = feature_store
        self.feature_pipelines[symbol] = pipeline

        if self.feature_engine is None:
            self.feature_engine = feature_engine
            self.feature_store = feature_store
            self.feature_graph = feature_graph
            self.obs_builder = observation_builder
            self.pipeline = pipeline

        return pipeline


    def _ensure_feature_pipelines(self, symbols: List[str]) -> None:
        for symbol in symbols:
            if symbol not in self.feature_pipelines:
                self._build_feature_pipeline(symbol)


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
    def register_data_listener(self, callback: Callable[[Any], None]) -> None:
        """ثبت شنونده برای دریافت داده‌های جدید (لایه features, risk, ...)"""
        self._data_listeners.append(callback)
        self.logger.debug(f"Data listener registered. Total: {len(self._data_listeners)}")


    def _notify_data_listeners(self, value: Any) -> None:
        for cb in self._data_listeners:
            try:
                cb(value)
            except Exception as e:
                self.logger.exception(f"Listener callback error: {e}")


    # ===============================================================
    # DATA -> FEATURE
    # ===============================================================
    # def _process_features(self, df: pd.DataFrame) -> pd.DataFrame:
    #     return self.feature_engine.execute(
    #         df,
    #         self.feature_specs,
    #         mode=self.mode,
    #     )

    # ===============================================================
    # FEATURE -> OBSERVATION
    # ===============================================================
    def _process_pipeline(self, dataset: MTFDataset):
        if dataset is None:
            raise ValueError("dataset is required")

        symbol = dataset.symbol
        pipeline = self.feature_pipelines.get(symbol)

        if pipeline is None:
            pipeline = self._build_feature_pipeline(symbol)

        if self.mode in {"live", "paper", "shadow"}:
            return pipeline.process_live(dataset)

        result = pipeline.run(dataset, mode=self.mode)
        return result["observation"]

    # ===============================================================
    # PIPELINE ENTRY
    # ===============================================================
    def _on_data_updated(self, dataset: MTFDataset) -> None:
        # if not hasattr(self.data_handler, "_latest_dataset"):
        #     return
        # dataset = self.data_handler._latest_dataset

        if not self._running:
            return
        observation = self._process_pipeline(dataset)

        if observation is None:
            return

        self._notify_data_listeners(observation)
        self.logger.debug(
            "Observation ready : shape=%s",
            getattr(observation, "shape", None),
        )
        
    # ---------------------------------------------------------------
    def _init_components(self, symbols: Optional[List[str]] = None) -> None:
        """ایجاد مؤلفه‌ها بر اساس mode."""
        if symbols is None:
            symbols = list(self.cfg.get("__warmups_dicts", {}).keys())

        symbols = list(symbols)

        if self.mode in ["live", "paper", "shadow"]:
            self.engine = MarketDataEngine(cfg=self.cfg)
            self._components["engine"] = self.engine
            self.logger.info("MarketDataEngine initialized (live mode)")

            self._ensure_feature_pipelines(symbols)

            for symbol in symbols:
                handler = DataHandler(
                    cfg=self.cfg,
                    symbol=symbol,
                    event_bus=None,
                )
                self.engine.attach_data_handler(handler)
                handler.set_data_callback(self._on_data_updated)

                self.data_handlers[symbol] = handler
                self.data_sources[symbol] = LiveDataSource(
                    self.engine,
                    handler,
                )

            if len(symbols) == 1:
                self.data_handler = self.data_handlers[symbols[0]]
                self.data_source = self.data_sources[symbols[0]]

            self._components["data_handlers"] = self.data_handlers

        elif self.mode in ["train", "backtest", "optimize", "evaluate"]:
            if len(symbols) != 1:
                raise ValueError(
                    "Batch mode currently requires exactly one symbol."
                )

            symbol = symbols[0]
            self._ensure_feature_pipelines(symbols)

            handler = DataHandler(
                cfg=self.cfg,
                symbol=symbol,
                event_bus=None,
            )
            self.data_handlers[symbol] = handler
            self.data_handler = handler
            self._components["data_handler"] = handler

        dl_cfg = self.cfg.get("download_defaults", {})
        if dl_cfg.get("symbols") and dl_cfg.get("timeframes"):
            self.downloader = MT5DataLoader_batch(cfg=self.cfg)
            self._components["downloader"] = self.downloader
            self.logger.info("MT5DataLoader initialized")


    def start_live(
        self,
        symbols: List[str],
        timeframes: List[str],
        poll_interval: float = 2.0,
    ) -> None:
        """شروع حالت live/paper/shadow."""
        symbols = list(symbols)
        self._ensure_feature_pipelines(symbols)

        warmups_dicts = self.cfg.get("__warmups_dicts", {})
        missing = [s for s in symbols if s not in warmups_dicts]
        if missing:
            raise ValueError(
                f"No warmup configuration found for symbols: {missing}"
            )

        self._running = True

        threading.Thread(
            target=self.engine.start,
            args=(warmups_dicts, poll_interval),
            daemon=True,
            name="MarketDataEngine",
        ).start()

        for symbol, handler in self.data_handlers.items():
            threading.Thread(
                target=handler.start_consuming,
                daemon=True,
                name=f"DataHandler-{symbol}",
            ).start()

        self.logger.info(
            "Live started for symbols=%s, timeframes=%s",
            symbols,
            timeframes,
        )


    def start_batch(
        self,
        symbol: str,
        base_tf: str,
        timeframes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        self._ensure_feature_pipelines([symbol])

        handler = self.data_handlers.get(symbol)
        if handler is None:
            raise RuntimeError(
                f"No DataHandler initialized for symbol={symbol}"
            )

        self.data_source = BatchDataSource(
            handler,
            symbol,
            base_tf,
            timeframes,
            start_date,
            end_date,
        )
        self._batch_dataset = self.data_source.dataset
        self._running = True

        self.logger.info(
            "Batch mode started: %d rows",
            len(self.data_source.df),
        )


    def run_batch_loop(self):
        if self._batch_dataset is None:
            raise RuntimeError("Batch dataset is not initialized.")

        observation = self._process_pipeline(self._batch_dataset)
        self._notify_data_listeners(observation)

        self._running = False

    # ---------------------------------------------------------------
    def start(self, cli_args: argparse.Namespace) -> None:
        """شروع بر اساس mode و پارامترهای خط فرمان."""
        if self._running:
            self.logger.warning("Bot is already running")
            return

        self.load_configuration()
        self._setup_signal_handlers()

        try:
            if self.mode in ["live", "paper", "shadow"]:

                symbols = cli_args.symbols or list(
                    self.cfg.get("__warmups_dicts", {}).keys()
                )
                if not symbols:
                    symbols = self.cfg.get(
                        "download_defaults", {}
                    ).get("symbols", ["EURUSD"])

                timeframes = cli_args.timeframes or list(
                    self.cfg.get("__timeframes_dict", {}).get(
                        symbols[0], []
                    )
                )
                if not timeframes:
                    timeframes = self.cfg.get(
                        "download_defaults", {}
                    ).get("timeframes", ["M5"])

                self._init_components(symbols)

                poll_interval = self.cfg.get(
                    "executor", {}
                ).get("poll_interval_sec", 2.0)

                self.start_live(
                    symbols,
                    timeframes,
                    poll_interval,
                )

                self.logger.info(
                    "Entering listener-based wait (no polling). "
                    "Press Ctrl+C to stop."
                )
                self._stop_event.wait()

            elif self.mode in ["train", "backtest", "optimize", "evaluate"]:

                configured_symbols = list(
                    self.cfg.get("__warmups_dicts", {}).keys()
                )
                symbol = (
                    cli_args.symbols[0]
                    if cli_args.symbols
                    else (
                        configured_symbols[0]
                        if configured_symbols
                        else "EURUSD"
                    )
                )

                base_tf = cli_args.base_tf or self.cfg.get(
                    "__base_tfs_dict", {}
                ).get(symbol, "M5")

                timeframes = cli_args.timeframes or list(
                    self.cfg.get("__timeframes_dict", {}).get(
                        symbol,
                        [base_tf, "H1"],
                    )
                )

                self._init_components([symbol])

                self.start_batch(
                    symbol,
                    base_tf,
                    timeframes,
                    cli_args.start,
                    cli_args.end,
                )
                self.run_batch_loop()

            else:
                raise ValueError(
                    f"Unsupported mode: {self.mode}"
                )

        except Exception as e:
            self.logger.exception(
                f"Failed to start bot: {e}"
            )
            self.shutdown()
            raise

    # ---------------------------------------------------------------
    def stop(self) -> None:
        self._running = False
        self.logger.info("Stopping Bot-RL-3...")
        if self.engine:
            self.engine.stop()

        for handler in self.data_handlers.values():
            try:
                handler.stop_consuming()
            except Exception:
                self.logger.exception(
                    "Failed to stop DataHandler for %s",
                    getattr(handler, "symbol", "unknown"),
                )

        for source in self.data_sources.values():
            try:
                source.close()
            except Exception:
                self.logger.exception(
                    "Failed to close live data source"
                )

        if self.data_source:
            self.data_source.close()
        self._stop_event.set()   # آزاد کردن نخ اصلی
        self.logger.info("Bot-RL-3 stopped")

    
    def shutdown(self) -> None:
        self.logger.info("Initiating shutdown...")
        self.stop()
        self.logger.info("Shutdown complete")

    # ---------------------------------------------------------------
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
        if self.data_handlers:
            status["components"]["data_handlers"] = {
                symbol: {
                    "subscribed": handler._subscription_key is not None
                }
                for symbol, handler in self.data_handlers.items()
            }
        return status

    # def _append_batch_candle(self, candle: dict) -> pd.DataFrame:
    #     self._batch_buffer.append(candle)

    #     df = pd.DataFrame(self._batch_buffer)
    #     if "time" in df.columns:
    #         df["time"] = pd.to_datetime(df["time"])
    #         df = df.set_index("time", drop=True)
    #     self._batch_df = df

    #     return df
    

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
            symbols    = args.symbols    or cfg.get("download_defaults", {}).get("symbols")
            timeframes = args.timeframes or cfg.get("download_defaults", {}).get("timeframes")
            lookback   = args.lookback if hasattr(args, 'lookback') and args.lookback else None
            bot._init_components(symbols or [])
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
            symbols = args.symbols or list(
                bot.cfg.get("__warmups_dicts", {}).keys()
            )
            bot._init_components(symbols)
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