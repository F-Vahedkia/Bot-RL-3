"""
تستر متد update_live از کلاس DataHandler_A

روش کار:
1. ابتدا داده‌های زنده (live) را از MT5 دریافت می‌کند
2. برای هر کندل جدید، update_live را صدا می‌زند
3. بعد از جمع‌آوری تعداد کافی کندل، داده‌های batch را دانلود می‌کند
4. خروجی update_live را با build() مقایسه می‌کند


Run: python -m f03_data.tests_data_handler.test_live_datahandler_A
"""

from __future__ import annotations

import sys, os, threading, logging # , time
import pandas as pd

from pathlib import Path
from datetime import timedelta     # , datetime, timezone
from typing import Dict, Any, List # , Optional

sys.path.insert(0, os.path.dirname(__file__) + "/../..")

from f03_data.data_handler_G import DataHandler, BuildParams
from f03_data.mt5_data_loader_E import MT5DataLoader_batch, DownloadPlan
from f03_data.live_market_engine import EventBus, MT5StreamWorker
from f02_utils.config_completer import config_completer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-6s | %(filename)-28s | %(lineno)-4d : %(funcName)-24s | %(message)s"
)
logger = logging.getLogger(__name__)


class LiveDataHandlerTester:

    # =========================================================================
    def __init__(
        self,
        cfg: Dict[str, Any],
        symbol: str,
        timeframe: str,
        required_bars: Dict[str, int]
    ):
        self.cfg = cfg
        
        # تنظیمات تست
        dl = (cfg.get("download_defaults") or {})
        self.symbol = symbol
        self.base_tf = timeframe
        self.required_bars = dict(required_bars)

        if not self.required_bars:
            raise ValueError("required_bars cannot be empty.")

        self.timeframes = list(self.required_bars.keys())

        if self.base_tf not in self.timeframes:
            raise ValueError(
                f"Base timeframe '{self.base_tf}' is not present in required_bars."
            )
        
        self.num_live_candles = 3  # تعداد کندل‌های زنده برای جمع‌آوری
        
        self.event_bus = EventBus(queue_size=1000)
        self.data_handler = DataHandler(
            cfg=cfg,
            symbol=symbol,
            required_bars=self.required_bars,
            event_bus=self.event_bus
        )
        
        self.live_rows: List[pd.DataFrame] = []
        self.collected = 0
        self.last_candle_time = None
        self.worker = None
        self.sub_id = None
        self.running = True
    
    # =========================================================================
    def _on_candle(self, event: Dict) -> None:
        if event.get("event_type") != "NEW_CANDLE":
            return

        timeframe = event["timeframe"]
        if timeframe != self.base_tf:
            return

        all_dfs = event["all_dfs"]

        base_key = f"{self.symbol}:{self.base_tf.upper()}"
        base_df = all_dfs[base_key]

        candle_time = base_df.index[-1]

        if self.last_candle_time == candle_time:
            return

        self.last_candle_time = candle_time
        self.collected += 1
        logger.info(
            f"Candle {self.collected}/{self.num_live_candles} at {candle_time}"
        )

        dataset = self.data_handler.update_live(event)

        if dataset is not None:
            df = dataset.get(self.base_tf)

            if df is not None and not df.empty:
                self.live_rows.append(df.iloc[-1:].copy())
                logger.info(
                    f"  -> Row shape: {df.iloc[-1:].shape[1]} columns, "
                    f"total rows: {len(df)}"
                )

        if self.collected >= self.num_live_candles:
            logger.info("Enough candles collected, stopping...")
            self.running = False
            if self.worker:
                self.worker.stop()

    # =========================================================================
    def _consumer_loop(self):
        while self.running:
            event = self.event_bus.get_event(self.sub_id, timeout=1.0)
            if event:
                self._on_candle(event)
    
    # =========================================================================
    def _download_batch_data(self):
        """دانلود داده‌های batch برای همان بازه زمانی"""
        logger.info("Downloading batch data...")
        
        # محاسبه بازه زمانی بر اساس کندل‌های جمع‌آوری شده
        if not self.live_rows:
            logger.warning("No live data collected")
            return False
        
        first_candle_time = self.live_rows[0].index[0]
        last_candle_time = self.live_rows[-1].index[0]
        
        # اضافه کردن حاشیه امن (چند کندل قبل)
        margin = timedelta(minutes=30)
        
        # live timestamps are UTC-aware.
        # DownloadPlan expects broker-local naive datetimes.
        broker_tz = self.data_handler.broker_timezone
        date_from = (
            first_candle_time
            .tz_convert(broker_tz)
            .tz_localize(None)
            - margin
        )
        date_to = (
            last_candle_time
            .tz_convert(broker_tz)
            .tz_localize(None)
            + margin
            )

        loader = MT5DataLoader_batch(cfg=self.cfg)
        loader.save_format = "parquet"

        plans = []
        for tf in self.timeframes:
            plans.append(DownloadPlan(
                symbol=self.symbol,
                timeframe=tf,
                date_from=date_from,
                date_to=date_to,
                date_tz=self.data_handler.broker_timezone,
                result_tz="UTC",
                lookback_bars=None,
                range_policy="date"
            ))
        
        results = loader.run_plan(plans)
        
        for r in results:
            if "error" in r:
                logger.error(f"Download failed for {r.get('symbol')}/{r.get('timeframe')}: {r['error']}")
                return False
        
        logger.info("Batch data downloaded successfully")
        return True
    
    # =========================================================================
    def _compare_results(self) -> bool:
        """مقایسه خروجی update_live با build()"""
        logger.info("=" * 60)
        logger.info("Comparing update_live vs build()")
        logger.info("=" * 60)
        
        build_dataset = self.data_handler.build(BuildParams(
            symbol=self.symbol,
            base_tf=self.base_tf,
            timeframes=self.timeframes,
            load_format="parquet"
        ))
        build_df = build_dataset.get(self.base_tf)


        live_start = self.live_rows[0].index[0]
        live_end = self.live_rows[-1].index[0]
        build_df = build_df[(build_df.index >= live_start) & (build_df.index <= live_end)]

        logger.info(f"build() -> {len(build_df)} rows, {len(build_df.columns)} cols")
        logger.info(f"update_live -> {len(self.live_rows)} rows collected")
        
        if build_df.empty:
            logger.error("build() returned empty dataframe")
            return False
        
        # آخرین ردیف update_live را با آخرین ردیف build مقایسه کن
        last_live = self.live_rows[-1] if self.live_rows else None
        last_build = build_df.iloc[-1:] if not build_df.empty else None
        
        if last_live is None or last_build is None:
            logger.error("No data to compare")
            return False
        
        # مقایسه ستون‌ها
        live_cols = set(last_live.columns)
        build_cols = set(last_build.columns)
        
        if live_cols != build_cols:
            logger.warning(f"Column mismatch!")
            logger.warning(f"  Only in live: {live_cols - build_cols}")
            logger.warning(f"  Only in build: {build_cols - live_cols}")
        
        # مقایسه مقادیر (با tolerance برای floating point)
        all_match = True
        for col in build_cols.intersection(live_cols):
            if col.startswith(("qc_", "session_")):
                continue  # QC و session flags را نادیده بگیر
            
            live_val = last_live[col].iloc[0]
            build_val = last_build[col].iloc[0]
            
            if pd.isna(live_val) and pd.isna(build_val):
                continue
            
            if pd.isna(live_val) or pd.isna(build_val):
                logger.warning(f"Mismatch in {col}: live={live_val}, build={build_val}")
                all_match = False
                continue
            
            if isinstance(live_val, (int, float)) and isinstance(build_val, (int, float)):
                if abs(live_val - build_val) > 1e-6:
                    logger.warning(f"Mismatch in {col}: live={live_val}, build={build_val}")
                    all_match = False
            else:
                if live_val != build_val:
                    logger.warning(f"Mismatch in {col}: live={live_val}, build={build_val}")
                    all_match = False
        
        if all_match:
            logger.info("✓ All matching! update_live and build() produce same output.")
        else:
            logger.warning("✗ Some mismatches found.")
        
        # ----- ذخیره نتایج
        output_dir = Path(__file__).parent.parent.parent / ""
        # output_dir.mkdir(exist_ok=True)
        
        build_df.to_csv(output_dir / "batch_output.csv")
        
        # ذخیره ردیف‌های live
        live_df = pd.concat(self.live_rows) if self.live_rows else pd.DataFrame()
        live_df.to_csv(output_dir / "live_output.csv")
        
        logger.info(f"Results saved to {output_dir}")
        
        return all_match
    
    # =========================================================================
    def run(self) -> bool:
        print("=" * 60)
        print(f"Testing DataHandler.update_live()")
        print(f"Symbol: {self.symbol}")
        print(f"Base TF: {self.base_tf}")
        print(f"Timeframes: {self.timeframes}")
        print(f"Collecting {self.num_live_candles} live candles...")
        print("=" * 60)
        
        # 1) دریافت داده‌های زنده
        self.worker = MT5StreamWorker(
            cfg=self.cfg,
            event_bus=self.event_bus,
            # warmups_dicts=self.cfg["__warmups_dicts"], # اصلاح نیاز دارد
            warmups_dicts={
                self.symbol: self.required_bars
            },
            poll_interval_sec=1.0
        )
        
        if not self.worker.connector.initialize():
            logger.error("MT5 connection failed")
            return False
        
        self.sub_id = self.event_bus.subscribe(self.symbol)
        
        consumer_thread = threading.Thread(target=self._consumer_loop, daemon=True)
        consumer_thread.start()
        
        logger.info("Waiting for live candles...")
        self.worker.start()
        # بعد از برگشتن از start (به دلیل stop شدن در _on_candle)
        self.running = False
        self.worker.connector.shutdown()
        
        if self.collected < self.num_live_candles:
            logger.warning(f"Only collected {self.collected}/{self.num_live_candles} candles")
        
        # 2) دانلود داده‌های batch (بعد از لایو)
        if not self._download_batch_data():
            logger.error("Failed to download batch data")
            return False
        
        # 3) مقایسه نتایج
        return self._compare_results()

# =============================================================================
# MAIN TEST
# =============================================================================
def main():
    # cfg = load_config()
    cfg = config_completer(enable_env_override=True)

    symbol = "BITCOIN"
    timeframe = "M1"

    required_bars = {
        timeframe: cfg["__warmups_dicts"][symbol][timeframe]
    }

    print(f"symbol = {symbol}")
    print(f"timeframe = {timeframe}")
    print(f"required_bars = {required_bars}")

    tester = LiveDataHandlerTester(
        cfg,
        symbol,
        timeframe,
        required_bars,
    )

    success = tester.run()
    
    print("\n" + "=" * 60)
    print(f"TEST {'PASSED' if success else 'FAILED'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
