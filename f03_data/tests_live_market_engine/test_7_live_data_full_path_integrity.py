# f03_data/tests_live_market_engine/test_7_live_data_full_path_integrity.py
#
# Date Reviewed:
#     1405/06/02- 

# Run:
#     python -m f03_data.tests_live_market_engine.test_7_live_data_full_path_integrity

# =============================================================================
""" LIVE DATA FULL-PATH INTEGRITY TEST
==================================

این تستر مسیر واقعی داده‌های Live را با داده‌های واقعی MT5 از
MT5Connector تا خروجی نهایی DataHandler بررسی می‌کند.

Pipeline تحت تست:

    MT5Connector
        ↓
    MT5StreamWorker
        ↓
    _fetch_closed()
        ↓
    CandleDetector
        ↓
    _fetch_all_tfs()
        ↓
    EventBus.publish()
        ↓
    DataHandler EventBus subscription
        ↓
    DataHandler.update_live()
        ↓
    DataHandler._update_cache()
        ↓
    MTFDataset
        ↓
    DataHandler._latest_dataset
        ↓
    DataHandler callback

این تستر برخلاف test_6_live_market_engine_integrity_1.py
فقط دریافت NEW_CANDLE از EventBus را بررسی نمی‌کند؛ بلکه همان
event واقعی را مستقیماً به update_live() تحویل می‌دهد و نتیجه
پردازش آن در DataHandler را تا MTFDataset و callback نهایی
اعتبارسنجی می‌کند.

هیچ داده مصنوعی تولید نمی‌شود.
داده‌ها مستقیماً از MT5 دریافت می‌شوند.

تست شامل موارد زیر است:

1. بارگذاری Config واقعی پروژه.
2. ساخت MarketDataEngine واقعی.
3. ساخت DataHandler واقعی برای هر Symbol.
4. اتصال DataHandler به EventBus.
5. اجرای واقعی MT5StreamWorker و MarketDataEngine.
6. دریافت NEW_CANDLE واقعی از EventBus.
7. اعتبارسنجی نوع و ساختار event.
8. اعتبارسنجی symbol و timeframe رویداد.
9. اعتبارسنجی all_dfs.
10. اعتبارسنجی DataFrameهای all_dfs.
11. اعتبارسنجی DatetimeIndex و ترتیب زمانی.
12. اعتبارسنجی timezone-aware بودن index.
13. اعتبارسنجی timezone = UTC.
14. اعتبارسنجی وجود OHLCVS در DataFrameهای Live.
15. اعتبارسنجی وجود timeframe مورد انتظار در all_dfs.
16. استخراج و اعتبارسنجی timestamp آخرین کندل از all_dfs.
17. تحویل event واقعی به DataHandler.update_live().
18. اعتبارسنجی عدم تغییر in-place داده‌های event توسط DataHandler.
19. اعتبارسنجی DataHandler._cache_dict.
20. اعتبارسنجی تعداد رکوردهای cache در محدوده warmup.
21. اعتبارسنجی آخرین timestamp موجود در cache.
22. اعتبارسنجی MTFDataset ساخته‌شده توسط DataHandler.
23. اعتبارسنجی symbol و base_tf در MTFDataset.
24. اعتبارسنجی وجود frameهای مورد انتظار در MTFDataset.
25. اعتبارسنجی DataHandler._latest_dataset.
26. اعتبارسنجی اجرای callback مصرف‌کننده.
27. اعتبارسنجی اینکه callback همان latest dataset را دریافت کرده است.
28. گزارش خروجی نهایی DataHandler برای تمام timeframeهای موجود.
29. امکان اجرای تست برای تمام Symbolهای تنظیم‌شده در Config.
30. توقف صحیح DataHandlerها.
31. توقف صحیح MT5StreamWorker و MarketDataEngine.
32. اعتبارسنجی خاتمه‌ی صحیح thread مربوط به MarketDataEngine.

این تستر عمداً قراردادهای واقعی فعلی پروژه را دنبال می‌کند و
ناسازگاری‌های واقعی در مسیر Live را با workaround پنهان نمی‌کند.
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict

import pandas as pd

from f03_data.live_market_engine import MarketDataEngine
from f03_data.data_handler_G import DataHandler
from f02_utils.config_completer import config_completer

# =============================================================================
# Configuration
# =============================================================================
TEST_TIMEOUT_SEC = 120.0
EVENT_TIMEOUT_SEC = 1.0
POLL_INTERVAL_SEC = 2.0

# حداقل تعداد event مورد نیاز برای هر Symbol
EVENTS_REQUIRED_PER_SYMBOL = 1

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)-6s | %(filename)-40s |"
        " %(lineno)-4d : %(funcName)-22s | %(message)s"
    ),
)
logger = logging.getLogger("test_live_full_path")

# =============================================================================
# Assertion helpers
# =============================================================================
def fail(message: str) -> None:
    raise AssertionError(message)

def assert_true(condition: bool, message: str) -> None:
    if not condition:
        fail(message)

# =============================================================================
# DataFrame validation
# =============================================================================
def validate_dataframe(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    source_name: str,
) -> None:
    """
    اعتبارسنجی ساختاری یک DataFrame دریافتی از Live pipeline.
    """

    assert_true(
        isinstance(df, pd.DataFrame),
        (
            f"{symbol}/{timeframe}: {source_name} is not a "
            f"pandas.DataFrame: {type(df)!r}"
        ),
    )

    assert_true(
        not df.empty,
        f"{symbol}/{timeframe}: {source_name} is empty.",
    )

    assert_true(
        isinstance(df.index, pd.DatetimeIndex),
        (
            f"{symbol}/{timeframe}: {source_name} index must be "
            f"DatetimeIndex, got {type(df.index)!r}"
        ),
    )

    assert_true(
        df.index.is_monotonic_increasing,
        (
            f"{symbol}/{timeframe}: {source_name} index is not "
            f"monotonically increasing."
        ),
    )

    assert_true(
        not df.index.has_duplicates,
        (
            f"{symbol}/{timeframe}: {source_name} contains "
            f"duplicated timestamps."
        ),
    )

    assert_true(
        df.index.tz is not None,
        (
            f"{symbol}/{timeframe}: {source_name} index is timezone-naive."
        ),
    )

    assert_true(
        str(df.index.tz) == "UTC",
        (
            f"{symbol}/{timeframe}: {source_name} timezone must be UTC, "
            f"got {df.index.tz!r}"
        ),
    )

    required_columns = {"open", "high", "low", "close", "volume", "spread"}

    missing = required_columns.difference(df.columns)

    assert_true(
        not missing,
        (
            f"{symbol}/{timeframe}: {source_name} is missing columns: "
            f"{sorted(missing)}"
        ),
    )

# =============================================================================
# Event validation
# =============================================================================
def validate_event(
    event: Dict[str, Any],
    *,
    expected_symbol: str,
) -> Dict[str, pd.DataFrame]:
    """
    اعتبارسنجی event واقعی دریافت‌شده از EventBus.
    """

    assert_true(
        isinstance(event, dict),
        f"Received event is not dict: {type(event)!r}",
    )

    assert_true(
        event.get("event_type") == "NEW_CANDLE",
        f"Unexpected event_type: {event.get('event_type')!r}",
    )

    symbol = event.get("symbol")

    assert_true(
        symbol == expected_symbol,
        (
            f"Event symbol mismatch. "
            f"Expected={expected_symbol!r}, got={symbol!r}"
        ),
    )

    timeframe = event.get("timeframe")

    assert_true(
        isinstance(timeframe, str) and timeframe,
        f"Invalid event timeframe: {timeframe!r}",
    )

    all_dfs = event.get("all_dfs")

    assert_true(
        isinstance(all_dfs, dict),
        (
            f"{expected_symbol}: event['all_dfs'] must be dict, "
            f"got {type(all_dfs)!r}"
        ),
    )

    assert_true(
        len(all_dfs) > 0,
        f"{expected_symbol}: event['all_dfs'] is empty.",
    )

    for key, df in all_dfs.items():

        assert_true(
            isinstance(key, str),
            f"{expected_symbol}: all_dfs key is not str: {key!r}",
        )

        expected_prefix = f"{expected_symbol}:"

        assert_true(
            key.startswith(expected_prefix),
            (
                f"Invalid all_dfs key={key!r}; "
                f"expected prefix={expected_prefix!r}"
            ),
        )

        tf = key.split(":", 1)[1].upper()

        validate_dataframe(
            df,
            symbol=expected_symbol,
            timeframe=tf,
            source_name=f"event.all_dfs[{key!r}]",
        )

    event_key = f"{expected_symbol}:{timeframe.upper()}"

    assert_true(
        event_key in all_dfs,
        (
            f"Event timeframe {timeframe!r} is missing from all_dfs. "
            f"Expected key={event_key!r}; "
            f"available={sorted(all_dfs.keys())}"
        ),
    )

    candle_df = all_dfs[event_key]

    assert_true(
        not candle_df.empty,
        f"{event_key}: event dataframe is empty.",
    )

    candle_time = candle_df.index[-1]

    assert_true(
        isinstance(candle_time, pd.Timestamp),
        (
            f"{event_key}: latest candle timestamp must be "
            f"pandas.Timestamp, got {type(candle_time)!r}"
        ),
    )

    assert_true(
        candle_time.tz is not None,
        f"{event_key}: latest candle timestamp is timezone-naive.",
    )

    assert_true(
        str(candle_time.tz) == "UTC",
        (
            f"{event_key}: latest candle timestamp must be UTC, "
            f"got {candle_time.tz!r}"
        ),
    )

    return all_dfs

# =============================================================================
# Snapshot helpers
# =============================================================================
def snapshot_dataframes(
    all_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    قبل از تحویل event به DataHandler از DataFrameها snapshot می‌گیرد.

    هدف:
        بررسی اینکه DataHandler داده‌های موجود در event را
        در محل تغییر نداده باشد.
    """

    return {
        key: df.copy(deep=True)
        for key, df in all_dfs.items()
    }


def assert_event_data_unchanged(
    before: Dict[str, pd.DataFrame],
    after: Dict[str, pd.DataFrame],
    *,
    symbol: str,
) -> None:
    """
    بررسی می‌کند که DataHandler event.all_dfs را mutate نکرده باشد.
    """

    assert_true(
        before.keys() == after.keys(),
        (
            f"{symbol}: all_dfs keys changed after DataHandler processing. "
            f"before={sorted(before.keys())}, "
            f"after={sorted(after.keys())}"
        ),
    )

    for key in before:

        left = before[key]
        right = after[key]

        assert_true(
            left.equals(right),
            (
                f"{symbol}: DataHandler modified event dataframe "
                f"{key!r} in-place."
            ),
        )

# =============================================================================
# DataHandler cache validation
# =============================================================================
def validate_data_handler_cache(
    handler: DataHandler,
    *,
    symbol: str,
    event_all_dfs: Dict[str, pd.DataFrame],
) -> None:
    """
    بررسی می‌کند که DataHandler تمام TFهای مورد انتظار را
    در _cache_dict دریافت کرده باشد.
    """

    warmup_dict = handler._required_bars

    assert_true(
        isinstance(warmup_dict, dict),
        (
            f"{symbol}: DataHandler._required_bars must be dict, "
            f"got {type(warmup_dict)!r}"
        ),
    )

    assert_true(
        len(warmup_dict) > 0,
        f"{symbol}: DataHandler._required_bars is empty.",
    )

    cache = handler._cache_dict

    assert_true(
        isinstance(cache, dict),
        (
            f"{symbol}: DataHandler._cache_dict must be dict, "
            f"got {type(cache)!r}"
        ),
    )

    for tf, warmup in warmup_dict.items():

        tf = tf.upper()
        key = f"{symbol}:{tf}"

        assert_true(
            key in event_all_dfs,
            (
                f"{symbol}: required timeframe {tf} is missing "
                f"from EventBus all_dfs."
            ),
        )

        assert_true(
            key in cache,
            (
                f"{symbol}: required timeframe {tf} did not reach "
                f"DataHandler._cache_dict."
            ),
        )

        cached_df = cache[key]
        source_df = event_all_dfs[key]

        validate_dataframe(
            cached_df,
            symbol=symbol,
            timeframe=tf,
            source_name=f"DataHandler._cache_dict[{key!r}]",
        )

        assert_true(
            len(cached_df) <= warmup,
            (
                f"{symbol}/{tf}: cache exceeds warmup. "
                f"rows={len(cached_df)}, warmup={warmup}"
            ),
        )

        assert_true(
            cached_df.index[-1] == source_df.index[-1],
            (
                f"{symbol}/{tf}: cache latest timestamp mismatch. "
                f"cache={cached_df.index[-1]!r}, "
                f"event={source_df.index[-1]!r}"
            ),
        )

        # آخرین ردیف event باید در cache وجود داشته باشد.
        source_last = source_df.iloc[-1]
        cache_last = cached_df.loc[source_df.index[-1]]

        for column in ("open", "high", "low", "close", "volume", "spread"):
            assert_true(
                cache_last[column] == source_last[column],
                (
                    f"{symbol}/{tf}: cached {column} mismatch. "
                    f"event={source_last[column]!r}, "
                    f"cache={cache_last[column]!r}"
                ),
            )

# =============================================================================
# MTFDataset validation
# =============================================================================
def validate_latest_dataset(
    handler: DataHandler,
    *,
    symbol: str,
) -> None:
    """
    بررسی خروجی نهایی DataHandler._latest_dataset.
    """

    dataset = handler.get_latest_dataset()

    assert_true(
        dataset is not None,
        f"{symbol}: DataHandler._latest_dataset is None.",
    )

    assert_true(
        dataset.symbol == symbol,
        (
            f"{symbol}: MTFDataset.symbol mismatch. "
            f"got={dataset.symbol!r}"
        ),
    )

    assert_true(
        dataset.base_tf == handler._base_tf,
        (
            f"{symbol}: MTFDataset.base_tf mismatch. "
            f"dataset={dataset.base_tf!r}, "
            f"handler={handler._base_tf!r}"
        ),
    )
    frames = dataset.frames

    assert_true(
        isinstance(frames, dict),
        (
            f"{symbol}: MTFDataset.frames must be dict, "
            f"got={type(frames)!r}"
        ),
    )

    assert_true(
        len(frames) > 0,
        f"{symbol}: MTFDataset.frames is empty.",
    )

    for tf in handler._required_bars:
        tf = tf.upper()
        assert_true(
            tf in frames,
            (
                f"{symbol}: timeframe {tf} missing from "
                f"DataHandler._latest_dataset."
            ),
        )
        df = frames[tf]
        validate_dataframe(
            df,
            symbol=symbol,
            timeframe=tf,
            source_name=f"DataHandler._latest_dataset.frames[{tf!r}]",
        )
        warmup = handler._required_bars[tf]
        assert_true(
            len(df) <= warmup,
            (
                f"{symbol}/{tf}: MTFDataset frame exceeds warmup. "
                f"rows={len(df)}, warmup={warmup}"
            ),
        )


# =============================================================================
# Callback validation
# =============================================================================

class CallbackRecorder:
    """
    Callback واقعی DataHandler.

    برای بررسی اینکه dataset نهایی DataHandler به consumer
    تحویل داده شده است.
    """

    def __init__(self) -> None:
        self.datasets = []

    def __call__(self, dataset) -> None:
        self.datasets.append(dataset)


# =============================================================================
# Per-symbol test
# =============================================================================

def test_symbol(
    *,
    symbol: str,
    handler: DataHandler,
    event_bus,
    engine_thread: threading.Thread,
    engine_errors: list[Exception],
) -> Dict[str, Any]:
    """
    یک مسیر کامل Live را برای یک Symbol تست می‌کند.
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info("TESTING SYMBOL: %s", symbol)
    logger.info("=" * 80)

    subscriber_id = handler._subscription_key

    assert_true(
        subscriber_id is not None,
        f"{symbol}: DataHandler has no EventBus subscription.",
    )

    logger.info("[1/10] DataHandler subscribed: %s", subscriber_id)

    callback_recorder = CallbackRecorder()
    handler.set_data_callback(callback_recorder)

    logger.info("[2/10] DataHandler callback registered.")

    deadline = time.monotonic() + TEST_TIMEOUT_SEC

    event = None

    while time.monotonic() < deadline:

        event = event_bus.get_event(
            subscriber_id,
            timeout=EVENT_TIMEOUT_SEC,
        )

        if event is None:
            # اگر موتور در نخ جداگانه خطا داده است،
            # همان خطای اصلی را به تست اصلی منتقل کن.
            if engine_errors:
                raise RuntimeError(
                    f"{symbol}: MarketDataEngine thread failed."
                ) from engine_errors[0]

            # پایان عادی یا غیرمنتظرهٔ نخ، بدون دریافت event،
            # نباید به انتظار تا پایان timeout منجر شود.
            if not engine_thread.is_alive():
                fail(
                    f"{symbol}: MarketDataEngine thread exited "
                    "before publishing NEW_CANDLE."
                )

            continue

        if event.get("event_type") != "NEW_CANDLE":
            logger.warning(
                "%s: ignoring unexpected event type=%r",
                symbol,
                event.get("event_type"),
            )
            continue

        if event.get("symbol") != symbol:
            fail(
                f"{symbol}: received event for another symbol: "
                f"{event.get('symbol')!r}"
            )

        break

    assert_true(
        event is not None,
        (
            f"{symbol}: no NEW_CANDLE event received within "
            f"{TEST_TIMEOUT_SEC:.1f} seconds."
        ),
    )

    logger.info("[3/10] REAL NEW_CANDLE received from EventBus.")

    all_dfs_before = snapshot_dataframes(
        event["all_dfs"]
    )

    all_dfs = validate_event(
        event,
        expected_symbol=symbol,
    )

    logger.info("[4/10] Event structure and all_dfs validated.")

    event_timeframe = event["timeframe"].upper()
    event_key = f"{symbol}:{event_timeframe}"

    logger.info(
        "Event candle: %s/%s",
        symbol,
        event_timeframe,
    )

    # logger.info(
    #     "Event candle time: %s",
    #     event["candle_time"],
    # )
    event_candle_time = all_dfs[event_key].index[-1]

    logger.info(
        "Event candle time: %s",
        event_candle_time,
    )

    logger.info(
        "Event all_dfs: %s",
        sorted(all_dfs.keys()),
    )

    # -----------------------------------------------------------------
    # تحویل همان event واقعی به DataHandler
    # -----------------------------------------------------------------

    dataset_from_update = handler.update_live(event)
    assert_true(
        dataset_from_update is not None,
        f"{symbol}: DataHandler.update_live() returned None.",
    )
    logger.info("[5/10] Event delivered to DataHandler.update_live().")

    all_dfs_after = event["all_dfs"]
    assert_event_data_unchanged(
        all_dfs_before,
        all_dfs_after,
        symbol=symbol,
    )
    logger.info("[6/10] Event data was not modified in-place by DataHandler.")

    # -----------------------------------------------------------------
    # Cache
    # -----------------------------------------------------------------

    validate_data_handler_cache(
        handler,
        symbol=symbol,
        event_all_dfs=all_dfs,
    )

    logger.info("[7/10] DataHandler._cache_dict validated.")

    # -----------------------------------------------------------------
    # Latest Dataset
    # -----------------------------------------------------------------

    validate_latest_dataset(
        handler,
        symbol=symbol,
    )

    logger.info("[8/10] DataHandler._latest_dataset / MTFDataset validated.")

    # -----------------------------------------------------------------
    # Callback
    # -----------------------------------------------------------------

    assert_true(
        len(callback_recorder.datasets) >= 1,
        (
            f"{symbol}: DataHandler callback was not invoked "
            f"after live update."
        ),
    )

    callback_dataset = callback_recorder.datasets[-1]

    assert_true(
        callback_dataset is handler.get_latest_dataset(),
        (
            f"{symbol}: callback dataset is not the same object "
            f"as DataHandler._latest_dataset."
        ),
    )

    logger.info("[9/10] DataHandler consumer callback received latest dataset.")

    # -----------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------

    dataset = handler.get_latest_dataset()

    logger.info("[10/10] FINAL DATAHANDLER OUTPUT")
    logger.info("Symbol      : %s", dataset.symbol)
    logger.info("Base TF     : %s", dataset.base_tf)

    for tf, df in dataset.frames.items():
        logger.info(
            "  %-5s rows=%-5d first=%s last=%s",
            tf,
            len(df),
            df.index[0],
            df.index[-1],
        )

    logger.info("")
    logger.info("PASS: FULL LIVE PATH %s", symbol)
    logger.info("=" * 80)

    return {
        "symbol": symbol,
        "event_timeframe": event_timeframe,
        # "event_candle_time": event["candle_time"],
        "event_candle_time": event_candle_time,
        "event_all_dfs": all_dfs,
        "dataset": dataset,
        "callback_count": len(callback_recorder.datasets),
    }

# =============================================================================
# Main
# =============================================================================

def main() -> int:

    logger.info("")
    logger.info("=" * 80)
    logger.info("LIVE DATA FULL-PATH INTEGRITY TEST")
    logger.info("=" * 80)

    # -----------------------------------------------------------------
    # 1. Config
    # -----------------------------------------------------------------

    cfg = config_completer()

    assert_true(
        isinstance(cfg, dict),
        f"config_completer() returned {type(cfg)!r}, expected dict.",
    )

    logger.info("[CONFIG] Configuration loaded.")

    # -----------------------------------------------------------------
    # 2. Required project configuration
    # -----------------------------------------------------------------

    broker_timezone = cfg["project"]["broker_timezone"]
    logger.info("[CONFIG] broker timezone = %s", broker_timezone)

    symbols = list(cfg["__symbols"])
    print(f" ============================================================,,,= {symbols}")
    symbols = list(["BITCOIN", "ETHEREUM"])
    print(f" ============================================================,,,= {symbols}")
    assert_true(
        symbols,
        "No symbols found in cfg['__symbols'].",
    )

    timeframes_dict = cfg["__timeframes_dict"]
    logger.info("[CONFIG] symbols = %s", symbols)
    logger.info("[CONFIG] timeframes_dict = %s", timeframes_dict)

    # -----------------------------------------------------------------
    # 3. MarketDataEngine
    # -----------------------------------------------------------------

    engine = MarketDataEngine(cfg)
    event_bus = engine.get_event_bus()

    logger.info("[ENGINE] MarketDataEngine created.")

    # -----------------------------------------------------------------
    # 4. DataHandlers
    # -----------------------------------------------------------------

    data_handlers: Dict[str, DataHandler] = {}
    
    for symbol in symbols:
        
        handler = DataHandler(cfg, symbol=symbol, required_bars=None )
        handler.subscribe_to_event_bus(event_bus)
        data_handlers[symbol] = handler

        logger.info(
            "[HANDLER] %s subscribed with id=%s",
            symbol,
            handler._subscription_key,
        )

    assert_true(
        event_bus.subscriber_count() == len(symbols),
        (
            "EventBus subscriber count mismatch. "
            f"expected={len(symbols)}, "
            f"actual={event_bus.subscriber_count()}"
        ),
    )
    logger.info(
        "[ENGINE] EventBus subscribers = %d",
        event_bus.subscriber_count(),
    )

    # -----------------------------------------------------------------
    # 5. Start real Live MarketDataEngine
    # -----------------------------------------------------------------
    engine_errors: list[Exception] = []

    # فقط نمادهای انتخاب‌شده برای تست، با تمام کندل‌های موردنیازشان
    all_required_bars = cfg["__all_required_bars"]
    required_bars_for_test = {
        symbol: all_required_bars[symbol]
        for symbol in symbols
    }

    def run_engine() -> None:
        try:
            engine.start(
                warmups_dicts=required_bars_for_test,
                poll_interval_sec=POLL_INTERVAL_SEC,
            )
        except Exception as exc:
            engine_errors.append(exc)
            logger.exception("[ENGINE] MarketDataEngine thread failed.")

    engine_thread = threading.Thread(
        target=run_engine,
        daemon=True,
        name="test-live-market-engine",
    )

    results: Dict[str, Any] = {}

    try:
        logger.info("[ENGINE] Starting real MarketDataEngine...")
        engine_thread.start()

        # -------------------------------------------------------------
        # 6. Test every configured symbol
        # -------------------------------------------------------------

        for symbol in symbols:
            results[symbol] = test_symbol(
                symbol=symbol,
                handler=data_handlers[symbol],
                event_bus=event_bus,
                engine_thread=engine_thread,
                engine_errors=engine_errors,
            )

        # -------------------------------------------------------------
        # 7. Final global validation
        # -------------------------------------------------------------

        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL GLOBAL VALIDATION")
        logger.info("=" * 80)

        for symbol in symbols:
            result = results[symbol]
            dataset = result["dataset"]
            assert_true(
                dataset is not None,
                f"{symbol}: final dataset is None.",
            )
            assert_true(
                result["callback_count"] >= 1,
                f"{symbol}: no callback notification recorded.",
            )
            logger.info(
                "PASS %-10s | event=%s | candle=%s | "
                "frames=%s | callbacks=%d",
                symbol,
                result["event_timeframe"],
                result["event_candle_time"],
                list(dataset.frames.keys()),
                result["callback_count"],
            )

        logger.info("")
        logger.info("=" * 80)
        logger.info("LIVE FULL-PATH TEST PASSED — %d symbol(s) tested.", len(symbols))
        logger.info("Verified path:")
        logger.info("MT5Connector")
        logger.info("  -> MT5StreamWorker")
        logger.info("  -> CandleDetector")
        logger.info("  -> EventBus")
        logger.info("  -> DataHandler.update_live")
        logger.info("  -> DataHandler._update_cache")
        logger.info("  -> MTFDataset / _latest_dataset")
        logger.info("  -> DataHandler callback")        
        logger.info("=" * 80)

        return 0

    finally:

        # -------------------------------------------------------------
        # 8. Stop DataHandlers
        # -------------------------------------------------------------

        logger.info("[CLEANUP] Stopping DataHandlers...")

        for symbol, handler in data_handlers.items():
            try:
                handler.stop_consuming()
            except Exception:
                logger.exception("[CLEANUP] Failed to stop DataHandler %s", symbol)

        # -------------------------------------------------------------
        # 9. Stop MarketDataEngine
        # -------------------------------------------------------------

        logger.info("[CLEANUP] Stopping MarketDataEngine...")

        try:
            engine.stop()
        except Exception:
            logger.exception(
                "[CLEANUP] Failed to stop MarketDataEngine."
            )

        # MarketDataEngine.start() may still be blocked inside
        # MT5StreamWorker.start() / _loop(). Ensure the worker itself stops.
        if engine.worker is not None:
            try:
                engine.worker.stop()
                logger.info("[CLEANUP] MT5StreamWorker stop requested.")
            except Exception:
                logger.exception("[CLEANUP] Failed to stop MT5StreamWorker.")

        # -------------------------------------------------------------
        # 10. Wait for worker thread
        # -------------------------------------------------------------

        if engine_thread.is_alive():
            engine_thread.join(timeout=10.0)

        if engine_thread.is_alive():
            logger.warning(
                "[CLEANUP] MarketDataEngine thread did not "
                "terminate within timeout."
            )
        else:
            logger.info("[CLEANUP] MarketDataEngine thread stopped.")

# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception:
        logger.exception("`LIVE FULL-PATH TEST FAILED.`")
        raise

# ============================================================================= END