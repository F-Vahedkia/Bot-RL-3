# f03_data/tests_live_market_engine/test_6_live_market_engine_integrity_1.py
# Date Reviewed:
#    1405/05/25-07:00 ==> run result is OK.

# Run: python -m f03_data.tests_live_market_engine.test_6_live_market_engine_integrity_1

"""
    این ماژول، هماهنگی و اقدام مشترک چهار کلاس:
        - event_bus.py
        - candle_detector.py
        - mt5_stream_worker.py
        - market_data_engine.py
    که همگی در فایل live_market_engine.py قرار دارند را تست میکند و 
    تا ابتدای اتصال به data_handler.py پیش میرود.
    یعنی تا آنجا که رویداد اتفاق افتاده را در سطر زیر
    event = event_bus.get_event(data_handlers[sym]._subscription_key, timeout=0.1)
    از data_handler میگیرد.
"""
# f03_data/tests_live_market_engine/test_6_live_market_engine_integrity_1.py
# =======================================================================================
# Imports
# =======================================================================================
from typing import Dict
import threading
import logging
# from zoneinfo import ZoneInfo
from f03_data.live_market_engine import MarketDataEngine
from f03_data.data_handler_G import DataHandler
from f02_utils.config_completer import config_completer

# =======================================================================================
# Logging
# =======================================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-6s | %(filename)-40s | %(lineno)-4d : %(funcName)-20s | %(message)s"
)
logger = logging.getLogger("tester")

# =======================================================================================
# main
# =======================================================================================
def main():
    # -------------------------------------------
    # بدست آوردن پیش نیازها
    #     1) config
    #     2) broker_timezone
    #     3) symbols
    #     4) timeframes_dict
    # -------------------------------------------
    # 1) --- config
    cfg = config_completer()

    # 2) --- broker timezone
    broker_timezone = cfg["project"]["broker_timezone"]
    logger.info(f"broker timezone = {broker_timezone}")

    # 3) --- warmups dicts
    warmups_dicts: Dict[str, Dict[str, int]] = cfg["__all_required_bars"]
    # logger.info(f"warmups dicts = {warmups_dicts}")

    # 4) --- symbols
    # symbols = list(warmups_dicts.keys())
    symbols = cfg["__symbols"]
    logger.info(f"symbols = {symbols}")

    # 5) --- ساخت Scope مستقل required_bars برای هر symbol
    #
    # فقط timeframeهایی وارد Scope می‌شوند که واقعاً تعداد candle
    # موردنیاز مثبت دارند.
    required_bars_dicts: Dict[str, Dict[str, int]] = {}

    for sym in symbols:
        source = warmups_dicts.get(sym, {})

        required_bars_dicts[sym] = {
            tf.upper().replace(" ", ""): int(bars)
            for tf, bars in source.items()
            if isinstance(bars, int) and not isinstance(bars, bool) and bars > 0
        }

        if not required_bars_dicts[sym]:
            raise ValueError(
                f"No positive required bars found for symbol '{sym}'."
            )

    logger.info(f"required_bars_dicts = {required_bars_dicts}")

    # -------------------------------------------
    # ساخت موتور و گرفتن event_bus که نقش دفتر یادداشت مشترکین را دارد
    # -------------------------------------------
    # 6) --- MarketDataEngine
    engine = MarketDataEngine(cfg, )
    event_bus = engine.get_event_bus()



    logger.info("✅ Tester started. Press Ctrl+C to stop.")

    # 7) --- loop over all symbols to create DataHandlers
    data_handlers = {}
    for sym in symbols:
        data_handlers[sym] = DataHandler(cfg, symbol=sym, required_bars=required_bars_dicts[sym])    #, event_bus=None)

        # engine.attach_data_handler(data_handlers[sym])      # متد اتصال موتور به دیتا هندلر که در کلاس موتور بود را حذف کردم تا ایمپورت حلقه ای بوجود نیاید
        data_handlers[sym].subscribe_to_event_bus(event_bus)  # برای اتصال موتور و دیتاهندلر، باید از متد اتصالی که در دیتاهندلر است استفاده بشود

    # 8) --- Start Engine (یک بار برای همه نمادها)
    # threading.Thread(target=engine.start, args=(2.0), daemon=True).start()
    threading.Thread(
        target=engine.start,
        args=(required_bars_dicts,),
        daemon=True
    ).start()        


    # 9) --- Event loop for all symbols
    last_printed_time = {}

    while True:
        # برای هر نماد چک کن
        for sym in symbols:
            # event = event_bus.get_event(sym, timeout=0.1)
            event = event_bus.get_event(data_handlers[sym]._subscription_key, timeout=0.1)
            if event and event.get("event_type") == "NEW_CANDLE":
                # if sym == symbols[0]: print("="*60, "Symbols from first") # for debug

                tf = event.get("timeframe")
                all_dfs = event.get("all_dfs", {})
                key = f"{sym}:{tf}"
                df = all_dfs.get(key)

                if df is not None and not df.empty:
                    last_row = df.iloc[-1]      # Recent closed candle OHLCVS
                    candle_time = df.index[-1]  # Recent closed candle time

                    if last_printed_time.get(f"{sym}_{tf}") != candle_time:
                        last_printed_time[f"{sym}_{tf}"] = candle_time

                        print(f"==========////=====> {candle_time}")  # for debug

                        logger.info(f"\n📊 New Candle [{sym}/{tf}] at {candle_time}")

                        print(f"   time : {candle_time}")

                        print(f"   Open  : {last_row.get('open', 'N/A'):.5f}")
                        print(f"   High  : {last_row.get('high', 'N/A'):.5f}")
                        print(f"   Low   : {last_row.get('low', 'N/A'):.5f}")
                        print(f"   Close : {last_row.get('close', 'N/A'):.5f}")
                        print(f"   Volume: {last_row.get('volume', 'N/A'):.5f}")
                        print(f"   Spread: {last_row.get('spread', 'N/A'):.5f}")
                        print("-" * 40)

# =======================================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")

# ======================================================================================= END
