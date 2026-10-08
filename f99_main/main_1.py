from f03_data.market_data_engine.market_data_engine_2 import MarketDataEngine
from f03_data.OLD.data_handler import DataHandler
from f03_data.OLD.mt5_data_loader import MT5DataLoader_batch
from f02_utils.config_loader import load_config
from datetime import time

# راه‌اندازی کامل سیستم
cfg = load_config()
engine = MarketDataEngine(cfg)

# DataHandler با EventBus یکپارچه
data_handler = DataHandler(cfg, event_bus=engine.get_event_bus())
engine.attach_data_handler(data_handler)

# دانلودر نیز با EventBus یکپارچه
downloader = MT5DataLoader_batch(cfg, event_bus=engine.get_event_bus())

# شروع استریم زنده
engine.start(symbols=["EURUSD", "XAUUSD"], timeframes=["M5", "H1"])



def main():
    cfg = load_config()
    engine = MarketDataEngine(cfg)
    engine.start(
        symbols=["EURUSD", "XAUUSD"],
        timeframes=["M5", "H1"],
        poll_interval_sec=2.0
    )
    
    # منتظر ماندن برای Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        engine.stop()

if __name__ == "__main__":
    main()