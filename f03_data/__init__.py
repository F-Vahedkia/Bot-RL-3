"""
[MT5 Connector]
       ↓
[MT5StreamWorker] ← Polling هر ۲ ثانیه
       ↓
[CandleDetector] ← تشخیص کندل جدید
       ↓
[EventBus] ← انتشار رویداد "NEW_CANDLE"
       ↓
[DataHandlerLiveConsumer] ← مصرف رویدادها
       ↓
[DataHandler] ← پردازش استراتژی و سیگنال‌ها

"""
# =========================================================

"""
جریان اتصال صحیح:

1. MarketDataEngine.start() (فایل 4)
      ↓
2. MT5StreamWorker.__init__() (فایل 3)
      ↓
3. MT5Connector(cfg) (فایل mt5_connector)
      ↓
4. connector.initialize() → اتصال به MT5
      ↓
5. شروع polling داده
"""