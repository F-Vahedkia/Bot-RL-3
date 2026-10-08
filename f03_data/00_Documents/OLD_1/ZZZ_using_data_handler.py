
"""
در زیر 3 روش برای  استفاده از داده های زنده یا غیرزنده 
که توسط data_handler فراهم میشوند ارائه شده است.
 روشهای 1 و 2 مناسب هستند و روش 3 فقط و فقط آموزشی است.

"""

############################################################################### 1 +++ for batch
import pandas as pd
from f03_data.data_handler_G import DataHandler, BuildParams
from f02_utils.config_loader import load_config

cfg = load_config()   # کانفیگ اصلی

# 1) ایجاد نمونه از DataHandler (با کانفیگ پیش‌فرض)
handler = DataHandler(cfg)   # یا می‌توانید cfg دلخواه بدهید

# 2) تعیین پارامترهای ساخت دیتاست
params = BuildParams(
    symbol="XAUUSD",
    base_tf="M1",
    timeframes=["M5", "H1", "H4"],   # تایم‌فریم‌های اضافی (اختیاری)
    format_="parquet"
)

# 3) ساخت دیتافریم نهایی (ایندکس = زمان شروع کندل پایه)
df_processed = handler.build(params)

# 4) ذخیره دیتافریم (اختیاری – قبلاً هم در build ذخیره نمی‌شود؛ بلکه باید خودتان save کنید)
output_path = handler.save(df_processed, symbol="XAUUSD", base_tf="M1", fmt="parquet")

# 5) حالا df_processed را می‌توانید برای آموزش RL یا بک‌تست استفاده کنید
print(df_processed.head())
print(f"Shape: {df_processed.shape}")


############################################################################### 2 +++ for live
# ----- بخش 1: راه‌اندازی MarketDataEngine و EventBus -----
import time
import pandas as pd
from f03_data.live_market_engine import MarketDataEngine
from f03_data.data_handler_G import DataHandler
from f02_utils.config_loader import load_config

cfg = load_config()   # کانفیگ اصلی

# 1) ساختن DataHandler
data_handler = DataHandler(cfg)

# 2) ساختن موتور داده زنده
engine = MarketDataEngine(cfg)

# 2.1) اتصال موتور داده زنده به DataHandler
engine.attach_data_handler(data_handler)                      
# data_handler.subscribe_to_event_bus(engine.get_event_bus)

# متغیر شرط برای ادامه حلقه (True یعنی ادامه بده)
keep_running = True

# 3) (اختیاری) یک تابع callback برای دریافت دیتافریم‌های به‌روز تعریف کنید
def on_new_data(df: pd.DataFrame):
    global keep_running
    # این تابع هر بار که یک کندل جدید base_tf بسته می‌شود، فراخوانی می‌شود
    # df همان دیتافریم کامل (شامل همه تایم‌فریم‌ها) است که ایندکس آن زمان شروع کندل پایه می‌باشد
    print("New data received:")
    print(df.tail(2))
    # ... اینجا می‌توانید به agent RL بدهید یا معامله کنید

    # بعد از دریافت اولین دیتا، حلقه را متوقف کن
    keep_running = False


data_handler.set_data_callback(on_new_data)

# 4) شروع موتور (با نمادها و تایم‌فریم‌های مورد نظر)
engine.start(
    symbols=["XAUUSD"],
    timeframes=["M1", "M5", "H1"],   # تمام تایم‌فریم‌هایی که نیاز دارید
    poll_interval_sec=2.0
)

# 5) در یک حلقه اصلی برنامه منتظر بمانید (یا از signal برای توقف استفاده کنید)
try:
    # while True:
    while keep_running:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
finally:
    engine.stop()


############################################################################### 3 --- for live (+++)
import time
import threading
from f03_data.live_market_engine import MarketDataEngine
from f03_data.data_handler_G import DataHandler
from f02_utils.config_loader import load_config

cfg = load_config()
data_handler = DataHandler(cfg)
engine = MarketDataEngine(cfg)

# اتصال DataHandler به EventBus (این کار فقط subscribe را انجام می‌دهد)
engine.attach_data_handler(data_handler)

# تعریف callback برای دریافت دیتافریم‌های به‌روز
def on_new_data(df):
    print("New data received:", df.tail(2))
data_handler.set_data_callback(on_new_data)

# راه‌اندازی حلقه مصرف رویدادها در یک ترد جداگانه
threading.Thread(target=data_handler.start_consuming,
                 daemon=True
                 ).start()

# شروع موتور
# راه‌اندازی موتور در یک ترد جداگانه (چون start مسدود است)
threading.Thread(target=engine.start,
                 args=(["XAUUSD"], ["M1"]),
                 kwargs={"poll_interval_sec": 2.0},
                 daemon=True
                 ).start()

# نگه‌داشتن برنامه
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    engine.stop()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++
latest_df = None

def on_new_data(df):
    global latest_df
    latest_df = df.copy()
    print("New data received:", df.tail(2))

# حالا هر جا که نیاز دارید، از latest_df استفاده کنید

############################################################################### 4 --- for live
import time
import logging
from datetime import datetime, timezone
import pandas as pd

from f03_data.mt5_connector import MT5Connector          # برای دریافت کندل از MT5
from f03_data.data_handler_G import DataHandler, BuildParams

# تنظیم لاگ (اختیاری)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# 1) آماده‌سازی DataHandler
# ------------------------------
cfg = {
    # کانفیگ ساده (معمولاً از config.yaml لود می‌شود)

    "download_defaults": {
        "symbols": ["XAUUSD"],
        "timeframes": ["M1", "M5", "H1"],
        "save_format": "parquet",
    },
    "features": {
        "base_timeframe": "M1",
        "time_features": {
            "add_hour_of_day": True,
            "add_day_of_week": True,
            "add_session_flags": True,
        }
    }
}

handler = DataHandler(cfg=cfg)

# (اختیاری) برای ذخیره دیتافریم به‌روز شده در یک فایل یا متغیر
latest_df = None

def on_new_data(df: pd.DataFrame):
    """این تابع هر بار که update_live یک دیتافریم جدید برگرداند، فراخوانی می‌شود."""
    global latest_df
    latest_df = df
    logger.info(f"New live data received: shape={df.shape}, last index={df.index[-1]}")
    # در اینجا می‌توانید به agent RL بدهید یا معامله کنید
    print(df.tail(2))

handler.set_data_callback(on_new_data)

# ------------------------------
# 2) اتصال مستقیم به MT5 برای دریافت کندل‌ها (به جای EventBus)
# ------------------------------
connector = MT5Connector(config=cfg)
if not connector.initialize():
    raise RuntimeError("MT5 connection failed")

symbol = "XAUUSD"
base_tf = "M1"   # همان default_base_tf

# ------------------------------
# 3) حلقه اصلی: هر چند ثانیه یک بار کندل جدید را چک می‌کنیم
# ------------------------------
last_fetched_time = None

try:
    while True:
        # دریافت آخرین کندل‌ها (مثلاً 2 کندل آخر برای تشخیص جدید بودن)
        df_candles = connector.get_candles_num(symbol, base_tf, num_candles=2)
        if df_candles.empty:
            time.sleep(2)
            continue
        
        # آخرین زمان شروع کندل (ایندکس دیتافریم خام = زمان شروع)
        latest_start = df_candles.index[-1]   # زمان شروع آخرین کندل
        
        if last_fetched_time is None or latest_start > last_fetched_time:
            # کندل جدیدی بسته شده است (آخرین کندل کامل، کندل ماقبل آخر است)
            # اما در get_candles_num، آخرین ردیف همان کندل جاری ناقص است.
            # بنابراین باید از ردیف ماقبل آخر استفاده کنیم (کندل بسته شده)
            if len(df_candles) >= 2:
                closed_candle = df_candles.iloc[-2]   # کندل بسته شده (قدیمی‌تر)
                candle_start = closed_candle.name     # ایندکس = زمان شروع
                
                # ساخت payload شبیه به رویداد NEW_CANDLE
                payload = {
                    "symbol": symbol,
                    "timeframe": base_tf,
                    "candle_time": candle_start,      # زمان شروع کندل
                    "open": closed_candle['open'],
                    "high": closed_candle['high'],
                    "low": closed_candle['low'],
                    "close": closed_candle['close'],
                    "volume": closed_candle['volume'],
                    "spread": closed_candle['spread'],
                }
                
                # فراخوانی دستی update_live – اینجا متد اجرا می‌شود
                df_updated = handler.update_live(symbol, base_tf, payload)
                # (اختیاری) می‌توانید df_updated را مستقیم هم بگیرید
                
                last_fetched_time = latest_start
                logger.debug(f"Updated live cache, new row at {candle_start}")
            else:
                # هنوز یک کندل کامل نداریم
                pass
        
        time.sleep(2)   # هر 2 ثانیه یکبار چک کن

except KeyboardInterrupt:
    logger.info("Stopping...")
finally:
    connector.shutdown()

###############################################################################