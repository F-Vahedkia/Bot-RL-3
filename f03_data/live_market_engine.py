# f03_data/live_market_engine.py (5)
#
# Last reviewed: 1405/06/25
# =============================================================================
""" ---> Docstring:
موتور داده زنده بازار برای لایه f03_data.

این ماژول مسیر live را از دریافت دوره‌ای داده از MT5 تا انتشار event برای مصرف‌کنندگان
اجرا می‌کند. معماری آن شامل EventBus، CandleDetector، MT5StreamWorker و MarketDataEngine است.

اجزای اصلی:
    - EventBus: پیاده‌سازی Pub/Sub با Queue مستقل برای هر subscriber و دسترسی thread-safe.
        در مسیر publish() فعلی، رویداد به صف subscriber متناظر با symbol ارسال می‌شود و در صورت
        پر بودن صف، event کنار گذاشته می‌شود.
    - CandleDetector: برای هر ترکیب symbol/timeframe وضعیت آخرین timestamp را نگه می‌دارد و
        هنگام مشاهده timestamp جدید، event داده‌ای مربوط به آخرین کندل بسته را تولید می‌کند.
    - MT5StreamWorker: با polling از MT5 داده می‌گیرد، کندل جاری را کنار می‌گذارد، بسته‌شده‌ها
        را بررسی می‌کند، و هنگام کشف کندل جدید داده تمام timeframeهای موردنیاز آن symbol را
        دریافت و به EventBus منتشر می‌کند.
    - MarketDataEngine: EventBus و worker را orchestration می‌کند و ورودی اصلی مسیر live است.

قرارداد event فعلی:
{
    "event_type": str,
    "symbol": str,
    "timeframe": str,
    "all_dfs": Dict[str, pd.DataFrame],
}

کلیدهای all_dfs به صورت "SYMBOL:TF" و در ارتباط با warmups_dicts ساخته می‌شوند.
DataFrameهای موجود در all_dfs شامل کندل‌های بسته‌شده موردنیاز هر timeframe هستند.

قرارداد زمانی:
    - broker_timezone از project.broker_timezone گرفته می‌شود.
    - CandleDetector زمان کندل را برای event به UTC تبدیل می‌کند.
    - MT5StreamWorker می‌تواند result_timezone را از download_defaults بخواند و آن را به connector
        منتقل کند؛ مسیرهای بالادستی در پروژه فعلی داده را در چارچوب UTC مصرف می‌کنند.


این ماژول منبع اصلی رویدادهای live برای DataHandler و سایر مصرف‌کنندگان بالادستی است.
"""
# =============================================================================
# """
# نهایی هستند:
#    - API عمومی
#    - معماری Pub/Sub
#    - یک Queue برای هر Consumer
#    - Thread-safe
#    - publish()
#    - subscribe()
#    - unsubscribe()

# هنوز نهایی نشده اند:
#    - event payload schema
#    - overflow policy
#    - shutdown
#    - health monitoring
#    - backpressure
#    - metrics
#    - consumer naming
#    - event persistence
# """

# =============================================================================
# Imports
# =============================================================================
# f03_data/live_market_engine.py

from __future__ import annotations

import time
import logging
import pandas as pd
import uuid
from dataclasses import dataclass
from queue import Queue, Full, Empty
from threading import Lock
from typing import Any, Dict, Optional
# ---------------------------
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
# ---------------------------
from f03_data.mt5_connector import MT5Connector
from f02_utils.functions._to_zoneinfo import _to_zoneinfo
# from f03_data.data_handler_F_3 import DataHandler

# =============================================================================
# Logging
# =============================================================================
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =============================================================================
# 1) EventBus
# =============================================================================
class EventBus:
    """
    Thread-safe Pub/Sub EventBus.
    هر subscriber یک queue اختصاصی دریافت می‌کند.
    Producer:
        publish(...)
    Consumer:
        queue = subscribe(...)
        event = queue.get(...)
    """
    # --------------------------------------------------------- OK
    def __init__(self, queue_size: int = 1000) -> None:
        """
        در ابتدای ساخت یک شیئ از این کلاس، این موارد معلوم یا ساخته میشود:
            - اندازه طول هر صف
            - دیشکنری خالی برای مصرف کنندگان، که شماره اشتراک را میگیرد و صف را برمیگرداند
            - یک کلید، برای جلوگیری از ایجاد تداخل ها در مواقع ضروری
        """
        self._queue_size = queue_size

        # subscriber_id -> Queue
        self._subscribers: Dict[str, Queue] = {}
        self._lock = Lock()

    # --------------------------------------------------------- OK
    def subscribe_by_id(self) -> str:
        """
        هربار که این متد فراخوانی شود، در دیکشنری مصرف کنندگان:
            - key: یک آیدی رشته ای به عنوان کد مصرف کننده جدید تولید میکند
            - value: یک صف اختصاصی برای آن مصرف کننده ایجاد میکند

        Create dedicated queue for consumer.
        یعنی: ایجاد صف اختصاصی برای مصرف کننده.
        Returns:
            subscriber_id
        """
        subscriber_id = str(uuid.uuid4()) 
        with self._lock:
            self._subscribers[subscriber_id] = Queue(maxsize=self._queue_size)
        logger.info("EventBus subscriber registered: %s", subscriber_id)
        return subscriber_id
    
    # -------------
    def subscribe(self, symbol: str) -> str:
        """
        ثبت‌نام برای دریافت رویدادهای یک نماد خاص.
        اگر symbol = "all" باشد، همه رویدادها را دریافت می‌کند.
        """
        with self._lock:
            if symbol not in self._subscribers:
                self._subscribers[symbol] = Queue(maxsize=self._queue_size)
            else:
                logger.warning(f"Symbol '{symbol}' already subscribed.")
        return symbol  # خود symbol به عنوان شناسه برمی‌گردد
    
    # --------------------------------------------------------- OK
    def unsubscribe_by_id(self, subscriber_id: str) -> None:
        """
        این متد، آیدی مصرف کننده را از دیکشنری مصرف کنندگان حذف میکند.
        """
        with self._lock:
            # pop: متدی است برای حذف یک کلید از دیکشنری و دریافت مقدار مربوط به آن
            self._subscribers.pop(subscriber_id, None)
        logger.info("EventBus subscriber removed: %s", subscriber_id)
    
    # -------------
    def unsubscribe(self, symbol: str) -> None:
        with self._lock:
            self._subscribers.pop(symbol, None)
        logger.info("EventBus subscriber removed: %s", symbol)

    # --------------------------------------------------------- OK
    def get_queue_by_id(self, subscriber_id: str) -> Optional[Queue]:
        """
        این متد، صف مربوط به یک مصرف کننده را برمی گرداند
        """
        with self._lock:
            return self._subscribers.get(subscriber_id)
    
    # -------------
    def get_queue(self, symbol: str) -> Optional[Queue]:
        """
        این متد، صف مربوط به یک مصرف کننده را برمی گرداند
        """
        with self._lock:
            return self._subscribers.get(symbol)
        
    # --------------------------------------------------------- OK
    def publish_by_id(self, event_type: str, symbol: str, timeframe: str, all_dfs: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        این متد، بدون صبر کردن، -دیکشنری- رویداد را در صف های همه مصرف کنندگان قرار می دهد
        """
        event = {
            "event_type": event_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "all_dfs": all_dfs
        }
        # ✅ ایمن - توسط ویت، ابتدا یک کپی از وضعیت فعلی می‌گیریم
        with self._lock:
            subscribers = list(self._subscribers.items())

        # ✅ حالا می‌توانیم بدون قفل، روی لیست حلقه بزنیم
        for subscriber_id, q in subscribers:
            try:
                q.put_nowait(event) # put_nowait: متدی ،برای قرار دادن یک آیتم در صف، بدون منتظر ماندن است
                # logger.info("put event dict in queue")
            except Full:            # Full: استثنائی است که وقتی صف پُر باشد، توسط متد put_nowait() تولید میشود
                logger.warning(
                    "Subscriber queue full. "
                    "Dropping event. "
                    "subscriber=%s "
                    "event=%s",
                    subscriber_id,
                    event_type,
                )

    # -------------
    def publish(self, event_type: str, symbol: str, timeframe: str, all_dfs: Dict[str, Dict[str, Any]]) -> None:
        event = {
            "event_type": event_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "all_dfs": all_dfs
        }
        with self._lock:
            # ارسال به همان نماد
            q = self._subscribers.get(symbol)
            if q:
                try:
                    q.put_nowait(event)
                except Full:
                    logger.warning("Queue full for %s. Dropping event.", symbol)

    # --------------------------------------------------------- OK
    def get_event_by_id(self, subscriber_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        این متد، از صف یک مصرف کننده معلوم، یک -دیکشنری- رویداد را دریافت میکند
        """
        q = self._subscribers.get(subscriber_id) # q: مخفف است برای queue
        if q is None:
            return None
        try:
            return q.get(timeout=timeout)
        except Empty:  # Empty: استثنائی است که وقتی تایم اوت تمام شد، توسط متد get() تولید میشود
            return None
    
    # -------------
    def get_event(self, symbol: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        این متد، از صف یک مصرف کننده معلوم، یک -دیکشنری- رویداد را دریافت میکند
        """
        q = self.get_queue(symbol) # q: مخفف است برای queue
        if q is None:
            return None
        try:
            return q.get(timeout=timeout)
        except Empty: # Empty: استثنائی است که وقتی تایم اوت تمام شد، توسط متد get() تولید میشود
            return None

    # --------------------------------------------------------- OK
    def subscriber_count(self) -> int:
        """
        تعداد مصرف کنندگان را برمیگرداند
        """
        with self._lock:
            return len(self._subscribers)

    # --------------------------------------------------------- OK
    def stats(self) -> Dict[str, int]:
        """
        یک دیکشنری برمیگرداند که کلیدها، شماره های مصرف کنندگان است و
        مقادیر، برابر است با تعداد رویدادهای موجود در صف آن مصرف کننده
        """
        with self._lock:
            return {              # sid: مخفف شده است برای subscriber_id
                sid: q.qsize()    # qsize(): متدی برای گرفتن تعداد آیتم‌های موجود در صف، در همان لحظه است
                for sid, q in self._subscribers.items()
            }


# =============================================================================
# 2) Candle State Tracker
# =============================================================================
@dataclass
class CandleState:
    """
    نگهداری آخرین وضعیت کندل برای تشخیص بسته شدن کندل جدید
    """
    last_candle_time: Optional[datetime] = None

# =============================================================================
# 2) Candle Detector
# =============================================================================
class CandleDetector:
    """
    Detects newly closed candles from streaming OHLCVS data.
    Logic:
        - receive latest dataframe snapshot
        - compare last timestamp
        - emit event only when a NEW candle is closed
    """
    # -------------------------------------------------------- OK
    def __init__(self, tzinfo) -> None:
        """
        در ابتدای ساخت یک شیئ از این کلاس، این موارد معلوم یا ساخته میشود:
            - یک دیکشنری خالی به نام _state که :
                کلیدهای آن رشته های "نماد-تایمفریم" هستند و 
                مقادیر آن شیئی از کلاس CandleState است.
        """
        self._state: Dict[str, CandleState] = {}
        self.tzinfo = tzinfo

    # -------------------------------------------------------- OK
    def _get_state(self, symbol: str, timeframe: str) -> CandleState:
        """
        در ابتدا بررسی میکند که کلید "نماد-تایمفریم" در دیکشنری _state وجود دارد یا نه
        اگر وجود نداشته باشد، کلید مزبور را میسازد و همراه با مقدار متناظر با آن کلید در دیکشنری قرار میدهد
        اگر وجود داشته باشد، مقدار معادل با ان کلید را برمیگرداند.
        """
        key = f"{symbol}:{timeframe.upper()}"
        if key not in self._state:
            self._state[key] = CandleState()
        return self._state[key]

    # --------------------------------------------------------
    def detect(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        این متد، در صورت تشخیص کندل جدید، دیکشنری مربوطه را برمیگرداند
        در غیر اینصورت نَن را برمیگرداند
        Returns event "payload" if a new candle is detected.
        Otherwise returns None.
        """
        if df is None or df.empty:
            return None

        state = self._get_state(symbol, timeframe)

        # در تابع _fetch_closed این بازه خروجی داده شده است: df=[-n-1:-1]
        # یعنی در آنجا، اِن-تا کندل ((بسته شده،)) در قالب یک دیتافریم، برگردانده شده است
        if len(df) == 0:
            return None
    
        # --- گرفتن داده های آخرین کندل بسته شده
        last_time_raw = df.index[-1]

        # --- first run -------------------------
        if state.last_candle_time is None:
            state.last_candle_time = last_time_raw
            return None
        
        # --- no new candle ---------------------
        if last_time_raw <= state.last_candle_time:
            return None

        # --- کندل جدید تشخیص داده شد ---------
        # فقط در اینجا زمان را یک بار تبدیل کن
        try:
            # تبدیل به Timestamp اگر نبود
            if not isinstance(last_time_raw, pd.Timestamp):
                last_time = pd.to_datetime(last_time_raw)
            else:
                last_time = last_time_raw
            # در مورخ 1405/06/02-11:40 بررسی کردم و دیدم که در این تقطه last_time is UTC and is aware
            
            # اضافه کردن منطقه زمانی بروکر (اگر naive باشد)
            if last_time.tzinfo is None:
                last_time = last_time.tz_localize(self.tzinfo)

            # تبدیل به UTC
            last_time = last_time.tz_convert("UTC")

            # تبدیل به datetime پایتون برای خروجی
            if hasattr(last_time, 'to_pydatetime'):
                last_time = last_time.to_pydatetime()

        except Exception as e:
            logger.error(f"Time conversion failed for {symbol}/{timeframe}: {e}")
            return None

        # --- update state ----------------------
        state.last_candle_time = last_time_raw
        logger.info(
            f"New candle detected {symbol}/{timeframe} at "
            f"{last_time} UTC / {last_time_raw.tz_convert(self.tzinfo)} Broker")
    
        # --- استخراج داده های کندل ------------
        # داده‌های کندل بسته شده را از ردیف ماقبل آخر بگیر
        # در تابع _fetch_closed این بازه خروجی داده شده است: df=[-n-1:-1]
        # یعنی در آنجا، 3 تا کندل ((بسته شده،)) در قالب یک دیتافریم، برگردانده شده است
        row = df.iloc[-1]
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candle_time": last_time,
            "open"  : float(row["open"  ]) if "open"   in df.columns else None,
            "high"  : float(row["high"  ]) if "high"   in df.columns else None,
            "low"   : float(row["low"   ]) if "low"    in df.columns else None,
            "close" : float(row["close" ]) if "close"  in df.columns else None,
            "volume": float(row["volume"]) if "volume" in df.columns else None,
            "spread": float(row["spread"]) if "spread" in df.columns else None,
        }
    

# =============================================================================
# 3) MT5 Live Stream Worker
# =============================================================================
class MT5StreamWorker:
    """
    Live polling worker:  (polling: کشیدن  ,  pushing: فشار دادن)
    """
    # -------------------------------------------------------- OK
    def __init__(
        self,
        cfg: Dict[str, Any],
        event_bus: EventBus,
        warmups_dicts: Dict[str, Dict[str, int]],   #////change_1405/05/20-16:30
        poll_interval_sec: float = 2.0,
    ) -> None:
        """
        در ابتدا این اشیائ را میسازد: MT5Connector, CandleDetector
        """
        self.cfg = cfg
        self.event_bus = event_bus

        self.warmups_dicts = warmups_dicts   #////change_1405/05/20-16:30
        # self.warmups_dicts: Dict[str, Dict[str, int]] = cfg["__warmups_dicts"]

        self.symbols = list(self.warmups_dicts.keys())
        self.timeframes_dict = {sym: list(warmup.keys()) for sym, warmup in self.warmups_dicts.items()}
        self.poll_interval_sec = poll_interval_sec
        self.connector = MT5Connector(config=cfg)

        # --- broker_timezone ----------------------------- start
        # project_cfg = self.cfg.get("project")
        # if not project_cfg:
        #     raise ValueError("'project' key not found in config !")
            
        # self.broker_timezone = project_cfg.get("broker_timezone")
        # if not self.broker_timezone:
        #     raise ValueError("'broker_timezone' key not set in 'project' key!")

        # اگر بلوک زیری جواب بدهد، باید بلوک بالایی را حذف کنم.
        project_cfg = self.cfg.get("project")
        if not project_cfg:
            raise ValueError("'project' key not found in config !")

        broker_timezone = project_cfg.get("broker_timezone")
        if not broker_timezone:
            raise ValueError("'broker_timezone' key not found or empty in 'project' config!")

        try:
            self.broker_timezone = ZoneInfo(broker_timezone)
        except ZoneInfoNotFoundError:
            raise ValueError(f"Invalid broker_timezone: '{broker_timezone}'")
        
        # --- result_timezone -----------------------------------
        dl = (self.cfg.get("download_defaults") or {})
        temp = dl.get("result_timezone")

        if temp is None:
            self.result_timezone = None
        else:
            self.result_timezone = _to_zoneinfo(temp)

        # ------------------------------------------------- end
        self.detector = CandleDetector(self.broker_timezone)
        self._running = False
       
    # -------------------------------------------------------- OK
    def start(self) -> None:
        """
        این متد اتصال به متاتریدر را راه اندازی میکند و 
        حلقه اصلی را شروع میکند.
        در نهایت همیشه اتصال به متاتریدر را قطع میکند
        """
        if not self.connector.initialize():
            raise RuntimeError("MT5 connection failed")
        self._running = True # کنترل اجرای حقله- مادامیکه صحیح است، حلقه اجرا میشود
        logger.info("MT5StreamWorker started")
        try:
            self._loop() # حلقه پایش بازار
        finally:
            self.connector.shutdown()
            logger.info("MT5StreamWorker stopped")

    # -------------------------------------------------------- OK
    def _loop(self) -> None:
        """
        این متد این کارها را انجام میدهد:
            - پایش بازار هر چند ثانیه یکبار توسط دانلود کندل جدید
            - در صورت تشخیص کندل جدید، دانلود تمام تایم فرمهای مورد نیاز فیچرها در آن نماد
            -
        """
        logger.debug("===> start loop at streamworker")
        logger.debug(f"{self.symbols}")
        
        while self._running:
            # ---------------------------------/
            for symbol in self.symbols:
                for tf in self.timeframes_dict[symbol]:
                    # -------------------------//
                    try:
                        df = self._fetch_closed(symbol, tf, 3)
                        event = self.detector.detect(symbol, tf, df)  # will be "None" or "dict"
                        # event: is a dict contains: symbol, timeframe, candle_time, O,H,L,C,V,S
                        if event:
                            # سطر زیر:
                            # در تمام تایمفریهای وارم آپ، دیتاها را دانلود نموده
                            # و همگی را در قالب یک دیکشنری برمیگرداند
                            all_dfs = self._fetch_all_tfs(symbol)
                            self.event_bus.publish(
                                event_type="NEW_CANDLE",
                                symbol=symbol,
                                timeframe=tf,
                                all_dfs=all_dfs,
                            )

                    except Exception as ex:
                        logger.exception("Stream error %s/%s: %s", symbol, tf, ex)
                    # -------------------------//
            # ---------------------------------/
            time.sleep(self.poll_interval_sec)

    # -------------------------------------------------------- OK
    def _fetch_closed(self, symbol: str, timeframe: str, num_candles: Optional[int] = None) -> pd.DataFrame:
        """
        این متد فقط وظیفه دانلود داده های کندلی ((بسته شده)) را به عهده دارد.
        و فقط تعداد n کندل بسته شده نهایی را برمیگرداند.
        """
        n = num_candles if num_candles is not None else (self.cfg.get("executor") or {}).get("lookback_bars", 30)
        df = self.connector.get_candles_num(
            symbol=symbol,
            timeframe=timeframe,
            num_candles=n+1,
            result_tz=self.result_timezone
        )
        # سطر زیر کندل -1 را برنمی گرداند. چون هنوز بسته نشده است و کندل جاری است.
        result = df.iloc[-n-1:-1] if df is not None else pd.DataFrame()
        # logger.info(f"Candles for {symbol}/{timeframe} was lowmloaded. rows = {len(result)}")
        return result

    # -------------------------------------------------------- OK
    def _fetch_all_tfs(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        مثال برای خروجی این تابع:
        all_dfs = {
            "XAUUSD:M1" : DataFrame of "XAUUSD", at "M1" , contains (self.warmups_dicts["XAUUSD"]["M1" ]) closed candles
            "XAUUSD:M15": DataFrame of "XAUUSD", at "M15", contains (self.warmups_dicts["XAUUSD"]["M15"]) closed candles
            "XAUUSD:H1" : DataFrame of "XAUUSD", at "H1" , contains (self.warmups_dicts["XAUUSD"]["H1" ]) closed candles
            "XAUUSD:H4" : DataFrame of "XAUUSD", at "H4" , contains (self.warmups_dicts["XAUUSD"]["H4" ]) closed candles
        }
        """

        all_dfs = {
            f"{symbol}:{tf.upper()}": self._fetch_closed(symbol, tf, self.warmups_dicts[symbol][tf])
            for tf in self.warmups_dicts[symbol]
        }
        return all_dfs
    
    # -------------------------------------------------------- OK=
    def stop(self) -> None:
        """
        این متد، چوب لای چرخ حلقه اصلی میکند
        """
        self._running = False


# =============================================================================
# 4) Market Data Engine (Orchestrator)
# =============================================================================
class MarketDataEngine:
    """
    Central orchestrator of live market data pipeline.
    """
    # -------------------------------------------------------- OK
    def __init__(self, cfg: Dict[str, Any]) -> None:
        """
        یک شیئ self.event_bus = EventBus میسازد
        """
        self.cfg = cfg
        self.event_bus = EventBus(
            queue_size=(cfg.get("event_bus") or {}).get("queue_size", 1000) # config-readable
        )
        self.worker: Optional[MT5StreamWorker] = None
        self._running: bool = False
        # self.data_handler: Optional[DataHandler] = None

    # -------------------------------------------------------- OK
    def start(
        self,
        warmups_dicts: Dict[str, Dict[str, int]],   #////change_1405/05/20-16:30
        poll_interval_sec: float = 2.0,
    ) -> None:
        """
        این متد:
            - یک شیئ MT5StreamWorker میسازد و 
            - آنرا استارت میکند
            - وضعیت اجرای شیئ MarketDataEngine را در حالت True قرار میدهد
        """
        if self._running:
            logger.warning("MarketDataEngine already running")
            return

        self.worker = MT5StreamWorker(
            cfg=self.cfg,
            event_bus=self.event_bus,
            warmups_dicts=warmups_dicts,   #////change_1405/05/20-16:30
            poll_interval_sec=poll_interval_sec,
        )
        self.worker.start()
        self._running = True
        
        logger.info("MarketDataEngine starting...")

    # -------------------------------------------------------- OK
    def stop(self) -> None:
        """
        - اجرای شیئ self.worker = MT5StreamWorker را متوقف میکند
        - وضعیت اجرای شیئ MarketDataEngine را در حالت False قرار میدهد
        """
        if not self._running:
            return
        logger.info("MarketDataEngine stopping...")
        if self.worker:
            self.worker.stop()
        self._running = False

    # -------------------------------------------------------- OK
    def get_event_bus(self) -> EventBus:
        """
        شیئ self.event_bus = EventBus را برمیگرداند
        """
        return self.event_bus
    

    # --------------------------------------------------------
    """ چرا فعلاً این تابع را کامنت/حذف کرده ام؟

    چون با جداسازی مسیر نمادها تا ابتدای observation_builder ما
    دارای DataHandler های متعددی خواهیم بود. بنابراین بهتر است
    این کلاس خودش حق انتخاب آبجکت DataHandler را نداشته باشد.
    """

    # def attach_data_handler(self, data_handler: DataHandler) -> None:
        
    #     """اتصال DataHandler به EventBus"""
    #     self.data_handler = data_handler
    #     data_handler.subscribe_to_event_bus(self.event_bus)
    #     logger.info("DataHandler attached to MarketDataEngine")

    # --------------------------------------------------------
# ============================================================================= END
