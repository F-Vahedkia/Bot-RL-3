# f03_data/mt5_connector.py (1)
#
# Last reviewed: 1405/06/25
# =======================================================================================
""" ---> Docstring:
اتصال‌دهنده MetaTrader 5 (MT5) برای لایه داده ربات Bot-RL-3.

این ماژول مسئول برقراری، پایش و قطع اتصال به ترمینال MT5 و دریافت
داده‌های کندلی برای نمادها و تایم‌فریم‌های موردنیاز لایه داده است.

مسئولیت‌های اصلی:
    - خواندن مشخصات اتصال و گزینه‌های retry از config و ساخت یک اتصال قابل‌کنترل به MT5.
    - انجام initialize/login با retry و backoff و در صورت نیاز انتخاب خودکار نمادهای پیکربندی‌شده.
    - اطمینان از معتبر بودن اتصال پیش از فراخوانی‌های داده‌ای با ensure_connection().
    - دریافت کندل‌ها بر اساس تعداد با get_candles_num() یا بر اساس بازه زمانی با
        get_candles_range().
    - نرمال‌سازی ساختار نرخ‌های MT5 به ستون‌های استاندارد time/open/high/low/close/
        volume/spread بدون تحمیل timezone نهایی در مرحله دریافت.
    - امکان تبدیل timezone خروجی با پارامتر result_tz؛ در حالت عدم تعیین، زمان خروجی
        به‌صورت naive باقی می‌ماند و مصرف‌کننده مسئول سیاست timezone خود است.
    - فراهم‌کردن health_check() و get_symbol_specs() برای بررسی وضعیت حساب، دسترسی داده
        و مشخصات قرارداد نماد.

قرارداد زمانی:
    - broker_timezone از project.broker_timezone خوانده و به عنوان timezone بروکر نگهداری می‌شود.
    - get_candles_range() ورودی‌های زمانی را با normalize_datetime در چارچوب timezone بروکر
        استاندارد می‌کند و منطق فعلی آن باید به عنوان قرارداد موجود حفظ شود.
    - تعیین timezone نهایی داده بر عهده پارامتر result_tz و مصرف‌کننده است.

این ماژول منبع مستقیم داده خام بازار برای mt5_data_loader_E.py و مسیر live بازار است.


نکاتی که خودم اضافه کرده ام:
    *** این متد، دیتافریم را با اندکس زمانی zone-aware و با broker_timezone برمیگرداند ***
    ***      این کار در انتهای متد _normalize_rates از کلاس MT5Connector انجام میشود     ***

Test Run: python -m f03_data.mt5_connector
"""
# =======================================================================================

# =======================================================================================
# Imports & Logger
# ======================================================================================= OK
from __future__ import annotations

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import time
import pandas as pd
import logging

from f02_utils.functions.normalize_datetime import normalize_datetime
from f02_utils.functions.constants import _TF_MAP

# ----------------- Logger for this module ---------------------------------------------- OK
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ----------------- Trying to import MT5 ------------------------------------------------ OK
try:
    import MetaTrader5 as mt5  # type: ignore
    _HAS_MT5 = True
except Exception as ex:        # pragma: no cover
    mt5 = None                 # type: ignore
    _HAS_MT5 = False
    logger.error("MetaTrader5 package not available: %s", ex)

# ----------------- Trying to import config_loader -------------------------------------- OK
try:
    from f02_utils.config_loader import load_config
except Exception:
    load_config = None  # type: ignore
    logger.warning("Could not import load_config from f02_utils.config_loader; please ensure it exists.")

# =======================================================================================
# ساختار تنظیمات Connector (اختیاری، برای خوانایی و توسعه‌پذیری) 
# ======================================================================================= OK
@dataclass
class MT5Credentials:
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    terminal_path: Optional[str] = None   # مسیر نصب ترمینال (متاتریدر) - اختیاری

@dataclass
class MT5ConnectorOptions:
    max_retries: int = 10                 #60  # تعداد تلاش‌ها برای initialize/login 
    retry_delay: float = 5.0              # فاصله بین تلاش‌ها (ثانیه) 
    backoff_multiplier: float = 1.0       # ضریب backoff (می‌توان 1.2 گذاشت) 
    auto_select_symbols: bool = True      # نمادهای کانفیگ را auto-select کند 
    check_trade_allowed: bool = False     # در health_check بررسی trade_allowed را هم انجام دهد 
    symbol_activation_timeout: float = 4  # مکث کوتاه پس از انتخاب نماد (ثانیه) 
    ensure_on_each_call: bool = True      # قبل از هر fetch یک ensure_connection بزن 

# =======================================================================================
# نگاشت تایم‌فریم‌ها 
# ======================================================================================= OK
def _to_mt5_timeframe(tf: str) -> int:
    """ نگاشت رشتهٔ تایم‌فریم به ثابت متناظر در MetaTrader5 
    اگر نامعتبر باشد ValueError می‌دهد
    """
    if not _HAS_MT5:
        raise RuntimeError("MetaTrader5 package not available")

    key = str(tf).replace(" ","").upper()  # key = str(tf).strip().upper()
    if key not in _TF_MAP:
        raise ValueError(f"Unsupported timeframe key: {tf}")

    # تبدیل کلید به ثابت mt5.TIMEFRAME_*
    name = _TF_MAP[key]
    const = getattr(mt5, f"TIMEFRAME_{name}")
    if const is None:
        raise ValueError(f"Unsupported timeframe (missing constant): {tf}")
    return const

# =======================================================================================
# Connector
# ======================================================================================= OK
class MT5Connector:
    """ Connector عمومی برای اتصال به MetaTrader5 
    - با کانفیگ `f01_config/config.yaml` کار می‌کند (از طریق load_config())
    - خواص مورد نیاز سایر بخش‌ها:
        * self.connected  (bool)
        * initialize(), ensure_connection(), shutdown()
        * get_candles_num(symbol, timeframe, num_candles, result_tz)
        * get_candles_range(symbol, timeframe, date_from, date_to, date_tz, result_tz)
    """   
    # ---------------------------------------------------------------
    # متد سازنده
    # --------------------------------------------------------------- OK
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 credentials: Optional[MT5Credentials] = None,
                 options: Optional[MT5ConnectorOptions] = None):
        """
        اگر config ندهید، از load_config() استفاده می‌شود.
        اگر credentials ندهید، از config['mt5_credentials'] خوانده می‌شود.
        اگر options ندهید، از config['initialize_retry'] خوانده می شود.
        """
        # -- 1 -- بررسی وجود کتابخانه متاتریدر
        if not _HAS_MT5:
            raise RuntimeError("MetaTrader5 package not installed or failed to import.")

        # -- 2 -- تعیین و تنظیم نمودن مقادیر config, credentials, options
        self.cfg: Dict[str, Any] = config or (load_config() if callable(load_config) else {})
        self.creds: MT5Credentials = credentials or self._read_credentials_from_config(self.cfg)
        self.opts: MT5ConnectorOptions = options or self._read_options_from_config(self.cfg)

        # -- 3 -- تعیین متغیرهای کنترلی برای وضعیتهای "اتصال" و "آخرین خطا" حادث شده
        self.connected: bool = False
        self._last_init_error: Optional[str] = None

        # -- 4 -- broker_timezone -------------------------
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
        
    # ---------------------------------------------------------------
    # helpers برای ساخت تنظیمات  
    # --------------------------------------------------------------- OK
    @staticmethod
    def _read_credentials_from_config(cfg: Dict[str, Any]) -> MT5Credentials:
        mt5c = (((cfg.get("connection") or {}).get("mt5_credentials")) or {}) if isinstance(cfg, dict) else {}
        login = mt5c.get("login")
        # تلاش برای تبدیل login به int (در صورت رشته بودن)
        try:
            login = int(login) if login is not None else None
        except Exception:
            pass
        return MT5Credentials(
            login = login,
            password = mt5c.get("password"),
            server = mt5c.get("server"),
            terminal_path = mt5c.get("terminal_path") or mt5c.get("path") or None,
        )


    @staticmethod
    def _read_options_from_config(cfg: Dict[str, Any]) -> MT5ConnectorOptions:
        init_opts = (((cfg.get("connection") or {}).get("initialize_retry")) or {}) if isinstance(cfg, dict) else {}
        return MT5ConnectorOptions(
            max_retries = int(init_opts.get("max_retries", 10)),
            retry_delay = float(init_opts.get("retry_delay", 5)),
            backoff_multiplier = float(init_opts.get("backoff_multiplier", 1.0)),
            auto_select_symbols = bool(init_opts.get("auto_select_symbols", True)),
            check_trade_allowed = bool(init_opts.get("check_trade_allowed", False)),
            symbol_activation_timeout = float(init_opts.get("symbol_activation_timeout", 4.0)),
            ensure_on_each_call = bool(init_opts.get("ensure_on_each_call", True)),
        )

    # ---------------------------------------------------------------
    # اتصال/لاگین 
    # --------------------------------------------------------------- OK
    def initialize(self,
                   max_retries: Optional[int] = None,
                   retry_delay: Optional[float] = None,
                   backoff_multiplier: Optional[float] = None) -> bool:
        """اتصال به ترمینال MetaTrader5 و ورود به حساب کاربری. 
        پارامترها اگر None باشند از self.opts خوانده می‌شوند.
        """
        # -- 1 -- بررسی وجود کتابخانه متاتریدر
        if not _HAS_MT5:
            raise RuntimeError("MetaTrader5 package not available")

        # -- 2 -- تعیین پارامترهای اتصال
        max_retries        = int  (max_retries        or self.opts.max_retries)
        retry_delay        = float(retry_delay        or self.opts.retry_delay)
        backoff_multiplier = float(backoff_multiplier or self.opts.backoff_multiplier or 1.0)

        self._last_init_error = None

        # -- 3 -- تنظیم مسیر نصب متاتریدر
        init_kwargs: Dict[str, Any] = {}
        if self.creds.terminal_path:
            init_kwargs["path"] = str(self.creds.terminal_path)

        # --- try loops ----------------------------------- start
        delay = retry_delay
        for attempt in range(1, max_retries + 1):
            try:
                # -- L1 -- اجرای نرم افزار متاتریدر 5 و اتصال پایتون به آن. (ولی وارد هیچ اکانتی نمیشود)
                ok_init = bool(mt5.initialize(**init_kwargs)) if init_kwargs else bool(mt5.initialize())
                if not ok_init:
                    self._last_init_error = "initialize_failed"
                    logger.error("MT5 initialize failed. (attempt %d/%d)", attempt, max_retries)
                    time.sleep(delay)
                    delay *= backoff_multiplier
                    continue  #اگر اتصال برقرار نشد، شمارنده بعدی حلقه اجرا می‌شود 

                # -- L2 -- بررسی وجود مشخصات اکانت جهت ورود به اکانت
                if self.creds.login is None or not self.creds.password or not self.creds.server:
                    self._last_init_error = "missing_credentials"
                    logger.critical("MT5 credentials incomplete: login/password/server required.")
                    mt5.shutdown()
                    return False

                # -- L3 -- ورود به اکانت با داشتن مشخصات آن اکانت
                authorized = bool(mt5.login(login=int(self.creds.login),
                                            password=str(self.creds.password),
                                            server=str(self.creds.server)))
                # -- L4 -- بررسی ورود موفق به اکانت
                if not authorized:
                    self._last_init_error = "login_failed"
                    logger.error("MT5 login failed. (attempt %d/%d)", attempt, max_retries)
                    mt5.shutdown()
                    time.sleep(delay)
                    delay *= backoff_multiplier
                    continue  #اگر لاگین انجام نشد، شمارنده بعدی حلقه اجرا می‌شود 

                # -- L5 -- گرفتن "اطلاعات حساب" برای اطمینان از اینکه لاگین واقعاً موفق بوده است
                acct = mt5.account_info()  # sanity check: account_info
                
                if acct is None:
                    self._last_init_error = mt5.last_error()  # "no_account_info"
                    logger.error("MT5 login returned True but account_info() is None. (attempt %d/%d)", attempt, max_retries)
                    mt5.shutdown()
                    time.sleep(delay)
                    delay *= backoff_multiplier
                    continue  #اگر اطلاعات اکانت وجود ندارد، شمارنده بعدی حلقه اجرا می‌شود 

                # -- L6 -- اتصال و لاگین موفق بوده است
                #در صورتی به این نقطه میرسیم که اتصال و لاگین موفق بوده باشد و account_info موجود باشد 
                self.connected = True
                logger.info("MT5 connected: login=%s server=%s name=%s",
                            getattr(acct, "login", None), getattr(acct, "server", None), getattr(acct, "name", None))
                
                # -- L7 -- فعال نمودن نمادها در مارکت واچ (اختیاری)
                if self.opts.auto_select_symbols:
                    self._auto_select_symbols()   # نمادها را در market watch فعال می‌کند 

                # -- L8 -- موفقیت‌آمیز بودن اتصال و لاگین
                return True

            except Exception as ex:
                self._last_init_error = f"exception:{ex}"
                logger.exception("Exception during MT5 initialize/login (attempt %d/%d): %s", attempt, max_retries, ex)
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                time.sleep(delay)
                delay *= backoff_multiplier
        # --- try loops ----------------------------------- end
        
        # -- 4 -- اگر به اینجا رسیدیم، همه تلاش‌ها ناموفق بوده‌اند
        logger.critical("All attempts to connect/login to MT5 failed. last_error=%s", self._last_init_error)
        self.connected = False
        
        return False
    
    # ---------------------------------------------------------------
    # اطمینان از اتصال 
    # --------------------------------------------------------------- OK
    def ensure_connection(self) -> bool:
        """
        بررسی زنده‌بودن اتصال با mt5.account_info() و در صورت نیاز تلاش به reconnect
        """
        # -- 1 -- بررسی وجود کتابخانه متاتریدر
        if not _HAS_MT5:
            return False
        
        # -- 2 -- بررسی اولیه اتصال
        #         گرفتن "اطلاعات حساب" برای اطمینان از اینکه لاگین واقعاً موفق بوده است
        try:
            acct = mt5.account_info()
            if acct is not None:
                self.connected = True
                return True
        except Exception:
            pass

        # -- 3 -- در صورتی به این نقظه میرسیم که اتصال برقرار نباشد. 
        #         بنابراین باید اتصال مجدد برقرار بشود
        logger.warning("MT5 not connected — attempting re-initialize()")
        try:
            connected = self.initialize()
            self.connected = bool(connected)
            if connected:
                logger.info("MT5 reconnected successfully.")
            else:
                logger.error("MT5: unable to re-establish connection.")
            return connected
        except Exception as ex:
            logger.exception("re-initialize() failed with exception during ensure_connection: %s", ex)
            self.connected = False
            return False

    # ---------------------------------------------------------------
    # خاموش‌کردن اتصال 
    # --------------------------------------------------------------- OK
    def shutdown(self) -> None:
        """قطع اتصال از MT5 و بروزرسانی وضعیت داخلی."""
        # -- 1 -- بررسی وجود کتابخانه متاتریدر
        if not _HAS_MT5:
            return
        
        # -- 2 -- تلاش برای قطع اتصال متاتریدر
        try:
            mt5.shutdown()
        except Exception as ex:
            logger.exception("Error while shutting down MT5: %s", ex)
        finally:
            self.connected = False
            logger.info("MT5 shutdown complete.")

    # ---------------------------------------------------------------
    # انتخاب/اشتراک نمادها 
    # --------------------------------------------------------------- OK
    def _auto_select_symbols(self) -> None:
        """
        نمادهایی که در کانفیگ آمده‌اند را در ترمینال انتخاب (subscribe) می‌کند تا copy_rates_* کار کند.
        ابتدا از download_defaults.symbols می خواند و اگر نبود از مسیرهای قدیمی‌تر.

        در واقع نمادها را مطابق با کانفیگ در پنجره Market Watch فعال می‌کند.
        """
        # -- 1 -- خواندن لیست سمبل ها  از کانفیگ
        symbols: List[str] = []
        try:
            dd = ((self.cfg.get("download_defaults")) or {})
            symbols = list(dd.get("symbols") or [])
        except Exception:
            symbols = []
        
        # -- 2 -- اگر لیست سمبلها خالی است، متد را خاتمه بده
        if not symbols:
            logger.info("No symbols to auto-select from config.")
            return

        # -- 3 -- سمبل های موجود در لیست را در مارکت واچ فعال کن
        for sym in symbols:
            try:
                ok = mt5.symbol_select(sym, True)    # سمبل را در پنجره Market Watch فعال می‌کند 
                if ok:
                    logger.debug("Symbol selected: %s", sym)
                    time.sleep(self.opts.symbol_activation_timeout)  # مکث کوتاه پس از انتخاب نماد (ثانیه) 
                else:
                    logger.warning("Failed to select symbol: %s", sym)
            except Exception:
                logger.exception("Exception while selecting symbol: %s", sym)

    # ---------------------------------------------------------------
    # دریافت کندل‌ها (آخرین/رنج) 
    # --------------------------------------------------------------- OK 3 func.s
    def get_candles_num(
        self,
        symbol: str,
        timeframe: str,
        num_candles: int = 1000,
        result_tz: tzinfo | str | None = None,
    ) -> "pd.DataFrame":
        """
        دریافت آخرین کندل‌ها برای یک نماد و تایم‌فریم مشخص.

        پارامترها:
        - symbol: نام نماد (مثلاً "EURUSD")
        - timeframe: تایم‌فریم به صورت string (مثلاً "M1", "H1")
        - num_candles: تعداد آخرین کندل‌هایی که باید دریافت شوند (پیش‌فرض 1000)
        - result_tz: منطقه زمانی برای دیتافریمی که حاصل این تابع است
                     هرگاه این پارامتر نان باشد، اندکس دیتافریم خروجی naive خواهد بود.
        خروجی:
        - DataFrame با index از نوع datetime  (aware or naive)
        - ستون‌ها:
            - open, high, low, close
            - volume: ستون حجم، که می‌تواند از real_volume یا tick_volume استخراج شود
            - spread
        - اگر داده‌ای دریافت نشود، DataFrame خالی برمی‌گردد

        توضیحات:
        - ستون volume همیشه با نام یکنواخت 'volume' بازگردانده می‌شود
        - داده‌ها بر اساس زمان مرتب شده‌اند
        - اگر opts.ensure_on_each_call فعال باشد، قبل از دریافت داده، اتصال به MT5 بررسی می‌شود
        """
        # -- 1 -- بررسی وضعیت اتصال به متاتریدر
        if self.opts.ensure_on_each_call:  # اگر true باشد، یعنی باید قبل از هر fetch یکبار ensure_connection را اجرا کنیم 
            if not self.ensure_connection():
                raise ConnectionError("Cannot connect to MT5")

        # -- 2 -- تبدیل رشته تایمفریم به نوع مورد قبول متاتریدر و سپس دانلود داده ها از متاتریدر
        tf = _to_mt5_timeframe(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(num_candles))

        logger.debug("=== After mt5.copy_rates_num ===============================")
        logger.debug(f"Downloaded {len(rates)} candles for {symbol} {timeframe}")

        # -- 3 -- نرمال سازی ستونهای داده های دریافتی از متاتریدر
        df = self._normalize_rates(rates, symbol, timeframe)

        logger.debug("=== After _normalize_rates() ===========")
        logger.debug(f"type of df: {type(df)}")
        logger.debug(f"columns of df: {df.columns}")
        logger.debug(f"rows of df: {len(df)}")

        # -- 4 -- نرمال سازی ستون زمان
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=False)   #.dt.tz_localize(self.broker_timezone)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        if result_tz is not None:
            df = df.tz_localize(self.broker_timezone)
            df = df.tz_convert(result_tz)

        logger.debug("=== After normalize time column ========")
        logger.debug(f"first row = {df.index[0]}")
        logger.debug(f"last row = {df.index[-1]}")
        logger.debug("============================================================\n")

        return df

    # -------------------------------------------
    def get_candles_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: datetime,
        date_to: datetime,
        date_tz: tzinfo | None = None,
        result_tz: tzinfo | str | None = None,
    ) -> "pd.DataFrame":
        # این تابع اصلاً تغییر نکند و بخصوص دو سطری که دارای کامنت ### *** ### هستند

        """
        دریافت کندل‌ها برای یک نماد و تایم‌فریم مشخص بین دو تاریخ مشخص [inclusive].

        پارامترها:
        - symbol: نام نماد (مثلاً "EURUSD")
        - timeframe: تایم‌فریم به صورت string (مثلاً "M1", "H1")
        - date_from: تاریخ شروع
        - date_to: تاریخ پایان
        - date_tz: منطقه زمانی مربوط به دو پارامتر بالایی
        - result_tz: منطقه زمانی برای دیتافریمی که حاصل این تابع است
                    هرگاه این پارامتر نان باشد، اندکس دیتافریم خروجی naive خواهد بود.
        خروجی:
        - DataFrame با index از نوع datetime (aware or naive)
        - ستون‌ها:
            - open, high, low, close
            - volume: ستون حجم، که می‌تواند از real_volume یا tick_volume استخراج شود
            - spread
        - اگر داده‌ای دریافت نشود، DataFrame خالی برمی‌گردد

        توضیحات:
        - ستون volume همیشه با نام یکنواخت 'volume' بازگردانده می‌شود
        - داده‌ها بر اساس زمان مرتب شده‌اند
        - اگر opts.ensure_on_each_call فعال باشد، قبل از دریافت داده، اتصال به MT5 بررسی می‌شود
        """

        # -- 1 -- بررسی اتصال MT5
        if self.opts.ensure_on_each_call:
            if not self.ensure_connection():
                raise ConnectionError("Cannot connect to MT5")

        # -- 2 -- نرمال سازی زمان‌ها به timezone بروکر
        try:
            dt_from = normalize_datetime(date_from, input_tz=date_tz, output_tz=self.broker_timezone)
            dt_to = normalize_datetime(date_to, input_tz=date_tz, output_tz=self.broker_timezone)
        except Exception as e:
            raise ValueError(
                f"Invalid datetime range: "
                f"from={date_from}, to={date_to}"
            ) from e

        # -- 3 -- تبدیل رشته تایمفریم به نوع مورد قبول متاتریدر و سپس دانلود داده ها از متاتریدر
        tf = _to_mt5_timeframe(timeframe)

        dt_from = dt_from.replace(tzinfo=timezone.utc) ### *** ###
        dt_to = dt_to.replace(tzinfo=timezone.utc)     ### *** ###

        rates = mt5.copy_rates_range(symbol, tf, dt_from, dt_to)

        logger.debug("=== After mt5.copy_rates_range =============================")
        logger.debug(f"Downloaded {len(rates)} candles for {symbol} {timeframe}")

        # -- 4 -- نرمال سازی ستونهای داده های دریافتی از متاتریدر
        df = self._normalize_rates(rates, symbol, timeframe)

        logger.debug("=== After _normalize_rates() ===========")
        logger.debug(f"type of df: {type(df)}")
        logger.debug(f"columns of df: {df.columns}")
        logger.debug(f"rows of df: {len(df)}")

        # -- 5 -- نرمال سازی ستون زمان
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=False)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        
        if result_tz is not None:
            df = df.tz_localize(self.broker_timezone)
            df = df.tz_convert(result_tz)

        logger.debug("=== After normalize time column ========")
        logger.debug(f"first row = {df.index[0]}")
        logger.debug(f"last row = {df.index[-1]}")
        logger.debug("============================================================\n")

        return df

    # -------------------------------------------
    def _normalize_rates(self, rates, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        خروجی:
            دیتافریم با اندکس زمانی در self.broker_timezone
        """
        
        # -- 2 -- کنترل دیتای ورودی به این تابع
        if rates is None or len(rates) == 0:
            logger.warning("No rates for %s %s", symbol, timeframe)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "spread"])

        df = pd.DataFrame(rates)

        # -- 3 -- تعیین ستون حجم
        volume_col = next(
            (c for c in ("tick_volume", "real_volume")
            if c in df.columns and df[c].notna().any() and (df[c] != 0).any()),
            None
        )
        # -- 4 -- ساختاردهی ستونها
        cols = ["time", "open", "high", "low", "close"]
        if volume_col is not None:
            cols.append(volume_col)
        cols.append("spread")
        df = df[cols]
        
        # -- 5 -- تغییر نام ستون حجم
        if volume_col is not None and volume_col != "volume":
            df.rename(columns={volume_col: "volume"}, inplace=True)

        """-- 6 -- تنظیم ستون زمان به عنوان اندکس و سورت نمودن دیتافریم نهایی
        به دلیل تفاوتی که بین نرمال سازی ستونهای زمان در دو حالت دانلود از متاتریدر وجود داشت،
        این بخش به داخل توابع copy_rates_... منتقل شد.

        df["time"] = pd.to_datetime(df["time"], unit="s", utc=False).dt.tz_localize(self.broker_timezone)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        """

        return df
    
    # ---------------------------------------------------------------
    # Health & Diagnostics
    # --------------------------------------------------------------- OK
    def health_check(self, sample_symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        گزارش سریع سلامت اتصال و حساب.
        sample_symbol: اگر داده شود، یک symbol_info_tick می‌گیرد تا تاخیر/دسترسی سنجیده شود.
        """
        # -- 1 -- ساخت دیکشنری خالی
        info: Dict[str, Any] = {
            "connected": False,
            "login": None,
            "server": None,
            "name": None,
            "currency": None,
            "trade_allowed": None,
            "sample_symbol_ok": None,
            "last_init_error": self._last_init_error,
        }

        # -- 2 -- بررسی وجود کتابخانه متاتریدر
        if not _HAS_MT5:
            info["connected"] = False
            return info

        # -- 3 -- گرفتن "اطلاعات حساب" مربوط به اکانت
        try:
            acct = mt5.account_info()
            info["connected"] = acct is not None
            if acct is not None:
                info["login"   ] = getattr(acct, "login"   , None)
                info["server"  ] = getattr(acct, "server"  , None)
                info["name"    ] = getattr(acct, "name"    , None)
                info["currency"] = getattr(acct, "currency", None)
                if self.opts.check_trade_allowed:
                    info["trade_allowed"] = bool(getattr(acct, "trade_allowed", False))
        except Exception as ex:
            info["connected"] = False
            info["last_init_error"] = f"account_info_exception:{ex}"
            return info
        
        # -- 4 -- تست دسترسی و تاخیر
        """
        بلوک زیر یک تست سریعِ دسترسی و تاخیر برای sample_symbol انجام می‌دهد
        و نتیجهٔ موفقیت/عدم موفقیت و مقدار لاتنسی را در info می‌گذارد
        تا health_check بتواند گزارش دهد.
        """
        if sample_symbol:
            try:
                t0 = time.perf_counter()
                tick = mt5.symbol_info_tick(sample_symbol)  # گرفتن آخرین تیک برای نماد نمونه از متاتریدر 
                dt = time.perf_counter() - t0
                info["sample_symbol_ok"] = tick is not None  # مقداری بولی دارد که نشان می‌دهد آیا تیک با موفقیت گرفته شده یا نه
                info["latency_ms"] = round(dt * 1000.0, 2)
            except Exception as ex:
                info["sample_symbol_ok"] = False
                info["last_init_error"] = f"tick_exception:{ex}"

        return info


    # ---------------------------------------------------------------
    # Context Manager
    # --------------------------------------------------------------- OK
    # برای استفاده از این کلاس در دستوراتی مانند with MT5Connector() as my_conn:
    def __enter__(self) -> "MT5Connector":
        ok = self.initialize()
        if not ok:
            raise RuntimeError(f"MT5Connector failed to initialize: {self._last_init_error}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    # ---------------------------------------------------------------
    # Symbol specs snapshot (for overlay)  — aligns with project style
    # --------------------------------------------------------------- OK
    def get_symbol_specs(self, symbols: list[str]) -> dict:
        """دریافت مشخصات نمادها از MT5 و بازگرداندن ساختاری مناسب برای ذخیره در overlay 
        خروجی:
            {
                "meta": {"as_of": "...Z", "account_currency": "...", "login": 12345, "server": "..."},
                "symbol_specs": {
                    ...,
                    "XAUUSD": {
                        "digits": 2,
                        "point": 0.01,
                        "trade_tick_value": 1.0,
                        "trade_tick_size": 0.01,
                        "contract_size": 100.0,
                        "volume_min": 0.01,
                        "volume_step": 0.01,
                        "volume_max": 100.0,
                        "stops_level": 50,
                        "pip_value_per_lot": 0.01,    
                        "raw": { ... }               
                    },
                    ...,
                }
            }
        خروجی نهایی:
            یک دیکشنری با دو کلید اصلی برمیگرداند:
            "meta": شامل اطلاعات حساب و زمان اسنپ‌شات.
            "symbol_specs": دیکشنری است که کلیدهایش نام نمادها و مقادیرش همان مشخصات فنی استخراج شده است.
        """
        
        # -- 1 -- بررسی اتصال به متاتریدر
        try:
            #  بررسی زنده‌بودن اتصال با mt5.account_info() و در صورت نیاز تلاش به reconnect 
            self.ensure_connection()
        except Exception as ex:  # pragma: no cover
            logger.error("Could not ensure MT5 connection: %s", ex)
            return {"meta": {"connected": False}, "symbol_specs": {}}

        # -- 2 -- تلاش برای گرفتن "اطلاعات حساب".
        #         اما ممکن است که None باشد 
        acct = None
        try:
            acct = mt5.account_info() if _HAS_MT5 else None
        except Exception:
            acct = None

        # -- 3 -- ساخت دیکشنری مشخصات عمومی اکانت به نام meta
        meta = {
            # as_of = زمان معتبر بودن داده در لحظهٔ گرفته شدن snapshot است 
            "as_of": datetime.now(timezone.utc).replace(microsecond=0).isoformat() + "Z",
            "connected": bool(acct is not None),
            "account_currency": getattr(acct, "currency", None) if acct else None,
            "login": getattr(acct, "login", None) if acct else None,
            "server": getattr(acct, "server", None) if acct else None,
        }

        # -- 4 -- اگر نمادی وجود ندارد، اسپک ها را تهی بگذار و خروجی بده
        specs: dict[str, dict] = {}
        if not symbols:
            logger.warning("No symbols provided to get_symbol_specs().")
            return {"meta": meta, "symbol_specs": specs}
        
        # -- 5 -- حلقه روی تمام نمادها جهت بدست آوردن مشخصات هر نماد
        #         و تولید دیکشنری specs
        for sym in symbols:
            # -- L1 --------- گرفتن اطلاعات سمبل --------------------
            si = None
            try:
                si = mt5.symbol_info(sym) if _HAS_MT5 else None
                if si is None and _HAS_MT5:
                    mt5.symbol_select(sym, True)   # سمبل را در پنجره Market Watch فعال می‌کند 
                    si = mt5.symbol_info(sym)      # تلاش مجدد برای گرفتن info 
            except Exception as ex:
                logger.exception("Error fetching symbol_info for %s: %s", sym, ex)
                si = None

            if si is None:
                logger.warning("Symbol not available or info missing: %s", sym)
                continue
            # -- L2 --------- ساخت آیتم مشخصات ----------------------
            try:
                item = {
                    "digits"          : getattr(si, "digits"             , None),
                    "point"           : getattr(si, "point"              , None),
                    "trade_tick_value": getattr(si, "trade_tick_value"   , None),
                    "trade_tick_size" : getattr(si, "trade_tick_size"    , None),
                    "contract_size"   : getattr(si, "trade_contract_size", None) or getattr(si, "contract_size", None),
                    "volume_min"      : getattr(si, "volume_min"         , None),
                    "volume_step"     : getattr(si, "volume_step"        , None),
                    "volume_max"      : getattr(si, "volume_max"         , None),
                    "stops_level"     : getattr(si, "stops_level"        , None),
                }

                # -- L2-2 -- ذخیرهٔ نسخهٔ خام بازگشتی از متاتریدر برای دیباگ/آرشیو ---
                """
                بطور کلی:
                    خیلی از اشیاء کتابخانه های پایتون مانند namedtuple ها یا برخی از Dataclass ها
                    دارای متد استانداردی به نام _asdict() هستند
                    که دقیقاً کار تبدیل شیء به دیکشنری را انجام میدهد.
                    اگر si این متد را داشت، آن را صدا میزند و نتیجه را به dict تبدیل میکند
                    برخی متدها ممکن است OrderedDict برگردانند.
                    این تمیزترین و مطمئنترین راه است.
                در اینجا:
                    اگر object دارای _asdict باشد (مثلاً namedtuple) از آن استفاده می‌کنیم،
                    در غیر اینصورت تلاش می‌کنیم فیلدهای عمومی را استخراج کنیم.
                """
                raw_spec = None
                try:
                    """ مرحلهٔ 1:
                    بررسی وجود متد _asdict 
                    اگر object دارای _asdict باشد (مثلاً namedtuple) از آن استفاده می‌کنیم،
                    """
                    if hasattr(si, "_asdict"):
                        raw_spec = dict(si._asdict())
                    else:
                        """ مرحله 2
                        با dir(si) لیست تمام نامهای ویژگیها و متدهای آن شیء را میگیرد.
                        فیلتر اول: not k.startswith("_")
                            ویژگیهای خصوصی (که با زیرخط شروع میشوند) را حذف میکند تا خروجی تمیز باشد.
                        فیلتر دوم: not callable(getattr(si, k, None))
                            بررسی میکند که آیا آن ویژگی قابل فراخوانی است (متد است) یا خیر.
                            اگر متد باشد، آن را حذف میکند تا فقط دادهها (Properties/Attributes) باقی بمانند و توابع اضافی وارد دیکشنری نشوند.
                        در نهایت با getattr مقدار هر کلید را میگیرد و دیکشنری نهایی را میسازد.
                        """
                        raw_spec = {k: getattr(si, k, None) for k in dir(si) if not k.startswith("_") and not callable(getattr(si, k, None))}
                except Exception:
                    """ مرحله 3
                    کل این عملیات درون یک try/except قرار گرفته است.
                    اگر به هر دلیلی (مثلاً شیء ساختار عجیبی داشت یا دسترسی به یکی از ویژگیها خطا داد)، برنامه کرش نمیکند،
                    بلکه به سادگی raw_spec=None قرار داده و متد به کار خود ادامه میدهد.
                    این یعنی گرفتن اطلاعات raw یک امتیاز اضافی (Nice-to-have) است، نه یک الزام حیاتی.
                    """
                    raw_spec = None
                
                if raw_spec is not None:
                    item["raw"] = raw_spec

                # -- L2-3 -- محاسبهٔ ایمن pip_value_per_lot ---------
                try:
                    p = float(item.get("point") or 0)
                    tv = float(item.get("trade_tick_value") or 0)
                    ts = float(item.get("trade_tick_size") or 1)
                    # بررسی اینکه مقادیر معنادار و غیر صفر باشند
                    if p > 0 and tv > 0 and ts > 0:
                        item["pip_value_per_lot"] = round((tv / ts) * (10.0 * p), 6)   # pip := 10*point
                except Exception as ex:
                    # اگر خطایی در تبدیل/محاسبه بود، لاگ کن ولی اجرای تابع را ادامه بده
                    logger.debug("Could not compute pip_value_per_lot for %s: %s", sym, ex)
                
            except Exception as ex:
                # -- L3 -- اگر مشکلی در ساخت آیتم بود، به سمبل بعدی می‌رود
                logger.exception("Failed to build specs for %s: %s", sym, ex)
                continue # اگر مشکلی در ساخت آیتم بود، به سمبل بعدی می‌رود 
            
            # -- L4 -- ذخیره آیتم در دیکشنری اسپک ها
            specs[sym] = item

        # -- 6 -- ساخت دیکشنری نهایی شامل دو کلید اصلی meta, symbol_specs
        return {"meta": meta, "symbol_specs": specs}


# =======================================================================================
# نمونهٔ اجرا (اختیاری) 
# =======================================================================================
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-6s | %(filename)-18s | %(lineno)-4d : %(funcName)-24s | %(message)s",
    )
    conn = MT5Connector()
    
    if conn.initialize():
        hc = conn.health_check(sample_symbol="BITCOIN")
        logger.info("Health: %s", hc)

        df_by_num = conn.get_candles_num("BITCOIN", "M1", 5, "UTC")
        df_by_range = conn.get_candles_range("BITCOIN", "M1", "2026-08-23 09:25:00", "noW", conn.broker_timezone, "UTC")

        conn.shutdown()

        print(f"df_by_num = {df_by_num}")
        print(f"df_by_range = {df_by_range}")
    


# =======================================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# =======================================================================================
""" Func Names                                                Used in Functions: ...
                                   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
1  MT5Credentials                 --  --  --  --  ok  ok  --  --  --  --  --  --  --  --  --  --  --  --
2  MT5ConnectorOptions            --  --  --  --  ok  --  ok  --  --  --  --  --  --  --  --  --  --  --
3  _to_mt5_timeframe              --  --  --  --  --  --  --  --  --  --  --  ok  ok  --  --  --  --  --
4  MT5Connector                   --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok
5  __init__                       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
6  _read_credentials_from_config  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --
7  _read_options_from_config      --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --
8  initialize                     --  --  --  --  --  --  --  --  ok  --  --  --  --  ok  --  --  --  ok 
9  ensure_connection              --  --  --  --  --  --  --  --  --  --  --  ok  ok  --  --  --  ok  -- 
10 shutdown                       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  ok 
11 _auto_select_symbols           --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --
12 get_candles_nmu                --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok
13 get_candles_range  (NOT USED)  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
14 health_check                   --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok
15 __enter__          (NOT USED)  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
16 __exit__           (NOT USED)  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
17 get_symbol_specs   (NOT USED)  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
18 (Global code)                  -/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
"""

# =======================================================================================
# نمونه‌های استفاده از توابع MT5 (برای توسعه‌دهندگان) 
# =======================================================================================
"""
mt5.initialize(path, login=LOGIN, password="PASSWORD", server="SERVER", timeout=TIMEOUT, portable=False)
mt5.shutdown()
mt5.login(login, password="PASSWORD", server="SERVER", timeout=TIMEOUT)
mt5.account_info()
mt5.symbol_select(symbol, enable=None)
mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
mt5.symbol_info_tick(symbol)
mt5.symbol_info(symbol)
"""

# =======================================================================================
# نمونهٔ خروجی mt5.account_info() (برای توسعه‌دهندگان) 
# =======================================================================================
""" 
mt5.account_info() sample output:
    {
    'login': 52623142,
    'trade_mode': 0,
    'leverage': 100,
    'limit_orders': 10000,
    'margin_so_mode': 0,
    'trade_allowed': True,
    'trade_expert': True,
    'margin_mode': 2,
    'currency_digits': 2,
    'fifo_close': False,
    'balance': 1200.0,
    'credit': 0.0,
    'profit': 0.0,
    'equity': 1200.0,
    'margin': 0.0,
    'margin_free': 1200.0,
    'margin_level': 0.0,
    'margin_so_call': 80.0,
    'margin_so_so': 50.0,
    'margin_initial': 0.0,
    'margin_maintenance': 0.0,
    'assets': 0.0,
    'liabilities': 0.0,
    'commission_blocked': 0.0,
    'name': 'Farhad Vahedkia',
    'server': 'Alpari-MT5-Demo',
    'currency': 'USD',
    'company': 'Alpari'
    }
"""