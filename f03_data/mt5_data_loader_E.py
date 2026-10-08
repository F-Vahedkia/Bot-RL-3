# f03_data/mt5_data_loader_E.py (2)
#
# Last reviewed: 1405/06/25
# =======================================================================================
""" ---> Docstring:
لودر batch داده‌های بازار از MetaTrader 5 برای لایه f03_data.

این ماژول لایه دریافت داده از MT5Connector را به یک فرایند قابل‌تکرار برای
ساخت DownloadPlan، دریافت کندل‌ها، نرمال‌سازی، ادغام با فایل موجود و ذخیره داده
خام تبدیل می‌کند.

مسئولیت‌های اصلی:
    - تعریف DownloadPlan برای یک symbol/timeframe و مشخص‌کردن روش دریافت بر اساس
        lookback_bars یا بازه date_from/date_to.
    - خواندن و تکمیل تنظیمات download_defaults و project.broker_timezone از config.
    - اجرای سیاست‌های range_policy شامل count، date، min و max در زمان ساخت/اجرای plan.
    - دریافت کندل‌ها از MT5Connector و انتخاب مسیر count/date متناسب با plan.
    - نرمال‌سازی DataFrame بدون تحمیل UTC در normalize_df؛ timezone خروجی در مرحله دریافت
        می‌تواند توسط connector تعیین شود.
    - الحاق داده جدید به فایل موجود، حذف timestampهای تکراری با نگه‌داشتن آخرین رکورد،
        مرتب‌سازی و ذخیره در CSV یا Parquet.
    - نوشتن metadata JSON در کنار فایل داده و بازگرداندن گزارش اجرای هر plan.
    - فراهم‌کردن CLI برای اجرای batch download از config یا overrideهای command line.

قرارداد خروجی:
    - داده خام در raw_dir و در مسیر نماد/تایم‌فریم ذخیره می‌شود.
    - format ذخیره‌سازی می‌تواند csv یا parquet باشد؛ در شکست ذخیره Parquet، fallback به CSV
        در کد فعلی وجود دارد.
    - DataFrameهای بازگردانده‌شده در results علاوه بر اطلاعات گزارش، کل DataFrame را در کلید
        dataframe حمل می‌کنند.

این ماژول تولیدکننده داده batch خام است و DataHandler از فایل‌های تولیدشده توسط آن
برای ساخت MTFDataset استفاده می‌کند.


Run: روش های اجرا از خظ فرمان یا CIL (Command Line Interface)

نمونه اجرا (از ریشه‌ی ریپو):
python -m f03_data.mt5_data_loader_E `
    --config f01_config/config.yaml  `
    --symbols XAUUSD                 `
    --timeframes M1 M2 M4 M20 H1 H4  `
    --lookback 10000000              `
    --save_format parquet            `
    --log-level DEBUG

python -m f03_data.mt5_data_loader_E `
    -c ./f01_config/config.yaml      `
    --symbols XAUUSD EURUSD BITCOIN  `
    --timeframes M1 M2 M5 M10 H1 H4 D1  `
    --lookback 1000000               `
    --range_policy count             `
    --save_format parquet

python -m f03_data.mt5_data_loader_E `
    --symbols XAUUSD                 `
    --timeframes M5                  `
    --brk_date_from "2026-09-16 00:00:00+03:00" `
    --brk_date_to "2026-09-17 00:00:00+03:00"   `
    --range_policy date              `
    --save_format csv

python -m f03_data.mt5_data_loader_E
    در این حالت، اگر آرگومان‌ها را ندهید، از مقادیر بخش download_defaults در کانفیگ استفاده میشود.
"""

# f03_data/mt5_data_loader_E.py (2)
# =======================================================================================

# =======================================================================================
# Imports & Logger
# =======================================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import json
import logging
import argparse

# ------------------ Importing Internal Modules -----------------------------------------
from f03_data.mt5_connector import MT5Connector
from f02_utils.functions.normalize_datetime import normalize_datetime
from f02_utils.config_completer import config_completer
from f02_utils.config_path_funcs import project_root, resolve_raw_dir, full_file_path
from f02_utils.functions._to_zoneinfo import _to_zoneinfo

# -------------------- Logger for this module -------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =======================================================================================
# ساختار برنامه و کمکی‌ها 
# =======================================================================================
# -------------------------------------------------------------------
@dataclass
class DownloadPlan:
    """طرح دانلود برای یک جفت ارز (نماد/تایم‌فریم)."""
    symbol: str
    timeframe: str
    # ----- یکی از lookback_bars یا (date_from, date_to) باید مشخص باشد
    lookback_bars: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime|str] = None
    date_tz: Optional[tzinfo|None] = None
    result_tz: tzinfo|str|None = None
    # ----- 
    range_policy: Literal["min", "max", "date", "count"] = "count"

# -------------------------------------------------------------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """ نرمال‌سازی دیتافریم دریافتی از متاتریدر
    - تنظیم ایندکس زمانی در UTC      <<<=== این تنظیم کامنت شد. این تنظیم کار را خراب کرده بود.
    - انتخاب ستون‌های استاندارد
    - تبدیل نوع‌ها و مرتب‌سازی
    - حذف رکوردهای تکراری بر اساس ایندکس
    """
    
    # -- 1 -- بررسی وجود و خالی نبودن دیتافریم ورودی ---------------------------------
    if df is None or df.empty:
        logger.debug('df is None or empty')
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "spread"],
                            index=pd.DatetimeIndex([], name="time")
                            )
    cols = list(df.columns)
    # logger.debug(f'columns of df at end of part 1 is : {cols}')   # save for debug

    # -- 2 -- معرفی ستون time به عنوان اندکس زمانی دیتافریم --------------------------
    if "time" in cols:
        df["time"] = pd.to_datetime(df["time"]) #, utc=True)
        df.set_index("time", inplace=True)
    # logger.debug(f'columns of df at end of part 2 is : {cols}')   # save for debug

    # -- 3 -- مرتب‌سازی بر اساس زمان ---------------------------------------------------
    df.sort_index(inplace=True)

    # -- 4 -- ساخت لیست نام ستونهای موجود و مورد نیاز به عنوان فیلتر ----------------
    keep = [c for c in ["open", "high", "low", "close", "volume", "spread"] if c in cols]

    # -- 5 -- انتخاب ستونهای مشخصی از دیتافریم، طبق لیست قسمت قبل  ------------------
    if keep:
        df = df[keep].copy()
    
    # -- 6 -- معقول سازی نوع دادع ها ----------------------------------------------------
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # float64 با NaN
    
    for col in ["volume", "spread"]:
        if col in df.columns:
            # Int64 از NaN پشتیبانی میکند (برای RL حیاتی است)
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    
    # -- 7 -- معرفی نام ایندکس ----------------------------------------------------------
    df.index.name = "time"

    # -- 8 -- حذف رکوردهای با ایندکس تکراری ---------------------------------------------
    df = df[~df.index.duplicated(keep="last")]

    return df

# -------------------------------------------------------------------
def _append_or_write(
    df_new: pd.DataFrame,
    out_path: Path,
    fmt: str
) -> Tuple[int, int, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    
    fmt = fmt.lower().replace(" ", "")

    # --- بخش خواندن فایل موجود (با مدیریت خطای کامل برای هر دو فرمت) ---
    if out_path.exists():
        df_old = pd.DataFrame()
        
        if fmt == "parquet":
            try:
                df_old = pd.read_parquet(out_path)
            except Exception:
                # اگر پارکت خراب است، سعی کن CSV هم‌نام را بخوان
                csv_fallback = out_path.with_suffix(".csv")
                if csv_fallback.exists():
                    try:
                        df_old = pd.read_csv(csv_fallback, parse_dates=["time"], index_col="time")
                    except Exception:
                        df_old = pd.DataFrame()  # اگر CSV هم خراب بود، از خالی شروع کن
                else:
                    df_old = pd.DataFrame()
        else:  # fmt == "csv"
            try:
                df_old = pd.read_csv(out_path, parse_dates=["time"], index_col="time")
            except Exception:
                # اگر CSV خراب است، از خالی شروع کن (مشابه رفتار پارکت)
                df_old = pd.DataFrame()
        
        df_old = normalize_df(df_old)
        before = len(df_old)

        parts = [x for x in (df_old, df_new) if x is not None and not x.empty]
        if parts:
            df_all = pd.concat(parts, axis=0)
        else:
            df_all = pd.DataFrame(columns=["open","high","low","close","volume","spread"],
                                  index=pd.DatetimeIndex([], name="time"))
        df_all = normalize_df(df_all)
    else:
        before = 0
        df_all = normalize_df(df_new)

    first_index = df_all.index[0] if len(df_all) > 0 else None
    last_index = df_all.index[-1] if len(df_all) > 0 else None
    
    # --- بخش ذخیره سازی (با پاکسازی فایل خراب قبلی در صورت لزوم) ---
    if fmt == "parquet":
        try:
            df_all.to_parquet(out_path)
        except Exception:
            # اگر پارکت ذخیره نشد، فایل پارکت قبلی (اگر خراب است) را حذف کن تا باعث سردرگمی نشود
            if out_path.exists():
                out_path.unlink()  # حذف فایل پارکت خراب/ناقص
            csv_path = out_path.with_suffix(".csv")
            df_all.to_csv(csv_path)
            return (before, len(df_all), first_index, last_index)
    else:  # fmt == "csv"
        df_all.to_csv(out_path)

    return (before, len(df_all), first_index, last_index)

# -------------------------------------------------------------------
def _write_metadata(
    raw_dir: Path,
    symbol: str,
    timeframe: str,
    first_index: pd.Timestamp,
    last_index: pd.Timestamp,
    rows: int,
    columns: List,
    fmt: str
) -> Path:
    """
    متادیتا (فایل JSON) را کنار داده ذخیره می‌کند تا برنامه‌های دیگر بتوانند سریع گزارش بگیرند.
    """
    meta = {
        "symbol": symbol,
        "timeframe": timeframe.upper(),

        # "first_index": df.index[0].isoformat(),
        # "last_index": df.index[-1].isoformat(),
        "first_index": first_index.isoformat(),
        "last_index": last_index.isoformat(),

        "rows": int(rows),
        "columns": list(columns),
        "format": fmt.lower(),
        "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "updated_at_tehran_time": datetime.now(ZoneInfo("Asia/Tehran")).replace(microsecond=0).isoformat()
    }
    meta_path = raw_dir / symbol / f"{timeframe.upper()}.meta.json"
    
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"metadata stored in {meta_path}")

    return meta_path

# -------------------------------------------------------------------
def _fetch_candles(connector: MT5Connector, p: DownloadPlan) -> pd.DataFrame:
    """ دریافت داده‌های کندلی از متاتریدر بر اساس طرح دانلود (DownloadPlan) """

    df = pd.DataFrame()
    _downloaded_by = ""

    if not p.symbol or not p.timeframe:
        raise ValueError("symbol and timeframe must be non-empty strings.")
    
    logger.debug("=== Before: connector.get_candles ==========================")  # save
    logger.debug(f"symbol = {p.symbol}")                                          # save
    logger.debug(f"timeframe = {p.timeframe}")                                    # save
    logger.debug(f"lookback_bars = {p.lookback_bars}")                            # save
    logger.debug(f"date_from = {p.date_from}")                                    # save
    logger.debug(f"date_to = {p.date_to}")                                        # save
    logger.debug(f"date_tz = {p.date_tz}")                                        # save
    logger.debug(f"result_tz = {p.result_tz}")                                    # save
    logger.debug(f"range_policy = {p.range_policy}")                              # save
    logger.debug("============================================================\n")  # save

    if p.date_from and p.date_to and p.lookback_bars:
        df = connector.get_candles_range(p.symbol, p.timeframe, p.date_from, p.date_to, p.date_tz, p.result_tz)
        _downloaded_by = "date"
        
        policy = str(p.range_policy).lower()    

        if   policy=="min" and len(df)> p.lookback_bars:
            df = df[ - p.lookback_bars:]
        elif policy=="min" and len(df)<=p.lookback_bars:
            pass
        elif policy=="max" and len(df)>=p.lookback_bars:
            pass
        elif policy=="max" and len(df)< p.lookback_bars:
            df = connector.get_candles_num(p.symbol, p.timeframe, p.lookback_bars, p.result_tz)
            _downloaded_by = "count"
        elif policy=="date":
            # df = connector.get_candles_range(p.symbol, p.timeframe, p.date_from, p.date_to, p.date_tz, p.result_tz)
            # _downloaded_by = "date"
            pass
        elif policy=="count":
            df = connector.get_candles_num(p.symbol, p.timeframe, p.lookback_bars, p.result_tz)
            _downloaded_by = "count"

        else:
            logger.warning("Check policy, date_from, date_to, lookback_bars.")

    elif (p.date_from and p.date_to) and not p.lookback_bars:
        df = connector.get_candles_range(p.symbol, p.timeframe, p.date_from, p.date_to, p.date_tz, p.result_tz)
        _downloaded_by = "date"

    elif (not p.date_from or not p.date_to) and p.lookback_bars:
        df = connector.get_candles_num(p.symbol, p.timeframe, p.lookback_bars, p.result_tz)
        _downloaded_by = "count"

    
    else:  # if every 3 ones is None or empty
        logger.error(
            f"Invalid DownloadPlan for {p.symbol} {p.timeframe}: "
            "either (lookback_bars) or (date_from and date_to) must be provided. Skipping..."
        )

    if _downloaded_by != "count" and _downloaded_by != "date":
        logger.debug("Check '_download_by' at '_fetch_candles()' !!!")
        return df, _downloaded_by

    logger.debug(f"=== Result of get_candles ==================================")  # save for debug
    logger.debug(f"symbol={p.symbol}")                                             # save for debug
    logger.debug(f"timeframe={p.timeframe}")                                       # save for debug
    if _downloaded_by == "count":                                                  # save for debug
        logger.debug(f"fetched candles by 'count'")                                # save for debug
        logger.debug(f"lookback_bars={p.lookback_bars}")                           # save for debug
    elif _downloaded_by == "date":                                                 # save for debug
        logger.debug(f"fetched candles by 'date'")                                 # save for debug
        logger.debug(f"date_from={p.date_from}")                                   # save for debug
        logger.debug(f"date_to={p.date_to}")                                       # save for debug
        logger.debug(f"date_tz = {p.date_tz}")                                     # save for debug
        logger.debug(f"result_tz = {p.result_tz}")                                 # save for debug
    logger.debug(f"columns of df is: {list(df.columns)}")                          # save for debug
    logger.debug(f"rows of df is: {len(df)}")                                      # save for debug
    logger.debug(f"============================================================\n")  # save for debug

    
    return df, _downloaded_by

# -------------------------------------------------------------------
# تبدیل datetime / str به datetime
# -------------------------------------------------------------------
def _to_datetime(value: datetime|str) -> datetime:
    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        return datetime.fromisoformat(value)

    raise TypeError("Datetime value must be datetime or str.")


# =======================================================================================
# هسته‌ی دانلود برای حالت batch
# =======================================================================================
class MT5DataLoader_batch:
    """لودر داده‌ی MT5. با MT5Connector کار می‌کند و داده‌ها را در data/raw ذخیره می‌کند """
    # ---------------------------------------------------------------
    # سازنده 
    # ---------------------------------------------------------------
    def __init__(self, 
                 cfg:       Optional[Dict[str, Any]] = None,
                 connector: Optional[MT5Connector]   = None,
                 ) -> None:
        
        # -- 1 -- config, raw_dir -----------------------------------

        logger.debug("=== start of init of MT5DataLoader_batch ===================")

        self.cfg: Dict[str, Any] = cfg or config_completer()
        self.raw_dir: Path = resolve_raw_dir(self.cfg)

        # -- 2 -- broker_timezone -----------------------------------

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

        # -- 3 -- download defaults ---------------------------------
        # گزینه‌های دانلود از config.download_defaults خوانده میشود 

        dl = (self.cfg.get("download_defaults") or {})

        self.default_symbols:    List[str] = list(dl.get("symbols") or [])
        self.default_timeframes: List[str] = list(dl.get("timeframes") or [])
        self.default_lookback:   int       = int (dl.get("lookback_bars") or 5_000_000)
                
        # -- 4 -- result_timezone -----------------------------------
        temp = dl.get("result_timezone")

        if temp is None:
            self.result_timezone = None
        else:
            self.result_timezone = _to_zoneinfo(temp)

        # -- 5 -- broker_date_from & broker_date_to -----------------   # <= گیت ورودی از config

        temp = dl.get("broker_date_from")
        self.date_from: datetime = normalize_datetime(temp, self.broker_timezone, "UTC")

        temp = dl.get("broker_date_to")
        self.date_to: datetime = normalize_datetime(temp, self.broker_timezone, "UTC")

        # -- 6 -- policy, save_format -------------------------------
        self.range_policy: str = str(dl.get("range_policy")).lower()

        self.save_format: str = str(dl.get("save_format", "csv")).lower()
        if self.save_format not in ("csv", "parquet"):
            logger.warning("save_format is unknown; falling back to csv.")
            self.save_format = "csv"

        # logger.debug(f"self.default_symbols = {self.default_symbols}")         # save for debug
        # logger.debug(f"self.default_timeframes = {self.default_timeframes}")   # save for debug
        # logger.debug(f"self.default_lookback = {self.default_lookback}")       # save for debug
        # logger.debug(f"self.date_from = {self.date_from}")                     # save for debug
        # logger.debug(f"self.date_to = {self.date_to}")                         # save for debug
        # logger.debug(f"self.broker_timezone = {self.broker_timezone}")         # save for debug
        # logger.debug(f"self.result_timezone = {self.result_timezone}")         # save for debug
        # logger.debug(f"self.range_policy = {self.range_policy}")               # save for debug
        # logger.debug(f"self.save_format = {self.save_format}")                 # save for debug

        # # -- 7 -- save_at_utc_time ----------------------------------
        # self.save_at_utc_time = bool(dl["save_at_utc_time"])
        
        # -- 8 -- connection to mt5 ---------------------------------
        self.conn = connector or MT5Connector(config=self.cfg)
        logger.debug("============================================================\n")

    # ---------------------------------------------------------------
    # ساخت طرح دانلود- متد کمکی
    # ---------------------------------------------------------------
    def _check_downloadplan_params(
        self,
        lookback_bars: Optional[int] = None,
        date_from: Optional[datetime | str] = None,
        date_to: Optional[datetime | str] = None,
        date_tz: Optional[tzinfo | str] = None,
        range_policy: Optional[str] = None,
    ):

        # ----- count -----------------------------------------------

        if range_policy == "count":
            if not isinstance(lookback_bars, int):
                raise ValueError("Check 'lookback_bars', 'range_policy'.")

            return (lookback_bars, None, None, None, range_policy)

        # ----- normalize date_tz -----------------------------------

        tz = _to_zoneinfo(date_tz)

        # ----- normalize / validate dates --------------------------

        if date_from is None or date_to is None:
            raise ValueError("Check 'date_from', 'date_to'.")

        if tz is not None:
            dt_from = normalize_datetime(date_from, tz, tz)
            dt_to = normalize_datetime(date_to, tz, tz)
        else:
            dt_from = _to_datetime(date_from)
            dt_to = _to_datetime(date_to)
            cond1 = (dt_from.tzinfo is None) or (dt_from.utcoffset() is None)
            cond2 = (dt_to.tzinfo is None) or (dt_to.utcoffset() is None)
            if cond1 or cond2:
                raise ValueError("Check 'date_from', 'date_to', 'date_tz'.")

        if dt_from is None or dt_to is None:
            raise ValueError("Check 'date_from', 'date_to'.")

        return (None, dt_from, dt_to, tz, range_policy)


    # ---------------------------------------------------------------
    # ساخت طرح دانلود- متد اصلی
    # ---------------------------------------------------------------
    def build_plan(
        self,
        symbols: Optional[Iterable[str]] = None,
        timeframes: Optional[Iterable[str]] = None,
        lookback_bars: Optional[int] = None,
        date_from: Optional[datetime | str] = None,
        date_to: Optional[datetime | str] = None,
        date_tz: Optional[tzinfo | str | None] = None,
        result_tz: Optional[tzinfo | str | None] = None,
        range_policy: Optional[str] = None,
    ) -> List[DownloadPlan]:

        """
        این تابع بر اساس آرگومان‌ها یا پیش‌فرض‌های کانفیگ، لیست DownloadPlan تولید می‌کند.
        """
        date_tz = _to_zoneinfo(date_tz)
        result_tz = _to_zoneinfo(result_tz)
        
        logger.debug("=== Class MT5DataLoader_batch: build_plan() is start =======")
        logger.debug(f"symbols = {symbols}")
        logger.debug(f"timeframes = {timeframes}")
        logger.debug(f"lookback_bars = {lookback_bars}")
        logger.debug(f"date_from = {date_from}")
        logger.debug(f"date_to = {date_to}")
        logger.debug(f"date_tz = {date_tz}") # فقط زمانی استفاده میشود که داده های ورودی منطقه زمانی نداشته باشند.
        logger.debug(f"result_tz = {result_tz}")
        logger.debug(f"range_policy = {range_policy}")
        logger.debug("============================================================\n")

        # -- 1 -- Assigning and fallbacks ---------------------------

        syms = list(symbols) if symbols is not None else self.default_symbols
        tfs = list(timeframes) if timeframes is not None else self.default_timeframes
        lb = int(lookback_bars) if lookback_bars is not None else self.default_lookback
        dt_from = date_from if date_from is not None else self.date_from
        dt_to = date_to if date_to is not None else self.date_to
        dt_tz = date_tz if date_tz is not None else self.broker_timezone
        rs_tz =result_tz if result_tz is not None else self.result_timezone
        rng_plcy = str(range_policy) if range_policy is not None else self.range_policy

        logger.debug("=== After assigning and fallbacks ==========================")
        logger.debug(f"symbols = {syms}")
        logger.debug(f"timeframes = {tfs}")
        logger.debug(f"lookback_bars = {lb}")
        logger.debug(f"date_from = {dt_from}")
        logger.debug(f"date_to = {dt_to}")
        logger.debug(f"date_tz = {dt_tz}")
        logger.debug(f"result_tz = {rs_tz}")
        logger.debug(f"range_policy = {rng_plcy}")
        logger.debug("============================================================\n")

        # -- 2 -- Standardize date_from & date_to -------------------

        if rng_plcy == "date":
            dt_from = normalize_datetime(dt_from, dt_tz, dt_tz)
            dt_to = normalize_datetime(dt_to, dt_tz, dt_tz)
            logger.debug(f"After normalize_datetime: dt_from = {dt_from}, dt_to = {dt_to}")
        
        # -- 3 -- Checking symbols & TFs ----------------------------

        if not syms or not tfs:
            raise ValueError(
                "symbols/timeframes are empty. "
                "Set them in config or arguments."
            )

        # -- 4 -- Validate range_policy -----------------------------

        valid_policies = {"min", "max", "date", "count"}

        if rng_plcy not in valid_policies:
            raise ValueError(
                f"Invalid range_policy: {rng_plcy!r}. "
                f"Expected one of {sorted(valid_policies)}."
            )

        # -- 5 -- Checking range ------------------------------------

        if rng_plcy == "date":
            lb = None
            if dt_from is None or dt_to is None:
                raise ValueError(
                    "date_from and date_to are required "
                    "for range_policy='date'."
                )
            if dt_from >= dt_to:
                raise ValueError("Check 'date_from' and 'date_to'.")

        elif rng_plcy == "count":
            if lb <= 0:
                raise ValueError("Check 'lookback_bars'.")

            dt_from = None
            dt_to = None

        elif rng_plcy in ("max", "min"):
            pass

        # -- 6 -- Wrapping plans ------------------------------------

        plans: List[DownloadPlan] = []

        logger.debug("=== Before _check_downloadplan_params() ====================")
        logger.debug(f"symbols = {syms}")
        logger.debug(f"timeframes = {tfs}")
        logger.debug(f"lookback_bars = {lb}")
        logger.debug(f"date_from = {dt_from}")
        logger.debug(f"date_to = {dt_to}")
        logger.debug(f"date_tz = {dt_tz}")
        logger.debug(f"result_tz = {rs_tz}")
        logger.debug(f"range_policy = {rng_plcy}")
        logger.debug("============================================================\n")

        (
            _lb,
            _dt_from,
            _dt_to,
            _dt_tz,
            _rng_plc,
        ) = self._check_downloadplan_params(
            lookback_bars=lb,
            date_from=dt_from,
            date_to=dt_to,
            date_tz=dt_tz,
            range_policy=rng_plcy,
        )
        logger.debug("=== After checking download plans ==========================")
        logger.debug(f"_lb = {_lb}, _dt_from = {_dt_from}, _dt_to = {_dt_to}")
        logger.debug(f"_tz = {_dt_tz}, result_tz = {rs_tz}, _rng_plc = {_rng_plc}")
        
        for s in syms:
            for tf in tfs:
                plans.append(
                    DownloadPlan(
                        symbol=s,
                        timeframe=tf,
                        lookback_bars=_lb,
                        date_from=_dt_from,
                        date_to=_dt_to,
                        date_tz=_dt_tz,
                        result_tz=rs_tz,
                        range_policy=_rng_plc,
                    )
                )

        return plans


    # ---------------------------------------------------------------
    # اجرای طرح دانلود برای حالت batch
    # ---------------------------------------------------------------
    def run_plan(self,
            plans: List[DownloadPlan],
            ) -> List[Dict[str, Any]]:
        """
        طرح را اجرا می‌کند و خلاصه‌ی هر کار را برمی‌گرداند.
        """
        # -- 1 -- اتصال به متاتریدر ---------------------------------

        if not self.conn.initialize():
            raise RuntimeError("Unable to connect to MT5. Check the credentials/terminal.")

        results: List[Dict[str, Any]] = []  # (Future) شاید بهتر باشد که این متغیر، یک دیکشنری از دیکشنری ها باشد
        # -- 2 -- حلقه روی تمام پلان های دانلود ---------------------
        for p in plans:
            try:
                # -- L1 -- fetching candles -------------------------

                logger.debug(f"=== Before _fetch_candles() ================================")   # save for debug
                logger.debug(f"symbol = {p.symbol}")                                            # save for debug
                logger.debug(f"timeframe = {p.timeframe}")                                      # save for debug
                logger.debug(f"lookback_bars = {p.lookback_bars}")                              # save for debug
                logger.debug(f"date_from = {p.date_from}")                                      # save for debug
                logger.debug(f"date_to = {p.date_to}")                                          # save for debug
                logger.debug(f"date_tz = {p.date_tz}")                                          # save for debug
                logger.debug(f"result_tz = {p.result_tz}")                                      # save for debug
                logger.debug(f"range_policy = {p.range_policy}")                                # save for debug
                logger.debug(f"============================================================\n") # save for debug

                df, _downloaded_by = _fetch_candles(self.conn, p)

                logger.debug(f"=== Result of _fetch_candles() =============================")   # save for debug
                logger.debug(f"is a pd.DataFrame: {isinstance(df, pd.DataFrame)}")              # save for debug
                logger.debug(f"clomuns of df= {df.columns}")                                    # save for debug
                logger.debug(f"rows of df= {len(df)}")                                          # save for debug
                logger.debug(f"============================================================\n") # save for debug


                # if not self.save_at_utc_time:                             # *** درست نمودن زمان داده های خام دانلود شده
                #     df.index = df.index.tz_convert(self.broker_timezone)  # *** درست نمودن زمان داده های خام دانلود شده
                

                # -- L2 -- Normalizing DataFrame --------------------

                df = normalize_df(df)
                logger.debug(f"=== Result of normalize_df() ===============================")   # save for debug
                logger.debug(f"Is a pd.DataFrame: {isinstance(df, pd.DataFrame)}")              # save for debug

                # -- L3 -- Logging requested and returned candles ---

                req = int(p.lookback_bars or self.default_lookback)
                if df is None or df.empty:
                    if _downloaded_by == "count":
                        logger.info("TF=%s | requested=%d | returned=0", p.timeframe, req )
                    else:
                        logger.info("TF=%s | date_from=%s | date_to=%s | returned=0",
                            p.timeframe, p.date_from, p.date_to)
                else:
                    if _downloaded_by == "count":
                        logger.info("\n")
                        logger.info("TF=%s | requested=%d | returned=%d | range=%s → %s",
                            p.timeframe, req, len(df), df.index.min(), df.index.max() )
                    else:
                        logger.info("TF=%s | date_from=%s | date_to=%s", p.timeframe, p.date_from, p.date_to )
                        logger.info("returned=%d | range=%s → %s", len(df), df.index.min(), df.index.max() )


                # -- L4 -- Writing downloaded dataframes to files ---

                out_path = full_file_path(self.raw_dir, p.symbol, p.timeframe, self.save_format)
                before, after, first_index, last_index = _append_or_write(df, out_path, self.save_format)

                # -- L5 -- Logging saved files & sizes --------------

                logger.info("Saved: %s %s → %s (rows: +%d / total %d)",
                            p.symbol, p.timeframe, out_path, after - before, after)

                # -- L6 -- Writing metadatas to files ---------------

                _write_metadata(self.raw_dir, p.symbol, p.timeframe, first_index, last_index, after, df.columns, self.save_format)
                # لاگر در داخل تابع لاگ را ثبت میکند

                # -- L6 -- Wrapping list of dictionaries ------------

                results.append({
                    "symbol": p.symbol,    #.upper(),
                    "timeframe": p.timeframe.upper(),
                    "rows_written": after - before,
                    "rows_total": after,
                    "file": str(out_path),
                    "dataframe": df,
                })
                logger.info(
                    f"resulted dict appent to 'result list': " 
                    f"symbol = {p.symbol}, timeframe = {p.timeframe}"
                )

            except Exception as ex:
                # -- L7 -- Exception handling -----------------------

                logger.exception("Error downloading/saving %s %s: %s", p.symbol, p.timeframe, ex)
                results.append({
                    "symbol": p.symbol,
                    "timeframe": p.timeframe.upper(),
                    "error": str(ex),
                })

        # -- 3 -- disconnetcing mt5 ---------------------------------
        self.conn.shutdown()

        return results      


# =============================================================================
# CLI
# =============================================================================
# -------------------------------------------------------------------
def _setup_logging_all(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-6s | %(filename)-22s | %(lineno)-4d : %(funcName)-18s | %(message)s",
        datefmt="%H:%M:%S",
    )

def _setup_logging_funcname(
    level: str = "debug",
    allowed_functions: list[str] | None = None,
) -> None:

    class FunctionFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if allowed_functions is None:
                return True
            return record.funcName in allowed_functions

    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler.addFilter(FunctionFilter(allowed_functions))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-6s | %(filename)-28s | "
            "%(lineno)-4d : %(funcName)-24s | %(message)s",
        datefmt="%H:%M:%S",
    )

    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # جلوگیری از باقی ماندن handlerهای قبلی
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

# -------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    path = str(project_root() / "f01_config" / "config.yaml")
    parser = argparse.ArgumentParser(description="Download data from MT5 and save to f03_data/raw (CSV/Parquet).")
    
    parser.add_argument("-c","--config", type=str, default=path, help="Config file path (default: f01_config/config.yaml)")
    parser.add_argument("--symbols", nargs="*", default=None, help="List of symbols (Example: XAUUSD EURUSD)")
    parser.add_argument("--timeframes", nargs="*", default=None, help="List of time frames (example: M5 H1)")
    parser.add_argument("--lookback", type=int, default=None, help="Number of closing candles to receive") #تعداد کندلهای انتهایی
    parser.add_argument("--brk_date_from", type=str, default=None, help="Start of interval (ISO 8601 like 2024-01-01T00:00:00Z)")
    parser.add_argument("--brk_date_to", type=str, default=None, help="End of interval (ISO8601)")
    parser.add_argument("--range_policy", type=str, default=None, choices=["date", "count"], help="Range policy")
    parser.add_argument("--save_format", type=str, default=None, choices=["csv", "parquet"], help="Storage format")
    parser.add_argument("--log_level", type=str, default="INFO", help="Log level: DEBUG/INFO/WARN/ERROR")
    
    return parser.parse_args()

# -------------------------------------------------------------------
def main() -> int:
    # --1 -- استخراج مقادیر از خط فرمان 
    args = _parse_args()
    
    # -- 2 -- ساخت لاگر و تعیین سطح آن، همراه با تعیین فرمت و فرمت زمان 
    _setup_logging_all(args.log_level)
    
    # --- این بلوک کد زیری باید حفظ بشود ---
    # _setup_logging_funcname(
    #     "debug",
    #     allowed_functions=[
    #         "main",
    #         "build_plan",
    #         "run_plan",
    #         "normalize_df",
    #         "_fetch_candles",
    #         "get_candles_num",
    #         "get_candles_range",
    #         "_normalize_date"
    #     ]
    # )

    # -- 3 -- بارگذاری کانفیگ (با ENV Override فعال)
    cfg = config_completer(args.config, enable_env_override=True)
    
    # -- 4 -- اوور راید موقتی فرمت بر روی کانفیگ
    # اگر کاربر فرمت را در CLI تعیین کرد، آن را در cfg منعکس کنیم (Override موقتی) 
    #setdefault راه کوتاهی است برای «دریافت مقدار یا ایجاد/قرار دادن مقدار پیش‌فرض در صورت نبودن» — خواندن + نوشتن هم‌زمان 
    if args.save_format:
        # cfg.setdefault("download_defaults", {})
        # cfg["download_defaults"]["save_format"] = args.format
        cfg.setdefault("download_defaults", {})["save_format"] = args.save_format
    # -- 5 -- ساخت کانکتور
    my_connector = MT5Connector(config=cfg)

    # -- 6 -- ساخت بچ لودر
    loader = MT5DataLoader_batch(cfg=cfg, connector=my_connector)


    # -- 7 -- ساخت پلان های دانلود
    # if args.brk_date_from is not None:
    #     date_from = normalize_datetime(args.brk_date_from, ZoneInfo("Europe/Athens"))   # <= گیت ورودی از CLI
    # if args.brk_date_to is not None:
    #     date_to   = normalize_datetime(args.brk_date_to  , ZoneInfo("Europe/Athens"))   # <= گیت ورودی از CLI

    # --- پلان اولی برای دیباگ است ---
    # plans = loader.build_plan(
    #     symbols=None,
    #     timeframes=None,
    #     lookback_bars=None,
    #     date_from=None,
    #     date_to=None,
    #     ## date_tz=ZoneInfo("Asia/Tehran"),
    #     ## result_tz="Asia/Tehran",
    #     range_policy="count",
    # )

    # --- پلان دومی برای اصل برنامه است ---
    if args.range_policy.lower() == "date" or args.range_policy.lower() == "count":
        rng_policy = args.range_policy.lower()
    else:
        rng_policy = cfg.get("download_defaults").get("range_policy")

    if rng_policy == "date":
        plans = loader.build_plan(
            symbols=args.symbols,
            timeframes=args.timeframes,
            # lookback_bars=args.lookback,
            date_from=datetime.fromisoformat(args.brk_date_from),
            date_to=datetime.fromisoformat(args.brk_date_to),
            date_tz=ZoneInfo("Europe/Athens"),
            result_tz="UTC",
            range_policy=args.range_policy,
        )
    elif rng_policy == "count":
        plans = loader.build_plan(
            symbols=args.symbols,
            timeframes=args.timeframes,
            lookback_bars=args.lookback,
            # date_from=datetime.fromisoformat(args.brk_date_from),
            # date_to=datetime.fromisoformat(args.brk_date_to),
            date_tz=ZoneInfo("Europe/Athens"),
            result_tz="UTC",
            range_policy=args.range_policy,
        )


    # -- 8 -- اجرای لودر و دریافت نتیجه دانلودها
    results = loader.run_plan(plans)

    # -- 9 -- نوشتن نام ستونهای محصول
    for res in results:
        symbol = res.get("symbol")
        tf = res.get("timeframe")
        df = res.get("dataframe")  # اگر کلید موجود نباشد None برمی‌گرداند

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        # df.to_csv(f"___{symbol}_{tf}.csv")
        # logger.info(f"Df saved in root of project: ___{symbol}_{tf}.csv")
        logger.info(f"Df can save in root of project: ___{symbol}_{tf}.csv")

    # -- 10 -- گزارش خلاصه 
    ok  = [r for r in results if "error" not in r]
    bad = [r for r in results if "error"     in r]

    logger.info("Summary: Successful %d | Error %d", len(ok), len(bad))
    if bad:
        for r in bad:
            logger.error("Failed: %s %s → %s", r.get("symbol"), r.get("timeframe"), r.get("error"))
        return 2
    return 0

# -------------------------------------------------------------------
# اجرای از طریق فراخوانی مستقیم این فایل، سبب میشود که داده های جدید در ریشه پروژه ذخیره شوند
if __name__ == "__main__":
    raise SystemExit(main())

# ============================================================================= END