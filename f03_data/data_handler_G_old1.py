# f03_data/data_handler_G.py (3)
#
# Last reviewed: 1405/06/25
# =======================================================================================
""" ---> Docstring:
DataHandler چندتایم‌فریمی لایه f03_data برای مسیر batch و live.

این ماژول داده خام یک نماد را از فایل‌های data/raw می‌خواند، timezone آن را در مرز
ورودی به UTC تبدیل می‌کند، و داده را در قالب MTFDataset با یک DataFrame مستقل برای
هر تایم‌فریم نگهداری می‌کند. همچنین در مسیر live، cache هر نماد را با داده‌های دریافتی
از MarketDataEngine به‌روزرسانی می‌کند.

مسئولیت‌های اصلی:
    - خواندن فایل خام CSV/Parquet با _read_raw_df().
    - تفسیر فایل‌های ذخیره‌شده روی دیسک به عنوان داده‌های naive در صورت نداشتن timezone،
        نسبت‌دادن broker_timezone به آنها و سپس تبدیل index به UTC.
    - پشتیبانی از انتخاب داده در modeهای number و time؛ در mode time بازه به صورت
        [start_time, end_time) در نظر گرفته می‌شود و بازه بدون overlap یا نامعتبر DataFrame
        خالی ایجاد می‌کند.
    - ساخت BuildParams و تعیین base timeframe و سایر timeframeها با check_tfs().
    - ساخت MTFDataset در build() بدون هم‌ترازسازی یا merge کردن DataFrameهای تایم‌فریم‌های مختلف.
    - ذخیره مجموعه پردازش‌شده و manifest در save().
    - دریافت eventهای live، به‌روزرسانی cache مستقل هر نماد/تایم‌فریم و بازگرداندن MTFDataset
        جدید از cache.
    - نگهداری latest dataset و امکان ثبت callback برای مصرف داده live.

قرارداد timezone:
    - خروجی _read_raw_df() و در نتیجه build() بر اساس کد فعلی UTC-aware است.
    - داده live دریافتی از مسیر MarketDataEngine نیز در مسیر cache به عنوان داده زمانی پروژه
        با مبنای UTC مصرف می‌شود.

قرارداد symbol/timeframe:
    - DataHandler برای یک symbol ساخته می‌شود و cache آن نیز برای همان symbol نگهداری می‌شود.
    - MTFDataset شامل frames مستقل است و هیچ alignment یا cross-timeframe merge در این لایه انجام نمی‌دهد.

این ماژول نقطه اصلی تحویل داده batch/live به لایه‌های بالادستی است.

Run: روش های اجرا از خظ فرمان یا CIL (Command Line Interface)
    - بدون base_tf ، یعنی براساس آنچه در کانفیگ داده شده:

python -m f03_data.data_handler_F_3 `
    -c ./f01_config/config_0_1.yaml `
    --symbol XAUUSD_i `
    --timeframes M10 M30 H1 `
    --format parquet

python -m f03_data.data_handler_F_3 `
    --symbol BITCOIN `
    --save_format csv

"""
# =======================================================================================

# f03_data/data_handler_G.py (3)
# =======================================================================================
# Imports & Logger
# =======================================================================================
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING   # , Iterable

from pathlib import Path
from datetime import datetime, timezone
import logging
import pandas as pd
import json

# ------------------ Importing Internal Modules -----------------------------------------
from f03_data.mt5_data_loader_E import _setup_logging_funcname,  MT5DataLoader_batch
from f03_data.data_layer_functions import check_tfs    #, prefix_columns

from f03_data.mt5_data_loader_E import normalize_df
from f03_data.mtf_dataset import MTFDataset

from f02_utils.config_path_funcs import project_root, resolve_raw_dir, resolve_process_dir, full_file_path
from f02_utils.functions.constants import _TF_MINUTES, _TF_MAP, normalize_and_sort_timeframe_dicts as normalize_dict
if TYPE_CHECKING:
    from f03_data.live_market_engine import EventBus

# -------------------- Logger for this module -------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =======================================================================================
# 1-- دانلود و نرمال‌سازی دیتافریم‌های خام
# =======================================================================================
def _read_raw_df(
    path: Path,
    mode: Literal["number", "time"] = "number",      # can be: "number" or "time"
    start_lastrows: Optional[int] = None,               # contain 
    end_lastrows: Optional[int] = None,                 # NOT contain
    start_time: Optional[Union[str, datetime]] = None,     # contain
    end_time: Optional[Union[str, datetime]] = None,       # NOT contain
    columns: Optional[List[str]] = None,
    broker_timezone: str = "UTC",
) -> pd.DataFrame:
    # logger.debug("==== start _read_raw_df ========================")  # for debug
    # logger.debug(f"path = {path}")                                    # for debug
    # logger.debug(f"mode = {mode}")                                    # for debug
    # logger.debug(f"start_lastrows = {start_lastrows}")                # for debug
    # logger.debug(f"end_lastrows = {end_lastrows}")                    # for debug
    # logger.debug(f"start_time = {start_time}")                        # for debug
    # logger.debug(f"end_time = {end_time}")                            # for debug

    """
    *** تا قبل از این تابع تمام داده های ذخیره شده روی هارد naive هستند.
    *** در انتهای این تابع ابتدا منطقه زمانی بروکر به داده ها نسبت داده میشود،
    *** سپس منطقه زمانی آنها به UTC تبدیل میشود.
    *** به این ترتیب از ابتدای dataHandler ربات بر اساس داده های UTC کار میکند

    1) تلاش برای خواندن parquet و در صورت خطا/نبود، فرمت CSV را امتحان می‌کند.
    2) مقدار mode اجباراً باید تعیین شده باشد. وگرنه با خطا مواجه میشود.
    3) مقادیر start, end میتوانند تعریف نشوند و None باشند.
    4) دیتا فریم محصول را همانند نرمال سازی موجود در فایل mt5_data_loader.py نرمال میکند
    5) به دیتافریم نهایی شیفت زمانی داده میشود تا داده ها مطابق با utc بشوند.
    """
    # ===============================================================
    # اعمال شرطهای کنترلی برای انتهای بازه ها
    # ===============================================================
    end_lastrows_is_none = False
    end_time_is_none = False
    if end_lastrows is None: end_lastrows_is_none = True
    if end_time is None: end_time_is_none = True

    df = pd.DataFrame()
    # ===============================================================
    # Read parquet file
    # ===============================================================
    readparquet = False
    if path.suffix.lower() == ".parquet" and path.exists():
        logger.debug(f" ====> {path} is exist and found.")
        try:
            df = pd.read_parquet(path)
            # logger.debug(f" ====> after read parquet: type of index = {type(df.index[0])}")  # for debug
            # logger.debug(f" ====>     index class = {type(df.index)}")
            # logger.debug(f" ====>     index dtype = {df.index.dtype}")
            # logger.debug(f" ====>     index tz = {getattr(df.index, 'tz', None)}")
            # logger.debug(f" ====> after read parquet: len of DF is {len(df)}")               # for debug
            # logger.debug(f" ====> after read parquet: columns of DF is {df.columns}")        # for debug
            # logger.debug(f" ====> after read parquet: 1st row index is {df.index[0]}")       # for debug
            # logger.debug(f" ====> after read parquet: last row index is {df.index[-1]}")     # for debug

            readparquet = True 
        except Exception as ex:
            logger.warning("Failed to read Parquet (%s). Switching to CSV.", ex)

    # ===============================================================
    # Read csv file
    # ===============================================================
    if readparquet == False:
        csv_path = path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")
        if csv_path.exists():
            logger.debug(f" ====> {csv_path} is exist and found.")
            try:
                df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
                # logger.debug(f" ====> after read csv: type of index = {type(df.index[0])}")  # for debug
                # logger.debug(f" ====>     index class = {type(df.index)}")
                # logger.debug(f" ====>     index dtype = {df.index.dtype}")
                # logger.debug(f" ====>     index tz = {getattr(df.index, 'tz', None)}")
                # logger.debug(f" ====> after read csv: len of DF is {len(df)}")               # for debug
                # logger.debug(f" ====> after read csv: columns of DF is {df.columns}")        # for debug
                # logger.debug(f" ====> after read csv: 1st row index is {df.index[0]}")       # for debug
                # logger.debug(f" ====> after read csv: last row index is {df.index[-1]}")     # for debug
            except Exception as ex:
                logger.warning("Failed to read CSV (%s). Return None", ex)

    # ===============================================================
    # بررسی و تبدیل ایندکس به DatetimeIndex و هم‌سوسازی با UTC
    # ===============================================================
    if broker_timezone is None: broker_timezone="UTC"
    
    if isinstance(df.index, pd.DatetimeIndex):           # اگر اندکس از نوع زمانی است
        if df.index.tz is not None:                # اگر در اندکس، منطقه زمانی وجود دارد
            if str(df.index.tz) != "UTC":    # اگر منطقه زمانی برابر با UTC نیست
                df.index = df.index.tz_convert("UTC")   # منطقه زمانی را به UTC تبدیل کن
            else:                            # در غیر اینصورت
                pass                         # بیخیال عبور کن
        else:   # df.index.tz is None              # اگر منطقه زمانی ندارد
            df.index = df.index.tz_localize(broker_timezone)  # به داده ها منطقه زمانی بروکر را نسبت بده
            df.index = df.index.tz_convert("UTC")             # زمان را توسط تبدیل منطقه زمانی، به زمان UTC تبدیل کن
    
    elif not isinstance(df.index, pd.DatetimeIndex):     # اگر اندکس از نوع زمانی نیست
        if 'time' in df.columns:                   #  اگر ستون time در لیست ستونها وجود دارد
            df['time'] = pd.to_datetime(df['time'], tz=broker_timezone)   # ستون time را به نوع زمان با منطقه زمانی بروکر تبدیل کن
            df = df.set_index('time')              # سپس همان ستون time را اندکس دیتافریم قرار بده
            df.index = df.index.tz_convert('UTC')  # منطقه زمانی را به UTC تبدیل کن 
        else:                                      # اگر ستون time وجود ندارد، خطا بده
            raise ValueError("DataFrame index is not DatetimeIndex and no 'time' column found.")    

    # logger.debug(f" ====> after check timezones: type of index = {type(df.index[0])}")  # for debug
    # logger.debug(f" ====> after check timezones: len of DF is {len(df)}")               # for debug
    # logger.debug(f" ====> after check timezones: columns of DF is {df.columns}")        # for debug
    # logger.debug(f" ====> after check timezones: 1st row index is {df.index[0]}")       # for debug
    # logger.debug(f" ====> after check timezones: last row index is {df.index[-1]}")     # for debug

    # ===============================================================
    # Check & outputs
    # ===============================================================
    if len(df) == 0:
        selected_rows = pd.DataFrame()
    else:
        # --- modes ------------------------------------------------- start
        if mode == "number":
            if start_lastrows is None: start_lastrows = len(df)
            if end_lastrows is None: end_lastrows  = 0

            if (start_lastrows > 0) and (end_lastrows >= 0) and (start_lastrows > end_lastrows):
                first_row = max(0, len(df) - start_lastrows)
                last_row  = max(0, len(df) -   end_lastrows)
                # ---/-start-----------
                if end_lastrows_is_none:
                    selected_rows = df[first_row:]
                else:
                    selected_rows = df[first_row:last_row]
                # ---/-end-------------
            elif start_lastrows == 0:
                selected_rows = pd.DataFrame()
            elif (start_lastrows < 0) or (end_lastrows < 0):
                raise ValueError("'start_lastrows'/'end_lastrows' cannot be negative")
        
        elif mode == "time":
            # logger.debug("=============== > up to here.")  # for debug
            first_time = df.index[0] if start_time is None else pd.to_datetime(start_time, utc=True) # old time
            last_time =  pd.Timestamp.now(tz='UTC') if end_time is None else pd.to_datetime(end_time, utc=True)  # new time

            # logger.debug(f" ===== > first_time = {first_time}")  # for debug
            # logger.debug(f" ===== > last_time = {last_time}")  # for debug

            if first_time < last_time:

                # start = max(first_time, df.index[0])
                # # end = min(last_time, df.index[-1])
                # end = last_time


                data_start_time = df.index[0]   # ---------------- new part1 start
                data_end_time = df.index[-1]

                # Requested interval and available-data interval do not overlap.
                if last_time <= data_start_time or first_time > data_end_time:
                    logger.info(
                        "Requested time range has no overlap with available data. "
                        "requested=[%s, %s), available=[%s, %s]. "
                        "Returning empty DataFrame.",
                        first_time,
                        last_time,
                        data_start_time,
                        data_end_time,
                    )
                    selected_rows = pd.DataFrame()
                else:
                    start = max(first_time, data_start_time)
                    end = last_time          # ---------------- new part1 end


                    # ---/-start-----------
                    if end_time_is_none:
                        myfilter = (df.index >= start)
                    else:
                        myfilter = (df.index >= start) & (df.index < end)   # <<<<<<<<<<==========
                    # ---/-end-------------
                    selected_rows = df[myfilter]
                logger.debug(f" =====> len(selected_rows) = {selected_rows}")
            else:
                # logger.warning("Start_time is bigger than end_time! Return empty dataframe")

                logger.info(                          # ---------------- new part2 start
                    "Invalid/empty requested time range: "
                    "start_time=%s, end_time=%s. Returning empty DataFrame.",
                    first_time,
                    last_time,
                )                                     # ---------------- new part2 end

                selected_rows = pd.DataFrame()
                
        else:  # mode not in ["number", "time"]
            raise ValueError("mode can be: 'number' or 'time'. Check mode")
        # --- modes ------------------------------------------------- end

    # ===============================================================
    # columns
    # ===============================================================
    if columns is None:
        cols = selected_rows.columns
    if columns is not None:
        cols = [c for c in columns if c in selected_rows.columns]
    result = normalize_df(selected_rows[cols])

    logger.debug(f"len of result of _read_raw_df = {len(result)}")

    # --- output ----------------------------------------------------
    # logger.debug("==== end _read_raw_df ==========================")  # for debug
    return result

# =======================================================================================
# 2-- (پنجره انتخاب دادهای خام) برای پردازش
# =======================================================================================
def _get_range_by_timeframe(
    timeframe: str,           # اجباری
    from_last_n: int,         # اجباری
    to_last_n: int,           # اجباری
    base_time: Optional[Union[str, pd.Timestamp]] = None  # اختیاری
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    محاسبه بازه زمانی بر اساس تعداد کندل‌های کامل و بسته‌شده از تایم‌فریم tf1.
    شروع هفته = دوشنبه ساعت 00:00:00 UTC
    پایان هفته = یکشنبه ساعت 23:59:59.999999 UTC

    پارامترها:
        from_last_n: تعداد کندل‌های tf1 به عقب برای شروع بازه (بزرگتر)
        to_last_n: تعداد کندل‌های tf1 به عقب برای پایان بازه (کوچکتر)
        tf1: تایم‌فریم مبنا (مثلاً 'W1' برای هفته، 'D1' برای روز)
        base_time: زمان مبنا (پیش‌فرض: زمان حال UTC)
    
    خروجی:
        (start_time, end_time) به صورت pandas Timestamp با timezone UTC
    
    ---------------------------------------------------------
    برای استفاده از این تابع
    ---------------------------------------------------------
    import pandas as pd

    - روش ۱: استفاده از pd.Timestamp
    ts1 = pd.Timestamp("2026-02-03 04:05:06")

    - روش ۲: استفاده از pd.to_datetime
    ts2 = pd.to_datetime("2026-02-03 04:05:06")

    - تنظیم timezone به UTC (توصیه می‌شود)
    ts3 = pd.Timestamp("2026-02-03 04:05:06", tz='UTC')
    ts4 = pd.to_datetime("2026-02-03 04:05:06", utc=True)

    ---------------------------
    from datetime import datetime

    dt = datetime(2026, 2, 3, 4, 5, 6)
    ts = pd.Timestamp(dt)  # تبدیل می‌شود
    ---------------------------------------------------------
    """        

    # ===============================================================
    # Internal Func-1
    # =============================================================== OK
    def _get_candle_start_base(time: pd.Timestamp, minutes: int) -> pd.Timestamp:
        """محاسبه زمان شروع کندل برای تایم‌فریم‌های دقیقه‌ای و ساعتی."""
        total_minutes = time.hour * 60 + time.minute
        adjusted_minutes = (total_minutes // minutes) * minutes
        return time.floor('D') + pd.Timedelta(minutes=adjusted_minutes)

    # ===============================================================
    # Internal Func-2
    # =============================================================== OK
    def _last_closedcandle_time(
        timeframe: str,
        steps: int,
        base_time: Union[str, pd.Timestamp, datetime]
    ) -> pd.Timestamp:
        """
        محاسبه زمان شروع کندل شماره 'steps' از آخرین کندل کامل قبل از base_time.

        پارامترها:
            timeframe: تایم‌فریم (مثلاً 'D1', 'H4', 'M5', 'W1', 'MN1')
            steps: تعداد کندل‌های کامل به عقب (1 = آخرین کندل کامل)
            base_time: زمان مبنا (پیش‌فرض: زمان حال)

        خروجی:
            زمان شروع کندل (pd.Timestamp با timezone UTC)
        """

        # =======================================
        # 1) اعتبار سنجی timeframe
        # =======================================
        temp_tf = timeframe.upper().replace(" ", "")
        tf_key = _TF_MAP.get(temp_tf, temp_tf)
        minutes = _TF_MINUTES.get(tf_key)
        if minutes is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        offset = pd.Timedelta(minutes=minutes)

        # =======================================
        # 2) محاسبه زمان شروع آخرین کندل کامل قبل از base_time
        # =======================================
        if tf_key == 'W1':
            # start_of_period = base_time - pd.Timedelta(days=base_time.weekday())            # ابتدای هفته (دوشنبه 00:00:00)
            start_of_period = base_time - pd.Timedelta(days=(base_time.weekday() + 1) % 7)  # ابتدای هفته (یکشنبه 00:00:00)
            start_of_period = start_of_period.floor('D')
            period_offset = pd.Timedelta(days=7)
        elif tf_key == 'MN1':
            # ابتدای ماه (روز اول 00:00:00)
            start_of_period = base_time.floor('D') - pd.Timedelta(days=base_time.day - 1)
            period_offset = pd.DateOffset(months=1)  # برای مقایسه باید به Timestamp تبدیل شود
        else:
            # دقیقه‌ای یا ساعتی
            start_of_period = _get_candle_start_base(base_time, minutes)
            period_offset = offset

        # بررسی کامل بودن کندل
        if tf_key == 'MN1':
            # برای ماه، پایان دوره را با DateOffset محاسبه می‌کنیم
            end_of_period = start_of_period + period_offset
        else:
            end_of_period = start_of_period + period_offset

        # اگر base_time بعد از پایان کندل باشد، کندل کامل شده است
        if base_time >= end_of_period:
            last_complete_start = start_of_period
        else:
            last_complete_start = start_of_period - (period_offset if tf_key in ('W1', 'MN1') else offset)

        # برگرداندن کندل شماره 'steps' به عقب
        if tf_key == 'MN1':
            return last_complete_start - (period_offset * (steps - 2))
        else:
            return last_complete_start - offset * (steps - 2)

    # ===============================================================
    # 1) اعتبارسنجی ها
    # ===============================================================
    from_none = from_last_n is None
    to_none = to_last_n is None

    if (from_none and not to_none) or (not from_none and to_none):   # 1=None, 1=not None
        if timeframe is None:
            raise ValueError("'timeframe' must be defined.")
    if (not from_none) and (not to_none):
        if from_last_n < to_last_n:
            raise ValueError("'from_last_n' must be greater than 'to_last_n'.")
        if from_last_n < 0 or to_last_n < 0:
            raise ValueError("'from_last_n' and 'to_last_n' must be positive.")
    # ===============================================================
    # 2) تنظیم base_time به UTC
    # ===============================================================
    if base_time is None:
        base_time = pd.Timestamp.now(tz='UTC')
        logger.debug(f"new base_time is {base_time}")
    # else:
    #     base_time = pd.to_datetime(base_time, utc=True)
    # ===============================================================
    # 3) یکسان‌سازی نام تایم‌فریم‌ها
    # ===============================================================
    if from_last_n is None:
        start = None
    else:
        start = _last_closedcandle_time(timeframe=timeframe, steps=from_last_n, base_time=base_time)
        start = ensure_timezone_aware(start, tz="UTC")
    if to_last_n is None:
        end = None
    else:
        end = _last_closedcandle_time(timeframe=timeframe, steps=to_last_n, base_time=base_time)
        end = ensure_timezone_aware(end, tz="UTC")

    return start, end, base_time

# =======================================================================================
# 3-- ویژگی‌های زمانی و سشن‌ها 
# =======================================================================================
def ensure_timezone_aware(dt, tz="UTC"):
    """
    بررسی می‌کند که متغیر زمانی از منطقه‌ی زمانی آگاه است یا نه.
    اگر نبود، منطقه‌ی زمانی مشخص شده را به آن اضافه می‌کند.

    پارامترها:
        dt: ورودی می‌تواند pd.Timestamp، datetime.datetime یا str باشد
        tz: منطقه‌ی زمانی مورد نظر (پیش‌فرض: "UTC")

    خروجی:
        pd.Timestamp با منطقه‌ی زمانی مشخص
    """
    # تبدیل به pandas Timestamp
    if not isinstance(dt, pd.Timestamp):
        dt = pd.to_datetime(dt)

    # بررسی آگاه بودن از منطقه‌ی زمانی
    if dt.tz is None:
        # اگر ناآگاه است، منطقه را اضافه کن
        return dt.tz_localize(tz)
    else:
        # اگر آگاه است، به منطقه‌ی مورد نظر تبدیل کن
        return dt.tz_convert(tz)
    
    """ مثالهایی برا استفاده از تابع بالا
        ==========================================
        # مثال 1: زمان ناآگاه (naive)
        naive_dt = datetime(2026, 7, 7, 10, 30)
        aware = ensure_timezone_aware(naive_dt, "UTC")
        print(aware)  # 2026-07-07 10:30:00+00:00

        # مثال 2: زمان آگاه (aware) با منطقه‌ی دیگر
        aware_dt = pd.Timestamp("2026-07-07 10:30:00", tz="Asia/Tehran")
        converted = ensure_timezone_aware(aware_dt, "UTC")
        print(converted)  # 2026-07-07 07:00:00+00:00

        # مثال 3: رشته
        str_dt = "2026-07-07 10:30:00"
        result = ensure_timezone_aware(str_dt, "UTC")
        print(result)  # 2026-07-07 10:30:00+00:00
        =============================================
    """

# =======================================================================================
# کلاس کمکی BuildParams
# =======================================================================================
class BuildParams:
    """ برای یک نماد و در یک تایمفریم مبنای مشخص """
    def __init__(self,
        symbol: str,                       
        base_tf: str,
        timeframes: Optional[List[str]],      # در این کلاس فرض شده است که این پارامتر شامل تمام تایم فرمها است.
        selected_tf: Optional[str] = None,    # برای ذخیره نمودن (نام تایم فریم جاری) که باید بارگیری بشود                       
        #--------------------
        load_format: Literal["csv", "parquet"] = "parquet",   # فرمت خواندن داده های خام از روی هارد
        mode: Literal["number", "time", "periods"] = "number",
        #-------------------- for mode = "number"
        start_lastrows: int = None,
        end_lastrows: int = None,
        #-------------------- for mode = "time"
        start_time: datetime = None,
        end_time: datetime = None,       
        #-------------------- for mode = "periods"
        period_size: str = None,    # برای استفاده در تابع _get_range_by_timeframe سه گانه پنجره انتخاب داده های خام
        from_last_n: int = None,    # برای استفاده در تابع _get_range_by_timeframe سه گانه پنجره انتخاب داده های خام
        to_last_n: int = None,      # برای استفاده در تابع _get_range_by_timeframe سه گانه پنجره انتخاب داده های خام
        base_time: datetime = None  # برای استفاده در تابع _get_range_by_timeframe سه گانه پنجره انتخاب داده های خام
    ) -> None:
        # --- Validations -------------------------------------------
        if symbol is None:
            raise ValueError("Must define 'symbol'")
        
        # -----
        # وقتی که مصرف کننده این کلاس از config_completer استفاده کرده باشد،
        # در بخش زیر، دیگر نیازی به check_tfs نیست.
        _base_tf, _other_tfs, all_tfs = check_tfs(base_tf, timeframes)
        if _base_tf != base_tf:
            logger.info(f"base_tf changed from {base_tf} to {_base_tf}")
        if set(_other_tfs) != set(timeframes[1:]):
            logger.info(f"timeframes changed from {timeframes[1:]} to {_other_tfs}")
        # -----
        
        if selected_tf is not None:
            selected_tf = _TF_MAP.get(selected_tf.upper().replace(" ", ""))
        # else:    # این بخش حذف شد. بجای آن فالبک شد به تایم فریم مبنا
        #     raise ValueError(f"selected_tf: {selected_tf} is invalid.")
            
        if period_size is not None:
            period_size = _TF_MAP.get(period_size.upper().replace(" ", ""))

        # --- Initial mountings -------------------------------------
        self.symbol = symbol
        self.base_tf = _base_tf
        self.timeframes = all_tfs  #_other_tfs # <=== توجه شود: در خروجی این کلاس- برخلاف ورودی آن- این پارامتر شامل تایمفریمهای غیر پایه است.
        self.selected_tf = selected_tf or _base_tf
        self.load_format = load_format
        self.mode = mode
        self.start_lastrows = start_lastrows
        self.end_lastrows = end_lastrows
        self.start_time = ensure_timezone_aware(start_time, tz="UTC") if start_time is not None else None
        self.end_time = ensure_timezone_aware(end_time, tz="UTC") if end_time is not None else None
        self.period_size = period_size
        self.from_last_n = from_last_n
        self.to_last_n = to_last_n
        self.base_time = ensure_timezone_aware(base_time, tz="UTC") if base_time is not None else None

        # محاسبه بازه زمانی با استفاده از تابع _get_range_by_timeframe
        now_time = None # ==>> برای اطمینان از اینکه تابع _get_range_by_timeframe اجرا شده است
        if mode == "periods":           
            start, end, now_time = _get_range_by_timeframe(
                from_last_n=from_last_n,
                to_last_n=to_last_n,
                timeframe=period_size,
                base_time=self.base_time,
            )
            if now_time is not None:
                self.mode = "time"
                self.start_time = start
                self.end_time = end
                self.base_time = now_time

        self.output_dict = {
            "symbol": self.symbol,
            "base_tf" : self.base_tf,
            "_other_tfs" : self.timeframes,
            "selected_tf" : self.selected_tf,
            "load_format" : self.load_format,
            "mode" : self.mode,
            "start_lastrows" : self.start_lastrows,
            "end_lastrows" : self.end_lastrows,
            "start_time" : self.start_time,
            "end_time" : self.end_time,
            "period_size" : self.period_size,
            "from_last_n" : self.from_last_n,
            "to_last_n" : self.to_last_n,
            "base_time" : self.base_time,
        }
    
    def print_params(self) -> None:
        print("-"*45)
        for item in self.output_dict:
            # print(item)
            print(f" {item}", " "*(17-len(item)) , ":", f" {self.output_dict[item]}")
        print("-"*45, "\n")


# =======================================================================================
# کلاس اصلی DataHandler 
# =======================================================================================
class DataHandler:
    """
    سازنده‌ی دیتاست پردازش‌شده‌ی چند-تایم‌فریم برای آموزش/بک‌تست/اجرا.
    """
    # -------------------------------------------------------------------------
    # 1- سازنده 
        # -------------------------------------------------------------------------
    def __init__(
        self,
        cfg: Dict[str, Any],
        symbol: str,
        required_bars: Optional[Dict[str, int]] = None,
        event_bus: Optional[EventBus] = None,  # Deleted at 1405/05/21-14:42
    ) -> None:

        # -1-- Setting config ---------------------------------------
        self.cfg: Dict[str, Any] = cfg
        self.symbol: str = symbol
        # print(f" === 1 === self.symbol ===> {self.symbol}")   # for debug

        # -2-- Setting directories ----------------------------------
        self.raw_dir: Path = resolve_raw_dir(self.cfg)
        self.proc_dir: Path = resolve_process_dir(self.cfg)
        # print(f" === 2 === self.raw_dir ===> {self.raw_dir}")   # for debug
        # print(f" === 3 === self.proc_dir ===> {self.proc_dir}")   # for debug

        # -3-- Saving format ----------------------------------------
        dl = (self.cfg.get("features") or {})
        self.save_format: str = str(
            dl.get("save_format", "parquet")
        ).lower()

        if self.save_format not in ("csv", "parquet"):
            self.save_format = "parquet"

        # -4-- Required bars / Timeframes / Base timeframe ----------
        # required_bars is the primary source for this DataHandler
        # instance.
        #
        # If required_bars is not supplied explicitly, use the final
        # normalized/merged requirement prepared by config_completer:
        #
        #     __warmups_dicts
        #            +
        #     __candles_required_bars
        #            ↓
        #     __all_required_bars
        #

        if required_bars is None:
            try:
                source_required_bars = self.cfg["__all_required_bars"][symbol]
            except KeyError as ex:
                raise ValueError(
                    f"Required bars for symbol '{symbol}' were not provided "
                    f"and '{symbol}' was not found in cfg['__all_required_bars']."
                ) from ex
        else:
            source_required_bars = required_bars

        if not isinstance(source_required_bars, dict):
            raise TypeError(
                "'required_bars' must be a dictionary of "
                "{timeframe: number_of_required_candles}."
            )

        if not source_required_bars:
            raise ValueError(
                f"required_bars for symbol '{symbol}' cannot be empty."
            )
        
        temp = normalize_dict({symbol: source_required_bars})
        source_required_bars = temp[symbol]

        # -----------------------------------------------------------
        # Validate the already-normalized required_bars.
        # No timeframe normalization is performed here.
        # -----------------------------------------------------------
        for tf, bars in source_required_bars.items():

            if not isinstance(tf, str) or not tf.strip():
                raise ValueError(
                    "Every timeframe in 'required_bars' must be a "
                    "non-empty string."
                )

            if isinstance(bars, bool) or not isinstance(bars, int):
                raise TypeError(
                    f"Required bar count for timeframe '{tf}' must be an int."
                )

            if bars <= 0:
                raise ValueError(
                    f"Required bar count for timeframe '{tf}' must be > 0."
                )

        # -----------------------------------------------------------
        # Determine canonical timeframe order and base timeframe.
        # check_tfs() guarantees that the smallest timeframe becomes
        # the base timeframe.
        # -----------------------------------------------------------
        first_tf = next(iter(source_required_bars))
        remaining_tfs = [
            tf for tf in source_required_bars
            if tf != first_tf
        ]

        base_tf, other_tfs, all_tfs = check_tfs(
            base_tf=first_tf,
            tfs=remaining_tfs,
        )

        self._base_tf = base_tf
        self.timeframes = list(all_tfs)

        # Build the warmup/required-bars dictionary using exactly the
        # same canonical timeframe names and order.
        #
        # Keep the existing DataHandler internal attribute name and
        # populate it from the final required_bars values.

        self._warmup_dict: Dict[str, int] = {   # ==> CAUTION: محتوای این پارامتر عبارت است از __all_required_bars
            tf: source_required_bars[tf]
            for tf in self.timeframes
        }

        # print(f" === 5 === self.timeframes ===> {self.timeframes}")   # for debug

        # -5-- broker_timezone --------------------------------------
        project_cfg = self.cfg.get("project")

        if not project_cfg:
            raise ValueError("'project' key not found in config !")

        self.broker_timezone = project_cfg.get("broker_timezone")

        if not self.broker_timezone:
            raise ValueError(
                "'broker_timezone' key not found in 'project' key !"
            )

        # print(f" === 7 === self.broker_timezone ===> {self.broker_timezone}")   # for debug

        # -6-- Connection to MarketDataEngine -----------------------
        # self.event_bus = event_bus  # Deleted at 1405/05/21-14:42
        self._subscription_key: Optional[str] = None
        self._running = False
        self._data_callback = None

        # -7-- Cache for live mode ----------------------------------
        # یک دیکشنری برای یک نماد.
        # شامل دیتافریمهای مختلف متناظر با تایمفریمهای متفاوت
        self._cache_dict: dict[str, pd.DataFrame] = {}
        self._cache_dict_extra_length: int = 0

        self._latest_dataset: Optional[MTFDataset] = None

    #/////////////////////////////////
    #/                               /
    #/       For BATCH               /
    #/                               /
    #/////////////////////////////////
    # -------------------------------------------------------------------------
    # BATCH-1- بارگذاری یک تایم‌فریم خام برای حالت batch
    # -------------------------------------------------------------------------

    def _load_raw(self, params: BuildParams) -> pd.DataFrame:
        symbol: str = params.symbol
        timeframe: str = params.selected_tf
        fmt: Literal["parquet", "csv"] = params.load_format
        mode: Literal["number", "time"] = params.mode
        start_lastrows: int | None = params.start_lastrows
        end_lastrows: int | None = params.end_lastrows
        start_time: datetime | None = params.start_time
        end_time: datetime | None = params.end_time

        # logger.debug("==== start _load_raw =================")  # for debug
        # logger.debug(f"symbol = {symbol}")                      # for debug
        # logger.debug(f"timeframe = {timeframe}")                # for debug
        # logger.debug(f"fmt = {fmt}")                            # for debug
        # logger.debug(f"mode = {mode}")                          # for debug
        # logger.debug(f"start_lastrows = {start_lastrows}")      # for debug
        # logger.debug(f"end_lastrows = {end_lastrows}")          # for debug
        # logger.debug(f"start_time = {start_time}")              # for debug
        # logger.debug(f"end_time = {end_time}")                  # for debug

        """
        *** تمام تایم فریمهای دریافتی از کلاس BuildParams استاندارد پروژه را دارا میباشند
            و نیازی به کنترل مجدد ندارند
        1) مقادیر مورد نیاز را از متغیرهای شیء params که از کلاس BuildParams است استخراج میکند.
        2) مسیر کامل دسترسی به فایل را بوسیله نماد، تایمفریم و فرمت میسازد.
        3) با استفاده از فراخوانی تابع _read_raw_df دیتافریم مورد نظر را بدست می آورد.
        """
        path = full_file_path(self.raw_dir, symbol, timeframe, fmt=fmt)
        # logger.debug(f"path = {path}")                 # for debug
        df = _read_raw_df(path=path, mode=mode,
                          start_lastrows=start_lastrows, end_lastrows=end_lastrows,
                          start_time=start_time, end_time=end_time,
                          broker_timezone=self.broker_timezone)
        
        if df.empty:
            logger.warning("Raw data %s/%s not found or empty: %s", symbol, timeframe, path)
            return df

        # logger.debug(f"Len(df) = {len(df)}")                    # for debug   
        # logger.debug("====  end  _load_raw =================")  # for debug
        return df
    
    # -------------------------------------------------------------------------
    # BATCH-2- بررسی فایلهای متادیتای موجود برای تعیین کفایت/عدم کفایت داده های موجود
    # -------------------------------------------------------------------------

    def _inspect_raw_metadata(self, params: BuildParams) -> Dict[str, Dict[str, Any]]:
        """
        بررسی سریع metadata تایم‌فریم‌های موردنیاز، بدون خواندن CSV/Parquet.
        این تابع از فایلهای متادیتا، فقط اطلاعات را میخواند.
        """

        result = {}
        for tf in params.timeframes:

            meta_path = self.raw_dir / params.symbol / f"{tf}.meta.json"
            info = {
                "status": "missing",
                "rows": 0,
                "first_index": None,
                "last_index": None,
            }
            if not meta_path.exists():
                result[tf] = info
                continue

            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))

                info["rows"] = int(metadata.get("rows", 0))
                info["first_index"] = pd.to_datetime(metadata.get("first_index"), utc=True)
                info["last_index"] = pd.to_datetime(metadata.get("last_index"), utc=True)

                if params.mode == "number":
                    required_rows = (
                        params.start_lastrows
                        if params.start_lastrows is not None
                        else 0
                    )
                    info["status"] = "ready" if (info["rows"] >= required_rows) else "insufficient"

                elif params.mode == "time":
                    start_ok = (
                        params.start_time is None
                        or info["first_index"]
                        <= pd.to_datetime(params.start_time, utc=True)
                    )
                    end_ok = (
                        params.end_time is None
                        or info["last_index"]
                        >= pd.to_datetime(params.end_time, utc=True)
                    )
                    info["status"] = "ready" if (start_ok and end_ok) else "insufficient"

            except Exception as exc:
                logger.warning("Invalid metadata for %s/%s: %s", params.symbol, tf, exc)
                info["status"] = "invalid"
            result[tf] = info

        return result

    # -------------------------------------------------------------------------
    # BATCH-3- تعیین کسری داده های موجود و اقدام برای دانلود آنها نوسط mt5_data_loader
    # -------------------------------------------------------------------------

    def download_required_data(
        self,
        metadata_info: Dict[str, Dict[str, Any]],
        params: BuildParams,
    ) -> List[Dict[str, Any]]:
        """
        شناسایی داده‌های خام ناقص و در صورت فعال بودن auto_download، دانلود آنها با پنجره موردنیاز.
        """

        download_jobs: List[Dict[str, Any]] = []

        auto_download = bool(
            (self.cfg.get("download_defaults") or {}).get(
                "auto_download", False
            )
        )

        for tf, info in metadata_info.items():
            if info["status"] == "ready":
                continue

            logger.warning(
                "Raw data is not ready for %s/%s: status=%s",
                self.symbol,
                tf,
                info["status"],
            )

            job = {
                "timeframe": tf,
                "required_rows": None,
                "start_time": None,
                "end_time": None,
            }

            if params.mode == "number":
                job["required_rows"] = (
                    params.start_lastrows
                    if params.start_lastrows is not None
                    else 0
                )

            elif params.mode == "time":
                job["start_time"] = params.start_time
                job["end_time"] = params.end_time

            else:
                raise ValueError(
                    f"Unsupported download mode: {params.mode}"
                )

            download_jobs.append(job)

        # هیچ دانلودی لازم نیست
        if not download_jobs:
            return []

        # دانلود خودکار خاموش است
        if not auto_download:
            logger.warning(
                "auto_download=False; required raw data was not downloaded."
            )
            return download_jobs

        # -------------------------------------------------
        # MT5 Data Loader
        # -------------------------------------------------

        downloader = MT5DataLoader_batch(cfg=self.cfg)

        timeframes = [
            job["timeframe"]
            for job in download_jobs
        ]

        # -------------------------------------------------
        # ساخت DownloadPlan
        # -------------------------------------------------
        if params.mode == "number":
            plans = []

            for job in download_jobs:
                tf = job["timeframe"]
                required_rows = job["required_rows"]

                tf_plans = downloader.build_plan(
                    symbols=[params.symbol],
                    timeframes=[tf],
                    lookback_bars=required_rows,
                    date_from=None,
                    date_to=None,
                    range_policy="count",
                )

                plans.extend(tf_plans)

        else:  # time

            plans = downloader.build_plan(
                symbols=[self.symbol],
                timeframes=timeframes,
                lookback_bars=None,
                date_from=params.start_time,
                date_to=params.end_time,
                date_tz=self.broker_timezone,
                result_tz="UTC",
                range_policy="date",
            )

        # -------------------------------------------------
        # اجرای دانلود
        # -------------------------------------------------
        logger.info(
            "Downloading required raw data: %s / %s",
            self.symbol,
            timeframes,
        )

        results = downloader.run_plan(plans)

        for result in results:
            if "error" in result:
                logger.error(
                    "Download failed for %s/%s: %s",
                    result.get("symbol"),
                    result.get("timeframe"),
                    result["error"],
                )
            else:
                logger.info(
                    "Downloaded %s/%s: rows=%s",
                    result.get("symbol"),
                    result.get("timeframe"),
                    result.get("rows_written"),
                )

        return download_jobs

    # -------------------------------------------------------------------------
    # BATCH-4- ساخت دیتاست برای حالت batch
    # -------------------------------------------------------------------------

    def build(self, params: BuildParams) -> MTFDataset:
        """
        ساخت MTFDataset برای حالت batch.
        جریان اجرا:
            1) تعیین Timeframeهای موردنیاز
            2) بررسی metadata بدون خواندن CSV/Parquet
            3) دانلود خودکار داده‌های ناقص در صورت فعال بودن auto_download
            4) خواندن واقعی داده‌ها
            5) ساخت MTFDataset
            
        جریان داده:
            BuildParams
                ↓
            check_tfs()
                ↓
            all_tfs
                ↓
            _inspect_raw_metadata()
                ↓
            کدام TF آماده نیست؟
                ↓
            download_required_data()
                ↓
            auto_download=False
                → دانلود نمی‌کند
                → هنگام load اگر داده نباشد → FileNotFoundError

            auto_download=True
                → MT5DataLoader_batch
                → build_plan()
                → run()
                ↓
            _load_raw() برای هر TF
                ↓
            MTFDataset
        """
        symbol = params.symbol
        _base_tf = params.base_tf

        _tfs: List[str] = (
            list(params.timeframes)
            if params.timeframes
            else self.timeframes
        )

        # -------------------------------------------------
        # 1) تعیین Timeframeها
        # -------------------------------------------------
        base_tf, other_tfs, all_tfs = check_tfs(_base_tf, _tfs)
        params.timeframes = all_tfs

        # -------------------------------------------------
        # 2) بررسی سریع metadata برای تمام Timeframeهای موردنیاز
        # -------------------------------------------------
        metadata_info = self._inspect_raw_metadata(params=params)

        # -------------------------------------------------
        # 3) شناسایی کسری داده و در صورت نیاز دانلود خودکار
        # -------------------------------------------------
        self.download_required_data(
            metadata_info=metadata_info,
            params=params,
        )

        # -------------------------------------------------
        # 4) ساخت MTFDataset
        # -------------------------------------------------
        dataset = MTFDataset(
            symbol=symbol,
            base_tf=base_tf,
        )

        # -------------------------------------------------
        # 5) خواندن base timeframe
        # -------------------------------------------------
        params.selected_tf = base_tf

        base_raw = self._load_raw(params)

        if base_raw.empty:
            raise FileNotFoundError(
                f"Raw data for {symbol}/{base_tf} is not available "
                f"or there is no overlap between existing and requested "
                f"time windows."
            )

        dataset.add(
            base_tf,
            base_raw,
        )

        # -------------------------------------------------
        # 6) خواندن سایر Timeframeهای موردنیاز
        # -------------------------------------------------
        for tf in other_tfs:
            params.selected_tf = tf

            odf = self._load_raw(params)

            if odf.empty:
                raise FileNotFoundError(
                    f"Raw data for required timeframe "
                    f"{symbol}/{tf} is not available "
                    f"or there is no overlap between existing "
                    f"and requested time windows."
                )

            dataset.add(
                tf,
                odf,
            )

        # -------------------------------------------------
        # 7) ذخیره آخرین Dataset
        # -------------------------------------------------
        self._latest_dataset = dataset

        return dataset

    # -------------------------------------------------------------------------
    # BATCH-5- ذخیره‌سازی دیتاست batch
    # -------------------------------------------------------------------------

    def save(self, dataset: MTFDataset, fmt: Optional[str] = None):
        
        symbol = dataset.symbol
        base_tf = _TF_MAP.get(dataset.base_tf)
        fmt = (fmt or self.save_format or "parquet").lower()

        df_metadata = {}
        for tf in list(dataset.frames.keys()):

            df = dataset.frames[tf]
            out = full_file_path(self.proc_dir, symbol, tf, fmt)  # برای ذخیره فایل پروسس شده 
            meta = {
                "start_time": df.index[0],
                "end_time": df.index[-1],
                "rows": int(len(df)),
            }
            # --------------------------------------------------
            # Save to file 
            # --------------------------------------------------
            if fmt == "parquet":
                try:
                    df.to_parquet(out)
                except Exception as ex:
                    logger.warning("Parquet save failed (%s). Reverting to CSV.", ex)
                    out = out.with_suffix(".csv")
                    df.to_csv(out)
                    fmt = "csv"
            else:     # if fmt == csv
                df.to_csv(out)
                fmt = "csv"
            # ----------------------------------- 
            df_metadata[tf] = meta


        # ------------------------------------------------------
        # Manifest برای تکرارپذیری
        # ------------------------------------------------------
        manifest = {
            "symbol": symbol,
            "base_timeframe": base_tf,
            "format": fmt.lower(),
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "config_version": (self.cfg.get("version") or "unknown"),
            "timeframes_used": list(dataset.frames.keys()),
            # -----------------------------------
            # "other_metadata": df_metadata,
            "other_metadata": {
                tf: {
                    **meta,
                    "start_time": pd.to_datetime(
                        meta["start_time"], unit="ms", utc=True
                    ).replace(microsecond=0).isoformat(),
                    "end_time": pd.to_datetime(
                        meta["end_time"], unit="ms", utc=True
                    ).replace(microsecond=0).isoformat(),
                }
                for tf, meta in df_metadata.items()
            },
            # -----------------------------------
            "columns": list(df.columns),
        }
        out = full_file_path(self.proc_dir, symbol, symbol, fmt)
        manifest_path = out.with_suffix(".manifest.json")
        manifest_path.write_text(
            pd.Series(manifest).to_json(force_ascii=False, indent=2),
            encoding="utf-8"
        )

        # ------------------------------------------------------
        # Output Log
        # ------------------------------------------------------
        logger.info(
            "Processed data saved: %s (rows=%d, cols=%d)",
            Path(self.proc_dir / symbol), len(df), len(df.columns)
        )
        return out
    

    #/////////////////////////////////
    #/                               /
    #/       For LIVE                /
    #/                               /
    #/////////////////////////////////
    # ---------------------------------------------------------------
    # LIVE-1- اتصال EventBus به کلاس DataHandler
    # ---------------------------------------------------------------

    def subscribe_to_event_bus(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._subscription_key = event_bus.subscribe(self.symbol)
        logger.info("DataHandler connected to EventBus with id=%s", self._subscription_key)

    # ---------------------------------------------------------------
    # LIVE-2- شروع مصرف خودکار - NOT USED IN ANY PLACE
    # ---------------------------------------------------------------

    def start_consuming(self) -> None:
        logger.debug("consuming start.")

        if not self.event_bus or not self._subscription_key: # اگر جدول اشتراکات وجود ندارد یا کلیدی موجود نیست،
            raise RuntimeError("EventBus not attached...")   # خطا بده.
        
        self._running = True              # مود را "در حالت اجرا" قرار بده.
        try:
            while self._running:                                       # تا زمانیکه مود "در حال اجرا" فعال است،
                event = self.event_bus.get_event(self._subscription_key, timeout=1.0)  # رویداد را از EVENTBUS بگیر.
                if event and event.get("event_type") == "NEW_CANDLE":  # اگر رویداد وجود داشت و از نوع "کندل جدید" بود،
                    self.update_live(event)  # run update_live(): ابتدا کش را آپدیت میکند و فراخوانی تابع کال بک مصرف کنند
        finally:
            if self._subscription_key:                              # اگر اشتراک در EVENTBUS وجود دارد
                self.event_bus.unsubscribe(self._subscription_key)  # اشتراک را لغو کن
                self._subscription_key = None                       # کلید اشتراک را از بین ببر
    
    # ---------------------------------------------------------------
    # LIVE-3- Live Update
    # ---------------------------------------------------------------

    def update_live(self, event: Dict[str, Any]) -> MTFDataset:
        """
        به‌روزرسانی کش برای یک کندل جدید و بازگرداندن یک دیکشنری از نوع MTFDataset
        پارامتر ورودی:
            event = {
                "event_type": event_type,
                "symbol": symbol,
                "timeframe": timeframe,
                "all_dfs": all_dfs
            }
        خروجی:
            MTFDataset
        """

        logger.debug("update_live start.")

        symbol = event["symbol"]
        timeframe = event["timeframe"]
        all_dfs = event["all_dfs"]
        
        if not self._warmup_dict:
            logger.warning("Required timeframes not set. Load config by config_completer, first.")
            return MTFDataset(symbol=symbol, base_tf=self._base_tf, frames={})
        
        # اگر تایم‌فریم دریافتی در لیست مورد نیاز نیست، نادیده بگیر
        if timeframe not in self._warmup_dict.keys():
            logger.debug(f"Ignoring {timeframe} (not required)")
            return MTFDataset(symbol=symbol, base_tf=self._base_tf, frames={})
        
        # به‌روزرسانی کش
        # dataset = self._update_cache(symbol, timeframe, all_dfs)
        dataset = self._update_cache(event)
        
        # ----- روش اول برای اجرای تابع کال بک ----------------------
        self._notify_new_data(dataset)

        # ----- روش دوم برای اجرای کال بک ---------------------------
        # if self._data_callback is not None:
        #     self._data_callback(dataset)

        
        return dataset       

    # ---------------------------------------------------------------
    # LIVE-4- Helpers for Live Update
    # ---------------------------------------------------------------

    def _update_cache(self, event: Dict[str, Any]) -> MTFDataset:
        """
        این تابع دیکشنری _cache_dict را بروزرسانی میکند
        ساختار دیکشنری _cache_dict مشابه با ساختار دیکشنری all_fds است:
        _cache_dict = {
            "XAUUSD:M1" : DataFrame of "XAUUSD", at "M1" , contains (self.warmups_dicts["XAUUSD"]["M1" ]) closed candles
            "XAUUSD:M15": DataFrame of "XAUUSD", at "M15", contains (self.warmups_dicts["XAUUSD"]["M15"]) closed candles
            "XAUUSD:H1" : DataFrame of "XAUUSD", at "H1" , contains (self.warmups_dicts["XAUUSD"]["H1" ]) closed candles
            "XAUUSD:H4" : DataFrame of "XAUUSD", at "H4" , contains (self.warmups_dicts["XAUUSD"]["H4" ]) closed candles
        }
        """

        symbol = event["symbol"]
        # timeframe = event["timeframe"]
        all_dfs = event["all_dfs"]

        extra = self._cache_dict_extra_length

        for tf in self._warmup_dict.keys():
            key = f"{symbol}:{tf.upper()}"
            if key not in self._cache_dict:
                self._cache_dict[key] = all_dfs[key].copy()
            else:
                combined = pd.concat([self._cache_dict[key], all_dfs[key]], axis=0)
                combined = combined[~combined.index.duplicated(keep='last')]
                
                warmup = self._warmup_dict.get(tf) + extra  # تعداد کندلهای مورد نیاز برای وارم-آپ
                if warmup and len(combined) > warmup:
                    combined = combined.iloc[-warmup:]

                self._cache_dict[key] = combined                   #     <<=== ***** آپدیت شدن _cache_dict *****

            logger.info(f"_cache_dict for {symbol}/{tf} was updated. rows = {len(self._cache_dict[key])}")
        logger.info("_cache_dict was updated")

        # --------- new added
        dataset = MTFDataset(symbol=symbol, base_tf=self._base_tf)
        for tf in self._warmup_dict:
            key = f"{symbol}:{tf.upper()}"
            if key in self._cache_dict:
                dataset.add(tf, self._cache_dict[key])
       
        self._latest_dataset = dataset
        # dataset._print("mtfdataset_debug.md", n_rows=5)        # ///////////////////////////// برای تست است فقط
        return dataset
           
    # ---------------------------------------------------------------
    # LIVE-5- گرفتن آخرین/جدیدترین دیتاست این کلاس
    # ---------------------------------------------------------------

    def get_latest_dataset(self) -> Optional[MTFDataset]:
        return self._latest_dataset

    # ---------------------------------------------------------------
    # LIVE-6- ثبت تابع کال بک مصرف کننده در این کلاس
    # ---------------------------------------------------------------

    def set_data_callback(self, callback):
        """ثبت تابع callback برای دریافت دیتافریم جدید"""
        self._data_callback = callback

    # ---------------------------------------------------------------
    # LIVE-7- فراخوانی تابع کال بک مصرف کننده از این کلاس
    # ---------------------------------------------------------------

    def _notify_new_data(self, dataset: MTFDataset):
        """هر جا که دیتافریم جدید ساخته شد (مثلاً در متد update یا در consumer)، این متد را صدا بزنید"""
        if self._data_callback is not None:
            self._data_callback(dataset)

    # ---------------------------------------------------------------
    # LIVE-8- متد جدید: توقف مصرف NOT USED IN ANY PLACE
    # ---------------------------------------------------------------

    def stop_consuming(self) -> None:
        self._running = False
        if self.event_bus and self._subscription_key:
            self.event_bus.unsubscribe(self._subscription_key)


# =======================================================================================
# CLI
# =======================================================================================
# -------------------------------------------------------------------
def _setup_logging(level: str = "DEBUG") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(filename)-28s | %(lineno)-4d : %(funcName)-24s | %(message)s",
        datefmt="%H:%M:%S",
    )

# -------------------------------------------------------------------
def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Create a processed multi-timeframe dataset from raw MT5 data.")
    
    p.add_argument("-c", "--config", type=str, default=str(project_root() / "f01_config" / "config.yaml"),
                                                                help="Path to the config file (default: f01_config/config.yaml)")
    p.add_argument("--symbol", type=str, required=True,         help="Symbol (example: XAUUSD)")
    p.add_argument("--base-tf", type=str, default=None,         help="Base timeframe")
    p.add_argument("--timeframes", nargs="*", default=None,     help="Timeframes to use. If not provided, config.download_defaults.timeframes will be used.")
    p.add_argument("--load_format", type=str, default=None, choices=["csv", "parquet"], help="Raw Data format (default: from config)")
    p.add_argument("--save_format", type=str, default=None, choices=["csv", "parquet"], help="Output format (default: from config)")
    p.add_argument("--log-level", type=str, default="info", help="Log level: DEBUG/INFO/WARN/ERROR")
    return p.parse_args()

# -------------------------------------------------------------------
def main_old1() -> int:
    from f02_utils.config_loader import load_config
    from f02_utils.config_completer import config_completer

    # --- 1 --- استخراج مقادیر از خط فرمان 
    args = _parse_args()

    # --- 2 --- ساخت لاگر و تعیین سطح آن، همراه با تعیین فرمت و فرمت زمان 
    _setup_logging(args.log_level)

    # --- 3 --- بارگذاری کانفیگ با ENV Override
    cfg = config_completer(args.config, enable_env_override=True)

    # --- 4 --- ساخت هندلر
    handler = DataHandler(cfg=cfg, symbol=args.symbol)

    # --- 5.1 --- ساخت دیتاست
    base_tf = (args.base_tf or handler._base_tf or "M1").upper()
    save_format = (args.save_format or handler.save_format or "parquet").lower()

    # --- 5.2 پارامترهای ساخت
    params = BuildParams(
        symbol=args.symbol,                # اجباری است
        base_tf=base_tf,                   # base_tf,
        timeframes= [" m   10"], #["m20", "1  0  M  "],   # args.timeframes,
        selected_tf=None,
        load_format=(args.load_format or "csv").lower(),    # فرمت داده های خام که باید خوانده شوند.

        mode = "number",  # "number", "time", "periods"
        # --- مربوط به مد number:
        start_lastrows=5,
        end_lastrows=1,
        # --- مربوط به مد time:
        start_time=pd.to_datetime("2026-05-31 00:00:00+00:00", utc=True),
        end_time=pd.to_datetime("2026-06-05 00:00:00+00:00", utc=True),
        # --- مربوط به مد periods:
        period_size = "20m",
        from_last_n = 8,
        to_last_n = 4,
        base_time = pd.to_datetime("2026-06-02 20:00:00").tz_localize(handler.broker_timezone),
    )
    # params.print_params()

    # --- 5.4 ساخت دیتاست
    _dataset = handler.build(params)

    # --- 6 --- ذخیره سازی دیتاست
    out = handler.save(dataset=_dataset, fmt=save_format)

    logger.info("Done. Output: %s", out)
    return 0

# -------------------------------------
def main() -> int:
    from f02_utils.config_completer import config_completer

    # ================================================================
    # 1. دریافت آرگومان‌های CLI
    # ================================================================
    args = _parse_args()

    # ================================================================
    # 2. راه‌اندازی logging
    # ================================================================
    # _setup_logging(args.log_level)

    _setup_logging_funcname(
        "info",
        allowed_functions=[
            "build_plan",
            "normalize_df",
            "main",
            "run_plan",
            "_fetch_candles",
            "get_candles_range",
            "_normalize_date"
        ]
    )
    # ================================================================
    # 3. بارگذاری و تکمیل config
    # ================================================================
    cfg = config_completer(args.config, enable_env_override=True)

    # ================================================================
    # 4. symbol
    # ================================================================
    symbol = args.symbol.upper() # بطور اجباری با ید نماد مشخص بشود. چون در کانفیگ میتواند همزمان چند نماد وجود داشته باشد.
                                 # حال آنکه یک کلاس دیتا هندلر بر اساس یک نماد ساخته میشود.

    base_tf = (
        args.base_tf if args.base_tf is not None
        else cfg["__base_tfs_dict"][symbol]
    )
    # print(f"====== base TF is {base_tf} ======")
    timeframes = (
        args.timeframes if args.timeframes is not None
        else cfg["__timeframes_dict"][symbol]
    )
    # ================================================================
    # 6. ساخت DataHandler
    # ================================================================
    handler = DataHandler(cfg=cfg, symbol=symbol)

    # ================================================================
    # 7. تنظیمات نهایی BuildParams
    # ================================================================
    # base_tf = handler._base_tf
    # timeframes = list(handler.timeframes)

    # ================================================================
    # 8. فرمت خواندن و ذخیره
    # ================================================================
    load_format = (
        args.load_format
        or handler.save_format
        or "parquet"
    ).lower()

    save_format = (
        args.save_format
        or handler.save_format
        or "parquet"
    ).lower()

    # ================================================================
    # 9. BuildParams
    #
    # چون CLI فعلی فقط برای انتخاب symbol / TF / format است،
    # mode پیش‌فرض number استفاده می‌شود.
    # ================================================================
    params = BuildParams(
        symbol=args.symbol,         # اجباری است
        base_tf=base_tf,            # base_tf,
        timeframes=timeframes,
        selected_tf=None,
        load_format=load_format,    # فرمت داده های خام که باید خوانده شوند.

        mode = "periods",           # "number", "time", "periods"
        # --- مربوط به مد number:
        start_lastrows=500,
        end_lastrows=200,
        # --- مربوط به مد time:
        start_time=pd.to_datetime("2026-08-01 00:00:00+00:00", utc=True),
        end_time=pd.to_datetime("2026-08-10 00:00:00+00:00", utc=True),
        # --- مربوط به مد periods:
        period_size = "20m",
        from_last_n = 8,
        to_last_n = 4,
        base_time = "now",
        # base_time = pd.to_datetime("2026-06-02 20:00:00").tz_localize(handler.broker_timezone),
    )
    params.print_params()   # =====>>>>>   تا اینجا درست است  <<<<<======

    # ================================================================
    # 10. ساخت MTFDataset
    # ================================================================
    dataset = handler.build(params)
    print(" //////     every thing is ok upto here ////////////////////////")
    print(dataset.symbol)
    print(dataset.base_tf)
    print(dataset.frames.keys())
    for tf in dataset.frames.keys():
        print("==========================")
        print(f" timeframe = {tf}")
        print(dataset.frames[tf].head(5))
        print(dataset.frames[tf].tail(5))
        print("==========================")

    # ================================================================
    # 11. ذخیره
    # ================================================================
    print(symbol)
    print(base_tf)
    print(save_format)    
    
    out = handler.save(
        dataset=dataset,
        # symbol=symbol,
        # base_tf=params.base_tf,
        fmt=save_format,
    )

    logger.info("Done. Output: %s", out)

    return 0    

# ------------------------------------------------------------------- OK
if __name__ == "__main__":
    raise SystemExit(main_old1())

"""
روش اجرا:

python -m f03_data.data_handler_F_3 `
    --symbol BITCOIN `
    --save_format csv `
"""
