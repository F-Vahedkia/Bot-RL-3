from typing import Any, Dict, List

from f01_utils.functions.parser import parse_spec
from f01_utils.functions.constants import _TF_MINUTES

# ------------------------------------------------------------------- OK= new 050315
def extract_indicator_timeframes(cfg: Dict[str, Any]) -> List[str]:
    """ استخراج تایم‌فریم‌های یکتا از لیست اندیکاتورهای تعریف‌شده در کانفیگ. """

    indicators = (((cfg.get("features") or {}).get("indicators")) or [])
    tfs = {
        ps.timeframe
        for spec in indicators
        if (ps := parse_spec(spec)).timeframe
    }
    result = sorted(tfs, key=lambda tf: _TF_MINUTES.get(tf.upper(), 999999))
    return result







# =================================================================================================
# 2) twin functions (for use in data_handler.py, ...)
# دانلود و نرمال‌سازی دیتافریم‌های خام 
# =================================================================================================
from typing import List, Optional
from pathlib import Path
import pyarrow.parquet as pq
import logging
from f03_data.mt5_data_loader_E import normalize_df
import pandas as pd

# -------------------- Logger for this module -----------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ------------------------------------------------------------------- OK= NOT USED
def _read_last_n_rows_parquet_A(
    file_path: Path,
    last_n_rows: int,
    columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
    
    # -- 1 -- Checking file_path ------------------------------------
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # -- 2 -- Checking last_n_rows, columns -------------------------
    parquet_file = pq.ParquetFile(file_path)                  # تعریف یک پارکت فایل
    if last_n_rows is None and columns is None:
        return parquet_file.read().to_pandas()                # خواندن یک پارکت فایل و تبدیل آن به دیتافریم پانداس
    if last_n_rows is None and columns is not None:
        return parquet_file.read(columns=columns).to_pandas() # خواندن یک پارکت فایل، استخراج ستونهای مشخصی از آن و تبدیل آن به دیتافریم پانداس
    
    # -- 3 -- Checking Last_n_rows ----------------------------------
    if last_n_rows < 0:
        raise ValueError("last_n_rows cannot be negative")
    if last_n_rows == 0:
        return pd.DataFrame()
        
    # -- 4 -- Checking num_rows of destination file -----------------
    total_rows = parquet_file.metadata.num_rows               # استخراج تعداد سطرهای یک پارکت فایل از متادیتای آن
    if total_rows == 0:
        return pd.DataFrame()

    # -- 5 -- Comparing total_rows, last_n_rows ---------------------
    if total_rows <= last_n_rows:
        return parquet_file.read(columns=columns).to_pandas()

    # -- 6 -- یافتن اندکس (گروه-سطر)ی که اولین سطر مورد نیاز در آن قرار میگیرد
    start_row = total_rows - last_n_rows                      # شماره سطر شروع/مورد نیاز
    rows_so_far = 0
    first_rg_idx = None
    for i in range(parquet_file.num_row_groups):              # استخراج تعداد (گروه-سطر)های یک پارکت فایل
        rg_rows = parquet_file.metadata.row_group(i).num_rows # استخراج تعداد سطرهای موجود در یک (گروه-سطر) با اندکس مشخص، از متادیتای پارکت فایل
        if rows_so_far + rg_rows > start_row:
            first_rg_idx = i                                  # اندکس (گروه-سطری) که (شماره سطر شروع) در آن گروه قرار گرفته است
            break
        rows_so_far += rg_rows

    # -- 7 -- خواندن تمام (گروه-سطر)هایی که تمام/بخشی از داده آنها را نیاز داریم
    row_groups_to_read = list(range(first_rg_idx, parquet_file.num_row_groups)) # لیستی شامل (اندکس گروه یافت شده) تا (تعداد کل گروه-سطر)ها
    table = parquet_file.read_row_groups(row_groups_to_read, columns=columns)   # خواندن گروه-سطرها از پارکت فایل، مطابق با لیست اندکسها در سطر قبل/بالا
    df = table.to_pandas()                                                      # تبدیل داده های خوانده شده به دیتافریم پانداس

    # # -- 8 -- برش نهایی
    # offset_in_first_rg = start_row - rows_so_far
    # if offset_in_first_rg > 0:
    #     df = df.iloc[offset_in_first_rg:]

    # -- 8 -- برش نهایی
    if len(df) > last_n_rows:
        df = df.iloc[-last_n_rows:]
    return df

# ------------------------------------------------------------------- OK= NOT USED
def _read_raw_df_A(path: Path, last_n_rows: int = None) -> pd.DataFrame:
    """
    تلاش برای خواندن parquet و در صورت خطا/نبود، فرمت CSV را امتحان می‌کند.
    دیتا فریم محصول را همانند نرمال سازی موجود در فایل mt5_data_loader.py نرمال میکند
    """
    if path.suffix.lower() == ".parquet" and path.exists():
        try:
            df = _read_last_n_rows_parquet_A(path, last_n_rows)
            return normalize_df(df)
        
        except Exception as ex:
            logger.warning("Failed to read Parquet (%s). Switching to CSV.", ex)

    csv_path = path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
        if last_n_rows is not None:     
            if last_n_rows > 0:
                start_row = max(0, len(df) - last_n_rows)
                return normalize_df(df[start_row:])
            elif last_n_rows == 0:
                return normalize_df(pd.DataFrame)
            else: # elif last_n_rows < 0
                raise ValueError("last_n_rows cannot be negative")
        else: # elif last_n_rows is None:
            return normalize_df(df)
        
    # اگر هیچکدام نبود، دیتافریم خالی
    return normalize_df(pd.DataFrame())

# ------------------------------------------------------------------- OK= NOT USED
# =======================================================================================
# 3) single functions (for use in data_handler.py, ...)
# تابع تبدیل زمان شروع به زمان بسته شدن
# ======================================================================================= OK
def _start_to_close_time(t_start: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    """
    تبدیل زمان شروع کندل به زمان بسته شدن (end of candle) بر اساس تایم‌فریم.
    فرض: t_start timezone-aware (UTC).
    """
    tf_upper = timeframe.upper().replace(" ", "")
    minutes = _TF_MINUTES.get(tf_upper)
    if minutes is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    delta = pd.Timedelta(minutes=minutes)
    return t_start + delta #- pd.Timedelta(seconds=1)   # لحظه قبل از شروع کندل بعدی

# =======================================================================================
