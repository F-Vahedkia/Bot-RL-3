# f03_data/data_layer_functions.py (4)
#
# Last reviewed: 1405/06/25
# =======================================================================================
""" ---> Docstring:
توابع کمکی عمومی لایه f03_data برای استانداردسازی تایم‌فریم‌ها و عملیات‌های وابسته.

تابع فعال اصلی در این ماژول check_tfs() است که مجموعه تایم‌فریم‌های ورودی را با استفاده
از _TF_MAP به نام‌های استاندارد تبدیل می‌کند، تکراری‌ها را حذف می‌کند، بر اساس
_TF_MINUTES به ترتیب صعودی مرتب می‌کند و کوچک‌ترین تایم‌فریم معتبر را به عنوان base_tf
انتخاب می‌کند.

قرارداد check_tfs():
    - ورودی base_tf و فهرست tfs می‌توانند دارای حروف کوچک یا فاصله باشند و پیش از نگاشت
        normalize می‌شوند.
    - اگر tfs خالی باشد، base_tf به‌تنهایی قابل استفاده است و در صورت نامعتبر بودن خطا رخ می‌دهد.
    - تایم‌فریم‌های نامعتبر در ترکیب چندتایم‌فریمی نادیده گرفته می‌شوند؛ اگر هیچ تایم‌فریم
        معتبری باقی نماند، ValueError ایجاد می‌شود.
    - خروجی سه‌گانه به شکل (base_tf, other_tfs, all_tfs) است و همه نام‌ها استاندارد و uppercase هستند.

prefix_columns() نیز یک DataFrame را کپی کرده و به نام همه ستون‌ها یک prefix اضافه می‌کند.

توابعی که با پسوند _old1 یا _old نام‌گذاری شده‌اند مسیر فعال این ماژول محسوب نمی‌شوند.
"""
# =======================================================================================


from __future__ import annotations

from typing import List,  Tuple
from datetime import datetime, date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import pandas as pd
import logging

from f02_utils.functions.constants import _TF_MINUTES, _TF_MAP

# -------------------- Logger for this module -------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =======================================================================================
# 7-- معرفی کوچکترین تایم فریم به عنوان تایم فریم مبنا و مرتب سازی بقیه تایم فریم ها
# =======================================================================================
def check_tfs(base_tf: str, tfs: List[str]) -> Tuple[str, List[str], List[str]]:
    """
    تایم فریمها را بصورت صعودی مرتب میکند.
    کوچکترین تایم فریم را به عنوان تایم فریم پایه انتخاب میکند و بعنوان اولین خروجی میدهد.
    بقیه تایم فریم ها را در قالب دومین خروجی میدهد.
    تمام تایم فریمها را در قالب خروجی سوم میدهد
    *** تمام تایمفریمها با حروف بزرگ و مطابق با _TF_MAP به خروجی فرستاده میشوند.
    """
    # اعتبار سنجی اولیه
    if not tfs:
        if base_tf is None:
            raise ValueError("Both base_tf and tfs are empty. Cannot determine any timeframe.")
        base_tf_upper = base_tf.upper().replace(" ", "")
        if base_tf_upper in _TF_MAP:
            mapped = _TF_MAP[base_tf_upper]
            logger.info("tfs is empty, using base_tf only.")
            return mapped, [], [mapped]
        else:
            raise ValueError(f"base_tf '{base_tf}' is not a valid timeframe.")
    
    # تبدیل به حروف بزرگ
    base_tf_upper = base_tf.upper().replace(" ", "")
    tfs_upper = [tf.upper().replace(" ", "") for tf in tfs]
    original_base = base_tf_upper

    # مجموعه همه تایم‌فریم‌ها
    all_tfs = set(tfs_upper)
    all_tfs.add(base_tf_upper)
    
    # نگاشت به استاندارد
    all_tfs_mapped = []
    invalid_count = 0
    for tf in all_tfs:
        if tf in _TF_MAP:
            all_tfs_mapped.append(_TF_MAP[tf])
        else:
            invalid_count += 1
            logger.warning(f"timeframe '{tf}' is not in _TF_MAP and will be ignored.")
    
    if not all_tfs_mapped:
        raise ValueError("No valid timeframes found after mapping.")
    
    # مرتب‌سازی بر اساس _TF_MINUTES
    temp = sorted(all_tfs_mapped, key=lambda x: _TF_MINUTES.get(x, 999999))
    
    # حذف تکراری‌ها با حفظ ترتیب
    unique = list(dict.fromkeys(temp))
    
    # فیلتر نهایی بر اساس _TF_MINUTES (اختیاری)
    final = [tf for tf in unique if tf in _TF_MINUTES]
    if len(final) != len(unique):
        logger.info(f"Ignored {len(unique)-len(final)} timeframe(s) not in _TF_MINUTES.")
    
    if not final:
        raise ValueError("No valid timeframes after filtering by _TF_MINUTES.")
    
    # انتخاب کوچک‌ترین به عنوان base
    _base_tf = final[0]
    _other_tfs = final[1:]
    
    if original_base != _base_tf:
        logger.info(f"base_tf changed from {original_base} to {_base_tf} (smallest timeframe in set).")
    
    return _base_tf, _other_tfs, final   # fina = all_tfs


# ======================================================================================= OK
def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """به همه‌ی ستون‌ها پیشوند اضافه می‌کند (برای تمایز تایم‌فریم‌ها)."""
    df2 = df.copy()
    df2.columns = [f"{prefix}_{c}" for c in df2.columns]
    return df2

""" فقط بایگانی- مورد استفاده قرار نگیرد
def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # این تابع فقط نام ستون های همان دیتافریم را عوض میکند
    df.columns = [f'{prefix}_{c}' for c in df.columns]
    return df
"""


# ======================================================================================= OK=me

#   تابع زیر را خودم نوشتم و برای ان وقت گذاشتم. تابع دوم را دیپ سیک نوشت.
def check_tfs_old1(base_tf: str, tfs: List[str]) -> Tuple[str, List[str], List[str]]:
    """
    تایم فریمها را بصورت صعودی مرتب میکند.
    کوچکترین تایم فریم را به عنوان تایم فریم پایه انتخاب میکند و بعنوان اولین خروجی میدهد.
    بقیه تایم فریم ها را در قالب دومین خروجی میدهد.
    تمام تایم فریمها را در قالب خروجی سوم میدهد
    """
    # اعتبار سنجی ها
    if len(tfs) == 0:
        if base_tf is None:
            logger.warning("check base_tf, tfs")
            return None, [], []
        elif base_tf in _TF_MAP.keys():
            logger.info("tfs is empty")
            return _TF_MAP[base_tf] , [], []
        else:
            raise ValueError("all timeframes are invalid !!!")
        
    # تبدیل همه به حروف بزرگ برای یکسانی
    base_tf = base_tf.upper()
    tfs_upper = [tf.upper() for tf in tfs]
    
    # ذخیره base_tf اولیه تا انتهای تابع
    original_base = base_tf

    # مجموعه همه تایم‌فریم‌ها به همراه base_tf
    all_tfs = set(tfs_upper)
    all_tfs.add(base_tf)
    
    # نگاشت تمام رشته ها به رشته های استاندارد توسط _TF_MAP
    all_tfs_mapped = []
    for tf in all_tfs:
        if tf in _TF_MAP.keys():
            all_tfs_mapped.append(_TF_MAP[tf])
        else:
            logger.warning(f"timeframe: {tf} not mapped and skiped.")

    # مرتب‌سازی بر اساس مقدار عددی (صعودی)
    temp1 = sorted(all_tfs_mapped, key=lambda x: _TF_MINUTES.get(x, 999999))

    # حذف تکراری ها
    unique_list = list(dict.fromkeys(temp1))
    
    # اعتبار سنجی مطابق با تایم فریمهای موجود در _TF_MINUTES
    _all_tfs_final = [tf for tf in unique_list if tf in  _TF_MINUTES.keys()]

    tfs_number = len(_all_tfs_final)
    if tfs_number == 0:
        logger.info("all timeframes are invalid !!!")
        return None, [], []
    elif tfs_number == 1:
        logger.info("there is only one invalid timeframe.")
        return _all_tfs_final[0], [], [_all_tfs_final[0]]
    
    # تعداد تایم فریمهای نامعتبر که حذف شده اند
    if tfs_number != len(unique_list):
        logger.info(f"some timeframes (n={len(unique_list)-tfs_number}) are invalid and ignored.)")

    # کوچک‌ترین عضو به عنوان base_tf جدید
    _base_tf = _all_tfs_final[0]
    _other_tfs = _all_tfs_final[1:]
    
    # اگر base_tf اصلی با کوچک‌ترین تایم‌فریم یکی نبود، لاگ ثبت کن
    if original_base != _base_tf:
        logger.info(f"base_tf changed from {original_base} to {_base_tf} (smallest timeframe in set).")
    
    return _base_tf, _other_tfs, _all_tfs_final



# ======================================================================================= OK=
def _merge_on_base_old(base_df: pd.DataFrame, other_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    ادغام DataFrame کمکی روی شبکه‌ی base_df با merge_asof (ffill).
    - فرض: ایندکس هر دو UTC و مرتب است.
    """
    if other_df.empty:
        # اگر دیتای کمکی خالی بود، فقط base را برگردان
        return base_df

    # merge_asof روی ستون زمان؛ بنابراین index را به ستون تبدیل می‌کنیم
    b = base_df.copy()
    o = prefix_columns(other_df, prefix)    #.copy()

    b.index.name = "time"         # نام ستون ایندکس را برابر با time قرار میدهد 
    b = b.reset_index()         # ایندکس فعلی دیتافریم b را به یک ستون معمولی تبدیل می‌کند و یک ایندکس عددی جدید 0,1,2,… می‌سازد 
    o.index.name = "time"
    o = o.reset_index()

    merged = pd.merge_asof(
        b.sort_values("time"),
        o.sort_values("time"),
        on="time",
        direction="backward",     #  "backward","forward"
        # allow_exact_matches=True
        )
    
    merged.set_index("time", inplace=True)   # برعکس متد reset_index عمل میکند
    merged.index = pd.to_datetime(merged.index, utc=True) 

# ======================================================================================= END
