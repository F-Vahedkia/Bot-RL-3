# f04_features/feature_B_cache_6.py
# Reviwed at 1405/05/16

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations
from typing import Any, Dict, Optional
from typing import Callable
from collections import OrderedDict   # مدیریت خود Cache و ترتیب آیتم‌های آن

import hashlib                        # تولید Hash برای ساخت Cache Key
import json
import pandas as pd

from f03_data.mtf_dataset import MTFDataset
from f02_utils.functions.parser import ParsedSpec

# =============================================================================
# HASH
# ============================================================================= 1 خوانده شد و فهمیده شد.
def _stable_hash(obj: Any) -> str:
    """
    این تابع دیکشنری مشخصات اجرای فیچرها را به یک رشته‌ی هش ثابت تبدیل می‌کند
    تا به‌عنوان کلید کش استفاده شود.
    این رشته هش نقش اثر انگشت یکتای دیکشنری مشخصات را ایفا میکند.
    """
    """ آموزشی:
    -----------
    json.dump():
        آیجکت را به یک رشته جیسان تبدیل میکند
    sort_keys=True:
        کلیدهای دیکشنری را مرتب می‌کند. چون ترتیب ورود کلیدها نباید باعث تولید هش متفاوت شود.
    default=str: 
        اگر داخل دیکشنری چیزی باشد که جیسان نمی‌تواند مستقیم تبدیل کند
        توسط تابع str()
        (مثلاً یک آبجکت یا یک تایم استمپ خاص)، آن را به متن تبدیل می‌کند.
    encode():
        رشته (جیسان) را به bytes تبدیل میکند، چون الگوریتم SHA-256 با bytes کار می‌کند.
    hashlib.sha256():
         اثر انگشت باینری payload را تولید میکند.
    hexdigest():
        خروجی باینری هش را به یک رشته قابل خواندن تبدیل میکند.
    """
    payload = json.dumps(
        obj,
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()

# =============================================================================
# EXECUTION CONTRACT
# ============================================================================= 2 خوانده شد و فهمیده شد.
class ExecutionContract:
    """
    قرارداد اجرای Feature Engine را با ذخیره‌سازی نسخه‌ی تمام مؤلفه‌هایی
    که بر محاسبه‌ی فیچرها تأثیر دارند، نمایش می‌دهد.
    این کلاس یک امضای پایدار (هش) از این قرارداد ایجاد می‌کند تا
    زمینه‌ی اجرای فیچرها به‌صورت یکتا شناسایی و اعتبارسنجی شود.
    """
    def __init__(
        self,
        engine_version: str,
        resolver_version: str,
        config_version: str,
        registry_version: str,
        feature_schema_version: str = "v3",
    ):
        self.engine_version = engine_version
        self.resolver_version = resolver_version
        self.config_version = config_version
        self.registry_version = registry_version
        self.feature_schema_version = feature_schema_version

    def signature(self) -> str:
        return _stable_hash(
            {
                "engine": self.engine_version,
                "resolver": self.resolver_version,
                "config": self.config_version,
                "registry": self.registry_version,
                "schema": self.feature_schema_version,
            }
        )

# =============================================================================
# CACHE
# ============================================================================= 3 خوانده شد و فهمیده شد.
class FeatureCache:
    """
    این کلاس یک (حافظه‌ی موقت) برای (ذخیره و بازیابی) (نتایج محاسبه‌ی فیچرها) ایجاد می‌کند.
    با استفاده از کلیدهای یکتا، از محاسبات تکراری جلوگیری کرده
    و مدیریت محدودیت ظرفیت کش را انجام می‌دهد.

    این حافظه موقت بصورت یک OrderedDict است و این کلاس متدهای get, set, clear را روی آن اجرا میکند.
    """
    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self._store: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: Any):
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        if len(self._store) >= self.max_size:
            self._store.popitem(last=False)
        self._store[key] = value

    def clear(self):
        self._store.clear()

# GLOBAL_FEATURE_CACHE = FeatureCache()

# ============================================================================= 4 خوانده شد و فهمیده شد.
class FeatureCacheManager:
    """
    Manages independent feature caches for different symbols.
    این کلاس یک دیکشنری ایجاد میکند که کلیدهای آن نمادها هستند و 
    مقدار متناظر با هر نماد یک آبجکت از کلاس FeatureCache است.
    این کلاس از متدهای get, clear برای مدیریت دیکشنری ایجاد شده استفاده میکند.
    """
    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self._caches: Dict[str, FeatureCache] = {}

    def get(self, symbol: str) -> FeatureCache:
        if symbol not in self._caches:
            self._caches[symbol] = FeatureCache(max_size=self.max_size)
        return self._caches[symbol]

    def clear(self, symbol: Optional[str] = None):
        if symbol is None:
            self._caches.clear()
        elif symbol in self._caches:
            self._caches[symbol].clear()

GLOBAL_FEATURE_CACHE_MANAGER = FeatureCacheManager()

# =============================================================================
# DATASET SIGNATURE
# ============================================================================= 5 خوانده شد و فهمیده شد.
def _frame_signature(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Lightweight deterministic fingerprint of one DataFrame.
    این تابع یک دیکشنری از مشخصات (دیتافریم ورودی) را برمیگرداند.
    """
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "first_index": str(df.index[0]) if len(df) else None,
        "last_index": str(df.index[-1]) if len(df) else None,
        "dtypes": {
            c: str(df[c].dtype)
            for c in df.columns
        },
    }

# ============================================================================= 6 خوانده شد و فهمیده شد.
def _dataset_signature(dataset: MTFDataset) -> Dict[str, Any]:
    """
    Stable signature for whole MTFDataset.
    این تابع یک دیکشنری از مشخصات (دیتاست ورودی) را برمیگرداند.
    frames:
        یک دیکشنری است
        که کلیدهای آن تایم فریم های سورت شده هستند و
        مقدار متناظر با هر کلید/تایمفریم عبارتند از دیکشنری مشخصات دیتافریم مربوط به آن تایم فریم
    """
    frames = {}
    for tf in sorted(dataset.frames.keys()):
        frames[tf] = _frame_signature(dataset.frames[tf])
    return {
        "symbol": dataset.symbol,
        "base_tf": dataset.base_tf,
        "frames": frames,
    }

# =============================================================================
# CACHE KEY
# ============================================================================= 7 خوانده شد و فهمیده شد.
def build_cache_key(
    dataset: MTFDataset,
    spec: str,
    mode: str,
    contract: ExecutionContract,
    *,     # یعنی از این نقطه به بعد، تمام آرگومان‌ها فقط به صورت keyword argument قابل ارسال هستند.
    tf: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    این تابع یک دیکشنری از موارد زیر میسازد:
        - اثرانگشت دیتاست
        - فیچر مربوطه
        - مود
        - اثرانگشت قرارداد محاسبه فیچر
        - تایم فرم فیچر مربوطه
        - موارد اضافی دیگر
    و (رشته-هش-قابل خواندن) مربوط به این دیکشنری را برمیگرداند
    """
    payload = {
        "dataset": _dataset_signature(dataset),
        "spec": spec,
        "mode": mode,
        "contract": contract.signature(),
        "tf": tf,
        "extra": extra or {},
    }
    return _stable_hash(payload)

# =============================================================================
# CACHED EXECUTION
# ============================================================================= 8 خوانده شد و تقریباً فهمیده شد.
def cached_compute(
    *,     # یعنی از این نقطه به بعد، تمام آرگومان‌ها فقط به صورت keyword argument قابل ارسال هستند.
    cache: FeatureCache,     # آبجکتی بود که یک OrderedDict داشت.
    dataset: MTFDataset,
    ps: ParsedSpec,
    mode: str,
    contract: ExecutionContract,
    compute_fn: Callable[[], MTFDataset],
) -> MTFDataset:
    """
    Cache wrapper for one parsed feature specification.
    """
    key = build_cache_key(                  # خروجی این تابع یک "رشته قابل خواندن هش" است
        dataset=dataset,
        spec=ps.canonical,   # canonical=ps.raw.replace("'", '"')
        mode=mode,
        contract=contract,
        tf=ps.timeframe,
        extra={
            "args": ps.args,
            "kwargs": ps.kwargs,
        },
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    result = compute_fn() # در این سطر (دستور میانی) محاسبه مقادیر اندیکاتور اجرا میشود
                          # دراینجاخروجی result ازنوع MTFDataset است.
                          # اگر در انتهای سطر دوتا پرانتز وجود نداشت، کالیبل بود.
                          # اما در اینجا فراخوانی تابع اتفاق افتاده است
    cache.set(key, result)

    return result

# ============================================================================= END