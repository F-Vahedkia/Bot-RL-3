من می‌خواهم در این چت فقط مسیرهای **Batch و Live داخل لایه Features** پروژه را بررسی کنیم و برای آن‌ها تست‌های End-to-End بنویسیم.

**لطفاً به فایل‌های لایه Data نیازی نداشته باش و از روی این قرارداد ورودی Features کار کن. اگر چیزی در این قرارداد مشخص نیست، حدس نزن و همان مورد را صریحاً اعلام کن.**

# قرارداد تحویل Data Layer به Features Layer

## 1. معماری کلی

پروژه Multi-Symbol و Multi-Timeframe است.

برای هر Symbol یک `DataHandler` مستقل وجود دارد.

`DataHandler` یک `MTFDataset` تولید می‌کند.

`MTFDataset` شامل DataFrameهای مستقل برای Timeframeهای مختلف یک Symbol است و خودش هیچ alignment انجام نمی‌دهد.

هر `MTFDataset` دارای:

```python
symbol: str
base_tf: str
frames: Dict[str, pd.DataFrame]
```

است.

---

## 2. قرارداد داده ورودی به Features در Batch

در حالت Batch، DataHandler یک `MTFDataset` تولید می‌کند:

```text
DataHandler.build(...)
        ↓
MTFDataset
        ↓
Features Layer
```

هر frame در `dataset.frames` یک DataFrame مستقل برای یک timeframe است.

ساختار DataFrameهای خام:

```text
index:
    pandas.DatetimeIndex

timezone:
    timezone-aware

timezone contract:
    UTC

columns:
    open
    high
    low
    close
    volume
    spread
```

نام ستون `volume` استاندارد است و ممکن است از `real_volume` یا `tick_volume` منبع داده ساخته شده باشد.

---

## 3. قرارداد داده ورودی به Features در Live

در حالت Live مسیر واقعی تا پایان DataHandler قبلاً End-to-End تست شده است:

```text
MT5Connector
    ↓
MT5StreamWorker
    ↓
CandleDetector
    ↓
EventBus
    ↓
DataHandler.update_live()
    ↓
DataHandler._update_cache()
    ↓
MTFDataset / _latest_dataset
    ↓
DataHandler callback
```

بنابراین Features Layer باید Dataset تولیدشده توسط DataHandler را مصرف کند.

در Live نیز DataFrameهای داخل `MTFDataset` دارای:

```text
pandas.DatetimeIndex
timezone-aware
UTC
```

هستند.

---

## 4. قرارداد Timeframe

Timeframeهای پروژه:

```python
_all_tfs = [
    "M1", "M2", "M3", "M4", "M5", "M6",
    "M10", "M12", "M15", "M20", "M30",
    "H1", "H2", "H3", "H4", "H6", "H8", "H12",
    "D1", "W1", "MN1",
]
```

Timeframeهای Intraday:

```python
_intraday_tfs = [
    "M1", "M2", "M3", "M4", "M5", "M6",
    "M10", "M12", "M15", "M20", "M30",
    "H1", "H2", "H3", "H4", "H6", "H8", "H12",
]
```

---

## 5. قرارداد base_tf

هر `MTFDataset` دقیقاً یک `base_tf` دارد.

قرارداد معماری Features این است:

> Time Features هر Symbol فقط روی `base_tf` همان `MTFDataset` اعمال می‌شوند.

در Config برای هر Symbol فقط لیست Time Featureها قرار می‌گیرد و `base_tf` در زیر Symbol تکرار نمی‌شود، زیرا `base_tf` متعلق به Dataset است و Single Source of Truth باید حفظ شود.

---

## 6. Time Features فعلی

در پروژه 25 Time Feature وجود دارد:

```text
minute
hour
day_of_week
day_of_month
day_of_year
week_of_year
month
quarter

session_asia
session_london
session_newyork

is_weekend
is_month_start
is_month_end
is_quarter_start
is_quarter_end

sin_hour
cos_hour
sin_day_of_week
cos_day_of_week
sin_month
cos_month

market_open_flag
market_close_flag
is_new_day
```

Registry برای هر Feature دارای:

```python
TimeFeatureSpec(
    name=...,
    function=...,
    output_columns=...,
    valid_timeframes=...,
    description=...,
)
```

است.

`valid_timeframes` قرارداد معتبر بودن Feature روی timeframe را تعیین می‌کند.

---

## 7. Time Feature Engine

سه فایل Time Feature هسته‌ای فعلاً تکمیل و تست شده‌اند:

```text
f04_features/time_features/time_feature_functions.py
f04_features/time_features/time_feature_registry.py
f04_features/time_features/time_feature_engine.py
```

برای این سه فایل قبلاً:

* تست Batch با داده واقعی انجام شده است.
* تست Live با داده واقعی انجام شده است.
* خروجی Batch و Live برای Time Features با یکدیگر مقایسه شده‌اند.
* هر 25 Feature در تست موفق شده‌اند.
* `valid_timeframes` در Registry برای کنترل اعتبار Feature استفاده می‌شود.
* در Live، `is_new_day` امکان دریافت `previous_timestamp` را دارد.

این سه فایل فعلاً جزء هسته‌های تکمیل‌شده محسوب می‌شوند؛ مگر اینکه در اتصال به مسیر اصلی Features یک نقص واقعی کشف شود.

---

## 8. تست مسیر Data تا Features

مسیر Data Layer تا انتهای DataHandler قبلاً با داده واقعی تست شده و PASS گرفته است.

تستر مورد استفاده:

```text
f03_data/tests_live_market_engine/test_7_live_data_full_path_integrity.py
```

این تستر برای `BITCOIN` و timeframeهای `M1` و `M2` موفق شده است.

بنابراین در این چت نباید دوباره Data Layer را از MT5 تا DataHandler بررسی کنیم، مگر اینکه یک خطای مشخص در مرز Data → Features مشاهده شود.

---

## 9. هدف این چت

هدف فعلی:

### Batch

بررسی دقیق مسیر:

```text
DataHandler.build()
    ↓
MTFDataset
    ↓
Features Layer
    ↓
Feature Engine
    ↓
Indicators / Time Features / سایر Features
    ↓
FeatureStore
    ↓
ObservationBuilder
```

### Live

بررسی دقیق مسیر:

```text
DataHandler callback
    ↓
MTFDataset
    ↓
Features Layer
    ↓
Feature Engine
    ↓
Indicators / Time Features / سایر Features
    ↓
FeatureStore
    ↓
ObservationBuilder
```

و در نهایت باید مشخص شود که Batch و Live در Features Layer:

* چه ورودی‌ای دریافت می‌کنند،
* چه پردازشی انجام می‌دهند،
* stateful یا stateless بودن هر Feature چگونه مدیریت می‌شود،
* چه تفاوت‌هایی میان Batch و Live وجود دارد،
* و آیا خروجی هر دو از نظر ساختار و مقدار با یکدیگر سازگار است.

---

## 10. قانون مهم برای بررسی

در این چت:

**هیچ فرضی درباره API یا ساختار فایل‌های Features نزن.**

اگر برای تعیین مسیر واقعی نیاز به فایل خاصی باشد، فقط همان فایل را درخواست کن.

همچنین بدون مشاهده نسخه واقعی فعلی فایل‌ها:

* تابع جدید اختراع نکن،
* نام متد حدس نزن،
* مسیر داده حدس نزن،
* قرارداد جدید ایجاد نکن.

ابتدا مسیر واقعی Batch و Live در Features Layer را از روی فایل‌های واقعی استخراج کن، سپس تست End-to-End مناسب طراحی کن.
