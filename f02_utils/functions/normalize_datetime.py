# Date reviewed: 1405/05/30 - 08:47


from datetime import datetime, date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


# === OLD ===================================================================== OK=
def _parse_dt_old1(value, input_tz, output_tz=None):
    """
    مقدار تاریخ/زمان را دریافت کرده و آن را به یک datetime دارای منطقهٔ زمانی تبدیل می‌کند.

    پارامترها
    ----------
    value : datetime, date, int, float, str یا None
        مقدار تاریخ/زمان ورودی.

        - اگر None یا رشتهٔ خالی باشد، مقدار None برگردانده می‌شود.
        - اگر datetime بدون منطقهٔ زمانی باشد، input_tz به آن نسبت داده می‌شود.
        - اگر datetime دارای منطقهٔ زمانی باشد، به output_tz تبدیل می‌شود.
        - اگر date باشد، زمان 00:00:00 به آن اضافه شده و در input_tz
          تفسیر می‌شود، سپس به output_tz تبدیل می‌شود.
        - اگر int یا float باشد، به‌عنوان Unix timestamp در نظر گرفته
          شده و مستقیماً به datetime در output_tz تبدیل می‌شود.
        - اگر رشته برابر با "now" باشد، زمان فعلی در output_tz برگردانده می‌شود.
        - سایر رشته‌ها با datetime.fromisoformat() تجزیه می‌شوند.
          اگر نتیجه بدون timezone باشد، input_tz به آن نسبت داده می‌شود؛
          در غیر این صورت مستقیماً به output_tz تبدیل می‌شود.

    input_tz : str یا tzinfo
        منطقهٔ زمانی مورد استفاده برای تفسیر مقادیر فاقد timezone.
        اگر رشته باشد، با ZoneInfo به منطقهٔ زمانی تبدیل می‌شود.

    output_tz : str یا tzinfo، اختیاری
        منطقهٔ زمانی datetime خروجی.
        اگر None باشد، input_tz به‌عنوان output_tz نیز استفاده می‌شود.

    خروجی
    -------
    datetime یا None
        یک datetime دارای timezone در منطقهٔ زمانی output_tz،
        یا None در صورتی که value برابر None یا رشتهٔ خالی باشد.

    خطاها
    ------
    ValueError
        اگر input_tz یا output_tz یک منطقهٔ زمانی معتبر نباشد.

    TypeError
        اگر نوع value توسط تابع پشتیبانی نشود.
    """
    # ---------- validations ----------
    if value is None or value == "":
        return None
    # -------------------------------------------------------------------
    if input_tz is None:
        if isinstance(value, str) and value.strip().lower() != "now":
            try:
                dt = datetime.fromisoformat(value)
                if not isinstance(dt, datetime) and not isinstance(dt, date):
                    raise ValueError(f"Invalid datetime/date.")
                else:
                    pass
            except Exception as e:
                pass

            if value.tzinfo is not None and output_tz is not None:
                return value.astimezone(output_tz)
        else:
            pass

    else:
        if output_tz is None:
            output_tz = input_tz
    # -------------------------------------------------------------------

    try:
        if isinstance(input_tz, str):
            input_tz = ZoneInfo(input_tz)

        if isinstance(output_tz, str):
            output_tz = ZoneInfo(output_tz)

    except ZoneInfoNotFoundError as e:
        raise ValueError(f"Invalid timezone: {e}")

    # ---------- datetime -------------
    if isinstance(value, datetime):

        if value.tzinfo is None:
            value = value.replace(tzinfo=input_tz)

        return value.astimezone(output_tz)

    # ---------- date -----------------
    if isinstance(value, date):

        dt = datetime.combine(value, datetime.min.time())
        dt = dt.replace(tzinfo=input_tz)
        return dt.astimezone(output_tz)

    # ---------- unix timestamp -------
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=output_tz)

    # ---------- string ---------------
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == "now":
            return datetime.now(input_tz).astimezone(output_tz)

        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=input_tz)
        return dt.astimezone(output_tz)

    raise TypeError(f"Unsupported type: {type(value)}")


# === NEW & RENAMED =========================================================== new
def normalize_datetime(value, input_tz, output_tz=None):
    """
    مقدار تاریخ/زمان را دریافت کرده و آن را به یک datetime دارای منطقهٔ زمانی
    (timezone-aware datetime) تبدیل می‌کند.

    پارامترها
    ----------
    value : datetime, date, int, float, str یا None
        مقدار تاریخ/زمان ورودی.

        - اگر None یا رشتهٔ خالی باشد، مقدار None برگردانده می‌شود.

        - اگر datetime باشد:
            * اگر دارای timezone باشد (aware):
                مستقیماً به output_tz تبدیل می‌شود.
                در این حالت input_tz مورد نیاز نیست.

            * اگر فاقد timezone باشد (naive):
                input_tz به‌عنوان timezone مبدأ به آن نسبت داده می‌شود،
                سپس به output_tz تبدیل می‌شود.
                در این حالت input_tz الزامی است.

        - اگر date باشد:
            زمان 00:00:00 به آن اضافه شده و مقدار حاصل در input_tz
            تفسیر می‌شود، سپس به output_tz تبدیل می‌شود.
            در این حالت input_tz الزامی است.

        - اگر int یا float باشد:
            به‌عنوان Unix timestamp در نظر گرفته می‌شود و به datetime
            در output_tz تبدیل می‌شود.
            در این حالت output_tz الزامی است (مگر اینکه از input_tz
            برای تعیین آن استفاده شود).

        - اگر رشته برابر با "now" باشد:
            زمان فعلی در input_tz گرفته شده و سپس به output_tz تبدیل می‌شود.
            در این حالت input_tz الزامی است.

        - سایر رشته‌ها با datetime.fromisoformat() تجزیه می‌شوند:
            * اگر رشته شامل timezone باشد (مانند +00:00 یا Z):
                مقدار به‌عنوان aware datetime در نظر گرفته شده و
                مستقیماً به output_tz تبدیل می‌شود.

            * اگر رشته فاقد timezone باشد:
                input_tz به آن نسبت داده می‌شود و سپس به output_tz
                تبدیل می‌شود.
                در این حالت input_tz الزامی است.

    input_tz : str یا tzinfo یا None
        منطقهٔ زمانی مورد استفاده برای تفسیر مقادیر فاقد timezone.

        اگر رشته باشد، با ZoneInfo به timezone تبدیل می‌شود.

        مقدار None فقط زمانی مجاز است که مقدار ورودی خود دارای timezone
        باشد (مانند aware datetime یا ISO string دارای timezone).

    output_tz : str یا tzinfo یا None
        منطقهٔ زمانی خروجی.

        اگر مقدار None باشد:
            - برای مقادیر naive، input_tz به‌عنوان output_tz استفاده می‌شود.
            - برای مقادیر aware، باید به‌صورت صریح مشخص شود.

    خروجی
    -------
    datetime یا None

        یک datetime دارای timezone در منطقهٔ زمانی output_tz،
        یا None در صورتی که value برابر None یا رشتهٔ خالی باشد.

    خطاها
    ------
    ValueError
        در موارد زیر ایجاد می‌شود:

        - timezone نامعتبر باشد.
        - برای مقدار naive، input_tz مشخص نشده باشد.
        - برای مقدار aware، output_tz مشخص نشده باشد.
        - برای Unix timestamp، timezone خروجی قابل تعیین نباشد.
        - برای رشته با فرمت نامعتبر.

    TypeError
        اگر نوع value توسط تابع پشتیبانی نشود.
    """
    # ---------- validations ----------
    if value is None or value == "":
        return None


    # ---------- normalize timezone ----------
    try:
        if isinstance(input_tz, str):
            input_tz = ZoneInfo(input_tz)
        if isinstance(output_tz, str):
            output_tz = ZoneInfo(output_tz)
    except ZoneInfoNotFoundError as e:
        raise ValueError(f"Invalid timezone: {e}")


    # ---------- datetime ----------
    if isinstance(value, datetime):

        # aware datetime
        if value.tzinfo is not None:
            if output_tz is None:
                raise ValueError("output_tz is required for aware datetime")
            return value.astimezone(output_tz)

        # naive datetime
        if input_tz is None:
            raise ValueError("input_tz is required for naive datetime")

        value = value.replace(tzinfo=input_tz)

        if output_tz is None:
            output_tz = input_tz

        return value.astimezone(output_tz)

    # ---------- date ----------
    if isinstance(value, date):
        if input_tz is None:
            raise ValueError("input_tz is required for date")
        dt = datetime.combine(value, datetime.min.time(), tzinfo=input_tz)
        if output_tz is None:
            output_tz = input_tz
        return dt.astimezone(output_tz)


    # ---------- unix timestamp ----------
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if output_tz is None:
            output_tz = input_tz
        if output_tz is None:
            raise ValueError("output_tz is required for timestamp")
        return datetime.fromtimestamp(value, tz=output_tz)


    # ---------- string ----------
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == "now":
            if input_tz is None:
                raise ValueError("input_tz is required for now")
            if output_tz is None:
                output_tz = input_tz
            return datetime.now(input_tz).astimezone(output_tz)

        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            raise ValueError(f"Invalid date format: {value}")


        # aware ISO string ----------------------
        if dt.tzinfo is not None:
            if output_tz is None:
                raise ValueError("output_tz is required for aware datetime")
            return dt.astimezone(output_tz)


        # naive ISO string ----------------------
        if input_tz is None:
            raise ValueError("input_tz is required for naive datetime string")
        dt = dt.replace(tzinfo=input_tz)
        if output_tz is None:
            output_tz = input_tz
        return dt.astimezone(output_tz)
    

    raise TypeError(f"Unsupported type: {type(value)}")


# ============================================================================= END