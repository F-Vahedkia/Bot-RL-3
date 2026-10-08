# Run: python -m f17_my_learnings.L1_datetime

from f03_data.data_handler_F_2 import ensure_timezone_aware

# print("============================================================================== Part-1")
# from datetime import datetime

# # 1. ساده‌ترین دستور ممکن: ساعت الان کامپیوتر خودت را بگیر
# my_time = datetime.now()

# # حالا این ساعت را چاپ کن تا ببینی دقیقاً همان ساعتی را نشان می‌دهد که روی سیستم داری.
# print(my_time) 
# # خروجی نمونه: 2026-07-08 17:45:30.123456
# # (این همان ساعت ۵ و ۴۵ دقیقه عصر به وقت خودت است، بدون اینکه پایتون بداند در کدام کشور هستی)

# print("============================================================================== Part-2")
# from datetime import datetime
# import zoneinfo

# # 1. منطقه زمانی تهران را تعریف کن (این یک نقشه است)
# tehran_zone = zoneinfo.ZoneInfo("Asia/Tehran")

# # 2. حالا از پایتون بخواه که ساعت الان را بر اساس این نقشه به تو بدهد
# now_tehran = datetime.now(tehran_zone)

# # 3. چاپ کن
# print(now_tehran) 
# # خروجی نمونه: 2026-07-08 17:15:30.123456+03:30
# now_tehran_naive = now_tehran.replace(tzinfo=None)
# print(now_tehran_naive)

# print("============================================================================== Part-3")
# from datetime import datetime
# import zoneinfo

# # گرفتن وقت دبی (که +۴ است)
# now_dubai = datetime.now(zoneinfo.ZoneInfo("Asia/Dubai"))
# print(f"Dubai time: {now_dubai}")

# # گرفتن وقت لندن (که در زمستان +۰ و در تابستان +۱ است)
# now_london = datetime.now(zoneinfo.ZoneInfo("Europe/London"))
# print(f"London time: {now_london}")

# print("============================================================================== Part-4")
# from datetime import datetime
# import zoneinfo

# # 1. وقت الان لندن را بگیر
# now_london = datetime.now(zoneinfo.ZoneInfo("Europe/London"))

# # 2. ساعت و دقیقه آن را به عدد تبدیل کن
# hour_int = now_london.hour
# minute_int = now_london.minute

# # 3. چک کن که آیا بین ۸ صبح تا ۵ عصر لندن است یا نه
# if 8 <= hour_int <= 17:
#     print("London market is open")
# else:
#     print("London market is closed")

# print("============================================================================== Part-5")
# temp = datetime(2026,1,2,3,4,5,tzinfo=zoneinfo.ZoneInfo("Asia/Dubai"))
# print(f"Asia/Dubai: {temp}")

# temp = datetime(2026,1,2,3,4,5,tzinfo=zoneinfo.ZoneInfo("Europe/London"))
# print(f"Europe/London: {temp}")

# temp = datetime.strptime("2026-04-05 01:02:03+03:30", "%Y-%m-%d %H:%M:%S%z")

print("============================================================================== Part-6")
from datetime import datetime, timezone
import zoneinfo
import pandas as pd

# timestamp_str = "170000000"
# timestamp_int = int(timestamp_str)
# dt10 = datetime.fromtimestamp(timestamp_int)  # زمان محلی سیستم
# print(dt10)  # output: 2023-11-15 01:43:20

# # برای دریافت UTC
# dt11 = datetime.fromtimestamp(timestamp_int, tz=timezone.utc)
# print(dt11)  # output: 1975-05-22 14:13:20+00:00

# # برای دریافت زمان آسیا/دوبی
# dt11 = datetime.fromtimestamp(timestamp_int, tz=zoneinfo.ZoneInfo("Asia/Dubai"))
# print(dt11)

# # تبدیل به وقت تهران
# dt_tehran = datetime.fromtimestamp(timestamp_int, tz=zoneinfo.ZoneInfo("Asia/Tehran"))
# print(dt_tehran)  # خروجی: 2023-11-14 23:33:20.500000+03:30 (زمستان)

# # تبدیل به وقت نیویورک
# dt_ny = datetime.fromtimestamp(timestamp_int, tz=zoneinfo.ZoneInfo("America/New_York"))
# print(dt_ny)  # خروجی: 2023-11-14 15:03:20.500000-05:00 (زمستان)

# # رشته بدون منطقه
# dt_rio = pd.to_datetime("2026-01-02 12:13:14", tz="America/Sao_Paulo")
# print(dt_rio)  
# # خروجی: 2026-01-02 12:13:14-03:00 (تابستان برزیل -۳ ساعت)



dt_naive = pd.to_datetime("2026-01-02 12:13:14")
dt_rio = dt_naive.tz_localize("America/Sao_Paulo")
print(f"dt_rio    : {dt_rio}")

dt_tehran = dt_naive.tz_localize("Asia/Tehran")
print(f"dt_tehran : {dt_tehran}")

tehran_from_rio = dt_rio.tz_convert("Asia/Tehran")
print(f"tehran_from_rio : {tehran_from_rio}")
