# Run: pytest f02_utils/functions/normalize_datetime_test.py -v

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
from f02_utils.functions.normalize_datetime import normalize_datetime

INPUT_TZ = "Asia/Tehran"   # منطقه ورودی فرضی
OUTPUT_TZ = "UTC"          # منطقه خروجی نهایی

# -- 1 --------------------------------
def test_none():
    assert normalize_datetime(None, INPUT_TZ, OUTPUT_TZ) is None

# -- 2 --------------------------------
def test_empty_string():
    assert normalize_datetime("", INPUT_TZ, OUTPUT_TZ) is None

# -- 3 --------------------------------
def test_datetime_aware():
    dt = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    result = normalize_datetime(dt, INPUT_TZ, OUTPUT_TZ)
    # ابتدا dt باید به INPUT_TZ تبدیل شود، سپس به OUTPUT_TZ
    expected = dt.astimezone(ZoneInfo(INPUT_TZ)).astimezone(ZoneInfo(OUTPUT_TZ))
    assert result == expected
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

# -- 4 --------------------------------
def test_datetime_naive():
    dt = datetime(2024, 1, 1, 12, 0)
    result = normalize_datetime(dt, INPUT_TZ, OUTPUT_TZ)
    expected = dt.replace(tzinfo=ZoneInfo(INPUT_TZ)).astimezone(ZoneInfo(OUTPUT_TZ))
    assert result == expected
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

# -- 5 --------------------------------
def test_date_object():
    d = date(2024, 1, 1)
    result = normalize_datetime(d, INPUT_TZ, OUTPUT_TZ)
    expected = datetime.combine(d, datetime.min.time(), tzinfo=ZoneInfo(INPUT_TZ)).astimezone(ZoneInfo(OUTPUT_TZ))
    assert result == expected
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

# -- 6 --------------------------------
def test_now_string():
    result = normalize_datetime("now", INPUT_TZ, OUTPUT_TZ)
    assert isinstance(result, datetime)
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

# -- 7 --------------------------------
def test_date_only_string():
    result = normalize_datetime("2024-01-01", INPUT_TZ, OUTPUT_TZ)
    expected = datetime(2024, 1, 1, 0, 0, tzinfo=ZoneInfo(INPUT_TZ)).astimezone(ZoneInfo(OUTPUT_TZ))
    assert result == expected
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

# -- 8 --------------------------------
def test_iso_with_z():
    result = normalize_datetime("2024-01-01T12:00:00Z", INPUT_TZ, OUTPUT_TZ)
    expected = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).astimezone(ZoneInfo(OUTPUT_TZ))
    assert result == expected
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

# -- 9 --------------------------------
def test_iso_naive():
    result = normalize_datetime("2024-01-01T12:00:00", INPUT_TZ, OUTPUT_TZ)
    expected = datetime(2024, 1, 1, 12, 0, tzinfo=ZoneInfo(INPUT_TZ)).astimezone(ZoneInfo(OUTPUT_TZ))
    assert result == expected
    assert result.tzinfo == ZoneInfo(OUTPUT_TZ)

# -- 10 --------------------------------
def test_invalid_format():
    with pytest.raises(ValueError, match="Invalid date format"):
        normalize_datetime("2024/01/01", INPUT_TZ, OUTPUT_TZ)

# -- 11 --------------------------------
def test_invalid_timezone():
    with pytest.raises(ValueError, match="Invalid timezone"):
        normalize_datetime("2024-01-01", "Invalid/Timezone", OUTPUT_TZ)

# -- END ----------------------------------------------------------------------
