# Run: python f02_utils/functions/normalize_datetime_test_print.py


#           این فایل باید حفظ شود



import sys
sys.path.insert(0, r"E:\Bot-RL-3")
from f02_utils.functions.normalize_datetime import normalize_datetime
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo

INPUT_TZ = "Europe/Athens"
# OUTPUT_TZ = "UTC"
OUTPUT_TZ = "Asia/Tehran"

# =============================================================================

def print_result(label, value, input_tz=INPUT_TZ, output_tz=OUTPUT_TZ):

    try:
        result = normalize_datetime(
            value,
            input_tz,
            output_tz
        )

        val_str = str(value) if value is not None else "None"

        print(
            f"{label:32}: "
            f"{val_str:35} → "
            f"{result} "
            f"(tz={result.tzinfo if result else None})"
        )

    except Exception as e:

        print(
            f"{label:32}: "
            f"{str(value):35} → ERROR: {e}"
        )


# =============================================================================
# Basic inputs
# =============================================================================

print_result("1) None", None)

print_result("2) Empty string", "")


# =============================================================================
# String inputs
# =============================================================================

print_result(
    "3) ISO date only",
    "2024-01-01"
)

print_result(
    "4) ISO with Z (UTC)",
    "2024-01-01T12:00:00Z"
)

print_result(
    "5) ISO with space + Z",
    "2024-01-01 12:00:00Z"
)

print_result(
    "6) ISO naive",
    "2024-01-01T12:00:00"
)

print_result(
    "7) ISO with offset",
    "2024-01-01T12:00:00+04:00"
)


print_result(
    "8) now",
    "now"
)


# =============================================================================
# datetime inputs
# =============================================================================

print_result(
    "9) datetime aware UTC",
    datetime(
        2024,
        1,
        1,
        12,
        0,
        tzinfo=timezone.utc
    )
)


print_result(
    "10) datetime aware Tehran",
    datetime(
        2024,
        1,
        1,
        12,
        0,
        tzinfo=ZoneInfo("Asia/Tehran")
    )
)


print_result(
    "11) datetime naive",
    datetime(
        2024,
        1,
        1,
        12,
        0
    )
)


# =============================================================================
# date input
# =============================================================================

print_result(
    "12) date object",
    date(2024, 1, 1)
)


# =============================================================================
# timestamp
# =============================================================================

print_result(
    "13) unix timestamp",
    1704103200
)


# =============================================================================
# IMPORTANT:
# aware datetime without input_tz
# =============================================================================

print_result(
    "14) aware datetime input_tz=None",
    datetime(
        2024,
        1,
        1,
        12,
        0,
        tzinfo=ZoneInfo("Asia/Tehran")
    ),
    input_tz=None,
    output_tz="UTC"
)


# =============================================================================
# invalid cases
# =============================================================================

print_result(
    "15) invalid date format",
    "2024/01/01"
)


print_result(
    "16) naive without input_tz",
    datetime(
        2024,
        1,
        1,
        12,
        0
    ),
    input_tz=None,
    output_tz="UTC"
)


print("\nDone.")

# ============================================================================= END
