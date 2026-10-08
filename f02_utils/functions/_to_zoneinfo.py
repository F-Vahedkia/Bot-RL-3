
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# =============================================================================
def _to_zoneinfo(value: str | ZoneInfo | None) -> ZoneInfo | None:
    if value is None:
        return None

    if isinstance(value, str):
        try:
            return ZoneInfo(value.strip())
        except ZoneInfoNotFoundError as e:
            raise ValueError(f"Invalid timezone: '{value}'") from e

    if isinstance(value, ZoneInfo):
        return value

    raise TypeError("Timezone must be 'ZoneInfo', 'str' or 'None'.")

# ============================================================================= END
