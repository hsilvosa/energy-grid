from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

MADRID = ZoneInfo("Europe/Madrid")


def delivery_intervals(delivery_date: date) -> list[tuple[datetime, datetime]]:
    """Return real UTC quarter hours for a local delivery day, including DST days."""
    local_start = datetime.combine(delivery_date, time.min, MADRID)
    local_end = datetime.combine(delivery_date + timedelta(days=1), time.min, MADRID)
    cursor = local_start.astimezone(UTC)
    end = local_end.astimezone(UTC)
    result: list[tuple[datetime, datetime]] = []
    while cursor < end:
        next_cursor = cursor + timedelta(minutes=15)
        result.append((cursor, next_cursor))
        cursor = next_cursor
    return result


def interval_number(timestamp: datetime) -> int:
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone aware")
    local_date = timestamp.astimezone(MADRID).date()
    intervals = delivery_intervals(local_date)
    instant = timestamp.astimezone(UTC)
    for index, (start, _) in enumerate(intervals, start=1):
        if start == instant:
            return index
    raise ValueError("timestamp is not aligned to a 15-minute delivery interval")


def floor_quarter_hour(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone aware")
    timestamp = timestamp.astimezone(UTC)
    return timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
