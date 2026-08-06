from datetime import UTC, date, datetime

import pytest

from energy_grid.time_utils import delivery_intervals, interval_number


def test_delivery_day_has_92_intervals_when_dst_starts() -> None:
    intervals = delivery_intervals(date(2025, 3, 30))
    assert len(intervals) == 92
    assert intervals[0][0] == datetime(2025, 3, 29, 23, tzinfo=UTC)
    assert intervals[-1][1] == datetime(2025, 3, 30, 22, tzinfo=UTC)


def test_delivery_day_has_100_intervals_when_dst_ends() -> None:
    intervals = delivery_intervals(date(2025, 10, 26))
    assert len(intervals) == 100
    assert intervals[0][0] == datetime(2025, 10, 25, 22, tzinfo=UTC)
    assert intervals[-1][1] == datetime(2025, 10, 26, 23, tzinfo=UTC)


def test_interval_number_handles_repeated_local_hour() -> None:
    intervals = delivery_intervals(date(2025, 10, 26))
    numbers = [interval_number(start) for start, _ in intervals]
    assert numbers == list(range(1, 101))


def test_interval_number_rejects_unaligned_timestamp() -> None:
    with pytest.raises(ValueError, match="not aligned"):
        interval_number(datetime(2025, 1, 1, 0, 1, tzinfo=UTC))

