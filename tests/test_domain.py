from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from energy_grid.domain import EventType, GridEvent, QualityStatus


def event(**updates: object) -> GridEvent:
    values = {
        "source": "entsoe",
        "event_type": EventType.DEMAND,
        "interval_start": datetime(2025, 1, 1, tzinfo=UTC),
        "interval_end": datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=15),
        "published_at": datetime(2024, 12, 31, 23, 55, tzinfo=UTC),
        "ingested_at": datetime(2025, 1, 1, 0, 16, tzinfo=UTC),
        "value": 25000.0,
        "unit": "MW",
    }
    values.update(updates)
    return GridEvent.model_validate(values)


def test_business_key_is_stable_across_revisions() -> None:
    first = event(revision=1)
    revised = event(revision=2, value=25100.0)
    assert first.event_id != revised.event_id
    assert first.business_key == revised.business_key


def test_event_round_trip_preserves_utc() -> None:
    original = event()
    parsed = GridEvent.from_message(original.to_message())
    assert parsed == original
    assert parsed.interval_start.tzinfo == UTC


def test_missing_value_must_be_null() -> None:
    with pytest.raises(ValidationError, match="missing events"):
        event(quality_status=QualityStatus.MISSING, value=1.0)


def test_unit_is_checked_by_event_type() -> None:
    with pytest.raises(ValidationError, match="invalid for"):
        event(unit="EUR/MWh")


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        event(interval_start=datetime(2025, 1, 1))

