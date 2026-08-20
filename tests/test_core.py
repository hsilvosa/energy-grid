from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from energy_grid.domain import EventType, GridEvent
from energy_grid.features import asof_features, build_multiscale_features
from energy_grid.replay import ReplayProfile, load_fixture, replay
from energy_grid.rollback import RollbackController, RollbackState, promotion_gate
from energy_grid.time_utils import delivery_intervals, interval_number

MADRID = ZoneInfo("Europe/Madrid")


def test_grid_event_and_time_utils() -> None:
    now = datetime(2025, 6, 1, 10, 0, tzinfo=UTC)
    event = GridEvent(
        source="entsoe",
        event_type=EventType.DEMAND,
        interval_start=now,
        interval_end=now + timedelta(minutes=15),
        published_at=now,
        value=28500.0,
        unit="MW",
    )
    assert event.event_type == EventType.DEMAND
    assert event.area == "10YES-REE------0"
    assert len(event.business_key) == 64

    # Test delivery intervals: regular day has 96, spring DST has 92, autumn DST has 100
    reg = delivery_intervals(date(2025, 6, 1))
    dst_spring = delivery_intervals(date(2025, 3, 30))
    dst_autumn = delivery_intervals(date(2025, 10, 26))
    assert len(reg) == 96
    assert len(dst_spring) == 92
    assert len(dst_autumn) == 100
    assert interval_number(reg[0][0]) == 1


def test_features_and_multiscale() -> None:
    origins = pd.DataFrame(
        {
            "forecast_origin": [pd.Timestamp("2025-01-01 10:00:00+00:00")],
            "valid_time": [pd.Timestamp("2025-01-01 10:15:00+00:00")],
        }
    )
    measurements = pd.DataFrame(
        {
            "published_at": [pd.Timestamp("2025-01-01 09:45:00+00:00")],
            "interval_start": [pd.Timestamp("2025-01-01 09:30:00+00:00")],
            "value": [25000.0],
        }
    )
    asof = asof_features(measurements, origins)
    assert not asof.empty
    assert asof["value"].iloc[0] == 25000.0

    # Multiscale lag generation
    series = pd.Series(
        range(1500),
        index=pd.date_range("2025-01-01", periods=1500, freq="15min", tz="UTC"),
        name="demand",
    )
    df = build_multiscale_features(series, steps_per_day=96)
    assert "lag_24h" in df.columns
    assert "lag_7d" in df.columns
    assert "sin_hour" in df.columns
    assert "rolling_mean_24h" in df.columns
    assert not df.isna().any().any()


def test_replay_and_rollback() -> None:
    events = load_fixture(Path("data/fixtures/events.jsonl"))
    assert len(events) > 0

    class MockProducer:
        def __init__(self) -> None:
            self.published: list[str] = []

        def send(self, topic: str, key: bytes, value: bytes) -> None:
            self.published.append(topic)

        def flush(self) -> None:
            pass

    producer = MockProducer()
    count = replay(
        events,
        producer,
        ReplayProfile(
            duplicate_rate=0.0,
            missing_rate=0.0,
            late_rate=0.0,
            corrupt_rate=0.0,
            speed=0.0,
        ),
    )
    assert count == len(events)
    assert len(producer.published) == len(events)

    # Rollback controller
    ctrl = RollbackController()
    state = RollbackState()
    assert not ctrl.evaluate(state, champion_error=125.0, reference_error=100.0)
    assert not ctrl.evaluate(state, champion_error=125.0, reference_error=100.0)
    # Third consecutive breach triggers rollback
    assert ctrl.evaluate(state, champion_error=125.0, reference_error=100.0)
    result = promotion_gate(champion_error=100.0, challenger_error=97.0, slice_regressions=[0.05])
    assert result.promote
