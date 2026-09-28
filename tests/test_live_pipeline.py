from datetime import UTC, datetime, timedelta

from energy_grid.domain import EventType, GridEvent
from energy_grid.live_pipeline import (
    build_training_frame,
    events_to_series,
    source_snapshot_id,
    weather_to_frame,
)


def grid_event(
    target: EventType,
    start: datetime,
    value: float,
    *,
    minutes: int = 15,
    dimension: str | None = None,
    checksum: str = "checksum",
) -> GridEvent:
    return GridEvent(
        source="test",
        event_type=target,
        interval_start=start,
        interval_end=start + timedelta(minutes=minutes),
        published_at=start,
        ingested_at=start,
        value=value,
        unit=(
            "MW"
            if target == EventType.DEMAND
            else "EUR/MWh"
            if target == EventType.PRICE
            else {
                "temperature_2m": "degC",
                "wind_speed_10m": "m/s",
                "relative_humidity_2m": "%",
                "shortwave_radiation": "W/m2",
            }[dimension]
        ),
        dimension=dimension,
        payload_checksum=checksum,
    )


def test_hourly_source_interval_expands_to_quarter_hours() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    series = events_to_series(
        [grid_event(EventType.PRICE, start, 50.0, minutes=60)], EventType.PRICE
    )
    assert len(series) == 4
    assert series.tolist() == [50.0] * 4


def test_newer_revision_wins() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first = grid_event(EventType.DEMAND, start, 100.0)
    revised = first.model_copy(update={"revision": 2, "value": 110.0})
    series = events_to_series([first, revised], EventType.DEMAND)
    assert series.iloc[0] == 110.0


def test_real_training_frame_contains_lagged_weather_features() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    demand = [
        grid_event(EventType.DEMAND, start + timedelta(minutes=15 * index), 20000 + index)
        for index in range(96 * 9)
    ]
    weather = []
    for hour in range(24 * 9):
        timestamp = start + timedelta(hours=hour)
        weather.extend(
            [
                grid_event(
                    EventType.WEATHER,
                    timestamp,
                    15.0,
                    minutes=60,
                    dimension="temperature_2m",
                ),
                grid_event(
                    EventType.WEATHER,
                    timestamp,
                    3.0,
                    minutes=60,
                    dimension="wind_speed_10m",
                ),
                grid_event(
                    EventType.WEATHER,
                    timestamp,
                    60.0,
                    minutes=60,
                    dimension="relative_humidity_2m",
                ),
                grid_event(
                    EventType.WEATHER,
                    timestamp,
                    100.0,
                    minutes=60,
                    dimension="shortwave_radiation",
                ),
            ]
        )
    frame = build_training_frame(demand, weather, EventType.DEMAND)
    assert not frame.empty
    assert frame.iloc[0]["lag_672"] == 20000
    assert frame.iloc[0]["temperature_2m"] == 15.0


def test_snapshot_is_stable_and_identifies_real_inputs() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first = [grid_event(EventType.DEMAND, start, 1.0, checksum="a")]
    second = [grid_event(EventType.PRICE, start, 2.0, checksum="b")]
    assert source_snapshot_id(first, second) == source_snapshot_id(second, first)
    assert source_snapshot_id(first, second).startswith("live-")


def test_live_weather_excludes_revisions_published_after_cutoff() -> None:
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    original = grid_event(
        EventType.WEATHER, start + timedelta(hours=1), 10.0,
        minutes=60, dimension="temperature_2m",
    ).model_copy(update={"published_at": start - timedelta(minutes=5)})
    future_revision = original.model_copy(update={
        "revision": 2,
        "value": 99.0,
        "published_at": start + timedelta(minutes=5),
    })
    frame = weather_to_frame([original, future_revision], available_at=start)
    assert frame.iloc[0]["temperature_2m"] == 10.0
