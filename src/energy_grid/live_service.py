from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta

from energy_grid.config import Settings
from energy_grid.domain import EventType, ForecastRecord, GridEvent
from energy_grid.live_pipeline import TrainingResult, log_training_result, train_and_forecast
from energy_grid.monitoring import EVENTS_PUBLISHED
from energy_grid.release import (
    ReleaseDecision,
    evaluate_and_record_candidate,
    forecast_with_approved_model,
)
from energy_grid.replay import TOPICS, KafkaProducerAdapter
from energy_grid.sources.entsoe import EntsoeClient
from energy_grid.sources.open_meteo import OpenMeteoClient
from energy_grid.storage import ForecastStore


@dataclass(frozen=True)
class LiveRunSummary:
    demand_events: int
    price_events: int
    generation_events: int
    weather_events: int
    kafka_events: int
    demand_result: TrainingResult
    price_result: TrainingResult
    mlflow_run_ids: dict[str, str]
    releases: dict[str, ReleaseDecision]


@dataclass(frozen=True)
class LiveInferenceSummary:
    demand_events: int
    price_events: int
    weather_events: int
    forecasts_written: int


def _publish(events: list[GridEvent], producer: KafkaProducerAdapter) -> int:
    for event in events:
        topic = TOPICS[event.event_type.value]
        producer.send(topic, event.business_key.encode(), event.to_message())
        EVENTS_PUBLISHED.labels(topic).inc()
    return len(events)


def run_live_pipeline(
    settings: Settings,
    *,
    history_days: int = 45,
    generation_days: int = 2,
    publish_kafka: bool = True,
    register_mlflow: bool = True,
    now: datetime | None = None,
) -> LiveRunSummary:
    if not settings.entsoe_token:
        raise ValueError("ENTSOE_TOKEN is missing; add it to .env")
    if history_days < 15:
        raise ValueError("history_days must be at least 15 to construct seven-day lags")
    reference_time = (now or datetime.now(UTC)).astimezone(UTC)
    history_end = datetime.combine(reference_time.date(), time.min, UTC)
    history_start = history_end - timedelta(days=history_days)
    generation_start = history_end - timedelta(days=min(generation_days, history_days))

    entsoe = EntsoeClient(settings.entsoe_token, timeout=90)
    weather = OpenMeteoClient(timeout=90)
    demand_events = entsoe.events(EventType.DEMAND, history_start, history_end)
    price_events = entsoe.events(EventType.PRICE, history_start, history_end)
    generation_events = entsoe.events(EventType.GENERATION, generation_start, history_end)
    historical_weather = weather.events(
        start_date=history_start.date().isoformat(),
        end_date=(history_end.date() - timedelta(days=1)).isoformat(),
        historical=True,
    )
    live_weather = weather.events(historical=False)
    current = (now or datetime.now(UTC)).astimezone(UTC)

    demand_result = train_and_forecast(
        demand_events,
        historical_weather,
        live_weather,
        EventType.DEMAND,
        issue_time=current,
    )
    price_result = train_and_forecast(
        price_events,
        historical_weather,
        live_weather,
        EventType.PRICE,
        issue_time=current,
    )

    store = ForecastStore(settings.database_url)
    store.create_schema()
    releases: dict[str, ReleaseDecision] = {}
    for result in (demand_result, price_result):
        decision = evaluate_and_record_candidate(
            store, result, settings.model_artifact_root
        )
        releases[result.target.value] = decision
        store.upsert_forecasts(decision.forecasts)

    kafka_count = 0
    if publish_kafka:
        producer = KafkaProducerAdapter(settings.kafka_bootstrap_servers)
        for group in (
            demand_events,
            price_events,
            generation_events,
            historical_weather,
            live_weather,
        ):
            kafka_count += _publish(group, producer)
        producer.flush()

    mlflow_runs: dict[str, str] = {}
    if register_mlflow:
        for result in (demand_result, price_result):
            mlflow_runs[result.target.value] = log_training_result(
                result, settings.mlflow_tracking_uri
            )

    return LiveRunSummary(
        demand_events=len(demand_events),
        price_events=len(price_events),
        generation_events=len(generation_events),
        weather_events=len(historical_weather) + len(live_weather),
        kafka_events=kafka_count,
        demand_result=demand_result,
        price_result=price_result,
        mlflow_run_ids=mlflow_runs,
        releases=releases,
    )


def run_live_inference(
    settings: Settings,
    *,
    history_days: int = 15,
    targets: tuple[EventType, ...] = (EventType.DEMAND, EventType.PRICE),
    now: datetime | None = None,
) -> LiveInferenceSummary:
    """Fetch current inputs and forecast using approved artifacts only."""
    if not settings.entsoe_token:
        raise ValueError("ENTSOE_TOKEN is missing; add it to the runtime secret store")
    reference_time = (now or datetime.now(UTC)).astimezone(UTC)
    history_end = datetime.combine(reference_time.date(), time.min, UTC)
    history_start = history_end - timedelta(days=history_days)
    entsoe = EntsoeClient(settings.entsoe_token, timeout=90)
    weather = OpenMeteoClient(timeout=90)
    demand_events = (
        entsoe.events(EventType.DEMAND, history_start, history_end)
        if EventType.DEMAND in targets else []
    )
    price_events = (
        entsoe.events(EventType.PRICE, history_start, history_end)
        if EventType.PRICE in targets else []
    )
    live_weather = weather.events(historical=False)
    current = (now or datetime.now(UTC)).astimezone(UTC)
    store = ForecastStore(settings.database_url)
    store.create_schema()
    forecasts: list[ForecastRecord] = []
    for target, events in (
        (EventType.DEMAND, demand_events),
        (EventType.PRICE, price_events),
    ):
        if target not in targets:
            continue
        forecasts.extend(
            forecast_with_approved_model(
                store, events, live_weather, target, issue_time=current
            )
        )
    store.upsert_forecasts(forecasts)
    return LiveInferenceSummary(
        demand_events=len(demand_events),
        price_events=len(price_events),
        weather_events=len(live_weather),
        forecasts_written=len(forecasts),
    )
