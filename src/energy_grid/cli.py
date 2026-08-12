from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from energy_grid.config import get_settings
from energy_grid.domain import EventType
from energy_grid.forecasting import materialize_demo_forecasts
from energy_grid.live_service import run_live_pipeline
from energy_grid.monitoring import EVENTS_PUBLISHED
from energy_grid.replay import KafkaProducerAdapter, ReplayProfile, load_fixture
from energy_grid.replay import replay as replay_events
from energy_grid.rollback import RollbackController, RollbackState
from energy_grid.storage import ForecastStore

app = typer.Typer(no_args_is_help=True)


@app.command()
def replay(
    fixture: Annotated[Path, typer.Option(exists=True)] = Path("data/fixtures/events.jsonl"),
    duplicate_rate: float = 0.08,
    missing_rate: float = 0.03,
    late_rate: float = 0.08,
    corrupt_rate: float = 0.01,
    speed: float = 0.0,
) -> None:
    settings = get_settings()
    producer = KafkaProducerAdapter(settings.kafka_bootstrap_servers)
    profile = ReplayProfile(
        duplicate_rate=duplicate_rate,
        missing_rate=missing_rate,
        late_rate=late_rate,
        corrupt_rate=corrupt_rate,
        speed=speed,
    )
    count = replay_events(
        load_fixture(fixture),
        producer,
        profile,
        on_publish=lambda topic: EVENTS_PUBLISHED.labels(topic).inc(),
    )
    typer.echo(f"Published {count} events")


@app.command("materialize-demo")
def materialize_demo() -> None:
    store = ForecastStore(get_settings().database_url)
    store.create_schema()
    demand = materialize_demo_forecasts(EventType.DEMAND)
    price = materialize_demo_forecasts(EventType.PRICE)
    count = store.upsert_forecasts(demand + price)
    for target in (EventType.DEMAND, EventType.PRICE):
        store.set_model_status(
            target,
            champion_version=(demand if target == EventType.DEMAND else price)[0].model_version,
            previous_version=None,
            baseline_version="sarimax-v1",
            last_evaluated_at=datetime.now(UTC),
            rolling_error=None,
            baseline_error=None,
            consecutive_breaches=0,
            status="fixture_demo",
        )
    typer.echo(f"Materialized {count} forecasts")


@app.command()
def materialize(
    target: Annotated[EventType, typer.Option()] = EventType.DEMAND,
) -> None:
    if target not in (EventType.DEMAND, EventType.PRICE):
        raise typer.BadParameter("target must be demand or price")
    store = ForecastStore(get_settings().database_url)
    store.create_schema()
    forecasts = materialize_demo_forecasts(target)
    store.upsert_forecasts(forecasts)
    typer.echo(f"Materialized {len(forecasts)} {target.value} forecasts")


def _simulate_degradation(store: ForecastStore, target: EventType, degraded_version: str) -> None:
    controller = RollbackController()
    state = RollbackState()
    rolled_back = False
    for error in (125.0, 128.0, 130.0):
        rolled_back = controller.evaluate(state, champion_error=error, reference_error=100.0)
    store.set_model_status(
        target,
        champion_version="lightgbm-stable-v0" if rolled_back else degraded_version,
        previous_version=degraded_version if rolled_back else "lightgbm-stable-v0",
        baseline_version="sarimax-v1",
        last_evaluated_at=datetime.now(UTC),
        rolling_error=130.0,
        baseline_error=100.0,
        consecutive_breaches=state.consecutive_breaches,
        status="rolled_back" if rolled_back else "degraded",
    )


@app.command("simulate-degradation")
def simulate_degradation(
    target: Annotated[EventType, typer.Option()] = EventType.DEMAND,
    degraded_version: str = "lightgbm-degraded-demo",
) -> None:
    store = ForecastStore(get_settings().database_url)
    store.create_schema()
    _simulate_degradation(store, target, degraded_version)
    typer.echo(f"Simulated three degradation windows and rolled back {target.value}")


@app.command()
def backfill(
    fixture: Annotated[Path, typer.Option(exists=True)] = Path("data/fixtures/events.jsonl"),
) -> None:
    """Validate an idempotent bounded backfill input and emit its deterministic run id."""
    raw = fixture.read_bytes()
    events = load_fixture(fixture)
    if not events:
        raise typer.BadParameter("fixture contains no events")
    checksum = hashlib.sha256(raw).hexdigest()
    start = min(event.interval_start for event in events)
    end = max(event.interval_end for event in events)
    identity = f"{checksum}|{start.isoformat()}|{end.isoformat()}"
    run_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    typer.echo(f"backfill_run_id={run_id} records={len(events)} checksum={checksum}")


@app.command()
def demo(
    fixture: Annotated[Path, typer.Option(exists=True)] = Path("data/fixtures/events.jsonl"),
) -> None:
    """Run the credential-free replay and forecast materialization scenario."""
    settings = get_settings()
    last_error: Exception | None = None
    producer: KafkaProducerAdapter | None = None
    for _ in range(20):
        try:
            producer = KafkaProducerAdapter(settings.kafka_bootstrap_servers)
            break
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    if producer is None:
        raise RuntimeError("Kafka did not become ready") from last_error
    count = replay_events(
        load_fixture(fixture),
        producer,
        ReplayProfile(duplicate_rate=0.15, missing_rate=0.05, late_rate=0.15, corrupt_rate=0.05),
        on_publish=lambda topic: EVENTS_PUBLISHED.labels(topic).inc(),
    )
    store = ForecastStore(settings.database_url)
    store.create_schema()
    forecasts = materialize_demo_forecasts(EventType.DEMAND) + materialize_demo_forecasts(
        EventType.PRICE
    )
    store.upsert_forecasts(forecasts)
    for target in (EventType.DEMAND, EventType.PRICE):
        degraded = next(item.model_version for item in forecasts if item.target == target)
        _simulate_degradation(store, target, degraded)
    typer.echo(f"Demo complete: {count} events and {len(forecasts)} forecasts")


@app.command("live-demo")
def live_demo(
    history_days: Annotated[int, typer.Option(min=15, max=365)] = 45,
    generation_days: Annotated[int, typer.Option(min=1, max=30)] = 2,
    publish_kafka: Annotated[bool, typer.Option()] = True,
    register_mlflow: Annotated[bool, typer.Option()] = True,
) -> None:
    """Download real Spanish data, train models and materialize live forecasts."""
    summary = run_live_pipeline(
        get_settings(),
        history_days=history_days,
        generation_days=generation_days,
        publish_kafka=publish_kafka,
        register_mlflow=register_mlflow,
    )
    typer.echo(
        "Real-data run complete: "
        f"demand_events={summary.demand_events}, "
        f"price_events={summary.price_events}, "
        f"generation_events={summary.generation_events}, "
        f"weather_events={summary.weather_events}, "
        f"kafka_events={summary.kafka_events}"
    )
    for result in (summary.demand_result, summary.price_result):
        typer.echo(
            f"{result.target.value}: forecasts={len(result.forecasts)}, "
            f"training_rows={result.training_rows}, "
            f"mae={result.metrics.mae:.3f}, baseline_mae={result.baseline_mae:.3f}, "
            f"snapshot={result.snapshot_id}"
        )
    if summary.mlflow_run_ids:
        typer.echo(f"MLflow runs: {summary.mlflow_run_ids}")


if __name__ == "__main__":
    app()
