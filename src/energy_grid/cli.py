from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from energy_grid.backtest import run_rolling_backtest
from energy_grid.config import get_settings
from energy_grid.domain import EventType
from energy_grid.features import build_multiscale_features
from energy_grid.forecasting import evaluate, materialize_demo_forecasts
from energy_grid.hf_hub import export_model_to_hf_package, upload_model_to_hf
from energy_grid.live_service import run_live_pipeline
from energy_grid.monitoring import EVENTS_PUBLISHED
from energy_grid.replay import KafkaProducerAdapter, ReplayProfile, load_fixture
from energy_grid.replay import replay as replay_events
from energy_grid.rollback import RollbackController, RollbackState
from energy_grid.sources.dataset_reader import DEFAULT_DATASET_PATH, EntsoeDatasetReader
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


@app.command("summarize-dataset")
def summarize_dataset(
    data_dir: Annotated[Path, typer.Option()] = DEFAULT_DATASET_PATH,
    country_code: Annotated[str, typer.Option()] = "ES",
) -> None:
    """Inspect and summarize local parquet datasets in data_dir."""
    reader = EntsoeDatasetReader(data_dir)
    if not reader.exists():
        typer.echo(f"Dataset directory not found at {data_dir}")
        raise typer.Exit(code=1)

    for target in (EventType.DEMAND, EventType.PRICE):
        try:
            summary = reader.summarize(target, country_code=country_code)
            typer.echo(
                f"[{summary.dataset_type.upper()}] {summary.country_code} ({summary.zone_key}): "
                f"{summary.total_records:,} records from "
                f"{summary.start_time.date()} to {summary.end_time.date()} | "
                f"Range: {summary.min_value:.2f} to {summary.max_value:.2f} {summary.unit} "
                f"(Mean: {summary.mean_value:.2f} {summary.unit})"
            )
        except Exception as exc:
            typer.echo(f"Could not load summary for {target.value}: {exc}")


@app.command("train-historical")
def train_historical(
    data_dir: Annotated[Path, typer.Option()] = DEFAULT_DATASET_PATH,
    target: Annotated[EventType, typer.Option()] = EventType.DEMAND,
    country_code: Annotated[str, typer.Option()] = "ES",
    zone_key: Annotated[str | None, typer.Option()] = None,
    model_type: Annotated[
        str, typer.Option(help="lightgbm, xgboost, catboost, stacked, dual_horizon")
    ] = "dual_horizon",
    start_year: Annotated[int, typer.Option()] = 2022,
    end_year: Annotated[int, typer.Option()] = 2026,
    calibrate: Annotated[bool, typer.Option()] = True,
    output_model_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Train quantile forecaster on multi-year dataset."""
    from energy_grid.forecasting import create_forecaster

    reader = EntsoeDatasetReader(data_dir)
    typer.echo(
        f"Loading {country_code} {target.value} dataset from {data_dir} "
        f"({start_year}-{end_year})..."
    )
    series = reader.load_series(
        target,
        country_code=country_code,
        zone_key=zone_key,
        start_year=start_year,
        end_year=end_year,
        resample_freq="15min" if target == EventType.DEMAND else "60min",
    )
    if series.empty:
        typer.echo(f"No series data found for {target.value}")
        raise typer.Exit(code=1)

    steps_per_day = 96 if target == EventType.DEMAND else 24
    typer.echo(f"Constructing multiscale features for {len(series):,} observations...")
    df = build_multiscale_features(series, steps_per_day=steps_per_day)
    feature_names = [col for col in df.columns if col != "target"]

    # Holdout validation (last 20%)
    holdout_size = int(len(df) * 0.20)
    train_df = df.iloc[:-holdout_size]
    val_df = df.iloc[-holdout_size:]

    typer.echo(
        f"Training {model_type.upper()} on {len(train_df):,} rows, "
        f"validating on {len(val_df):,} rows "
        f"({val_df.index.min().date()} to {val_df.index.max().date()})..."
    )
    model = create_forecaster(model_type)  # type: ignore[arg-type]
    model.fit(
        train_df,
        train_df["target"],
        feature_names=feature_names,
        calibration_fraction=0.15 if calibrate else 0.0,
    )

    raw_p10, raw_p50, raw_p90 = model.predict(val_df, apply_calibration=False)
    cal_p10, cal_p50, cal_p90 = model.predict(val_df, apply_calibration=True)

    actuals = val_df["target"].to_numpy()
    metrics = evaluate(
        actuals, cal_p50, cal_p10, cal_p90, allow_wape=target == EventType.DEMAND
    )
    base_7d = val_df["lag_7d"].to_numpy()
    b7d_metrics = evaluate(
        actuals, base_7d, base_7d, base_7d, allow_wape=target == EventType.DEMAND
    )

    uncal_cov = float(np.mean((actuals >= raw_p10) & (actuals <= raw_p90)))
    unit = "MW" if target == EventType.DEMAND else "EUR/MWh"

    typer.echo(f"--- Results for {target.value.upper()} ---")
    typer.echo(f"Model MAE: {metrics.mae:.3f} {unit} (RMSE: {metrics.rmse:.3f} {unit})")
    if metrics.wape is not None:
        typer.echo(f"Model WAPE: {metrics.wape * 100:.2f}%")
    typer.echo(f"7-day Baseline MAE: {b7d_metrics.mae:.3f} {unit}")
    typer.echo(
        f"MAE Improvement: {((b7d_metrics.mae - metrics.mae) / b7d_metrics.mae) * 100:+.1f}%"
    )
    typer.echo(f"Raw P10-P90 Coverage: {uncal_cov * 100:.1f}%")
    typer.echo(f"Calibrated P10-P90 Coverage: {metrics.interval_coverage * 100:.1f}%")
    typer.echo(f"Winkler Score: {metrics.winkler_score:.3f}")

    if output_model_dir:
        model.save(output_model_dir)
        typer.echo(f"Saved model artifacts to {output_model_dir}")


@app.command("backtest")
def backtest(
    data_dir: Annotated[Path, typer.Option()] = DEFAULT_DATASET_PATH,
    target: Annotated[EventType, typer.Option()] = EventType.DEMAND,
    country_code: Annotated[str, typer.Option()] = "ES",
    start_year: Annotated[int, typer.Option()] = 2023,
    end_year: Annotated[int, typer.Option()] = 2026,
    train_days: Annotated[int, typer.Option()] = 365,
    test_days: Annotated[int, typer.Option()] = 30,
    output_report: Annotated[Path | None, typer.Option()] = None,
    output_json: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Run multi-year rolling origin cross-validation and export slice report."""
    reader = EntsoeDatasetReader(data_dir)
    typer.echo(f"Loading {target.value} dataset from {data_dir} ({start_year}-{end_year})...")
    series = reader.load_series(
        target,
        country_code=country_code,
        start_year=start_year,
        end_year=end_year,
        resample_freq="15min" if target == EventType.DEMAND else "60min",
    )
    if series.empty:
        typer.echo(f"No series data found for {target.value}")
        raise typer.Exit(code=1)

    steps_per_day = 96 if target == EventType.DEMAND else 24
    typer.echo(f"Running rolling-origin backtest on {len(series):,} observations...")
    report = run_rolling_backtest(
        series,
        target=target,
        train_days=train_days,
        test_days=test_days,
        step_days=test_days,
        steps_per_day=steps_per_day,
    )

    typer.echo(report.to_markdown())

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(report.to_markdown(), encoding="utf-8")
        typer.echo(f"Wrote report to {output_report}")

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        import json

        output_json.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        typer.echo(f"Wrote JSON metrics to {output_json}")


@app.command("list-zones")
def list_zones(
    data_dir: Annotated[Path, typer.Option()] = DEFAULT_DATASET_PATH,
    target: Annotated[EventType, typer.Option()] = EventType.DEMAND,
) -> None:
    """List all European countries and bidding zones available in the dataset."""
    reader = EntsoeDatasetReader(data_dir)
    if not reader.exists():
        typer.echo(f"Dataset directory not found at {data_dir}")
        raise typer.Exit(code=1)

    zones = reader.list_available_zones(target)
    typer.echo(f"=== Available European Zones for {target.value.upper()} ({len(zones)} zones) ===")
    for z in zones:
        typer.echo(f"  {z['country_code']:5s} | {z['zone_key']:20s} | {z['zone_name']}")


@app.command("export-hf")
def export_hf(
    data_dir: Annotated[Path, typer.Option()] = DEFAULT_DATASET_PATH,
    target: Annotated[EventType, typer.Option()] = EventType.DEMAND,
    country_code: Annotated[str, typer.Option()] = "ES",
    zone_key: Annotated[str | None, typer.Option()] = None,
    model_type: Annotated[
        str, typer.Option(help="lightgbm, xgboost, catboost, stacked, dual_horizon")
    ] = "dual_horizon",
    start_year: Annotated[int, typer.Option()] = 2022,
    end_year: Annotated[int, typer.Option()] = 2026,
    output_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Train and export a production-ready model repository for Hugging Face."""
    from energy_grid.forecasting import create_forecaster

    reader = EntsoeDatasetReader(data_dir)
    target_name = f"{country_code.lower()}-{target.value}-forecaster"
    destination = output_dir or Path(f"hf_models/{target_name}")

    zone_label = zone_key or country_code
    typer.echo(
        f"Preparing Hugging Face model export for {country_code} ({zone_label}) "
        f"{target.value.upper()} [{model_type.upper()}]..."
    )
    series = reader.load_series(
        target,
        country_code=country_code,
        zone_key=zone_key,
        start_year=start_year,
        end_year=end_year,
        resample_freq="15min" if target == EventType.DEMAND else "60min",
    )
    steps_per_day = 96 if target == EventType.DEMAND else 24
    df = build_multiscale_features(series, steps_per_day=steps_per_day)
    feature_names = [col for col in df.columns if col != "target"]

    holdout_size = int(len(df) * 0.20)
    train_df = df.iloc[:-holdout_size]
    val_df = df.iloc[-holdout_size:]

    model = create_forecaster(model_type)  # type: ignore[arg-type]
    model.fit(
        train_df, train_df["target"], feature_names=feature_names, calibration_fraction=0.15
    )

    p10, p50, p90 = model.predict(val_df, apply_calibration=True)
    actuals = val_df["target"].to_numpy()
    metrics = evaluate(actuals, p50, p10, p90, allow_wape=target == EventType.DEMAND)
    base_7d = val_df["lag_7d"].to_numpy()
    b7d_metrics = evaluate(
        actuals, base_7d, base_7d, base_7d, allow_wape=target == EventType.DEMAND
    )

    export_model_to_hf_package(
        forecaster=model,
        target=target,
        output_dir=destination,
        metrics=metrics,
        baseline_mae=b7d_metrics.mae,
        training_rows=len(train_df),
        start_year=start_year,
        end_year=end_year,
        country_code=country_code,
        zone_key=zone_key,
        sample_df=val_df,
    )
    improvement = ((b7d_metrics.mae - metrics.mae) / b7d_metrics.mae) * 100
    typer.echo(f"Successfully exported Hugging Face model package to: {destination.resolve()}")
    typer.echo(f"Model MAE: {metrics.mae:.3f} (Improvement vs 7d: {improvement:+.1f}%)")
    typer.echo(f"Calibrated P10-P90 Coverage: {metrics.interval_coverage*100:.1f}%")



@app.command("upload-hf")
def upload_hf(
    model_dir: Annotated[Path, typer.Option(exists=True)],
    repo_id: Annotated[str, typer.Option(help="e.g. org/spanish-demand-forecaster")],
    token: Annotated[str | None, typer.Option(envvar="HF_TOKEN")] = None,
    private: Annotated[bool, typer.Option()] = False,
) -> None:
    """Upload a packaged model repository directly to the Hugging Face Hub."""
    if not token:
        typer.echo("Error: HF_TOKEN must be set or passed with --token")
        raise typer.Exit(code=1)

    typer.echo(f"Uploading {model_dir} to Hugging Face Hub: {repo_id}...")
    url = upload_model_to_hf(model_dir=model_dir, repo_id=repo_id, token=token, private=private)
    typer.echo(f"Upload complete! Model URL: {url}")


@app.command("launch-demo")
def launch_demo(
    data_dir: Annotated[Path, typer.Option()] = DEFAULT_DATASET_PATH,
    port: Annotated[int, typer.Option()] = 7860,
    share: Annotated[bool, typer.Option()] = False,
) -> None:
    """Launch the interactive Gradio web application."""
    from energy_grid.app_gradio import build_gradio_app

    demo_app = build_gradio_app(data_dir)
    typer.echo(f"Starting Gradio web demo on port {port}...")
    demo_app.launch(server_port=port, share=share)


if __name__ == "__main__":
    app()
