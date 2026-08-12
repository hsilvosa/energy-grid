from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from energy_grid.domain import MADRID, EventType, ForecastRecord, GridEvent
from energy_grid.forecasting import ForecastMetrics, LightGBMQuantileForecaster, evaluate
from energy_grid.time_utils import delivery_intervals, floor_quarter_hour

LIVE_FEATURES = [
    "hour",
    "quarter",
    "day_of_week",
    "month",
    "is_weekend",
    "temperature_2m",
    "wind_speed_10m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "lag_96",
    "lag_672",
    "rolling_96",
]


@dataclass(frozen=True)
class TrainingResult:
    target: EventType
    model: LightGBMQuantileForecaster
    forecasts: list[ForecastRecord]
    metrics: ForecastMetrics
    baseline_mae: float
    training_rows: int
    training_start: datetime
    training_end: datetime
    snapshot_id: str


def events_to_series(events: list[GridEvent], target: EventType) -> pd.Series:
    """Convert source intervals to a revision-aware 15-minute UTC series."""
    rows: list[tuple[datetime, float, int, datetime]] = []
    for event in events:
        if event.event_type != target or event.value is None:
            continue
        cursor = event.interval_start.astimezone(UTC)
        end = event.interval_end.astimezone(UTC)
        while cursor < end:
            rows.append((cursor, float(event.value), event.revision, event.ingested_at))
            cursor += timedelta(minutes=15)
    if not rows:
        raise ValueError(f"no usable {target.value} events were downloaded")
    frame = pd.DataFrame(rows, columns=["timestamp", "value", "revision", "ingested_at"])
    frame = frame.sort_values(["timestamp", "revision", "ingested_at"])
    frame = frame.drop_duplicates("timestamp", keep="last")
    series = frame.set_index("timestamp")["value"].sort_index()
    series.index = pd.DatetimeIndex(series.index).tz_convert("UTC")
    return series.astype(float)


def weather_to_frame(events: list[GridEvent]) -> pd.DataFrame:
    rows: list[tuple[datetime, str, float]] = []
    for event in events:
        if event.event_type != EventType.WEATHER or event.value is None or not event.dimension:
            continue
        cursor = event.interval_start.astimezone(UTC)
        end = event.interval_end.astimezone(UTC)
        while cursor < end:
            rows.append((cursor, event.dimension, float(event.value)))
            cursor += timedelta(minutes=15)
    if not rows:
        raise ValueError("no usable weather events were downloaded")
    frame = pd.DataFrame(rows, columns=["timestamp", "dimension", "value"])
    result = frame.pivot_table(index="timestamp", columns="dimension", values="value")
    result.index = pd.DatetimeIndex(result.index).tz_convert("UTC")
    return result.sort_index()


def _calendar(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    local = result.index.tz_convert(MADRID)
    result["hour"] = local.hour
    result["quarter"] = local.minute // 15
    result["day_of_week"] = local.dayofweek
    result["month"] = local.month
    result["is_weekend"] = (local.dayofweek >= 5).astype(int)
    return result


def build_training_frame(
    target_events: list[GridEvent],
    weather_events: list[GridEvent],
    target: EventType,
) -> pd.DataFrame:
    target_series = events_to_series(target_events, target)
    weather = weather_to_frame(weather_events)
    frame = pd.DataFrame({"target": target_series}).join(weather, how="left")
    weather_columns = [
        "temperature_2m",
        "wind_speed_10m",
        "relative_humidity_2m",
        "shortwave_radiation",
    ]
    for column in weather_columns:
        if column not in frame:
            frame[column] = np.nan
    frame[weather_columns] = frame[weather_columns].interpolate(limit_direction="both")
    frame = _calendar(frame)
    frame["lag_96"] = frame["target"].shift(96)
    frame["lag_672"] = frame["target"].shift(672)
    frame["rolling_96"] = frame["target"].shift(1).rolling(96, min_periods=96).mean()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=["target", *LIVE_FEATURES])


def _value_at_or_before(series: pd.Series, timestamp: datetime) -> float:
    eligible = series.loc[:pd.Timestamp(timestamp)]
    if eligible.empty:
        return float(series.iloc[0])
    return float(eligible.iloc[-1])


def build_future_frame(
    intervals: list[tuple[datetime, datetime]],
    target_history: pd.Series,
    live_weather_events: list[GridEvent],
) -> pd.DataFrame:
    weather = weather_to_frame(live_weather_events)
    index = pd.DatetimeIndex([start for start, _ in intervals]).tz_convert("UTC")
    expanded_weather = weather.reindex(weather.index.union(index)).sort_index().ffill().bfill()
    frame = expanded_weather.reindex(index)
    for column in (
        "temperature_2m",
        "wind_speed_10m",
        "relative_humidity_2m",
        "shortwave_radiation",
    ):
        if column not in frame:
            frame[column] = 0.0
        frame[column] = frame[column].fillna(float(weather[column].median()))
    frame = _calendar(frame)
    recent_mean = float(target_history.tail(96).mean())
    frame["lag_96"] = [
        _value_at_or_before(target_history, timestamp.to_pydatetime() - timedelta(days=1))
        for timestamp in index
    ]
    frame["lag_672"] = [
        _value_at_or_before(target_history, timestamp.to_pydatetime() - timedelta(days=7))
        for timestamp in index
    ]
    frame["rolling_96"] = recent_mean
    return frame


def source_snapshot_id(*event_groups: list[GridEvent]) -> str:
    checksums = sorted(
        {
            event.payload_checksum
            for group in event_groups
            for event in group
            if event.payload_checksum
        }
    )
    raw = "|".join(checksums).encode()
    return f"live-{hashlib.sha256(raw).hexdigest()[:20]}"


def _forecast_intervals(target: EventType, issue_time: datetime) -> list[tuple[datetime, datetime]]:
    if target == EventType.DEMAND:
        return [
            (
                issue_time + timedelta(minutes=15 * step),
                issue_time + timedelta(minutes=15 * (step + 1)),
            )
            for step in range(1, 25)
        ]
    delivery_date = issue_time.astimezone(MADRID).date() + timedelta(days=1)
    return delivery_intervals(delivery_date)


def train_and_forecast(
    target_events: list[GridEvent],
    historical_weather: list[GridEvent],
    live_weather: list[GridEvent],
    target: EventType,
    *,
    issue_time: datetime | None = None,
) -> TrainingResult:
    if target not in (EventType.DEMAND, EventType.PRICE):
        raise ValueError("live forecasting supports demand and price")
    frame = build_training_frame(target_events, historical_weather, target)
    minimum_rows = 96 * 5
    if len(frame) < minimum_rows:
        raise ValueError(
            f"not enough complete {target.value} training rows: {len(frame)}; "
            f"need at least {minimum_rows}"
        )
    holdout_size = min(96 * 3, max(96, len(frame) // 5))
    training = frame.iloc[:-holdout_size]
    holdout = frame.iloc[-holdout_size:]
    validation_model = LightGBMQuantileForecaster().fit(
        training, training["target"], feature_names=LIVE_FEATURES
    )
    p10, p50, p90 = validation_model.predict(holdout)
    metrics = evaluate(
        holdout["target"].to_numpy(),
        p50,
        p10,
        p90,
        allow_wape=target == EventType.DEMAND,
    )
    baseline = holdout["lag_672"].to_numpy()
    baseline_mae = float(np.mean(np.abs(holdout["target"].to_numpy() - baseline)))

    model = LightGBMQuantileForecaster().fit(frame, frame["target"], feature_names=LIVE_FEATURES)
    issue = floor_quarter_hour(issue_time or datetime.now(UTC))
    intervals = _forecast_intervals(target, issue)
    target_history = events_to_series(target_events, target)
    future = build_future_frame(intervals, target_history, live_weather)
    p10, p50, p90 = model.predict(future)
    snapshot_id = source_snapshot_id(target_events, historical_weather, live_weather)
    unit = "MW" if target == EventType.DEMAND else "EUR/MWh"
    forecasts = [
        ForecastRecord(
            target=target,
            issue_time=issue,
            valid_from=start,
            valid_to=end,
            point=float(p50[index]),
            p10=float(p10[index]),
            p50=float(p50[index]),
            p90=float(p90[index]),
            unit=unit,
            model_name=model.model_name,
            model_version=model.model_version,
            data_snapshot_id=snapshot_id,
            quality_flags=[
                "real_source_data",
                "entsoe_actuals",
                "open_meteo_historical_forecast",
            ],
        )
        for index, (start, end) in enumerate(intervals)
    ]
    return TrainingResult(
        target=target,
        model=model,
        forecasts=forecasts,
        metrics=metrics,
        baseline_mae=baseline_mae,
        training_rows=len(frame),
        training_start=frame.index.min().to_pydatetime(),
        training_end=frame.index.max().to_pydatetime(),
        snapshot_id=snapshot_id,
    )


def log_training_result(result: TrainingResult, tracking_uri: str) -> str:
    """Log real-data lineage, metrics and the three LightGBM quantile models to MLflow."""
    import mlflow
    import mlflow.lightgbm

    mlflow.set_tracking_uri(tracking_uri)
    run_name = f"live-{result.target.value}-{result.model.model_version}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "target": result.target.value,
                "training_rows": result.training_rows,
                "training_start": result.training_start.isoformat(),
                "training_end": result.training_end.isoformat(),
                "feature_count": len(LIVE_FEATURES),
                "source": "ENTSO-E + Open-Meteo",
            }
        )
        metric_values = {
            key: value
            for key, value in asdict(result.metrics).items()
            if value is not None
        }
        mlflow.log_metrics({**metric_values, "baseline_mae": result.baseline_mae})
        mlflow.set_tags(
            {
                "data_snapshot_id": result.snapshot_id,
                "data_class": "real_source_data",
                "area": "10YES-REE------0",
            }
        )
        for quantile, model in result.model.models.items():
            mlflow.lightgbm.log_model(model, name=f"model_q{int(quantile * 100)}")
        return str(run.info.run_id)
