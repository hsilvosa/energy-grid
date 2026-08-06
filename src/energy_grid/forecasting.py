from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd

from energy_grid.domain import MADRID, EventType, ForecastRecord
from energy_grid.time_utils import delivery_intervals, floor_quarter_hour


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    wape: float | None
    pinball_p10: float
    pinball_p90: float
    interval_coverage: float


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def evaluate(
    actual: np.ndarray,
    point: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    *,
    allow_wape: bool = True,
) -> ForecastMetrics:
    if not (len(actual) == len(point) == len(p10) == len(p90)) or not len(actual):
        raise ValueError("metric arrays must be non-empty and have equal length")
    error = actual - point
    denominator = float(np.sum(np.abs(actual)))
    wape = float(np.sum(np.abs(error)) / denominator) if allow_wape and denominator else None
    return ForecastMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(math.sqrt(np.mean(np.square(error)))),
        wape=wape,
        pinball_p10=pinball_loss(actual, p10, 0.1),
        pinball_p90=pinball_loss(actual, p90, 0.9),
        interval_coverage=float(np.mean((actual >= p10) & (actual <= p90))),
    )


class QuantileForecaster(Protocol):
    model_name: str
    model_version: str

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class LightGBMQuantileForecaster:
    model_name = "lightgbm-global-horizon"

    def __init__(self, model_version: str = "untrained", random_state: int = 42) -> None:
        self.model_version = model_version
        self.random_state = random_state
        self.feature_names: list[str] = []
        self.models: dict[float, Any] = {}

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        feature_names: list[str],
    ) -> LightGBMQuantileForecaster:
        from lightgbm import LGBMRegressor

        self.feature_names = feature_names
        for quantile in (0.1, 0.5, 0.9):
            model = LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=160,
                learning_rate=0.05,
                num_leaves=31,
                random_state=self.random_state,
                verbosity=-1,
            )
            model.fit(features[feature_names], target)
            self.models[quantile] = model
        self.model_version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return self

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if set(self.models) != {0.1, 0.5, 0.9}:
            raise RuntimeError("forecaster is not trained")
        predictions = [
            self.models[q].predict(features[self.feature_names]) for q in (0.1, 0.5, 0.9)
        ]
        stacked = np.sort(np.vstack(predictions), axis=0)
        return stacked[0], stacked[1], stacked[2]


class SarimaxBaseline:
    model_name = "sarimax"

    def __init__(self, seasonal_period: int = 96) -> None:
        self.seasonal_period = seasonal_period
        self.result: Any | None = None

    def fit(self, values: pd.Series) -> SarimaxBaseline:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.result = SARIMAX(
            values.astype(float),
            order=(1, 0, 1),
            seasonal_order=(1, 0, 0, self.seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=50)
        return self

    def predict(self, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.result is None:
            raise RuntimeError("baseline is not trained")
        forecast = self.result.get_forecast(steps=steps)
        mean = np.asarray(forecast.predicted_mean)
        interval = np.asarray(forecast.conf_int(alpha=0.2))
        return interval[:, 0], mean, interval[:, 1]


def synthetic_training_data(days: int = 35, seed: int = 42) -> pd.DataFrame:
    """Deterministic fixture expansion used only by the credential-free demo."""
    rng = np.random.default_rng(seed)
    periods = days * 96
    times = pd.date_range("2025-01-01", periods=periods, freq="15min", tz="UTC")
    local = times.tz_convert("Europe/Madrid")
    daily = np.sin(2 * np.pi * (local.hour * 4 + local.minute // 15) / 96 - 1.2)
    weekly = np.where(local.dayofweek < 5, 1.0, 0.91)
    temperature = 11 + 7 * np.sin(2 * np.pi * np.arange(periods) / 96 - 1.5)
    demand = 25500 + 4300 * daily + 1800 * weekly + 140 * np.abs(18 - temperature)
    demand += rng.normal(0, 280, periods)
    price = 35 + demand / 1100 - 0.65 * temperature + rng.normal(0, 5, periods)
    return pd.DataFrame(
        {
            "valid_time": times,
            "hour": local.hour,
            "quarter": local.minute // 15,
            "day_of_week": local.dayofweek,
            "month": local.month,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "temperature": temperature,
            "horizon_step": np.tile(np.arange(1, 97), days),
            "demand": demand,
            "price": price,
        }
    )


def materialize_demo_forecasts(
    target: EventType,
    *,
    issue_time: datetime | None = None,
    delivery_date: date | None = None,
) -> list[ForecastRecord]:
    frame = synthetic_training_data()
    feature_names = [
        "hour",
        "quarter",
        "day_of_week",
        "month",
        "is_weekend",
        "temperature",
        "horizon_step",
    ]
    target_column = "demand" if target == EventType.DEMAND else "price"
    model = LightGBMQuantileForecaster().fit(
        frame.iloc[:-96], frame[target_column].iloc[:-96], feature_names=feature_names
    )
    issue = floor_quarter_hour(issue_time or datetime.now(UTC))

    if target == EventType.DEMAND:
        intervals = [
            (issue + timedelta(minutes=15 * step), issue + timedelta(minutes=15 * (step + 1)))
            for step in range(1, 25)
        ]
    else:
        local_date = delivery_date or (issue.astimezone(MADRID).date() + timedelta(days=1))
        intervals = delivery_intervals(local_date)

    rows: list[dict[str, float | int]] = []
    for step, (valid_from, _) in enumerate(intervals, start=1):
        local = valid_from.astimezone(MADRID)
        rows.append(
            {
                "hour": local.hour,
                "quarter": local.minute // 15,
                "day_of_week": local.weekday(),
                "month": local.month,
                "is_weekend": int(local.weekday() >= 5),
                "temperature": 15.0,
                "horizon_step": step,
            }
        )
    prediction_frame = pd.DataFrame(rows)
    p10, p50, p90 = model.predict(prediction_frame)
    unit = "MW" if target == EventType.DEMAND else "EUR/MWh"
    return [
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
            data_snapshot_id="fixture-snapshot-v1",
            quality_flags=["fixture_data"],
        )
        for index, (start, end) in enumerate(intervals)
    ]
