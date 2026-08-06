from __future__ import annotations

from datetime import datetime

import pandas as pd


def asof_features(
    measurements: pd.DataFrame,
    origins: pd.DataFrame,
    *,
    value_column: str = "value",
) -> pd.DataFrame:
    """Build leakage-safe features using publication time rather than observation time."""
    required_measurements = {"published_at", "interval_start", value_column}
    required_origins = {"forecast_origin", "valid_time"}
    if missing := required_measurements - set(measurements.columns):
        raise ValueError(f"measurements missing columns: {sorted(missing)}")
    if missing := required_origins - set(origins.columns):
        raise ValueError(f"origins missing columns: {sorted(missing)}")

    left = origins.copy().sort_values("forecast_origin")
    right = measurements.copy().sort_values("published_at")
    result = pd.merge_asof(
        left,
        right,
        left_on="forecast_origin",
        right_on="published_at",
        direction="backward",
        allow_exact_matches=True,
    )
    if (result["published_at"] > result["forecast_origin"]).fillna(False).any():
        raise AssertionError("future-published values leaked into features")
    result["is_missing"] = result[value_column].isna()
    result["imputation_method"] = result["is_missing"].map(
        {True: "seasonal_median", False: "none"}
    )
    result[value_column] = result[value_column].fillna(
        right[value_column].tail(96 * 7).median()
    )
    return result


def calendar_features(frame: pd.DataFrame, column: str = "valid_time") -> pd.DataFrame:
    result = frame.copy()
    timestamps = pd.to_datetime(result[column], utc=True).dt.tz_convert("Europe/Madrid")
    result["hour"] = timestamps.dt.hour
    result["quarter"] = timestamps.dt.minute // 15
    result["day_of_week"] = timestamps.dt.dayofweek
    result["month"] = timestamps.dt.month
    result["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)
    return result


def validate_asof(publication_time: datetime, forecast_origin: datetime) -> None:
    if publication_time > forecast_origin:
        raise ValueError("feature publication time is after forecast origin")

