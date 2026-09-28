from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from energy_grid.domain import MADRID

MARKET_TIMEZONES = {
    "ES": "Europe/Madrid", "PT": "Europe/Lisbon", "FR": "Europe/Paris",
    "DE": "Europe/Berlin", "IT": "Europe/Rome", "NL": "Europe/Amsterdam",
    "BE": "Europe/Brussels", "PL": "Europe/Warsaw", "AT": "Europe/Vienna",
    "NO": "Europe/Oslo", "SE": "Europe/Stockholm", "GR": "Europe/Athens",
    "CH": "Europe/Zurich", "GB": "Europe/London", "FI": "Europe/Helsinki",
    "AL": "Europe/Tirane", "BA": "Europe/Sarajevo", "BG": "Europe/Sofia",
    "CY": "Asia/Nicosia", "CZ": "Europe/Prague", "DK": "Europe/Copenhagen",
    "EE": "Europe/Tallinn", "HR": "Europe/Zagreb", "HU": "Europe/Budapest",
    "IE": "Europe/Dublin", "LT": "Europe/Vilnius", "LU": "Europe/Luxembourg",
    "LV": "Europe/Riga", "ME": "Europe/Podgorica", "MK": "Europe/Skopje",
    "RO": "Europe/Bucharest", "RS": "Europe/Belgrade", "SI": "Europe/Ljubljana",
    "SK": "Europe/Bratislava", "XK": "Europe/Belgrade",
}

FEATURE_COLUMNS_CORE = [
    "hour",
    "quarter",
    "day_of_week",
    "day_of_year",
    "month",
    "is_weekend",
    "is_holiday",
    "is_morning_peak",
    "is_evening_peak",
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
    "sin_day_of_year",
    "cos_day_of_year",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_24h",
    "lag_48h",
    "lag_7d",
    "lag_14d",
    "diff_1h",
    "diff_24h",
    "diff_7d",
    "acceleration_1h",
    "ema_4step",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_min_24h",
    "rolling_max_24h",
    "rolling_mean_7d",
    "rolling_std_7d",
]

SHORT_HORIZON_FEATURES = [
    "hour",
    "quarter",
    "day_of_week",
    "sin_hour",
    "cos_hour",
    "is_morning_peak",
    "is_evening_peak",
    "lag_1h",
    "lag_2h",
    "lag_3h",
    "lag_4h",
    "diff_1h",
    "diff_2h",
    "acceleration_1h",
    "ema_4step",
    "ema_12step",
    "rolling_std_4step",
    "rolling_mean_24h",
]

LONG_HORIZON_FEATURES = [
    "hour",
    "quarter",
    "day_of_week",
    "day_of_year",
    "month",
    "is_weekend",
    "is_holiday",
    "is_morning_peak",
    "is_evening_peak",
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
    "sin_day_of_year",
    "cos_day_of_year",
    "lag_24h",
    "lag_48h",
    "lag_7d",
    "lag_14d",
    "diff_24h",
    "diff_7d",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_min_24h",
    "rolling_max_24h",
    "rolling_mean_7d",
    "rolling_std_7d",
]


def is_spanish_holiday(dates: pd.DatetimeIndex | pd.Series) -> pd.Series:
    """Return boolean Series indicating fixed/standard Spanish national holidays."""
    if isinstance(dates, pd.Series):
        local_dates = dates.dt.tz_convert(MADRID) if dates.dt.tz is not None else dates
        months = local_dates.dt.month
        days = local_dates.dt.day
        idx = dates.index
    else:
        local_dates = dates.tz_convert(MADRID) if dates.tz is not None else dates
        months = local_dates.month
        days = local_dates.day
        idx = dates

    fixed_holiday = (
        ((months == 1) & (days == 1))
        | ((months == 1) & (days == 6))
        | ((months == 5) & (days == 1))
        | ((months == 8) & (days == 15))
        | ((months == 10) & (days == 12))
        | ((months == 11) & (days == 1))
        | ((months == 12) & (days == 6))
        | ((months == 12) & (days == 8))
        | ((months == 12) & (days == 25))
    )
    return pd.Series(fixed_holiday.astype(int), index=idx, name="is_holiday")


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
    # A global fallback would use values published after some forecast origins.
    prior = right[["published_at", value_column]].dropna(subset=[value_column]).copy()
    prior = prior.sort_values("published_at")
    prior["prior_median"] = prior[value_column].expanding().median()
    fallback = pd.merge_asof(
        left[["forecast_origin"]],
        prior[["published_at", "prior_median"]],
        left_on="forecast_origin", right_on="published_at", direction="backward",
    )["prior_median"]
    result[value_column] = result[value_column].fillna(fallback)
    result["imputation_method"] = np.where(
        ~result["is_missing"], "none",
        np.where(result[value_column].notna(), "prior_median", "unavailable"),
    )
    return result


def calendar_features(
    frame: pd.DataFrame, column: str = "valid_time", *, country_code: str = "ES"
) -> pd.DataFrame:
    """Derive calendar, peak, and cyclical time features from datetime column."""
    result = frame.copy()
    try:
        timezone = ZoneInfo(MARKET_TIMEZONES[country_code.upper()])
    except KeyError as exc:
        raise ValueError(f"unsupported country_code: {country_code}") from exc
    timestamps = pd.to_datetime(result[column], utc=True).dt.tz_convert(timezone)
    result["hour"] = timestamps.dt.hour
    result["quarter"] = timestamps.dt.minute // 15
    result["day_of_week"] = timestamps.dt.dayofweek
    result["day_of_year"] = timestamps.dt.dayofyear
    result["month"] = timestamps.dt.month
    result["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)
    result["is_holiday"] = (
        is_spanish_holiday(timestamps.dt.tz_convert("UTC")).values
        if country_code.upper() == "ES" else 0
    )

    # Peak hour indicators (Morning: 8-11h, Evening: 19-22h)
    result["is_morning_peak"] = ((result["hour"] >= 8) & (result["hour"] <= 11)).astype(int)
    result["is_evening_peak"] = ((result["hour"] >= 19) & (result["hour"] <= 22)).astype(int)

    # Cyclical sin/cos encodings
    result["sin_hour"] = np.sin(2 * np.pi * result["hour"] / 24.0)
    result["cos_hour"] = np.cos(2 * np.pi * result["hour"] / 24.0)
    result["sin_day_of_week"] = np.sin(2 * np.pi * result["day_of_week"] / 7.0)
    result["cos_day_of_week"] = np.cos(2 * np.pi * result["day_of_week"] / 7.0)
    result["sin_day_of_year"] = np.sin(2 * np.pi * result["day_of_year"] / 365.25)
    result["cos_day_of_year"] = np.cos(2 * np.pi * result["day_of_year"] / 365.25)
    return result


def lag_features(
    frame: pd.DataFrame,
    value_column: str,
    *,
    steps_per_day: int = 96,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute autoregressive multi-scale lags and differences."""
    result = frame.copy()
    step_1h = steps_per_day // 24

    # High frequency intraday lags
    result["lag_1h"] = result[value_column].shift(step_1h)
    result["lag_2h"] = result[value_column].shift(step_1h * 2)
    result["lag_3h"] = result[value_column].shift(step_1h * 3)
    result["lag_4h"] = result[value_column].shift(step_1h * 4)

    # Diurnal & weekly lags
    result["lag_24h"] = result[value_column].shift(steps_per_day)
    result["lag_48h"] = result[value_column].shift(steps_per_day * 2)
    result["lag_7d"] = result[value_column].shift(steps_per_day * 7)
    result["lag_14d"] = result[value_column].shift(steps_per_day * 14)

    # Autoregressive differences & momentum
    result["diff_1h"] = result["lag_1h"] - result["lag_2h"]
    result["diff_2h"] = result["lag_1h"] - result["lag_3h"]
    result["diff_24h"] = result["lag_24h"] - result["lag_48h"]
    result["diff_7d"] = result["lag_7d"] - result["lag_14d"]
    result["acceleration_1h"] = result["diff_1h"] - (result["lag_2h"] - result["lag_3h"])

    # Exponential moving averages & spread
    result["ema_4step"] = result[value_column].shift(1).ewm(span=4).mean()
    result["ema_12step"] = result[value_column].shift(1).ewm(span=12).mean()
    result["ema_spread"] = result["ema_4step"] - result["ema_12step"]

    # Rolling statistics strictly shifted by 1 step to avoid leakage
    shifted = result[value_column].shift(1)
    result["rolling_std_4step"] = shifted.rolling(step_1h).std().fillna(0)
    result["rolling_mean_24h"] = shifted.rolling(steps_per_day).mean()
    result["rolling_std_24h"] = shifted.rolling(steps_per_day).std()
    result["rolling_min_24h"] = shifted.rolling(steps_per_day).min()
    result["rolling_max_24h"] = shifted.rolling(steps_per_day).max()

    result["rolling_mean_7d"] = shifted.rolling(steps_per_day * 7).mean()
    result["rolling_std_7d"] = shifted.rolling(steps_per_day * 7).std()

    # Relative diurnal position and volatility ratio
    result["ratio_24h_mean"] = result["lag_1h"] / (result["rolling_mean_24h"] + 1e-3)
    result["volatility_24h"] = (
        result["rolling_max_24h"] - result["rolling_min_24h"]
    ) / (result["rolling_mean_24h"] + 1e-3)
    return result


def build_multiscale_features(
    series: pd.Series,
    *,
    steps_per_day: int = 96,
    additional_exog: pd.DataFrame | None = None,
    exogenous_series: pd.Series | None = None,
    country_code: str = "ES",
) -> pd.DataFrame:
    """Build unified multi-scale feature matrix from continuous time-series."""
    df = pd.DataFrame({"valid_time": series.index, "target": series.values})
    df = calendar_features(df, column="valid_time", country_code=country_code)
    df = lag_features(df, value_column="target", steps_per_day=steps_per_day)

    if exogenous_series is not None:
        exog_df = pd.DataFrame(
            {
                "valid_time": exogenous_series.index,
                "exog_lag_24h": exogenous_series.shift(steps_per_day),
            }
        )
        df = pd.merge(df, exog_df, on="valid_time", how="left")

    if additional_exog is not None and not additional_exog.empty:
        df = df.set_index("valid_time").join(additional_exog, how="left").reset_index()

    df = df.set_index("valid_time")
    # Drop rows where long-term lags are NaN
    df = df.dropna()
    df["horizon_step"] = 1
    return df


def build_horizon_features(
    series: pd.Series, *, horizon_steps: int, steps_per_day: int = 96,
    country_code: str = "ES",
) -> pd.DataFrame:
    """Create features for a valid time using observations available at its forecast origin.

    The origin is ``horizon_steps`` intervals before the target. Historical
    statistics are shifted so their newest observation is at or before the origin.
    """
    if horizon_steps < 1 or steps_per_day < 24 or steps_per_day % 24:
        raise ValueError("horizon_steps must be positive and steps_per_day divisible by 24")
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is None:
        raise ValueError("series must have a timezone-aware DatetimeIndex")
    if series.isna().any():
        raise ValueError("series contains missing observations; repair gaps before backtesting")
    expected = pd.Timedelta(days=1) / steps_per_day
    if not series.index.is_monotonic_increasing or not series.index.is_unique or not (
        series.index.to_series().diff().iloc[1:] == expected
    ).all():
        raise ValueError("series must have a regular, unique time index")

    frame = pd.DataFrame({"valid_time": series.index, "target": series.to_numpy()})
    calendar = calendar_features(frame[["valid_time"]], country_code=country_code)
    historical = lag_features(frame, "target", steps_per_day=steps_per_day)
    historical = historical.drop(columns=["valid_time", "target"]).shift(horizon_steps - 1)
    result = pd.concat([calendar, historical], axis=1).set_index("valid_time")
    result["horizon_step"] = horizon_steps
    result["target"] = series.to_numpy()
    return result.dropna()


def build_short_horizon_features(
    series: pd.Series,
    steps_per_day: int = 96,
) -> pd.DataFrame:
    """Build feature matrix tailored for short-term (1-6h) intraday forecasting."""
    full_df = build_multiscale_features(series, steps_per_day=steps_per_day)
    features = [c for c in SHORT_HORIZON_FEATURES if c in full_df.columns]
    return full_df[features + ["target"]]


def build_long_horizon_features(
    series: pd.Series,
    steps_per_day: int = 96,
) -> pd.DataFrame:
    """Build feature matrix tailored for long-term (24-48h) day-ahead forecasting."""
    full_df = build_multiscale_features(series, steps_per_day=steps_per_day)
    features = [c for c in LONG_HORIZON_FEATURES if c in full_df.columns]
    return full_df[features + ["target"]]
