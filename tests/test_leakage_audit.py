from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from energy_grid.features import asof_features, build_multiscale_features, lag_features


def test_asof_features_leakage_rejection() -> None:
    """Verify that asof_features strictly rejects future-published measurements."""
    origins = pd.DataFrame(
        {
            "forecast_origin": [
                datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
                datetime(2025, 1, 1, 13, 0, tzinfo=UTC),
            ],
            "valid_time": [
                datetime(2025, 1, 1, 13, 0, tzinfo=UTC),
                datetime(2025, 1, 1, 14, 0, tzinfo=UTC),
            ],
        }
    )

    measurements = pd.DataFrame(
        {
            "published_at": [
                datetime(2025, 1, 1, 11, 45, tzinfo=UTC),
                datetime(2025, 1, 1, 12, 30, tzinfo=UTC),
            ],
            "interval_start": [
                datetime(2025, 1, 1, 11, 0, tzinfo=UTC),
                datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
            ],
            "value": [25000.0, 26000.0],
        }
    )

    res = asof_features(measurements, origins)
    assert len(res) == 2
    # First origin (12:00) must see publication from 11:45 (25000.0), NOT 12:30
    assert res.iloc[0]["value"] == 25000.0
    assert res.iloc[0]["published_at"] <= res.iloc[0]["forecast_origin"]
    # Second origin (13:00) sees publication from 12:30 (26000.0)
    assert res.iloc[1]["value"] == 26000.0
    assert res.iloc[1]["published_at"] <= res.iloc[1]["forecast_origin"]


def test_lag_features_strict_causality() -> None:
    """Modifying future values at t + k must NEVER alter feature vectors at time t."""
    n_steps = 2500
    times = pd.date_range("2025-01-01", periods=n_steps, freq="15min", tz="UTC")
    values_base = 20000.0 + 5000.0 * np.sin(np.linspace(0, 50, n_steps))

    series_base = pd.Series(values_base, index=times)
    features_base = build_multiscale_features(series_base, steps_per_day=96)

    # Check for arbitrary historical timestamp t_eval in valid range (>14 days)
    t_eval = times[1800]

    # Create modified series where ALL values AFTER t_eval are perturbed (+1,000,000 MW)
    values_perturbed = values_base.copy()
    future_idx = np.where(times > t_eval)[0]
    values_perturbed[future_idx] += np.random.normal(1_000_000, 50_000, size=len(future_idx))

    series_perturbed = pd.Series(values_perturbed, index=times)
    features_perturbed = build_multiscale_features(series_perturbed, steps_per_day=96)

    # Verify that every feature at time t_eval is IDENTICAL to the bit level
    assert t_eval in features_base.index
    assert t_eval in features_perturbed.index

    row_base = features_base.loc[t_eval].drop("target")
    row_perturbed = features_perturbed.loc[t_eval].drop("target")

    pd.testing.assert_series_equal(
        row_base,
        row_perturbed,
        check_exact=True,
        obj="Causality check: future values leaked into past features!",
    )


def test_rolling_window_excludes_current_step() -> None:
    """Verify that rolling mean and standard deviation strictly exclude current step t."""
    times = pd.date_range("2025-01-01", periods=200, freq="15min", tz="UTC")
    values = np.zeros(200)
    # Put a massive spike at step 100
    values[100] = 999999.0

    df = pd.DataFrame({"valid_time": times, "target": values})
    df_lags = lag_features(df, value_column="target", steps_per_day=96)

    # At step 100, rolling_mean_24h and rolling_std_4step MUST NOT include the spike at step 100
    assert df_lags.loc[100, "rolling_mean_24h"] == 0.0
    assert df_lags.loc[100, "rolling_std_4step"] == 0.0

    # Only at step 101 should the spike be visible in rolling statistics
    assert df_lags.loc[101, "rolling_std_4step"] > 0.0
