from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from energy_grid.backtest import run_rolling_backtest
from energy_grid.domain import EventType
from energy_grid.features import (
    asof_features,
    build_horizon_features,
    build_multiscale_features,
    calendar_features,
    lag_features,
)
from energy_grid.forecasting import DualHorizonEnsembleForecaster, StackedQuantileEnsemble


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


@pytest.mark.parametrize("lead", [4, 96, 192])
def test_horizon_features_ignore_observations_after_origin(lead: int) -> None:
    times = pd.date_range("2025-01-01", periods=2500, freq="15min", tz="UTC")
    values = np.arange(len(times), dtype=float)
    valid_pos = 2000
    base = build_horizon_features(pd.Series(values, index=times), horizon_steps=lead)
    changed = values.copy()
    changed[valid_pos - lead + 1 : valid_pos] += 100_000
    perturbed = build_horizon_features(pd.Series(changed, index=times), horizon_steps=lead)
    pd.testing.assert_series_equal(base.loc[times[valid_pos]], perturbed.loc[times[valid_pos]])


def test_asof_missing_value_has_no_future_fallback() -> None:
    origin = pd.Timestamp("2025-01-01T12:00:00Z")
    measurements = pd.DataFrame({
        "published_at": [origin + pd.Timedelta(hours=1)],
        "interval_start": [origin], "value": [999.0],
    })
    origins = pd.DataFrame({"forecast_origin": [origin], "valid_time": [origin]})
    result = asof_features(measurements, origins)
    assert pd.isna(result.loc[0, "value"])
    assert result.loc[0, "imputation_method"] == "unavailable"


def test_calendar_uses_market_timezone_and_no_spanish_holiday_abroad() -> None:
    frame = pd.DataFrame({"valid_time": [pd.Timestamp("2025-01-06T00:00:00Z")]})
    spanish = calendar_features(frame, country_code="ES")
    portuguese = calendar_features(frame, country_code="PT")
    assert spanish.loc[0, "hour"] == 1
    assert portuguese.loc[0, "hour"] == 0
    assert spanish.loc[0, "is_holiday"] == 1
    assert portuguese.loc[0, "is_holiday"] == 0


def test_calendar_handles_daylight_saving_transition() -> None:
    frame = pd.DataFrame({"valid_time": pd.to_datetime([
        "2025-03-30T00:30:00Z", "2025-03-30T01:30:00Z",
    ])})
    spanish = calendar_features(frame, country_code="ES")
    assert spanish["hour"].tolist() == [1, 3]


def test_dual_horizon_prediction_is_independent_of_batch_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DualHorizonEnsembleForecaster(switch_horizon_steps=24)

    def short_predict(frame: pd.DataFrame, *, apply_calibration: bool = True) -> tuple:
        n = len(frame)
        return np.full(n, 8.0), np.full(n, 10.0), np.full(n, 12.0)

    def long_predict(frame: pd.DataFrame, *, apply_calibration: bool = True) -> tuple:
        n = len(frame)
        return np.full(n, 18.0), np.full(n, 20.0), np.full(n, 22.0)

    monkeypatch.setattr(model.short_model, "predict", short_predict)
    monkeypatch.setattr(model.long_model, "predict", long_predict)
    frame = pd.DataFrame({"horizon_step": [24, 1, 48]})
    batch = model.predict(frame)
    single = model.predict(frame.iloc[[0]])
    assert batch[1][0] == single[1][0]
    assert batch[1][0] > batch[1][1]


def test_backtest_purges_labels_after_first_forecast_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import energy_grid.backtest as backtest_module

    class FakeForecaster:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit(
            self, features: pd.DataFrame, target: pd.Series, **kwargs: object
        ) -> FakeForecaster:
            return self

        def predict(self, features: pd.DataFrame, **kwargs: object) -> tuple:
            point = features["lag_24h"].to_numpy()
            return point - 1, point, point + 1

    monkeypatch.setattr(backtest_module, "LightGBMQuantileForecaster", FakeForecaster)
    times = pd.date_range("2025-01-01", periods=45 * 96, freq="15min", tz="UTC")
    series = pd.Series(100 + np.sin(np.arange(len(times)) / 96), index=times)
    report = run_rolling_backtest(
        series, EventType.DEMAND, train_days=20, test_days=5,
        step_days=5, horizon_steps=96,
    )
    assert report.fold_results
    assert all(
        fold.train_end < fold.test_start - pd.Timedelta(days=1)
        for fold in report.fold_results
    )


def test_dual_horizon_trains_on_distinct_leads_and_calibrates_later_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DualHorizonEnsembleForecaster(switch_horizon_steps=6)
    fitted: dict[str, pd.DataFrame] = {}

    def fake_fit(name: str):
        def fit(features: pd.DataFrame, target: pd.Series, **kwargs: object) -> None:
            assert len(features) == len(target)
            fitted[name] = features
        return fit

    def fake_predict(frame: pd.DataFrame, *, apply_calibration: bool = True) -> tuple:
        point = np.full(len(frame), 10.0)
        return point - 2, point, point + 2

    monkeypatch.setattr(model.short_model, "fit", fake_fit("short"))
    monkeypatch.setattr(model.long_model, "fit", fake_fit("long"))
    monkeypatch.setattr(model.short_model, "predict", fake_predict)
    monkeypatch.setattr(model.long_model, "predict", fake_predict)
    times = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC").repeat(2)
    frame = pd.DataFrame({
        "hour": times.hour, "lag_1h": 1.0, "lag_24h": 2.0,
        "horizon_step": [1, 24] * 10,
    }, index=times)
    target = pd.Series(np.full(len(frame), 10.0), index=times)
    model.fit(frame, target, feature_names=list(frame.columns), calibration_fraction=0.2)
    assert set(fitted["short"]["horizon_step"]) == {1}
    assert set(fitted["long"]["horizon_step"]) == {24}
    assert fitted["short"].index.max() < times.unique()[-2]
    assert model.calibrator is not None and model.calibrator.is_fitted


def test_stacked_calibration_uses_blended_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = StackedQuantileEnsemble()

    def fake_fit(features: pd.DataFrame, target: pd.Series, **kwargs: object) -> None:
        assert len(features) == 8

    def fake_predict(low: float):
        def predict(frame: pd.DataFrame, *, apply_calibration: bool = True) -> tuple:
            return (
                np.full(len(frame), low), np.full(len(frame), low + 1),
                np.full(len(frame), low + 2),
            )
        return predict

    for component, low in ((model.lgbm, 0.0), (model.xgb, 10.0), (model.cat, 20.0)):
        monkeypatch.setattr(component, "fit", fake_fit)
        monkeypatch.setattr(component, "predict", fake_predict(low))
    frame = pd.DataFrame({"hour": range(10)})
    target = pd.Series(np.full(10, 11.0))
    model.fit(frame, target, feature_names=["hour"], calibration_fraction=0.2)
    assert model.calibrator is not None
    assert model.calibrator.q_correction > 0
    raw_low, _, raw_high = model.predict(frame.iloc[[0]], apply_calibration=False)
    cal_low, _, cal_high = model.predict(frame.iloc[[0]], apply_calibration=True)
    assert raw_low[0] == pytest.approx(7.5)
    assert raw_high[0] == pytest.approx(9.5)
    assert cal_low[0] < raw_low[0]
    assert cal_high[0] > raw_high[0]
