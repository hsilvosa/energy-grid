from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from energy_grid.backtest import run_rolling_backtest
from energy_grid.domain import EventType
from energy_grid.forecasting import (
    LightGBMQuantileForecaster,
    evaluate,
    synthetic_training_data,
)
from energy_grid.sources.dataset_reader import EntsoeDatasetReader


def test_forecasting_and_calibration() -> None:
    df = synthetic_training_data(days=40)
    features = ["hour", "quarter", "day_of_week", "month", "is_weekend", "temperature"]

    forecaster = LightGBMQuantileForecaster(n_estimators=30, num_leaves=15)
    forecaster.fit(
        df.iloc[:-96],
        df["demand"].iloc[:-96],
        feature_names=features,
        calibration_fraction=0.20,
    )

    test_df = df.iloc[-96:]
    p10_raw, p50_raw, p90_raw = forecaster.predict(test_df, apply_calibration=False)
    p10_cal, p50_cal, p90_cal = forecaster.predict(test_df, apply_calibration=True)

    # Monotonicity check
    assert np.all(p10_raw <= p50_raw)
    assert np.all(p50_raw <= p90_raw)
    assert np.all(p10_cal <= p50_cal)
    assert np.all(p50_cal <= p90_cal)

    metrics = evaluate(test_df["demand"].to_numpy(), p50_cal, p10_cal, p90_cal)
    assert metrics.mae > 0
    assert metrics.interval_coverage >= 0.0


def test_dataset_reader_and_backtest() -> None:
    reader = EntsoeDatasetReader(Path(r"D:\datasets\entsoe-transparency"))
    assert reader.exists()

    series = reader.load_series(
        EventType.DEMAND,
        country_code="ES",
        start_year=2024,
        end_year=2024,
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 1, 15, tzinfo=UTC),
    )
    assert not series.empty
    assert len(series) > 100

    # Backtest on synthetic series
    df = synthetic_training_data(days=60)
    rep = run_rolling_backtest(
        df.set_index("valid_time")["demand"],
        target=EventType.DEMAND,
        train_days=20,
        test_days=5,
        step_days=5,
        steps_per_day=96,
    )
    assert len(rep.fold_results) >= 2
    assert rep.overall_metrics.mae > 0


def test_hf_export_and_forecaster_api(tmp_path: Path) -> None:
    from energy_grid.forecaster import SpanishElectricityForecaster
    from energy_grid.hf_hub import export_model_to_hf_package

    df = synthetic_training_data(days=30)
    features = ["hour", "quarter", "day_of_week", "month", "is_weekend", "temperature"]
    forecaster = LightGBMQuantileForecaster(n_estimators=20, num_leaves=10)
    forecaster.fit(df.iloc[:-48], df["demand"].iloc[:-48], feature_names=features)

    test_df = df.iloc[-48:]
    p10, p50, p90 = forecaster.predict(test_df)
    metrics = evaluate(test_df["demand"].to_numpy(), p50, p10, p90)

    export_dir = tmp_path / "hf_model"
    export_model_to_hf_package(
        forecaster=forecaster,
        target=EventType.DEMAND,
        output_dir=export_dir,
        metrics=metrics,
        baseline_mae=metrics.mae * 1.5,
        training_rows=len(df) - 48,
        start_year=2024,
        end_year=2024,
        sample_df=test_df,
    )

    assert (export_dir / "README.md").exists()
    assert (export_dir / "config.json").exists()
    assert (export_dir / "models.joblib").exists()
    assert (export_dir / "inference.py").exists()

    # Load via high-level API
    loaded = SpanishElectricityForecaster.load(export_dir)
    preds = loaded.predict(test_df)
    assert len(preds) == 48
    assert "point_forecast" in preds.columns
    assert "p10" in preds.columns
    assert "p90" in preds.columns

