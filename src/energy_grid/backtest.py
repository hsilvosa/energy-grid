from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from energy_grid.domain import MADRID, EventType
from energy_grid.features import build_multiscale_features
from energy_grid.forecasting import (
    ForecastMetrics,
    LightGBMQuantileForecaster,
    evaluate,
)


@dataclass(frozen=True)
class FoldResult:
    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    model_metrics: ForecastMetrics
    baseline_24h_metrics: ForecastMetrics
    baseline_7d_metrics: ForecastMetrics
    uncalibrated_coverage: float
    calibrated_coverage: float


@dataclass
class BacktestReport:
    target: EventType
    overall_metrics: ForecastMetrics
    baseline_24h_metrics: ForecastMetrics
    baseline_7d_metrics: ForecastMetrics
    mae_improvement_vs_24h: float
    mae_improvement_vs_7d: float
    uncalibrated_coverage: float
    calibrated_coverage: float
    total_test_rows: int
    fold_results: list[FoldResult]
    hourly_mae: dict[int, float]
    seasonal_mae: dict[str, float]
    day_type_mae: dict[str, float]
    regime_mae: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.value,
            "overall_metrics": self.overall_metrics.to_dict(),
            "baseline_24h_metrics": self.baseline_24h_metrics.to_dict(),
            "baseline_7d_metrics": self.baseline_7d_metrics.to_dict(),
            "mae_improvement_vs_24h": self.mae_improvement_vs_24h,
            "mae_improvement_vs_7d": self.mae_improvement_vs_7d,
            "uncalibrated_coverage": self.uncalibrated_coverage,
            "calibrated_coverage": self.calibrated_coverage,
            "total_test_rows": self.total_test_rows,
            "hourly_mae": self.hourly_mae,
            "seasonal_mae": self.seasonal_mae,
            "day_type_mae": self.day_type_mae,
            "regime_mae": self.regime_mae,
            "fold_count": len(self.fold_results),
        }

    def to_markdown(self) -> str:
        target_name = self.target.value.upper()
        unit = "MW" if self.target == EventType.DEMAND else "EUR/MWh"

        m_wape = f"{self.overall_metrics.wape * 100:.2f}%" if self.overall_metrics.wape else "N/A"
        b24_wape = (
            f"{self.baseline_24h_metrics.wape * 100:.2f}%"
            if self.baseline_24h_metrics.wape
            else "N/A"
        )
        b7d_wape = (
            f"{self.baseline_7d_metrics.wape * 100:.2f}%"
            if self.baseline_7d_metrics.wape
            else "N/A"
        )

        lines = [
            f"# Rolling-Origin Multi-Year Backtest Report: {target_name}",
            "",
            "## Summary Metrics",
            "",
            "| Metric | LightGBM Model | 24h Baseline | 7d Baseline | Improvement vs 7d |",
            "|---|---:|---:|---:|---:|",
            (
                f"| **MAE** | **{self.overall_metrics.mae:.3f} {unit}** | "
                f"{self.baseline_24h_metrics.mae:.3f} {unit} | "
                f"{self.baseline_7d_metrics.mae:.3f} {unit} | "
                f"**{self.mae_improvement_vs_7d:+.1f}%** |"
            ),
            (
                f"| **RMSE** | **{self.overall_metrics.rmse:.3f} {unit}** | "
                f"{self.baseline_24h_metrics.rmse:.3f} {unit} | "
                f"{self.baseline_7d_metrics.rmse:.3f} {unit} | — |"
            ),
            f"| **WAPE** | {m_wape} | {b24_wape} | {b7d_wape} | — |",
            f"| **P10 Pinball Loss** | {self.overall_metrics.pinball_p10:.3f} | — | — | — |",
            f"| **P90 Pinball Loss** | {self.overall_metrics.pinball_p90:.3f} | — | — | — |",
            (
                f"| **Raw P10-P90 Coverage** | {self.uncalibrated_coverage * 100:.1f}% | "
                "— | — | (Nominal: 80%) |"
            ),
            (
                f"| **Calibrated Coverage** | **{self.calibrated_coverage * 100:.1f}%** | "
                "— | — | (Target: 75–85%) |"
            ),
            f"| **Winkler Score** | {self.overall_metrics.winkler_score:.3f} | — | — | — |",
            "",
            (
                f"**Evaluated Folds**: {len(self.fold_results)} | "
                f"**Total Test Observations**: {self.total_test_rows:,}"
            ),
            "",
            "## Slice Performance: Seasonal MAE",
            "",
            f"| Season | MAE ({unit}) |",
            "|---|---:|",
        ]
        for season, mae in sorted(self.seasonal_mae.items()):
            lines.append(f"| {season} | {mae:.3f} |")

        lines.extend([
            "",
            "## Slice Performance: Day Type MAE",
            "",
            f"| Day Type | MAE ({unit}) |",
            "|---|---:|",
        ])
        for day_type, mae in sorted(self.day_type_mae.items()):
            lines.append(f"| {day_type} | {mae:.3f} |")

        lines.extend([
            "",
            "## Slice Performance: Regime MAE",
            "",
            f"| Regime | MAE ({unit}) |",
            "|---|---:|",
        ])
        for regime, mae in sorted(self.regime_mae.items()):
            lines.append(f"| {regime} | {mae:.3f} |")

        lines.extend([
            "",
            "## Fold-by-Fold Details",
            "",
            "| Fold | Test Start | Test End | LightGBM MAE | 7d Baseline MAE | Calibrated Cov |",
            "|---:|---|---|---:|---:|---:|",
        ])
        for f in self.fold_results:
            lines.append(
                f"| {f.fold_index} | {f.test_start.strftime('%Y-%m-%d')} | "
                f"{f.test_end.strftime('%Y-%m-%d')} | {f.model_metrics.mae:.2f} {unit} | "
                f"{f.baseline_7d_metrics.mae:.2f} {unit} | "
                f"{f.calibrated_coverage * 100:.1f}% |"
            )

        return "\n".join(lines)


def run_rolling_backtest(
    target_series: pd.Series,
    target: EventType,
    exogenous_series: pd.Series | None = None,
    *,
    train_days: int = 365,
    test_days: int = 30,
    step_days: int = 30,
    calibration_fraction: float = 0.15,
    steps_per_day: int = 96,
) -> BacktestReport:
    """Perform multi-year rolling-origin time-series cross validation with slice analysis."""
    df = build_multiscale_features(
        target_series, exogenous_series=exogenous_series, steps_per_day=steps_per_day
    )
    feature_names = [col for col in df.columns if col != "target"]

    step_size = step_days * steps_per_day
    train_size = train_days * steps_per_day
    test_size = test_days * steps_per_day

    if len(df) < (train_size + test_size):
        raise ValueError(
            f"Not enough data for rolling backtest: have {len(df)} rows, "
            f"need at least {train_size + test_size}"
        )

    all_actuals: list[float] = []
    all_points: list[float] = []
    all_p10_raw: list[float] = []
    all_p90_raw: list[float] = []
    all_p10_cal: list[float] = []
    all_p90_cal: list[float] = []

    all_base_24h: list[float] = []
    all_base_7d: list[float] = []
    all_timestamps: list[datetime] = []

    fold_results: list[FoldResult] = []
    fold_idx = 1

    for start_idx in range(train_size, len(df) - test_size + 1, step_size):
        train_df = df.iloc[start_idx - train_size : start_idx]
        test_df = df.iloc[start_idx : start_idx + test_size]

        # Fit model on training slice
        forecaster = LightGBMQuantileForecaster(
            n_estimators=180, learning_rate=0.05, num_leaves=31
        ).fit(
            train_df,
            train_df["target"],
            feature_names=feature_names,
            calibration_fraction=calibration_fraction,
        )

        raw_p10, raw_p50, raw_p90 = forecaster.predict(test_df, apply_calibration=False)
        cal_p10, cal_p50, cal_p90 = forecaster.predict(test_df, apply_calibration=True)

        actuals = test_df["target"].to_numpy()
        base_24h = test_df["lag_24h"].to_numpy()
        base_7d = test_df["lag_7d"].to_numpy()

        # Fold metrics
        m_metrics = evaluate(
            actuals, cal_p50, cal_p10, cal_p90, allow_wape=target == EventType.DEMAND
        )
        b24_metrics = evaluate(
            actuals, base_24h, base_24h, base_24h, allow_wape=target == EventType.DEMAND
        )
        b7d_metrics = evaluate(
            actuals, base_7d, base_7d, base_7d, allow_wape=target == EventType.DEMAND
        )

        uncal_cov = float(np.mean((actuals >= raw_p10) & (actuals <= raw_p90)))
        cal_cov = float(np.mean((actuals >= cal_p10) & (actuals <= cal_p90)))

        fold_results.append(
            FoldResult(
                fold_index=fold_idx,
                train_start=train_df.index.min().to_pydatetime(),
                train_end=train_df.index.max().to_pydatetime(),
                test_start=test_df.index.min().to_pydatetime(),
                test_end=test_df.index.max().to_pydatetime(),
                model_metrics=m_metrics,
                baseline_24h_metrics=b24_metrics,
                baseline_7d_metrics=b7d_metrics,
                uncalibrated_coverage=uncal_cov,
                calibrated_coverage=cal_cov,
            )
        )
        fold_idx += 1

        all_actuals.extend(actuals)
        all_points.extend(cal_p50)
        all_p10_raw.extend(raw_p10)
        all_p90_raw.extend(raw_p90)
        all_p10_cal.extend(cal_p10)
        all_p90_cal.extend(cal_p90)
        all_base_24h.extend(base_24h)
        all_base_7d.extend(base_7d)
        all_timestamps.extend([t.to_pydatetime() for t in test_df.index])

    arr_actuals = np.array(all_actuals)
    arr_points = np.array(all_points)
    arr_p10_raw = np.array(all_p10_raw)
    arr_p90_raw = np.array(all_p90_raw)
    arr_p10_cal = np.array(all_p10_cal)
    arr_p90_cal = np.array(all_p90_cal)
    arr_base_24h = np.array(all_base_24h)
    arr_base_7d = np.array(all_base_7d)

    overall_metrics = evaluate(
        arr_actuals, arr_points, arr_p10_cal, arr_p90_cal, allow_wape=target == EventType.DEMAND
    )
    overall_b24 = evaluate(
        arr_actuals,
        arr_base_24h,
        arr_base_24h,
        arr_base_24h,
        allow_wape=target == EventType.DEMAND,
    )
    overall_b7d = evaluate(
        arr_actuals,
        arr_base_7d,
        arr_base_7d,
        arr_base_7d,
        allow_wape=target == EventType.DEMAND,
    )

    uncal_cov = float(np.mean((arr_actuals >= arr_p10_raw) & (arr_actuals <= arr_p90_raw)))
    cal_cov = float(np.mean((arr_actuals >= arr_p10_cal) & (arr_actuals <= arr_p90_cal)))

    imp_24h = ((overall_b24.mae - overall_metrics.mae) / overall_b24.mae) * 100.0
    imp_7d = ((overall_b7d.mae - overall_metrics.mae) / overall_b7d.mae) * 100.0

    # Slices Analysis
    eval_df = pd.DataFrame(
        {
            "timestamp": all_timestamps,
            "actual": arr_actuals,
            "predicted": arr_points,
            "abs_error": np.abs(arr_actuals - arr_points),
        }
    )
    eval_df["local_time"] = pd.DatetimeIndex(eval_df["timestamp"]).tz_convert(MADRID)
    eval_df["hour"] = eval_df["local_time"].dt.hour
    eval_df["month"] = eval_df["local_time"].dt.month
    eval_df["day_of_week"] = eval_df["local_time"].dt.dayofweek

    # Hourly MAE
    hourly_mae = eval_df.groupby("hour")["abs_error"].mean().to_dict()

    # Seasonal MAE
    def to_season(month: int) -> str:
        if month in (12, 1, 2):
            return "Winter"
        elif month in (3, 4, 5):
            return "Spring"
        elif month in (6, 7, 8):
            return "Summer"
        return "Autumn"

    eval_df["season"] = eval_df["month"].apply(to_season)
    seasonal_mae = eval_df.groupby("season")["abs_error"].mean().to_dict()

    # Day Type MAE
    eval_df["day_type"] = eval_df["day_of_week"].apply(
        lambda d: "Weekend" if d >= 5 else "Weekday"
    )
    day_type_mae = eval_df.groupby("day_type")["abs_error"].mean().to_dict()

    # Regime MAE
    q95 = float(np.quantile(arr_actuals, 0.95))
    eval_df["is_peak_95"] = eval_df["actual"] >= q95
    peak_mae = float(eval_df[eval_df["is_peak_95"]]["abs_error"].mean())
    normal_mae = float(eval_df[~eval_df["is_peak_95"]]["abs_error"].mean())
    regime_mae = {"Top 5% Peak Regime": peak_mae, "Normal Regime": normal_mae}

    if target == EventType.PRICE:
        eval_df["is_negative"] = eval_df["actual"] < 0
        if eval_df["is_negative"].any():
            regime_mae["Negative Price Regime"] = float(
                eval_df[eval_df["is_negative"]]["abs_error"].mean()
            )
        eval_df["is_spike_150"] = eval_df["actual"] > 150
        if eval_df["is_spike_150"].any():
            regime_mae["Price Spikes > 150 EUR"] = float(
                eval_df[eval_df["is_spike_150"]]["abs_error"].mean()
            )

    return BacktestReport(
        target=target,
        overall_metrics=overall_metrics,
        baseline_24h_metrics=overall_b24,
        baseline_7d_metrics=overall_b7d,
        mae_improvement_vs_24h=imp_24h,
        mae_improvement_vs_7d=imp_7d,
        uncalibrated_coverage=uncal_cov,
        calibrated_coverage=cal_cov,
        total_test_rows=len(eval_df),
        fold_results=fold_results,
        hourly_mae=hourly_mae,
        seasonal_mae=seasonal_mae,
        day_type_mae=day_type_mae,
        regime_mae=regime_mae,
    )
