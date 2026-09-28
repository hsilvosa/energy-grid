from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from energy_grid.domain import MADRID, EventType, ForecastRecord
from energy_grid.features import LONG_HORIZON_FEATURES, SHORT_HORIZON_FEATURES
from energy_grid.time_utils import delivery_intervals, floor_quarter_hour


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    wape: float | None
    pinball_p10: float
    pinball_p90: float
    interval_coverage: float
    winkler_score: float | None = None
    mape: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def pinball_loss(
    actual: np.ndarray[Any, Any], predicted: np.ndarray[Any, Any], quantile: float
) -> float:
    """Compute asymmetric pinball loss for a specific quantile alpha."""
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def winkler_score(
    actual: np.ndarray[Any, Any],
    lower: np.ndarray[Any, Any],
    upper: np.ndarray[Any, Any],
    alpha: float = 0.2,
) -> float:
    """Compute Winkler Interval Score for a (1 - alpha) prediction interval."""
    spread = upper - lower
    under = (2.0 / alpha) * (lower - actual) * (actual < lower)
    over = (2.0 / alpha) * (actual - upper) * (actual > upper)
    return float(np.mean(spread + under + over))


def evaluate(
    actual: np.ndarray[Any, Any],
    point: np.ndarray[Any, Any],
    p10: np.ndarray[Any, Any],
    p90: np.ndarray[Any, Any],
    *,
    allow_wape: bool = True,
    alpha: float = 0.2,
) -> ForecastMetrics:
    """Evaluate point and quantile forecasts with complete probabilistic and accuracy metrics."""
    if not (len(actual) == len(point) == len(p10) == len(p90)) or not len(actual):
        raise ValueError("metric arrays must be non-empty and have equal length")
    error = actual - point
    abs_error = np.abs(error)
    denominator = float(np.sum(np.abs(actual)))
    wape = float(np.sum(abs_error) / denominator) if allow_wape and denominator else None

    # MAPE where non-zero actuals
    non_zero = np.abs(actual) > 1e-3
    mape = (
        float(np.mean(abs_error[non_zero] / np.abs(actual[non_zero]))) * 100.0
        if np.any(non_zero)
        else None
    )

    coverage = float(np.mean((actual >= p10) & (actual <= p90)))
    w_score = winkler_score(actual, p10, p90, alpha=alpha)

    return ForecastMetrics(
        mae=float(np.mean(abs_error)),
        rmse=float(math.sqrt(np.mean(np.square(error)))),
        wape=wape,
        mape=mape,
        pinball_p10=pinball_loss(actual, p10, 0.1),
        pinball_p90=pinball_loss(actual, p90, 0.9),
        interval_coverage=coverage,
        winkler_score=w_score,
    )


class ConformalCalibrator:
    """Split-conformal prediction calibrator with adaptive interval scaling."""

    def __init__(self, target_coverage: float = 0.80, adaptive: bool = True) -> None:
        self.target_coverage = target_coverage
        self.adaptive = adaptive
        self.q_correction: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self, actual: np.ndarray[Any, Any], p10: np.ndarray[Any, Any], p90: np.ndarray[Any, Any]
    ) -> ConformalCalibrator:
        """Fit empirical conformity adjustment on a dedicated calibration split."""
        if len(actual) == 0:
            raise ValueError("Calibration data cannot be empty")

        raw_scores = np.maximum(p10 - actual, actual - p90)
        if self.adaptive:
            widths = (p90 - p10) + 1e-4
            scores = raw_scores / widths
        else:
            scores = raw_scores

        # Compute empirical quantile of conformity scores
        n = len(scores)
        q_level = min(1.0, math.ceil((n + 1) * self.target_coverage) / n)
        self.q_correction = float(np.quantile(scores, q_level, method="higher"))
        self.is_fitted = True
        return self

    def calibrate(
        self, p10: np.ndarray[Any, Any], p90: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Apply conformal correction to broaden or narrow prediction intervals."""
        if not self.is_fitted:
            return p10, p90
        if self.adaptive:
            widths = (p90 - p10) + 1e-4
            cal_p10 = p10 - self.q_correction * widths
            cal_p90 = p90 + self.q_correction * widths
        else:
            cal_p10 = p10 - self.q_correction
            cal_p90 = p90 + self.q_correction
        return cal_p10, cal_p90


class QuantileForecaster(Protocol):
    model_name: str
    model_version: str

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        feature_names: list[str],
        calibration_fraction: float = 0.0,
    ) -> Any: ...

    def predict(
        self,
        features: pd.DataFrame,
        *,
        apply_calibration: bool = True,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]: ...


class LightGBMQuantileForecaster:
    """Multi-quantile LightGBM forecaster with monotonicity enforcement."""

    model_name = "lightgbm-quantile"

    def __init__(
        self,
        model_version: str = "untrained",
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self.model_version = model_version
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.feature_names: list[str] = []
        self.models: dict[float, Any] = {}
        self.calibrator: ConformalCalibrator | None = None

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        feature_names: list[str],
        calibration_fraction: float = 0.0,
    ) -> LightGBMQuantileForecaster:
        from lightgbm import LGBMRegressor

        self.feature_names = list(feature_names)
        X = features[self.feature_names]
        y = target.astype(float)

        if calibration_fraction > 0.0:
            cal_size = max(96, int(len(X) * calibration_fraction))
            X_train, y_train = X.iloc[:-cal_size], y.iloc[:-cal_size]
            X_cal, y_cal = X.iloc[-cal_size:], y.iloc[-cal_size:]
        else:
            X_train, y_train = X, y
            X_cal, y_cal = None, None

        for quantile in (0.1, 0.5, 0.9):
            model = LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                verbosity=-1,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            self.models[quantile] = model

        if X_cal is not None and y_cal is not None:
            raw_p10 = self.models[0.1].predict(X_cal)
            raw_p50 = self.models[0.5].predict(X_cal)
            raw_p90 = self.models[0.9].predict(X_cal)
            stacked = np.sort(np.vstack([raw_p10, raw_p50, raw_p90]), axis=0)
            self.calibrator = ConformalCalibrator(target_coverage=0.80).fit(
                y_cal.to_numpy(), stacked[0], stacked[2]
            )

        self.model_version = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        return self

    def predict(
        self,
        features: pd.DataFrame,
        *,
        apply_calibration: bool = True,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Predict P10, P50, P90 quantiles with monotonicity and optional conformal calibration."""
        if set(self.models) != {0.1, 0.5, 0.9}:
            raise RuntimeError("forecaster is not trained")
        X = features[self.feature_names]
        raw_p10 = self.models[0.1].predict(X)
        raw_p50 = self.models[0.5].predict(X)
        raw_p90 = self.models[0.9].predict(X)

        # Monotonicity post-processing: p10 <= p50 <= p90
        stacked = np.sort(np.vstack([raw_p10, raw_p50, raw_p90]), axis=0)
        p10, p50, p90 = stacked[0], stacked[1], stacked[2]

        if apply_calibration and self.calibrator is not None and self.calibrator.is_fitted:
            p10, p90 = self.calibrator.calibrate(p10, p90)
            stacked_cal = np.sort(np.vstack([p10, p50, p90]), axis=0)
            p10, p50, p90 = stacked_cal[0], stacked_cal[1], stacked_cal[2]

        return p10, p50, p90

    def get_feature_importances(self) -> dict[str, float]:
        """Get feature importance from point forecast (P50) model."""
        if 0.5 not in self.models:
            return {}
        p50_model = self.models[0.5]
        importances = p50_model.feature_importances_
        return dict(zip(self.feature_names, [float(v) for v in importances], strict=True))

    def save(self, directory: Path | str) -> None:
        """Save model artifacts and configuration to directory."""
        import joblib

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, path / "models.joblib")
        meta = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_names": self.feature_names,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "calibrator_fitted": self.calibrator.is_fitted if self.calibrator else False,
            "q_correction": self.calibrator.q_correction if self.calibrator else 0.0,
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str) -> LightGBMQuantileForecaster:
        """Load forecaster from serialized directory."""
        import joblib

        path = Path(directory)
        meta_file = (
            path / "config.json" if (path / "config.json").exists() else path / "metadata.json"
        )
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        instance = cls(
            model_version=meta.get("model_version", "unknown"),
            n_estimators=meta.get("n_estimators", 100),
            learning_rate=meta.get("learning_rate", 0.05),
            num_leaves=meta.get("num_leaves", 31),
        )
        instance.feature_names = meta["feature_names"]
        instance.models = joblib.load(path / "models.joblib")
        cal_data = meta.get("calibrator", {})
        if meta.get("calibrator_fitted") or cal_data.get("is_fitted"):
            q_corr = meta.get("q_correction") or cal_data.get("q_correction", 0.0)
            instance.calibrator = ConformalCalibrator()
            instance.calibrator.q_correction = float(q_corr)
            instance.calibrator.is_fitted = True
        return instance


class XGBoostQuantileForecaster:
    """Multi-quantile XGBoost forecaster."""

    model_name = "xgboost-quantile"

    def __init__(
        self,
        model_version: str = "untrained",
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42,
    ) -> None:
        self.model_version = model_version
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.feature_names: list[str] = []
        self.models: dict[float, Any] = {}
        self.calibrator: ConformalCalibrator | None = None

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        feature_names: list[str],
        calibration_fraction: float = 0.0,
    ) -> XGBoostQuantileForecaster:
        from xgboost import XGBRegressor

        self.feature_names = list(feature_names)
        X = features[self.feature_names]
        y = target.astype(float)

        if calibration_fraction > 0.0:
            cal_size = max(96, int(len(X) * calibration_fraction))
            X_train, y_train = X.iloc[:-cal_size], y.iloc[:-cal_size]
            X_cal, y_cal = X.iloc[-cal_size:], y.iloc[-cal_size:]
        else:
            X_train, y_train = X, y
            X_cal, y_cal = None, None

        for quantile in (0.1, 0.5, 0.9):
            model = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=quantile,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            self.models[quantile] = model

        if X_cal is not None and y_cal is not None:
            raw_p10 = self.models[0.1].predict(X_cal)
            raw_p50 = self.models[0.5].predict(X_cal)
            raw_p90 = self.models[0.9].predict(X_cal)
            stacked = np.sort(np.vstack([raw_p10, raw_p50, raw_p90]), axis=0)
            self.calibrator = ConformalCalibrator(target_coverage=0.80).fit(
                y_cal.to_numpy(), stacked[0], stacked[2]
            )

        self.model_version = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        return self

    def predict(
        self,
        features: pd.DataFrame,
        *,
        apply_calibration: bool = True,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        X = features[self.feature_names]
        raw_p10 = self.models[0.1].predict(X)
        raw_p50 = self.models[0.5].predict(X)
        raw_p90 = self.models[0.9].predict(X)

        stacked = np.sort(np.vstack([raw_p10, raw_p50, raw_p90]), axis=0)
        p10, p50, p90 = stacked[0], stacked[1], stacked[2]

        if apply_calibration and self.calibrator is not None and self.calibrator.is_fitted:
            p10, p90 = self.calibrator.calibrate(p10, p90)
            stacked_cal = np.sort(np.vstack([p10, p50, p90]), axis=0)
            p10, p50, p90 = stacked_cal[0], stacked_cal[1], stacked_cal[2]

        return p10, p50, p90

    def get_feature_importances(self) -> dict[str, float]:
        if 0.5 not in self.models:
            return {}
        p50_model = self.models[0.5]
        importances = p50_model.feature_importances_
        return dict(zip(self.feature_names, [float(v) for v in importances], strict=True))


class CatBoostQuantileForecaster:
    """Multi-quantile CatBoost forecaster."""

    model_name = "catboost-quantile"

    def __init__(
        self,
        model_version: str = "untrained",
        iterations: int = 300,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 42,
    ) -> None:
        self.model_version = model_version
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self.feature_names: list[str] = []
        self.models: dict[float, Any] = {}
        self.calibrator: ConformalCalibrator | None = None

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        feature_names: list[str],
        calibration_fraction: float = 0.0,
    ) -> CatBoostQuantileForecaster:
        from catboost import CatBoostRegressor

        self.feature_names = list(feature_names)
        X = features[self.feature_names]
        y = target.astype(float)

        if calibration_fraction > 0.0:
            cal_size = max(96, int(len(X) * calibration_fraction))
            X_train, y_train = X.iloc[:-cal_size], y.iloc[:-cal_size]
            X_cal, y_cal = X.iloc[-cal_size:], y.iloc[-cal_size:]
        else:
            X_train, y_train = X, y
            X_cal, y_cal = None, None

        for quantile in (0.1, 0.5, 0.9):
            model = CatBoostRegressor(
                loss_function=f"Quantile:alpha={quantile}",
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                random_seed=self.random_seed,
                verbose=False,
            )
            model.fit(X_train, y_train)
            self.models[quantile] = model

        if X_cal is not None and y_cal is not None:
            raw_p10 = self.models[0.1].predict(X_cal)
            raw_p50 = self.models[0.5].predict(X_cal)
            raw_p90 = self.models[0.9].predict(X_cal)
            stacked = np.sort(np.vstack([raw_p10, raw_p50, raw_p90]), axis=0)
            self.calibrator = ConformalCalibrator(target_coverage=0.80).fit(
                y_cal.to_numpy(), stacked[0], stacked[2]
            )

        self.model_version = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        return self

    def predict(
        self,
        features: pd.DataFrame,
        *,
        apply_calibration: bool = True,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        X = features[self.feature_names]
        raw_p10 = self.models[0.1].predict(X)
        raw_p50 = self.models[0.5].predict(X)
        raw_p90 = self.models[0.9].predict(X)

        stacked = np.sort(np.vstack([raw_p10, raw_p50, raw_p90]), axis=0)
        p10, p50, p90 = stacked[0], stacked[1], stacked[2]

        if apply_calibration and self.calibrator is not None and self.calibrator.is_fitted:
            p10, p90 = self.calibrator.calibrate(p10, p90)
            stacked_cal = np.sort(np.vstack([p10, p50, p90]), axis=0)
            p10, p50, p90 = stacked_cal[0], stacked_cal[1], stacked_cal[2]

        return p10, p50, p90

    def get_feature_importances(self) -> dict[str, float]:
        if 0.5 not in self.models:
            return {}
        p50_model = self.models[0.5]
        importances = p50_model.get_feature_importance()
        return dict(zip(self.feature_names, [float(v) for v in importances], strict=True))


class StackedQuantileEnsemble:
    """Blended ensemble of LightGBM, XGBoost, and CatBoost quantile forecasters."""

    model_name = "stacked-ensemble-quantile"

    def __init__(
        self,
        model_version: str = "untrained",
        weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
    ) -> None:
        self.model_version = model_version
        self.weights = weights
        self.lgbm = LightGBMQuantileForecaster()
        self.xgb = XGBoostQuantileForecaster()
        self.cat = CatBoostQuantileForecaster()
        self.calibrator: ConformalCalibrator | None = None
        self.feature_names: list[str] = []

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        feature_names: list[str],
        calibration_fraction: float = 0.15,
    ) -> StackedQuantileEnsemble:
        self.feature_names = list(feature_names)
        if not 0 <= calibration_fraction < 1:
            raise ValueError("calibration_fraction must be in [0, 1)")
        if not features.index.is_monotonic_increasing:
            raise ValueError("ensemble training rows must be ordered by valid time")
        cal_periods = (
            max(1, int(features.index.nunique() * calibration_fraction))
            if calibration_fraction else 0
        )
        if cal_periods:
            cal_start = features.index.unique()[-cal_periods]
            fit_mask = features.index < cal_start
            cal_mask = ~fit_mask
            fit_features = features.loc[fit_mask]
            fit_target = target.loc[fit_mask]
        else:
            fit_features, fit_target = features, target
        if fit_features.empty:
            raise ValueError("not enough rows before calibration split")
        self.lgbm.fit(
            fit_features, fit_target, feature_names=feature_names, calibration_fraction=0
        )
        self.xgb.fit(
            fit_features, fit_target, feature_names=feature_names, calibration_fraction=0
        )
        self.cat.fit(
            fit_features, fit_target, feature_names=feature_names, calibration_fraction=0
        )
        if cal_periods:
            p10, _, p90 = self.predict(features.loc[cal_mask], apply_calibration=False)
            self.calibrator = ConformalCalibrator().fit(
                target.loc[cal_mask].to_numpy(), p10, p90
            )
        self.model_version = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        return self

    def predict(
        self,
        features: pd.DataFrame,
        *,
        apply_calibration: bool = True,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        w1, w2, w3 = self.weights
        p10_1, p50_1, p90_1 = self.lgbm.predict(features, apply_calibration=False)
        p10_2, p50_2, p90_2 = self.xgb.predict(features, apply_calibration=False)
        p10_3, p50_3, p90_3 = self.cat.predict(features, apply_calibration=False)

        p10 = w1 * p10_1 + w2 * p10_2 + w3 * p10_3
        p50 = w1 * p50_1 + w2 * p50_2 + w3 * p50_3
        p90 = w1 * p90_1 + w2 * p90_2 + w3 * p90_3

        stacked = np.sort(np.vstack([p10, p50, p90]), axis=0)
        p10, p50, p90 = stacked[0], stacked[1], stacked[2]

        if apply_calibration and self.calibrator is not None:
            p10, p90 = self.calibrator.calibrate(p10, p90)
            stacked_cal = np.sort(np.vstack([p10, p50, p90]), axis=0)
            p10, p50, p90 = stacked_cal[0], stacked_cal[1], stacked_cal[2]

        return p10, p50, p90

    def get_feature_importances(self) -> dict[str, float]:
        lgbm_imp = self.lgbm.get_feature_importances()
        xgb_imp = self.xgb.get_feature_importances()
        merged = {}
        for k in self.feature_names:
            merged[k] = float(0.6 * lgbm_imp.get(k, 0.0) + 0.4 * xgb_imp.get(k, 0.0))
        return merged


class DualHorizonEnsembleForecaster:
    """Dual-model strategy combining a specialized Short-Horizon (1-6h) model

    and a Long-Horizon (24-48h) model with smooth lead-time transition.
    """

    model_name = "dual-horizon-hybrid"

    def __init__(
        self,
        model_version: str = "untrained",
        switch_horizon_steps: int = 24,  # e.g. 24 steps = 6 hours on 15-min data
    ) -> None:
        self.model_version = model_version
        self.switch_horizon_steps = switch_horizon_steps
        self.short_model = LightGBMQuantileForecaster(n_estimators=180, learning_rate=0.04)
        self.long_model = LightGBMQuantileForecaster(n_estimators=240, learning_rate=0.04)
        self.calibrator: ConformalCalibrator | None = None
        self.feature_names: list[str] = []

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        feature_names: list[str],
        calibration_fraction: float = 0.15,
    ) -> DualHorizonEnsembleForecaster:
        self.feature_names = list(feature_names)
        if "horizon_step" not in features:
            raise ValueError("dual-horizon forecasts require a horizon_step column")
        if not 0 <= calibration_fraction < 1:
            raise ValueError("calibration_fraction must be in [0, 1)")
        if not features.index.is_monotonic_increasing:
            raise ValueError("dual-horizon training rows must be ordered by valid time")
        cal_periods = (
            max(1, int(features.index.nunique() * calibration_fraction))
            if calibration_fraction else 0
        )
        if cal_periods:
            cal_start = features.index.unique()[-cal_periods]
            fit_mask = features.index < cal_start
            cal_mask = ~fit_mask
            fit_features = features.loc[fit_mask]
            fit_target = target.loc[fit_mask]
        else:
            fit_features, fit_target = features, target
        if fit_features.empty:
            raise ValueError("not enough rows before calibration split")
        short_features = [name for name in SHORT_HORIZON_FEATURES if name in feature_names]
        long_features = [name for name in LONG_HORIZON_FEATURES if name in feature_names]
        if not short_features or not long_features:
            raise ValueError("short and long horizon feature sets must both be available")
        short_mask = fit_features["horizon_step"] <= self.switch_horizon_steps
        long_mask = fit_features["horizon_step"] >= self.switch_horizon_steps
        short_train = fit_features.loc[short_mask] if short_mask.any() else fit_features
        long_train = fit_features.loc[long_mask] if long_mask.any() else fit_features
        short_target = fit_target.loc[short_mask] if short_mask.any() else fit_target
        long_target = fit_target.loc[long_mask] if long_mask.any() else fit_target
        self.short_model.fit(
            short_train, short_target,
            feature_names=short_features, calibration_fraction=0,
        )
        self.long_model.fit(
            long_train, long_target,
            feature_names=long_features, calibration_fraction=0,
        )
        if cal_periods:
            p10, _, p90 = self.predict(features.loc[cal_mask], apply_calibration=False)
            self.calibrator = ConformalCalibrator().fit(
                target.loc[cal_mask].to_numpy(), p10, p90
            )
        self.model_version = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
        return self

    def predict(
        self,
        features: pd.DataFrame,
        *,
        apply_calibration: bool = True,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Predict with dynamic decay from short-horizon to long-horizon model."""
        if "horizon_step" not in features:
            raise ValueError("dual-horizon forecasts require a horizon_step column")
        p10_s, p50_s, p90_s = self.short_model.predict(features, apply_calibration=False)
        p10_l, p50_l, p90_l = self.long_model.predict(features, apply_calibration=False)

        steps = features["horizon_step"].to_numpy(dtype=float) - 1.0
        alpha = np.maximum(0.0, 1.0 - (steps / float(self.switch_horizon_steps)))
        alpha = np.clip(alpha, 0.0, 1.0)

        p10 = alpha * p10_s + (1.0 - alpha) * p10_l
        p50 = alpha * p50_s + (1.0 - alpha) * p50_l
        p90 = alpha * p90_s + (1.0 - alpha) * p90_l

        stacked = np.sort(np.vstack([p10, p50, p90]), axis=0)
        if apply_calibration and self.calibrator is not None:
            low, high = self.calibrator.calibrate(stacked[0], stacked[2])
            stacked = np.sort(np.vstack([low, stacked[1], high]), axis=0)
        return stacked[0], stacked[1], stacked[2]

    def get_feature_importances(self) -> dict[str, float]:
        return self.short_model.get_feature_importances()


def create_forecaster(
    model_type: (
        Literal["lightgbm", "xgboost", "catboost", "stacked", "dual_horizon"]
    ) = "lightgbm",
    **kwargs: Any,
) -> Any:
    """Factory to instantiate the desired quantile forecaster architecture."""
    if model_type == "lightgbm":
        return LightGBMQuantileForecaster(**kwargs)
    elif model_type == "xgboost":
        return XGBoostQuantileForecaster(**kwargs)
    elif model_type == "catboost":
        return CatBoostQuantileForecaster(**kwargs)
    elif model_type == "stacked":
        return StackedQuantileEnsemble()
    elif model_type == "dual_horizon":
        return DualHorizonEnsembleForecaster(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class PersistenceBaseline:
    """Seasonal persistence baseline for 24-hour or 7-day benchmarks."""

    def __init__(self, steps_lag: int = 96) -> None:
        self.steps_lag = steps_lag

    def predict(self, series: pd.Series, horizon_steps: int) -> np.ndarray[Any, Any]:
        """Predict using the values from steps_lag ago."""
        if len(series) < self.steps_lag:
            raise ValueError(f"Series has {len(series)} points; need at least {self.steps_lag}")
        history = series.iloc[-self.steps_lag :].to_numpy()
        repeated = np.tile(history, math.ceil(horizon_steps / len(history)))
        return repeated[:horizon_steps].astype(float)


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

    def predict(
        self, steps: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
