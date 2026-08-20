from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from energy_grid.domain import EventType
from energy_grid.forecasting import LightGBMQuantileForecaster


class EuropeanElectricityForecaster:
    """High-level Python forecaster interface for European Electricity Demand and Prices."""

    def __init__(
        self,
        forecaster: LightGBMQuantileForecaster,
        target: EventType = EventType.DEMAND,
        country_code: str = "ES",
        zone_key: str | None = None,
    ) -> None:
        self.forecaster = forecaster
        self.target = target
        self.country_code = country_code
        self.zone_key = zone_key or country_code

    @classmethod
    def load(cls, model_dir: Path | str) -> EuropeanElectricityForecaster:
        """Load forecaster from a local directory or Hugging Face repository."""
        path = Path(model_dir)
        config_file = path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found in {path}")

        config = json.loads(config_file.read_text(encoding="utf-8"))
        target_type = EventType(config.get("target", "demand"))
        country = str(config.get("country_code", "ES"))
        zone = str(config.get("zone_key", country))
        forecaster = LightGBMQuantileForecaster.load(path)
        return cls(
            forecaster=forecaster,
            target=target_type,
            country_code=country,
            zone_key=zone,
        )

    @classmethod
    def from_pretrained(
        cls, repo_id: str, token: str | None = None
    ) -> EuropeanElectricityForecaster:
        """Download and load model directly from the Hugging Face Hub."""
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(repo_id=repo_id, token=token)
        return cls.load(local_dir)

    def predict(
        self,
        features: pd.DataFrame,
        *,
        apply_calibration: bool = True,
    ) -> pd.DataFrame:
        """Generate Point (P50) and 80% Calibrated Prediction Intervals (P10, P90)."""
        p10, p50, p90 = self.forecaster.predict(features, apply_calibration=apply_calibration)
        unit = "MW" if self.target == EventType.DEMAND else "EUR/MWh"
        return pd.DataFrame(
            {
                "p10": p10,
                "point_forecast": p50,
                "p90": p90,
                "unit": unit,
            },
            index=features.index,
        )

    def feature_importances(self) -> dict[str, float]:
        """Return relative feature importance weights from point model."""
        return self.forecaster.get_feature_importances()


# Alias for backward compatibility
SpanishElectricityForecaster = EuropeanElectricityForecaster
