from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from energy_grid.domain import EventType
from energy_grid.forecasting import ForecastMetrics, LightGBMQuantileForecaster

COUNTRY_NAMES = {
    "ES": "Spain",
    "FR": "France",
    "DE": "Germany",
    "IT": "Italy",
    "PT": "Portugal",
    "NL": "Netherlands",
    "BE": "Belgium",
    "PL": "Poland",
    "AT": "Austria",
    "CH": "Switzerland",
    "NO": "Norway",
    "SE": "Sweden",
    "FI": "Finland",
    "DK": "Denmark",
    "GR": "Greece",
    "CZ": "Czechia",
    "RO": "Romania",
    "HU": "Hungary",
    "IE": "Ireland",
    "HR": "Croatia",
    "BG": "Bulgaria",
    "SK": "Slovakia",
    "SI": "Slovenia",
}


def generate_model_card_markdown(
    target: EventType,
    metrics: ForecastMetrics,
    baseline_mae: float,
    training_rows: int,
    start_year: int,
    end_year: int,
    feature_names: list[str],
    country_code: str = "ES",
    zone_key: str | None = None,
) -> str:
    c_name = COUNTRY_NAMES.get(country_code, country_code)
    target_name = (
        "Electricity Demand" if target == EventType.DEMAND else "Day-Ahead Electricity Price"
    )
    unit = "MW" if target == EventType.DEMAND else "EUR/MWh"
    model_id = f"{c_name.lower().replace(' ', '-')}-{target.value}-forecaster"
    resolution = "15-minute intervals" if target == EventType.DEMAND else "hourly intervals"
    improvement = ((baseline_mae - metrics.mae) / baseline_mae) * 100.0

    wape_line = f"- **WAPE**: {metrics.wape * 100:.2f}%\n" if metrics.wape is not None else ""

    yaml_header = f"""---
language:
- en
- {country_code.lower()}
license: apache-2.0
tags:
- time-series-forecasting
- energy-forecasting
- electricity-demand
- day-ahead-prices
- lightgbm
- tabular-regression
- conformal-prediction
- {c_name.lower()}
- europe
- entsoe
pipeline_tag: tabular-regression
datasets:
- hsilvosa/entsoe-day-ahead
metrics:
- mae
- rmse
- pinball_loss
- winkler_score
model-index:
- name: {model_id}
  results:
  - task:
      type: tabular-regression
      name: Electricity {target.value.capitalize()} Forecasting
    dataset:
      name: ENTSO-E {c_name} Bidding Zone ({zone_key or country_code})
      type: hsilvosa/entsoe-day-ahead
    metrics:
    - name: MAE
      type: mae
      value: {metrics.mae:.3f}
    - name: RMSE
      type: rmse
      value: {metrics.rmse:.3f}
    - name: Empirical Interval Coverage (80% Nominal)
      type: coverage
      value: {metrics.interval_coverage * 100:.1f}%
---
"""

    features_str = "\n".join(f"- `{f}`" for f in feature_names)

    body = f"""# {target_name} Forecaster for {c_name} ({zone_key or country_code})

High-accuracy calibrated quantile LightGBM model for forecasting {c_name} **{target.value}**.
Resolution: {resolution}. Trained on multi-year data ({start_year}–{end_year}) from **ENTSO-E**,
featuring multi-scale lags, cyclical encodings, and **conformal calibration**
for prediction intervals targeting 80% coverage ($P10, P50, P90$).

## Model Highlights

- **Country / Zone**: {c_name} (`{zone_key or country_code}`)
- **Target**: {target_name} in `{unit}`
- **Resolution**: {resolution}
- **Outputs**: Point forecast ($P50$), 80% prediction interval ($P10$ to $P90$)
- **Algorithm**: LightGBM Multi-Quantile Regressor with Conformal Calibration & Monotonicity
- **Dataset**: ENTSO-E European Transparency Platform (Zone: `{zone_key or country_code}`)
- **Training Samples**: {training_rows:,} observations ({start_year}–{end_year})

## Performance & Benchmark Comparison

Evaluated on out-of-sample test sets against official seasonal persistence benchmarks:

| Metric | LightGBM Forecaster | 7-Day Seasonal Persistence | Improvement |
|---|---:|---:|---:|
| **MAE** | **{metrics.mae:.3f} {unit}** | {baseline_mae:.3f} {unit} | **{improvement:+.1f}%** |
| **RMSE** | **{metrics.rmse:.3f} {unit}** | — | — |
{wape_line}| **P10 Pinball Loss** | {metrics.pinball_p10:.3f} | — | — |
| **P90 Pinball Loss** | {metrics.pinball_p90:.3f} | — | — |
| **P10–P90 Interval Coverage** | **{metrics.interval_coverage * 100:.1f}%** | — | Target: 75–85% |
| **Winkler Score** | {metrics.winkler_score:.3f} | — | — |

## Quickstart: Python Inference

```python
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# 1. Download model artifacts
model_path = hf_hub_download(repo_id="ORGANIZATION/{model_id}", filename="models.joblib")
models = joblib.load(model_path)

# 2. Predict P10, P50 (point), and P90 quantiles
X_test = pd.read_csv("sample_input.csv")
p10 = models[0.1].predict(X_test)
p50 = models[0.5].predict(X_test)
p90 = models[0.9].predict(X_test)

print("Forecast Point Estimate:", p50[:5])
print("80% Lower Bound (P10):", p10[:5])
print("80% Upper Bound (P90):", p90[:5])
```

## Features Used

The model uses {len(feature_names)} leakage-safe features:
{features_str}

## Intended Use & Advisory

This model is intended for research, energy market analytics, grid load planning,
and educational forecasting demonstrations. It is advisory only and not intended
for automated trading execution or real-time grid dispatch.

## Citation & Attribution

Data published under the ENTSO-E Transparency framework:
- Transparency Platform: https://transparency.entsoe.eu/
"""
    return yaml_header + body


def export_model_to_hf_package(
    forecaster: LightGBMQuantileForecaster,
    target: EventType,
    output_dir: Path | str,
    metrics: ForecastMetrics,
    baseline_mae: float,
    training_rows: int,
    start_year: int,
    end_year: int,
    country_code: str = "ES",
    zone_key: str | None = None,
    sample_df: pd.DataFrame | None = None,
) -> Path:
    """Package model into a complete, standalone Hugging Face repository directory."""
    import joblib

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Save standard LightGBM text models & joblib dictionary
    joblib.dump(forecaster.models, out / "models.joblib")
    for q, m in forecaster.models.items():
        q_tag = f"q{int(q * 100)}"
        m.booster_.save_model(str(out / f"model_{q_tag}.txt"))

    # 2. Save metadata and calibrator configuration
    cal_info = {
        "is_fitted": forecaster.calibrator.is_fitted if forecaster.calibrator else False,
        "q_correction": forecaster.calibrator.q_correction if forecaster.calibrator else 0.0,
        "target_coverage": (
            forecaster.calibrator.target_coverage if forecaster.calibrator else 0.80
        ),
    }
    meta = {
        "target": target.value,
        "country_code": country_code,
        "zone_key": zone_key or country_code,
        "model_name": forecaster.model_name,
        "model_version": forecaster.model_version,
        "feature_names": forecaster.feature_names,
        "n_estimators": forecaster.n_estimators,
        "learning_rate": forecaster.learning_rate,
        "num_leaves": forecaster.num_leaves,
        "training_rows": training_rows,
        "start_year": start_year,
        "end_year": end_year,
        "metrics": metrics.to_dict(),
        "baseline_mae": baseline_mae,
        "calibrator": cal_info,
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 3. Generate Hugging Face Model Card (README.md)
    readme_content = generate_model_card_markdown(
        target=target,
        metrics=metrics,
        baseline_mae=baseline_mae,
        training_rows=training_rows,
        start_year=start_year,
        end_year=end_year,
        feature_names=forecaster.feature_names,
        country_code=country_code,
        zone_key=zone_key,
    )
    (out / "README.md").write_text(readme_content, encoding="utf-8")

    # 4. Generate standalone inference.py script inside package
    inference_code = f"""# Standalone inference helper for {target.value.capitalize()} Forecaster
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def load_forecaster(model_dir="."):
    path = Path(model_dir)
    config = json.loads((path / "config.json").read_text())
    models = joblib.load(path / "models.joblib")
    features = config["feature_names"]
    q_correction = config["calibrator"]["q_correction"] if config.get("calibrator") else 0.0

    def predict(df_features, apply_calibration=True):
        X = df_features[features]
        p10 = models[0.1].predict(X)
        p50 = models[0.5].predict(X)
        p90 = models[0.9].predict(X)
        stacked = np.sort(np.vstack([p10, p50, p90]), axis=0)
        p10, p50, p90 = stacked[0], stacked[1], stacked[2]
        if apply_calibration:
            p10 -= q_correction
            p90 += q_correction
            stacked_cal = np.sort(np.vstack([p10, p50, p90]), axis=0)
            p10, p50, p90 = stacked_cal[0], stacked_cal[1], stacked_cal[2]
        return pd.DataFrame({{"p10": p10, "p50_point": p50, "p90": p90}}, index=df_features.index)

    return predict
"""
    (out / "inference.py").write_text(inference_code, encoding="utf-8")

    # 5. Save sample input & output for validation if available
    if sample_df is not None and not sample_df.empty:
        sample_in = sample_df[forecaster.feature_names].head(24)
        sample_in.to_csv(out / "sample_input.csv", index=False)
        p10, p50, p90 = forecaster.predict(sample_in)
        sample_out = {
            "p10": [round(float(v), 2) for v in p10],
            "p50_point": [round(float(v), 2) for v in p50],
            "p90": [round(float(v), 2) for v in p90],
        }
        pred_json = json.dumps(sample_out, indent=2)
        (out / "sample_prediction.json").write_text(pred_json, encoding="utf-8")

    return out


def upload_model_to_hf(
    model_dir: Path | str,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Upload packaged model folder to Hugging Face Hub using HfApi."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload calibrated {Path(model_dir).name} model",
    )
    return f"https://huggingface.co/{repo_id}"
