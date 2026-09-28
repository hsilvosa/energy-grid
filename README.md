# Pan-European Energy Grid Intelligence and AI Forecasting Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![ML Stack](https://img.shields.io/badge/ML-LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-success.svg)](https://github.com/)
[![Lakehouse](https://img.shields.io/badge/Lakehouse-Apache%20Iceberg%20%7C%20PySpark-orange.svg)](https://iceberg.apache.org/)
[![Streaming](https://img.shields.io/badge/Streaming-Apache%20Kafka-red.svg)](https://kafka.apache.org/)
[![API](https://img.shields.io/badge/API-FastAPI%20%7C%20PostgreSQL-009688.svg)](https://fastapi.tiangolo.com/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Hub%20%26%20Spaces-yellow.svg)](https://huggingface.co/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked](https://img.shields.io/badge/type%20checked-mypy%20strict-blue.svg)](https://mypy-lang.org/)

A Python project for European electricity data ingestion, forecasting, evaluation, and serving. It includes a local synthetic demo and tools for working with ENTSO-E historical data when that dataset is available.

The repository includes streaming and lakehouse components, quantile models, chronological backtesting, a FastAPI service, and a Gradio demo. Historical forecast evaluation now uses an explicit forecast origin and lead time. Source publication times are needed to verify availability of real measurements at each origin.

---

## System Architecture

```mermaid
flowchart TD
    subgraph INGESTION["Ingestion and Event Streaming"]
        ENTSOE["ENTSO-E API / 10-Yr Parquet Dataset<br/>(34 Countries, 48 Bidding Zones)"]
        METEO["Open-Meteo Weather API<br/>(ECMWF/IFS Temperature and Wind)"]
        KAFKA["Apache Kafka Event Bus<br/>(Strict JSON Schema Contract)"]
        DLQ["Dead-Letter Queue<br/>(Late and Corrupted Records)"]
        ENTSOE --> KAFKA
        METEO --> KAFKA
        KAFKA -->|Schema Violations| DLQ
    end

    subgraph LAKEHOUSE["Apache Iceberg Medallion Lakehouse"]
        SPARK["PySpark Streaming Engine"]
        BRONZE["Bronze Layer: Raw Immutable Append Log"]
        SILVER["Silver Layer: Deduplicated and As-Of Aligned"]
        GOLD["Gold Layer: Multi-Scale Feature Store"]
        KAFKA --> SPARK
        SPARK --> BRONZE
        BRONZE --> SILVER
        SILVER --> GOLD
    end

    subgraph FORECASTING["Multi-Model Forecasting and Uncertainty Calibration"]
        GOLD --> DUAL["Dual-Horizon Forecaster<br/>(Short-Term 1-6h + Long-Term 24-48h)"]
        GOLD --> STACKED["Stacked Multi-Model Ensemble<br/>(LightGBM + XGBoost + CatBoost)"]
        CONFORMAL["Split-Conformal Prediction Calibrator<br/>(80% Nominal P10-P90 Interval)"]
        DUAL --> CONFORMAL
        STACKED --> CONFORMAL
    end

    subgraph SERVING["Production Serving and User Interfaces"]
        POSTGRES[("PostgreSQL Forecast Sink")]
        FASTAPI["FastAPI REST and Telemetry Service"]
        GRAFANA["Grafana Analytics Dashboards"]
        GRADIO["Gradio and Hugging Face Spaces UI"]
        HF_HUB["Hugging Face Model Hub (16 Packages)"]
        CONFORMAL --> POSTGRES
        POSTGRES --> FASTAPI
        FASTAPI --> GRAFANA
        CONFORMAL --> GRADIO
        CONFORMAL --> HF_HUB
    end
```

---

## Pan-European Coverage (34 Countries and 48 Bidding Zones)

The reader supports the following ENTSO-E market identifiers when the corresponding parquet data is supplied:

| Region | Countries and Bidding Zones | Target Variables | Native Resolutions |
|---|---|---|---|
| **Western Europe** | Spain (`ES`), France (`FR`), Germany (`DE`, `DE_LU`, `DE_AT_LU`), Portugal (`PT`), Netherlands (`NL`), Belgium (`BE`), Austria (`AT`), Switzerland (`CH`), Ireland (`IE_SEM`), Luxembourg (`LU`) | Electricity Demand (MW)<br/>Day-Ahead Price (EUR/MWh) | 15-min / 60-min |
| **Southern Europe and Nordics** | Italy (`IT_NORD`, `IT_CNOR`, `IT_CSUD`, `IT_SUD`, `IT_SICI`, `IT_SARD`), Greece (`GR`), Norway (`NO_1`..`NO_5`), Sweden (`SE_1`..`SE_4`), Finland (`FI`), Denmark (`DK_1`, `DK_2`) | Electricity Demand (MW)<br/>Day-Ahead Price (EUR/MWh) | 15-min / 60-min |
| **Central and Eastern Europe** | Poland (`PL`), Czechia (`CZ`), Romania (`RO`), Hungary (`HU`), Slovakia (`SK`), Slovenia (`SI`), Croatia (`HR`), Bulgaria (`BG`), Serbia (`RS`), Bosnia (`BA`), Montenegro (`ME`), North Macedonia (`MK`), Albania (`AL`), Kosovo (`XK`), Estonia (`EE`), Latvia (`LV`), Lithuania (`LT`), Cyprus (`CY`) | Electricity Demand (MW)<br/>Day-Ahead Price (EUR/MWh) | 15-min / 60-min |

---

## Evaluating forecasts

The earlier benchmark table was removed because its test-window features used observations that would not be available for a day-ahead forecast. Regenerate results with the current backtest before quoting model gains or interval coverage.

The backtest makes predictions at a fixed lead time and excludes training labels that would still be unknown at the first test origin. It reports MAE, RMSE, baseline error, interval coverage, and slice results. For example:

```powershell
python -m energy_grid.cli backtest --data-dir "PATH_TO_ENTSOE_DATA" --country-code ES --target demand --horizon-hours 24 --start-year 2024 --end-year 2026 --output-report reports/es_demand_24h.md --output-json reports/es_demand_24h.json
```

Run separate backtests for each country, target, and lead time (for example 1, 24, and 48 hours). Generate a comparison table from the resulting JSON files with `python scripts/benchmark_table.py reports/*.json`. The ENTSO-E parquet reader currently has observation timestamps but no publication timestamps. These results assume a measurement is available by its timestamp; delayed publication must be modeled before treating results as a fully operational simulation. Calibration targets 80% coverage, but measured holdout coverage can differ. Local calendar time follows the selected country; the current holiday flag is implemented for Spain only and is zero elsewhere.

---

## Key Modeling Strategies

### 1. Dual-Horizon Hybrid Architecture
Power grid dynamics operate under two distinct physical regimes:
- **Near-term Intraday (1-6 Hours)**: Governed by thermal inertia, real-time power dispatch deviation, and high-frequency momentum. Features: `lag_1h`, `lag_2h`, `diff_1h`, `acceleration_1h`, `ema_4step`, `rolling_std_4step`.
- **Day-Ahead (24-48 Hours)**: Governed by diurnal solar patterns, industrial workday schedules, weather degree days (HDD/CDD), and calendar effects. Features: `lag_24h`, `lag_48h`, `lag_7d`, `sin/cos` calendar harmonics, `is_holiday`, `is_morning_peak`, `is_evening_peak`.
- **Dynamic Transition Function**: Smoothly blends predictions across forecast lead-time steps $h \in [1, H]$ using linear decay:
  $$\alpha(h) = \max\left(0, 1 - \frac{h-1}{h_{\text{switch}}}\right), \quad \hat{y}(h) = \alpha(h) \hat{y}_{\text{short}}(h) + (1 - \alpha(h)) \hat{y}_{\text{long}}(h)$$

### 2. Multi-Model Quantile Ensembling
- **LightGBM Quantile**: Fast leaf-wise tree growth with exact quantile pinball objective.
- **XGBoost Quantile**: Histogram-partitioned tree boosting with `objective="reg:quantileerror"`.
- **CatBoost Quantile**: Symmetric oblivious decision trees with `loss_function="Quantile:alpha"`.
- **Ensemble Blending**: Fixed weighted blending ($0.45 \cdot \text{LGBM} + 0.35 \cdot \text{XGB} + 0.20 \cdot \text{CAT}$) followed by calibration of the combined predictions on a chronological holdout.

---

## Quickstart and Execution Guide

### 1. Environment Setup

Run the following commands to install dependencies:

```powershell
# Clone the repository
git clone https://github.com/hsilvosa/energy-grid.git
cd energy-grid

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install in editable mode
pip install -e ".[dev,demo]"
```

> **Note on running commands**: You can run CLI commands either via `python -m energy_grid.cli <command>` (recommended, works in any activated Python environment) or directly via `energy-grid <command>` (available after `pip install -e .`).

### 2. Try the demo without external data

Create deterministic synthetic demand and price data, then open the Gradio app. The sample covers Spain only and is for checking the workflow, not measuring real-world accuracy.

```powershell
python -m energy_grid.cli create-sample-data
python -m energy_grid.cli launch-demo --data-dir data/runtime/sample --port 7860
```

Open `http://localhost:7860` and choose the LightGBM model. The results show the 24-hour forecast lead, origins, latest available observation, and previous-day baseline.

### 3. Launch with your own ENTSO-E dataset

Launch the interactive web application featuring country selection, model strategy selection, and Plotly ribbon charts:

```powershell
python -m energy_grid.cli launch-demo --data-dir "PATH_TO_ENTSOE_DATA" --port 7860
```

### 4. European Bidding Zones and Dataset Exploration

```powershell
# List the bidding zones present in your dataset
python -m energy_grid.cli list-zones --data-dir "PATH_TO_ENTSOE_DATA"

# Summarize historical data coverage for any European country
python -m energy_grid.cli summarize-dataset --data-dir "PATH_TO_ENTSOE_DATA" --country-code FR
python -m energy_grid.cli summarize-dataset --data-dir "PATH_TO_ENTSOE_DATA" --country-code DE --zone-key DE_LU
python -m energy_grid.cli summarize-dataset --data-dir "PATH_TO_ENTSOE_DATA" --country-code ES
```

### 5. Model Training and Conformal Calibration

```powershell
# Train Dual-Horizon Hybrid model on Spanish Demand (15-min intervals)
python -m energy_grid.cli train-historical --data-dir "PATH_TO_ENTSOE_DATA" --country-code ES --target demand --model-type dual_horizon

# Train Stacked Multi-Model Ensemble on Spanish Day-Ahead Price
python -m energy_grid.cli train-historical --data-dir "PATH_TO_ENTSOE_DATA" --country-code ES --target price --model-type stacked

# Train models for France, Germany, Italy, Netherlands, Belgium, Poland
python -m energy_grid.cli train-historical --data-dir "PATH_TO_ENTSOE_DATA" --country-code FR --target demand --model-type dual_horizon
python -m energy_grid.cli train-historical --data-dir "PATH_TO_ENTSOE_DATA" --country-code DE --zone-key DE_LU --target price --model-type dual_horizon
```

### 6. Multi-Year Rolling-Origin Backtesting

```powershell
# Run rolling-origin evaluation with a 24-hour lead and slice reports
python -m energy_grid.cli backtest --data-dir "PATH_TO_ENTSOE_DATA" --country-code ES --target demand --start-year 2024 --end-year 2026 --output-report "reports/backtest_demand_es.md"
python -m energy_grid.cli backtest --data-dir "PATH_TO_ENTSOE_DATA" --country-code FR --target demand --start-year 2024 --end-year 2026 --output-report "reports/backtest_demand_fr.md"
```

---

## Hugging Face Model Hub Ecosystem

Model packages can be generated locally when the corresponding market data is available. Portable export currently supports LightGBM:

```powershell
python -m energy_grid.cli export-hf --data-dir "PATH_TO_ENTSOE_DATA" --country-code ES --target demand --model-type lightgbm --horizon-hours 24 --output-dir hf_models/es-demand
```


## High-Level Python Forecaster API

Use trained European forecasters downstream in any Python application:

```python
from energy_grid.forecaster import EuropeanElectricityForecaster
import pandas as pd

# Load the local package created by export-hf
forecaster = EuropeanElectricityForecaster.load("hf_models/es-demand")

# Predict calibrated P10, P50, P90 intervals
forecast_df = forecaster.predict(df_features, apply_calibration=True)
print(forecast_df[["p10", "p50", "p90"]].head())

# Inspect top feature importances
print(list(forecaster.feature_importances().items())[:5])
```

---

## Docker Stack and Operational Infrastructure

### Model release workflow

Production forecasting separates candidate training from inference:

1. `energy-grid live-demo --no-publish-kafka --no-register-mlflow` trains and evaluates
   candidates. A candidate must improve on the current champion (or the seven-day baseline for
   the first release) before promotion. Rejected candidate forecasts are stored as shadow data.
2. Approved artifacts are written beneath `MODEL_ARTIFACT_ROOT`. ECS mounts an encrypted,
   backed-up EFS filesystem at `/models`, allowing training and inference tasks to share the same
   immutable artifact.
3. `energy-grid live-inference --target demand` and `--target price` load the approved artifact
   recorded for that area, target, and forecast product. They never train or promote a model.
4. `/v1/models/status` exposes both legacy status records and the market-specific deployment
   records used by the release workflow.

Use `energy-grid rollback-model --target demand` (or `price`) to atomically restore the prior
approved artifact and its evaluation score. The command fails without changing release state
when no prior artifact is available.

Set `entsoe_token_secret_arn` when applying Terraform to enable the weekly candidate-training
schedule and the demand and price inference schedules. The secret value must contain the raw
ENTSO-E token. The deployment workflow registers a new ECS task-definition revision using the
Git-SHA image, waits for service stability, and optionally checks `API_HEALTHCHECK_URL`.

```bash
# Spin up Kafka, Spark, PostgreSQL, Grafana, and Prometheus
docker compose up -d

# Verify streaming pipeline health
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

- **FastAPI Documentation**: `http://localhost:8000/docs`
- **Grafana Monitoring**: `http://localhost:3000` (pre-configured dashboards for forecast vs actuals, WAPE, and Kafka consumer lag)
- **Prometheus Telemetry**: `http://localhost:9090`

---

## Testing and Code Quality

The codebase enforces linting, type checking, and automated tests:

```powershell
# Run unit and contract test suite
pytest -v

# Run linter and formatting checks
ruff check src tests

# Run strict type checking across the source package
mypy src
```

---

## License

This project is licensed under the Apache License 2.0. Data provided by the ENTSO-E Transparency Platform and Open-Meteo.
