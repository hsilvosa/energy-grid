# Model Card: Spanish Demand and Day-Ahead Price Forecasts

## Purpose and current status

This document describes the two LightGBM models produced by the local real-data pipeline. Its purpose is to distinguish three separate facts:

- The source measurements are real ENTSO-E and Open-Meteo data.
- The training and serving pipeline works end to end.
- The current models have not yet been validated well enough for production, trading, or grid-operation decisions.

Both models must therefore be treated as **portfolio MVP candidates**. The API database currently uses the status `live` to mean that a model was trained and materialized from live-source data. It does not mean that the model passed a production quality gate.

The platform is advisory only. These forecasts must not be used to place market orders or issue grid-control instructions.

## Forecast products

| Target | Product | Resolution | Current output |
|---|---|---:|---:|
| Demand | Rolling forecast for the next six hours | 15 minutes | 24 intervals |
| Price | Next Spanish delivery day | 15 minutes | 92, 96, or 100 intervals depending on DST |

Each interval contains a point estimate and P10, P50, and P90 quantile estimates.

## Data used by the verified run

The documented run was issued on 12 August 2026 at 08:30 UTC. It downloaded approximately 45 days of history and produced:

| Source data | Parsed events |
|---|---:|
| ENTSO-E actual demand | 1,428 |
| ENTSO-E day-ahead price | 4,132 |
| ENTSO-E generation by type | 758 |
| Open-Meteo historical and live forecast variables | 2,112 |

Generation data was downloaded and can be published to Kafka, but it is **not yet used as a model feature**. Its presence in the ingestion result must not be interpreted as evidence that renewable generation or residual load is represented by the current models.

After resampling, lag construction, weather alignment, and removal of incomplete rows, the effective datasets were:

| Target | Complete rows | Training rows | Holdout rows | Approximate holdout duration |
|---|---:|---:|---:|---:|
| Demand | 756 | 605 | 151 | 37.75 hours |
| Price | 842 | 674 | 168 | 42 hours |

This is a very small evaluation sample. It does not cover multiple seasons, holidays, scarcity regimes, major renewable ramps, or enough price spikes.

## Features currently implemented

Both models use the same feature set:

- Local hour and quarter-hour.
- Day of week, month, and weekend indicator.
- Population-weighted temperature, wind speed, relative humidity, and shortwave radiation from Madrid, Barcelona, Valencia, Seville, and Bilbao.
- Target value at the same interval one day earlier (`lag_96`).
- Target value at the same interval seven days earlier (`lag_672`).
- Mean target value over the previous 96 intervals (`rolling_96`).

The training weather interface uses Open-Meteo historical forecast data rather than observed weather. This is the correct direction for reducing training-serving skew. However, the simplified live pipeline aligns records by valid timestamp and does not yet reproduce a fully revision-aware publication-time as-of dataset. A production evaluation must reconstruct exactly what was available at every historical forecast origin.

## Algorithm and baseline

For each target, three `LGBMRegressor` models are trained with quantile objectives at 0.1, 0.5, and 0.9. The P50 model is used as the point forecast. Current main parameters are 160 estimators, a learning rate of 0.05, and 31 leaves.

The reported comparison is seven-day seasonal persistence: the value from the same quarter-hour one week earlier. This is a useful emergency fallback, but it is a weak benchmark on its own. Beating it once is not sufficient evidence of model quality.

SARIMAX code exists in the repository, but the verified live run did not include SARIMAX in the reported comparison.

## Validation method used in the verified run

The rows are ordered chronologically. The last 20 percent, bounded to between 96 and 288 rows, forms one holdout block. A validation model is trained on the preceding rows and evaluated once on that block. A final model is then trained on all complete rows for materialization.

This avoids a random train/test split, but it is **not rolling-origin validation**. The result depends heavily on the conditions in one short period and cannot estimate performance stability.

## Verified metrics

| Metric | Demand | Price |
|---|---:|---:|
| LightGBM MAE | 861.055 MW | 29.629 EUR/MWh |
| LightGBM RMSE | 1,137.469 MW | 38.196 EUR/MWh |
| Seven-day persistence MAE | 1,315.523 MW | 57.918 EUR/MWh |
| MAE improvement over persistence | 34.5% | 48.8% |
| WAPE | 2.75% | Not used |
| P10-P90 empirical coverage | 34.4% | 45.8% |
| P10 pinball loss | 139.901 | 5.134 |
| P90 pinball loss | 582.835 | 7.361 |

MLflow run identifiers for this exact verification are:

- Demand: `b90dc9856eb3461bb323470f0efbf5c4`
- Price: `be00e26c68864d55be8c30156c1caadc`

The exact values will change when the source window and forecast origin change.

## Interpretation

### Demand

The demand result is promising for an MVP. A WAPE of 2.75% and a 34.5% MAE improvement over weekly persistence show that the model learned useful structure in this particular holdout.

It is not yet strong evidence of generalization. The model was trained on only 605 complete quarter-hour rows and tested on fewer than two days. Its nominal 80% interval covers only 34.4% of actual values, so uncertainty is severely underestimated.

### Price

The price result should not be described as good. It beats weekly persistence in the short holdout, but an MAE of 29.629 EUR/MWh and RMSE of 38.196 EUR/MWh leave large operational errors. Its P10-P90 interval covers only 45.8% of actual prices.

Electricity price formation depends on information not represented in the current feature set, including renewable and thermal availability, residual load, interconnector conditions, outages, fuel and carbon prices, and bidding behavior. Price spikes also require explicit regime evaluation.

### Prediction intervals

P10-P90 is intended to contain approximately 80% of observations over a representative sample. Coverage of 34.4% and 45.8% shows that both interval models are miscalibrated. Sorting the three independently trained quantile outputs prevents crossing in a single prediction, but it does not calibrate their probabilities.

## Known limitations

1. The effective training history is too short.
2. Validation uses one short holdout instead of many rolling forecast origins.
3. Downloaded generation data is not included in the feature table.
4. The price model lacks residual load, renewable forecasts, outages, interconnection, fuel, and carbon features.
5. The current feature table does not reconstruct source revisions and publication availability at every historical origin end to end.
6. Weather values are interpolated when gaps occur, but prediction records do not yet expose per-feature imputation flags.
7. The same generic feature design is used for two targets with very different market dynamics.
8. The day-ahead price model has no explicit horizon-step or forecast-origin feature.
9. Quantile estimates are not calibrated on an independent calibration window.
10. There are no reported metrics by horizon, hour, season, temperature regime, peak demand, renewable ramp, or price-spike slice.
11. The reported baseline is only weekly persistence; the live comparison does not yet include daily persistence, official ENTSO-E forecasts, or SARIMAX.
12. No inference should be made about performance outside the short verified August window.

## Required validation before promotion

A future model should remain `candidate` until all of the following are demonstrated:

- At least one full year of leakage-safe training data, preferably two or more years.
- Archived source versions and weather forecasts reconstructed as they were known at each origin.
- Rolling-origin evaluation across seasons with non-overlapping reporting periods.
- Comparisons against daily persistence, weekly persistence, SARIMAX, and relevant official forecasts where permitted.
- Metrics by horizon and by unusual-period slice: holidays, extreme temperatures, demand peaks, renewable ramps, and price spikes.
- A material improvement over the strongest baseline across folds, not just one aggregate block.
- P10-P90 empirical coverage close to its nominal 80%, assessed overall and by horizon.
- No critical slice regression greater than the configured 10% promotion limit.
- Shadow operation followed by evaluation against actuals before changing a champion alias.
- Reproducible source snapshot, feature schema, parameters, code revision, and model artifact lineage in MLflow.

Reasonable initial engineering targets, subject to review after longer backtests, are 75-85% P10-P90 coverage, consistent baseline improvement across monthly folds, and no severe deterioration during peak or spike periods. These are promotion gates for this project, not universal market-performance standards.

## Planned improvements

The highest-value next steps are:

1. Download one to two years of ENTSO-E data in bounded API requests and preserve raw responses.
2. Build a true publication-time feature snapshot for every forecast origin.
3. Add official load forecasts, generation forecasts by type, residual load, interconnection, outages, holidays, and renewable ramp features.
4. Give the price model explicit delivery horizon and market-timing features.
5. Implement rolling-origin backtests and persist fold, horizon, and regime metrics in MLflow.
6. Calibrate quantiles using a separate calibration period or conformal methods.
7. Rename operational model states so source freshness and validation status are separate fields.

## Reproducing and inspecting the run

Run the real-data pipeline:

```powershell
docker compose up -d --build
docker compose run --rm live-demo
```

Inspect forecasts at `http://localhost:8000/docs`, charts at `http://localhost:3000`, and run metrics and artifacts at `http://localhost:5000`.

When presenting the project, use the following wording:

> The platform is verified end to end with real ENTSO-E and Open-Meteo data. The current LightGBM models are MVP candidates. They outperform weekly persistence on one short chronological holdout, but their uncertainty intervals are poorly calibrated and broader rolling-origin validation is still required.
