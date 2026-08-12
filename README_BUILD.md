# Repository Build Journal

This document explains how the Energy Grid Intelligence repository was created, why the main design decisions were made, which problems appeared during implementation, and how to obtain an ENTSO-E API token.

It complements the main [README.md](README.md), which focuses on running and operating the finished platform.

## 1. Starting point and design decisions

The repository started empty, with no existing source files or commits. The first task was therefore to define a small but complete vertical slice instead of integrating with an existing application.

The initial decisions were:

- Target the Spanish ENTSO-E bidding zone.
- Build a portfolio MVP that runs locally but maps to AWS services later.
- Replay historical fixtures through Kafka instead of depending on live APIs for every demonstration.
- Use Apache Iceberg with a local Hadoop catalog and MinIO. AWS uses S3 and Glue Catalog.
- Produce rolling six-hour demand forecasts and next-day price forecasts.
- Materialize forecasts in PostgreSQL so FastAPI does not have to train a model or reconstruct features during an HTTP request.
- Use Grafana for both forecast presentation and operational monitoring.

These decisions kept the normal demo reproducible while preserving the interfaces needed for a small production deployment.

## 2. Project and dependency scaffolding

The Python package, dependency groups, test configuration, linting, and typing rules were defined in `pyproject.toml`. Python 3.11 and 3.12 are supported because this range is compatible with the selected Spark and scientific Python stack.

The base structure added:

- `src/energy_grid` for application code.
- `tests` for unit and contract tests.
- `data/fixtures` for credential-free source events.
- `monitoring` for Prometheus and Grafana configuration.
- `infra/terraform` for the AWS deployment path.
- `.github/workflows` for continuous integration and deployment.
- `.env.example` for local configuration without storing secrets.

The `.gitignore` excludes credentials, virtual environments, Terraform state, Spark checkpoints, local databases, model artifacts, and generated caches.

## 3. Canonical event and time model

The canonical `GridEvent` model was implemented before the source connectors or streaming jobs. Every event contains:

- A versioned schema and event identifier.
- Source, event type, bidding zone, unit, value, and optional dimension.
- Interval start and end timestamps.
- Publication and ingestion timestamps.
- Revision and quality status.
- Source request metadata and payload checksum.

All timestamps are normalized to UTC. The local Spanish delivery date, UTC offset, and interval number can be derived without discarding timezone information.

The delivery interval generator creates real UTC intervals between two Europe/Madrid midnights. This naturally produces:

- 92 intervals when daylight saving starts.
- 96 intervals on an ordinary day.
- 100 intervals when daylight saving ends.

This is safer than generating 96 local timestamps and trying to repair ambiguous or nonexistent hours afterward.

The Pydantic contract is accompanied by `schemas/grid-event-v1.json`, allowing non-Python producers to validate the same message envelope.

## 4. Source connectors and fixtures

The ENTSO-E connector was implemented for actual demand, generation by type, and day-ahead prices. It records a checksum and request parameters while deliberately excluding the API token from recorded metadata.

The Open-Meteo connector retrieves live or historical forecast weather and produces fixed population-weighted features from Madrid, Barcelona, Valencia, Seville, and Bilbao.

The default fixture contains demand, generation, price, weather, and a revised late demand measurement. It is intentionally small. The model demo expands a deterministic scenario into a longer training series instead of committing a large third-party dataset.

## 5. Kafka replay and failure injection

The fixture replayer maps each event type to its own Kafka topic. A seeded replay profile can inject:

- Duplicate messages.
- Missing messages.
- Out-of-order and late messages.
- Malformed payloads.
- Controlled replay speed.

Using a fixed random seed makes the same failure scenario reproducible in tests and demonstrations.

## 6. Spark and Iceberg processing

Spark Structured Streaming reads the Kafka topics and first appends every message to the Bronze Iceberg table. This preserves duplicates and malformed source payloads for audit and replay.

The Silver processing path:

- Parses the versioned event schema.
- Validates required fields, time ranges, event types, quality states, and units.
- Routes invalid messages to a dead-letter Iceberg table.
- Routes events arriving more than two hours late to a late-event table.
- Creates a deterministic business key.
- Keeps only the highest revision of a business key within a micro-batch.
- Uses an Iceberg `MERGE` so newer revisions replace older curated values.

The separate backfill job selects a bounded range from the late-event table and merges it into Silver. It stores a deterministic run ID, input checksum, record count, and before and after Iceberg snapshot IDs. If the same completed run is requested again, it exits without applying the correction twice.

## 7. Feature construction and forecasting

The repository contains publication-time as-of feature utilities and tests that demonstrate the required leakage rule. The simplified real-data training path currently aligns ENTSO-E targets and Open-Meteo historical forecast variables by valid timestamp. It does not yet reconstruct all source revisions exactly as they were available at every historical forecast origin. This distinction is recorded as a model limitation and must be resolved before production evaluation.

The forecasting layer includes:

- LightGBM global-horizon quantile models for P10, P50, and P90.
- A SARIMAX statistical benchmark.
- Seasonal persistence as the operational fallback concept.
- MAE, RMSE, WAPE, pinball loss, and interval coverage metrics.
- Promotion rules requiring at least a 2 percent primary-metric improvement.
- A critical-slice guard preventing regressions greater than 10 percent.
- A rollback controller requiring three consecutive error breaches above 120 percent of the reference error.

The intended MLflow registry flow stores model metrics, parameters, Git commit, feature schema hash, Iceberg snapshot, source versions, and training window. Registered candidates should receive the `challenger` alias, and production rollback should change the `champion` alias to the previous accepted version. The current real-data command logs metrics, lineage tags, and P10/P50/P90 model artifacts, but it does not yet register model versions or move registry aliases automatically.

## 8. Serving and monitoring

Forecasts are upserted into PostgreSQL and exposed through FastAPI. The response contains forecast provenance, quantiles, quality flags, age, and staleness.

The API exposes health checks, demand forecasts, next-day prices, model status, and Prometheus metrics. Grafana combines PostgreSQL forecast panels with Prometheus service and pipeline metrics.

The credential-free demo deliberately simulates three degraded evaluation windows. The stored model status then shows that the degraded model was replaced by the previous stable version.

## 9. Containers, AWS, and CI

Docker Compose defines Kafka in KRaft mode, MinIO, PostgreSQL, MLflow, FastAPI, Prometheus, Grafana, Kafka Exporter, and an optional Spark/Iceberg streaming service.

Terraform defines the AWS migration path:

- S3 and Glue Catalog for Iceberg.
- MSK Serverless for Kafka.
- EMR Serverless for Spark jobs.
- RDS PostgreSQL.
- ECR and ECS/Fargate.
- Secrets Manager, IAM, GitHub OIDC, CloudWatch, HTTPS load balancing, schedules, and a cost budget.

GitHub Actions separates validation from deployment. CI runs tests, Ruff, mypy, dependency auditing, image builds, Terraform validation, and infrastructure security scanning. Deployment is manual and uses an immutable Git commit image plus GitHub OIDC.

## 10. Problems encountered during implementation

### Empty repository

There was no application or configuration to extend. The solution was to establish data contracts and time semantics first, then add source, processing, modeling, serving, and deployment layers around those contracts.

### Git repository ownership warning

The sandbox user did not own the repository, so Git reported dubious ownership. Repository inspection used a per-command `safe.directory` setting. The global Git configuration was not changed.

### Python runtime mismatch

The machine's default interpreter was Python 3.14, while the project targets Python 3.11-3.12 for Spark and scientific-library compatibility. Python 3.12 was available through the Windows application installation, but sandbox access initially prevented creation and execution of the virtual environment. The environment was created with the required permission and kept inside `.venv`. Normal users do not need this environment because the recommended real-data path runs Python inside Docker.

### Dependency installation timeout

The first installation pass timed out while downloading the full MLflow and scientific stack. It had installed only part of the dependency graph. Running the same declared installation again safely resumed from the local package cache and completed successfully.

### Strict typing and third-party stubs

Pandas, Statsmodels, Spark, and Confluent Kafka do not all expose complete type information. Mypy was kept in strict mode for project code, with narrow missing-import overrides for these third-party modules. Missing return and parameter annotations in Spark entry points were added rather than disabling strict checking globally.

### DST interval counts

Assuming 96 price intervals per local day would fail twice each year. The implementation derives intervals from UTC instants between Spanish local midnights and includes tests for both 92- and 100-interval days.

### Duplicate revisions in one Spark micro-batch

An Iceberg `MERGE` becomes ambiguous when multiple source rows match the same target key. The replay scenario can produce both duplicates and revisions in a single micro-batch. Silver processing therefore ranks rows by revision and ingestion time and merges only the highest-ranked row, while Bronze still preserves every message.

### Weather feature leakage

Observed weather can make historical model scores unrealistically good because the live system only knows forecast weather for future intervals. The design uses historical weather forecasts for training and publication-time as-of joins for all source features.

### Local and AWS Iceberg catalogs

Running AWS Glue locally would make the demo unnecessarily difficult. The Spark configuration isolates the catalog choice: local execution uses the Hadoop catalog with MinIO, while AWS uses Glue with S3.

### ENTSO-E day-ahead price requests

The first real price request returned HTTP 400 even though the same token worked for demand. ENTSO-E's price endpoint requires both `in_Domain` and `out_Domain` for the Spanish bidding zone. Supplying both parameters fixed the request. HTTP error handling was also changed so a request URL containing the security token is never copied into an exception message.

### Gaps in operational source series

Real ENTSO-E documents do not always form the perfectly complete matrix produced by fixtures. After weekly lag creation and publication-time alignment, a 45-day request produced 756 eligible demand rows rather than the original hard-coded minimum of 768. The minimum was changed to five complete days after feature construction, while missing values remain explicit and carry quality metadata.

### MLflow client and server compatibility

The first container run reached ENTSO-E, Open-Meteo, Kafka, and PostgreSQL but failed while logging models. The application used MLflow 3.15.1 while the server image used 3.1.4, so their database expectations differed. Pinning both sides to 3.15.1 fixed the schema mismatch. MLflow 3.15 also rejects unknown HTTP Host headers by default; the local server now explicitly permits only `mlflow:5000`, localhost, and loopback hosts.

### Optional Spark image defaults

The Spark/Iceberg image starts several Spark services through an entrypoint and evaluates the submitted command as one shell argument. The Compose command therefore had to be supplied as one folded value. The image also defaults to a bundled REST catalog named `demo`; the application explicitly selects the local Hadoop catalog backed by MinIO. Kafka and S3 connector packages are declared on `spark-submit` because they are not included in the base runtime.

This work made it clear that Spark, Kafka, and Iceberg add substantial local startup cost. They are now an optional `streaming` profile. The recommended real-data path uses PostgreSQL directly and keeps the distributed path for demonstrations that specifically need event-time and lakehouse behavior.

## 11. Verification completed

The final local verification included:

- Real authenticated ENTSO-E demand, price, and generation downloads.
- Real Open-Meteo historical and live forecast downloads.
- 1,428 demand, 4,132 price, 758 generation, and 2,112 weather source events in the verified run.
- 24 six-hour demand forecasts and 96 next-day price forecasts in PostgreSQL and FastAPI.
- Demand validation MAE of 861.055 MW versus a 1,315.523 MW seasonal baseline.
- Price validation MAE of 29.629 EUR/MWh versus a 57.918 EUR/MWh seasonal baseline.
- Two MLflow runs containing P10, P50, and P90 LightGBM artifacts and source lineage.
- Kafka publication of the real canonical events when the optional flag was enabled.
- 29 passing tests after the real-data integration and Compose simplification.
- Clean Ruff linting.
- Strict mypy success across the source package.
- Valid JSON Schema and Grafana dashboard JSON.
- Valid Docker Compose configuration.

The exact forecast metrics change when the live data window changes. They are recorded here to distinguish a verified real run from the deterministic fixture demonstration, not as a stable model-performance claim.

The short holdout results are not sufficient to call either model production-ready. Demand showed useful point-forecast performance but poor interval coverage. Price error and interval coverage remain weak. The full evaluation, interpretation, missing features, and promotion requirements are documented in [MODEL_CARD.md](MODEL_CARD.md).

## 12. Getting an ENTSO-E API token

ENTSO-E does not currently issue a token immediately when an account is created. Web API access must first be enabled for the account by ENTSO-E.

1. Open the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) and create an account using the registration option in the user menu.
2. Confirm the account and sign in successfully.
3. Send an email to `transparency@entsoe.eu`.
4. Use `Restful API access` as the email subject.
5. In the email body, state that you are requesting Web API access and include the exact email address used to register the Transparency Platform account.
6. Wait for ENTSO-E to enable access. This is a manual administrative step and may not be immediate.
7. Sign in again and open the account settings.
8. Find the `Web API Security Token` section and select `Generate and overwrite`.
9. Copy the generated token before closing the dialog. ENTSO-E states that it is shown only once; generating another token invalidates the previous one.

These steps follow ENTSO-E's current [API Token Management guidance](https://transparency.entsoe.eu/content/static_content/download?path=%2FStatic+content%2FAPI-Token-Management.pdf).

### Configure the token locally

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set:

```dotenv
ENTSOE_TOKEN=your_token_here
```

Alternatively, set it for the current PowerShell session:

```powershell
$env:ENTSOE_TOKEN = "your_token_here"
```

The application reads this value through `Settings.entsoe_token`. The ENTSO-E connector sends it as the `securityToken` request parameter but removes it from stored request metadata.

Never commit `.env`, paste the token into logs, include it in screenshots, or store it in fixture data. The repository already ignores `.env`. In AWS, store the token in Secrets Manager and inject it into the relevant ingestion task.

If the `Web API Security Token` section is missing after registration, API access has not yet been enabled. Recheck that the request email used the registered address and the required subject before contacting ENTSO-E support again.
