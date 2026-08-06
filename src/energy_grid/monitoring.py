from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

EVENTS_PUBLISHED = Counter(
    "energy_events_published_total", "Published grid events", ["topic"]
)
EVENTS_LATE = Counter("energy_late_events_total", "Events received after the watermark", ["source"])
EVENTS_DEAD_LETTER = Counter(
    "energy_dead_letter_events_total", "Events rejected by validation", ["reason"]
)
SOURCE_FRESHNESS_SECONDS = Gauge(
    "energy_source_freshness_seconds", "Age of the newest source event", ["source"]
)
FORECAST_ERROR = Gauge(
    "energy_forecast_error", "Latest forecast error", ["target", "metric", "horizon"]
)
FORECAST_ERROR_RATIO = Gauge(
    "energy_forecast_error_ratio",
    "Champion forecast error divided by baseline error",
    ["target", "horizon"],
)
MODEL_ROLLBACK_STATE = Gauge(
    "energy_model_rollback_state",
    "Whether the persisted model status is rolled back",
    ["target"],
)
FEATURE_DRIFT = Gauge(
    "energy_feature_drift_psi", "Population stability index by feature", ["feature"]
)
ROLLBACKS = Counter("energy_model_rollbacks_total", "Automatic model rollbacks", ["target"])
API_LATENCY = Histogram(
    "energy_api_request_duration_seconds",
    "FastAPI request duration",
    ["method", "path", "status"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
