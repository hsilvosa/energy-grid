from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelLineage:
    git_commit: str
    feature_schema_hash: str
    iceberg_snapshot_id: str
    training_window_start: str
    training_window_end: str
    source_versions: str


def log_lightgbm_candidate(
    model: Any,
    *,
    registered_model_name: str,
    target: str,
    metrics: dict[str, float],
    parameters: dict[str, str | int | float | bool],
    lineage: ModelLineage,
    tracking_uri: str,
) -> str:
    """Register a fully traceable challenger and return its model version."""
    import mlflow
    import mlflow.lightgbm

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run() as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"target": target, **asdict(lineage)})
        model_info = mlflow.lightgbm.log_model(
            model,
            name="model",
            registered_model_name=registered_model_name,
        )
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"run_id='{run.info.run_id}'")
        if not versions:
            raise RuntimeError("MLflow did not create a registered model version")
        version = str(versions[0].version)
        client.set_registered_model_alias(registered_model_name, "challenger", version)
        mlflow.set_tag("model_uri", model_info.model_uri)
        return version

