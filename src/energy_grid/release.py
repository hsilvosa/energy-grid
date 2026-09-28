from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from energy_grid.domain import EventType, ForecastRecord, GridEvent
from energy_grid.forecasting import LightGBMQuantileForecaster
from energy_grid.live_pipeline import (
    TrainingResult,
    _forecast_intervals,
    build_future_frame,
    events_to_series,
    source_snapshot_id,
)
from energy_grid.rollback import promotion_gate
from energy_grid.storage import ForecastStore

DEFAULT_AREA = "10YES-REE------0"
DEFAULT_PRODUCT = "operational"


def _safe_component(value: str, name: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"invalid {name}: {value!r}")
    return value


@dataclass(frozen=True)
class ReleaseDecision:
    promoted: bool
    reason: str
    version: str
    artifact_uri: str
    forecasts: list[ForecastRecord]


def save_candidate_artifact(
    result: TrainingResult,
    artifact_root: Path,
    *,
    area: str = DEFAULT_AREA,
    forecast_product: str = DEFAULT_PRODUCT,
) -> str:
    """Persist a candidate under an immutable, product-specific directory."""
    area = _safe_component(area, "area")
    forecast_product = _safe_component(forecast_product, "forecast_product")
    version = _safe_component(result.model.model_version, "model_version")
    destination = (
        artifact_root / area / result.target.value / forecast_product / version
    ).resolve()
    if destination.exists():
        raise FileExistsError(f"model artifact already exists: {destination}")
    temporary = destination.with_name(f".{version}.tmp-{uuid4().hex}")
    try:
        result.model.save(temporary)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return str(destination)


def evaluate_and_record_candidate(
    store: ForecastStore,
    result: TrainingResult,
    artifact_root: Path,
    *,
    area: str = DEFAULT_AREA,
    forecast_product: str = DEFAULT_PRODUCT,
    minimum_improvement: float = 0.02,
) -> ReleaseDecision:
    """Record a candidate and promote it only when the release gate passes."""
    artifact_uri = save_candidate_artifact(
        result, artifact_root, area=area, forecast_product=forecast_product
    )
    current = store.get_model_deployment(
        area=area, target=result.target, forecast_product=forecast_product
    )
    reference_error = (
        float(current["champion_error"])
        if current and current["champion_error"] is not None
        else result.baseline_mae
    )
    gate = promotion_gate(
        champion_error=reference_error,
        challenger_error=result.metrics.mae,
        slice_regressions=[],
        minimum_improvement=minimum_improvement,
    )
    promoted = gate.promote
    champion_version = result.model.model_version if promoted else (
        current["champion_version"] if current else None
    )
    champion_uri = artifact_uri if promoted else (
        current["champion_artifact_uri"] if current else None
    )
    store.set_model_deployment(
        area=area,
        target=result.target,
        forecast_product=forecast_product,
        champion_version=champion_version,
        champion_artifact_uri=champion_uri,
        previous_version=current["champion_version"] if promoted and current else None,
        previous_artifact_uri=(
            current["champion_artifact_uri"] if promoted and current else None
        ),
        previous_error=current["champion_error"] if promoted and current else None,
        candidate_version=result.model.model_version,
        candidate_artifact_uri=artifact_uri,
        champion_error=result.metrics.mae if promoted else (
            current["champion_error"] if current else None
        ),
        candidate_error=result.metrics.mae,
        baseline_error=result.baseline_mae,
        last_evaluated_at=datetime.now(UTC),
        status="champion" if promoted else "candidate_rejected",
    )
    forecasts = [
        item.model_copy(update={"area": area, "is_shadow": not promoted})
        for item in result.forecasts
    ]
    return ReleaseDecision(
        promoted=promoted,
        reason=gate.reason,
        version=result.model.model_version,
        artifact_uri=artifact_uri,
        forecasts=forecasts,
    )


def rollback_model_deployment(
    store: ForecastStore,
    *,
    area: str = DEFAULT_AREA,
    target: EventType,
    forecast_product: str = DEFAULT_PRODUCT,
) -> None:
    """Restore the previous approved artifact for one forecast product."""
    current = store.get_model_deployment(
        area=area, target=target, forecast_product=forecast_product
    )
    if not current or not current["previous_version"] or not current["previous_artifact_uri"]:
        raise RuntimeError(f"no previous model for {area}/{target.value}/{forecast_product}")
    previous_artifact = Path(current["previous_artifact_uri"])
    if not previous_artifact.is_dir():
        raise RuntimeError(f"previous model artifact is unavailable: {previous_artifact}")
    store.set_model_deployment(
        area=area,
        target=target,
        forecast_product=forecast_product,
        champion_version=current["previous_version"],
        champion_artifact_uri=current["previous_artifact_uri"],
        previous_version=current["champion_version"],
        previous_artifact_uri=current["champion_artifact_uri"],
        previous_error=current["champion_error"],
        champion_error=current["previous_error"],
        last_evaluated_at=datetime.now(UTC),
        status="rolled_back",
    )


def forecast_with_approved_model(
    store: ForecastStore,
    target_events: list[GridEvent],
    live_weather: list[GridEvent],
    target: EventType,
    *,
    area: str = DEFAULT_AREA,
    forecast_product: str = DEFAULT_PRODUCT,
    issue_time: datetime | None = None,
) -> list[ForecastRecord]:
    """Generate forecasts from the approved artifact without training a model."""
    deployment = store.get_model_deployment(
        area=area, target=target, forecast_product=forecast_product
    )
    if not deployment or not deployment["champion_artifact_uri"]:
        raise RuntimeError(f"no approved model for {area}/{target.value}/{forecast_product}")
    artifact = Path(deployment["champion_artifact_uri"])
    if not artifact.is_dir():
        raise RuntimeError(f"approved model artifact is unavailable: {artifact}")
    model = LightGBMQuantileForecaster.load(artifact)
    issue = (issue_time or datetime.now(UTC)).astimezone(UTC)
    intervals = _forecast_intervals(target, issue)
    history = events_to_series(target_events, target)
    future = build_future_frame(
        intervals, history, live_weather, available_at=issue
    )
    p10, p50, p90 = model.predict(future)
    snapshot_id = source_snapshot_id(target_events, live_weather)
    unit = "MW" if target == EventType.DEMAND else "EUR/MWh"
    return [
        ForecastRecord(
            target=target,
            area=area,
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
            data_snapshot_id=snapshot_id,
            quality_flags=["approved_model", "real_source_data"],
        )
        for index, (start, end) in enumerate(intervals)
    ]
