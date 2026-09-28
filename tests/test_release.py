from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from energy_grid.domain import EventType, ForecastRecord
from energy_grid.forecasting import ForecastMetrics, LightGBMQuantileForecaster
from energy_grid.live_pipeline import TrainingResult
from energy_grid.release import evaluate_and_record_candidate, rollback_model_deployment
from energy_grid.storage import ForecastStore


class FakeModel:
    model_name = "fake-quantile"

    def __init__(self, version: str) -> None:
        self.model_version = version

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True)
        (directory / "model.txt").write_text(self.model_version, encoding="utf-8")


def result(version: str, mae: float, baseline: float = 100.0) -> TrainingResult:
    issue = datetime(2026, 1, 1, tzinfo=UTC)
    forecast = ForecastRecord(
        target=EventType.DEMAND,
        issue_time=issue,
        valid_from=issue + timedelta(minutes=15),
        valid_to=issue + timedelta(minutes=30),
        point=100.0,
        p10=90.0,
        p50=100.0,
        p90=110.0,
        unit="MW",
        model_name="fake-quantile",
        model_version=version,
        data_snapshot_id="snapshot",
    )
    metrics = ForecastMetrics(
        mae=mae,
        rmse=mae,
        wape=0.1,
        pinball_p10=1.0,
        pinball_p90=1.0,
        interval_coverage=0.8,
    )
    return TrainingResult(
        target=EventType.DEMAND,
        model=cast(LightGBMQuantileForecaster, FakeModel(version)),
        forecasts=[forecast],
        metrics=metrics,
        baseline_mae=baseline,
        training_rows=1000,
        training_start=issue - timedelta(days=30),
        training_end=issue - timedelta(days=1),
        snapshot_id="snapshot",
    )


def test_release_gate_preserves_champion_and_shadows_rejected_candidate(tmp_path: Path) -> None:
    store = ForecastStore(f"sqlite:///{tmp_path / 'release.db'}")
    store.create_schema()
    first = evaluate_and_record_candidate(store, result("v1", 80.0), tmp_path / "models")
    assert first.promoted
    assert not first.forecasts[0].is_shadow

    second = evaluate_and_record_candidate(store, result("v2", 85.0), tmp_path / "models")
    assert not second.promoted
    assert second.forecasts[0].is_shadow
    deployment = store.get_model_deployment(
        area="10YES-REE------0", target=EventType.DEMAND, forecast_product="operational"
    )
    assert deployment is not None
    assert deployment["champion_version"] == "v1"
    assert deployment["candidate_version"] == "v2"
    assert deployment["status"] == "candidate_rejected"


def test_release_state_is_isolated_by_area(tmp_path: Path) -> None:
    store = ForecastStore(f"sqlite:///{tmp_path / 'areas.db'}")
    store.create_schema()
    root = tmp_path / "models"
    evaluate_and_record_candidate(store, result("es-v1", 80.0), root, area="ES")
    evaluate_and_record_candidate(store, result("fr-v1", 70.0), root, area="FR")
    deployments = store.model_deployments()
    assert {(item["area"], item["champion_version"]) for item in deployments} == {
        ("ES", "es-v1"),
        ("FR", "fr-v1"),
    }


def test_rollback_restores_previous_artifact_and_error(tmp_path: Path) -> None:
    store = ForecastStore(f"sqlite:///{tmp_path / 'rollback.db'}")
    store.create_schema()
    root = tmp_path / "models"
    evaluate_and_record_candidate(store, result("v1", 80.0), root)
    evaluate_and_record_candidate(store, result("v2", 70.0), root)

    rollback_model_deployment(store, target=EventType.DEMAND)

    deployment = store.get_model_deployment(
        area="10YES-REE------0",
        target=EventType.DEMAND,
        forecast_product="operational",
    )
    assert deployment is not None
    assert deployment["champion_version"] == "v1"
    assert deployment["champion_error"] == 80.0
    assert deployment["previous_version"] == "v2"
    assert deployment["previous_error"] == 70.0
    assert deployment["status"] == "rolled_back"
