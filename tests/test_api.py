from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from energy_grid.api import create_app
from energy_grid.config import Settings
from energy_grid.domain import EventType, ForecastRecord
from energy_grid.storage import ForecastStore
from energy_grid.time_utils import delivery_intervals


def forecast(target: EventType, start: datetime, issue: datetime) -> ForecastRecord:
    return ForecastRecord(
        target=target,
        issue_time=issue,
        valid_from=start,
        valid_to=start + timedelta(minutes=15),
        point=100.0,
        p10=90.0,
        p50=100.0,
        p90=110.0,
        unit="MW" if target == EventType.DEMAND else "EUR/MWh",
        model_name="test",
        model_version="v1",
        data_snapshot_id="snapshot-1",
    )


def test_health_and_forecast_contracts(tmp_path: Path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'api.db'}")
    store = ForecastStore(settings.database_url)
    store.create_schema()
    issue = datetime.now(UTC).replace(second=0, microsecond=0)
    demand = [
        forecast(EventType.DEMAND, issue + timedelta(minutes=15 * i), issue)
        for i in range(1, 25)
    ]
    delivery_date = date(2025, 10, 26)
    price = [
        forecast(EventType.PRICE, start, issue)
        for start, _ in delivery_intervals(delivery_date)
    ]
    store.upsert_forecasts(demand + price)

    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").status_code == 200
        demand_response = client.get("/v1/forecasts/demand?horizon_hours=2")
        assert demand_response.status_code == 200
        assert len(demand_response.json()["forecasts"]) == 8
        price_response = client.get(
            "/v1/forecasts/prices/day-ahead", params={"delivery_date": delivery_date.isoformat()}
        )
        assert price_response.status_code == 200
        assert len(price_response.json()["forecasts"]) == 100
        assert client.get("/metrics").status_code == 200


def test_unavailable_forecast_has_structured_error(tmp_path: Path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'empty.db'}")
    with TestClient(create_app(settings)) as client:
        response = client.get("/v1/forecasts/demand")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "forecast_unavailable"
        invalid = client.get("/v1/forecasts/demand?horizon_hours=7")
        assert invalid.status_code == 422
