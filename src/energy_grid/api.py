from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from energy_grid.config import Settings, get_settings
from energy_grid.domain import MADRID, EventType, ForecastRecord
from energy_grid.monitoring import API_LATENCY, FORECAST_ERROR_RATIO, MODEL_ROLLBACK_STATE
from energy_grid.storage import ForecastStore

AREA_ES = "10YES-REE------0"


def get_store(settings: Annotated[Settings, Depends(get_settings)]) -> ForecastStore:
    return ForecastStore(settings.database_url)


def _payload(
    records: list[ForecastRecord], settings: Settings, now: datetime | None = None
) -> dict[str, Any]:
    if not records:
        return {"forecasts": [], "stale": True, "age_seconds": None}
    current = now or datetime.now(UTC)
    age = max(0.0, (current - records[0].issue_time).total_seconds())
    return {
        "forecasts": [record.model_dump(mode="json") for record in records],
        "stale": age > settings.forecast_stale_after_minutes * 60,
        "age_seconds": age,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        ForecastStore(resolved.database_url).create_schema()
        yield

    app = FastAPI(
        title="Energy Grid Intelligence API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.dependency_overrides[get_settings] = lambda: resolved

    @app.middleware("http")
    async def observe_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        API_LATENCY.labels(request.method, request.url.path, response.status_code).observe(
            time.perf_counter() - started
        )
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": str(exc)}},
        )

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def readiness(store: Annotated[ForecastStore, Depends(get_store)]) -> JSONResponse:
        ready = store.ready()
        return JSONResponse({"status": "ready" if ready else "not_ready"}, 200 if ready else 503)

    @app.get("/metrics")
    def metrics(store: Annotated[ForecastStore, Depends(get_store)]) -> Response:
        for status in store.model_statuses():
            rolling_error = status["rolling_error"]
            baseline_error = status["baseline_error"]
            if rolling_error is not None and baseline_error:
                FORECAST_ERROR_RATIO.labels(status["target"], "rolling_7d").set(
                    rolling_error / baseline_error
                )
            MODEL_ROLLBACK_STATE.labels(status["target"]).set(
                int(status["status"] == "rolled_back")
            )
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/forecasts/demand")
    def demand_forecasts(
        store: Annotated[ForecastStore, Depends(get_store)],
        settings: Annotated[Settings, Depends(get_settings)],
        area: str = AREA_ES,
        horizon_hours: int = Query(default=6, ge=1, le=6),
    ) -> JSONResponse:
        records = store.query_forecasts(
            EventType.DEMAND, area=area, limit=horizon_hours * 4
        )
        if not records:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "forecast_unavailable",
                        "message": "No demand forecast",
                    }
                },
            )
        return JSONResponse(_payload(records, settings))

    @app.get("/v1/forecasts/prices/day-ahead")
    def price_forecasts(
        store: Annotated[ForecastStore, Depends(get_store)],
        settings: Annotated[Settings, Depends(get_settings)],
        delivery_date: date,
        area: str = AREA_ES,
    ) -> JSONResponse:
        records = store.query_forecasts(EventType.PRICE, area=area, limit=100)
        records = [
            record
            for record in records
            if record.valid_from.astimezone(MADRID).date() == delivery_date
        ]
        if not records:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "code": "delivery_date_unavailable",
                        "message": f"No price forecast for {delivery_date.isoformat()}",
                    }
                },
            )
        return JSONResponse(_payload(records, settings))

    @app.get("/v1/models/status")
    def model_status(store: Annotated[ForecastStore, Depends(get_store)]) -> dict[str, Any]:
        return {"models": store.model_statuses()}

    return app


app = create_app()
