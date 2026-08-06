from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

from energy_grid.domain import EventType, GridEvent


@dataclass(frozen=True)
class WeatherLocation:
    name: str
    latitude: float
    longitude: float
    weight: float


SPANISH_LOCATIONS = (
    WeatherLocation("madrid", 40.4168, -3.7038, 0.30),
    WeatherLocation("barcelona", 41.3874, 2.1686, 0.24),
    WeatherLocation("valencia", 39.4699, -0.3763, 0.17),
    WeatherLocation("seville", 37.3891, -5.9845, 0.16),
    WeatherLocation("bilbao", 43.2630, -2.9350, 0.13),
)


class OpenMeteoClient:
    LIVE_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
    HISTORICAL_ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    def __init__(self, timeout: float = 30) -> None:
        self.timeout = timeout

    def download(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        historical: bool = False,
    ) -> tuple[list[dict[str, object]], dict[str, str | bool]]:
        params: dict[str, str | bool] = {
            "latitude": ",".join(str(item.latitude) for item in SPANISH_LOCATIONS),
            "longitude": ",".join(str(item.longitude) for item in SPANISH_LOCATIONS),
            "hourly": "temperature_2m,wind_speed_10m,relative_humidity_2m,shortwave_radiation",
            "timezone": "UTC",
        }
        if start_date and end_date:
            params.update({"start_date": start_date, "end_date": end_date})
        endpoint = self.HISTORICAL_ENDPOINT if historical else self.LIVE_ENDPOINT
        response = httpx.get(endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        return body if isinstance(body, list) else [body], {"endpoint": endpoint, **params}

    def events(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        historical: bool = False,
    ) -> list[GridEvent]:
        payloads, params = self.download(
            start_date=start_date, end_date=end_date, historical=historical
        )
        retrieved = datetime.now(UTC)
        variables = {
            "temperature_2m": "degC",
            "wind_speed_10m": "m/s",
            "relative_humidity_2m": "%",
            "shortwave_radiation": "W/m2",
        }
        by_time: dict[tuple[str, str], float] = {}
        for location, payload in zip(SPANISH_LOCATIONS, payloads, strict=True):
            hourly = payload["hourly"]
            assert isinstance(hourly, dict)
            times = hourly["time"]
            assert isinstance(times, list)
            for variable in variables:
                values = hourly[variable]
                assert isinstance(values, list)
                for timestamp, value in zip(times, values, strict=True):
                    if value is not None:
                        key = (str(timestamp), variable)
                        by_time[key] = by_time.get(key, 0.0) + float(value) * location.weight

        raw = json.dumps(payloads, sort_keys=True).encode()
        checksum = hashlib.sha256(raw).hexdigest()
        endpoint = str(params.pop("endpoint"))
        events: list[GridEvent] = []
        for (timestamp, variable), value in sorted(by_time.items()):
            start = datetime.fromisoformat(timestamp).replace(tzinfo=UTC)
            events.append(
                GridEvent(
                    source="open-meteo",
                    event_type=EventType.WEATHER,
                    interval_start=start,
                    interval_end=start + timedelta(hours=1),
                    published_at=retrieved,
                    ingested_at=retrieved,
                    value=value,
                    unit=variables[variable],
                    dimension=variable,
                    source_url=endpoint,
                    request_params=params,
                    payload_checksum=checksum,
                )
            )
        return events
