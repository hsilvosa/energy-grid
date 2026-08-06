from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from xml.etree import ElementTree

import httpx

from energy_grid.domain import EventType, GridEvent

ENTSOE_ENDPOINT = "https://web-api.tp.entsoe.eu/api"
AREA_ES = "10YES-REE------0"
DOCUMENT_TYPES = {
    EventType.DEMAND: "A65",
    EventType.GENERATION: "A75",
    EventType.PRICE: "A44",
}
PROCESS_TYPES = {EventType.DEMAND: "A16", EventType.GENERATION: "A16", EventType.PRICE: "A01"}
RESOLUTIONS = {
    "PT15M": timedelta(minutes=15),
    "PT30M": timedelta(minutes=30),
    "PT60M": timedelta(hours=1),
}


class EntsoeClient:
    def __init__(self, token: str, timeout: float = 30) -> None:
        if not token:
            raise ValueError("ENTSO-E token is required for live downloads")
        self.token = token
        self.timeout = timeout

    def download(
        self,
        target: EventType,
        start: datetime,
        end: datetime,
        *,
        area: str = AREA_ES,
    ) -> tuple[bytes, dict[str, str]]:
        if target not in DOCUMENT_TYPES:
            raise ValueError(f"ENTSO-E connector does not support {target}")
        params = {
            "securityToken": self.token,
            "documentType": DOCUMENT_TYPES[target],
            "processType": PROCESS_TYPES[target],
            "periodStart": start.astimezone(UTC).strftime("%Y%m%d%H%M"),
            "periodEnd": end.astimezone(UTC).strftime("%Y%m%d%H%M"),
        }
        area_key = "in_Domain" if target == EventType.PRICE else "outBiddingZone_Domain"
        params[area_key] = area
        if target == EventType.GENERATION:
            params = {**params, "in_Domain": area}
            params.pop("outBiddingZone_Domain", None)
        response = httpx.get(ENTSOE_ENDPOINT, params=params, timeout=self.timeout)
        response.raise_for_status()
        safe_params = {key: value for key, value in params.items() if key != "securityToken"}
        return response.content, safe_params

    def events(
        self,
        target: EventType,
        start: datetime,
        end: datetime,
        *,
        area: str = AREA_ES,
    ) -> list[GridEvent]:
        payload, params = self.download(target, start, end, area=area)
        return list(parse_entsoe_document(payload, target, area=area, request_params=params))


def _text(element: ElementTree.Element, name: str) -> str | None:
    match = element.find(f".//{{*}}{name}")
    return match.text if match is not None else None


def parse_entsoe_document(
    payload: bytes,
    target: EventType,
    *,
    area: str = AREA_ES,
    request_params: dict[str, str] | None = None,
    retrieved_at: datetime | None = None,
) -> Iterable[GridEvent]:
    root = ElementTree.fromstring(payload)
    retrieved = retrieved_at or datetime.now(UTC)
    checksum = hashlib.sha256(payload).hexdigest()
    for series in root.findall(".//{*}TimeSeries"):
        dimension = _text(series, "psrType")
        for period in series.findall(".//{*}Period"):
            start_text = _text(period, "start")
            resolution_text = _text(period, "resolution")
            if start_text is None or resolution_text not in RESOLUTIONS:
                continue
            period_start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
            resolution = RESOLUTIONS[resolution_text]
            for point in period.findall("./{*}Point"):
                position_text = _text(point, "position")
                value_text = _text(point, "quantity") or _text(point, "price.amount")
                if position_text is None or value_text is None:
                    continue
                interval_start = period_start + resolution * (int(position_text) - 1)
                unit = "EUR/MWh" if target == EventType.PRICE else "MW"
                yield GridEvent(
                    source="entsoe",
                    event_type=target,
                    area=area,
                    interval_start=interval_start,
                    interval_end=interval_start + resolution,
                    published_at=retrieved,
                    ingested_at=retrieved,
                    value=float(value_text),
                    unit=unit,
                    dimension=dimension,
                    source_url=ENTSOE_ENDPOINT,
                    request_params=request_params or {},
                    payload_checksum=checksum,
                )
