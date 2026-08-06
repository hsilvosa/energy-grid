from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MADRID = ZoneInfo("Europe/Madrid")


class EventType(StrEnum):
    DEMAND = "demand"
    GENERATION = "generation"
    PRICE = "price"
    WEATHER = "weather"


class QualityStatus(StrEnum):
    VALID = "valid"
    MISSING = "missing"
    ESTIMATED = "estimated"
    INVALID = "invalid"


EXPECTED_UNITS: dict[EventType, set[str]] = {
    EventType.DEMAND: {"MW"},
    EventType.GENERATION: {"MW"},
    EventType.PRICE: {"EUR/MWh"},
    EventType.WEATHER: {"degC", "m/s", "%", "W/m2"},
}


class GridEvent(BaseModel):
    """Versioned canonical event shared by producers and streaming consumers."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    event_id: UUID = Field(default_factory=uuid4)
    source: str
    event_type: EventType
    area: str = "10YES-REE------0"
    interval_start: datetime
    interval_end: datetime
    published_at: datetime
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    value: float | None
    unit: str
    revision: int = Field(default=1, ge=1)
    quality_status: QualityStatus = QualityStatus.VALID
    dimension: str | None = None
    source_url: str | None = None
    request_params: dict[str, Any] = Field(default_factory=dict)
    payload_checksum: str | None = None

    @field_validator("interval_start", "interval_end", "published_at", "ingested_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_event(self) -> GridEvent:
        if self.interval_end <= self.interval_start:
            raise ValueError("interval_end must be after interval_start")
        if self.unit not in EXPECTED_UNITS[self.event_type]:
            raise ValueError(f"unit {self.unit!r} is invalid for {self.event_type}")
        if self.quality_status == QualityStatus.MISSING and self.value is not None:
            raise ValueError("missing events must have a null value")
        if self.quality_status == QualityStatus.VALID and self.value is None:
            raise ValueError("valid events must have a value")
        return self

    @property
    def business_key(self) -> str:
        parts = (
            self.source,
            self.event_type,
            self.area,
            self.interval_start.isoformat(),
            self.dimension or "",
        )
        return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()

    @property
    def local_delivery_date(self) -> date:
        return self.interval_start.astimezone(MADRID).date()

    @property
    def timezone_offset_minutes(self) -> int:
        offset = self.interval_start.astimezone(MADRID).utcoffset()
        assert offset is not None
        return int(offset.total_seconds() // 60)

    def to_message(self) -> bytes:
        return self.model_dump_json().encode()

    @classmethod
    def from_message(cls, value: bytes | str) -> GridEvent:
        return cls.model_validate_json(value)

    def with_checksum(self, raw_payload: bytes) -> GridEvent:
        return self.model_copy(
            update={"payload_checksum": hashlib.sha256(raw_payload).hexdigest()}
        )


class ForecastRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecast_id: UUID = Field(default_factory=uuid4)
    target: EventType
    area: str = "10YES-REE------0"
    issue_time: datetime
    valid_from: datetime
    valid_to: datetime
    point: float
    p10: float
    p50: float
    p90: float
    unit: str
    model_name: str
    model_version: str
    data_snapshot_id: str
    quality_flags: list[str] = Field(default_factory=list)
    is_shadow: bool = False

    @field_validator("issue_time", "valid_from", "valid_to")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("forecast timestamps must be timezone aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def ordered_quantiles(self) -> ForecastRecord:
        if not self.p10 <= self.p50 <= self.p90:
            raise ValueError("forecast quantiles must be ordered")
        if self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
        return self

    def stable_key(self) -> str:
        raw = json.dumps(
            [self.target, self.area, self.issue_time.isoformat(), self.valid_from.isoformat()],
            separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode()).hexdigest()

