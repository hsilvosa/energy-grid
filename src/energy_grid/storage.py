from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from energy_grid.domain import MADRID, EventType, ForecastRecord


class Base(DeclarativeBase):
    pass


class ForecastRow(Base):
    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint(
            "target", "area", "issue_time", "valid_from", "model_version", "is_shadow"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    forecast_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    target: Mapped[str] = mapped_column(String(32), index=True)
    area: Mapped[str] = mapped_column(String(32), index=True)
    issue_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    valid_to: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    point: Mapped[float] = mapped_column(Float)
    p10: Mapped[float] = mapped_column(Float)
    p50: Mapped[float] = mapped_column(Float)
    p90: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(16))
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))
    data_snapshot_id: Mapped[str] = mapped_column(String(128))
    quality_flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    is_shadow: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class ModelStatusRow(Base):
    __tablename__ = "model_status"

    target: Mapped[str] = mapped_column(String(32), primary_key=True)
    champion_version: Mapped[str] = mapped_column(String(64))
    previous_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    baseline_version: Mapped[str] = mapped_column(String(64))
    last_evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rolling_error: Mapped[float | None] = mapped_column(Float)
    baseline_error: Mapped[float | None] = mapped_column(Float)
    consecutive_breaches: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="healthy")


class BackfillManifestRow(Base):
    __tablename__ = "backfill_manifests"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(64))
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    input_checksum: Mapped[str] = mapped_column(String(64))
    record_count: Mapped[int] = mapped_column(Integer)
    before_snapshot_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    after_snapshot_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class ForecastStore:
    def __init__(self, database_url: str) -> None:
        if database_url.startswith("sqlite"):
            database_path = database_url.removeprefix("sqlite:///")
            if database_path and database_path != ":memory:":
                Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        connect_args: dict[str, Any] = {}
        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        self.engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def ready(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
            return True
        except Exception:
            return False

    def upsert_forecasts(self, forecasts: list[ForecastRecord]) -> int:
        with Session(self.engine) as session:
            for forecast in forecasts:
                existing = (
                    session.query(ForecastRow)
                    .filter_by(
                        target=forecast.target.value,
                        area=forecast.area,
                        issue_time=forecast.issue_time,
                        valid_from=forecast.valid_from,
                        model_version=forecast.model_version,
                        is_shadow=forecast.is_shadow,
                    )
                    .one_or_none()
                )
                values = forecast.model_dump(mode="python")
                values["forecast_id"] = str(values["forecast_id"])
                values["target"] = forecast.target.value
                if existing:
                    for name, value in values.items():
                        if name != "forecast_id":
                            setattr(existing, name, value)
                else:
                    session.add(ForecastRow(**values))
            session.commit()
        return len(forecasts)

    def query_forecasts(
        self,
        target: EventType,
        *,
        area: str,
        delivery_date: date | None = None,
        issue_time: datetime | None = None,
        limit: int = 100,
    ) -> list[ForecastRecord]:
        with Session(self.engine) as session:
            query = session.query(ForecastRow).filter_by(
                target=target.value, area=area, is_shadow=False
            )
            if issue_time is None:
                latest = (
                    session.query(ForecastRow.issue_time)
                    .filter_by(target=target.value, area=area, is_shadow=False)
                    .order_by(ForecastRow.issue_time.desc())
                    .limit(1)
                    .scalar()
                )
                if latest is None:
                    return []
                query = query.filter(ForecastRow.issue_time == latest)
            else:
                query = query.filter(ForecastRow.issue_time == issue_time)
            rows = query.order_by(ForecastRow.valid_from).limit(limit).all()

        records = [self._to_record(row) for row in rows]
        if delivery_date is not None:
            records = [
                item
                for item in records
                if item.valid_from.astimezone(MADRID).date() == delivery_date
            ]
        return records

    def model_statuses(self) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            rows = session.query(ModelStatusRow).order_by(ModelStatusRow.target).all()
            return [
                {
                    "target": row.target,
                    "champion_version": row.champion_version,
                    "previous_version": row.previous_version,
                    "baseline_version": row.baseline_version,
                    "last_evaluated_at": row.last_evaluated_at,
                    "rolling_error": row.rolling_error,
                    "baseline_error": row.baseline_error,
                    "consecutive_breaches": row.consecutive_breaches,
                    "status": row.status,
                }
                for row in rows
            ]

    def set_model_status(self, target: EventType, **values: Any) -> None:
        with Session(self.engine) as session:
            row = session.get(ModelStatusRow, target.value)
            if row is None:
                row = ModelStatusRow(target=target.value, **values)
                session.add(row)
            else:
                for name, value in values.items():
                    setattr(row, name, value)
            session.commit()

    @staticmethod
    def _to_record(row: ForecastRow) -> ForecastRecord:
        return ForecastRecord(
            forecast_id=row.forecast_id,
            target=EventType(row.target),
            area=row.area,
            issue_time=_as_utc(row.issue_time),
            valid_from=_as_utc(row.valid_from),
            valid_to=_as_utc(row.valid_to),
            point=row.point,
            p10=row.p10,
            p50=row.p50,
            p90=row.p90,
            unit=row.unit,
            model_name=row.model_name,
            model_version=row.model_version,
            data_snapshot_id=row.data_snapshot_id,
            quality_flags=row.quality_flags or [],
            is_shadow=row.is_shadow,
        )
