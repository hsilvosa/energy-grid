from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from energy_grid.domain import EventType, GridEvent, QualityStatus

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path(r"D:\datasets\entsoe-transparency")


@dataclass(frozen=True)
class DatasetSummary:
    dataset_type: str
    zone_key: str
    country_code: str
    total_records: int
    start_time: datetime
    end_time: datetime
    resolution: str
    unit: str
    min_value: float
    max_value: float
    mean_value: float


class EntsoeDatasetReader:
    """High-performance reader for local normalized ENTSO-E parquet datasets."""

    def __init__(self, root_dir: Path | str = DEFAULT_DATASET_PATH) -> None:
        self.root_dir = Path(root_dir)
        self._check_paths()

    def _check_paths(self) -> None:
        # Check standard layout (hf_staging/data or data/processed or data)
        self.actual_load_dir = self._resolve_subdir("actual_load")
        self.prices_dir = self._resolve_subdir("day_ahead_prices")

    def _resolve_subdir(self, name: str) -> Path:
        candidates = [
            self.root_dir / "hf_staging" / "data" / name,
            self.root_dir / "data" / "processed" / name,
            self.root_dir / "data" / name,
            self.root_dir / name,
        ]
        for candidate in candidates:
            if candidate.exists() and any(candidate.glob("*.parquet")):
                return candidate
        # Default to standard hf_staging path
        return self.root_dir / "hf_staging" / "data" / name

    def exists(self) -> bool:
        return self.actual_load_dir.exists() or self.prices_dir.exists()

    def load_raw_table(
        self,
        target: EventType | Literal["actual_load", "day_ahead_prices"],
        *,
        country_code: str = "ES",
        zone_key: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Load and filter parquet shards for the specified zone and time range."""
        target_name = (
            "actual_load" if target in (EventType.DEMAND, "actual_load") else "day_ahead_prices"
        )
        data_dir = self.actual_load_dir if target_name == "actual_load" else self.prices_dir
        if not data_dir.exists():
            raise FileNotFoundError(f"Parquet directory not found: {data_dir}")

        dataset = ds.dataset(str(data_dir), format="parquet")

        # Build filter expression
        expr = ds.field("country_code") == country_code
        if zone_key:
            expr = expr & ((ds.field("zone_key") == zone_key) | (ds.field("eic_code") == zone_key))

        if start_year is not None:
            dt_start = pd.Timestamp(datetime(start_year, 1, 1, tzinfo=UTC))
            expr = expr & (ds.field("timestamp_utc") >= dt_start)
        if end_year is not None:
            dt_end = pd.Timestamp(datetime(end_year, 12, 31, 23, 59, 59, tzinfo=UTC))
            expr = expr & (ds.field("timestamp_utc") <= dt_end)

        if start_time is not None:
            expr = expr & (ds.field("timestamp_utc") >= pd.Timestamp(start_time.astimezone(UTC)))
        if end_time is not None:
            expr = expr & (ds.field("timestamp_utc") <= pd.Timestamp(end_time.astimezone(UTC)))

        scanner = dataset.scanner(filter=expr)
        table = scanner.to_table()
        frame = table.to_pandas()

        if frame.empty:
            logger.warning(
                "No records found for target %s (country=%s, zone=%s)",
                target_name,
                country_code,
                zone_key,
            )
            return frame

        # Ensure UTC timezone and proper timestamp sorting
        if not isinstance(frame["timestamp_utc"].dtype, pd.DatetimeTZDtype):
            frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        else:
            frame["timestamp_utc"] = frame["timestamp_utc"].dt.tz_convert("UTC")

        # Sort and deduplicate by latest revision number if present
        sort_cols = ["timestamp_utc"]
        if "revision_number" in frame.columns:
            sort_cols.append("revision_number")
        frame = frame.sort_values(sort_cols)
        frame = frame.drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)
        return frame

    def load_series(
        self,
        target: EventType | Literal["actual_load", "day_ahead_prices"],
        *,
        country_code: str = "ES",
        zone_key: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        resample_freq: str | None = None,
    ) -> pd.Series:
        """Load a clean continuous UTC time-series indexed by timestamp."""
        frame = self.load_raw_table(
            target,
            country_code=country_code,
            zone_key=zone_key,
            start_year=start_year,
            end_year=end_year,
            start_time=start_time,
            end_time=end_time,
        )
        if frame.empty:
            return pd.Series(dtype=float, name="value")

        val_col = "load" if "load" in frame.columns else "price"
        series = frame.set_index("timestamp_utc")[val_col].astype(float)
        series.name = "demand" if val_col == "load" else "price"
        series.index = pd.DatetimeIndex(series.index).tz_convert("UTC")

        if resample_freq:
            # Resample and interpolate small missing gaps linearly
            series = series.resample(resample_freq).mean().interpolate(method="time", limit=4)

        return series

    def load_as_events(
        self,
        target: EventType,
        *,
        country_code: str = "ES",
        zone_key: str = "10YES-REE------0",
        start_year: int | None = None,
        end_year: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> list[GridEvent]:
        """Convert parquet records to canonical GridEvent domain objects."""
        frame = self.load_raw_table(
            target,
            country_code=country_code,
            zone_key=zone_key,
            start_year=start_year,
            end_year=end_year,
            start_time=start_time,
            end_time=end_time,
        )
        if limit:
            frame = frame.iloc[:limit]

        events: list[GridEvent] = []
        val_col = "load" if target == EventType.DEMAND else "price"
        unit = "MW" if target == EventType.DEMAND else "EUR/MWh"

        for _, row in frame.iterrows():
            ts = row["timestamp_utc"].to_pydatetime()
            res = str(row.get("resolution") or "PT15M")
            delta_mins = 60 if "60" in res or "1H" in res else (30 if "30" in res else 15)
            end_ts = ts + timedelta(minutes=delta_mins)
            val = float(row[val_col]) if not pd.isna(row[val_col]) else None
            rec_id = str(row.get("record_id") or "")
            event_uuid = uuid5(NAMESPACE_URL, f"entsoe-hist-{rec_id}")

            events.append(
                GridEvent(
                    schema_version="1.0",
                    event_id=event_uuid,
                    source="entsoe_dataset",
                    event_type=target,
                    area=str(row.get("zone_key") or zone_key),
                    interval_start=ts,
                    interval_end=end_ts,
                    published_at=ts,
                    ingested_at=ts,
                    revision=int(row.get("revision_number") or 1),
                    quality_status=QualityStatus.VALID
                    if val is not None
                    else QualityStatus.MISSING,
                    value=val,
                    unit=unit,
                    dimension=None,
                    payload_checksum=rec_id,
                )
            )
        return events

    def list_available_zones(
        self,
        target: EventType | Literal["actual_load", "day_ahead_prices"] = EventType.DEMAND,
    ) -> list[dict[str, str]]:
        """List all European countries and bidding zones present in the dataset."""
        target_name = (
            "actual_load" if target in (EventType.DEMAND, "actual_load") else "day_ahead_prices"
        )
        data_dir = self.actual_load_dir if target_name == "actual_load" else self.prices_dir
        if not data_dir.exists():
            return []

        dataset = ds.dataset(str(data_dir), format="parquet")
        table = dataset.to_table(columns=["country_code", "zone_key", "zone_name"])
        df = table.to_pandas().drop_duplicates()
        df = df.sort_values(by=["country_code", "zone_key"])
        return [
            {
                "country_code": str(row["country_code"]),
                "zone_key": str(row["zone_key"]),
                "zone_name": str(row["zone_name"]),
            }
            for _, row in df.iterrows()
        ]

    def summarize(
        self,
        target: EventType | Literal["actual_load", "day_ahead_prices"],
        country_code: str = "ES",
        zone_key: str | None = None,
    ) -> DatasetSummary:
        """Produce statistical and coverage summary of the offline dataset."""
        series = self.load_series(target, country_code=country_code, zone_key=zone_key)
        if series.empty:
            raise ValueError(f"No data available for {target} in {country_code}")

        target_name = (
            "actual_load" if target in (EventType.DEMAND, "actual_load") else "day_ahead_prices"
        )
        unit = "MW" if target_name == "actual_load" else "EUR/MWh"

        return DatasetSummary(
            dataset_type=target_name,
            zone_key=zone_key or country_code,
            country_code=country_code,
            total_records=len(series),
            start_time=series.index.min().to_pydatetime(),
            end_time=series.index.max().to_pydatetime(),
            resolution="15min/60min",
            unit=unit,
            min_value=float(np.nanmin(series)),
            max_value=float(np.nanmax(series)),
            mean_value=float(np.nanmean(series)),
        )

