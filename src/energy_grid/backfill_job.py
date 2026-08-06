from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime

from energy_grid.streaming import build_spark, ensure_tables


def run(start: str, end: str, source: str) -> str:
    spark = build_spark()
    ensure_tables(spark)
    start_time = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(UTC)
    end_time = datetime.fromisoformat(end.replace("Z", "+00:00")).astimezone(UTC)
    input_checksum = hashlib.sha256(f"{source}|{start}|{end}".encode()).hexdigest()
    run_id = input_checksum[:16]
    existing = spark.sql(
        f"SELECT count(*) count FROM local.gold.backfill_manifests WHERE run_id = '{run_id}'"
    ).first()["count"]
    if existing:
        spark.stop()
        return run_id

    before = spark.sql(
        "SELECT snapshot_id FROM local.silver.grid_events.snapshots "
        "ORDER BY committed_at DESC LIMIT 1"
    ).first()
    candidates = spark.sql(
        f"""
        SELECT * FROM local.silver.late_events
        WHERE source = '{source}'
          AND interval_start >= TIMESTAMP '{start_time.isoformat()}'
          AND interval_start < TIMESTAMP '{end_time.isoformat()}'
        """
    )
    candidates.createOrReplaceTempView("backfill_candidates")
    spark.sql(
        """
        MERGE INTO local.silver.grid_events target
        USING backfill_candidates source
        ON target.business_key = source.business_key
        WHEN MATCHED AND source.revision > target.revision THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    after = spark.sql(
        "SELECT snapshot_id FROM local.silver.grid_events.snapshots "
        "ORDER BY committed_at DESC LIMIT 1"
    ).first()
    manifest = spark.createDataFrame(
        [
            (
                run_id,
                source,
                start_time,
                end_time,
                input_checksum,
                candidates.count(),
                str(before["snapshot_id"]) if before else None,
                str(after["snapshot_id"]) if after else None,
                datetime.now(UTC),
            )
        ],
        "run_id string, source string, start_time timestamp, end_time timestamp, "
        "input_checksum string, record_count long, before_snapshot_id string, "
        "after_snapshot_id string, completed_at timestamp",
    )
    manifest.writeTo("local.gold.backfill_manifests").append()
    spark.stop()
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--source", required=True)
    arguments = parser.parse_args()
    print(run(arguments.start, arguments.end, arguments.source))
