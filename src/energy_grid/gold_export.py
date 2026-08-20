from __future__ import annotations

import os

from energy_grid.streaming import build_spark, ensure_tables


def run() -> None:
    spark = build_spark()
    ensure_tables(spark)
    jdbc_url = os.getenv("POSTGRES_JDBC_URL", "jdbc:postgresql://postgres:5432/energy")
    forecasts = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "forecasts")
        .option("user", os.getenv("POSTGRES_USER", "energy"))
        .option("password", os.getenv("POSTGRES_PASSWORD", "energy"))
        .option("driver", "org.postgresql.Driver")
        .load()
        .selectExpr(
            "forecast_id",
            "target",
            "area",
            "issue_time",
            "valid_from",
            "valid_to",
            "point",
            "p10",
            "p50",
            "p90",
            "unit",
            "model_name",
            "model_version",
            "data_snapshot_id",
            "cast(quality_flags as string) quality_flags",
            "is_shadow",
        )
    )
    forecasts.createOrReplaceTempView("forecast_export")
    spark.sql(
        """
        MERGE INTO local.gold.forecasts target
        USING forecast_export source
        ON target.forecast_id = source.forecast_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    spark.stop()


if __name__ == "__main__":
    run()
