from __future__ import annotations

import os
from typing import Any


def build_spark() -> Any:
    from pyspark.sql import SparkSession

    warehouse = os.getenv("ICEBERG_WAREHOUSE", "s3a://energy-lakehouse/warehouse")
    return (
        SparkSession.builder.appName("energy-grid-streaming")
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.local.type", "hadoop")
        .config("spark.sql.catalog.local.warehouse", warehouse)
        # The tabulario image defaults to its bundled `demo` REST catalog.
        # Use the self-contained Hadoop catalog backed by MinIO instead.
        .config("spark.sql.defaultCatalog", "local")
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("AWS_ENDPOINT_URL", "http://minio:9000"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID", "minio"))
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"))
        .getOrCreate()
    )


def ensure_tables(spark: Any) -> None:
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.bronze")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.silver")
    spark.sql("CREATE NAMESPACE IF NOT EXISTS local.gold")
    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS local.bronze.kafka_events (
          topic STRING, partition INT, offset BIGINT, kafka_timestamp TIMESTAMP,
          key STRING, raw_payload STRING, ingested_at TIMESTAMP
        ) USING iceberg PARTITIONED BY (days(kafka_timestamp))
        """
    )
    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS local.gold.forecasts (
          forecast_id STRING, target STRING, area STRING, issue_time TIMESTAMP,
          valid_from TIMESTAMP, valid_to TIMESTAMP, point DOUBLE, p10 DOUBLE,
          p50 DOUBLE, p90 DOUBLE, unit STRING, model_name STRING, model_version STRING,
          data_snapshot_id STRING, quality_flags STRING, is_shadow BOOLEAN
        ) USING iceberg PARTITIONED BY (target, days(valid_from))
        """
    )
    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS local.gold.backfill_manifests (
          run_id STRING, source STRING, start_time TIMESTAMP, end_time TIMESTAMP,
          input_checksum STRING, record_count BIGINT, before_snapshot_id STRING,
          after_snapshot_id STRING, completed_at TIMESTAMP
        ) USING iceberg
        """
    )
    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS local.silver.grid_events (
          business_key STRING, schema_version STRING, event_id STRING, source STRING,
          event_type STRING, area STRING, interval_start TIMESTAMP, interval_end TIMESTAMP,
          published_at TIMESTAMP, ingested_at TIMESTAMP, value DOUBLE, unit STRING,
          revision INT, quality_status STRING, dimension STRING, payload_checksum STRING
        ) USING iceberg PARTITIONED BY (event_type, days(interval_start))
        """
    )
    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS local.silver.late_events
        USING iceberg AS SELECT * FROM local.silver.grid_events WHERE 1 = 0
        """
    )
    spark.sql(
        """
        CREATE TABLE IF NOT EXISTS local.silver.dead_letter_events (
          topic STRING, partition INT, offset BIGINT, raw_payload STRING,
          reason STRING, rejected_at TIMESTAMP
        ) USING iceberg PARTITIONED BY (days(rejected_at))
        """
    )


def run() -> None:
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        MapType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )
    from pyspark.sql.window import Window

    spark = build_spark()
    ensure_tables(spark)
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    checkpoint_root = os.getenv("CHECKPOINT_ROOT", "/checkpoints")
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribePattern", "grid\\..*|market\\..*|weather\\..*|energy\\..*")
        .option("startingOffsets", "earliest")
        .load()
    )
    bronze = raw.select(
        "topic",
        "partition",
        "offset",
        F.col("timestamp").alias("kafka_timestamp"),
        F.col("key").cast("string").alias("key"),
        F.col("value").cast("string").alias("raw_payload"),
        F.current_timestamp().alias("ingested_at"),
    )
    bronze_query = (
        bronze.writeStream.format("iceberg")
        .outputMode("append")
        .option("checkpointLocation", f"{checkpoint_root}/bronze")
        .toTable("local.bronze.kafka_events")
    )

    schema = StructType(
        [
            StructField("schema_version", StringType()),
            StructField("event_id", StringType()),
            StructField("source", StringType()),
            StructField("event_type", StringType()),
            StructField("area", StringType()),
            StructField("interval_start", TimestampType()),
            StructField("interval_end", TimestampType()),
            StructField("published_at", TimestampType()),
            StructField("ingested_at", TimestampType()),
            StructField("value", DoubleType()),
            StructField("unit", StringType()),
            StructField("revision", IntegerType()),
            StructField("quality_status", StringType()),
            StructField("dimension", StringType()),
            StructField("payload_checksum", StringType()),
            StructField("request_params", MapType(StringType(), StringType())),
        ]
    )
    parsed = bronze.withColumn("event", F.from_json("raw_payload", schema))

    def curate_batch(batch: Any, batch_id: int) -> None:
        batch.createOrReplaceTempView("incoming_batch")
        spark.sql(
            """
            INSERT INTO local.silver.dead_letter_events
            SELECT topic, partition, offset, raw_payload, 'schema_validation', current_timestamp()
            FROM incoming_batch
            WHERE event IS NULL
               OR event.event_id IS NULL
               OR event.source IS NULL
               OR event.area IS NULL
               OR event.interval_start IS NULL
               OR event.interval_end IS NULL
               OR event.published_at IS NULL
               OR event.ingested_at IS NULL
               OR event.unit IS NULL
               OR event.revision IS NULL
               OR event.interval_end <= event.interval_start
               OR event.event_type NOT IN ('demand', 'generation', 'price', 'weather')
               OR event.quality_status NOT IN ('valid', 'missing', 'estimated', 'invalid')
               OR (event.quality_status = 'valid' AND event.value IS NULL)
               OR (event.quality_status = 'missing' AND event.value IS NOT NULL)
               OR (event.event_type IN ('demand', 'generation') AND event.unit <> 'MW')
               OR (event.event_type = 'price' AND event.unit <> 'EUR/MWh')
               OR (event.event_type = 'weather' AND event.unit NOT IN ('degC', 'm/s', '%', 'W/m2'))
            """
        )
        valid = spark.sql(
            """
            SELECT sha2(
                       concat_ws('|', event.source, event.event_type, event.area,
                       cast(event.interval_start AS STRING), coalesce(event.dimension, '')),
                       256
                   ) business_key,
                   event.schema_version, event.event_id, event.source, event.event_type, event.area,
                   event.interval_start, event.interval_end, event.published_at, event.ingested_at,
                   event.value, event.unit, event.revision, event.quality_status, event.dimension,
                   event.payload_checksum
            FROM incoming_batch
            WHERE event IS NOT NULL
              AND event.event_id IS NOT NULL
              AND event.source IS NOT NULL
              AND event.area IS NOT NULL
              AND event.interval_start IS NOT NULL
              AND event.interval_end IS NOT NULL
              AND event.published_at IS NOT NULL
              AND event.ingested_at IS NOT NULL
              AND event.unit IS NOT NULL
              AND event.revision IS NOT NULL
              AND event.interval_end > event.interval_start
              AND event.event_type IN ('demand', 'generation', 'price', 'weather')
              AND event.quality_status IN ('valid', 'missing', 'estimated', 'invalid')
              AND NOT (event.quality_status = 'valid' AND event.value IS NULL)
              AND NOT (event.quality_status = 'missing' AND event.value IS NOT NULL)
              AND ((event.event_type IN ('demand', 'generation') AND event.unit = 'MW')
                OR (event.event_type = 'price' AND event.unit = 'EUR/MWh')
                OR (event.event_type = 'weather' AND event.unit IN ('degC', 'm/s', '%', 'W/m2')))
            """
        )
        late = valid.filter(
            F.col("ingested_at") > F.col("interval_end") + F.expr("INTERVAL 2 HOURS")
        )
        late.writeTo("local.silver.late_events").append()
        merge_window = Window.partitionBy("business_key").orderBy(
            F.col("revision").desc(), F.col("ingested_at").desc()
        )
        on_time = (
            valid.subtract(late)
            .withColumn("merge_rank", F.row_number().over(merge_window))
            .filter(F.col("merge_rank") == 1)
            .drop("merge_rank")
        )
        on_time.createOrReplaceTempView("valid_batch")
        spark.sql(
            """
            MERGE INTO local.silver.grid_events target
            USING valid_batch source
            ON target.business_key = source.business_key
            WHEN MATCHED AND source.revision > target.revision THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
            """
        )

    curated_query = (
        parsed.writeStream.foreachBatch(curate_batch)
        .option("checkpointLocation", f"{checkpoint_root}/silver")
        .start()
    )
    spark.streams.awaitAnyTermination()
    bronze_query.stop()
    curated_query.stop()


if __name__ == "__main__":
    run()
