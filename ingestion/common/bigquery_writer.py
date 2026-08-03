import pandas as pd
from google.cloud import bigquery

from ingestion.common.config import settings
from ingestion.common.logging_utils import get_logger

logger = get_logger(__name__)


BQ_PEGELONLINE_SCHEMA = [
    bigquery.SchemaField("station_id", "STRING"),
    bigquery.SchemaField("station_name", "STRING"),
    bigquery.SchemaField("timeseries_name", "STRING"),
    bigquery.SchemaField("timestamp_utc", "TIMESTAMP"),
    bigquery.SchemaField("value", "FLOAT64"),
    bigquery.SchemaField("unit", "STRING"),
    bigquery.SchemaField("latitude", "FLOAT64"),
    bigquery.SchemaField("longitude", "FLOAT64"),
    bigquery.SchemaField("ingestion_ts_utc", "TIMESTAMP"),
    bigquery.SchemaField("source", "STRING"),
    bigquery.SchemaField("source_record_hash", "STRING"),
    bigquery.SchemaField("source_url", "STRING"),
]

BQ_DWD_HOURLY_SCHEMA = [
    bigquery.SchemaField("dwd_station_id", "STRING"),
    bigquery.SchemaField("dwd_station_name", "STRING"),
    bigquery.SchemaField("timestamp_utc", "TIMESTAMP"),
    bigquery.SchemaField("latitude", "FLOAT64"),
    bigquery.SchemaField("longitude", "FLOAT64"),
    bigquery.SchemaField("temperature_c", "FLOAT64"),
    bigquery.SchemaField("precipitation_mm", "FLOAT64"),
    bigquery.SchemaField("wind_speed_ms", "FLOAT64"),
    bigquery.SchemaField("pressure_hpa", "FLOAT64"),
    bigquery.SchemaField("relative_humidity_pct", "FLOAT64"),
    bigquery.SchemaField("is_proxy_backfilled", "BOOL"),
    bigquery.SchemaField("proxy_source_station_id", "STRING"),
    bigquery.SchemaField("proxy_source_variable", "STRING"),
    bigquery.SchemaField("proxy_source_distance_km", "FLOAT64"),
    bigquery.SchemaField("proxy_fill_method", "STRING"),
    bigquery.SchemaField("ingestion_ts_utc", "TIMESTAMP"),
    bigquery.SchemaField("source", "STRING"),
    bigquery.SchemaField("source_record_hash", "STRING"),
    bigquery.SchemaField("source_url", "STRING"),
]


def normalize_pegelonline_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    if "ingestion_ts_utc" in df.columns:
        df["ingestion_ts_utc"] = pd.to_datetime(df["ingestion_ts_utc"], utc=True, errors="coerce")

    for col in ["value", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["station_id", "station_name", "timeseries_name", "unit", "source", "source_record_hash", "source_url"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def normalize_dwd_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["timestamp_utc", "ingestion_ts_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    for col in [
        "latitude",
        "longitude",
        "temperature_c",
        "precipitation_mm",
        "wind_speed_ms",
        "pressure_hpa",
        "relative_humidity_pct",
        "proxy_source_distance_km",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_proxy_backfilled" in df.columns:
        df["is_proxy_backfilled"] = df["is_proxy_backfilled"].astype("boolean")

    for col in [
        "dwd_station_id",
        "dwd_station_name",
        "proxy_source_station_id",
        "proxy_source_variable",
        "proxy_fill_method",
        "source",
        "source_record_hash",
        "source_url",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df


def write_dataframe_to_bigquery(
    df: pd.DataFrame,
    table_name: str,
    schema: list[bigquery.SchemaField] | None = None,
) -> None:
    if df.empty:
        logger.info("bq_write_skipped table=%s reason=empty_dataframe", table_name)
        return

    if not settings.project_id:
        raise ValueError("GCP_PROJECT_ID is empty. Check your .env file.")

    table_id = f"{settings.project_id}.{settings.dataset_raw}.{table_name}"

    logger.info(
        "bq_write_start table=%s full_table_id=%s rows=%s",
        table_name,
        table_id,
        len(df),
    )

    if table_name == "pegelonline_measurements":
        df = normalize_pegelonline_dataframe(df)
        schema = schema or BQ_PEGELONLINE_SCHEMA

    if table_name == "dwd_hourly_observations":
        df = normalize_dwd_dataframe(df)
        schema = schema or BQ_DWD_HOURLY_SCHEMA

    logger.info(
        "bq_dtypes table=%s dtypes=%s",
        table_name,
        {col: str(dtype) for col, dtype in df.dtypes.items()},
    )

    client = bigquery.Client(project=settings.project_id)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=schema,
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    logger.info(
        "bq_write_complete table=%s rows=%s project=%s dataset=%s",
        table_name,
        len(df),
        settings.project_id,
        settings.dataset_raw,
    )