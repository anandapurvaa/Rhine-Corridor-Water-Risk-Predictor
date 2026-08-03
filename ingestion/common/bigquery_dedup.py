from __future__ import annotations

import uuid

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from ingestion.common.bigquery_writer import write_dataframe_to_bigquery
from ingestion.common.config import settings
from ingestion.common.logging_utils import get_logger

logger = get_logger(__name__)


def _normalize_merge_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["timestamp_utc", "ingestion_ts_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    return df


def merge_dataframe_to_bigquery(
    df: pd.DataFrame,
    table_name: str,
    key_column: str = "source_record_hash",
    key_columns: list[str] | None = None,
    dataset: str | None = None,
) -> None:
    if df.empty:
        logger.info("bq_merge_skip_empty_df table=%s", table_name)
        return

    if not settings.project_id:
        logger.info("bq_merge_skip_no_project_id table=%s", table_name)
        return

    target_dataset = dataset or settings.dataset_raw
    project_id = settings.project_id
    location = settings.gcp_region

    merge_keys = key_columns or [key_column]
    missing_keys = [col for col in merge_keys if col not in df.columns]
    if missing_keys:
        raise ValueError(f"Missing required key columns: {missing_keys}")

    stage_df = (
        df.copy()
        .drop_duplicates(subset=merge_keys, keep="last")
        .reset_index(drop=True)
    )

    stage_df = _normalize_merge_dataframe(stage_df)

    client = bigquery.Client(project=project_id, location=location)
    target_table_id = f"{project_id}.{target_dataset}.{table_name}"
    staging_table_id = f"{project_id}.{target_dataset}._stg_{table_name}_{uuid.uuid4().hex[:12]}"

    try:
        target_table = client.get_table(target_table_id)
    except NotFound:
        logger.info("bq_target_missing_bootstrap_start table=%s", target_table_id)
        write_dataframe_to_bigquery(stage_df, table_name=table_name)
        logger.info("bq_target_missing_bootstrap_complete table=%s", target_table_id)
        return

    logger.info(
        "bq_merge_stage_start target_table=%s staging_table=%s rows=%s",
        target_table_id,
        staging_table_id,
        len(stage_df),
    )

    target_schema_by_name = {field.name: field for field in target_table.schema}

    missing_target_keys = [col for col in merge_keys if col not in target_schema_by_name]
    if missing_target_keys:
        raise ValueError(f"Key columns {missing_target_keys} not found in target table {target_table_id}")

    stage_columns = [col for col in stage_df.columns if col in target_schema_by_name]
    missing_stage_columns = [col for col in stage_df.columns if col not in target_schema_by_name]
    if missing_stage_columns:
        logger.warning(
            "bq_merge_stage_extra_columns_ignored table=%s columns=%s",
            target_table_id,
            ",".join(missing_stage_columns),
        )
        stage_df = stage_df[stage_columns].copy()

    load_schema = [target_schema_by_name[col] for col in stage_columns]

    job_config = bigquery.LoadJobConfig(schema=load_schema)

    load_job = client.load_table_from_dataframe(
        stage_df,
        staging_table_id,
        job_config=job_config,
    )
    load_job.result()

    staging_table = client.get_table(staging_table_id)
    staging_schema_by_name = {field.name: field.field_type for field in staging_table.schema}

    logger.info(
        "bq_merge_schema_check target_table=%s merge_keys=%s target_types=%s staging_types=%s",
        target_table_id,
        merge_keys,
        {k: target_schema_by_name[k].field_type for k in merge_keys},
        {k: staging_schema_by_name.get(k) for k in merge_keys},
    )

    merge_columns = [field.name for field in target_table.schema if field.name in stage_columns]
    update_columns = [col for col in merge_columns if col not in merge_keys]

    insert_columns_sql = ", ".join(f"`{col}`" for col in merge_columns)
    insert_values_sql = ", ".join(f"S.`{col}`" for col in merge_columns)
    update_set_sql = ", ".join(f"T.`{col}` = S.`{col}`" for col in update_columns)
    merge_on_sql = " AND ".join(f"T.`{col}` = S.`{col}`" for col in merge_keys)
    partition_by_sql = ", ".join(f"`{col}`" for col in merge_keys)

    merge_sql = f"""
    MERGE `{target_table_id}` T
    USING (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY {partition_by_sql}
            ORDER BY ingestion_ts_utc DESC
          ) AS rn
        FROM `{staging_table_id}`
      )
      WHERE rn = 1
    ) S
    ON {merge_on_sql}
    WHEN MATCHED THEN
      UPDATE SET {update_set_sql}
    WHEN NOT MATCHED THEN
      INSERT ({insert_columns_sql})
      VALUES ({insert_values_sql})
    """

    try:
        logger.info("bq_merge_start target_table=%s staging_table=%s", target_table_id, staging_table_id)
        client.query(merge_sql).result()
        logger.info("bq_merge_complete target_table=%s", target_table_id)
    finally:
        client.delete_table(staging_table_id, not_found_ok=True)
        logger.info("bq_merge_stage_deleted staging_table=%s", staging_table_id)