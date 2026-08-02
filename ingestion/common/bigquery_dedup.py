from __future__ import annotations

import uuid

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from ingestion.common.bigquery_writer import write_dataframe_to_bigquery
from ingestion.common.config import settings
from ingestion.common.logging_utils import get_logger

logger = get_logger(__name__)


def merge_dataframe_to_bigquery(
    df: pd.DataFrame,
    table_name: str,
    key_column: str = "source_record_hash",
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

    if key_column not in df.columns:
        raise ValueError(f"Missing required key column: {key_column}")

    stage_df = (
        df.copy()
        .drop_duplicates(subset=[key_column], keep="last")
        .reset_index(drop=True)
    )

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

    load_job = client.load_table_from_dataframe(stage_df, staging_table_id)
    load_job.result()

    columns = [field.name for field in target_table.schema]
    if key_column not in columns:
        client.delete_table(staging_table_id, not_found_ok=True)
        raise ValueError(f"Key column {key_column} not found in target table {target_table_id}")

    update_columns = [col for col in columns if col != key_column]
    insert_columns_sql = ", ".join(f"`{col}`" for col in columns)
    insert_values_sql = ", ".join(f"S.`{col}`" for col in columns)
    update_set_sql = ", ".join(f"T.`{col}` = S.`{col}`" for col in update_columns)

    merge_sql = f"""
    MERGE `{target_table_id}` T
    USING (
      SELECT * EXCEPT(rn)
      FROM (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY `{key_column}`
            ORDER BY ingestion_ts_utc DESC
          ) AS rn
        FROM `{staging_table_id}`
      )
      WHERE rn = 1
    ) S
    ON T.`{key_column}` = S.`{key_column}`
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