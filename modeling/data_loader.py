from __future__ import annotations

from typing import Iterable

import pandas as pd
from google.cloud import bigquery

from ingestion.common.config import settings


def _client() -> bigquery.Client:
    return bigquery.Client(project=settings.project_id)


def get_table_columns(table_name: str, dataset: str = "rhein_curated") -> list[str]:
    query = f"""
    SELECT column_name
    FROM `{settings.project_id}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table_name}'
    ORDER BY ordinal_position
    """
    df = _client().query(query).to_dataframe()
    return df["column_name"].tolist()


def load_bigquery_table(
    table_name: str,
    dataset: str = "rhein_curated",
    columns: Iterable[str] | None = None,
    where_sql: str | None = None,
    order_by: str | None = None,
    allow_missing_columns: bool = False,
) -> pd.DataFrame:
    available_columns = get_table_columns(table_name, dataset)

    if columns:
        requested_columns = list(columns)
        if allow_missing_columns:
            selected_columns = [c for c in requested_columns if c in available_columns]
        else:
            missing = [c for c in requested_columns if c not in available_columns]
            if missing:
                raise ValueError(
                    f"Missing columns in {dataset}.{table_name}: {missing}. "
                    f"Available columns: {available_columns}"
                )
            selected_columns = requested_columns
        selected = ", ".join(selected_columns)
    else:
        selected = "*"

    query = f"SELECT {selected} FROM `{settings.project_id}.{dataset}.{table_name}`"
    if where_sql:
        query += f" WHERE {where_sql}"
    if order_by:
        query += f" ORDER BY {order_by}"

    return _client().query(query).to_dataframe()


def write_bigquery_table(
    df: pd.DataFrame,
    table_name: str,
    dataset: str = "rhein_curated",
    if_exists: str = "replace",
) -> None:
    client = _client()
    table_id = f"{settings.project_id}.{dataset}.{table_name}"

    if if_exists == "replace":
        write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    elif if_exists == "append":
        write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    else:
        raise ValueError("if_exists must be 'replace' or 'append'")

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
    )

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()