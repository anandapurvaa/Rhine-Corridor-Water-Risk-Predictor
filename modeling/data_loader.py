from __future__ import annotations

from typing import Iterable

import pandas as pd
from google.cloud import bigquery

from ingestion.common.config import settings


def _client() -> bigquery.Client:
    return bigquery.Client(
        project=settings.project_id,
        location=settings.gcp_region,
    )


def _split_table_ref(table_name: str, dataset: str) -> tuple[str, str, str]:
    parts = table_name.split(".")
    if len(parts) == 3:
        project_id, dataset_name, bare_table_name = parts
        return project_id, dataset_name, bare_table_name
    if len(parts) == 2:
        dataset_name, bare_table_name = parts
        return settings.project_id, dataset_name, bare_table_name
    return settings.project_id, dataset, table_name


def _quote_table(project_id: str, dataset: str, table_name: str) -> str:
    return f"`{project_id}.{dataset}.{table_name}`"


def get_table_columns(table_name: str, dataset: str = "rhein_curated") -> list[str]:
    project_id, dataset_name, bare_table_name = _split_table_ref(table_name, dataset)

    query = f"""
    SELECT column_name
    FROM `{project_id}.{dataset_name}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{bare_table_name}'
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
    project_id, dataset_name, bare_table_name = _split_table_ref(table_name, dataset)
    available_columns = get_table_columns(bare_table_name, dataset_name)

    if columns is None:
        selected_columns = ["*"]
    else:
        requested_columns = [str(c).strip() for c in columns if str(c).strip()]

        if allow_missing_columns:
            selected_columns = [c for c in requested_columns if c in available_columns]
        else:
            missing = [c for c in requested_columns if c not in available_columns]
            if missing:
                raise ValueError(
                    f"Missing columns in {dataset_name}.{bare_table_name}: {missing}. "
                    f"Available columns: {available_columns}"
                )
            selected_columns = requested_columns

        if not selected_columns:
            selected_columns = ["*"]

    selected = ", ".join(selected_columns)
    query = f"SELECT {selected} FROM {_quote_table(project_id, dataset_name, bare_table_name)}"

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
    project_id, dataset_name, bare_table_name = _split_table_ref(table_name, dataset)
    table_id = f"{project_id}.{dataset_name}.{bare_table_name}"

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