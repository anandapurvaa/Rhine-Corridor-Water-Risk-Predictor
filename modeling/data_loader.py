from __future__ import annotations

from typing import Iterable
import os

import pandas as pd
from google.cloud import bigquery


def _get_project_id() -> str:
    raw_project = os.getenv(
        "GCP_PROJECT_ID",
        "rhine-corridor-navigator",
    ).strip()
    return raw_project.split()[0]


def _get_region() -> str:
    raw_region = os.getenv(
        "GCP_REGION",
        "europe-west3",
    ).strip()
    return raw_region.split()[0]


def _client() -> bigquery.Client:
    return bigquery.Client(
        project=_get_project_id(),
        location=_get_region(),
    )


def _split_table_ref(
    table_name: str,
    dataset: str,
) -> tuple[str, str, str]:
    parts = table_name.split(".")

    if len(parts) == 3:
        return parts[0], parts[1], parts[2]

    if len(parts) == 2:
        return _get_project_id(), parts[0], parts[1]

    return _get_project_id(), dataset, table_name


def _quote_table(
    project_id: str,
    dataset: str,
    table_name: str,
) -> str:
    return f"`{project_id}.{dataset}.{table_name}`"


def get_table_columns(
    table_name: str,
    dataset: str = "rhein_curated",
) -> list[str]:
    project_id, dataset_name, bare_table_name = _split_table_ref(
        table_name,
        dataset,
    )

    query = f"""
        SELECT column_name
        FROM `{project_id}.{dataset_name}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = @table_name
        ORDER BY ordinal_position
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "table_name",
                "STRING",
                bare_table_name,
            )
        ]
    )

    rows = _client().query(
        query,
        job_config=job_config,
        location=_get_region(),
    ).result()

    return [row["column_name"] for row in rows]


def load_bigquery_table(
    table_name: str,
    dataset: str = "rhein_curated",
    columns: Iterable[str] | None = None,
    where_sql: str | None = None,
    order_by: str | None = None,
    allow_missing_columns: bool = False,
) -> pd.DataFrame:
    project_id, dataset_name, bare_table_name = _split_table_ref(
        table_name,
        dataset,
    )

    if columns is None:
        selected_columns = ["*"]
    else:
        requested_columns = [
            str(column).strip()
            for column in columns
            if str(column).strip()
        ]

        if allow_missing_columns:
            available_columns = get_table_columns(
                bare_table_name,
                dataset_name,
            )
            selected_columns = [
                column
                for column in requested_columns
                if column in available_columns
            ]
        else:
            selected_columns = requested_columns

        if not selected_columns:
            selected_columns = ["*"]

    selected = ", ".join(selected_columns)

    query = f"""
        SELECT {selected}
        FROM {_quote_table(project_id, dataset_name, bare_table_name)}
    """

    if where_sql:
        query += f" WHERE {where_sql}"

    if order_by:
        query += f" ORDER BY {order_by}"

    query_job = _client().query(
        query,
        location=_get_region(),
    )

    # Avoid BigQuery Storage API's larger memory footprint.
    return query_job.to_dataframe(
        create_bqstorage_client=False,
    )


def write_bigquery_table(
    df: pd.DataFrame,
    table_name: str,
    dataset: str = "rhein_curated",
    if_exists: str = "replace",
) -> None:
    client = _client()

    project_id, dataset_name, bare_table_name = _split_table_ref(
        table_name,
        dataset,
    )
    table_id = f"{project_id}.{dataset_name}.{bare_table_name}"

    if if_exists == "replace":
        write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    elif if_exists == "append":
        write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    else:
        raise ValueError(
            "if_exists must be 'replace' or 'append'"
        )

    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        autodetect=True,
    )

    job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=job_config,
    )
    job.result()