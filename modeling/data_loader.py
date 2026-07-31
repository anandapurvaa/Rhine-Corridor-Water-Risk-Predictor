from google.cloud import bigquery
import pandas as pd

from ingestion.common.config import settings


def load_bigquery_table(table_name: str, dataset: str = "rhein_curated") -> pd.DataFrame:
    client = bigquery.Client(project=settings.project_id)
    query = f"SELECT * FROM `{settings.project_id}.{dataset}.{table_name}`"
    return client.query(query).to_dataframe()