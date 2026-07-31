import os
from pathlib import Path
from google.cloud import bigquery

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "rhine-corridor-navigator")
SQL_ROOT = Path("sql")

SQL_FILES = [
    "raw/raw_tables.sql",
    "raw/raw_dwd_hourly_observations.sql",
    "raw/curated_measurements.sql",
    "dims/dim_station.sql",
    "dims/map_gauge_to_dwd_station.sql",
    "features/feature_gauge_timeseries.sql",
    "features/feature_gauge_weather_join.sql",
    "features/feature_gauge_weather_enriched.sql",
    "features/feature_modeling_multisource_v2.sql",
]

def load_sql(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("\u00a0", " ")
    return text.strip()

def run_sql_file(client: bigquery.Client, file_path: Path) -> None:
    sql = load_sql(file_path)
    if not sql:
        print(f"[SKIP] {file_path} is empty")
        return

    print(f"[RUN ] {file_path}")
    job = client.query(sql)
    job.result()
    print(f"[DONE] {file_path} | job_id={job.job_id}")

if __name__ == "__main__":
    client = bigquery.Client(project=PROJECT_ID)

    for rel_path in SQL_FILES:
        file_path = SQL_ROOT / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"Missing SQL file: {file_path}")
        run_sql_file(client, file_path)

    print("All first 9 SQL files completed successfully.")