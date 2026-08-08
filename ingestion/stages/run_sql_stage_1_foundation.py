import os
from pathlib import Path

from google.cloud import bigquery

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "rhine-corridor-navigator")
LOCATION = os.getenv("GCP_REGION", "europe-west3")
SQL_ROOT = Path("sql")

SQL_FILES = [
    "raw/raw_tables.sql",
    "raw/raw_dwd_hourly_observations.sql",
    "raw/v_pegelonline_measurements_dedup.sql",
    "raw/curated_measurements.sql",
    "dims/dim_station.sql",
    "dims/map_gauge_to_dwd_station.sql",
    "features/feature_gauge_timeseries.sql",
    "features/feature_gauge_weather_join.sql",
    "features/feature_gauge_weather_enriched.sql",
    "features/feature_modeling_multisource_v2.sql",
    "features/feature_segment_aggregation.sql",
]


def load_sql(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("\u00a0", " ")
    return text.strip()


def run_sql_file(client: bigquery.Client, file_path: Path) -> dict:
    sql = load_sql(file_path)
    if not sql:
        print(f"[SKIP] {file_path} is empty")
        return {
            "file": str(file_path),
            "status": "skipped",
        }

    print(f"[RUN ] {file_path}")
    job = client.query(sql)
    job.result()
    print(f"[DONE] {file_path} | job_id={job.job_id}")

    return {
        "file": str(file_path),
        "status": "success",
        "job_id": job.job_id,
    }


def run_stage1_sql() -> dict:
    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)

    results = []
    for i, rel_path in enumerate(SQL_FILES, start=1):
        file_path = SQL_ROOT / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"Missing SQL file: {file_path}")
        print(f"\n[Stage 1: {i}/{len(SQL_FILES)}]")
        results.append(run_sql_file(client, file_path))

    print("Stage 1 SQL files completed successfully.")
    return {
        "stage": "stage1",
        "files_executed": len(SQL_FILES),
        "results": results,
    }


if __name__ == "__main__":
    run_stage1_sql()