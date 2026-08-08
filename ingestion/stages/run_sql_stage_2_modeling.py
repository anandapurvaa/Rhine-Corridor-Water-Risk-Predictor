import os
from pathlib import Path

from google.cloud import bigquery

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "rhine-corridor-navigator")
LOCATION = os.getenv("GCP_REGION", "europe-west3")
SQL_ROOT = Path("sql")

SQL_FILES = [
    "labels/label_low_water_events.sql",
    "labels/supervised_gauge_24h_multisource.sql",
    "labels/supervised_gauge_72h.sql",
    "labels/supervised_segment_24h.sql",
    "splits/dataset_splits_gauge_24h.sql",
    "qa/qa_feature_coverage.sql",
    "qa/qa_station_coverage.sql",
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
    result = job.result()
    print(f"[DONE] {file_path} | job_id={job.job_id}")

    preview_rows = []
    try:
        preview = list(result[:10])
        if preview:
            print(f"[INFO] {file_path} previewed {len(preview)} row(s)")
            for row in preview:
                preview_rows.append(dict(row))
                print(dict(row))
    except Exception:
        pass

    return {
        "file": str(file_path),
        "status": "success",
        "job_id": job.job_id,
        "preview_rows": preview_rows,
    }


def run_stage2_sql() -> dict:
    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)

    results = []
    for i, rel_path in enumerate(SQL_FILES, start=1):
        file_path = SQL_ROOT / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"Missing SQL file: {file_path}")
        print(f"\n[Stage 2: {i}/{len(SQL_FILES)}]")
        results.append(run_sql_file(client, file_path))

    print("Stage 2 SQL files completed successfully.")
    return {
        "stage": "stage2",
        "files_executed": len(SQL_FILES),
        "results": results,
    }


if __name__ == "__main__":
    run_stage2_sql()