import os
from pathlib import Path
from google.cloud import bigquery

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "rhine-corridor-navigator")
SQL_ROOT = Path("sql")

SQL_FILES = [
    "labels/label_low_water_events.sql",
    "labels/supervised_gauge_24h_multisource.sql",   # use this OR supervised_gauge_24h.sql
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

def run_sql_file(client: bigquery.Client, file_path: Path) -> None:
    sql = load_sql(file_path)
    if not sql:
        print(f"[SKIP] {file_path} is empty")
        return

    print(f"[RUN ] {file_path}")
    job = client.query(sql)
    result = job.result()
    print(f"[DONE] {file_path} | job_id={job.job_id}")

    try:
        rows = list(result)
        if rows:
            print(f"[INFO] {file_path} returned {len(rows)} row(s)")
            for row in rows[:10]:
                print(dict(row))
    except Exception:
        pass

if __name__ == "__main__":
    client = bigquery.Client(project=PROJECT_ID)

    for rel_path in SQL_FILES:
        file_path = SQL_ROOT / rel_path
        if not file_path.exists():
            raise FileNotFoundError(f"Missing SQL file: {file_path}")
        run_sql_file(client, file_path)

    print("Labels, splits, and QA SQL files completed successfully.")