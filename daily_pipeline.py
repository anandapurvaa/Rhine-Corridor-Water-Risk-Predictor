from __future__ import annotations

from datetime import datetime, timezone
import os

from google.cloud import bigquery

from ingestion.main import run_source_ingestion
from ingestion.stages.run_sql_stage_1_foundation import run_stage1_sql
from ingestion.stages.run_sql_stage_2_modeling import run_stage2_sql
from modeling.predict_gauge_24h_production import main as run_prediction


PROJECT_ID = os.getenv(
    "GCP_PROJECT_ID",
    "rhine-corridor-navigator",
).strip()

REGION = os.getenv(
    "GCP_REGION",
    "europe-west3",
).strip()

PRED_SPLIT = os.getenv(
    "GAUGE24H_PRED_SPLIT",
    "test",
).strip().lower()

MAX_INPUT_AGE_HOURS = float(
    os.getenv(
        "GAUGE24H_MAX_INPUT_AGE_HOURS",
        "72",
    )
)


def _ensure_utc(value) -> datetime:
    if value is None:
        raise RuntimeError("Timestamp value is NULL")

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def _query_latest_complete_input() -> datetime:
    client = bigquery.Client(
        project=PROJECT_ID,
        location=REGION,
    )

    query = """
        SELECT MAX(timestamp_utc) AS latest_timestamp_utc
        FROM `rhine-corridor-navigator.rhein_curated.dataset_splits_gauge_24h`
        WHERE split_name = @split_name
          AND temperature_c IS NOT NULL
          AND precipitation_mm IS NOT NULL
          AND wind_speed_ms IS NOT NULL
          AND pressure_hpa IS NOT NULL
          AND relative_humidity_pct IS NOT NULL
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "split_name",
                "STRING",
                PRED_SPLIT,
            )
        ]
    )

    row = next(
        client.query(
            query,
            job_config=job_config,
            location=REGION,
        ).result()
    )

    return _ensure_utc(row["latest_timestamp_utc"])


def assert_input_freshness() -> None:
    latest_input = _query_latest_complete_input()
    now_utc = datetime.now(timezone.utc)
    age_hours = (now_utc - latest_input).total_seconds() / 3600

    print(
        "Input freshness check: "
        f"split={PRED_SPLIT} "
        f"latest_complete_input_utc={latest_input.isoformat()} "
        f"age_hours={age_hours:.1f} "
        f"max_age_hours={MAX_INPUT_AGE_HOURS:.1f}"
    )

    if age_hours > MAX_INPUT_AGE_HOURS:
        raise RuntimeError(
            "Prediction input is stale: "
            f"latest_complete_input_utc={latest_input.isoformat()}, "
            f"age_hours={age_hours:.1f}, "
            f"max_age_hours={MAX_INPUT_AGE_HOURS:.1f}"
        )


def main() -> None:
    print("=== Daily ingestion ===")
    run_source_ingestion(
        source="all",
        mode="recent",
        hours=72,
        from_date=None,
        to_date=None,
        chunk_months=1,
    )

    print("=== Stage 1 foundation ===")
    run_stage1_sql()

    print("=== Stage 2 modeling ===")
    run_stage2_sql()

    print("=== Input freshness check ===")
    assert_input_freshness()

    print("=== Production prediction ===")
    run_prediction()

    print("=== Daily pipeline completed successfully ===")


if __name__ == "__main__":
    main()