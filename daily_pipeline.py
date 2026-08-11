from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import pandas as pd
from typing import Any, Callable

from google.cloud import bigquery

from ingestion.main import run_source_ingestion
from ingestion.stages.run_sql_stage_1_foundation import (
    run_stage1_sql,
)
from ingestion.stages.run_sql_stage_2_modeling import (
    run_stage2_sql,
)
from modeling.predict_gauge_24h_production import (
    main as run_prediction,
)
from mlops.audit import (
    record_pipeline_run,
    record_stage_event,
)
from mlops.data_quality import (
    run_data_quality_checks,
)
from mlops.run_context import (
    PipelineRun,
    StageContext,
)
from mlops.structured_logging import (
    configure_logging,
)
from mlops.summary import (
    apply_summary,
    normalize_summary,
)


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

JOB_TYPE = "daily_ingestion_prediction"

logger = logging.getLogger(__name__)


def _ensure_utc(value):
    # 1. If it's a string (due to our BigQuery formatting), parse it first
    if isinstance(value, str):
        value = pd.to_datetime(value, utc=True)
        
    # 2. Check for missing timezone info
    if getattr(value, "tzinfo", None) is None:
        value = value.replace(tzinfo=timezone.utc)
        
    return value


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


def assert_input_freshness() -> dict[str, Any]:
    latest_input = _query_latest_complete_input()
    now_utc = datetime.now(timezone.utc)

    age_hours = (
        now_utc - latest_input
    ).total_seconds() / 3600

    logger.info(
        "input_freshness_check",
        extra={
            "input_split": PRED_SPLIT,
            "latest_complete_input_utc": (
                latest_input.isoformat()
            ),
            "input_age_hours": round(age_hours, 2),
            "max_input_age_hours": MAX_INPUT_AGE_HOURS,
        },
    )

    if age_hours > MAX_INPUT_AGE_HOURS:
        raise RuntimeError(
            "Prediction input is stale: "
            f"latest_complete_input_utc="
            f"{latest_input.isoformat()}, "
            f"age_hours={age_hours:.1f}, "
            f"max_age_hours={MAX_INPUT_AGE_HOURS:.1f}"
        )

    return {
        "latest_input_timestamp": (
            latest_input.isoformat()
        ),
        "input_age_hours": round(age_hours, 2),
        "prediction_split": PRED_SPLIT,
    }


def run_stage(
    run: PipelineRun,
    stage_name: str,
    function: Callable[[], Any],
) -> dict[str, Any]:
    with StageContext(stage_name) as stage:
        logger.info(
            "stage_started",
            extra={
                "run_id": run.run_id,
                "stage_name": stage_name,
            },
        )

        result = function()
        summary = normalize_summary(result)

        stage.metadata.update(summary)
        apply_summary(run.metadata, summary)

        logger.info(
            "stage_completed",
            extra={
                "run_id": run.run_id,
                "stage_name": stage_name,
                "duration_seconds": (
                    stage.duration_seconds
                ),
                **summary,
            },
        )

    record_stage_event(
        run=run,
        stage=stage,
    )

    return summary


def _resolve_data_quality_status(
    summary: dict[str, Any],
) -> str:
    for key in (
        "data_quality_status",
        "quality_status",
        "status",
        "result",
    ):
        value = summary.get(key)

        if value is not None:
            normalized = str(value).strip().lower()

            if normalized in {
                "pass",
                "passed",
                "success",
                "successful",
                "ok",
            }:
                return "pass"

            if normalized in {
                "fail",
                "failed",
                "failure",
                "error",
            }:
                return "fail"

    failed_checks = summary.get("failed_checks")

    if isinstance(failed_checks, int):
        return "fail" if failed_checks > 0 else "pass"

    if isinstance(failed_checks, (list, tuple, set)):
        return "fail" if failed_checks else "pass"

    return "pass"


def main() -> None:
    configure_logging(
        os.getenv("LOG_LEVEL", "INFO")
    )

    run = PipelineRun(
        job_type=JOB_TYPE,
        metadata={
            "cloud_run_job": os.getenv(
                "CLOUD_RUN_JOB",
                "rhine-daily-pipeline",
            ),
            "project_id": PROJECT_ID,
            "region": REGION,
            "input_split": PRED_SPLIT,
        },
    ).start()

    os.environ["MLOPS_RUN_ID"] = run.run_id

    logger.info(
        "pipeline_started",
        extra={
            "run_id": run.run_id,
            "job_type": JOB_TYPE,
            "input_split": PRED_SPLIT,
        },
    )

    try:
        run_stage(
            run,
            "daily_ingestion",
            lambda: run_source_ingestion(
                source="all",
                mode="recent",
                hours=72,
                from_date=None,
                to_date=None,
                chunk_months=1,
            ),
        )

        run_stage(
            run,
            "stage_1_foundation",
            run_stage1_sql,
        )

        run_stage(
            run,
            "stage_2_modeling",
            run_stage2_sql,
        )

        run_stage(
            run,
            "input_freshness_check",
            assert_input_freshness,
        )

        run_stage(
            run,
            "production_prediction",
            run_prediction,
        )

        data_quality_summary = run_stage(
            run,
            "data_quality_checks",
            lambda: run_data_quality_checks(
                run_id=run.run_id,
            ),
        )

        data_quality_status = _resolve_data_quality_status(
            data_quality_summary
        )

        logger.info(
            "data_quality_checks_completed",
            extra={
                "run_id": run.run_id,
                "data_quality_status": data_quality_status,
                "failed_checks": (
                    data_quality_summary.get(
                        "failed_checks",
                        [],
                    )
                ),
                **data_quality_summary,
            },
        )

        if data_quality_status == "fail":
            raise RuntimeError(
                "Data-quality checks failed"
            )

        run.complete(status="success")

        logger.info(
            "pipeline_completed",
            extra={
                "run_id": run.run_id,
                "status": run.status,
                "duration_seconds": (
                    run.duration_seconds
                ),
                **run.metadata,
            },
        )

    except Exception as exc:
        run.fail(exc)

        logger.exception(
            "pipeline_failed",
            extra={
                "run_id": run.run_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )

        raise

    finally:
        record_pipeline_run(run)
        run.close()


if __name__ == "__main__":
    main()