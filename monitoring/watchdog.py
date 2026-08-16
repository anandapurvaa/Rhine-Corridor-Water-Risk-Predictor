from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from google.cloud import bigquery


PROJECT_ID = os.getenv(
    "GCP_PROJECT_ID",
    "rhine-corridor-navigator",
).strip()

REGION = os.getenv(
    "GCP_REGION",
    "europe-west3",
).strip()

MLOPS_DATASET = os.getenv(
    "MLOPS_DATASET",
    "mlops",
).strip()

EXPECTED_STATIONS = int(
    os.getenv("EXPECTED_STATION_COUNT", "19")
)

MIN_PREDICTION_ROWS = int(
    os.getenv("MIN_PREDICTION_ROWS", "19")
)

MAX_PREDICTION_AGE_HOURS = float(
    os.getenv("MAX_PREDICTION_AGE_HOURS", "26")
)

MAX_VALIDATION_EVALUATION_AGE_HOURS = float(
    os.getenv(
        "MAX_VALIDATION_EVALUATION_AGE_HOURS",
        "840",
    )
)

MAX_MAE = float(
    os.getenv("MAX_MAE", "999999")
)

MAX_RMSE = float(
    os.getenv("MAX_RMSE", "999999")
)

FAIL_ON_MISSING_EVALUATION = (
    os.getenv(
        "FAIL_ON_MISSING_EVALUATION",
        "false",
    ).lower()
    == "true"
)

logging.basicConfig(
    stream=sys.stdout,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(message)s",
)

logger = logging.getLogger("gauge24h-watchdog")

client = bigquery.Client(
    project=PROJECT_ID,
    location=REGION,
)


def emit(
    event: str,
    status: str,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "status": status,
        "service": "gauge24h-watchdog",
        "project_id": PROJECT_ID,
        "region": REGION,
        "measured_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        **fields,
    }

    logger.info(
        json.dumps(
            payload,
            default=str,
            separators=(",", ":"),
        )
    )


def query(
    sql: str,
) -> list[dict[str, Any]]:
    return [
        dict(row.items())
        for row in client.query(
            sql,
            location=REGION,
        ).result()
    ]


def parse_utc(
    value: Any,
) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()

        if text.lower() in {
            "",
            "none",
            "null",
            "nan",
            "nat",
        }:
            return None

        normalized = text

        if normalized.endswith(" UTC"):
            normalized = (
                normalized[:-4].strip()
                + "+00:00"
            )
        elif normalized.endswith("Z"):
            normalized = (
                normalized[:-1]
                + "+00:00"
            )

        try:
            parsed = datetime.fromisoformat(
                normalized
            )
        except ValueError:
            formats = (
                "%Y-%m-%d %H:%M:%S UTC",
                "%Y-%m-%d %H:%M:%S.%f UTC",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
            )

            for date_format in formats:
                try:
                    parsed = datetime.strptime(
                        text,
                        date_format,
                    )
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(
                    f"Unsupported timestamp format: {value!r}"
                )

    if parsed.tzinfo is None:
        return parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(timezone.utc)


def age_hours(
    value: Any,
) -> float | None:
    parsed = parse_utc(value)

    if parsed is None:
        return None

    return (
        datetime.now(timezone.utc) - parsed
    ).total_seconds() / 3600


def check_pipeline() -> bool:
    sql = f"""
        SELECT *
        FROM `{PROJECT_ID}.{MLOPS_DATASET}.v_latest_pipeline_health_by_job`
        ORDER BY job_type
    """

    rows = query(sql)

    if not rows:
        emit(
            "pipeline_health",
            "fail",
            reason="no_pipeline_health_rows",
        )
        return False

    valid_statuses = {
        "success",
        "succeeded",
        "ok",
        "pass",
    }

    failed_jobs = [
        row
        for row in rows
        if str(
            row.get("status", "")
        ).lower()
        not in valid_statuses
    ]

    ok = not failed_jobs

    emit(
        "pipeline_health",
        "pass" if ok else "fail",
        job_count=len(rows),
        failed_job_count=len(failed_jobs),
        jobs=rows,
    )

    return ok


def check_quality() -> bool:
    sql = f"""
        SELECT *
        FROM `{PROJECT_ID}.{MLOPS_DATASET}.v_latest_quality_health`
        ORDER BY metric_name, metric_scope
    """

    rows = query(sql)

    if not rows:
        emit(
            "data_quality_health",
            "fail",
            reason="no_quality_health_rows",
        )
        return False

    valid_statuses = {
        "pass",
        "passed",
        "ok",
        "success",
    }

    failed_metrics = [
        row
        for row in rows
        if str(
            row.get("status", "")
        ).lower()
        not in valid_statuses
    ]

    ok = not failed_metrics

    emit(
        "data_quality_health",
        "pass" if ok else "fail",
        metric_count=len(rows),
        failed_metric_count=len(
            failed_metrics
        ),
        metrics=rows,
    )

    return ok


def check_stages() -> bool:
    sql = f"""
        SELECT *
        FROM `{PROJECT_ID}.{MLOPS_DATASET}.v_latest_stage_health`
        ORDER BY stage_name
    """

    rows = query(sql)

    if not rows:
        emit(
            "stage_health",
            "fail",
            reason="no_stage_health_rows",
        )
        return False

    valid_statuses = {
        "success",
        "succeeded",
        "ok",
        "pass",
    }

    failed_stages = [
        row
        for row in rows
        if str(
            row.get("status", "")
        ).lower()
        not in valid_statuses
    ]

    ok = not failed_stages

    emit(
        "stage_health",
        "pass" if ok else "fail",
        stage_count=len(rows),
        failed_stage_count=len(
            failed_stages
        ),
        stages=rows,
    )

    return ok


def check_validation_evaluation() -> bool:
    sql = f"""
        SELECT *
        FROM `{PROJECT_ID}.{MLOPS_DATASET}.v_latest_evaluation_health`
        ORDER BY evaluated_at_utc DESC
    """

    rows = query(sql)

    if not rows:
        ok = not FAIL_ON_MISSING_EVALUATION

        emit(
            "validation_evaluation_health",
            "pass" if ok else "fail",
            reason="no_evaluation_health_rows",
            max_age_hours=(
                MAX_VALIDATION_EVALUATION_AGE_HOURS
            ),
        )

        return ok

    evaluation = rows[0]

    evaluation_age = age_hours(
        evaluation.get("evaluated_at_utc")
    )

    mae = (
        float(evaluation["mae"])
        if evaluation.get("mae") is not None
        else None
    )

    rmse = (
        float(evaluation["rmse"])
        if evaluation.get("rmse") is not None
        else None
    )

    checks = {
        "available": (
            evaluation.get("evaluation_status")
            == "available"
        ),
        "freshness_ok": (
            evaluation_age is None
            or evaluation_age
            <= MAX_VALIDATION_EVALUATION_AGE_HOURS
        ),
        "mae_ok": (
            mae is None
            or mae <= MAX_MAE
        ),
        "rmse_ok": (
            rmse is None
            or rmse <= MAX_RMSE
        ),
    }

    ok = all(checks.values())

    emit(
        "validation_evaluation_health",
        "pass" if ok else "fail",
        checks=checks,
        age_hours=evaluation_age,
        max_age_hours=(
            MAX_VALIDATION_EVALUATION_AGE_HOURS
        ),
        mae=mae,
        rmse=rmse,
        evaluation=evaluation,
    )

    return ok


def main() -> int:
    emit(
        "watchdog_started",
        "ok",
        views={
            "pipeline": (
                f"{MLOPS_DATASET}."
                "v_latest_pipeline_health_by_job"
            ),
            "quality": (
                f"{MLOPS_DATASET}."
                "v_latest_quality_health"
            ),
            "stage": (
                f"{MLOPS_DATASET}."
                "v_latest_stage_health"
            ),
            "validation_evaluation": (
                f"{MLOPS_DATASET}."
                "v_latest_evaluation_health"
            ),
        },
    )

    try:
        results = [
            check_pipeline(),
            check_quality(),
            check_stages(),
            check_validation_evaluation(),
        ]
    except Exception as exc:
        emit(
            "watchdog_failed",
            "fail",
            error=repr(exc),
        )
        return 1

    passed = sum(
        1
        for result in results
        if result
    )

    ok = all(results)

    emit(
        "watchdog_completed",
        "pass" if ok else "fail",
        checks_passed=passed,
        checks_total=len(results),
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())