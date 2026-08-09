from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from google.cloud import bigquery


logger = logging.getLogger(__name__)


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

MAX_MISSING_FRACTION = float(
    os.getenv(
        "GAUGE24H_MAX_MISSING_FRACTION",
        "0.20",
    )
)

MIN_PREDICTION_ROWS = int(
    os.getenv(
        "GAUGE24H_MIN_PREDICTION_ROWS",
        "1",
    )
)

QUALITY_TABLE = (
    f"{PROJECT_ID}.mlops.data_quality_metrics"
)

TRAIN_TABLE = (
    f"{PROJECT_ID}.rhein_curated."
    "dataset_splits_gauge_24h"
)

if PRED_SPLIT == "production":
    PREDICTION_TABLE = (
        f"{PROJECT_ID}.rhein_curated."
        "gauge_24h_production_predictions"
    )
else:
    PREDICTION_TABLE = (
        f"{PROJECT_ID}.rhein_curated."
        f"gauge_24h_production_predictions_{PRED_SPLIT}"
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _client() -> bigquery.Client:
    return bigquery.Client(
        project=PROJECT_ID,
        location=REGION,
    )


def _safe_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def _query_one(
    query: str,
    parameters: list[bigquery.QueryParameter] | None = None,
) -> dict[str, Any]:
    job_config = None

    if parameters:
        job_config = bigquery.QueryJobConfig(
            query_parameters=parameters
        )

    row = next(
        _client()
        .query(
            query,
            job_config=job_config,
            location=REGION,
        )
        .result()
    )

    return dict(row)


def _build_metric(
    run_id: str,
    metric_name: str,
    metric_scope: str,
    metric_value: float | None,
    threshold_value: float | None,
    status: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "metric_id": uuid4().hex,
        "run_id": run_id,
        "metric_name": metric_name,
        "metric_scope": metric_scope,
        "metric_value": metric_value,
        "threshold_value": threshold_value,
        "status": status,
        "measured_at_utc": utc_now().isoformat(),
        "details_json": _safe_json(details or {}),
    }


def _write_metrics(
    metrics: list[dict[str, Any]],
) -> None:
    if not metrics:
        return

    job = _client().load_table_from_json(
        metrics,
        QUALITY_TABLE,
        job_config=bigquery.LoadJobConfig(
            write_disposition=(
                bigquery.WriteDisposition.WRITE_APPEND
            )
        ),
    )

    job.result()


def check_training_input(
    run_id: str,
) -> list[dict[str, Any]]:
    query = f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNTIF(
                temperature_c IS NULL
                OR precipitation_mm IS NULL
                OR wind_speed_ms IS NULL
                OR pressure_hpa IS NULL
                OR relative_humidity_pct IS NULL
            ) AS incomplete_rows,
            MAX(timestamp_utc) AS latest_timestamp_utc
        FROM `{TRAIN_TABLE}`
        WHERE split_name = @split_name
    """

    row = _query_one(
        query,
        [
            bigquery.ScalarQueryParameter(
                "split_name",
                "STRING",
                PRED_SPLIT,
            )
        ],
    )

    total_rows = int(row["total_rows"] or 0)
    incomplete_rows = int(
        row["incomplete_rows"] or 0
    )

    missing_fraction = (
        incomplete_rows / total_rows
        if total_rows
        else 1.0
    )

    latest_timestamp = row.get(
        "latest_timestamp_utc"
    )

    if latest_timestamp is not None:
        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.replace(
                tzinfo=timezone.utc
            )

        input_age_hours = (
            utc_now() - latest_timestamp
        ).total_seconds() / 3600
    else:
        input_age_hours = None

    metrics = [
        _build_metric(
            run_id=run_id,
            metric_name="training_input_rows",
            metric_scope=PRED_SPLIT,
            metric_value=float(total_rows),
            threshold_value=1.0,
            status=(
                "pass"
                if total_rows >= 1
                else "fail"
            ),
            details={
                "incomplete_rows": incomplete_rows,
            },
        ),
        _build_metric(
            run_id=run_id,
            metric_name="training_input_missing_fraction",
            metric_scope=PRED_SPLIT,
            metric_value=missing_fraction,
            threshold_value=MAX_MISSING_FRACTION,
            status=(
                "pass"
                if missing_fraction <= MAX_MISSING_FRACTION
                else "fail"
            ),
            details={
                "total_rows": total_rows,
                "incomplete_rows": incomplete_rows,
            },
        ),
        _build_metric(
            run_id=run_id,
            metric_name="training_input_age_hours",
            metric_scope=PRED_SPLIT,
            metric_value=input_age_hours,
            threshold_value=MAX_INPUT_AGE_HOURS,
            status=(
                "pass"
                if (
                    input_age_hours is not None
                    and input_age_hours <= MAX_INPUT_AGE_HOURS
                )
                else "fail"
            ),
            details={
                "latest_timestamp_utc": (
                    latest_timestamp.isoformat()
                    if latest_timestamp is not None
                    else None
                ),
            },
        ),
    ]

    return metrics


def check_predictions(
    run_id: str,
) -> list[dict[str, Any]]:
    query = f"""
        SELECT
            COUNT(*) AS prediction_rows,
            COUNT(DISTINCT station_name) AS station_count,
            COUNT(DISTINCT model_version) AS model_count,
            MAX(prediction_ready_utc)
                AS latest_prediction_timestamp
        FROM `{PREDICTION_TABLE}`
        WHERE run_id = @run_id
    """

    row = _query_one(
        query,
        [
            bigquery.ScalarQueryParameter(
                "run_id",
                "STRING",
                run_id,
            )
        ],
    )

    prediction_rows = int(
        row["prediction_rows"] or 0
    )

    station_count = int(
        row["station_count"] or 0
    )

    model_count = int(
        row["model_count"] or 0
    )

    return [
        _build_metric(
            run_id=run_id,
            metric_name="prediction_rows",
            metric_scope=PREDICTION_TABLE,
            metric_value=float(prediction_rows),
            threshold_value=float(
                MIN_PREDICTION_ROWS
            ),
            status=(
                "pass"
                if prediction_rows >= MIN_PREDICTION_ROWS
                else "fail"
            ),
            details={
                "station_count": station_count,
                "model_count": model_count,
            },
        ),
        _build_metric(
            run_id=run_id,
            metric_name="prediction_station_count",
            metric_scope=PREDICTION_TABLE,
            metric_value=float(station_count),
            threshold_value=1.0,
            status=(
                "pass"
                if station_count >= 1
                else "fail"
            ),
            details={
                "prediction_rows": prediction_rows,
            },
        ),
        _build_metric(
            run_id=run_id,
            metric_name="prediction_model_count",
            metric_scope=PREDICTION_TABLE,
            metric_value=float(model_count),
            threshold_value=1.0,
            status=(
                "pass"
                if model_count == 1
                else "fail"
            ),
            details={
                "prediction_rows": prediction_rows,
            },
        ),
    ]


def run_data_quality_checks(
    run_id: str | None = None,
) -> dict[str, Any]:
    effective_run_id = run_id or os.getenv(
        "MLOPS_RUN_ID",
        "",
    ).strip()

    if not effective_run_id:
        raise ValueError(
            "MLOPS_RUN_ID is required for "
            "data-quality checks."
        )

    metrics: list[dict[str, Any]] = []

    metrics.extend(
        check_training_input(effective_run_id)
    )

    metrics.extend(
        check_predictions(effective_run_id)
    )

    _write_metrics(metrics)

    failed_metrics = [
        metric["metric_name"]
        for metric in metrics
        if metric["status"] != "pass"
    ]

    status = (
        "fail"
        if failed_metrics
        else "pass"
    )

    logger.info(
        "data_quality_checks_completed",
        extra={
            "run_id": effective_run_id,
            "metric_count": len(metrics),
            "failed_metric_count": len(
                failed_metrics
            ),
            "data_quality_status": status,
            "failed_metrics": failed_metrics,
        },
    )

    return {
        "data_quality_status": status,
        "quality_metrics_written": len(metrics),
        "quality_failed_metrics": failed_metrics,
    }