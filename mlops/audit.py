from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import uuid4

from google.cloud import bigquery

from mlops.run_context import (
    PipelineRun,
    StageContext,
    isoformat_utc,
)

logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv(
    "GCP_PROJECT_ID",
    "rhine-corridor-navigator",
).strip()

REGION = os.getenv(
    "GCP_REGION",
    "europe-west3",
).strip()

PIPELINE_RUNS_TABLE = f"{PROJECT_ID}.mlops.pipeline_runs"
STAGE_EVENTS_TABLE = f"{PROJECT_ID}.mlops.stage_events"


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


def _first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _write_rows(table_id: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    try:
        job = _client().load_table_from_json(
            rows,
            table_id,
            job_config=bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND
            ),
        )
        job.result()
    except Exception:
        logger.exception(
            "audit_write_failed",
            extra={
                "table_id": table_id,
                "row_count": len(rows),
            },
        )


def record_pipeline_run(run: PipelineRun) -> None:
    record = run.as_record()

    row = {
        "run_id": record["run_id"],
        "job_type": record["job_type"],
        "cloud_run_job": record.get("cloud_run_job"),
        "project_id": record.get("project_id"),
        "region": record.get("region"),
        "started_at_utc": record["started_at_utc"],
        "ended_at_utc": record["ended_at_utc"],
        "duration_seconds": record["duration_seconds"],
        "status": record["status"],
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
        "input_split": record.get("input_split"),
        "model_version": record.get("model_version"),
        "rows_ingested": record.get("rows_ingested"),
        "rows_predicted": record.get("rows_predicted"),
        "stations_processed": record.get("stations_processed"),
        "data_window_start_utc": record.get("data_window_start_utc"),
        "data_window_end_utc": record.get("data_window_end_utc"),
        "created_at_utc": isoformat_utc(run.ended_at) or isoformat_utc(run.started_at),
    }

    _write_rows(PIPELINE_RUNS_TABLE, [row])


def record_stage_event(run: PipelineRun, stage: StageContext) -> None:
    record = stage.as_record()

    row = {
        "event_id": uuid4().hex,
        "run_id": run.run_id,
        "job_type": run.job_type,
        "stage_name": record["stage_name"],
        "started_at_utc": record["started_at_utc"],
        "ended_at_utc": record["ended_at_utc"],
        "duration_seconds": record["duration_seconds"],
        "status": record["status"],
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
        "rows_read": _first_present(record, "rows_read", "rows_ingested", "rows_predicted"),
        "rows_written": _first_present(record, "rows_written", "rows_ingested", "rows_predicted"),
        "station_count": _first_present(record, "station_count", "stations_processed", "stations_predicted"),
        "table_name": record.get("table_name"),
        "metadata_json": _safe_json(stage.metadata),
        "created_at_utc": isoformat_utc(stage.ended_at) or isoformat_utc(stage.started_at),
    }

    _write_rows(STAGE_EVENTS_TABLE, [row])