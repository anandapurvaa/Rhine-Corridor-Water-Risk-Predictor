from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from google.cloud import bigquery

PROJECT_ID = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "rhine-corridor-navigator")).strip()
REGION = os.getenv("GCP_REGION", "europe-west3").strip()
CURATED_DATASET = os.getenv("CURATED_DATASET", "rhein_curated").strip()
MLOPS_DATASET = os.getenv("MLOPS_DATASET", "mlops").strip()
PREDICTIONS_TABLE = os.getenv("PREDICTIONS_TABLE", "gauge_24h_production_predictions").strip()
EVALUATIONS_TABLE = os.getenv("EVALUATIONS_TABLE", "gauge_24h_prediction_evaluations").strip()
QUALITY_TABLE = os.getenv("QUALITY_TABLE", "data_quality_metrics").strip()
TRAINING_JOB = os.getenv("TRAINING_JOB_NAME", "gauge24h-train").strip()
DAILY_JOB = os.getenv("DAILY_JOB_NAME", "rhine-daily-pipeline").strip()
EVALUATION_JOB = os.getenv("EVALUATION_JOB_NAME", "rhine-gauge-24h-evaluation").strip()
EXPECTED_STATIONS = int(os.getenv("EXPECTED_STATION_COUNT", "19"))
MIN_PREDICTION_ROWS = int(os.getenv("MIN_PREDICTION_ROWS", "19"))
MAX_PREDICTION_AGE_HOURS = float(os.getenv("MAX_PREDICTION_AGE_HOURS", "26"))
MAX_EVALUATION_AGE_HOURS = float(os.getenv("MAX_EVALUATION_AGE_HOURS", "50"))
MAX_MAE = float(os.getenv("MAX_MAE", "999999"))
MAX_RMSE = float(os.getenv("MAX_RMSE", "999999"))
FAIL_ON_MISSING_EVALUATION = os.getenv("FAIL_ON_MISSING_EVALUATION", "false").lower() == "true"

logging.basicConfig(stream=sys.stdout, level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
logger = logging.getLogger("gauge24h-watchdog")
client = bigquery.Client(project=PROJECT_ID, location=REGION)


def emit(event: str, status: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "status": status,
        "service": "gauge24h-watchdog",
        "project_id": PROJECT_ID,
        "region": REGION,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    logger.info(json.dumps(payload, default=str, separators=(",", ":")))


def query(sql: str) -> list[dict[str, Any]]:
    return [dict(row.items()) for row in client.query(sql, location=REGION).result()]


def parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "nat", "null"}:
            return None
        normalized = text
        if normalized.endswith(" UTC"):
            normalized = normalized[:-4].strip() + "+00:00"
        elif normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            formats = (
                "%Y-%m-%d %H:%M:%S UTC",
                "%Y-%m-%d %H:%M:%S.%f UTC",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
            )
            for fmt in formats:
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Unsupported timestamp format: {text!r}")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def age_hours(value: Any) -> float | None:
    parsed = parse_utc(value)
    if parsed is None:
        return None
    return (datetime.now(timezone.utc) - parsed).total_seconds() / 3600


def check_predictions() -> bool:
    sql = f"""
    SELECT COUNT(*) AS prediction_rows,
           COUNT(DISTINCT station_name) AS station_count,
           MAX(prediction_ready_utc) AS latest_ready_utc,
           MAX(forecast_timestamp_utc) AS latest_forecast_utc,
           ANY_VALUE(run_id) AS run_id,
           ANY_VALUE(model_version) AS model_version
    FROM `{PROJECT_ID}.{CURATED_DATASET}.{PREDICTIONS_TABLE}`
    WHERE split_name = 'production'
      AND run_id = (
        SELECT run_id FROM `{PROJECT_ID}.{CURATED_DATASET}.{PREDICTIONS_TABLE}`
        WHERE split_name = 'production'
        ORDER BY prediction_ready_utc DESC LIMIT 1
      )
    """
    row = (query(sql) or [{}])[0]
    prediction_age = age_hours(row.get("latest_ready_utc"))
    checks = {
        "rows_ok": int(row.get("prediction_rows") or 0) >= MIN_PREDICTION_ROWS,
        "stations_ok": int(row.get("station_count") or 0) >= EXPECTED_STATIONS,
        "freshness_ok": prediction_age is not None and prediction_age <= MAX_PREDICTION_AGE_HOURS,
    }
    ok = all(checks.values())
    emit("prediction_health", "pass" if ok else "fail", checks=checks,
         prediction_rows=row.get("prediction_rows"), station_count=row.get("station_count"),
         age_hours=prediction_age, run_id=row.get("run_id"), model_version=row.get("model_version"))
    return ok


def check_quality() -> bool:
    sql = f"""
    SELECT metric_name, metric_value, threshold_value, status, measured_at_utc
    FROM `{PROJECT_ID}.{MLOPS_DATASET}.{QUALITY_TABLE}`
    WHERE run_id = (
      SELECT run_id FROM `{PROJECT_ID}.{MLOPS_DATASET}.{QUALITY_TABLE}`
      ORDER BY measured_at_utc DESC LIMIT 1
    )
    ORDER BY metric_name
    """
    rows = query(sql)
    if not rows:
        emit("data_quality_health", "fail", reason="no_quality_metrics")
        return False
    ok = all(str(row.get("status", "")).lower() in {"pass", "passed", "ok", "success"} for row in rows)
    emit("data_quality_health", "pass" if ok else "fail", metric_count=len(rows), metrics=rows)
    return ok


def check_evaluations() -> bool:
    sql = f"""
    SELECT COUNT(*) AS evaluation_rows,
           AVG(absolute_error) AS mae,
           SQRT(AVG(squared_error)) AS rmse,
           MAX(evaluated_at_utc) AS latest_evaluated_utc
    FROM `{PROJECT_ID}.{CURATED_DATASET}.{EVALUATIONS_TABLE}`
    WHERE forecast_timestamp_utc >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    """
    row = (query(sql) or [{}])[0]
    count = int(row.get("evaluation_rows") or 0)
    mae = float(row.get("mae")) if row.get("mae") is not None else None
    rmse = float(row.get("rmse")) if row.get("rmse") is not None else None
    evaluation_age = age_hours(row.get("latest_evaluated_utc"))
    checks = {
        "rows_ok": count > 0 or not FAIL_ON_MISSING_EVALUATION,
        "freshness_ok": evaluation_age is None or evaluation_age <= MAX_EVALUATION_AGE_HOURS,
        "mae_ok": mae is None or mae <= MAX_MAE,
        "rmse_ok": rmse is None or rmse <= MAX_RMSE,
    }
    ok = all(checks.values())
    emit("evaluation_health", "pass" if ok else "fail", checks=checks,
         evaluation_rows=count, mae=mae, rmse=rmse, age_hours=evaluation_age)
    return ok


def main() -> int:
    emit("watchdog_started", "ok", jobs={"daily": DAILY_JOB, "evaluation": EVALUATION_JOB, "training": TRAINING_JOB})
    try:
        results = [check_predictions(), check_quality(), check_evaluations()]
    except Exception as exc:
        emit("watchdog_failed", "fail", error=repr(exc))
        return 1
    ok = all(results)
    emit("watchdog_completed", "pass" if ok else "fail", checks_passed=sum(results), checks_total=len(results))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())