from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import bigquery

PROJECT_ID = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "rhine-corridor-navigator")).strip()
GCP_REGION = os.getenv("GCP_REGION", "europe-west3").strip()
CURATED_DATASET = os.getenv("CURATED_DATASET", "rhein_curated").strip()
MLOPS_DATASET = os.getenv("MLOPS_DATASET", "mlops").strip()
PREDICTIONS_TABLE = os.getenv("PREDICTIONS_TABLE", "gauge_24h_production_predictions").strip()
PREDICTIONS_HISTORY_TABLE = os.getenv("PREDICTIONS_HISTORY_TABLE", "gauge_24h_prediction_history").strip()
EVALUATIONS_TABLE = os.getenv("EVALUATIONS_TABLE", "gauge_24h_prediction_evaluations").strip()
STATIONS_TABLE = os.getenv("STATIONS_TABLE", "dim_station").strip()
SEGMENTS_CONFIG_PATH = Path(os.getenv("GAUGE24H_SEGMENTS_CONFIG_PATH", "config/segments.yaml").strip())
CACHE_TTL_SECONDS = int(os.getenv("API_CACHE_TTL_SECONDS", "300"))
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("API_ALLOWED_ORIGINS", "*").split(",") if x.strip()]

LATEST_PIPELINE_HEALTH_VIEW = os.getenv("LATEST_PIPELINE_HEALTH_VIEW", "v_latest_pipeline_health").strip()
LATEST_PIPELINE_HEALTH_BY_JOB_VIEW = os.getenv("LATEST_PIPELINE_HEALTH_BY_JOB_VIEW", "v_latest_pipeline_health_by_job").strip()
LATEST_QUALITY_HEALTH_VIEW = os.getenv("LATEST_QUALITY_HEALTH_VIEW", "v_latest_quality_health").strip()
LATEST_STAGE_HEALTH_VIEW = os.getenv("LATEST_STAGE_HEALTH_VIEW", "v_latest_stage_health").strip()
LATEST_RUN_QUALITY_SUMMARY_VIEW = os.getenv("LATEST_RUN_QUALITY_SUMMARY_VIEW", "v_latest_run_quality_summary").strip()
LATEST_EVALUATION_HEALTH_VIEW = os.getenv("LATEST_EVALUATION_HEALTH_VIEW", "v_latest_evaluation_health").strip()

app = FastAPI(title="Rhine Corridor Gauge 24h Prediction API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_bq_client: bigquery.Client | None = None
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()
_segments_cache: dict[str, Any] | None = None
_segments_cache_mtime: float | None = None


def get_bq_client() -> bigquery.Client:
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=PROJECT_ID, location=GCP_REGION)
    return _bq_client


def table_id(dataset: str, table: str) -> str:
    return f"{PROJECT_ID}.{dataset}.{table}"


def cache_get(key: str) -> Any | None:
    with _cache_lock:
        value = _cache.get(key)
        if value is None:
            return None
        expires_at, payload = value
        if expires_at <= time.monotonic():
            _cache.pop(key, None)
            return None
        return payload


def cache_set(key: str, value: Any, ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic() + ttl_seconds, value)


def normalize_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").map(
        lambda value: value.isoformat() if pd.notna(value) else None
    )


def clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    result = df.copy()
    for column in [
        "timestamp_utc", "forecast_timestamp_utc", "prediction_ready_utc",
        "prediction_timestamp_utc", "actual_available_utc", "evaluated_at_utc",
        "measured_at_utc", "started_at_utc", "ended_at_utc", "created_at_utc",
        "last_measured_at_utc",
    ]:
        if column in result.columns:
            result[column] = normalize_timestamp_series(result[column])
    result = result.where(pd.notna(result), None)
    return [
        {key: value.item() if hasattr(value, "item") else value for key, value in row.items()}
        for row in result.to_dict(orient="records")
    ]


def run_query(query: str, parameters: list[bigquery.ScalarQueryParameter] | None = None) -> pd.DataFrame:
    config = bigquery.QueryJobConfig(query_parameters=parameters) if parameters else None
    job = get_bq_client().query(query, job_config=config, location=GCP_REGION)
    return job.to_dataframe(create_bqstorage_client=False)


def normalize_station_name(name: str) -> str:
    return str(name).strip().upper().replace(" ", "-")


def load_segments_config() -> dict[str, Any]:
    global _segments_cache, _segments_cache_mtime
    if not SEGMENTS_CONFIG_PATH.exists():
        return {"segments": {}}
    mtime = SEGMENTS_CONFIG_PATH.stat().st_mtime
    if _segments_cache is not None and _segments_cache_mtime == mtime:
        return _segments_cache
    with SEGMENTS_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    raw = data.get("segments", {})
    normalized = {"segments": {}}
    for segment_id, segment in raw.items() if isinstance(raw, dict) else []:
        if isinstance(segment, dict):
            gauges = segment.get("support_gauges", []) or []
            normalized["segments"][str(segment_id)] = {
                "segment_id": str(segment_id),
                "label": str(segment.get("label", segment_id)),
                "decision_gauge": str(segment.get("decision_gauge", "")).strip(),
                "support_gauges": [str(g).strip() for g in gauges if str(g).strip()],
            }
    _segments_cache, _segments_cache_mtime = normalized, mtime
    return normalized


def build_station_segment_lookup() -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for segment_id, segment in load_segments_config().get("segments", {}).items():
        decision = normalize_station_name(segment.get("decision_gauge", ""))
        gauges = {normalize_station_name(g) for g in segment.get("support_gauges", []) if g}
        if decision:
            gauges.add(decision)
        for gauge in gauges:
            lookup.setdefault(gauge, []).append({
                "segment_id": segment_id,
                "segment_label": segment.get("label", segment_id),
                "decision_gauge": decision,
                "is_decision_gauge": gauge == decision,
            })
    return lookup


def attach_segment_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "station_name" not in df.columns:
        return df
    lookup = build_station_segment_lookup()
    result = df.copy()
    matches = [lookup.get(normalize_station_name(x), []) for x in result["station_name"].astype(str)]
    result["segment_ids"] = [[x["segment_id"] for x in items] for items in matches]
    result["segment_labels"] = [[x["segment_label"] for x in items] for items in matches]
    result["is_decision_gauge"] = [any(x["is_decision_gauge"] for x in items) for items in matches]
    return result


def latest_prediction_run_id() -> str:
    query = f"""
        SELECT run_id FROM `{table_id(CURATED_DATASET, PREDICTIONS_TABLE)}`
        WHERE split_name = 'production'
        ORDER BY prediction_ready_utc DESC LIMIT 1
    """
    df = run_query(query)
    if df.empty or pd.isna(df.iloc[0]["run_id"]):
        raise HTTPException(status_code=404, detail="No production prediction run found.")
    return str(df.iloc[0]["run_id"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "gauge24h-api"}


@app.get("/health/ready")
def readiness() -> dict[str, Any]:
    checks = {"bigquery": "unknown", "predictions": "unknown"}
    try:
        checks["bigquery"] = "ok" if not run_query("SELECT 1 AS ok").empty else "fail"
    except Exception as exc:
        checks["bigquery"] = f"fail: {exc}"
    try:
        latest_prediction_run_id()
        checks["predictions"] = "ok"
    except Exception as exc:
        checks["predictions"] = f"fail: {exc}"
    ready = all(str(value).startswith("ok") for value in checks.values())
    payload = {"status": "ready" if ready else "not_ready", "checks": checks}
    if not ready:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.get("/metadata/segments")
def get_segments_metadata() -> dict[str, Any]:
    key = "metadata:segments"
    cached = cache_get(key)
    if cached is not None:
        return cached
    segments = [
        {
            "segment_id": segment_id,
            "segment_label": segment.get("label", segment_id),
            "decision_gauge": segment.get("decision_gauge"),
            "support_gauges": segment.get("support_gauges", []),
        }
        for segment_id, segment in load_segments_config().get("segments", {}).items()
    ]
    payload = {"segments": segments, "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    cache_set(key, payload, 3600)
    return payload


@app.get("/predictions/latest", response_model=list[dict[str, Any]])
def get_latest_predictions() -> list[dict[str, Any]]:
    key = "predictions:latest"
    cached = cache_get(key)
    if cached is not None:
        return cached
    run_id = latest_prediction_run_id()
    query = f"""
        SELECT run_id, station_name, timeseries_name, unit, source,
               timestamp_utc, forecast_timestamp_utc, prediction_ready_utc,
               prediction_horizon_hours, prediction, actual_if_available,
               actual_available_now, model_version, split_name, target_mode,
               target_column
        FROM `{table_id(CURATED_DATASET, PREDICTIONS_TABLE)}`
        WHERE run_id = @run_id AND split_name = 'production'
        ORDER BY station_name
    """
    params = [bigquery.ScalarQueryParameter("run_id", "STRING", run_id)]
    try:
        records = clean_records(attach_segment_metadata(run_query(query, params)))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BigQuery error: {exc}") from exc
    cache_set(key, records)
    return records


@app.get("/predictions/history", response_model=list[dict[str, Any]])
def get_predictions_history(
    days: int = Query(default=7, ge=1, le=90),
    station: str | None = Query(default=None, min_length=1, max_length=100),
    segment: str | None = Query(default=None, min_length=1, max_length=100),
) -> list[dict[str, Any]]:
    key = f"predictions:history:{days}:{station or '*'}:{segment or '*'}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    query = f"""
        SELECT run_id, station_name, timeseries_name, unit, source,
               timestamp_utc, forecast_timestamp_utc, prediction_ready_utc,
               prediction_horizon_hours, prediction, actual_if_available,
               actual_available_now, model_version, split_name, target_mode,
               target_column
        FROM `{table_id(CURATED_DATASET, PREDICTIONS_HISTORY_TABLE)}`
        WHERE split_name = 'production'
          AND prediction_ready_utc >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
    """
    params = [bigquery.ScalarQueryParameter("days", "INT64", days)]
    if station:
        query += " AND LOWER(station_name) LIKE LOWER(@station_pattern)"
        params.append(bigquery.ScalarQueryParameter("station_pattern", "STRING", f"%{station}%"))
    query += " ORDER BY station_name, forecast_timestamp_utc"
    try:
        df = run_query(query, params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BigQuery error: {exc}") from exc
    if segment:
        allowed = set()
        for segment_id, item in load_segments_config().get("segments", {}).items():
            if str(segment_id).lower() == segment.lower() or str(item.get("label", "")).lower() == segment.lower():
                allowed.update(normalize_station_name(g) for g in [item.get("decision_gauge", "")] + item.get("support_gauges", []) if g)
        if allowed:
            df = df[df["station_name"].astype(str).map(normalize_station_name).isin(allowed)].copy()
    records = clean_records(attach_segment_metadata(df))
    cache_set(key, records)
    return records


@app.get("/metadata/stations", response_model=list[dict[str, Any]])
def get_station_metadata() -> list[dict[str, Any]]:
    key = "metadata:stations"
    cached = cache_get(key)
    if cached is not None:
        return cached
    query = f"""
        SELECT station_id, station_name, latitude, longitude
        FROM `{table_id(CURATED_DATASET, STATIONS_TABLE)}`
        WHERE station_name IS NOT NULL ORDER BY station_name
    """
    try:
        df = run_query(query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"BigQuery error: {exc}") from exc
    lookup = build_station_segment_lookup()
    normalized = df["station_name"].astype(str).map(normalize_station_name)
    df["segment_ids"] = normalized.map(lambda x: [m["segment_id"] for m in lookup.get(x, [])])
    df["segment_labels"] = normalized.map(lambda x: [m["segment_label"] for m in lookup.get(x, [])])
    df["is_decision_gauge"] = normalized.map(lambda x: any(m["is_decision_gauge"] for m in lookup.get(x, [])))
    records = clean_records(df)
    cache_set(key, records, 3600)
    return records


@app.get("/system/status")
def get_system_status() -> dict[str, Any]:
    key = "system:status"
    cached = cache_get(key)
    if cached is not None:
        return cached
    queries = {
        "pipeline": f"SELECT * FROM `{table_id(MLOPS_DATASET, LATEST_PIPELINE_HEALTH_BY_JOB_VIEW)}` ORDER BY job_type",
        "quality": f"SELECT * FROM `{table_id(MLOPS_DATASET, LATEST_QUALITY_HEALTH_VIEW)}` ORDER BY metric_name, metric_scope",
        "stage": f"SELECT * FROM `{table_id(MLOPS_DATASET, LATEST_STAGE_HEALTH_VIEW)}` ORDER BY stage_name",
        "quality_summary": f"SELECT * FROM `{table_id(MLOPS_DATASET, LATEST_RUN_QUALITY_SUMMARY_VIEW)}` ORDER BY last_measured_at_utc DESC LIMIT 1",
        "evaluation": f"SELECT * FROM `{table_id(MLOPS_DATASET, LATEST_EVALUATION_HEALTH_VIEW)}` ORDER BY evaluated_at_utc DESC LIMIT 1",
    }
    try:
        data = {name: clean_records(run_query(query)) for name, query in queries.items()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Monitoring views unavailable: {exc}") from exc
    pipeline = data["pipeline"]
    quality = data["quality"]
    stages = data["stage"]
    evaluation = data["evaluation"]
    good_pipeline = {"success", "succeeded", "ok", "pass"}
    good_quality = {"pass", "passed", "success", "ok"}
    pipeline_status = "pass" if pipeline and all(str(x.get("status", "")).lower() in good_pipeline for x in pipeline) else "unknown"
    quality_status = "pass" if quality and all(str(x.get("status", "")).lower() in good_quality for x in quality) else "unknown"
    stage_status = "pass" if stages and all(str(x.get("status", "")).lower() in good_pipeline for x in stages) else "unknown"
    evaluation_status = evaluation[0].get("evaluation_status", "unknown") if evaluation else "unknown"
    if any(str(x).lower() in {"fail", "failed", "error"} for x in [pipeline_status, quality_status, stage_status, evaluation_status]):
        overall = "degraded"
    elif not pipeline or not quality or not stages:
        overall = "degraded"
    else:
        overall = "ok"
    payload = {
        "status": overall,
        "project_id": PROJECT_ID,
        "region": GCP_REGION,
        "prediction_table": f"{CURATED_DATASET}.{PREDICTIONS_TABLE}",
        "prediction_history_table": f"{CURATED_DATASET}.{PREDICTIONS_HISTORY_TABLE}",
        "pipeline_status": pipeline_status,
        "data_quality_status": quality_status,
        "stage_status": stage_status,
        "evaluation_status": evaluation_status,
        "latest_pipeline_health": pipeline,
        "latest_quality_health": quality,
        "latest_stage_health": stages,
        "latest_quality_summary": data["quality_summary"][0] if data["quality_summary"] else {},
        "latest_evaluation": evaluation[0] if evaluation else {},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    cache_set(key, payload, 60)
    return payload


@app.get("/evaluations/history", response_model=list[dict[str, Any]])
def get_evaluation_history(days: int = Query(default=30, ge=1, le=365), station: str | None = Query(default=None, min_length=1, max_length=100)) -> list[dict[str, Any]]:
    key = f"evaluations:history:{days}:{station or '*'}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    query = f"""
        SELECT * FROM `{table_id(CURATED_DATASET, EVALUATIONS_TABLE)}`
        WHERE forecast_timestamp_utc >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
    """
    params = [bigquery.ScalarQueryParameter("days", "INT64", days)]
    if station:
        query += " AND LOWER(station_name) LIKE LOWER(@station_pattern)"
        params.append(bigquery.ScalarQueryParameter("station_pattern", "STRING", f"%{station}%"))
    query += " ORDER BY station_name, forecast_timestamp_utc"
    try:
        records = clean_records(run_query(query, params))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation table unavailable: {exc}") from exc
    cache_set(key, records)
    return records


@app.get("/evaluations/metrics")
def get_evaluation_metrics(days: int = Query(default=30, ge=1, le=365)) -> list[dict[str, Any]]:
    key = f"evaluations:metrics:{days}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    query = f"""
        SELECT station_name, COUNT(*) AS evaluated_predictions,
               AVG(absolute_error) AS mae,
               SQRT(AVG(squared_error)) AS rmse,
               AVG(error) AS bias,
               MAX(evaluated_at_utc) AS latest_evaluation_utc
        FROM `{table_id(CURATED_DATASET, EVALUATIONS_TABLE)}`
        WHERE forecast_timestamp_utc >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
        GROUP BY station_name ORDER BY mae ASC
    """
    try:
        records = clean_records(run_query(query, [bigquery.ScalarQueryParameter("days", "INT64", days)]))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation table unavailable: {exc}") from exc
    cache_set(key, records)
    return records