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

PROJECT_ID = os.getenv(
    "GCP_PROJECT_ID",
    os.getenv("GOOGLE_CLOUD_PROJECT", "rhine-corridor-navigator"),
).strip()
GCP_REGION = os.getenv("GCP_REGION", "europe-west3").strip()
CURATED_DATASET = os.getenv("CURATED_DATASET", "rhein_curated").strip()
MLOPS_DATASET = os.getenv("MLOPS_DATASET", "mlops").strip()
PREDICTIONS_TABLE = os.getenv(
    "PREDICTIONS_TABLE", "gauge_24h_production_predictions"
).strip()
PREDICTIONS_HISTORY_TABLE = os.getenv(
    "PREDICTIONS_HISTORY_TABLE", "gauge_24h_prediction_history"
).strip()
EVALUATIONS_TABLE = os.getenv(
    "EVALUATIONS_TABLE", "gauge_24h_prediction_evaluations"
).strip()
STATIONS_TABLE = os.getenv("STATIONS_TABLE", "dim_station").strip()
SEGMENTS_CONFIG_PATH = Path(
    os.getenv("GAUGE24H_SEGMENTS_CONFIG_PATH", "config/segments.yaml").strip()
)
CACHE_TTL_SECONDS = int(os.getenv("API_CACHE_TTL_SECONDS", "300"))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("API_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app = FastAPI(
    title="Rhine Corridor Gauge 24h Prediction API",
    version="1.1.0",
)
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
        _bq_client = bigquery.Client(
            project=PROJECT_ID,
            location=GCP_REGION,
        )
    return _bq_client


def table_id(dataset: str, table: str) -> str:
    return f"{PROJECT_ID}.{dataset}.{table}"


def cache_get(key: str) -> Any | None:
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached is None:
            return None
        expires_at, value = cached
        if expires_at <= now:
            _cache.pop(key, None)
            return None
        return value


def cache_set(
    key: str,
    value: Any,
    ttl_seconds: int = CACHE_TTL_SECONDS,
) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic() + ttl_seconds, value)


def normalize_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    ).map(
        lambda value: value.isoformat() if pd.notna(value) else None
    )


def clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []

    result = df.copy()
    timestamp_columns = [
        "timestamp_utc",
        "forecast_timestamp_utc",
        "prediction_ready_utc",
        "prediction_timestamp_utc",
        "actual_available_utc",
        "evaluated_at_utc",
        "measured_at_utc",
    ]
    for column in timestamp_columns:
        if column in result.columns:
            result[column] = normalize_timestamp_series(result[column])

    result = result.where(pd.notna(result), None)
    cleaned: list[dict[str, Any]] = []
    for record in result.to_dict(orient="records"):
        cleaned.append(
            {
                key: value.item() if hasattr(value, "item") else value
                for key, value in record.items()
            }
        )
    return cleaned


def run_query(
    query: str,
    parameters: list[bigquery.ScalarQueryParameter] | None = None,
) -> pd.DataFrame:
    job_config = None
    if parameters:
        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
    job = get_bq_client().query(
        query,
        job_config=job_config,
        location=GCP_REGION,
    )
    return job.to_dataframe(create_bqstorage_client=False)


def latest_prediction_run_id() -> str:
    query = f"""
        SELECT run_id
        FROM `{table_id(CURATED_DATASET, PREDICTIONS_TABLE)}`
        WHERE split_name = 'production'
        ORDER BY prediction_ready_utc DESC
        LIMIT 1
    """
    df = run_query(query)
    if df.empty or pd.isna(df.iloc[0]["run_id"]):
        raise HTTPException(
            status_code=404,
            detail="No production prediction run found.",
        )
    return str(df.iloc[0]["run_id"])


def load_segments_config() -> dict[str, Any]:
    global _segments_cache, _segments_cache_mtime
    if not SEGMENTS_CONFIG_PATH.exists():
        return {"segments": {}}

    mtime = SEGMENTS_CONFIG_PATH.stat().st_mtime
    if _segments_cache is not None and _segments_cache_mtime == mtime:
        return _segments_cache

    with SEGMENTS_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    segments = data.get("segments", {})
    if not isinstance(segments, dict):
        segments = {}

    normalized: dict[str, Any] = {"segments": {}}
    for segment_id, segment in segments.items():
        if not isinstance(segment, dict):
            continue
        support_gauges = segment.get("support_gauges", []) or []
        normalized["segments"][str(segment_id)] = {
            "segment_id": str(segment_id),
            "label": str(segment.get("label", segment_id)),
            "decision_gauge": str(segment.get("decision_gauge", "")).strip(),
            "support_gauges": [str(g).strip() for g in support_gauges if str(g).strip()],
        }

    _segments_cache = normalized
    _segments_cache_mtime = mtime
    return normalized


def normalize_station_name(name: str) -> str:
    return str(name).strip().upper().replace(" ", "-")


def build_station_segment_lookup() -> dict[str, list[dict[str, Any]]]:
    segments = load_segments_config().get("segments", {})
    lookup: dict[str, list[dict[str, Any]]] = {}
    for segment_id, segment in segments.items():
        decision_gauge = normalize_station_name(segment.get("decision_gauge", ""))
        label = segment.get("label", segment_id)
        gauges = set(
            normalize_station_name(g)
            for g in segment.get("support_gauges", [])
            if g
        )
        if decision_gauge:
            gauges.add(decision_gauge)
        for gauge in gauges:
            lookup.setdefault(gauge, []).append(
                {
                    "segment_id": segment_id,
                    "segment_label": label,
                    "decision_gauge": decision_gauge,
                    "is_decision_gauge": gauge == decision_gauge,
                }
            )
    return lookup


def attach_segment_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "station_name" not in df.columns:
        return df

    lookup = build_station_segment_lookup()
    result = df.copy()
    segment_ids = []
    segment_labels = []
    is_decision_gauge = []

    for station in result["station_name"].astype(str).map(normalize_station_name):
        matches = lookup.get(station, [])
        if not matches:
            segment_ids.append([])
            segment_labels.append([])
            is_decision_gauge.append(False)
            continue
        segment_ids.append([m["segment_id"] for m in matches])
        segment_labels.append([m["segment_label"] for m in matches])
        is_decision_gauge.append(any(m["is_decision_gauge"] for m in matches))

    result["segment_ids"] = segment_ids
    result["segment_labels"] = segment_labels
    result["is_decision_gauge"] = is_decision_gauge
    return result


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "gauge24h-api",
    }


@app.get("/health/ready")
def readiness() -> dict[str, Any]:
    checks: dict[str, Any] = {
        "bigquery": "unknown",
        "predictions": "unknown",
    }
    try:
        result = run_query("SELECT 1 AS ok")
        checks["bigquery"] = "ok" if not result.empty else "fail"
    except Exception as exc:
        checks["bigquery"] = f"fail: {exc}"

    try:
        latest_prediction_run_id()
        checks["predictions"] = "ok"
    except Exception as exc:
        checks["predictions"] = f"fail: {exc}"

    ready = all(str(value).startswith("ok") for value in checks.values())
    payload = {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
    }
    if not ready:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.get("/metadata/segments")
def get_segments_metadata() -> dict[str, Any]:
    cache_key = "metadata:segments"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    config = load_segments_config()
    segments = []
    for segment_id, segment in config.get("segments", {}).items():
        segments.append(
            {
                "segment_id": segment_id,
                "segment_label": segment.get("label", segment_id),
                "decision_gauge": segment.get("decision_gauge"),
                "support_gauges": segment.get("support_gauges", []),
            }
        )

    payload = {
        "segments": segments,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    cache_set(cache_key, payload, ttl_seconds=3600)
    return payload


@app.get("/predictions/latest", response_model=list[dict[str, Any]])
def get_latest_predictions() -> list[dict[str, Any]]:
    cache_key = "predictions:latest"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    run_id = latest_prediction_run_id()
    query = f"""
        SELECT
            run_id,
            station_name,
            timeseries_name,
            unit,
            source,
            timestamp_utc,
            forecast_timestamp_utc,
            prediction_ready_utc,
            prediction_horizon_hours,
            prediction,
            actual_if_available,
            actual_available_now,
            model_version,
            split_name,
            target_mode,
            target_column
        FROM `{table_id(CURATED_DATASET, PREDICTIONS_TABLE)}`
        WHERE run_id = @run_id
          AND split_name = 'production'
        ORDER BY station_name
    """
    parameters = [
        bigquery.ScalarQueryParameter("run_id", "STRING", run_id)
    ]
    try:
        df = run_query(query, parameters)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"BigQuery error: {exc}",
        ) from exc

    df = attach_segment_metadata(df)
    records = clean_records(df)
    cache_set(cache_key, records)
    return records


@app.get("/predictions/history", response_model=list[dict[str, Any]])
def get_predictions_history(
    days: int = Query(default=7, ge=1, le=90),
    station: str | None = Query(
        default=None,
        min_length=1,
        max_length=100,
    ),
    segment: str | None = Query(
        default=None,
        min_length=1,
        max_length=100,
    ),
) -> list[dict[str, Any]]:
    cache_key = f"predictions:history:{days}:{station or '*'}:{segment or '*'}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    query = f"""
        SELECT
            run_id,
            station_name,
            timeseries_name,
            unit,
            source,
            timestamp_utc,
            forecast_timestamp_utc,
            prediction_ready_utc,
            prediction_horizon_hours,
            prediction,
            actual_if_available,
            actual_available_now,
            model_version,
            split_name,
            target_mode,
            target_column
        FROM `{table_id(CURATED_DATASET, PREDICTIONS_HISTORY_TABLE)}`
        WHERE split_name = 'production'
          AND prediction_ready_utc >= TIMESTAMP_SUB(
              CURRENT_TIMESTAMP(), INTERVAL @days DAY
          )
    """
    parameters = [
        bigquery.ScalarQueryParameter("days", "INT64", days)
    ]

    if station:
        query += """
            AND LOWER(station_name) LIKE LOWER(@station_pattern)
        """
        parameters.append(
            bigquery.ScalarQueryParameter(
                "station_pattern",
                "STRING",
                f"%{station}%",
            )
        )

    query += """
        ORDER BY station_name, forecast_timestamp_utc
    """

    try:
        df = run_query(query, parameters)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"BigQuery error: {exc}",
        ) from exc

    if segment:
        lookup = build_station_segment_lookup()
        allowed = set()
        for segment_id, seg in load_segments_config().get("segments", {}).items():
            if str(segment_id).lower() == segment.lower() or str(seg.get("label", "")).lower() == segment.lower():
                allowed.update(
                    normalize_station_name(g)
                    for g in ([seg.get("decision_gauge", "")] + seg.get("support_gauges", []))
                    if g
                )
        if allowed:
            df = df[df["station_name"].astype(str).map(normalize_station_name).isin(allowed)].copy()

    df = attach_segment_metadata(df)
    records = clean_records(df)
    cache_set(cache_key, records)
    return records


@app.get("/metadata/stations", response_model=list[dict[str, Any]])
def get_station_metadata() -> list[dict[str, Any]]:
    cache_key = "metadata:stations"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    query = f"""
        SELECT station_id, station_name, latitude, longitude
        FROM `{table_id(CURATED_DATASET, STATIONS_TABLE)}`
        WHERE station_name IS NOT NULL
        ORDER BY station_name
    """
    try:
        df = run_query(query)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"BigQuery error: {exc}",
        ) from exc

    df = df.copy()
    df["station_name_normalized"] = df["station_name"].astype(str).map(normalize_station_name)
    lookup = build_station_segment_lookup()
    df["segment_ids"] = df["station_name_normalized"].map(
        lambda s: [m["segment_id"] for m in lookup.get(s, [])]
    )
    df["segment_labels"] = df["station_name_normalized"].map(
        lambda s: [m["segment_label"] for m in lookup.get(s, [])]
    )
    df["is_decision_gauge"] = df["station_name_normalized"].map(
        lambda s: any(m["is_decision_gauge"] for m in lookup.get(s, []))
    )
    df = df.drop(columns=["station_name_normalized"])

    records = clean_records(df)
    cache_set(cache_key, records, ttl_seconds=3600)
    return records


@app.get("/system/status")
def get_system_status() -> dict[str, Any]:
    cache_key = "system:status"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    prediction_query = f"""
        SELECT
            COUNT(*) AS prediction_rows,
            COUNT(DISTINCT station_name) AS station_count,
            MAX(prediction_ready_utc) AS latest_prediction_ready_utc,
            MAX(forecast_timestamp_utc) AS latest_forecast_timestamp_utc,
            ANY_VALUE(model_version) AS model_version,
            ANY_VALUE(run_id) AS run_id
        FROM `{table_id(CURATED_DATASET, PREDICTIONS_TABLE)}`
        WHERE split_name = 'production'
          AND run_id = (
              SELECT run_id
              FROM `{table_id(CURATED_DATASET, PREDICTIONS_TABLE)}`
              WHERE split_name = 'production'
              ORDER BY prediction_ready_utc DESC
              LIMIT 1
          )
    """

    quality_query = f"""
        SELECT
            metric_name,
            metric_value,
            threshold_value,
            status,
            measured_at_utc
        FROM `{table_id(MLOPS_DATASET, 'data_quality_metrics')}`
        WHERE run_id = (
            SELECT run_id
            FROM `{table_id(MLOPS_DATASET, 'data_quality_metrics')}`
            ORDER BY measured_at_utc DESC
            LIMIT 1
        )
        ORDER BY metric_name
    """

    try:
        prediction_df = run_query(prediction_query)
        quality_df = run_query(quality_query)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"BigQuery error: {exc}",
        ) from exc

    prediction_records = clean_records(prediction_df)
    quality_records = clean_records(quality_df)
    prediction_status = prediction_records[0] if prediction_records else {}

    quality_status = "unknown"
    if quality_records:
        quality_status = (
            "pass"
            if all(record.get("status") == "pass" for record in quality_records)
            else "fail"
        )

    payload = {
        "status": "ok" if prediction_status else "degraded",
        "project_id": PROJECT_ID,
        "region": GCP_REGION,
        "prediction_table": f"{CURATED_DATASET}.{PREDICTIONS_TABLE}",
        "prediction_history_table": (
            f"{CURATED_DATASET}.{PREDICTIONS_HISTORY_TABLE}"
        ),
        "prediction_status": prediction_status,
        "data_quality_status": quality_status,
        "quality_metrics": quality_records,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    cache_set(cache_key, payload, ttl_seconds=60)
    return payload


@app.get("/evaluations/history", response_model=list[dict[str, Any]])
def get_evaluation_history(
    days: int = Query(default=30, ge=1, le=365),
    station: str | None = Query(
        default=None,
        min_length=1,
        max_length=100,
    ),
) -> list[dict[str, Any]]:
    cache_key = f"evaluations:history:{days}:{station or '*'}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    query = f"""
        SELECT *
        FROM `{table_id(CURATED_DATASET, EVALUATIONS_TABLE)}`
        WHERE forecast_timestamp_utc >= TIMESTAMP_SUB(
            CURRENT_TIMESTAMP(), INTERVAL @days DAY
        )
    """
    parameters = [
        bigquery.ScalarQueryParameter("days", "INT64", days)
    ]

    if station:
        query += """
            AND LOWER(station_name) LIKE LOWER(@station_pattern)
        """
        parameters.append(
            bigquery.ScalarQueryParameter(
                "station_pattern",
                "STRING",
                f"%{station}%",
            )
        )

    query += " ORDER BY station_name, forecast_timestamp_utc"
    try:
        df = run_query(query, parameters)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation table unavailable: {exc}",
        ) from exc

    records = clean_records(df)
    cache_set(cache_key, records)
    return records


@app.get("/evaluations/metrics")
def get_evaluation_metrics(
    days: int = Query(default=30, ge=1, le=365),
) -> list[dict[str, Any]]:
    cache_key = f"evaluations:metrics:{days}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    query = f"""
        SELECT
            station_name,
            COUNT(*) AS evaluated_predictions,
            AVG(absolute_error) AS mae,
            SQRT(AVG(squared_error)) AS rmse,
            AVG(error) AS bias,
            MAX(evaluated_at_utc) AS latest_evaluation_utc
        FROM `{table_id(CURATED_DATASET, EVALUATIONS_TABLE)}`
        WHERE forecast_timestamp_utc >= TIMESTAMP_SUB(
            CURRENT_TIMESTAMP(), INTERVAL @days DAY
        )
        GROUP BY station_name
        ORDER BY mae ASC
    """
    parameters = [
        bigquery.ScalarQueryParameter("days", "INT64", days)
    ]
    try:
        df = run_query(query, parameters)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation table unavailable: {exc}",
        ) from exc

    records = clean_records(df)
    cache_set(cache_key, records)
    return records