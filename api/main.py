from fastapi import FastAPI, HTTPException
from google.cloud import bigquery
from typing import List, Dict, Any
import pandas as pd

app = FastAPI(title="Gauge24h Prediction API")

PROJECT_ID = "rhine-corridor-navigator"
DATASET = "rhein_curated"
PREDICTIONS_TABLE = "gauge_24h_production_predictions_test"

bq_client = bigquery.Client(project=PROJECT_ID)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predictions/latest", response_model=List[Dict[str, Any]])
def get_latest_predictions():
    query = f"""
        SELECT
          station_name,
          timestamp_utc,
          forecast_timestamp_utc,
          prediction,
          model_version,
          split_name
        FROM `{PROJECT_ID}.{DATASET}.{PREDICTIONS_TABLE}`
        WHERE forecast_timestamp_utc = (
          SELECT MAX(forecast_timestamp_utc)
          FROM `{PROJECT_ID}.{DATASET}.{PREDICTIONS_TABLE}`
        )
        ORDER BY station_name
    """
    try:
        df = bq_client.query(query).to_dataframe()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery error: {e}")

    # Convert timestamps to ISO strings for JSON
    for col in ["timestamp_utc", "forecast_timestamp_utc"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df.to_dict(orient="records")

@app.get("/predictions/history", response_model=List[Dict[str, Any]])
def get_predictions_history(
    days: int = 7,
    station: str | None = None,
):
    """
    Return recent predictions per station for the last `days` days.
    Optionally filter to a single station (case-insensitive substring match).
    """
    base_query = f"""
        SELECT
          station_name,
          timestamp_utc,
          forecast_timestamp_utc,
          prediction,
          model_version,
          split_name
        FROM `{PROJECT_ID}.{DATASET}.{PREDICTIONS_TABLE}`
        WHERE forecast_timestamp_utc >= TIMESTAMP_SUB(
            (SELECT MAX(forecast_timestamp_utc) FROM `{PROJECT_ID}.{DATASET}.{PREDICTIONS_TABLE}`),
            INTERVAL {days} DAY
        )
    """

    if station:
        base_query += f"""
          AND LOWER(station_name) LIKE LOWER('%{station.replace("'", "''")}%')
        """

    base_query += " ORDER BY station_name, forecast_timestamp_utc"

    try:
        df = bq_client.query(base_query).to_dataframe()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BigQuery error: {e}")

    for col in ["timestamp_utc", "forecast_timestamp_utc"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df.to_dict(orient="records")