from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ingestion.common.hashing import stable_record_hash
from ingestion.common.schemas import PegelMeasurement


def parse_archive_timestamp_to_utc(ts: str, timezone_name: str = "Europe/Berlin") -> str:
    dt_local = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M")
    localized = dt_local.replace(tzinfo=ZoneInfo(timezone_name))
    return localized.astimezone(ZoneInfo("UTC")).isoformat()


def normalize_archive_dataframe(
    df: pd.DataFrame,
    station_id: str,
    station_name: str,
    timeseries_name: str,
    unit: str,
    source_url: str,
    ingestion_ts_utc: str,
    timezone_name: str = "Europe/Berlin",
) -> pd.DataFrame:
    out = df.copy()

    colmap = {c.lower().strip(): c for c in out.columns}
    ts_col = colmap.get("timestamp") or colmap.get("date") or list(out.columns)[0]
    value_col = colmap.get("value") or list(out.columns)[1]

    out = out[[ts_col, value_col]].rename(columns={ts_col: "timestamp_raw", value_col: "value"})
    out["timestamp_utc"] = out["timestamp_raw"].astype(str).map(
        lambda x: parse_archive_timestamp_to_utc(x, timezone_name=timezone_name)
    )
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out["station_id"] = station_id
    out["station_name"] = station_name
    out["timeseries_name"] = timeseries_name
    out["unit"] = unit
    out["ingestion_ts_utc"] = ingestion_ts_utc
    out["source"] = "pegelonline"
    out["source_url"] = source_url

    records = []
    for row in out.to_dict(orient="records"):
        payload = {
            "station_id": row["station_id"],
            "station_name": row["station_name"],
            "timeseries_name": row["timeseries_name"],
            "timestamp_utc": row["timestamp_utc"],
            "value": row["value"],
            "unit": row["unit"],
            "latitude": None,
            "longitude": None,
            "ingestion_ts_utc": row["ingestion_ts_utc"],
            "source": row["source"],
            "source_url": row["source_url"],
        }
        payload["source_record_hash"] = stable_record_hash(
            payload,
            keys=["station_id", "timeseries_name", "timestamp_utc", "value", "source"],
        )
        records.append(PegelMeasurement(**payload).model_dump())

    return pd.DataFrame(records)