from datetime import datetime, timezone
from urllib.parse import quote

import pandas as pd

from ingestion.common.bigquery_dedup import merge_dataframe_to_bigquery
from ingestion.common.config import settings
from ingestion.common.hashing import stable_record_hash
from ingestion.common.http import get_json
from ingestion.common.logging_utils import get_logger
from ingestion.common.schemas import PegelMeasurement
from ingestion.common.station_mapping import load_segment_config, map_station_to_segment
from ingestion.common.storage import write_local_parquet, write_jsonl
from ingestion.common.utils import load_yaml
from ingestion.common.watermark import get_watermark, set_watermark

logger = get_logger(__name__)


def load_rhine_config() -> dict:
    return load_yaml("config/rhine_gauges.yaml")


def normalize_station_name(name: str) -> str:
    return (name or "").strip().upper()


def is_rhine_station(station: dict, rhine_cfg: dict) -> bool:
    name = normalize_station_name(station.get("shortname") or station.get("longname") or "")
    return name in {normalize_station_name(x) for x in rhine_cfg.get("rhine_priority_gauges", [])}


def build_station_identifier(station: dict) -> str:
    return str(
        station.get("uuid")
        or station.get("shortname")
        or station.get("number")
    )


def to_api_iso8601(value: str | None) -> str | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_measurements_url(
    station_identifier: str,
    timeseries_shortname: str,
    start: str | None = None,
) -> str:
    station_id = quote(str(station_identifier), safe="")
    ts_name = quote(str(timeseries_shortname), safe="")
    base = f"{settings.pegelonline_base_url}/stations/{station_id}/{ts_name}/measurements.json"
    return f"{base}?start={quote(start, safe='')}" if start else base


def attach_record_hash(row: dict) -> dict:
    row["source_record_hash"] = stable_record_hash(
        row,
        keys=["station_id", "timeseries_name", "timestamp_utc", "value", "source"],
    )
    return row


def extract_row_from_measurement(
    station: dict,
    ts: dict,
    measurement: dict,
    ingestion_ts: str,
    source_url: str,
    segment_cfg: dict,
) -> dict | None:
    timestamp_utc = measurement.get("timestamp")
    value = measurement.get("value")

    if timestamp_utc is None:
        return None

    station_name = station.get("shortname") or station.get("longname") or "unknown"
    segment_id = map_station_to_segment(station_name, segment_cfg)

    row = {
        "station_id": str(station.get("uuid", "")),
        "station_name": station_name,
        "timeseries_name": ts.get("shortname", "unknown"),
        "timestamp_utc": timestamp_utc,
        "value": value,
        "unit": ts.get("unit"),
        "latitude": station.get("latitude"),
        "longitude": station.get("longitude"),
        "ingestion_ts_utc": ingestion_ts,
        "source": "pegelonline",
        "source_url": source_url,
        "segment_id": segment_id,
    }

    row = attach_record_hash(row)
    return row


def fetch_station_timeseries_measurements(
    station: dict,
    start: str | None,
    segment_cfg: dict,
) -> list[dict]:
    ingestion_ts = datetime.now(timezone.utc).isoformat()
    station_identifier = build_station_identifier(station)
    station_name = station.get("shortname") or station.get("longname") or "unknown"
    records = []

    for ts in station.get("timeseries", []) or []:
        ts_name = ts.get("shortname")
        if not ts_name or ts_name != "W":
            continue

        measurements_url = build_measurements_url(
            station_identifier=station_identifier,
            timeseries_shortname=ts_name,
            start=start,
        )

        try:
            measurements_payload = get_json(measurements_url)
            if isinstance(measurements_payload, list):
                for item in measurements_payload:
                    row = extract_row_from_measurement(
                        station=station,
                        ts=ts,
                        measurement=item,
                        ingestion_ts=ingestion_ts,
                        source_url=measurements_url,
                        segment_cfg=segment_cfg,
                    )
                    if row:
                        records.append(
                            PegelMeasurement(
                                **{k: v for k, v in row.items() if k in PegelMeasurement.model_fields}
                            ).model_dump()
                            | {
                                "source_record_hash": row["source_record_hash"],
                                "source_url": row["source_url"],
                                "segment_id": row["segment_id"],
                            }
                        )
        except Exception as exc:
            logger.warning(
                "measurements_fetch_failed station=%s timeseries=%s url=%s error=%s",
                station_name,
                ts_name,
                measurements_url,
                exc,
            )
            continue

    return records


def run_pegelonline_ingestion(mode: str = "incremental", hours: int = 72) -> None:
    url = f"{settings.pegelonline_base_url}/stations.json?includeTimeseries=true&includeCurrentMeasurement=true"
    logger.info("fetch_start source=pegelonline url=%s mode=%s hours=%s", url, mode, hours)
    logger.info(
        "runtime_settings project_id=%s dataset_raw=%s region=%s",
        settings.project_id,
        settings.dataset_raw,
        settings.gcp_region,
    )

    previous_watermark = get_watermark("pegelonline")
    logger.info("watermark_loaded payload=%s", previous_watermark)

    payload = get_json(url)
    rhine_cfg = load_rhine_config()
    segment_cfg = load_segment_config()

    rhine_stations = [station for station in payload if is_rhine_station(station, rhine_cfg)]
    logger.info(
        "rhine_filter_complete total_stations=%s rhine_stations=%s",
        len(payload),
        len(rhine_stations),
    )

    api_start = to_api_iso8601(rhine_cfg.get("api_handoff_start"))
    if mode != "backfill" and previous_watermark and previous_watermark.get("max_observed_measurement_ts_utc"):
        watermark_start = to_api_iso8601(previous_watermark["max_observed_measurement_ts_utc"])
        if watermark_start:
            api_start = watermark_start

    records = []
    station_failed = 0

    for station in rhine_stations:
        try:
            station_records = fetch_station_timeseries_measurements(
                station=station,
                start=api_start,
                segment_cfg=segment_cfg,
            )
            records.extend(station_records)
        except Exception as exc:
            station_failed += 1
            logger.warning(
                "station_parse_failed station=%s error=%s",
                station.get("shortname"),
                exc,
            )

    df = pd.DataFrame(records)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df["ingestion_ts_utc"] = pd.to_datetime(df["ingestion_ts_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp_utc"]).reset_index(drop=True)

        df = (
            df.sort_values(["station_name", "timeseries_name", "timestamp_utc"])
              .drop_duplicates(subset=["source_record_hash"], keep="last")
              .reset_index(drop=True)
        )

        logger.info(
            "local_persist_start rows=%s unique_stations=%s unique_segments=%s api_start=%s",
            len(df),
            df["station_id"].nunique() if not df.empty else 0,
            df["segment_id"].nunique() if "segment_id" in df.columns and not df.empty else 0,
            api_start,
        )

        write_local_parquet(df, f"data/raw/pegelonline/pegelonline_{stamp}.parquet")

        json_df = df.copy()
        json_df["timestamp_utc"] = json_df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        json_df["ingestion_ts_utc"] = json_df["ingestion_ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        write_jsonl(json_df.to_dict(orient="records"), f"data/raw/pegelonline/pegelonline_{stamp}.jsonl")

        logger.info("bq_branch_check project_id_present=%s", bool(settings.project_id))

        if settings.project_id:
            bq_df = df.drop(columns=["segment_id"], errors="ignore").copy()
            merge_dataframe_to_bigquery(bq_df, table_name="pegelonline_measurements")

        max_ts = df["timestamp_utc"].max()
        set_watermark(
            "pegelonline",
            {
                "last_successful_ingestion_ts_utc": datetime.now(timezone.utc).isoformat(),
                "max_observed_measurement_ts_utc": max_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "rows_written": int(len(df)),
                "mode": mode,
                "api_start": api_start,
            },
        )

    logger.info(
        "fetch_complete source=pegelonline rows=%s rhine_stations=%s stations_failed=%s unique_stations=%s",
        len(df),
        len(rhine_stations),
        station_failed,
        df["station_id"].nunique() if not df.empty else 0,
    )