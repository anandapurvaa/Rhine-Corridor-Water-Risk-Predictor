from datetime import datetime, timezone
from urllib.parse import quote

import pandas as pd

from ingestion.common.bigquery_writer import write_dataframe_to_bigquery
from ingestion.common.config import settings
from ingestion.common.http import get_json
from ingestion.common.logging_utils import get_logger
from ingestion.common.storage import write_jsonl, write_local_parquet
from ingestion.common.utils import load_yaml
from ingestion.historical.archive_parser import normalize_archive_dataframe

logger = get_logger(__name__)


def month_chunks(start_date: str, end_date: str, chunk_months: int = 1) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    chunks = []
    current = start

    while current <= end:
        next_start = current + pd.DateOffset(months=chunk_months)
        chunk_end = next_start - pd.offsets.Day(1)
        if chunk_end > end:
            chunk_end = end

        chunks.append((current, chunk_end))
        current = chunk_end + pd.offsets.Day(1)

    return chunks


def load_backfill_config() -> dict:
    return load_yaml("config/historical_backfill.yaml")


def load_rhine_config() -> dict:
    return load_yaml("config/rhine_gauges.yaml")


def is_rhine_station(station: dict, rhine_cfg: dict) -> bool:
    name = (station.get("shortname") or station.get("longname") or "").upper()
    if name in set(rhine_cfg["rhine_priority_gauges"]):
        return True
    for keyword in rhine_cfg["rhine_name_keywords"]:
        if keyword in name:
            return True
    return False


def build_measurements_url(station_identifier: str, timeseries_shortname: str, start_iso: str, end_iso: str) -> str:
    station_id = quote(str(station_identifier), safe="")
    ts_name = quote(str(timeseries_shortname), safe="")
    return (
        f"{settings.pegelonline_base_url}/stations/{station_id}/{ts_name}/measurements.json"
        f"?start={quote(start_iso, safe='')}&end={quote(end_iso, safe='')}"
    )


def fetch_station_history_chunk(
    station: dict,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    ingestion_ts_utc: str,
    timezone_name: str,
) -> pd.DataFrame:
    station_id = str(station.get("uuid", ""))
    station_name = station.get("shortname") or station.get("longname") or "unknown"
    records = []

    start_iso = start_ts.strftime("%Y-%m-%dT00:00:00+01:00")
    end_iso = end_ts.strftime("%Y-%m-%dT23:59:00+01:00")

    for ts in station.get("timeseries", []) or []:
        ts_name = ts.get("shortname")
        if ts_name != "W":
            continue

        url = build_measurements_url(station_id, ts_name, start_iso, end_iso)

        try:
            payload = get_json(url)
            if not isinstance(payload, list) or not payload:
                continue

            raw_df = pd.DataFrame(payload)
            if "timestamp" not in raw_df.columns or "value" not in raw_df.columns:
                continue

            raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True, errors="coerce")
            raw_df["timestamp"] = raw_df["timestamp"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d %H:%M")
            raw_df = raw_df[["timestamp", "value"]]

            normalized = normalize_archive_dataframe(
                raw_df,
                station_id=station_id,
                station_name=station_name,
                timeseries_name=ts_name,
                unit=ts.get("unit"),
                source_url=url,
                ingestion_ts_utc=ingestion_ts_utc,
                timezone_name=timezone_name,
            )
            records.append(normalized)
        except Exception as exc:
            logger.warning(
                "historical_fetch_failed station=%s start=%s end=%s url=%s error=%s",
                station_name,
                start_ts,
                end_ts,
                url,
                exc,
            )

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def run_pegelonline_historical_backfill(
    from_date: str,
    to_date: str,
    chunk_months: int | None = None,
) -> None:
    cfg = load_backfill_config()
    rhine_cfg = load_rhine_config()

    chunk_months = chunk_months or cfg["pegelonline"]["default_chunk_months"]
    timezone_name = cfg["pegelonline"]["timezone_name"]
    raw_output_dir = cfg["pegelonline"]["raw_output_dir"]

    stations_url = f"{settings.pegelonline_base_url}/stations.json?includeTimeseries=true"
    stations = get_json(stations_url)
    rhine_stations = [s for s in stations if is_rhine_station(s, rhine_cfg)]

    logger.info(
        "historical_backfill_start from_date=%s to_date=%s chunk_months=%s rhine_stations=%s",
        from_date,
        to_date,
        chunk_months,
        len(rhine_stations),
    )

    for start_ts, end_ts in month_chunks(from_date, to_date, chunk_months=chunk_months):
        ingestion_ts_utc = datetime.now(timezone.utc).isoformat()
        batch_frames = []

        logger.info("historical_chunk_start start=%s end=%s", start_ts.date(), end_ts.date())

        for station in rhine_stations:
            df_station = fetch_station_history_chunk(
                station=station,
                start_ts=start_ts,
                end_ts=end_ts,
                ingestion_ts_utc=ingestion_ts_utc,
                timezone_name=timezone_name,
            )
            if not df_station.empty:
                batch_frames.append(df_station)

        if not batch_frames:
            logger.info("historical_chunk_empty start=%s end=%s", start_ts.date(), end_ts.date())
            continue

        df = pd.concat(batch_frames, ignore_index=True)
        df = (
            df.sort_values(["station_name", "timeseries_name", "timestamp_utc"])
              .drop_duplicates(subset=["source_record_hash"], keep="last")
              .reset_index(drop=True)
        )

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        parquet_path = f"{raw_output_dir}/pegelonline_hist_{start_ts.strftime('%Y%m')}_{stamp}.parquet"
        jsonl_path = f"{raw_output_dir}/pegelonline_hist_{start_ts.strftime('%Y%m')}_{stamp}.jsonl"

        write_local_parquet(df, parquet_path)
        write_jsonl(df.to_dict(orient="records"), jsonl_path)

        if settings.project_id:
            bq_df = df.drop(columns=["segment_id"], errors="ignore")
            write_dataframe_to_bigquery(bq_df, table_name="pegelonline_measurements")

        logger.info(
            "historical_chunk_complete start=%s end=%s rows=%s unique_stations=%s",
            start_ts.date(),
            end_ts.date(),
            len(df),
            df["station_id"].nunique() if not df.empty else 0,
        )

    logger.info("historical_backfill_complete from_date=%s to_date=%s", from_date, to_date)