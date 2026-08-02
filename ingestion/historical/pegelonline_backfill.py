from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import unicodedata

import pandas as pd

from ingestion.common.config import settings
from ingestion.common.hashing import stable_record_hash
from ingestion.common.http import get_json
from ingestion.common.logging_utils import get_logger
from ingestion.common.storage import write_jsonl, write_local_parquet
from ingestion.common.bigquery_dedup import merge_dataframe_to_bigquery
from ingestion.common.utils import load_yaml

logger = get_logger(__name__)

CSV_CHUNK_SIZE = 100_000
TIMESERIES_NAME = "W"

STATION_FILENAME_ALIASES = {
    "KÖLN": ["KLN"],
    "DÜSSELDORF": ["DSSELDORF"],
}


def load_backfill_config() -> dict:
    return load_yaml("config/historical_backfill.yaml")


def load_rhine_config() -> dict:
    return load_yaml("config/rhine_gauges.yaml")


def normalize_text(value: str) -> str:
    value = (value or "").strip().upper()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("ß", "SS")
    value = re.sub(r"[^A-Z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value


def compact_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", normalize_text(value))


def station_display_name(station: dict) -> str:
    return (station.get("shortname") or station.get("longname") or "unknown").strip()


def build_station_identifier(station: dict) -> str:
    return str(station.get("uuid") or "").strip()


def is_rhine_station(station: dict, rhine_cfg: dict) -> bool:
    station_name = normalize_text(station_display_name(station))
    allowed = {normalize_text(x) for x in rhine_cfg.get("rhine_priority_gauges", [])}
    excluded = {normalize_text(x) for x in rhine_cfg.get("excluded_stations", [])}
    return station_name in allowed and station_name not in excluded


def station_has_historical_backfill(station_name: str, rhine_cfg: dict) -> bool:
    missing = {normalize_text(x) for x in rhine_cfg.get("historical_missing_stations", [])}
    return normalize_text(station_name) not in missing


def station_filename_candidates(station_name: str) -> list[str]:
    raw_candidates = [
        station_name,
        normalize_text(station_name),
        compact_text(station_name),
    ]

    for alias in STATION_FILENAME_ALIASES.get(station_name.strip().upper(), []):
        raw_candidates.extend([alias, normalize_text(alias), compact_text(alias)])

    candidates = []
    seen = set()
    for item in raw_candidates:
        key = compact_text(item)
        if key and key not in seen:
            seen.add(key)
            candidates.append(item)

    return candidates


def find_station_csv_files(csv_root: Path, station_name: str) -> list[Path]:
    candidates = station_filename_candidates(station_name)
    normalized_candidates = [normalize_text(x) for x in candidates]
    compact_candidates = [compact_text(x) for x in candidates]

    matches: list[Path] = []

    for path in csv_root.rglob("*.csv"):
        stem_norm = normalize_text(path.stem)
        stem_compact = compact_text(path.stem)

        if any(
            candidate in stem_norm or candidate in stem_compact
            for candidate in normalized_candidates + compact_candidates
        ):
            matches.append(path)

    return sorted(matches)


def build_hashes(
    station_id: str,
    timestamps: pd.Series,
    values: pd.Series,
    source: str,
) -> list[str]:
    return [
        stable_record_hash(
            {
                "station_id": station_id,
                "timeseries_name": TIMESERIES_NAME,
                "timestamp_utc": ts,
                "value": val,
                "source": source,
            },
            keys=["station_id", "timeseries_name", "timestamp_utc", "value", "source"],
        )
        for ts, val in zip(timestamps, values)
    ]


def parse_station_csv_file(
    csv_path: Path,
    station: dict,
    ingestion_ts_utc: str,
) -> pd.DataFrame:
    station_id = build_station_identifier(station)
    station_name = station_display_name(station)
    ingestion_ts = pd.to_datetime(ingestion_ts_utc, utc=True, errors="coerce")
    latitude = station.get("latitude")
    longitude = station.get("longitude")

    chunks = []
    reader = pd.read_csv(
        csv_path,
        sep=";",
        usecols=["timestamp", "value"],
        dtype={"timestamp": "string", "value": "string"},
        engine="python",
        on_bad_lines="skip",
        chunksize=CSV_CHUNK_SIZE,
    )

    for chunk in reader:
        chunk.columns = [str(c).strip().lower() for c in chunk.columns]

        if set(chunk.columns) != {"timestamp", "value"}:
            logger.warning(
                "historical_csv_unexpected_columns file=%s columns=%s",
                csv_path,
                list(chunk.columns),
            )
            continue

        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        chunk["value"] = pd.to_numeric(
            chunk["value"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
            downcast="float",
        )
        chunk = chunk.dropna(subset=["timestamp", "value"]).reset_index(drop=True)

        if chunk.empty:
            continue

        chunk["timestamp"] = chunk["timestamp"].dt.tz_localize(
            "Europe/Berlin",
            ambiguous=False,
            nonexistent="shift_forward",
        ).dt.tz_convert("UTC")

        out_chunk = pd.DataFrame(
            {
                "station_id": station_id,
                "station_name": station_name,
                "timeseries_name": TIMESERIES_NAME,
                "timestamp_utc": chunk["timestamp"],
                "value": chunk["value"],
                "unit": "cm",
                "latitude": latitude,
                "longitude": longitude,
                "ingestion_ts_utc": ingestion_ts,
                "source": "pegelonline",
                "source_url": str(csv_path),
            }
        )

        out_chunk["source_record_hash"] = build_hashes(
            station_id=station_id,
            timestamps=out_chunk["timestamp_utc"],
            values=out_chunk["value"],
            source="pegelonline",
        )

        chunks.append(out_chunk)

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


def run_pegelonline_historical_backfill(
    from_date: str,
    to_date: str,
    chunk_months: int | None = None,
    station_filter: str | None = None,
) -> None:
    cfg = load_backfill_config()
    rhine_cfg = load_rhine_config()

    raw_output_dir = Path(cfg["pegelonline"]["raw_output_dir"])
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    csv_root = Path(rhine_cfg.get("historical_csv_root", "data/raw/pegelonline_historical"))
    historical_end = pd.Timestamp(rhine_cfg.get("historical_backfill_end", to_date)).date()

    stations = get_json(f"{settings.pegelonline_base_url}/stations.json?includeTimeseries=true")
    rhine_stations = [s for s in stations if is_rhine_station(s, rhine_cfg)]

    if station_filter:
        sf = normalize_text(station_filter)
        rhine_stations = [
            s for s in rhine_stations
            if sf in normalize_text(station_display_name(s)) or sf == normalize_text(str(s.get("uuid", "")))
        ]

    logger.info(
        "historical_backfill_start from_date=%s to_date=%s rhine_stations=%s csv_root=%s",
        from_date,
        to_date,
        len(rhine_stations),
        csv_root,
    )

    requested_from = pd.Timestamp(from_date).date()
    requested_to = min(pd.Timestamp(to_date).date(), historical_end)

    all_frames = []
    skipped_recent_only = []
    ingestion_ts_utc = datetime.now(timezone.utc).isoformat()

    for station in rhine_stations:
        station_name = station_display_name(station)

        if not station_has_historical_backfill(station_name, rhine_cfg):
            skipped_recent_only.append(station_name)
            logger.info("historical_skip_recent_only station=%s", station_name)
            continue

        csv_files = find_station_csv_files(csv_root, station_name)
        if not csv_files:
            logger.warning(
                "historical_csv_not_found station=%s csv_root=%s candidates=%s",
                station_name,
                csv_root,
                station_filename_candidates(station_name),
            )
            continue

        station_frames = []
        for csv_path in csv_files:
            try:
                df_file = parse_station_csv_file(
                    csv_path=csv_path,
                    station=station,
                    ingestion_ts_utc=ingestion_ts_utc,
                )
                if not df_file.empty:
                    station_frames.append(df_file)
            except Exception as exc:
                logger.warning(
                    "historical_csv_parse_failed station=%s file=%s error=%s",
                    station_name,
                    csv_path,
                    exc,
                )

        if not station_frames:
            logger.warning("historical_station_empty station=%s", station_name)
            continue

        df_station = pd.concat(station_frames, ignore_index=True)
        df_station = df_station[
            (df_station["timestamp_utc"].dt.date >= requested_from)
            & (df_station["timestamp_utc"].dt.date <= requested_to)
        ].reset_index(drop=True)

        if df_station.empty:
            logger.warning("historical_station_no_rows_in_window station=%s", station_name)
            continue

        all_frames.append(df_station)
        logger.info(
            "historical_station_ok station=%s rows=%s min_ts=%s max_ts=%s files=%s",
            station_name,
            len(df_station),
            df_station["timestamp_utc"].min(),
            df_station["timestamp_utc"].max(),
            len(csv_files),
        )

    if not all_frames:
        logger.info("historical_backfill_empty from_date=%s to_date=%s", from_date, to_date)
        return

    df = pd.concat(all_frames, ignore_index=True)
    df = (
        df.sort_values(["station_name", "timeseries_name", "timestamp_utc"])
        .drop_duplicates(subset=["source_record_hash"], keep="last")
        .reset_index(drop=True)
    )

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["ingestion_ts_utc"] = pd.to_datetime(df["ingestion_ts_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).reset_index(drop=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parquet_path = raw_output_dir / f"pegelonline_hist_{stamp}.parquet"
    jsonl_path = raw_output_dir / f"pegelonline_hist_{stamp}.jsonl"

    write_local_parquet(df, str(parquet_path))

    json_df = df.copy()
    json_df["timestamp_utc"] = json_df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    json_df["ingestion_ts_utc"] = json_df["ingestion_ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    write_jsonl(json_df.to_dict(orient="records"), str(jsonl_path))

    if settings.project_id:
        merge_dataframe_to_bigquery(df, table_name="pegelonline_measurements")

    logger.info(
        "historical_backfill_complete rows=%s unique_stations=%s skipped_recent_only=%s min_ts=%s max_ts=%s",
        len(df),
        df["station_id"].nunique() if not df.empty else 0,
        len(skipped_recent_only),
        df["timestamp_utc"].min() if not df.empty else None,
        df["timestamp_utc"].max() if not df.empty else None,
    )