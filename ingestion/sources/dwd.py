from datetime import datetime, timezone
import re

import pandas as pd
import requests

from ingestion.common.bigquery_dedup import merge_dataframe_to_bigquery
from ingestion.common.config import settings
from ingestion.common.logging_utils import get_logger
from ingestion.common.storage import write_jsonl, write_local_parquet
from ingestion.common.utils import load_yaml
from ingestion.sources.dwd_parser import build_canonical_dwd_frame, parse_dwd_zip_bytes

logger = get_logger(__name__)


def load_dwd_config() -> dict:
    return load_yaml("config/dwd.yaml")


def load_station_scope() -> dict:
    return load_yaml("config/dwd_station_scope.yaml")


def fetch_text(url: str) -> str:
    resp = requests.get(url, timeout=settings.timeout_seconds)
    resp.raise_for_status()
    return resp.text


def fetch_bytes(url: str) -> bytes:
    resp = requests.get(url, timeout=settings.timeout_seconds)
    resp.raise_for_status()
    return resp.content


def extract_zip_links(index_html: str) -> list[str]:
    return re.findall(r'href="([^"]+\.zip)"', index_html, flags=re.IGNORECASE)


def station_in_scope(station_id: str, station_name: str, scope_cfg: dict, latitude=None, longitude=None) -> bool:
    if pd.isna(station_id):
        return False

    station_id = str(station_id).strip().zfill(5)
    station_name = str(station_name).strip().upper() if pd.notna(station_name) else ""

    target_ids = {str(x).strip().zfill(5) for x in scope_cfg.get("target_station_ids", [])}
    if target_ids and station_id in target_ids:
        return True

    bbox = scope_cfg.get("bounding_box", {})
    in_bbox = False
    if latitude is not None and longitude is not None and bbox:
        try:
            lat = float(latitude)
            lon = float(longitude)
            in_bbox = (
                bbox["min_lat"] <= lat <= bbox["max_lat"]
                and bbox["min_lon"] <= lon <= bbox["max_lon"]
            )
        except Exception:
            in_bbox = False

    target_names = [str(s).strip().upper() for s in scope_cfg.get("target_station_names", [])]
    name_match = False
    for target in target_names:
        if target in station_name:
            name_match = True
            break
        if target.replace("OE", "Ö") in station_name:
            name_match = True
            break
        if target.replace("UE", "Ü") in station_name:
            name_match = True
            break
        if target.replace("AE", "Ä") in station_name:
            name_match = True
            break

    return in_bbox and name_match


def attach_station_metadata(df: pd.DataFrame, station_meta: pd.DataFrame) -> pd.DataFrame:
    meta = station_meta.copy()
    meta["station_id_norm"] = meta["Stations_id"].astype(str).str.zfill(5)

    out = df.merge(
        meta[["station_id_norm", "Stationsname", "geoBreite", "geoLaenge"]],
        how="left",
        left_on="dwd_station_id",
        right_on="station_id_norm",
    )

    out["dwd_station_name"] = out["Stationsname"]
    out["latitude"] = pd.to_numeric(out["geoBreite"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["geoLaenge"], errors="coerce")

    return out.drop(columns=["station_id_norm", "Stationsname", "geoBreite", "geoLaenge"], errors="ignore")


def fetch_station_meta(base_url: str) -> pd.DataFrame:
    stations_url = f"{base_url}/air_temperature/recent/TU_Stundenwerte_Beschreibung_Stationen.txt"
    raw_bytes = fetch_bytes(stations_url)
    lines = raw_bytes.decode("latin1").splitlines()

    parsed_data = []

    for line in lines[2:]:
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) < 8:
            continue

        parsed_data.append(
            {
                "Stations_id": parts[0].strip(),
                "von_datum": parts[1].strip(),
                "bis_datum": parts[2].strip(),
                "Stationshoehe": parts[3].strip(),
                "geoBreite": parts[4].strip(),
                "geoLaenge": parts[5].strip(),
                "Stationsname": " ".join(parts[6:-1]).strip(),
                "Bundesland": parts[-1].strip(),
            }
        )

    return pd.DataFrame(parsed_data)


def fetch_latest_variable_frame(
    base_url: str,
    variable_key: str,
    variable_cfg: dict,
    valid_station_ids: set[str],
) -> list[tuple[str, pd.DataFrame]]:
    index_url = f"{base_url}/{variable_cfg['folder']}/{variable_cfg['recent_subdir']}/"
    html = fetch_text(index_url)
    zip_links = extract_zip_links(html)

    outputs = []
    for link in zip_links:
        if not any(sid in link for sid in valid_station_ids):
            continue

        full_url = index_url + link
        content = fetch_bytes(full_url)
        raw_df = parse_dwd_zip_bytes(content)
        outputs.append((full_url, raw_df))

    return outputs


def merge_variable_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    cleaned_frames = [df.dropna(axis=1, how="all") for df in frames]
    combined = pd.concat(cleaned_frames, ignore_index=True)
    merged = combined.groupby(["dwd_station_id", "timestamp_utc"], as_index=False).first()

    return merged


def run_dwd_ingestion(mode: str = "recent") -> None:
    cfg = load_dwd_config()
    scope_cfg = load_station_scope()
    base_url = cfg["base_url"]
    raw_output_dir = cfg["raw_output_dir"]
    ingestion_ts_utc = datetime.now(timezone.utc).isoformat()

    logger.info("dwd_fetch_start mode=%s base_url=%s", mode, base_url)

    station_meta = fetch_station_meta(base_url)
    station_meta = station_meta[
        station_meta.apply(
            lambda row: station_in_scope(
                row["Stations_id"],
                row.get("Stationsname"),
                scope_cfg,
                latitude=row.get("geoBreite"),
                longitude=row.get("geoLaenge"),
            ),
            axis=1,
        )
    ].copy()

    max_station_count = int(scope_cfg.get("max_station_count", 25))
    station_meta = station_meta.sort_values(["Stations_id"]).head(max_station_count).copy()

    logger.info(
        "dwd_station_scope_complete scoped_stations=%s max_station_count=%s",
        len(station_meta),
        max_station_count,
    )

    if station_meta.empty:
        logger.info("dwd_fetch_complete rows=0")
        return

    valid_station_ids = set(station_meta["Stations_id"].astype(str).str.zfill(5))
    canonical_frames = []

    for variable_key, variable_cfg in cfg["variables"].items():
        payloads = fetch_latest_variable_frame(base_url, variable_key, variable_cfg, valid_station_ids)

        for source_url, raw_df in payloads:
            canonical = build_canonical_dwd_frame(
                df=raw_df,
                variable_family=variable_key,
                source_url=source_url,
                ingestion_ts_utc=ingestion_ts_utc,
            )
            canonical = canonical[canonical["dwd_station_id"].isin(valid_station_ids)]
            canonical_frames.append(canonical)

    merged = merge_variable_frames(canonical_frames)
    if merged.empty:
        logger.info("dwd_fetch_complete rows=0")
        return

    merged = attach_station_metadata(merged, station_meta)
    merged = merged.sort_values(["dwd_station_id", "timestamp_utc"]).reset_index(drop=True)

    merged = merged[
        [
            "dwd_station_id",
            "dwd_station_name",
            "timestamp_utc",
            "latitude",
            "longitude",
            "temperature_c",
            "precipitation_mm",
            "wind_speed_ms",
            "pressure_hpa",
            "relative_humidity_pct",
            "ingestion_ts_utc",
            "source",
            "source_record_hash",
            "source_url",
        ]
    ].drop_duplicates(subset=["source_record_hash"], keep="last")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    write_local_parquet(merged, f"{raw_output_dir}/dwd_hourly_{stamp}.parquet")

    json_df = merged.copy()
    json_df["timestamp_utc"] = json_df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    json_df["ingestion_ts_utc"] = json_df["ingestion_ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    write_jsonl(json_df.to_dict(orient="records"), f"{raw_output_dir}/dwd_hourly_{stamp}.jsonl")

    if settings.project_id:
        merge_dataframe_to_bigquery(merged, table_name="dwd_hourly_observations")

    logger.info(
        "dwd_fetch_complete rows=%s unique_stations=%s",
        len(merged),
        merged["dwd_station_id"].nunique() if not merged.empty else 0,
    )