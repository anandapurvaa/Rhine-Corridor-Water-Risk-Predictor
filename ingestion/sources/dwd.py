from datetime import datetime, timezone
import json
import re

import pandas as pd
import requests
from google.cloud import bigquery

from ingestion.common.bigquery_dedup import merge_dataframe_to_bigquery
from ingestion.common.config import settings
from ingestion.common.logging_utils import get_logger
from ingestion.common.storage import write_jsonl, write_local_parquet
from ingestion.common.utils import load_yaml
from ingestion.sources.dwd_audit import build_station_audit_outputs
from ingestion.sources.dwd_parser import build_canonical_dwd_frame, parse_dwd_zip_bytes

logger = get_logger(__name__)


VARIABLE_COLUMN_MAP = {
    "air_temperature": "temperature_c",
    "precipitation": "precipitation_mm",
    "wind": "wind_speed_ms",
    "pressure": "pressure_hpa",
}

IDW_VARIABLES = {"air_temperature", "precipitation"}

ALL_FEATURE_COLUMNS = [
    "temperature_c",
    "precipitation_mm",
    "wind_speed_ms",
    "pressure_hpa",
    "relative_humidity_pct",
]


def load_dwd_config() -> dict:
    return load_yaml("config/dwd.yaml")


def load_station_scope() -> dict:
    return load_yaml("config/dwd_station_scope.yaml")


def normalized_target_station_ids(scope_cfg: dict) -> list[str]:
    return sorted({str(x).strip().zfill(5) for x in scope_cfg.get("target_station_ids", []) if str(x).strip()})


def normalized_id_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return sorted({str(v).strip().zfill(5) for v in values if str(v).strip()})


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


def extract_station_id_from_filename(filename: str) -> str | None:
    m = re.search(r'_(\d{5})_', filename)
    return m.group(1) if m else None


def zip_overlaps_start_date(filename: str, start_date: str = "2018-01-01") -> bool:
    m = re.search(r'_(\d{8})_(\d{8})_', filename)
    if not m:
        return True
    end = datetime.strptime(m.group(2), "%Y%m%d").date()
    cutoff = datetime.strptime(start_date, "%Y-%m-%d").date()
    return end >= cutoff


def get_bigquery_max_timestamp(
    table_name: str,
    dataset: str | None = None,
    timestamp_column: str = "timestamp_utc",
) -> pd.Timestamp | None:
    if not settings.project_id:
        return None

    target_dataset = dataset or settings.dataset_raw
    project_id = settings.project_id
    location = settings.gcp_region
    table_id = f"{project_id}.{target_dataset}.{table_name}"

    client = bigquery.Client(project=project_id, location=location)
    sql = f"""
    SELECT MAX(`{timestamp_column}`) AS max_ts
    FROM `{table_id}`
    """
    try:
        rows = list(client.query(sql).result())
    except Exception as exc:
        logger.warning("dwd_watermark_query_failed table=%s error=%s", table_id, exc)
        return None

    if not rows or rows[0]["max_ts"] is None:
        return None

    return pd.to_datetime(rows[0]["max_ts"], utc=True, errors="coerce")


def station_in_scope(
    station_id: str,
    station_name: str,
    scope_cfg: dict,
    latitude=None,
    longitude=None,
) -> bool:
    if pd.isna(station_id):
        return False

    station_id = str(station_id).strip().zfill(5)
    target_ids = set(normalized_target_station_ids(scope_cfg))

    if target_ids:
        return station_id in target_ids

    station_name = str(station_name).strip().upper() if pd.notna(station_name) else ""
    target_names = [str(s).strip().upper() for s in scope_cfg.get("target_station_names", [])]

    for target in target_names:
        if target in station_name:
            return True
        if target.replace("OE", "Ö") in station_name:
            return True
        if target.replace("UE", "Ü") in station_name:
            return True
        if target.replace("AE", "Ä") in station_name:
            return True

    return False


def parse_station_metadata_text(raw_bytes: bytes) -> pd.DataFrame:
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


def fetch_station_meta_for_variable(base_url: str, variable_cfg: dict) -> pd.DataFrame:
    station_file = variable_cfg["station_description_file"]
    stations_url = f"{base_url}/{variable_cfg['folder']}/recent/{station_file}"
    raw_bytes = fetch_bytes(stations_url)
    meta = parse_station_metadata_text(raw_bytes)
    meta["variable_family"] = variable_cfg["name"]
    return meta


def build_scoped_station_meta(base_url: str, cfg: dict, scope_cfg: dict) -> pd.DataFrame:
    target_ids = set(normalized_target_station_ids(scope_cfg))
    all_meta = []

    for variable_key, variable_cfg in cfg["variables"].items():
        variable_cfg = {**variable_cfg, "name": variable_key}
        try:
            meta = fetch_station_meta_for_variable(base_url, variable_cfg)
            meta["station_id_norm"] = meta["Stations_id"].astype(str).str.zfill(5)

            if target_ids:
                meta = meta[meta["station_id_norm"].isin(target_ids)].copy()
            else:
                meta = meta[
                    meta.apply(
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

            meta["variable_family"] = variable_key
            all_meta.append(meta)
        except Exception as exc:
            logger.warning("dwd_station_meta_fetch_failed variable=%s error=%s", variable_key, exc)

    if not all_meta:
        return pd.DataFrame()

    combined = pd.concat(all_meta, ignore_index=True)

    station_counts = (
        combined.groupby("station_id_norm")["variable_family"]
        .nunique()
        .reset_index(name="variable_family_count")
    )

    representative = (
        combined.sort_values(["station_id_norm", "variable_family"])
        .drop_duplicates(subset=["station_id_norm"], keep="first")
        .merge(station_counts, on="station_id_norm", how="left")
    )

    if target_ids:
        representative = representative[representative["station_id_norm"].isin(target_ids)].copy()
        representative["target_order"] = representative["station_id_norm"].map(
            {sid: i for i, sid in enumerate(normalized_target_station_ids(scope_cfg))}
        )
        representative = representative.sort_values(["target_order", "station_id_norm"]).drop(columns=["target_order"])

        missing_target_ids = sorted(target_ids - set(representative["station_id_norm"]))
        if missing_target_ids:
            logger.warning("dwd_station_scope_missing_target_ids ids=%s", ",".join(missing_target_ids))
    else:
        min_variable_families = int(cfg.get("min_variable_families", 4))
        representative = representative[representative["variable_family_count"] >= min_variable_families].copy()

        max_station_count = int(scope_cfg.get("max_station_count", 25))
        representative = representative.sort_values(["station_id_norm"]).head(max_station_count).copy()

    return representative


def attach_station_metadata(df: pd.DataFrame, station_meta: pd.DataFrame) -> pd.DataFrame:
    meta = station_meta.copy()

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


def fetch_variable_frames(
    base_url: str,
    variable_key: str,
    variable_cfg: dict,
    valid_station_ids: set[str],
    mode: str = "both",
    historical_start_date: str = "2018-01-01",
) -> list[tuple[str, pd.DataFrame]]:
    if mode == "historical":
        subdirs = [variable_cfg["historical_subdir"]]
    elif mode == "recent":
        subdirs = [variable_cfg["recent_subdir"]]
    else:
        subdirs = [variable_cfg["historical_subdir"], variable_cfg["recent_subdir"]]

    outputs = []
    found_station_ids = set()

    for subdir in subdirs:
        index_url = f"{base_url}/{variable_cfg['folder']}/{subdir}/"
        html = fetch_text(index_url)
        zip_links = extract_zip_links(html)

        logger.info(
            "dwd_variable_index variable=%s subdir=%s zip_count=%s",
            variable_key,
            subdir,
            len(zip_links),
        )

        for link in zip_links:
            link_station_id = extract_station_id_from_filename(link)

            if link_station_id not in valid_station_ids:
                continue

            if subdir == variable_cfg["historical_subdir"] and not zip_overlaps_start_date(link, historical_start_date):
                logger.info(
                    "skipping variable=%s subdir=%s station_id=%s reason=no_overlap url=%s",
                    variable_key,
                    subdir,
                    link_station_id,
                    index_url + link,
                )
                continue

            full_url = index_url + link
            logger.info(
                "downloading variable=%s subdir=%s station_id=%s url=%s",
                variable_key,
                subdir,
                link_station_id,
                full_url,
            )
            content = fetch_bytes(full_url)
            raw_df = parse_dwd_zip_bytes(content)
            logger.info(
                "parsed variable=%s station_id=%s rows=%s",
                variable_key,
                link_station_id,
                len(raw_df),
            )
            outputs.append((full_url, raw_df))
            found_station_ids.add(link_station_id)

    missing_station_ids = sorted(valid_station_ids - found_station_ids)
    logger.info(
        "dwd_variable_station_coverage variable=%s found_station_ids=%s missing_station_ids=%s",
        variable_key,
        ",".join(sorted(found_station_ids)),
        ",".join(missing_station_ids),
    )

    return outputs


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else None


def merge_variable_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    combined = pd.concat([df.dropna(axis=1, how="all") for df in frames], ignore_index=True)
    combined = combined.sort_values(["dwd_station_id", "timestamp_utc", "ingestion_ts_utc"])

    grouped = combined.groupby(["dwd_station_id", "timestamp_utc"], as_index=False).agg(
        {
            "temperature_c": first_non_null,
            "precipitation_mm": first_non_null,
            "wind_speed_ms": first_non_null,
            "pressure_hpa": first_non_null,
            "relative_humidity_pct": first_non_null,
            "ingestion_ts_utc": "last",
            "source": "last",
            "source_url": "last",
            "source_record_hash": "last",
        }
    )

    return grouped


def apply_incremental_recent_filter(
    df: pd.DataFrame,
    mode: str,
    cfg: dict,
) -> pd.DataFrame:
    if df.empty or mode != "recent":
        return df

    target_table_name = cfg.get("incremental_table_name", "dwd_hourly_observations")
    lookback_hours = int(cfg.get("incremental_lookback_hours", 48))
    watermark = get_bigquery_max_timestamp(table_name=target_table_name)

    if watermark is None:
        logger.info("dwd_incremental_filter_skip reason=no_watermark")
        return df

    cutoff = watermark - pd.Timedelta(hours=lookback_hours)
    filtered = df[df["timestamp_utc"] > cutoff].copy()

    logger.info(
        "dwd_incremental_filter_applied watermark=%s cutoff=%s before_rows=%s after_rows=%s",
        watermark,
        cutoff,
        len(df),
        len(filtered),
    )
    return filtered


def filter_by_feature_overlap(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if df.empty:
        return df

    min_ratio = float(cfg.get("min_feature_overlap_ratio", 0.80))
    min_rows = int(cfg.get("min_fully_populated_rows", 5000))

    stats = (
        df.assign(
            fully_populated=lambda x: (
                x["temperature_c"].notna()
                & x["precipitation_mm"].notna()
                & x["wind_speed_ms"].notna()
                & x["pressure_hpa"].notna()
                & x["relative_humidity_pct"].notna()
            )
        )
        .groupby("dwd_station_id", as_index=False)
        .agg(
            total_rows=("timestamp_utc", "count"),
            fully_populated_rows=("fully_populated", "sum"),
        )
    )

    stats["fully_populated_ratio"] = stats["fully_populated_rows"] / stats["total_rows"]
    eligible = stats[
        (stats["fully_populated_ratio"] >= min_ratio)
        & (stats["fully_populated_rows"] >= min_rows)
    ]["dwd_station_id"]

    logger.info(
        "dwd_overlap_filter_complete eligible_stations=%s total_stations=%s min_ratio=%s min_rows=%s",
        eligible.nunique(),
        stats["dwd_station_id"].nunique(),
        min_ratio,
        min_rows,
    )

    return df[df["dwd_station_id"].isin(set(eligible))].copy()


def _idw_fill_from_candidates(proxy_values: list[tuple[float, float]]) -> float | None:
    valid = [(dist, val) for dist, val in proxy_values if pd.notna(val)]
    if not valid:
        return None

    if any(dist == 0 for dist, _ in valid):
        zero_vals = [val for dist, val in valid if dist == 0]
        return float(zero_vals[0]) if zero_vals else None

    weights = [1.0 / (dist ** 2) for dist, _ in valid]
    weighted_sum = sum(val * w for (dist, val), w in zip(valid, weights))
    total_weight = sum(weights)
    if total_weight == 0:
        return None
    return float(weighted_sum / total_weight)


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    safe_df = df.copy()

    for col in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
            safe_df[col] = pd.to_datetime(safe_df[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    safe_df = safe_df.where(pd.notna(safe_df), None)

    return json.loads(safe_df.to_json(orient="records", date_format="iso"))


def build_gap_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "phase",
                "dwd_station_id",
                "dwd_station_name",
                "rows_total",
                "temperature_missing",
                "precipitation_missing",
                "wind_missing",
                "pressure_missing",
                "humidity_missing",
                "temperature_missing_ratio",
                "precipitation_missing_ratio",
                "wind_missing_ratio",
                "pressure_missing_ratio",
                "humidity_missing_ratio",
                "any_gap_rows",
                "fully_populated_rows",
                "fully_populated_ratio",
            ]
        )

    work = df.copy()

    summary = (
        work.assign(
            temperature_missing=work["temperature_c"].isna(),
            precipitation_missing=work["precipitation_mm"].isna(),
            wind_missing=work["wind_speed_ms"].isna(),
            pressure_missing=work["pressure_hpa"].isna(),
            humidity_missing=work["relative_humidity_pct"].isna(),
            any_gap_row=(
                work["temperature_c"].isna()
                | work["precipitation_mm"].isna()
                | work["wind_speed_ms"].isna()
                | work["pressure_hpa"].isna()
                | work["relative_humidity_pct"].isna()
            ),
            fully_populated=(
                work["temperature_c"].notna()
                & work["precipitation_mm"].notna()
                & work["wind_speed_ms"].notna()
                & work["pressure_hpa"].notna()
                & work["relative_humidity_pct"].notna()
            ),
        )
        .groupby(["dwd_station_id", "dwd_station_name"], dropna=False, as_index=False)
        .agg(
            rows_total=("timestamp_utc", "count"),
            temperature_missing=("temperature_missing", "sum"),
            precipitation_missing=("precipitation_missing", "sum"),
            wind_missing=("wind_missing", "sum"),
            pressure_missing=("pressure_missing", "sum"),
            humidity_missing=("humidity_missing", "sum"),
            any_gap_rows=("any_gap_row", "sum"),
            fully_populated_rows=("fully_populated", "sum"),
        )
    )

    summary["phase"] = label
    summary["temperature_missing_ratio"] = summary["temperature_missing"] / summary["rows_total"]
    summary["precipitation_missing_ratio"] = summary["precipitation_missing"] / summary["rows_total"]
    summary["wind_missing_ratio"] = summary["wind_missing"] / summary["rows_total"]
    summary["pressure_missing_ratio"] = summary["pressure_missing"] / summary["rows_total"]
    summary["humidity_missing_ratio"] = summary["humidity_missing"] / summary["rows_total"]
    summary["fully_populated_ratio"] = summary["fully_populated_rows"] / summary["rows_total"]

    return summary[
        [
            "phase",
            "dwd_station_id",
            "dwd_station_name",
            "rows_total",
            "temperature_missing",
            "precipitation_missing",
            "wind_missing",
            "pressure_missing",
            "humidity_missing",
            "temperature_missing_ratio",
            "precipitation_missing_ratio",
            "wind_missing_ratio",
            "pressure_missing_ratio",
            "humidity_missing_ratio",
            "any_gap_rows",
            "fully_populated_rows",
            "fully_populated_ratio",
        ]
    ].sort_values(["dwd_station_id"]).reset_index(drop=True)


def build_gap_delta_summary(pre_gap_df: pd.DataFrame, post_gap_df: pd.DataFrame) -> pd.DataFrame:
    if pre_gap_df.empty and post_gap_df.empty:
        return pd.DataFrame()

    merge_cols = ["dwd_station_id", "dwd_station_name"]

    pre = pre_gap_df.copy().drop(columns=["phase"], errors="ignore")
    post = post_gap_df.copy().drop(columns=["phase"], errors="ignore")

    delta = pre.merge(
        post,
        how="outer",
        on=merge_cols,
        suffixes=("_before", "_after"),
    )

    for col in [
        "rows_total",
        "temperature_missing",
        "precipitation_missing",
        "wind_missing",
        "pressure_missing",
        "humidity_missing",
        "any_gap_rows",
        "fully_populated_rows",
    ]:
        before_col = f"{col}_before"
        after_col = f"{col}_after"
        delta[before_col] = delta[before_col].fillna(0)
        delta[after_col] = delta[after_col].fillna(0)
        delta[f"{col}_delta"] = delta[after_col] - delta[before_col]

    for col in [
        "temperature_missing_ratio",
        "precipitation_missing_ratio",
        "wind_missing_ratio",
        "pressure_missing_ratio",
        "humidity_missing_ratio",
        "fully_populated_ratio",
    ]:
        before_col = f"{col}_before"
        after_col = f"{col}_after"
        delta[before_col] = delta[before_col].fillna(0.0)
        delta[after_col] = delta[after_col].fillna(0.0)
        delta[f"{col}_delta"] = delta[after_col] - delta[before_col]

    return delta.sort_values(["dwd_station_id"]).reset_index(drop=True)


def log_gap_summary(gap_df: pd.DataFrame, label: str) -> None:
    if gap_df.empty:
        logger.info("dwd_gap_summary phase=%s rows=0", label)
        return

    totals = {
        "stations": int(gap_df["dwd_station_id"].nunique()),
        "rows_total": int(gap_df["rows_total"].sum()),
        "temperature_missing": int(gap_df["temperature_missing"].sum()),
        "precipitation_missing": int(gap_df["precipitation_missing"].sum()),
        "wind_missing": int(gap_df["wind_missing"].sum()),
        "pressure_missing": int(gap_df["pressure_missing"].sum()),
        "humidity_missing": int(gap_df["humidity_missing"].sum()),
        "any_gap_rows": int(gap_df["any_gap_rows"].sum()),
    }

    logger.info(
        "dwd_gap_summary phase=%s stations=%s rows_total=%s temperature_missing=%s precipitation_missing=%s wind_missing=%s pressure_missing=%s humidity_missing=%s any_gap_rows=%s",
        label,
        totals["stations"],
        totals["rows_total"],
        totals["temperature_missing"],
        totals["precipitation_missing"],
        totals["wind_missing"],
        totals["pressure_missing"],
        totals["humidity_missing"],
        totals["any_gap_rows"],
    )


def apply_proxy_backfill(
    merged: pd.DataFrame,
    proxy_df: pd.DataFrame,
    scope_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if merged.empty:
        return merged, pd.DataFrame()

    primary_ids = set(normalized_id_list(scope_cfg.get("primary_station_ids", [])))
    if not primary_ids:
        return merged, pd.DataFrame()

    out = merged.copy()
    out["is_proxy_backfilled"] = False
    out["proxy_source_station_id"] = None
    out["proxy_source_variable"] = None
    out["proxy_source_distance_km"] = None
    out["proxy_fill_method"] = None

    backfill_events = []

    if proxy_df.empty:
        return out, pd.DataFrame()

    proxy_df = proxy_df.sort_values(["target_station_id", "missing_variable", "distance_km"]).copy()

    for variable_name, target_column in VARIABLE_COLUMN_MAP.items():
        variable_proxies = proxy_df[proxy_df["missing_variable"] == variable_name].copy()
        if variable_proxies.empty:
            continue

        for target_station_id in sorted(primary_ids):
            station_proxy_options = variable_proxies[
                variable_proxies["target_station_id"] == target_station_id
            ].copy()
            if station_proxy_options.empty:
                continue

            target_mask = out["dwd_station_id"] == target_station_id
            if not target_mask.any():
                continue

            target_rows = out.loc[target_mask, ["timestamp_utc", target_column]].copy()
            missing_rows = target_rows[target_rows[target_column].isna()].copy()
            if missing_rows.empty:
                continue

            proxy_sources = []
            for _, proxy_row in station_proxy_options.iterrows():
                proxy_station_id = str(proxy_row["proxy_station_id"]).zfill(5)
                distance_km = float(proxy_row["distance_km"])

                proxy_series = out.loc[
                    out["dwd_station_id"] == proxy_station_id,
                    ["timestamp_utc", target_column],
                ].rename(columns={target_column: proxy_station_id})

                if proxy_series.empty:
                    continue

                proxy_sources.append((proxy_station_id, distance_km, proxy_series))

            if not proxy_sources:
                continue

            wide = missing_rows[["timestamp_utc"]].copy()
            for proxy_station_id, distance_km, proxy_series in proxy_sources:
                wide = wide.merge(proxy_series, on="timestamp_utc", how="left")

            filled_count = 0

            for _, row in wide.iterrows():
                timestamp = row["timestamp_utc"]
                candidate_values = []
                candidate_station_ids = []

                for proxy_station_id, distance_km, _ in proxy_sources:
                    proxy_val = row.get(proxy_station_id)
                    if pd.notna(proxy_val):
                        candidate_values.append((distance_km, proxy_val))
                        candidate_station_ids.append((proxy_station_id, distance_km))

                if not candidate_values:
                    continue

                fill_value = None
                fill_method = None
                source_station = None
                source_distance = None

                if variable_name in IDW_VARIABLES and len(candidate_values) >= 2:
                    fill_value = _idw_fill_from_candidates(candidate_values)
                    fill_method = "idw"
                    source_station = ",".join([sid for sid, _ in candidate_station_ids])
                    source_distance = min(dist for _, dist in candidate_station_ids)
                else:
                    nearest_station_id, nearest_distance = sorted(candidate_station_ids, key=lambda x: x[1])[0]
                    fill_value = float(row[nearest_station_id])
                    fill_method = "nearest"
                    source_station = nearest_station_id
                    source_distance = nearest_distance

                if fill_value is None:
                    continue

                global_fill_mask = (
                    (out["dwd_station_id"] == target_station_id)
                    & (out["timestamp_utc"] == timestamp)
                    & (out[target_column].isna())
                )

                if not global_fill_mask.any():
                    continue

                out.loc[global_fill_mask, target_column] = fill_value
                out.loc[global_fill_mask, "is_proxy_backfilled"] = True
                out.loc[global_fill_mask, "proxy_source_station_id"] = source_station
                out.loc[global_fill_mask, "proxy_source_variable"] = variable_name
                out.loc[global_fill_mask, "proxy_source_distance_km"] = source_distance
                out.loc[global_fill_mask, "proxy_fill_method"] = fill_method

                filled_count += int(global_fill_mask.sum())

            if filled_count > 0:
                backfill_events.append(
                    {
                        "target_station_id": target_station_id,
                        "variable_name": variable_name,
                        "fill_method": "idw" if variable_name in IDW_VARIABLES else "nearest",
                        "candidate_proxy_count": len(proxy_sources),
                        "filled_rows": filled_count,
                    }
                )

    backfill_df = pd.DataFrame(backfill_events)
    logger.info(
        "dwd_proxy_backfill_complete events=%s filled_rows=%s",
        len(backfill_df),
        int(backfill_df["filled_rows"].sum()) if not backfill_df.empty else 0,
    )

    return out, backfill_df


def run_dwd_ingestion(mode: str = "both") -> dict:
    cfg = load_dwd_config()
    scope_cfg = load_station_scope()
    base_url = cfg["base_url"]
    raw_output_dir = cfg["raw_output_dir"]
    historical_start_date = cfg.get("historical_start_date", "2018-01-01")
    start_ts = pd.Timestamp(historical_start_date, tz="UTC")
    ingestion_ts_utc = datetime.now(timezone.utc).isoformat()

    logger.info("dwd_fetch_start mode=%s base_url=%s", mode, base_url)

    station_meta = build_scoped_station_meta(base_url, cfg, scope_cfg)

    logger.info(
        "dwd_station_scope_complete scoped_stations=%s station_ids=%s",
        len(station_meta),
        ",".join(station_meta["station_id_norm"].astype(str).tolist()) if not station_meta.empty else "",
    )

    if station_meta.empty:
        logger.info("dwd_fetch_complete rows=0")
        return {
            "source": "dwd",
            "mode": mode,
            "rows_ingested": 0,
            "stations_processed": 0,
            "proxy_backfilled_rows": 0,
        }

    valid_station_ids = set(station_meta["station_id_norm"])
    canonical_frames = []

    for variable_key, variable_cfg in cfg["variables"].items():
        payloads = fetch_variable_frames(
            base_url=base_url,
            variable_key=variable_key,
            variable_cfg=variable_cfg,
            valid_station_ids=valid_station_ids,
            mode=mode,
            historical_start_date=historical_start_date,
        )

        logger.info("dwd_variable_payloads variable=%s files=%s", variable_key, len(payloads))

        for source_url, raw_df in payloads:
            canonical = build_canonical_dwd_frame(
                df=raw_df,
                variable_family=variable_key,
                source_url=source_url,
                ingestion_ts_utc=ingestion_ts_utc,
            )
            canonical = canonical[canonical["dwd_station_id"].isin(valid_station_ids)]
            canonical = canonical[canonical["timestamp_utc"].notna()]
            canonical = canonical[canonical["timestamp_utc"] >= start_ts]
            if not canonical.empty:
                canonical_frames.append(canonical)

    merged = merge_variable_frames(canonical_frames)
    if merged.empty:
        logger.info("dwd_fetch_complete rows=0")
        return {
            "source": "dwd",
            "mode": mode,
            "rows_ingested": 0,
            "stations_processed": 0,
            "proxy_backfilled_rows": 0,
        }

    merged = apply_incremental_recent_filter(merged, mode=mode, cfg=cfg)
    if merged.empty:
        logger.info("dwd_fetch_complete rows=0 after_incremental_filter")
        return {
            "source": "dwd",
            "mode": mode,
            "rows_ingested": 0,
            "stations_processed": 0,
            "proxy_backfilled_rows": 0,
        }

    overlap_cfg = dict(cfg)

    if mode == "recent":
        overlap_cfg["min_fully_populated_rows"] = int(
            cfg.get("recent_min_fully_populated_rows", 24)
        )

    logger.info(
        "dwd_overlap_filter_config mode=%s min_ratio=%s min_rows=%s",
        mode,
        overlap_cfg.get("min_feature_overlap_ratio", 0.80),
        overlap_cfg["min_fully_populated_rows"],
    )

    merged = filter_by_feature_overlap(merged, overlap_cfg)

    if merged.empty:
        logger.info("dwd_fetch_complete rows=0 after_overlap_filter")
        return {
            "source": "dwd",
            "mode": mode,
            "rows_ingested": 0,
            "stations_processed": 0,
            "proxy_backfilled_rows": 0,
        }

    merged = attach_station_metadata(merged, station_meta)
    merged = merged.sort_values(["dwd_station_id", "timestamp_utc"]).reset_index(drop=True)

    station_variable_map, audit_df, proxy_df = build_station_audit_outputs(
        station_meta=station_meta,
        merged=merged,
        cfg=cfg,
        scope_cfg=scope_cfg,
        proxy_distance_km=float(scope_cfg.get("proxy_station_max_distance_km", 175.0)),
    )

    pre_backfill_gap_df = build_gap_summary(merged, label="before_proxy_backfill")
    log_gap_summary(pre_backfill_gap_df, label="before_proxy_backfill")

    merged, backfill_df = apply_proxy_backfill(
        merged=merged,
        proxy_df=proxy_df,
        scope_cfg=scope_cfg,
    )

    post_backfill_gap_df = build_gap_summary(merged, label="after_proxy_backfill")
    log_gap_summary(post_backfill_gap_df, label="after_proxy_backfill")

    gap_delta_df = build_gap_delta_summary(
        pre_gap_df=pre_backfill_gap_df,
        post_gap_df=post_backfill_gap_df,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    final_columns = [
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
        "is_proxy_backfilled",
        "proxy_source_station_id",
        "proxy_source_variable",
        "proxy_source_distance_km",
        "proxy_fill_method",
        "ingestion_ts_utc",
        "source",
        "source_record_hash",
        "source_url",
    ]

    for col in final_columns:
        if col not in merged.columns:
            merged[col] = None

    merged = merged[final_columns].drop_duplicates(subset=["dwd_station_id", "timestamp_utc"], keep="last")
    merged = merged[pd.to_datetime(merged["timestamp_utc"], utc=True, errors="coerce") >= start_ts]

    write_local_parquet(merged, f"{raw_output_dir}/dwd_hourly_{stamp}.parquet")
    write_local_parquet(station_variable_map, f"{raw_output_dir}/dwd_station_variable_map_{stamp}.parquet")
    write_local_parquet(audit_df, f"{raw_output_dir}/dwd_station_audit_{stamp}.parquet")
    write_local_parquet(proxy_df, f"{raw_output_dir}/dwd_proxy_candidates_{stamp}.parquet")
    write_local_parquet(backfill_df, f"{raw_output_dir}/dwd_proxy_backfill_log_{stamp}.parquet")
    write_local_parquet(pre_backfill_gap_df, f"{raw_output_dir}/dwd_gap_summary_before_proxy_{stamp}.parquet")
    write_local_parquet(post_backfill_gap_df, f"{raw_output_dir}/dwd_gap_summary_after_proxy_{stamp}.parquet")
    write_local_parquet(gap_delta_df, f"{raw_output_dir}/dwd_gap_summary_delta_{stamp}.parquet")

    write_jsonl(dataframe_to_json_records(merged), f"{raw_output_dir}/dwd_hourly_{stamp}.jsonl")
    write_jsonl(dataframe_to_json_records(station_variable_map), f"{raw_output_dir}/dwd_station_variable_map_{stamp}.jsonl")
    write_jsonl(dataframe_to_json_records(audit_df), f"{raw_output_dir}/dwd_station_audit_{stamp}.jsonl")
    write_jsonl(dataframe_to_json_records(proxy_df), f"{raw_output_dir}/dwd_proxy_candidates_{stamp}.jsonl")
    write_jsonl(dataframe_to_json_records(backfill_df), f"{raw_output_dir}/dwd_proxy_backfill_log_{stamp}.jsonl")
    write_jsonl(dataframe_to_json_records(pre_backfill_gap_df), f"{raw_output_dir}/dwd_gap_summary_before_proxy_{stamp}.jsonl")
    write_jsonl(dataframe_to_json_records(post_backfill_gap_df), f"{raw_output_dir}/dwd_gap_summary_after_proxy_{stamp}.jsonl")
    write_jsonl(dataframe_to_json_records(gap_delta_df), f"{raw_output_dir}/dwd_gap_summary_delta_{stamp}.jsonl")

    if settings.project_id:
        merge_dataframe_to_bigquery(
            merged,
            table_name="dwd_hourly_observations",
            key_columns=["dwd_station_id", "timestamp_utc"],
        )

    logger.info(
        "dwd_fetch_complete rows=%s unique_stations=%s min_timestamp=%s max_timestamp=%s proxy_backfilled_rows=%s",
        len(merged),
        merged["dwd_station_id"].nunique() if not merged.empty else 0,
        merged["timestamp_utc"].min() if not merged.empty else None,
        merged["timestamp_utc"].max() if not merged.empty else None,
        int(merged["is_proxy_backfilled"].sum()) if "is_proxy_backfilled" in merged.columns else 0,
    )

    return {
        "source": "dwd",
        "mode": mode,
        "rows_ingested": int(len(merged)),
        "stations_processed": int(
            merged["dwd_station_id"].nunique()
        ) if not merged.empty else 0,
        "proxy_backfilled_rows": int(
            merged["is_proxy_backfilled"].sum()
        ) if (
            not merged.empty
            and "is_proxy_backfilled" in merged.columns
        ) else 0,
    }