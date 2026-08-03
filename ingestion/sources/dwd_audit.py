from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Any

import pandas as pd

from ingestion.common.logging_utils import get_logger

logger = get_logger(__name__)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return r * c


def _safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def normalized_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return sorted({str(v).strip().zfill(5) for v in values if str(v).strip()})


def build_station_variable_map(
    station_meta: pd.DataFrame,
    cfg: dict,
    scope_cfg: dict,
) -> pd.DataFrame:
    if station_meta.empty:
        return pd.DataFrame()

    expected_variables = list(cfg["variables"].keys())
    primary_ids = set(normalized_list(scope_cfg.get("primary_station_ids", [])))
    proxy_ids = set(normalized_list(scope_cfg.get("proxy_station_ids", [])))

    coverage = (
        station_meta.groupby("station_id_norm")["variable_family"]
        .agg(lambda s: sorted(set(s)))
        .reset_index(name="available_variables")
    )

    base = (
        station_meta.sort_values(["station_id_norm", "variable_family"])
        .drop_duplicates(subset=["station_id_norm"], keep="first")
        .copy()
    )

    out = base.merge(coverage, on="station_id_norm", how="left")
    out["variable_family_count"] = out["available_variables"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    for variable_name in expected_variables:
        out[f"has_{variable_name}"] = out["available_variables"].apply(
            lambda vals: variable_name in vals if isinstance(vals, list) else False
        )

    def station_role(station_id: str) -> str:
        if station_id in primary_ids:
            return "primary"
        if station_id in proxy_ids:
            return "proxy"
        return "unclassified"

    out["station_role"] = out["station_id_norm"].map(station_role)

    return out[
        [
            "station_id_norm",
            "Stationsname",
            "geoBreite",
            "geoLaenge",
            "station_role",
            "variable_family_count",
            "available_variables",
            *[f"has_{v}" for v in expected_variables],
        ]
    ].copy()


def build_observed_coverage_summary(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame()

    df = merged.copy()

    summary = (
        df.assign(
            temperature_present=df["temperature_c"].notna(),
            precipitation_present=df["precipitation_mm"].notna(),
            wind_present=df["wind_speed_ms"].notna(),
            pressure_present=df["pressure_hpa"].notna(),
            humidity_present=df["relative_humidity_pct"].notna(),
            fully_populated=(
                df["temperature_c"].notna()
                & df["precipitation_mm"].notna()
                & df["wind_speed_ms"].notna()
                & df["pressure_hpa"].notna()
                & df["relative_humidity_pct"].notna()
            ),
        )
        .groupby("dwd_station_id", as_index=False)
        .agg(
            rows_total=("timestamp_utc", "count"),
            min_timestamp_utc=("timestamp_utc", "min"),
            max_timestamp_utc=("timestamp_utc", "max"),
            temperature_non_null_ratio=("temperature_present", "mean"),
            precipitation_non_null_ratio=("precipitation_present", "mean"),
            wind_non_null_ratio=("wind_present", "mean"),
            pressure_non_null_ratio=("pressure_present", "mean"),
            humidity_non_null_ratio=("humidity_present", "mean"),
            fully_populated_ratio=("fully_populated", "mean"),
        )
    )

    return summary


def classify_station_status(
    coverage_df: pd.DataFrame,
    core_threshold: float = 0.80,
    partial_threshold: float = 0.40,
) -> pd.DataFrame:
    if coverage_df.empty:
        coverage_df["station_status"] = pd.Series(dtype="object")
        return coverage_df

    out = coverage_df.copy()

    def _classify(row):
        ratio = row.get("fully_populated_ratio")
        role = row.get("station_role", "unclassified")

        if role == "proxy":
            return "proxy_pool"

        if pd.isna(ratio):
            return "drop"
        if ratio >= core_threshold:
            return "core"
        if ratio >= partial_threshold:
            return "partial"
        return "drop"

    out["station_status"] = out.apply(_classify, axis=1)
    return out


def select_proxy_stations(
    station_variable_map: pd.DataFrame,
    target_variables: list[str],
    primary_station_ids: list[str],
    proxy_station_ids: list[str],
    max_proxy_distance_km: float = 175.0,
    max_proxies_per_variable: int = 3,
) -> pd.DataFrame:
    if station_variable_map.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    stations = station_variable_map.copy()
    primary_ids = set(normalized_list(primary_station_ids))
    proxy_ids = set(normalized_list(proxy_station_ids))

    primary_stations = stations[stations["station_id_norm"].isin(primary_ids)].copy()
    proxy_stations = stations[stations["station_id_norm"].isin(proxy_ids)].copy()

    for _, target in primary_stations.iterrows():
        target_id = str(target["station_id_norm"])
        target_name = target.get("Stationsname")
        target_lat = _safe_float(target.get("geoBreite"))
        target_lon = _safe_float(target.get("geoLaenge"))

        if target_lat is None or target_lon is None:
            continue

        for variable_name in target_variables:
            has_flag = bool(target.get(f"has_{variable_name}", False))
            if has_flag:
                continue

            candidates = []
            for _, proxy in proxy_stations.iterrows():
                proxy_id = str(proxy["station_id_norm"])
                if proxy_id == target_id:
                    continue
                if proxy_id not in proxy_ids:
                    continue
                if not bool(proxy.get(f"has_{variable_name}", False)):
                    continue

                proxy_lat = _safe_float(proxy.get("geoBreite"))
                proxy_lon = _safe_float(proxy.get("geoLaenge"))
                if proxy_lat is None or proxy_lon is None:
                    continue

                distance_km = haversine_km(target_lat, target_lon, proxy_lat, proxy_lon)
                if distance_km <= max_proxy_distance_km:
                    candidates.append(
                        {
                            "target_station_id": target_id,
                            "target_station_name": target_name,
                            "missing_variable": variable_name,
                            "proxy_station_id": proxy_id,
                            "proxy_station_name": proxy.get("Stationsname"),
                            "distance_km": round(distance_km, 2),
                        }
                    )

            candidates = sorted(candidates, key=lambda x: x["distance_km"])[:max_proxies_per_variable]
            rows.extend(candidates)

    return pd.DataFrame(rows)


def build_station_audit_outputs(
    station_meta: pd.DataFrame,
    merged: pd.DataFrame,
    cfg: dict,
    scope_cfg: dict,
    proxy_distance_km: float = 175.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    station_variable_map = build_station_variable_map(
        station_meta=station_meta,
        cfg=cfg,
        scope_cfg=scope_cfg,
    )
    observed_coverage = build_observed_coverage_summary(merged=merged)

    audit = station_variable_map.merge(
        observed_coverage,
        how="left",
        left_on="station_id_norm",
        right_on="dwd_station_id",
    )

    audit = classify_station_status(audit)

    proxy_candidates = select_proxy_stations(
        station_variable_map=station_variable_map,
        target_variables=list(cfg["variables"].keys()),
        primary_station_ids=scope_cfg.get("primary_station_ids", []),
        proxy_station_ids=scope_cfg.get("proxy_station_ids", []),
        max_proxy_distance_km=proxy_distance_km,
    )

    logger.info(
        "dwd_station_audit_complete stations=%s proxy_candidates=%s core=%s partial=%s drop=%s proxy_pool=%s",
        len(audit),
        len(proxy_candidates),
        int((audit["station_status"] == "core").sum()) if not audit.empty else 0,
        int((audit["station_status"] == "partial").sum()) if not audit.empty else 0,
        int((audit["station_status"] == "drop").sum()) if not audit.empty else 0,
        int((audit["station_status"] == "proxy_pool").sum()) if not audit.empty else 0,
    )

    return station_variable_map, audit, proxy_candidates