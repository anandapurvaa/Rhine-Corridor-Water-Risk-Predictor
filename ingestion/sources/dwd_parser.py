import io
import zipfile

import pandas as pd

from ingestion.common.hashing import stable_record_hash


def parse_dwd_zip_bytes(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        # CORRECTED: Only extract the file containing the actual data (starts with "produkt_")
        csv_names = [name for name in zf.namelist() if name.lower().startswith("produkt_") and name.lower().endswith(".txt")]
        if not csv_names:
            raise ValueError("No produkt_*.txt file found in DWD zip archive")

        with zf.open(csv_names[0]) as f:
            # CORRECTED: Added latin1 encoding
            df = pd.read_csv(f, sep=";", dtype=str, encoding="latin1")

    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_station_id(value: str) -> str:
    return str(value).strip().zfill(5)


def parse_dwd_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), format="%Y%m%d%H", utc=True, errors="coerce")


def build_canonical_dwd_frame(
    df: pd.DataFrame,
    variable_family: str,
    source_url: str,
    ingestion_ts_utc: str,
) -> pd.DataFrame:
    out = pd.DataFrame()

    out["dwd_station_id"] = df["STATIONS_ID"].map(normalize_station_id)
    out["timestamp_utc"] = parse_dwd_timestamp(df["MESS_DATUM"])
    out["ingestion_ts_utc"] = pd.to_datetime(ingestion_ts_utc, utc=True, errors="coerce")
    out["source"] = "dwd"
    out["source_url"] = source_url

    if "QN_9" in df.columns:
        out["quality_flag"] = df["QN_9"]
    elif "QN_8" in df.columns:
        out["quality_flag"] = df["QN_8"]
    else:
        out["quality_flag"] = None

    out["temperature_c"] = None
    out["precipitation_mm"] = None
    out["wind_speed_ms"] = None
    out["pressure_hpa"] = None
    out["relative_humidity_pct"] = None

    if variable_family == "air_temperature":
        if "TT_TU" in df.columns:
            out["temperature_c"] = pd.to_numeric(df["TT_TU"], errors="coerce")
        if "RF_TU" in df.columns:
            out["relative_humidity_pct"] = pd.to_numeric(df["RF_TU"], errors="coerce")

    elif variable_family == "precipitation":
        if "RS_IND" in df.columns:
            out["precip_indicator"] = pd.to_numeric(df["RS_IND"], errors="coerce")
        if "R1" in df.columns:
            out["precipitation_mm"] = pd.to_numeric(df["R1"], errors="coerce")
        elif "RSK" in df.columns:
            out["precipitation_mm"] = pd.to_numeric(df["RSK"], errors="coerce")

    elif variable_family == "wind":
        if "F" in df.columns:
            out["wind_speed_ms"] = pd.to_numeric(df["F"], errors="coerce")

    elif variable_family == "pressure":
        if "P0" in df.columns:
            out["pressure_hpa"] = pd.to_numeric(df["P0"], errors="coerce")
        elif "PP_10" in df.columns:
            out["pressure_hpa"] = pd.to_numeric(df["PP_10"], errors="coerce")

    out["dwd_station_name"] = None
    out["latitude"] = None
    out["longitude"] = None

    out["source_record_hash"] = out.apply(
        lambda row: stable_record_hash(
            {
                "dwd_station_id": row["dwd_station_id"],
                "timestamp_utc": str(row["timestamp_utc"]),
                "temperature_c": row.get("temperature_c"),
                "precipitation_mm": row.get("precipitation_mm"),
                "wind_speed_ms": row.get("wind_speed_ms"),
                "pressure_hpa": row.get("pressure_hpa"),
                "relative_humidity_pct": row.get("relative_humidity_pct"),
                "source": "dwd",
            },
            keys=[
                "dwd_station_id",
                "timestamp_utc",
                "temperature_c",
                "precipitation_mm",
                "wind_speed_ms",
                "pressure_hpa",
                "relative_humidity_pct",
                "source",
            ],
        ),
        axis=1,
    )

    return out