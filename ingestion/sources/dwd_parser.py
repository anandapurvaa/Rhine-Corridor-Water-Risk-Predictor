import io
import zipfile

import pandas as pd

from ingestion.common.hashing import stable_record_hash


def parse_dwd_zip_bytes(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        txt_names = [
            name for name in zf.namelist()
            if name.lower().startswith("produkt_") and name.lower().endswith(".txt")
        ]
        if not txt_names:
            raise ValueError("No produkt_*.txt file found in DWD zip archive")

        with zf.open(txt_names[0]) as f:
            df = pd.read_csv(f, sep=";", dtype=str, encoding="latin1")

    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    df = df.replace({"": None, "-999": None, "-999.0": None, "####": None})
    return df


def normalize_station_id(value: str) -> str:
    return str(value).strip().zfill(5)


def parse_dwd_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), format="%Y%m%d%H", utc=True, errors="coerce")


def _hash_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


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

    out["temperature_c"] = pd.Series([None] * len(df), dtype="float64")
    out["precipitation_mm"] = pd.Series([None] * len(df), dtype="float64")
    out["wind_speed_ms"] = pd.Series([None] * len(df), dtype="float64")
    out["pressure_hpa"] = pd.Series([None] * len(df), dtype="float64")
    out["relative_humidity_pct"] = pd.Series([None] * len(df), dtype="float64")

    if variable_family == "air_temperature":
        if "TT_TU" in df.columns:
            out["temperature_c"] = pd.to_numeric(df["TT_TU"], errors="coerce")
        if "RF_TU" in df.columns:
            out["relative_humidity_pct"] = pd.to_numeric(df["RF_TU"], errors="coerce")

    elif variable_family == "precipitation":
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

    out["source_record_hash"] = [
        stable_record_hash(
            {
                "dwd_station_id": sid,
                "timestamp_utc": _hash_value(ts),
                "temperature_c": _hash_value(temp),
                "precipitation_mm": _hash_value(prec),
                "wind_speed_ms": _hash_value(wind),
                "pressure_hpa": _hash_value(press),
                "relative_humidity_pct": _hash_value(rh),
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
        )
        for sid, ts, temp, prec, wind, press, rh in zip(
            out["dwd_station_id"].tolist(),
            out["timestamp_utc"].tolist(),
            out["temperature_c"].tolist(),
            out["precipitation_mm"].tolist(),
            out["wind_speed_ms"].tolist(),
            out["pressure_hpa"].tolist(),
            out["relative_humidity_pct"].tolist(),
        )
    ]

    return out