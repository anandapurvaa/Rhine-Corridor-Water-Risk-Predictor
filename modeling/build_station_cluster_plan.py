from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from modeling.data_loader import load_bigquery_table
from modeling.schemas import TABLE_NAME, TARGET_COLUMN

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUTPUT_DIR / "station_cluster_plan_v2.csv"
SUMMARY_PATH = OUTPUT_DIR / "station_cluster_plan_v2_summary.json"

REQUESTED_COLUMNS = [
    "station_name",
    "timestamp_utc",
    TARGET_COLUMN,
    "temperature_c",
    "precipitation_mm",
    "wind_speed_ms",
    "pressure_hpa",
    "relative_humidity_pct",
]

def main():
    df = load_bigquery_table(
        TABLE_NAME,
        columns=REQUESTED_COLUMNS,
        order_by="station_name, timestamp_utc",
        allow_missing_columns=True,
    )

    required = [c for c in ["station_name", "timestamp_utc", TARGET_COLUMN] if c not in df.columns]
    if required:
        raise ValueError(f"Missing required columns for cluster plan: {required}")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    available_numeric = [TARGET_COLUMN]
    for c in ["temperature_c", "precipitation_mm", "wind_speed_ms", "pressure_hpa", "relative_humidity_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            available_numeric.append(c)

    if len(available_numeric) < 2:
        raise ValueError("Not enough numeric columns available to build cluster plan.")

    agg = (
        df.groupby("station_name")[available_numeric]
        .agg(["mean", "std", "min", "max", "median"])
    )

    agg.columns = [f"{c1}_{c2}" for c1, c2 in agg.columns]
    agg = agg.reset_index()

    feature_cols = [c for c in agg.columns if c != "station_name"]
    X = agg[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    if X.shape[0] < 2:
        raise ValueError("Need at least 2 stations to cluster.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_components = min(3, Xs.shape[1], Xs.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(Xs)

    n_clusters = 4 if len(agg) >= 8 else max(2, len(agg) // 2)
    n_clusters = min(n_clusters, len(agg))

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster = km.fit_predict(Z)

    out = agg[["station_name"]].copy()
    out["cluster"] = cluster.astype(int)
    out.to_csv(OUT_PATH, index=False)

    summary = {
        "stations": int(len(out)),
        "clusters": int(n_clusters),
        "available_numeric_columns": available_numeric,
        "pca_components": int(n_components),
        "output": str(OUT_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()