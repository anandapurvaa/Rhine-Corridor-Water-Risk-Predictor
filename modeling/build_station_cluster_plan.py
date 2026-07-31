from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_STATION_METRICS = OUTPUT_DIR / "gauge_24h_backtest_station_metrics.csv"
INPUT_DATA = "supervised_gauge_24h_multisource"


def main():
    station_metrics = pd.read_csv(INPUT_STATION_METRICS)

    candidate_stations = (
        station_metrics.sort_values("rmse", ascending=False)
        .head(12)["station_name"]
        .astype(str)
        .tolist()
    )

    # This file is intentionally simple: it helps you define the first clustering plan.
    # Replace/extend the station metadata columns below once you confirm they exist in BigQuery.
    df = pd.read_csv(OUTPUT_DIR / "station_metadata_snapshot.csv")

    if "station_name" not in df.columns:
        raise ValueError("station_metadata_snapshot.csv must include station_name")

    target_df = df[df["station_name"].astype(str).isin(candidate_stations)].copy()

    numeric_cols = [c for c in ["distance_km", "mean_flow", "mean_level", "corridor_position"] if c in target_df.columns]
    if not numeric_cols:
        raise ValueError("No clustering features available in station_metadata_snapshot.csv")

    pipeline = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            numeric_cols,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("kmeans", KMeans(n_clusters=min(3, len(target_df)), random_state=42, n_init=10)),
        ]
    )

    cluster_labels = pipeline.fit_predict(target_df[numeric_cols])

    out = target_df[["station_name"]].copy()
    out["cluster"] = cluster_labels
    out.to_csv(OUTPUT_DIR / "station_cluster_plan.csv", index=False)

    print(out.sort_values(["cluster", "station_name"]).to_string(index=False))


if __name__ == "__main__":
    main()