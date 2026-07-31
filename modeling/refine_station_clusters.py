from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_PATH = OUTPUT_DIR / "station_metadata_snapshot.csv"
STATION_METRICS_PATH = OUTPUT_DIR / "gauge_24h_cluster_models_station_metrics.csv"
OUTPUT_PATH = OUTPUT_DIR / "station_cluster_plan_v2.csv"

TARGET_FEATURES = [
    "distance_km_mean",
    "target_value_mean",
    "target_value_std",
    "target_value_range",
    "coverage_days",
]

def main():
    meta = pd.read_csv(METADATA_PATH)
    metrics = pd.read_csv(STATION_METRICS_PATH)

    meta["station_name"] = meta["station_name"].astype(str)
    metrics["station_name"] = metrics["station_name"].astype(str)

    worst = (
        metrics.sort_values("rmse", ascending=False)
        .head(12)[["station_name", "rmse", "mae"]]
        .copy()
    )

    df = meta.merge(worst, on="station_name", how="inner")

    usable_features = [c for c in TARGET_FEATURES if c in df.columns]
    if len(usable_features) < 2:
        raise ValueError(f"Need at least 2 clustering features, found: {usable_features}")

    n_clusters = 4 if len(df) >= 8 else 3

    pipe = Pipeline(
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
                            usable_features,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=25)),
        ]
    )

    df["cluster_v2"] = pipe.fit_predict(df[usable_features])

    out = df[["station_name", "cluster_v2", "rmse", "mae"] + usable_features].copy()
    out = out.sort_values(["cluster_v2", "rmse"], ascending=[True, False]).reset_index(drop=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(out.to_string(index=False))
    print(f"\nWrote: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()