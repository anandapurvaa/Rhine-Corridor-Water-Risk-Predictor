from pathlib import Path
import pandas as pd


OUTPUT_DIR = Path("artifacts")


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def main():
    lean_fold = safe_read(OUTPUT_DIR / "gauge_24h_lean_fold_metrics.csv")
    cluster_fold = safe_read(OUTPUT_DIR / "gauge_24h_cluster_models_fold_metrics.csv")

    lean_station = safe_read(OUTPUT_DIR / "gauge_24h_backtest_station_metrics.csv")
    cluster_station = safe_read(OUTPUT_DIR / "gauge_24h_cluster_models_station_metrics.csv")

    print("\n=== Lean vs Cluster Models fold comparison ===")
    comp = lean_fold[["fold", "rmse", "mae", "r2"]].merge(
        cluster_fold[["fold", "rmse", "mae", "r2", "cluster_models_used"]],
        on="fold",
        suffixes=("_lean", "_cluster")
    )
    comp["rmse_delta"] = comp["rmse_cluster"] - comp["rmse_lean"]
    comp["mae_delta"] = comp["mae_cluster"] - comp["mae_lean"]
    print(comp.to_string(index=False))

    print("\n=== Worst fold comparison ===")
    print("Lean worst RMSE:", float(lean_fold["rmse"].max()))
    print("Cluster models worst RMSE:", float(cluster_fold["rmse"].max()))

    print("\n=== Mean / std comparison ===")
    summary = pd.DataFrame([
        {
            "model": "lean",
            "mean_rmse": float(lean_fold["rmse"].mean()),
            "std_rmse": float(lean_fold["rmse"].std(ddof=0)),
            "mean_mae": float(lean_fold["mae"].mean()),
            "mean_r2": float(lean_fold["r2"].mean()),
        },
        {
            "model": "cluster_models",
            "mean_rmse": float(cluster_fold["rmse"].mean()),
            "std_rmse": float(cluster_fold["rmse"].std(ddof=0)),
            "mean_mae": float(cluster_fold["mae"].mean()),
            "mean_r2": float(cluster_fold["r2"].mean()),
        },
    ])
    print(summary.to_string(index=False))

    print("\n=== Worst stations overall: lean ===")
    print(
        lean_station[["station_name", "rows", "rmse", "mae", "mean_residual", "p90_abs_error"]]
        .sort_values("rmse", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\n=== Worst stations overall: cluster models ===")
    print(
        cluster_station[["station_name", "rows", "rmse", "mae", "mean_residual", "p90_abs_error"]]
        .sort_values("rmse", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()