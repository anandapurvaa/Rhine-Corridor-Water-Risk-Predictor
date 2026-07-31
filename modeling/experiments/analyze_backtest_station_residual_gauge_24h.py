from pathlib import Path
import pandas as pd


OUTPUT_DIR = Path("artifacts")


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def main():
    lean_fold = safe_read(OUTPUT_DIR / "gauge_24h_lean_fold_metrics.csv")
    residual_fold = safe_read(OUTPUT_DIR / "gauge_24h_station_residual_fold_metrics.csv")

    lean_station = safe_read(OUTPUT_DIR / "gauge_24h_backtest_station_metrics.csv")
    residual_station = safe_read(OUTPUT_DIR / "gauge_24h_station_residual_station_metrics.csv")

    print("\n=== Lean vs Station Residual fold comparison ===")
    comp = lean_fold[["fold", "rmse", "mae", "r2"]].merge(
        residual_fold[["fold", "rmse", "mae", "r2"]],
        on="fold",
        suffixes=("_lean", "_station_residual")
    )
    comp["rmse_delta"] = comp["rmse_station_residual"] - comp["rmse_lean"]
    comp["mae_delta"] = comp["mae_station_residual"] - comp["mae_lean"]
    print(comp.to_string(index=False))

    print("\n=== Worst fold comparison ===")
    print("Lean worst RMSE:", float(lean_fold["rmse"].max()))
    print("Station residual worst RMSE:", float(residual_fold["rmse"].max()))

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
            "model": "station_residual",
            "mean_rmse": float(residual_fold["rmse"].mean()),
            "std_rmse": float(residual_fold["rmse"].std(ddof=0)),
            "mean_mae": float(residual_fold["mae"].mean()),
            "mean_r2": float(residual_fold["r2"].mean()),
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

    print("\n=== Worst stations overall: station residual ===")
    print(
        residual_station[["station_name", "rows", "rmse", "mae", "mean_residual", "p90_abs_error"]]
        .sort_values("rmse", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()