from pathlib import Path
import pandas as pd


OUTPUT_DIR = Path("artifacts")


def main():
    fold_metrics = pd.read_csv(OUTPUT_DIR / "gauge_24h_backtest_fold_metrics.csv")
    predictions = pd.read_csv(OUTPUT_DIR / "gauge_24h_backtest_predictions.csv")
    station_metrics = pd.read_csv(OUTPUT_DIR / "gauge_24h_backtest_station_metrics.csv")
    fold_station_metrics = pd.read_csv(OUTPUT_DIR / "gauge_24h_backtest_fold_station_metrics.csv")
    feature_importance = pd.read_csv(OUTPUT_DIR / "gauge_24h_backtest_feature_importance.csv")

    print("\n=== Fold metrics ===")
    print(fold_metrics[[
        "fold", "mae", "rmse", "r2", "mean_residual",
        "p90_abs_error", "max_abs_error", "test_start", "test_end"
    ]].to_string(index=False))

    print("\n=== Worst folds by RMSE ===")
    print(
        fold_metrics.sort_values("rmse", ascending=False)[[
            "fold", "rmse", "mae", "r2", "test_start", "test_end"
        ]].head(5).to_string(index=False)
    )

    print("\n=== Worst stations overall by RMSE ===")
    print(
        station_metrics.sort_values("rmse", ascending=False)[[
            "station_name", "rows", "rmse", "mae", "mean_residual", "p90_abs_error"
        ]].head(10).to_string(index=False)
    )

    print("\n=== Worst fold-station combinations by RMSE ===")
    print(
        fold_station_metrics.sort_values("rmse", ascending=False)[[
            "fold", "station_name", "rows", "rmse", "mae", "mean_residual", "p90_abs_error"
        ]].head(15).to_string(index=False)
    )

    print("\n=== Largest individual absolute errors ===")
    print(
        predictions.sort_values("abs_error", ascending=False)[[
            "fold", "station_name", "timestamp_utc", "target_value_t_plus_24h",
            "prediction", "residual", "abs_error"
        ]].head(20).to_string(index=False)
    )

    print("\n=== Top features by permutation importance ===")
    print(
        feature_importance[[
            "feature", "importance_mean", "importance_std"
        ]].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()