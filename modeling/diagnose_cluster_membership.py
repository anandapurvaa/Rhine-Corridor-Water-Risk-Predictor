from __future__ import annotations

from modeling.backtest_walkforward_gauge_24h_cluster import (
    MIN_CLUSTER_STATIONS,
    MIN_CLUSTER_TRAIN_ROWS,
    iter_walkforward_splits,
    load_cluster_plan,
    prepare_dataframe,
    summarize_cluster_coverage,
)


def main():
    df, target_column = prepare_dataframe()
    cluster_df = load_cluster_plan()

    print(f"Target column: {target_column}")
    print(f"MIN_CLUSTER_TRAIN_ROWS threshold: {MIN_CLUSTER_TRAIN_ROWS}")
    print(f"MIN_CLUSTER_STATIONS threshold: {MIN_CLUSTER_STATIONS}")
    print(f"Total clusters defined: {cluster_df['cluster'].nunique()}")
    print(f"Stations per cluster:\n{cluster_df.groupby('cluster')['station_name'].apply(list).to_string()}")
    print("-" * 80)

    for fold_num, (origin, train_df, test_df) in enumerate(iter_walkforward_splits(df), start=1):
        coverage, summary = summarize_cluster_coverage(train_df, cluster_df)

        print(f"Fold {fold_num} | origin={origin} | train_rows={len(train_df)}")
        print(coverage.to_string(index=False))
        print(
            "Summary:",
            {
                "planned_clusters": summary["planned_clusters"],
                "matched_clusters": summary["matched_clusters"],
                "eligible_clusters": summary["eligible_clusters"],
                "unmatched_clusters": summary["unmatched_clusters"],
            },
        )
        print("-" * 80)


if __name__ == "__main__":
    main()