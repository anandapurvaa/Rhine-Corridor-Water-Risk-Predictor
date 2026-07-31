# RheinKorridor Sentinel

RheinKorridor Sentinel is a water-risk and corridor-intelligence project focused on forecasting gauge behavior across the Rhine corridor using structured machine learning, external data enrichment, and reproducible experimentation. The project combines API-based ingestion, curated storage, supervised dataset construction, and forecasting experiments to identify a modeling strategy that performs well across multiple stations while remaining operationally simple.

## Project goals

The current modeling objective is to forecast `target_value_t_plus_24h` for gauge-related time series using a 24-hour horizon. The broader project goal is to build a strong forecasting and risk-monitoring foundation that can later support operational analytics, station-level diagnostics, and downstream applications.

Key goals so far:

- Build a reproducible ingestion-to-modeling pipeline.
- Pull and harmonize data from multiple upstream sources.
- Store curated data in BigQuery.
- Create a supervised learning table for 24-hour-ahead prediction.
- Establish a strong baseline model.
- Improve performance through disciplined experimentation rather than ad hoc tuning.
- Preserve experiment history and decisions for reproducibility.

## High-level pipeline

The work completed so far follows this sequence:

1. API ingestion and source acquisition.
2. Data cleaning, harmonization, and curation.
3. BigQuery storage and table organization.
4. Construction of a supervised dataset for 24-hour forecasting.
5. Baseline model training and backtesting.
6. Feature ablation experiments.
7. Regime-aware feature experiments.
8. Global residual-correction experiments.
9. Targeted station residual experiments.
10. Station clustering and pooled-by-cluster modeling.
11. Cluster refinement and final comparison.

## Data and storage flow

At a high level, the project uses external source data, transforms it into curated tables, and then prepares a supervised learning table used for modeling.

Typical flow:

- Raw or source data enters from external APIs and supporting sources.
- Data is cleaned and standardized.
- Curated tables are stored in BigQuery.
- A supervised table is built for model training and evaluation.
- Modeling scripts load data from BigQuery and write outputs to local `artifacts/`.

The main supervised table used in this phase is:

- `supervised_gauge_24h_multisource`

The target used in modeling is:

- `target_value_t_plus_24h`

## Modeling approach

The core estimator family used throughout the experiments is `HistGradientBoostingRegressor` from scikit-learn. This means the experiments did **not** change the underlying estimator family each time; instead, they changed the feature set, residual-correction logic, or the grouping strategy used to train the same supervised learner.

Why this is important:

- The “cluster model” experiments are still based on `HistGradientBoostingRegressor`.
- The difference is not the estimator itself, but how the training data is partitioned.
- The project therefore compares **global pooled training** versus **clustered pooled training**, not one estimator versus a completely different one.

This is a useful and legitimate design choice for structured tabular forecasting.

## Baseline setup

The initial strong baseline was a lean gauge forecasting model trained on structured temporal, weather, lag, and rolling features.

Main feature categories included:

- Gauge target and lag features.
- Rolling summary features.
- Weather features.
- Time-derived calendar features.
- Station and source categorical context.

The baseline was evaluated with rolling time-based backtesting instead of a random split, which is essential for forecasting validity.

## Feature ablation results

A feature ablation was run to understand which groups of features were actually helping.

The tested variants included:

- `full`
- `no_station_id`
- `lean`
- `no_target_value`

Observed result summary:

- Removing `station_id` did not materially hurt performance.
- The lean model stayed very competitive.
- Removing `target_value` severely damaged performance.

Interpretation:

- The model depends heavily on recent target behavior.
- Some categorical identity features were less important than expected.
- Simpler feature sets could perform nearly as well as more complex ones.

This gave confidence that the baseline could be simplified without large performance loss.

## Lean backtest baseline

The lean backtest established the practical benchmark for all later comparisons.

Lean backtest summary:

- Mean MAE: `2.4400417964939782`
- Mean RMSE: `6.051264621704021`
- RMSE standard deviation: `3.9712050749920382`
- Mean \(R^2\): `0.9977707576714604`
- Rows scored: `4080`

This became the main benchmark against which all later experiments were compared.

## Regime feature experiment

The next experiment added regime-aware features, such as trend and volatility-oriented summaries, to try to capture nonstationary behavior across stations and time windows.

What was tested:

- Short-vs-long trend features.
- Volatility and range features.
- Acceleration-style changes.
- Weather trend proxies.

Outcome:

- The regime feature set was informative as an exploration step.
- It did not beat the lean baseline convincingly enough to become the preferred model.
- One regime feature was entirely null in practice and generated repeated warnings, which also showed the value of checking feature availability carefully.

Decision:

- Keep the regime experiment as a documented branch.
- Do not promote it to the mainline model.

## Station residual experiment

A two-stage residual-correction experiment was then tested.

Design:

1. Train the global lean-style model.
2. Predict residual corrections on top of the global prediction.

Purpose:

- Reduce station-specific systematic bias without replacing the global model.

Outcome:

- This helped slightly in aggregate.
- It was not strong enough to clearly improve the worst-case behavior.
- Some stations improved while others worsened.

Decision:

- Keep as a valid experiment record.
- Do not use as the mainline solution.

## Targeted station residual experiment

The next variant restricted residual correction to the worst-performing stations instead of applying it broadly.

Purpose:

- Keep the global model for most stations.
- Apply local correction only where repeated high error suggested station-specific issues.

Outcome:

- Mean metrics changed only marginally.
- Worst-fold performance did not improve enough.
- The same difficult stations remained near the top of the error rankings.

Decision:

- Useful as a diagnostic.
- Not chosen as the final strategy.

## Station metadata snapshot

To support clustering, a station-level metadata snapshot was created from the supervised table.

This file summarizes each station using aggregated descriptors such as:

- Mean and variation of target values.
- Distance or corridor-position proxy.
- Coverage duration.
- Rolling variability summaries.
- Dominant source and time series identity.

Generated artifact:

- `artifacts/station_metadata_snapshot.csv`

This snapshot became the input for station clustering.

## First cluster plan

A first clustering pass was created using the worst-performing stations and their metadata summaries. The purpose was to test whether a grouped training strategy would outperform a single global pooled model.

Initial interpretation:

- Some stations naturally formed shared groups.
- One difficult station remained isolated.
- The grouping was useful enough to justify pooled-by-cluster backtesting.

This was the first point where the experiments showed a more structural improvement path beyond global feature tuning.

## Cluster pooled modeling

The clustered pooled model still used `HistGradientBoostingRegressor`, but it trained separate pooled models for station groups rather than one single model for all stations.

This is the key distinction:

- **Global pooled model**: one model for all stations.
- **Cluster pooled model**: one model per cluster, still using the same estimator family.

The first cluster version outperformed the lean baseline overall and became the first clearly better modeling family tested in the project.

## Refined cluster plan (v2)

The first clustering was then refined after inspecting which stations still behaved inconsistently inside the same pooled group.

The refined clustering produced a better separation of difficult station types, especially for stations that appeared structurally different from neighboring groups.

This led to the final tested version:

- `station_cluster_plan_v2.csv`

## Best result so far

The refined cluster-pooled backtest is currently the strongest model family tested in the project.

Best current summary:

- Mean MAE: `2.4110542399624117`
- Mean RMSE: `5.938162890501346`
- RMSE standard deviation: `4.020634885656827`
- Mean \(R^2\): `0.997809177156386`
- Worst fold RMSE: `13.939132403654272`
- Rows scored: `4080`

Compared with the lean baseline, this version improved:

- Mean RMSE.
- Mean MAE.
- Mean \(R^2\).
- Worst-fold RMSE slightly.

It also materially improved some previously problematic stations, although a few stubborn outliers still remain.

## Current modeling conclusion

At this stage, the strongest documented conclusion is:

- The lean global model is the best simple baseline.
- The refined clustered pooled model is the best-performing experimental strategy so far.
- The estimator family remains `HistGradientBoostingRegressor`.
- The main performance gain came from improving **grouping strategy**, not from switching to a new estimator family.

This means the project is doing the right thing structurally: the experiments moved from global feature tuning toward a better data-partitioning strategy when global refinements stopped yielding meaningful gains.

## Remaining hard stations

Even after clustering improvements, a few stations still appear structurally difficult and may need dedicated treatment later.

Examples include:

- `Koblenz UP`
- `RHEINE UNTERSCHLEUSE`

These stations may eventually require one of the following:

- Dedicated local models.
- Specialized feature engineering.
- More accurate station metadata.
- Hydrologic or corridor-aware grouping logic.

These steps were intentionally deferred until after proving that grouped pooled modeling could beat the global baseline.

## Files created during the process

Important modeling and documentation files created during this phase include:

- `modeling/build_station_metadata_snapshot.py`
- `modeling/build_station_cluster_plan.py`
- `modeling/refine_station_clusters.py`
- `modeling/backtest_gauge_24h_cluster_models.py`
- `modeling/analyze_backtest_cluster_models_gauge_24h.py`

Additional experiment files were also created for:

- Regime-aware features.
- Residual correction.
- Targeted residual correction.

These should be archived as experiments rather than deleted, because they document what was tested and why certain paths were not chosen.

## Recommended project organization

A clean structure going forward is:

```text
modeling/
  backtest_gauge_24h_lean.py
  analyze_backtest_gauge_24h_lean.py
  backtest_gauge_24h_cluster_models.py
  analyze_backtest_cluster_models_gauge_24h.py
  build_station_metadata_snapshot.py
  build_station_cluster_plan.py
  refine_station_clusters.py
  EXPERIMENT_LOG.md

modeling/experiments/
  backtest_gauge_24h_regime.py
  analyze_backtest_regime_gauge_24h.py
  backtest_gauge_24h_station_residual.py
  analyze_backtest_station_residual_gauge_24h.py
  backtest_gauge_24h_station_residual_targeted.py
  analyze_backtest_targeted_station_residual_gauge_24h.py
```

## Reproducibility notes

To keep this work reproducible, each major modeling milestone should preserve:

- The exact training script.
- The preprocessing logic.
- The supervised table name.
- The cluster plan version used.
- The output metrics.
- The git commit associated with the result.
- The Python package environment.

This is especially important in ML projects because keeping only model weights or a single script is not enough to fully reproduce a result.

## Suggested next steps

The best immediate next steps are:

1. Keep the lean model as the official simple baseline.
2. Keep the refined clustered pooled model as the best current performer.
3. Archive rejected or non-winning experiments in a dedicated `experiments/` folder.
4. Write an experiment log summarizing each tested approach and its outcome.
5. Consider a later dedicated local model for persistent outlier stations.
6. Begin integrating the chosen model family into the broader RheinKorridor Sentinel workflow.

## Status summary

Current status of the project so far:

- Ingestion and curation pipeline established.
- Supervised dataset for 24-hour forecasting established.
- Baseline modeling completed.
- Multiple structured experiments completed.
- Clustered pooled modeling validated.
- Refined cluster plan selected.
- Best current model family identified.
