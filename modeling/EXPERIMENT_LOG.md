# Modeling Experiment Log

## Baseline

### Lean global pooled model
- Estimator: `HistGradientBoostingRegressor`
- Strategy: one global pooled model across stations
- Result: strong baseline and comparison point for all later experiments
- Decision: keep as official simple baseline

## Feature experiments

### Regime-aware feature expansion
- Goal: capture time-varying trend and volatility regimes
- Result: did not outperform lean strongly enough; one feature was entirely null
- Decision: archive as experiment

## Residual experiments

### Global residual correction
- Goal: correct station-specific bias after global prediction
- Result: small gains, inconsistent station effects
- Decision: archive as experiment

### Targeted station residual correction
- Goal: apply correction only to worst stations
- Result: small average benefit, weak worst-fold improvement
- Decision: archive as experiment

## Cluster experiments

### Cluster pooled model v1
- Goal: replace one global pooled model with pooled models per station cluster
- Result: first clearly better family than lean baseline
- Decision: refine cluster assignment

### Cluster pooled model v2
- Goal: improve cluster homogeneity
- Result: best overall backtest so far
- Decision: keep as best current modeling strategy

## Current preferred models

### Simple baseline
- `lean` global pooled model

### Best current performer
- Cluster pooled model using `station_cluster_plan_v2.csv`

## Notes

- “Cluster model” here means clustered pooled training strategy, not a different estimator family.
- The estimator remained `HistGradientBoostingRegressor` throughout the winning modeling path.
- Future work should target persistent outlier stations with dedicated local treatment only if justified by backtest evidence.