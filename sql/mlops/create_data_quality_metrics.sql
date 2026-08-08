CREATE TABLE IF NOT EXISTS
  `rhine-corridor-navigator.mlops.data_quality_metrics`
(
  metric_id STRING NOT NULL,
  run_id STRING NOT NULL,
  metric_name STRING NOT NULL,
  metric_scope STRING NOT NULL,

  metric_value FLOAT64,
  threshold_value FLOAT64,
  status STRING NOT NULL,

  measured_at_utc TIMESTAMP NOT NULL,
  details_json STRING
)
PARTITION BY DATE(measured_at_utc)
CLUSTER BY metric_name, status, run_id
OPTIONS (
  description = 'Data quality and prediction quality metrics for ML pipeline runs.'
);