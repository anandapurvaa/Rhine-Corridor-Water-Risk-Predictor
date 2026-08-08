CREATE SCHEMA IF NOT EXISTS `rhine-corridor-navigator.mlops`;

CREATE TABLE IF NOT EXISTS
  `rhine-corridor-navigator.mlops.pipeline_runs`
(
  run_id STRING NOT NULL,
  job_type STRING NOT NULL,
  cloud_run_job STRING,
  project_id STRING,
  region STRING,

  started_at_utc TIMESTAMP NOT NULL,
  ended_at_utc TIMESTAMP,
  duration_seconds FLOAT64,

  status STRING NOT NULL,
  error_type STRING,
  error_message STRING,

  input_split STRING,
  model_version STRING,

  rows_ingested INT64,
  rows_predicted INT64,
  stations_processed INT64,

  data_window_start_utc TIMESTAMP,
  data_window_end_utc TIMESTAMP,

  created_at_utc TIMESTAMP NOT NULL
)
PARTITION BY DATE(started_at_utc)
CLUSTER BY job_type, status
OPTIONS (
  description = 'One audit record for every Cloud Run ML pipeline execution.'
);