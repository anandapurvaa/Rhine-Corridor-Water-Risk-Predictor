CREATE SCHEMA IF NOT EXISTS `rhine-corridor-navigator.mlops`;

CREATE TABLE IF NOT EXISTS
  `rhine-corridor-navigator.mlops.stage_events`
(
  event_id STRING NOT NULL,
  run_id STRING NOT NULL,
  job_type STRING NOT NULL,
  stage_name STRING NOT NULL,

  started_at_utc TIMESTAMP NOT NULL,
  ended_at_utc TIMESTAMP,
  duration_seconds FLOAT64,

  status STRING NOT NULL,
  error_type STRING,
  error_message STRING,

  rows_read INT64,
  rows_written INT64,
  station_count INT64,
  table_name STRING,
  metadata_json STRING,

  created_at_utc TIMESTAMP NOT NULL
)
PARTITION BY DATE(started_at_utc)
CLUSTER BY run_id, stage_name, status
OPTIONS (
  description = 'Stage-level audit events for Cloud Run ML pipeline executions.'
);