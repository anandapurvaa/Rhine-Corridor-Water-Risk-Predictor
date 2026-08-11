CREATE OR REPLACE VIEW
  `rhine-corridor-navigator.mlops.v_latest_pipeline_health`
AS
SELECT
  run_id,
  job_type,
  cloud_run_job,
  status,
  FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', started_at_utc) AS started_at_utc,
  FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', ended_at_utc) AS ended_at_utc,
  ROUND(duration_seconds, 2) AS duration_seconds,
  rows_ingested,
  rows_predicted,
  stations_processed,
  model_version,
  input_split,
  error_type,
  error_message,
  FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', created_at_utc) AS created_at_utc
FROM `rhine-corridor-navigator.mlops.pipeline_runs`
QUALIFY ROW_NUMBER() OVER (
  ORDER BY started_at_utc DESC
) = 1;


CREATE OR REPLACE VIEW
  `rhine-corridor-navigator.mlops.v_latest_quality_health`
AS
SELECT
  run_id,
  metric_name,
  metric_scope,
  ROUND(metric_value, 2) AS metric_value,
  ROUND(threshold_value, 2) AS threshold_value,
  status,
  FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', measured_at_utc) AS measured_at_utc,
  details_json
FROM `rhine-corridor-navigator.mlops.data_quality_metrics`
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY metric_name, metric_scope
  ORDER BY measured_at_utc DESC
) = 1;


CREATE OR REPLACE VIEW
  `rhine-corridor-navigator.mlops.v_latest_stage_health`
AS
SELECT
  run_id,
  job_type,
  stage_name,
  status,
  ROUND(duration_seconds, 2) AS duration_seconds,
  rows_read,
  rows_written,
  station_count,
  metadata_json,
  FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', started_at_utc) AS started_at_utc,
  FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', ended_at_utc) AS ended_at_utc
FROM `rhine-corridor-navigator.mlops.stage_events`
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY stage_name
  ORDER BY started_at_utc DESC
) = 1;


CREATE OR REPLACE VIEW
  `rhine-corridor-navigator.mlops.v_daily_pipeline_health`
AS
SELECT
  DATE(started_at_utc) AS execution_date,
  COUNT(*) AS total_runs,
  COUNTIF(status = 'success') AS successful_runs,
  COUNTIF(status = 'failed') AS failed_runs,
  ROUND(AVG(duration_seconds), 2) AS average_duration_seconds,
  MAX(rows_ingested) AS maximum_rows_ingested,
  MAX(rows_predicted) AS maximum_rows_predicted,
  MAX(stations_processed) AS maximum_stations_processed
FROM `rhine-corridor-navigator.mlops.pipeline_runs`
GROUP BY execution_date
ORDER BY execution_date DESC;


CREATE OR REPLACE VIEW
  `rhine-corridor-navigator.mlops.v_latest_run_quality_summary`
AS
SELECT
  run_id,
  COUNT(*) AS metric_count,
  COUNTIF(status = 'pass') AS passed_metrics,
  COUNTIF(status = 'fail') AS failed_metrics,
  ARRAY_AGG(
    IF(status = 'fail', metric_name, NULL)
    IGNORE NULLS
  ) AS failed_metric_names,
  FORMAT_TIMESTAMP('%Y-%m-%d %H:%M', MAX(measured_at_utc)) AS last_measured_at_utc
FROM `rhine-corridor-navigator.mlops.data_quality_metrics`
GROUP BY run_id
ORDER BY MAX(measured_at_utc) DESC;