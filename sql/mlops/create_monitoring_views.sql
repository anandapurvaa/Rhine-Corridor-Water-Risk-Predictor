-- Rhine Corridor Water Risk: MLOps monitoring views

CREATE OR REPLACE VIEW
  `rhine-corridor-navigator.mlops.v_latest_pipeline_health`
AS
SELECT
  run_id,
  job_type,
  cloud_run_job,
  project_id,
  region,
  status,
  started_at_utc,
  ended_at_utc,
  ROUND(duration_seconds, 2) AS duration_seconds,
  rows_ingested,
  rows_predicted,
  stations_processed,
  model_version,
  input_split,
  error_type,
  error_message,
  data_window_start_utc,
  data_window_end_utc,
  created_at_utc
FROM `rhine-corridor-navigator.mlops.pipeline_runs`
QUALIFY ROW_NUMBER() OVER (
  ORDER BY started_at_utc DESC
) = 1;


CREATE OR REPLACE VIEW
  `rhine-corridor-navigator.mlops.v_latest_pipeline_health_by_job`
AS
SELECT
  run_id,
  job_type,
  cloud_run_job,
  project_id,
  region,
  status,
  started_at_utc,
  ended_at_utc,
  ROUND(duration_seconds, 2) AS duration_seconds,
  rows_ingested,
  rows_predicted,
  stations_processed,
  model_version,
  input_split,
  error_type,
  error_message,
  data_window_start_utc,
  data_window_end_utc,
  created_at_utc
FROM `rhine-corridor-navigator.mlops.pipeline_runs`
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY job_type
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
  measured_at_utc,
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
  table_name,
  error_type,
  error_message,
  metadata_json,
  started_at_utc,
  ended_at_utc
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
  COUNTIF(
    LOWER(status) IN (
      'success',
      'succeeded',
      'ok',
      'pass'
    )
  ) AS successful_runs,
  COUNTIF(
    LOWER(status) IN (
      'failed',
      'failure',
      'error'
    )
  ) AS failed_runs,
  ROUND(
    AVG(duration_seconds),
    2
  ) AS average_duration_seconds,
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
  COUNTIF(
    LOWER(status) IN (
      'pass',
      'passed',
      'success',
      'ok'
    )
  ) AS passed_metrics,
  COUNTIF(
    LOWER(status) IN (
      'fail',
      'failed',
      'failure',
      'error'
    )
  ) AS failed_metrics,
  ARRAY_AGG(
    IF(
      LOWER(status) IN (
        'fail',
        'failed',
        'failure',
        'error'
      ),
      metric_name,
      NULL
    )
    IGNORE NULLS
  ) AS failed_metric_names,
  MAX(measured_at_utc) AS last_measured_at_utc
FROM `rhine-corridor-navigator.mlops.data_quality_metrics`
GROUP BY run_id
ORDER BY last_measured_at_utc DESC;


CREATE OR REPLACE VIEW
  `rhine-corridor-navigator.mlops.v_latest_evaluation_health`
AS
SELECT
  model_version,
  split_name,
  evaluated_at_utc,
  ROUND(mae, 2) AS mae,
  ROUND(rmse, 2) AS rmse,
  ROUND(mbe, 2) AS mbe,
  mae_by_station,
  'available' AS evaluation_status
FROM `rhine-corridor-navigator.mlops.model_evaluations`
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY split_name, model_version
  ORDER BY evaluated_at_utc DESC
) = 1;