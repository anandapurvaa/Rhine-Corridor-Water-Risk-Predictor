$ErrorActionPreference = "Stop"

$PROJECT_ID = "rhine-corridor-navigator"
$JOB_NAME = "rhine-daily-pipeline"

gcloud config set project $PROJECT_ID


$PIPELINE_FAILURE_FILTER = @'
resource.type="cloud_run_job"
resource.labels.job_name="rhine-daily-pipeline"
(
  jsonPayload.message="pipeline_failed"
  OR textPayload:"pipeline_failed"
)
'@

gcloud logging metrics create rhine_daily_pipeline_failures `
    --project=$PROJECT_ID `
    --description="Count of failed Rhine daily pipeline executions." `
    --log-filter="$PIPELINE_FAILURE_FILTER"


$DATA_QUALITY_FAILURE_FILTER = @'
resource.type="cloud_run_job"
resource.labels.job_name="rhine-daily-pipeline"
jsonPayload.message="data_quality_checks_completed"
jsonPayload.data_quality_status="fail"
'@

gcloud logging metrics create rhine_daily_pipeline_quality_failures `
    --project=$PROJECT_ID `
    --description="Count of failed data-quality checks." `
    --log-filter="$DATA_QUALITY_FAILURE_FILTER"


$PIPELINE_SUCCESS_FILTER = @'
resource.type="cloud_run_job"
resource.labels.job_name="rhine-daily-pipeline"
(
  jsonPayload.message="pipeline_completed"
  OR textPayload:"pipeline_completed"
)
'@

gcloud logging metrics create rhine_daily_pipeline_successes `
    --project=$PROJECT_ID `
    --description="Count of successful Rhine daily pipeline executions." `
    --log-filter="$PIPELINE_SUCCESS_FILTER"