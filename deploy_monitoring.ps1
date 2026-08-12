$ErrorActionPreference = "Stop"

$ProjectId = if ($env:GCP_PROJECT_ID) {
    $env:GCP_PROJECT_ID
} else {
    "rhine-corridor-navigator"
}

$Region = "europe-west3"
$Repository = "gauge24h"
$Image = "$Region-docker.pkg.dev/$ProjectId/$Repository/gauge24h-monitoring:latest"

$Job = "gauge24h-monitoring"
$SchedulerJob = "gauge24h-monitoring-daily"

$DailyJob = "rhine-daily-pipeline"
$EvaluationJob = "rhine-gauge-24h-evaluation"
$TrainingJob = "gauge24h-train"

$Timezone = "Europe/Berlin"
$SchedulerServiceAccount = "gauge24h-scheduler@$ProjectId.iam.gserviceaccount.com"


function Invoke-Gcloud {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & gcloud @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "gcloud command failed with exit code ${LASTEXITCODE}: gcloud $($Arguments -join ' ')"
    }
}


function Test-CloudRunJob {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $oldPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = "Continue"

        $null = & gcloud run jobs describe $Name `
            --project $ProjectId `
            --region $Region `
            --format="value(name)" 2>&1

        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
}


function Test-SchedulerJob {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $oldPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = "Continue"

        $null = & gcloud scheduler jobs describe $Name `
            --project $ProjectId `
            --location $Region `
            --format="value(name)" 2>&1

        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
}


$envs = @(
    "GCP_PROJECT_ID=$ProjectId",
    "GCP_REGION=$Region",
    "CURATED_DATASET=rhein_curated",
    "MLOPS_DATASET=mlops",
    "PREDICTIONS_TABLE=gauge_24h_production_predictions",
    "EVALUATIONS_TABLE=gauge_24h_prediction_evaluations",
    "QUALITY_TABLE=data_quality_metrics",
    "DAILY_JOB_NAME=$DailyJob",
    "EVALUATION_JOB_NAME=$EvaluationJob",
    "TRAINING_JOB_NAME=$TrainingJob",
    "EXPECTED_STATION_COUNT=19",
    "MIN_PREDICTION_ROWS=19",
    "MAX_PREDICTION_AGE_HOURS=26",
    "MAX_EVALUATION_AGE_HOURS=50",
    "FAIL_ON_MISSING_EVALUATION=false"
) -join ","


Write-Host "Building monitoring image..."

Invoke-Gcloud @(
    "builds",
    "submit",
    "--project",
    $ProjectId,
    "--region",
    $Region,
    "--config",
    "cloudbuild.monitoring.yaml",
    "--substitutions",
    "SHORT_SHA=latest",
    "."
)


Write-Host "Checking Cloud Run job [$Job]..."

if (Test-CloudRunJob -Name $Job) {
    Write-Host "Updating existing Cloud Run job [$Job]..."

    Invoke-Gcloud @(
        "run",
        "jobs",
        "update",
        $Job,
        "--project",
        $ProjectId,
        "--region",
        $Region,
        "--image",
        $Image,
        "--set-env-vars",
        $envs,
        "--max-retries",
        "0",
        "--task-timeout",
        "10m"
    )
}
else {
    Write-Host "Cloud Run job [$Job] does not exist. Creating it..."

    Invoke-Gcloud @(
        "run",
        "jobs",
        "create",
        $Job,
        "--project",
        $ProjectId,
        "--region",
        $Region,
        "--image",
        $Image,
        "--set-env-vars",
        $envs,
        "--max-retries",
        "0",
        "--task-timeout",
        "10m"
    )
}


Write-Host "Executing monitoring smoke test..."

Invoke-Gcloud @(
    "run",
    "jobs",
    "execute",
    $Job,
    "--project",
    $ProjectId,
    "--region",
    $Region,
    "--wait"
)


$RunUri = "https://run.googleapis.com/apis/run.googleapis.com/v1/projects/$ProjectId/locations/$Region/jobs/${Job}:run"


Write-Host "Checking Scheduler job [$SchedulerJob]..."

if (Test-SchedulerJob -Name $SchedulerJob) {
    Write-Host "Updating existing Scheduler job [$SchedulerJob]..."

    Invoke-Gcloud @(
        "scheduler",
        "jobs",
        "update",
        "http",
        $SchedulerJob,
        "--project",
        $ProjectId,
        "--location",
        $Region,
        "--schedule",
        "0 3 * * *",
        "--time-zone",
        $Timezone,
        "--uri",
        $RunUri,
        "--http-method",
        "POST",
        "--oauth-service-account-email",
        $SchedulerServiceAccount,
        "--oauth-token-scope",
        "https://www.googleapis.com/auth/cloud-platform",
        "--description",
        "Run Rhine Gauge 24h monitoring watchdog after daily prediction and evaluation"
    )
}
else {
    Write-Host "Scheduler job [$SchedulerJob] does not exist. Creating it..."

    Invoke-Gcloud @(
        "scheduler",
        "jobs",
        "create",
        "http",
        $SchedulerJob,
        "--project",
        $ProjectId,
        "--location",
        $Region,
        "--schedule",
        "0 3 * * *",
        "--time-zone",
        $Timezone,
        "--uri",
        $RunUri,
        "--http-method",
        "POST",
        "--oauth-service-account-email",
        $SchedulerServiceAccount,
        "--oauth-token-scope",
        "https://www.googleapis.com/auth/cloud-platform",
        "--description",
        "Run Rhine Gauge 24h monitoring watchdog after daily prediction and evaluation"
    )
}


Write-Host ""
Write-Host "Monitoring deployment completed successfully."
Write-Host "Cloud Run job: $Job"
Write-Host "Scheduler job: $SchedulerJob"
Write-Host "Schedule: 03:00 $Timezone"