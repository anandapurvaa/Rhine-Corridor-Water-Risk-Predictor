$ErrorActionPreference = "Stop"

$ProjectId = if ($env:GCP_PROJECT_ID) {
    $env:GCP_PROJECT_ID
} else {
    "rhine-corridor-navigator"
}


function Test-LogMetric {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $oldPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = "Continue"

        $null = & gcloud logging metrics describe $Name `
            --project=$ProjectId `
            --format="value(name)" 2>&1

        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
}


function Create-LogMetric {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$JobName,

        [Parameter(Mandatory = $true)]
        [string]$ServiceName,

        [Parameter(Mandatory = $true)]
        [string]$Event
    )

    $Filter = @"
resource.type="cloud_run_job"
resource.labels.job_name="$JobName"
jsonPayload.service="$ServiceName"
jsonPayload.event="$Event"
jsonPayload.status="fail"
"@

    Write-Host "Checking log metric: $Name"

    if (Test-LogMetric -Name $Name) {
        Write-Host "Log metric already exists: $Name"
        return
    }

    Write-Host "Creating log metric: $Name"

    & gcloud logging metrics create $Name `
        --project=$ProjectId `
        --description="Rhine Corridor monitoring failures for $Event" `
        --log-filter=$Filter

    if ($LASTEXITCODE -ne 0) {
        throw "Could not create log metric [$Name]. gcloud exit code: ${LASTEXITCODE}"
    }

    Write-Host "Created log metric: $Name"
}


Create-LogMetric `
    -Name "rhine_monitoring_prediction_failures" `
    -JobName "gauge24h-monitoring" `
    -ServiceName "gauge24h-watchdog" `
    -Event "prediction_health"


Create-LogMetric `
    -Name "rhine_monitoring_quality_failures" `
    -JobName "gauge24h-monitoring" `
    -ServiceName "gauge24h-watchdog" `
    -Event "data_quality_health"


Create-LogMetric `
    -Name "rhine_monitoring_stage_failures" `
    -JobName "gauge24h-monitoring" `
    -ServiceName "gauge24h-watchdog" `
    -Event "stage_health"


Create-LogMetric `
    -Name "rhine_monitoring_evaluation_failures" `
    -JobName "gauge24h-monitoring" `
    -ServiceName "gauge24h-watchdog" `
    -Event "evaluation_health"


Create-LogMetric `
    -Name "rhine_monitoring_watchdog_failures" `
    -JobName "gauge24h-monitoring" `
    -ServiceName "gauge24h-watchdog" `
    -Event "watchdog_failed"


Create-LogMetric `
    -Name "rhine_training_failures" `
    -JobName "gauge24h-train" `
    -ServiceName "gauge24h-training" `
    -Event "training_failed"


Write-Host ""
Write-Host "All monitoring log metrics are ready."