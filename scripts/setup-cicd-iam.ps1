$ErrorActionPreference = "Stop"

$PROJECT_ID = "rhine-corridor-navigator"
$REGION = "europe-west3"
$REPOSITORY = "gauge24h"
$JOB_NAME = "rhine-daily-pipeline"

gcloud config set project $PROJECT_ID

$PROJECT_NUMBER = gcloud projects describe $PROJECT_ID `
    --format="value(projectNumber)"

$BUILD_SERVICE_ACCOUNT = `
    "$PROJECT_NUMBER@cloudbuild.gserviceaccount.com"

$RUNTIME_SERVICE_ACCOUNT = gcloud run jobs describe $JOB_NAME `
    --project=$PROJECT_ID `
    --region=$REGION `
    --format="value(spec.template.spec.template.spec.serviceAccountName)"

Write-Host "Project number:"
Write-Host $PROJECT_NUMBER

Write-Host "Cloud Build service account:"
Write-Host $BUILD_SERVICE_ACCOUNT

Write-Host "Cloud Run runtime service account:"
Write-Host $RUNTIME_SERVICE_ACCOUNT


gcloud projects add-iam-policy-binding $PROJECT_ID `
    --member="serviceAccount:$BUILD_SERVICE_ACCOUNT" `
    --role="roles/artifactregistry.writer"


gcloud projects add-iam-policy-binding $PROJECT_ID `
    --member="serviceAccount:$BUILD_SERVICE_ACCOUNT" `
    --role="roles/run.developer"


gcloud iam service-accounts add-iam-policy-binding `
    $RUNTIME_SERVICE_ACCOUNT `
    --project=$PROJECT_ID `
    --member="serviceAccount:$BUILD_SERVICE_ACCOUNT" `
    --role="roles/iam.serviceAccountUser"


gcloud artifacts repositories add-iam-policy-binding $REPOSITORY `
    --location=$REGION `
    --project=$PROJECT_ID `
    --member="serviceAccount:$BUILD_SERVICE_ACCOUNT" `
    --role="roles/artifactregistry.writer"


Write-Host ""
Write-Host "CI/CD IAM configuration completed."