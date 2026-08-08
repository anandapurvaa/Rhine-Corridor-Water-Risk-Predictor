$ErrorActionPreference = "Stop"

$PROJECT_ID = "rhine-corridor-navigator"
$REGION = "europe-west3"

$REPO_OWNER = "anandapurvaa"
$REPO_NAME = "Rhine-Corridor-Water-Risk-Predictor"

$CI_TRIGGER = "rhine-daily-pipeline-ci"
$PRODUCTION_TRIGGER = "rhine-daily-pipeline-production"

gcloud config set project $PROJECT_ID


gcloud builds triggers create github `
    --project=$PROJECT_ID `
    --region=$REGION `
    --name=$CI_TRIGGER `
    --repo-owner=$REPO_OWNER `
    --repo-name=$REPO_NAME `
    --pull-request-pattern="^main$" `
    --build-config="cloudbuild.ci.yaml"

if ($LASTEXITCODE -ne 0) {
    throw "CI trigger creation failed."
}


gcloud builds triggers create github `
    --project=$PROJECT_ID `
    --region=$REGION `
    --name=$PRODUCTION_TRIGGER `
    --repo-owner=$REPO_OWNER `
    --repo-name=$REPO_NAME `
    --branch-pattern="^main$" `
    --build-config="cloudbuild.production.yaml" `
    --require-approval

if ($LASTEXITCODE -ne 0) {
    throw "Production trigger creation failed."
}


Write-Host ""
Write-Host "CI trigger created: $CI_TRIGGER"
Write-Host "Protected production trigger created: $PRODUCTION_TRIGGER"