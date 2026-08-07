from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from google.cloud import bigquery


PROJECT_ID = os.getenv(
    "GCP_PROJECT_ID",
    "rhine-corridor-navigator",
).strip()

REGION = os.getenv(
    "GCP_REGION",
    "europe-west3",
).strip()

REGISTRY_TABLE = (
    f"{PROJECT_ID}.mlops.model_registry"
)

RMSE_TOLERANCE = float(
    os.getenv(
        "GAUGE24H_PROMOTE_RMSE_TOLERANCE",
        "0.0",
    )
)


def get_latest_models(
    client: bigquery.Client,
) -> tuple[dict | None, dict | None]:
    query = f"""
        SELECT
            model_version,
            status,
            gcs_path,
            trained_at_utc,
            evaluation_metrics_json
        FROM `{REGISTRY_TABLE}`
        WHERE status IN ('prod', 'staging')
        ORDER BY trained_at_utc DESC
    """

    rows = list(
        client.query(
            query,
            location=REGION,
        ).result()
    )

    production = None
    staging = None

    for row in rows:
        record = dict(row)

        if record["status"] == "prod" and production is None:
            production = record

        if record["status"] == "staging" and staging is None:
            staging = record

        if production is not None and staging is not None:
            break

    return production, staging


def read_rmse(model_record: dict) -> float:
    raw_metrics = model_record.get("evaluation_metrics_json")

    if not raw_metrics:
        raise RuntimeError(
            f"Missing evaluation metrics for "
            f"{model_record['model_version']}"
        )

    if isinstance(raw_metrics, str):
        metrics = json.loads(raw_metrics)
    else:
        metrics = raw_metrics

    rmse = metrics.get("rmse")

    if rmse is None:
        raise RuntimeError(
            f"Missing RMSE for {model_record['model_version']}"
        )

    return float(rmse)


def promote_candidate(
    client: bigquery.Client,
    candidate_version: str,
) -> None:
    query = f"""
        BEGIN TRANSACTION;

        UPDATE `{REGISTRY_TABLE}`
        SET
            status = 'staging',
            promoted_at_utc = NULL
        WHERE status = 'prod';

        UPDATE `{REGISTRY_TABLE}`
        SET
            status = 'prod',
            promoted_at_utc = CURRENT_TIMESTAMP(),
            notes = CONCAT(
                COALESCE(notes, ''),
                ' Automatically promoted after RMSE gate at ',
                CAST(CURRENT_TIMESTAMP() AS STRING)
            )
        WHERE model_version = @candidate_version
          AND status = 'staging';

        COMMIT TRANSACTION;
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "candidate_version",
                "STRING",
                candidate_version,
            )
        ]
    )

    client.query(
        query,
        job_config=job_config,
        location=REGION,
    ).result()


def main() -> None:
    client = bigquery.Client(
        project=PROJECT_ID,
        location=REGION,
    )

    production, staging = get_latest_models(client)

    if staging is None:
        raise RuntimeError(
            "No staging model is available for promotion."
        )

    staging_version = staging["model_version"]
    staging_rmse = read_rmse(staging)

    if production is None:
        print(
            "No existing production model found. "
            f"Promoting {staging_version}."
        )
        promote_candidate(client, staging_version)
        return

    production_version = production["model_version"]
    production_rmse = read_rmse(production)

    threshold = production_rmse + RMSE_TOLERANCE

    print(f"Candidate model: {staging_version}")
    print(f"Candidate RMSE:  {staging_rmse:.6f}")
    print(f"Current prod:    {production_version}")
    print(f"Production RMSE: {production_rmse:.6f}")
    print(f"Promotion limit: {threshold:.6f}")

    if staging_rmse <= threshold:
        promote_candidate(client, staging_version)
        print(
            "PROMOTED: candidate RMSE passed the production gate."
        )
    else:
        print(
            "NOT PROMOTED: candidate RMSE did not pass the "
            "production gate. Current prod remains active."
        )


if __name__ == "__main__":
    main()