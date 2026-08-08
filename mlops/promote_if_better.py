from __future__ import annotations

import json
import os

from google.cloud import bigquery


PROJECT_ID = os.getenv(
    "GCP_PROJECT_ID",
    "rhine-corridor-navigator",
).strip()

REGION = os.getenv(
    "GCP_REGION",
    "europe-west3",
).strip()

REGISTRY_TABLE = f"{PROJECT_ID}.mlops.model_registry"

RMSE_TOLERANCE = float(
    os.getenv(
        "GAUGE24H_PROMOTE_RMSE_TOLERANCE",
        "0.0",
    )
)


def load_ranked_models(
    client: bigquery.Client,
) -> list[dict]:
    query = f"""
        SELECT
            model_version,
            status,
            gcs_path,
            trained_at_utc,
            promoted_at_utc,
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

    models = []

    for row in rows:
        record = dict(row)
        raw_metrics = record.get("evaluation_metrics_json")

        if not raw_metrics:
            print(
                f"Skipping {record['model_version']}: "
                "evaluation_metrics_json is empty."
            )
            continue

        try:
            metrics = (
                json.loads(raw_metrics)
                if isinstance(raw_metrics, str)
                else raw_metrics
            )

            rmse = float(metrics["rmse"])
            mae = float(metrics["mae"])

        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            print(
                f"Skipping {record['model_version']}: "
                f"invalid evaluation metrics: {exc}"
            )
            continue

        record["rmse"] = rmse
        record["mae"] = mae
        models.append(record)

    return sorted(
        models,
        key=lambda model: (
            model["rmse"],
            model["mae"],
            str(model.get("trained_at_utc") or ""),
        ),
    )


def promote_model(
    client: bigquery.Client,
    winning_model_version: str,
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
                ' Selected as best validation model at ',
                CAST(CURRENT_TIMESTAMP() AS STRING)
            )
        WHERE model_version = @winning_model_version
          AND status = 'staging';

        COMMIT TRANSACTION;
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "winning_model_version",
                "STRING",
                winning_model_version,
            )
        ]
    )

    client.query(
        query,
        job_config=job_config,
        location=REGION,
    ).result()


def main() -> str:
    client = bigquery.Client(
        project=PROJECT_ID,
        location=REGION,
    )

    ranked_models = load_ranked_models(client)

    if not ranked_models:
        raise RuntimeError(
            "No prod or staging models with valid evaluation metrics found."
        )

    print("=== Model ranking by validation RMSE ===")

    for rank, model in enumerate(ranked_models, start=1):
        print(
            f"{rank}. "
            f"{model['model_version']} | "
            f"status={model['status']} | "
            f"RMSE={model['rmse']:.6f} | "
            f"MAE={model['mae']:.6f}"
        )

    winner = ranked_models[0]
    winner_version = winner["model_version"]
    winner_status = winner["status"]

    current_prod = next(
        (
            model
            for model in ranked_models
            if model["status"] == "prod"
        ),
        None,
    )

    if current_prod is not None:
        promotion_limit = (
            current_prod["rmse"] + RMSE_TOLERANCE
        )

        print(
            f"Current production RMSE: "
            f"{current_prod['rmse']:.6f}"
        )
        print(
            f"Promotion limit: "
            f"{promotion_limit:.6f}"
        )

    if winner_status == "prod":
        print(
            "KEEPING current production model: "
            f"{winner_version}"
        )
        return winner_version

    if current_prod is not None:
        if winner["rmse"] > current_prod["rmse"] + RMSE_TOLERANCE:
            print(
                "NOT PROMOTED: winning staging model did not "
                "beat the current production model."
            )
            return current_prod["model_version"]

    print(
        "PROMOTING validation winner: "
        f"{winner_version}"
    )

    promote_model(client, winner_version)

    print(
        "PROMOTED: "
        f"{winner_version} is now the production model."
    )

    return winner_version


if __name__ == "__main__":
    main()