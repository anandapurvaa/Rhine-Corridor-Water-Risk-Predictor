from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

from mlops.promote_if_better import main as promote_main
from mlops.train_production import main as train_main


PROJECT_ID = os.getenv(
    "GCP_PROJECT_ID",
    "rhine-corridor-navigator",
).strip()

JOB_NAME = os.getenv(
    "TRAINING_JOB_NAME",
    "gauge24h-train",
).strip()

logging.basicConfig(
    stream=sys.stdout,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(message)s",
)

logger = logging.getLogger("gauge24h-training")


def emit(
    event: str,
    status: str,
    **fields: object,
) -> None:
    payload = {
        "event": event,
        "status": status,
        "service": "gauge24h-training",
        "job_name": JOB_NAME,
        "project_id": PROJECT_ID,
        "measured_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        **fields,
    }

    logger.info(
        json.dumps(
            payload,
            default=str,
            separators=(",", ":"),
        )
    )


def main() -> None:
    emit("training_started", "ok")

    try:
        train_main()
        emit(
            "training_candidate_completed",
            "pass",
        )

        winning_model_version = promote_main()

        emit(
            "training_promotion_completed",
            "pass",
            model_version=winning_model_version,
        )

        emit(
            "training_completed",
            "pass",
            model_version=winning_model_version,
        )

    except Exception as exc:
        emit(
            "training_failed",
            "fail",
            error=repr(exc),
        )
        raise


if __name__ == "__main__":
    main()