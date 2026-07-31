import os
from dataclasses import dataclass


@dataclass
class Settings:
    project_id: str = os.getenv("GCP_PROJECT_ID", "")
    dataset_raw: str = os.getenv("BQ_DATASET_RAW", "rhein_raw")
    pegelonline_base_url: str = os.getenv(
        "PEGELONLINE_BASE_URL",
        "https://www.pegelonline.wsv.de/webservices/rest-api/v2"
    )
    timeout_seconds: int = int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))


settings = Settings()