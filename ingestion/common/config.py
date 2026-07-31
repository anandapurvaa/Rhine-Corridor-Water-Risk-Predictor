import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_dotenv_file()


@dataclass
class Settings:
    project_id: str = os.getenv("GCP_PROJECT_ID", "")
    dataset_raw: str = os.getenv("BQ_DATASET_RAW", "rhein_raw")
    dataset_curated: str = os.getenv("BQ_DATASET_CURATED", "rhein_curated")
    gcp_region: str = os.getenv("GCP_REGION", "europe-west3")
    pegelonline_base_url: str = os.getenv(
        "PEGELONLINE_BASE_URL",
        "https://www.pegelonline.wsv.de/webservices/rest-api/v2"
    )
    timeout_seconds: int = int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))


settings = Settings()