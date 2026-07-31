import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ingestion.common.config import settings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def get_json(url: str):
    response = requests.get(
        url,
        timeout=settings.timeout_seconds,
        headers={"User-Agent": "rheinkorridor-sentinel/0.1"}
    )
    response.raise_for_status()
    return response.json()