from pathlib import Path
import json

from google.cloud import storage


BUCKET_NAME = "rhine-corridor-navigator-models"
CLIENT = storage.Client()


def get_bucket():
    return CLIENT.bucket(BUCKET_NAME)


def upload_blob(local_path: Path, gcs_path: str) -> str:
    bucket = get_bucket()
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{BUCKET_NAME}/{gcs_path}"


def upload_json(data: dict, gcs_path: str) -> str:
    bucket = get_bucket()
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(
        json.dumps(data, indent=2),
        content_type="application/json",
    )
    return f"gs://{BUCKET_NAME}/{gcs_path}"


def _split_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")

    value = gcs_uri[5:]
    bucket_name, blob_name = value.split("/", 1)
    return bucket_name, blob_name


def download_blob(gcs_uri: str, local_path: Path) -> Path:
    bucket_name, blob_name = _split_gcs_uri(gcs_uri)
    bucket = CLIENT.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))

    return local_path


def download_json(gcs_uri: str) -> dict:
    bucket_name, blob_name = _split_gcs_uri(gcs_uri)
    bucket = CLIENT.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    return json.loads(blob.download_as_text())