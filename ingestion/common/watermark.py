import json
from pathlib import Path


WATERMARK_DIR = Path(".watermarks")


def _watermark_path(source_name: str) -> Path:
    return WATERMARK_DIR / f"{source_name}.json"


def get_watermark(source_name: str) -> dict | None:
    path = _watermark_path(source_name)
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def set_watermark(source_name: str, payload: dict) -> None:
    WATERMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = _watermark_path(source_name)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")