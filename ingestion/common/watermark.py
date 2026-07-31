from pathlib import Path


WATERMARK_FILE = Path(".watermarks/pegelonline.txt")


def get_watermark() -> str | None:
    if WATERMARK_FILE.exists():
        return WATERMARK_FILE.read_text().strip() or None
    return None


def set_watermark(value: str) -> None:
    WATERMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATERMARK_FILE.write_text(value)