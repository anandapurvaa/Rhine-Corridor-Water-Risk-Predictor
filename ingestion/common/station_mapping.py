from ingestion.common.utils import load_yaml


def load_segment_config(path: str = "config/segments.yaml") -> dict:
    return load_yaml(path)


def map_station_to_segment(station_name: str, segment_cfg: dict) -> str | None:
    station_name = (station_name or "").upper()

    for segment_id, payload in segment_cfg.get("segments", {}).items():
        stations = [s.upper() for s in payload.get("stations", [])]
        if station_name in stations:
            return segment_id

    return None