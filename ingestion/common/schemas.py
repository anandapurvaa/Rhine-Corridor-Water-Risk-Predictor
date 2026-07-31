from pydantic import BaseModel


class PegelMeasurement(BaseModel):
    station_id: str
    station_name: str
    timeseries_name: str
    timestamp_utc: str | None = None
    value: float | None = None
    unit: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    ingestion_ts_utc: str
    source: str = "pegelonline"