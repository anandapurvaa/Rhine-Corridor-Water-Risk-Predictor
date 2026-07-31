from ingestion.common.schemas import PegelMeasurement


def test_pegel_measurement_nullable_timestamp():
    record = PegelMeasurement(
        station_id="abc",
        station_name="Kaub",
        timeseries_name="waterlevel",
        timestamp_utc=None,
        value=145.0,
        unit="cm",
        latitude=50.08,
        longitude=7.76,
        ingestion_ts_utc="2026-07-31T00:00:00Z"
    )
    assert record.timestamp_utc is None


def test_pegel_measurement_valid_record():
    record = PegelMeasurement(
        station_id="abc",
        station_name="Kaub",
        timeseries_name="waterlevel",
        timestamp_utc="2026-07-31T00:00:00Z",
        value=145.0,
        unit="cm",
        latitude=50.08,
        longitude=7.76,
        ingestion_ts_utc="2026-07-31T00:00:00Z"
    )
    assert record.station_name == "Kaub"