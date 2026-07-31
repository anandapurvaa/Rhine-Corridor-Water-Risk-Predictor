from ingestion.sources.pegelonline import (
    build_currentmeasurement_url,
    build_measurements_url,
    build_station_identifier,
)


def test_build_station_identifier_prefers_uuid():
    station = {"uuid": "abc-123", "shortname": "KAUB", "number": 1234}
    assert build_station_identifier(station) == "abc-123"


def test_build_currentmeasurement_url():
    url = build_currentmeasurement_url("abc-123", "W")
    assert url.endswith("/stations/abc-123/W/currentmeasurement.json")


def test_build_measurements_url():
    url = build_measurements_url("abc-123", "W")
    assert url.endswith("/stations/abc-123/W/measurements.json")