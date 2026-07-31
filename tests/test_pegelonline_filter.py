from ingestion.sources.pegelonline import is_rhine_station


def test_is_rhine_station_priority_match():
    cfg = {
        "rhine_priority_gauges": ["KAUB"],
        "rhine_name_keywords": ["RHEIN", "KAUB"]
    }
    station = {"shortname": "KAUB"}
    assert is_rhine_station(station, cfg) is True


def test_is_rhine_station_keyword_match():
    cfg = {
        "rhine_priority_gauges": [],
        "rhine_name_keywords": ["RHEIN"]
    }
    station = {"shortname": "BASEL-RHEINHALLE"}
    assert is_rhine_station(station, cfg) is True


def test_is_rhine_station_negative():
    cfg = {
        "rhine_priority_gauges": ["KAUB"],
        "rhine_name_keywords": ["RHEIN", "KAUB"]
    }
    station = {"shortname": "FLENSBURG"}
    assert is_rhine_station(station, cfg) is False