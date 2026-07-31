from ingestion.common.station_mapping import map_station_to_segment


def test_map_station_to_segment_upper():
    cfg = {
        "segments": {
            "upper_rhine": {"stations": ["MAXAU", "IFFEZHEIM"]},
            "lower_rhine": {"stations": ["KÖLN", "BONN"]},
        }
    }
    assert map_station_to_segment("MAXAU", cfg) == "upper_rhine"


def test_map_station_to_segment_none():
    cfg = {
        "segments": {
            "upper_rhine": {"stations": ["MAXAU", "IFFEZHEIM"]},
        }
    }
    assert map_station_to_segment("FLENSBURG", cfg) is None