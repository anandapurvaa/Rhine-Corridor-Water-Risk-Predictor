from ingestion.sources.dwd import station_in_scope


def test_station_in_scope_by_id():
    cfg = {"target_station_ids": ["00183"], "target_station_names": []}
    assert station_in_scope("183", "BASEL", cfg) is True


def test_station_in_scope_by_name():
    cfg = {"target_station_ids": [], "target_station_names": ["BASEL"]}
    assert station_in_scope("99999", "BASEL", cfg) is True


def test_station_not_in_scope():
    cfg = {"target_station_ids": ["00183"], "target_station_names": ["BASEL"]}
    assert station_in_scope("99999", "HAMBURG", cfg) is False