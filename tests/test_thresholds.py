from feature_store.labels import get_low_water_threshold


def test_get_low_water_threshold_found():
    cfg = {"low_water_thresholds_cm": {"KAUB": 120}}
    assert get_low_water_threshold("KAUB", cfg) == 120


def test_get_low_water_threshold_missing():
    cfg = {"low_water_thresholds_cm": {"KAUB": 120}}
    assert get_low_water_threshold("KÖLN", cfg) is None