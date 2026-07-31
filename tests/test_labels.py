from feature_store.labels import assign_low_water_label


def test_assign_low_water_label_positive():
    cfg = {"low_water_thresholds_cm": {"KAUB": 120}}
    assert assign_low_water_label("KAUB", 110, cfg) == 1


def test_assign_low_water_label_negative():
    cfg = {"low_water_thresholds_cm": {"KAUB": 120}}
    assert assign_low_water_label("KAUB", 130, cfg) == 0


def test_assign_low_water_label_unknown_station():
    cfg = {"low_water_thresholds_cm": {"KAUB": 120}}
    assert assign_low_water_label("FLENSBURG", 110, cfg) is None