def test_expected_multisource_feature_columns():
    expected = {
        "station_name",
        "timestamp_utc",
        "value",
        "lag_1",
        "lag_3",
        "lag_6",
        "diff_1",
        "diff_3",
        "rolling_mean_3",
        "rolling_std_3",
        "temperature_c",
        "precipitation_mm",
        "wind_speed_ms",
        "pressure_hpa",
        "relative_humidity_pct",
    }

    assert "temperature_c" in expected
    assert "precipitation_mm" in expected
    assert "lag_1" in expected