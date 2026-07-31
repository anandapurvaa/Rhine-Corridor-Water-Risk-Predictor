import pandas as pd

from feature_store.build_features import add_training_eligibility_flags


def test_add_training_eligibility_flags():
    df = pd.DataFrame(
        [
            {"lag_1": 1.0, "lag_3": 2.0, "rolling_mean_3": 3.0},
            {"lag_1": None, "lag_3": 2.0, "rolling_mean_3": 3.0},
        ]
    )

    out = add_training_eligibility_flags(df)
    assert list(out["eligible_for_training"]) == [1, 0]