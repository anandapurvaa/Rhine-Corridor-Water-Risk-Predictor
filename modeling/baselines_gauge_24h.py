from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_baseline(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["target_value"], errors="coerce").to_numpy()


def lag_24_baseline(df: pd.DataFrame) -> np.ndarray:
    if "lag_24" in df.columns:
        return pd.to_numeric(df["lag_24"], errors="coerce").to_numpy()
    return persistence_baseline(df)


def rolling_mean_baseline(df: pd.DataFrame) -> np.ndarray:
    if "rolling_mean_3" in df.columns:
        pred = pd.to_numeric(df["rolling_mean_3"], errors="coerce")
        return pred.fillna(pd.to_numeric(df["target_value"], errors="coerce")).to_numpy()
    return persistence_baseline(df)


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    err = pd.to_numeric(y_true, errors="coerce").to_numpy() - y_pred
    mse = float(np.nanmean(err ** 2))
    mae = float(np.nanmean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": float(np.nanmean(err)),
        "p90_abs_error": float(np.nanpercentile(np.abs(err), 90)),
    }