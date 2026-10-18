"""SARIMA + Elastic Net ensemble for CPI year-ended inflation forecasts.

The ensemble is a plain 50/50 average of the two models. Per-horizon weights fitted on the backtest
did no better than this once they had to be estimated in real time, so nothing is fitted here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.elastic_net import (
    ELASTIC_NET_FEATURE_COLUMNS,
    TARGET_COLUMN,
    forecast_elastic_net_direct,
    simulate_elastic_net_paths,
)
from src.models.sarima import (
    DEFAULT_ORDER as SARIMA_DEFAULT_ORDER,
    DEFAULT_SEASONAL_ORDER as SARIMA_DEFAULT_SEASONAL_ORDER,
    forecast_sarima,
    simulate_sarima_paths,
)


SARIMA_WEIGHT = 0.5
ELASTIC_NET_WEIGHT = 0.5
DEFAULT_SEED = 42


def combine_point_forecasts(sarima_forecast, elastic_net_forecast) -> np.ndarray:
    """Return the 50/50 average of two equal-length one-dimensional forecasts."""
    sarima_values = np.asarray(sarima_forecast, dtype=float)
    elastic_net_values = np.asarray(elastic_net_forecast, dtype=float)
    if sarima_values.ndim != 1 or elastic_net_values.ndim != 1:
        raise ValueError("both forecasts must be one-dimensional arrays.")
    if sarima_values.shape != elastic_net_values.shape:
        raise ValueError("both forecasts must have the same length.")
    return SARIMA_WEIGHT * sarima_values + ELASTIC_NET_WEIGHT * elastic_net_values


def combine_paths(sarima_paths, elastic_net_paths) -> np.ndarray:
    """50/50 average of paired (n_sims, steps) draws from the two component models."""
    sarima_values = np.asarray(sarima_paths, dtype=float)
    elastic_net_values = np.asarray(elastic_net_paths, dtype=float)
    if sarima_values.ndim != 2 or elastic_net_values.ndim != 2:
        raise ValueError("both path arrays must be two-dimensional with shape (n_sims, steps).")
    if sarima_values.shape != elastic_net_values.shape:
        raise ValueError("both path arrays must have the same shape.")
    return SARIMA_WEIGHT * sarima_values + ELASTIC_NET_WEIGHT * elastic_net_values


def recenter_paths_to_median(paths: np.ndarray, target_median) -> np.ndarray:
    """Shift each horizon's draws so their median equals ``target_median``; the spread is unchanged."""
    values = np.asarray(paths, dtype=float)
    if values.ndim != 2:
        raise ValueError("paths must be two-dimensional with shape (n_sims, steps).")
    target = np.asarray(target_median, dtype=float)
    if target.ndim != 1 or target.shape[0] != values.shape[1]:
        raise ValueError("target_median must be one-dimensional with one value per step.")
    current_median = np.percentile(values, 50, axis=0)
    return values + (target - current_median)


def forecast_ensemble(
    train_frame: pd.DataFrame,
    steps: int = 8,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Fit both components on raw training data and average their point forecasts."""
    sarima_forecast = forecast_sarima(
        train_frame[TARGET_COLUMN],
        steps=steps,
        order=SARIMA_DEFAULT_ORDER,
        seasonal_order=SARIMA_DEFAULT_SEASONAL_ORDER,
    )
    elastic_net_forecast = forecast_elastic_net_direct(
        train_frame,
        steps=steps,
        seed=seed,
        feature_columns=ELASTIC_NET_FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )
    return combine_point_forecasts(sarima_forecast, elastic_net_forecast)


def simulate_ensemble_paths(
    train_frame: pd.DataFrame,
    steps: int = 8,
    n_sims: int = 1000,
    seed: int = DEFAULT_SEED,
    sarima_series: pd.Series | None = None,
) -> np.ndarray:
    """Fit both components on raw training data and average their predictive draws."""
    sarima_train = (
        pd.Series(sarima_series).dropna().astype(float)
        if sarima_series is not None
        else train_frame[TARGET_COLUMN]
    )
    sarima_paths = simulate_sarima_paths(
        sarima_train,
        steps=steps,
        n_sims=n_sims,
        order=SARIMA_DEFAULT_ORDER,
        seasonal_order=SARIMA_DEFAULT_SEASONAL_ORDER,
        seed=seed,
    )
    elastic_net_paths = simulate_elastic_net_paths(
        train_frame,
        steps=steps,
        n_sims=n_sims,
        seed=seed + 1,
        feature_columns=ELASTIC_NET_FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )
    combined = combine_paths(sarima_paths, elastic_net_paths)
    # WHY recenter: the median of an average of two simulations is not the average of their point
    # forecasts, so shift the draws onto the ensemble point forecast to make the reported forecast
    # and the interval centre agree exactly.
    sarima_point = forecast_sarima(
        sarima_train,
        steps=steps,
        order=SARIMA_DEFAULT_ORDER,
        seasonal_order=SARIMA_DEFAULT_SEASONAL_ORDER,
    )
    elastic_net_point = forecast_elastic_net_direct(
        train_frame,
        steps=steps,
        seed=seed,
        feature_columns=ELASTIC_NET_FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )
    return recenter_paths_to_median(combined, combine_point_forecasts(sarima_point, elastic_net_point))


def ensemble_interval_from_paths(
    paths,
    lower: float = 0.1,
    upper: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ensemble draws into lower/upper forecast interval arrays."""
    lower_values, upper_values = np.percentile(
        np.asarray(paths, dtype=float),
        [lower * 100, upper * 100],
        axis=0,
    )
    return lower_values, upper_values


def _ensemble_prediction_frame(
    sarima_predictions: pd.DataFrame,
    elastic_net_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Build ensemble backtest rows from the SARIMA and Elastic Net rows that share an origin and horizon."""
    key_columns = ["forecast_origin", "target_quarter", "horizon"]
    merged = sarima_predictions.merge(
        elastic_net_predictions,
        on=key_columns,
        how="inner",
        suffixes=("_sarima", "_elastic_net"),
    )
    if merged.empty:
        raise ValueError("SARIMA and Elastic Net backtests have no common forecast grid.")
    if not np.allclose(merged["actual_sarima"], merged["actual_elastic_net"]):
        raise ValueError("SARIMA and Elastic Net backtests disagree on actual values.")

    forecast = combine_point_forecasts(
        merged["forecast_sarima"].to_numpy(dtype=float),
        merged["forecast_elastic_net"].to_numpy(dtype=float),
    )
    result = merged.loc[:, key_columns].copy()
    result.insert(0, "model", "ensemble")
    result["actual"] = merged["actual_sarima"].astype(float).to_numpy()
    result["forecast"] = forecast
    result["error"] = result["actual"] - result["forecast"]
    return result[
        [
            "model",
            "forecast_origin",
            "target_quarter",
            "horizon",
            "actual",
            "forecast",
            "error",
        ]
    ]
