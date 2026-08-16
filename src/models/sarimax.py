"""Reusable SARIMAX model helpers for quarterly Australian CPI inflation."""

from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.sarima import DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER


def _align_endog_exog(series: pd.Series, exog: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    frame = pd.concat(
        [pd.Series(series).rename("__target__"), pd.DataFrame(exog)],
        axis=1,
    ).dropna()
    if frame.empty:
        raise ValueError("SARIMAX requires overlapping non-missing target and exog values.")
    return frame["__target__"].astype(float), frame.drop(columns="__target__").astype(float)


def fit_sarimax(
    series: pd.Series,
    exog: pd.DataFrame,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
    trend: str = "n",
    maxiter: int = 100,
):
    """Fit a SARIMAX specification to one target series and fixed exog matrix."""
    clean_y, clean_exog = _align_endog_exog(series, exog)

    model = SARIMAX(
        clean_y,
        exog=clean_exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        return model.fit(disp=False, maxiter=maxiter)


def forecast_sarimax(
    series: pd.Series,
    exog: pd.DataFrame,
    future_exog: pd.DataFrame,
    steps: int = 8,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
    trend: str = "n",
    maxiter: int = 100,
) -> pd.Series:
    """Fit SARIMAX on ``series``/``exog`` and forecast with supplied future exog."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    future = pd.DataFrame(future_exog).iloc[:steps].astype(float)
    if len(future) < steps:
        raise ValueError("future_exog must contain at least ``steps`` rows.")
    if future.isna().any(axis=None):
        raise ValueError("future_exog must not contain missing values.")

    fitted = fit_sarimax(
        series=series,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        maxiter=maxiter,
    )
    train_columns = list(pd.DataFrame(exog).columns)
    future = future.loc[:, train_columns]
    forecast = fitted.get_forecast(steps=steps, exog=future).predicted_mean
    return pd.Series(forecast, name="forecast")
