"""Reusable SARIMA baseline for quarterly Australian CPI inflation.

The original assignment notebook selected SARIMA(0, 1, 1)x(0, 1, 1, 4) on the
trending CPI index. The production harness evaluates ``cpi_yoy`` so it uses a
development-only AIC/BIC grid-search winner for year-ended CPI inflation,
while still allowing callers to pass alternate specifications.
"""

from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX


DEFAULT_TARGET_COLUMN = "cpi_yoy"
CPI_INDEX_NOTEBOOK_ORDER = (0, 1, 1)
CPI_INDEX_NOTEBOOK_SEASONAL_ORDER = (0, 1, 1, 4)

# cpi_yoy should not reuse the CPI-index differencing tuned above. Selected by
# `python -m src.models.sarima_order_search --holdout-quarters 20`, i.e. AIC
# on development-only data (the series' final 20 quarters excluded), not the
# full sample -- so the walk-forward test tail never informed this choice.
# The raw development-only AIC winner, (2, 0, 1)x(0, 0, 2, 4) (AIC 167.38),
# produces explosive forecasts (>1e20) when refit on the shortest actual
# walk-forward training window (32 quarters) -- `enforce_stationarity=False`
# lets a short sample identify unstable AR/seasonal-MA roots that a larger
# sample masks. (2, 0, 2)x(0, 0, 2, 4) is statistically indistinguishable on
# development AIC (168.42, delta 1.04) and confirmed stable (bounded, finite
# forecasts) at every expanding-window size from 32 quarters up to the full
# development sample. The old full-sample winner was (1, 0, 2)x(1, 0, 2, 4).
DEFAULT_ORDER = (2, 0, 2)
DEFAULT_SEASONAL_ORDER = (0, 0, 2, 4)


def fit_sarima(
    series: pd.Series,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
    trend: str = "n",
    maxiter: int = 100,
):
    """Fit the cpi_yoy-tuned SARIMA specification to one training series."""
    clean = pd.Series(series).dropna().astype(float)
    if clean.empty:
        raise ValueError("SARIMA requires at least one non-missing training value.")

    model = SARIMAX(
        clean,
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


def forecast_sarima(
    series: pd.Series,
    steps: int = 8,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
    trend: str = "n",
    maxiter: int = 100,
) -> pd.Series:
    """Fit SARIMA on ``series`` and return ``steps`` out-of-sample forecasts."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    fitted = fit_sarima(
        series=series,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        maxiter=maxiter,
    )
    forecast = fitted.get_forecast(steps=steps).predicted_mean
    return pd.Series(forecast, name="forecast")
