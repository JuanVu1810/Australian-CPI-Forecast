"""Reusable SARIMA baseline for quarterly Australian CPI inflation.

The original assignment notebook selected SARIMA(0, 1, 1)x(0, 1, 1, 4) on the
trending CPI index. The production harness evaluates ``cpi_yoy`` so it uses a
development-only AIC/BIC grid-search winner for year-ended CPI inflation,
while still allowing callers to pass alternate specifications.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX


DEFAULT_TARGET_COLUMN = "cpi_yoy"
CPI_INDEX_NOTEBOOK_ORDER = (0, 1, 1)
CPI_INDEX_NOTEBOOK_SEASONAL_ORDER = (0, 1, 1, 4)

# cpi_yoy should not reuse the CPI-index differencing tuned above. These
# defaults are for the post-ETL ABS seasonally adjusted headline cpi_yoy basis.
# Phase 1 reran development-only order selection excluding the most recent 20
# quarters, then screened candidates across every expanding walk-forward window
# from the shortest training sample through the full development sample. The old
# NSA-basis default (2, 0, 2)x(0, 0, 2, 4) became unstable on the SA-sourced
# series, producing explosive short-window forecasts. The chosen replacement is
# documented in reports/model_refit_phase1_decisions.md.
DEFAULT_ORDER = (1, 0, 2)
DEFAULT_SEASONAL_ORDER = (1, 0, 2, 4)

# Separate trimmed-mean CPI YoY specification from the headline defaults above.
# Selected and stability-screened in reports/model_refit_phase1_decisions.md.
TRIMMED_MEAN_DEFAULT_TARGET_COLUMN = "trimmed_mean_cpi_yoy"
TRIMMED_MEAN_DEFAULT_ORDER = (1, 1, 1)
TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER = (0, 0, 1, 4)


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


def _simulation_result_to_paths(simulated, n_sims: int, steps: int) -> np.ndarray:
    paths = np.asarray(simulated, dtype=float)
    if paths.shape != (steps, n_sims):
        raise ValueError(
            "unexpected simulation result shape: "
            f"got {paths.shape}, expected {(steps, n_sims)} from statsmodels."
        )
    paths = paths.T
    if paths.shape != (n_sims, steps):
        raise ValueError(f"simulation paths have shape {paths.shape}, expected {(n_sims, steps)}.")
    return paths


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


def simulate_paths_from_fit(
    fitted,
    steps: int = 8,
    n_sims: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate future ``cpi_yoy`` paths from an already-fitted SARIMA model."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1.")

    # Innovation uncertainty only; this does not include parameter uncertainty.
    simulated = fitted.simulate(
        nsimulations=steps,
        anchor="end",
        repetitions=n_sims,
        random_state=seed,
    )
    return _simulation_result_to_paths(simulated, n_sims=n_sims, steps=steps)


def simulate_sarima_paths(
    series: pd.Series,
    steps: int = 8,
    n_sims: int = 1000,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
    trend: str = "n",
    maxiter: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Fit SARIMA and simulate future ``cpi_yoy`` paths with statsmodels innovations."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1.")

    fitted = fit_sarima(
        series=series,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        maxiter=maxiter,
    )
    return simulate_paths_from_fit(fitted, steps=steps, n_sims=n_sims, seed=seed)
