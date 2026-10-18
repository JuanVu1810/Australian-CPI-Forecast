"""SVAR-adjusted CPI scenario engine.

Scenario paths combine the existing SARIMA + Elastic Net ensemble predictive
draws with raw, paired SVAR bootstrap impulse-response draws. The IRFs come
from the documented Phase 1b VAR(2)-in-levels system, which still fails
multivariate residual whiteness and normality diagnostics after COVID
treatment checks and the block-bootstrap IRF fix. Scenario outputs are
illustrative under an imperfect specification, not precise causal estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.models import ensemble, svar
from src.models.elastic_net import ELASTIC_NET_FEATURE_COLUMNS, TARGET_COLUMN
from src.models.evaluation import CURATED_DATA_PATH, DEFAULT_HORIZONS


DEFAULT_N_SIMS = 1000
DEFAULT_SEED = 42
DEFAULT_LOWER_QUANTILE = 0.1
DEFAULT_UPPER_QUANTILE = 0.9
SCENARIO_CAVEAT = (
    "Scenario IRFs come from a Phase 1b VAR(2)-in-levels SVAR system that still "
    "fails multivariate residual whiteness and normality diagnostics after COVID "
    "treatment checks and the block-bootstrap IRF fix; adjusted forecasts are "
    "illustrative under a documented, imperfect specification, not precise "
    "causal estimates."
)


@dataclass(frozen=True)
class ScenarioResult:
    target_column: str
    shock_variable: str
    shock_value: float
    shock_size: float
    horizons: tuple[int, ...]
    forecast: list[float]
    interval_lower: list[float]
    interval_upper: list[float]
    quarters: list[str]
    forecast_origin: str
    caveat: str
    baseline_paths: np.ndarray
    shock_contribution: np.ndarray
    combined_paths: np.ndarray


def compute_shock_size(
    svar_forecast: pd.DataFrame,
    shock_variable: str,
    shock_value: float,
) -> float:
    """Return the surprise relative to the SVAR's horizon-1 macro forecast."""
    if shock_variable not in svar.SHARED_ESCALATION_SHOCKS:
        raise ValueError(
            "shock_variable must be one of "
            f"{tuple(svar.SHARED_ESCALATION_SHOCKS)}; got {shock_variable!r}."
        )
    forecast = pd.DataFrame(svar_forecast)
    if forecast.empty:
        raise ValueError("svar_forecast must contain at least one forecast row.")
    if shock_variable not in forecast.columns:
        raise ValueError(f"svar_forecast is missing shock variable {shock_variable!r}.")

    expected_h1 = float(forecast.iloc[0][shock_variable])
    if not np.isfinite(expected_h1):
        raise ValueError(f"horizon-1 SVAR forecast for {shock_variable!r} is not finite.")
    if not np.isfinite(shock_value):
        raise ValueError("shock_value must be a finite number.")
    return float(shock_value) - expected_h1


def combine_scenario_paths(
    tier1_paths: np.ndarray,
    irf_draws: np.ndarray,
    shock_size: float,
    response_index: int,
    shock_index: int,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair ensemble draw ``i`` with SVAR IRF draw ``i`` and add shock_size * IRF; return (combined, contribution).

    The shock happens at t+1 (it is measured against the SVAR's horizon-1 forecast), so CPI horizon h
    takes the IRF at index h - 1, and horizon 1 gets the same-quarter impact at index 0.
    """
    requested_horizons = _validate_horizons(horizons)
    baseline = np.asarray(tier1_paths, dtype=float)
    irfs = np.asarray(irf_draws, dtype=float)
    if baseline.ndim != 2:
        raise ValueError("tier1_paths must have shape (n_draws, steps).")
    if irfs.ndim != 4:
        raise ValueError("irf_draws must have shape (n_draws, periods, n_vars, n_vars).")
    if baseline.shape[0] != irfs.shape[0]:
        raise ValueError(
            "tier1_paths and irf_draws must contain the same number of paired draws."
        )
    if max(requested_horizons) > baseline.shape[1]:
        raise ValueError("tier1_paths do not cover the requested horizons.")
    if max(requested_horizons) - 1 >= irfs.shape[1]:
        raise ValueError("irf_draws do not cover the requested horizons.")
    if not 0 <= response_index < irfs.shape[2]:
        raise ValueError("response_index is outside the IRF response dimension.")
    if not 0 <= shock_index < irfs.shape[3]:
        raise ValueError("shock_index is outside the IRF shock dimension.")

    baseline_selected = baseline[:, [horizon - 1 for horizon in requested_horizons]]
    selected_irfs = irfs[
        :,
        [horizon - 1 for horizon in requested_horizons],
        response_index,
        shock_index,
    ]
    contribution = float(shock_size) * selected_irfs
    return baseline_selected + contribution, contribution


def run_scenario(
    shock_variable: str,
    shock_value: float,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_sims: int = DEFAULT_N_SIMS,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float = DEFAULT_UPPER_QUANTILE,
    seed: int = DEFAULT_SEED,
    curated_path: Path = CURATED_DATA_PATH,
) -> ScenarioResult:
    """Run one SVAR-adjusted headline CPI ensemble scenario."""
    target_column = TARGET_COLUMN
    requested_horizons = _validate_horizons(horizons)
    _validate_quantiles(lower_quantile, upper_quantile)
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1.")
    if shock_variable not in svar.SHARED_ESCALATION_SHOCKS:
        raise ValueError(
            "shock_variable must be one of "
            f"{tuple(svar.SHARED_ESCALATION_SHOCKS)}; got {shock_variable!r}."
        )

    steps = max(requested_horizons)
    frame = _load_curated_frame(curated_path)
    # Pinned to the same cutoff as the SVAR IRF export and credit-stress test
    # (svar.FORECAST_ORIGIN_PIN) so no component here fits on data later than
    # 2025Q4, regardless of how far the curated CSV itself has been refreshed.
    frame = frame.loc[frame.index <= svar.FORECAST_ORIGIN_PIN]
    svar_frame = _numeric_complete_rows(frame, svar.SYSTEM_A_COLUMNS)
    fitted_svar = svar.fit_svar(
        svar_frame,
        lag_order=svar.DEFAULT_SVAR_LAG_ORDER,
        ordering=svar.SYSTEM_A_CHOLESKY_ORDER,
    )
    svar_forecast = svar.forecast_from_fit(fitted_svar, steps=steps)
    shock_size = compute_shock_size(
        svar_forecast=svar_forecast,
        shock_variable=shock_variable,
        shock_value=shock_value,
    )

    tier1_paths = ensemble.simulate_ensemble_paths(
        frame,
        steps=steps,
        n_sims=n_sims,
        seed=seed,
        sarima_series=frame[target_column],
    )
    irf_draws = svar.bootstrap_orth_irf_draws(
        fitted_svar,
        horizons=requested_horizons,
        n_bootstrap=n_sims,
        seed=seed + 10_000,
    )
    names = tuple(str(name) for name in fitted_svar.names)
    combined_paths, contribution = combine_scenario_paths(
        tier1_paths=tier1_paths,
        irf_draws=irf_draws,
        shock_size=shock_size,
        response_index=names.index(target_column),
        shock_index=names.index(shock_variable),
        horizons=requested_horizons,
    )
    lower, upper = scenario_interval_from_paths(
        combined_paths,
        lower=lower_quantile,
        upper=upper_quantile,
    )
    ensemble_columns = (target_column, *ELASTIC_NET_FEATURE_COLUMNS)
    clean_frame = _numeric_complete_rows(frame, ensemble_columns).sort_index()
    forecast_origin = str(clean_frame.index[-1])
    future_quarters = []
    for horizon in requested_horizons:
        future_quarters.append(str(clean_frame.index[-1] + horizon))

    return ScenarioResult(
        target_column=target_column,
        shock_variable=str(shock_variable),
        shock_value=float(shock_value),
        shock_size=float(shock_size),
        horizons=requested_horizons,
        forecast=np.mean(combined_paths, axis=0).astype(float).tolist(),
        interval_lower=lower.astype(float).tolist(),
        interval_upper=upper.astype(float).tolist(),
        quarters=future_quarters,
        forecast_origin=forecast_origin,
        caveat=SCENARIO_CAVEAT,
        baseline_paths=tier1_paths[:, [horizon - 1 for horizon in requested_horizons]],
        shock_contribution=contribution,
        combined_paths=combined_paths,
    )


def scenario_interval_from_paths(
    paths: np.ndarray,
    lower: float = DEFAULT_LOWER_QUANTILE,
    upper: float = DEFAULT_UPPER_QUANTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scenario interval quantiles from combined paths."""
    _validate_quantiles(lower, upper)
    lower_values, upper_values = np.percentile(
        np.asarray(paths, dtype=float),
        [lower * 100, upper * 100],
        axis=0,
    )
    return lower_values, upper_values


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    """Return the horizons as ints; they must be unique and within 1-8."""
    requested = tuple(int(horizon) for horizon in horizons)
    if not requested:
        raise ValueError("horizons must contain at least one horizon.")
    if len(set(requested)) != len(requested):
        raise ValueError("horizons must not contain duplicates.")
    if min(requested) < min(DEFAULT_HORIZONS) or max(requested) > max(DEFAULT_HORIZONS):
        raise ValueError(
            f"horizons must be between {min(DEFAULT_HORIZONS)} and {max(DEFAULT_HORIZONS)}."
        )
    return requested


def _validate_quantiles(lower: float, upper: float) -> None:
    if not 0 <= lower < upper <= 1:
        raise ValueError("lower and upper quantiles must satisfy 0 <= lower < upper <= 1.")


def _load_curated_frame(path: Path) -> pd.DataFrame:
    """Read the curated CSV indexed by quarter."""
    data = pd.read_csv(path)
    if "quarter" not in data.columns:
        raise ValueError(f"{path} must contain a 'quarter' column.")
    data.index = pd.PeriodIndex(data.pop("quarter").astype(str), freq="Q")
    return data.sort_index()


def _numeric_complete_rows(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Keep ``columns`` as numbers and drop rows with any missing value."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"curated frame missing required columns: {missing}")
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").dropna()
