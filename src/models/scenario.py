"""SVAR-adjusted CPI scenario engine.

Scenario paths combine the existing SARIMA + Elastic Net ensemble predictive
draws with raw, paired SVAR bootstrap impulse-response draws. The IRFs come
from the documented Phase 1b VAR(2)-in-levels systems, which still fail
multivariate residual whiteness and normality diagnostics after COVID
treatment checks and the block-bootstrap IRF fix. Scenario outputs are
illustrative under an imperfect specification, not precise causal estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from src.models import ensemble, svar
from src.models.elastic_net import (
    ELASTIC_NET_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
    TRIMMED_MEAN_TARGET_COLUMN,
)
from src.models.evaluation import CURATED_DATA_PATH, DEFAULT_HORIZONS
from src.models.sarima import (
    DEFAULT_ORDER as SARIMA_DEFAULT_ORDER,
    DEFAULT_SEASONAL_ORDER as SARIMA_DEFAULT_SEASONAL_ORDER,
    TRIMMED_MEAN_DEFAULT_ORDER,
    TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER,
)


ScenarioTarget = Literal["headline", "trimmed_mean"]
DEFAULT_N_SIMS = 1000
DEFAULT_SEED = 42
DEFAULT_LOWER_QUANTILE = 0.1
DEFAULT_UPPER_QUANTILE = 0.9
SCENARIO_CAVEAT = (
    "Scenario IRFs come from Phase 1b VAR(2)-in-levels SVAR systems that still "
    "fail multivariate residual whiteness and normality diagnostics after COVID "
    "treatment checks and the block-bootstrap IRF fix; adjusted forecasts are "
    "illustrative under a documented, imperfect specification, not precise "
    "causal estimates."
)


@dataclass(frozen=True)
class ScenarioTargetConfig:
    target: ScenarioTarget
    target_column: str
    svar_columns: tuple[str, ...]
    svar_ordering: tuple[str, ...]
    sarima_order: tuple[int, int, int]
    sarima_seasonal_order: tuple[int, int, int, int]
    elastic_net_feature_columns: tuple[str, ...]
    dynamic_weights: bool


@dataclass(frozen=True)
class ScenarioResult:
    target: ScenarioTarget
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


TARGET_CONFIGS: dict[str, ScenarioTargetConfig] = {
    "headline": ScenarioTargetConfig(
        target="headline",
        target_column=TARGET_COLUMN,
        svar_columns=svar.SYSTEM_A_COLUMNS,
        svar_ordering=svar.SYSTEM_A_CHOLESKY_ORDER,
        sarima_order=SARIMA_DEFAULT_ORDER,
        sarima_seasonal_order=SARIMA_DEFAULT_SEASONAL_ORDER,
        elastic_net_feature_columns=ELASTIC_NET_FEATURE_COLUMNS,
        dynamic_weights=True,
    ),
    "trimmed_mean": ScenarioTargetConfig(
        target="trimmed_mean",
        target_column=TRIMMED_MEAN_TARGET_COLUMN,
        svar_columns=svar.SYSTEM_B_COLUMNS,
        svar_ordering=svar.SYSTEM_B_CHOLESKY_ORDER,
        sarima_order=TRIMMED_MEAN_DEFAULT_ORDER,
        sarima_seasonal_order=TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER,
        elastic_net_feature_columns=TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
        dynamic_weights=False,
    ),
}


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
    return float(shock_value) - expected_h1


def combine_scenario_paths(
    tier1_paths: np.ndarray,
    irf_draws: np.ndarray,
    shock_size: float,
    response_index: int,
    shock_index: int,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair ensemble draw ``i`` with SVAR IRF draw ``i`` and add the shock term.

    The scenario shock is realized at ``t+1`` because ``compute_shock_size``
    measures the user's value against the SVAR's horizon-1 macro forecast. CPI
    horizon 1 is also ``t+1``, so it receives the contemporaneous impact
    stored at IRF index 0; CPI horizon ``h`` receives IRF index ``h - 1``.
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
    target: ScenarioTarget,
    shock_variable: str,
    shock_value: float,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_sims: int = DEFAULT_N_SIMS,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float = DEFAULT_UPPER_QUANTILE,
    seed: int = DEFAULT_SEED,
    curated_path: Path = CURATED_DATA_PATH,
) -> ScenarioResult:
    """Run one headline or trimmed-mean SVAR-adjusted ensemble scenario."""
    config = _target_config(target)
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
    svar_frame = _svar_frame(frame, config.svar_columns)
    fitted_svar = svar.fit_svar(
        svar_frame,
        lag_order=svar.DEFAULT_SVAR_LAG_ORDER,
        ordering=config.svar_ordering,
    )
    svar_forecast = svar.forecast_from_fit(fitted_svar, steps=steps)
    shock_size = compute_shock_size(
        svar_forecast=svar_forecast,
        shock_variable=shock_variable,
        shock_value=shock_value,
    )

    weights = _ensemble_weights(config, steps)
    tier1_paths = ensemble.simulate_ensemble_paths(
        frame,
        steps=steps,
        n_sims=n_sims,
        weights=weights,
        seed=seed,
        target_column=config.target_column,
        sarima_order=config.sarima_order,
        sarima_seasonal_order=config.sarima_seasonal_order,
        elastic_net_feature_columns=config.elastic_net_feature_columns,
        sarima_series=frame[config.target_column],
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
        response_index=names.index(config.target_column),
        shock_index=names.index(shock_variable),
        horizons=requested_horizons,
    )
    lower, upper = scenario_interval_from_paths(
        combined_paths,
        lower=lower_quantile,
        upper=upper_quantile,
    )
    clean_frame = _complete_ensemble_frame(frame, config).sort_index()
    forecast_origin = str(clean_frame.index[-1])

    return ScenarioResult(
        target=config.target,
        target_column=config.target_column,
        shock_variable=str(shock_variable),
        shock_value=float(shock_value),
        shock_size=float(shock_size),
        horizons=requested_horizons,
        forecast=np.mean(combined_paths, axis=0).astype(float).tolist(),
        interval_lower=lower.astype(float).tolist(),
        interval_upper=upper.astype(float).tolist(),
        quarters=_future_quarters(forecast_origin, requested_horizons),
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


def _target_config(target: str) -> ScenarioTargetConfig:
    try:
        return TARGET_CONFIGS[str(target)]
    except KeyError as exc:
        raise ValueError("target must be 'headline' or 'trimmed_mean'.") from exc


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
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
    data = pd.read_csv(path)
    if "quarter" not in data.columns:
        raise ValueError(f"{path} must contain a 'quarter' column.")
    data = data.copy()
    data.index = pd.PeriodIndex(data.pop("quarter").astype(str), freq="Q")
    return data.sort_index()


def _svar_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"curated frame missing required SVAR columns: {missing}")
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").dropna()


def _complete_ensemble_frame(
    frame: pd.DataFrame,
    config: ScenarioTargetConfig,
) -> pd.DataFrame:
    columns = (config.target_column, *config.elastic_net_feature_columns)
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"curated frame missing required ensemble columns: {missing}")
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").dropna()


def _ensemble_weights(
    config: ScenarioTargetConfig,
    steps: int,
) -> dict[int, tuple[float, float]]:
    horizons = tuple(range(1, steps + 1))
    if config.dynamic_weights:
        return ensemble.horizon_rmse_weights(horizons=horizons)
    return {horizon: ensemble.DEFAULT_WEIGHTS for horizon in horizons}


def _future_quarters(forecast_origin: str, horizons: Sequence[int]) -> list[str]:
    origin = pd.Period(forecast_origin, freq="Q")
    return [str(origin + int(horizon)) for horizon in horizons]
