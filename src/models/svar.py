"""Five-variable SVAR utilities for CPI scenario and impulse-response work.

The Phase 1a gate checks live here alongside the Phase 1b VAR(2)-in-levels
wrappers, recursive Cholesky IRFs, bootstrap bands, diagnostics, and primary
System B walk-forward diagnostic backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.models.evaluation import (
    DEFAULT_HORIZONS,
    DEFAULT_INITIAL_TRAIN_SIZE,
    compute_metric_table,
    walk_forward_backtest_direct_multihorizon,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURATED_QUARTERLY_PATH = PROJECT_ROOT / "data/curated/quarterly_macro_features.parquet"

COMMON_MACRO_COLUMNS = (
    "unemployment_rate",
    "cash_rate",
    "commodity_growth",
    "inflation_expectations_business",
)
SYSTEM_A_COLUMNS = ("cpi_yoy", *COMMON_MACRO_COLUMNS)
SYSTEM_B_COLUMNS = ("trimmed_mean_cpi_yoy", *COMMON_MACRO_COLUMNS)
SYSTEM_A_CHOLESKY_ORDER = (
    "commodity_growth",
    "unemployment_rate",
    "cpi_yoy",
    "inflation_expectations_business",
    "cash_rate",
)
SYSTEM_B_CHOLESKY_ORDER = (
    "commodity_growth",
    "unemployment_rate",
    "trimmed_mean_cpi_yoy",
    "inflation_expectations_business",
    "cash_rate",
)
SHARED_ESCALATION_SHOCKS = (
    "unemployment_rate",
    "cash_rate",
    "commodity_growth",
    "inflation_expectations_business",
)

DEFAULT_MIN_OBS_PER_PARAMETER = 10.0
DEFAULT_DET_ORDER = 0
DEFAULT_JOHANSEN_SIGNIFICANCE_COLUMN = 1  # 5% column in statsmodels' cvt output.
DEFAULT_UNIT_ROOT_SIGNIFICANCE = 0.05
DEFAULT_STABILITY_MARGIN = 0.02
DEFAULT_SVAR_LAG_ORDER = 2
DEFAULT_LOWER_QUANTILE = 0.1
DEFAULT_UPPER_QUANTILE = 0.9
DEFAULT_BOOTSTRAP_REPLICATIONS = 1000
DEFAULT_ARCH_LAGS = 4
DEFAULT_WHITENESS_NLAGS = 10


@dataclass(frozen=True)
class LagOrderDecision:
    """BIC lag choice after applying the requested DoF cap."""

    selected_order: int
    max_lag_dof_cap: int
    hit_dof_cap: bool
    n_obs_complete: int
    n_vars: int
    min_obs_per_parameter: float
    selected_usable_obs: int
    selected_obs_per_parameter: float
    bic_by_lag: Mapping[int, float]


@dataclass(frozen=True)
class UnitRootPretestResult:
    """Per-series ADF pretest used to corroborate Johansen rank interpretation."""

    variable: str
    adf_statistic: float
    p_value: float
    used_lag: int
    n_obs: int
    critical_value_5pct: float
    reject_unit_root_5pct: bool


@dataclass(frozen=True)
class CointegrationDecision:
    """Johansen trace-test result mapped to a Phase 1b model-family decision."""

    rank: int
    n_vars: int
    det_order: int
    k_ar_diff: int
    trace_statistics: tuple[float, ...]
    critical_values_5pct: tuple[float, ...]
    phase_1b_model: str
    unit_root_pretests: Mapping[str, UnitRootPretestResult]


@dataclass(frozen=True)
class StabilityDecision:
    """Stability corroboration from a fitted positive-lag VAR."""

    is_stable: bool
    stable_with_margin: bool
    stability_margin: float
    max_companion_eigenvalue_modulus: float
    companion_eigenvalue_moduli: tuple[float, ...]
    inverse_root_moduli: tuple[float, ...]


@dataclass(frozen=True)
class SVARDiagnostics:
    """Diagnostics reported for systems receiving full Phase 1b treatment."""

    multivariate_tests: pd.DataFrame
    equation_tests: pd.DataFrame


@dataclass(frozen=True)
class SVARBacktestResult:
    """System-level point-forecast backtest and metric summary."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame


def load_svar_level_frame(
    columns: Sequence[str],
    path: Path = CURATED_QUARTERLY_PATH,
) -> pd.DataFrame:
    """Load complete level observations for one SVAR system."""

    data = pd.read_parquet(path)
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError(f"{path} missing required SVAR columns: {missing}")
    frame = data.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if "quarter" in data.columns:
        frame.index = pd.PeriodIndex(data["quarter"], freq="Q")
    return frame.dropna()


def calculate_dof_lag_cap(
    n_obs: int,
    n_vars: int,
    min_obs_per_parameter: float = DEFAULT_MIN_OBS_PER_PARAMETER,
) -> int:
    """Return the largest positive lag satisfying (n_obs - p) / (n_vars * p)."""

    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    if n_vars <= 0:
        raise ValueError("n_vars must be positive.")
    if min_obs_per_parameter <= 0:
        raise ValueError("min_obs_per_parameter must be positive.")

    max_lag = 0
    for lag_order in range(1, n_obs):
        usable_obs = n_obs - lag_order
        obs_per_parameter = usable_obs / (n_vars * lag_order)
        if obs_per_parameter >= min_obs_per_parameter:
            max_lag = lag_order
        else:
            break
    if max_lag < 1:
        raise ValueError(
            "No positive lag order satisfies the DoF rule for "
            f"n_obs={n_obs}, n_vars={n_vars}, "
            f"min_obs_per_parameter={min_obs_per_parameter}."
        )
    return max_lag


def select_bic_lag_order(
    data: pd.DataFrame,
    min_obs_per_parameter: float = DEFAULT_MIN_OBS_PER_PARAMETER,
    trend: str = "c",
) -> LagOrderDecision:
    """Select a positive VAR lag by BIC after capping candidates by DoF."""

    frame = _validate_var_frame(data)
    n_obs, n_vars = frame.shape
    max_lag = calculate_dof_lag_cap(n_obs, n_vars, min_obs_per_parameter)

    selection = VAR(frame).select_order(maxlags=max_lag, trend=trend)
    bic_by_lag = _bic_values_by_positive_lag(frame, selection, max_lag, trend)
    selected_order = min(bic_by_lag, key=bic_by_lag.__getitem__)
    selected_usable_obs = n_obs - selected_order
    selected_obs_per_parameter = selected_usable_obs / (n_vars * selected_order)

    return LagOrderDecision(
        selected_order=selected_order,
        max_lag_dof_cap=max_lag,
        hit_dof_cap=selected_order == max_lag,
        n_obs_complete=n_obs,
        n_vars=n_vars,
        min_obs_per_parameter=float(min_obs_per_parameter),
        selected_usable_obs=selected_usable_obs,
        selected_obs_per_parameter=float(selected_obs_per_parameter),
        bic_by_lag=bic_by_lag,
    )


def johansen_cointegration_check(
    data: pd.DataFrame,
    var_lag_order: int,
    det_order: int = DEFAULT_DET_ORDER,
    significance_column: int = DEFAULT_JOHANSEN_SIGNIFICANCE_COLUMN,
) -> CointegrationDecision:
    """Run the Johansen trace test using the VECM lag implied by VAR(p)."""

    if var_lag_order < 1:
        raise ValueError("var_lag_order must be a positive integer.")
    if significance_column != DEFAULT_JOHANSEN_SIGNIFICANCE_COLUMN:
        raise ValueError("Only the 5% Johansen critical-value column is supported.")

    frame = _validate_var_frame(data)
    n_vars = frame.shape[1]
    k_ar_diff = var_lag_order - 1
    result = coint_johansen(frame, det_order=det_order, k_ar_diff=k_ar_diff)
    trace_statistics = tuple(float(value) for value in result.lr1)
    critical_values_5pct = tuple(float(value) for value in result.cvt[:, significance_column])
    rank = min(_sequential_trace_rank(trace_statistics, critical_values_5pct), n_vars)
    phase_1b_model = _phase_1b_model_from_rank(rank, n_vars)

    return CointegrationDecision(
        rank=rank,
        n_vars=n_vars,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        trace_statistics=trace_statistics,
        critical_values_5pct=critical_values_5pct,
        phase_1b_model=phase_1b_model,
        unit_root_pretests=adf_unit_root_pretests(frame),
    )


def adf_unit_root_pretests(
    data: pd.DataFrame,
    significance_level: float = DEFAULT_UNIT_ROOT_SIGNIFICANCE,
) -> dict[str, UnitRootPretestResult]:
    """Run ADF unit-root pretests for each series in a VAR system."""

    frame = _validate_var_frame(data)
    results = {}
    for column in frame.columns:
        statistic, p_value, used_lag, n_obs, critical_values, *_ = adfuller(
            frame[column],
            autolag="AIC",
        )
        results[str(column)] = UnitRootPretestResult(
            variable=str(column),
            adf_statistic=float(statistic),
            p_value=float(p_value),
            used_lag=int(used_lag),
            n_obs=int(n_obs),
            critical_value_5pct=float(critical_values["5%"]),
            reject_unit_root_5pct=bool(p_value < significance_level),
        )
    return results


def check_var_stability(
    data: pd.DataFrame,
    lag_order: int,
    trend: str = "c",
    stability_margin: float = DEFAULT_STABILITY_MARGIN,
) -> StabilityDecision:
    """Fit a positive-lag VAR and report companion-matrix stability."""

    if lag_order < 1:
        raise ValueError("lag_order must be a positive integer.")
    if stability_margin < 0:
        raise ValueError("stability_margin must be non-negative.")

    frame = _validate_var_frame(data)
    fitted = VAR(frame).fit(lag_order, trend=trend)
    inverse_roots = np.asarray(fitted.roots, dtype=complex)
    companion_eigenvalues = 1 / inverse_roots
    companion_moduli = tuple(float(value) for value in np.sort(np.abs(companion_eigenvalues)))
    inverse_root_moduli = tuple(float(value) for value in np.sort(np.abs(inverse_roots)))
    max_companion_modulus = float(max(companion_moduli))

    return StabilityDecision(
        is_stable=bool(fitted.is_stable(verbose=False)),
        stable_with_margin=bool(max_companion_modulus <= 1.0 - stability_margin),
        stability_margin=float(stability_margin),
        max_companion_eigenvalue_modulus=max_companion_modulus,
        companion_eigenvalue_moduli=companion_moduli,
        inverse_root_moduli=inverse_root_moduli,
    )


def fit_svar(
    data: pd.DataFrame,
    lag_order: int = DEFAULT_SVAR_LAG_ORDER,
    ordering: Sequence[str] | None = None,
    trend: str = "c",
):
    """Fit the Phase 1b levels VAR using the supplied recursive ordering."""
    if lag_order < 1:
        raise ValueError("lag_order must be a positive integer.")

    frame = _prepare_ordered_var_frame(data, ordering=ordering)
    return VAR(frame).fit(lag_order, trend=trend)


def forecast_from_fit(fitted, steps: int = max(DEFAULT_HORIZONS)) -> pd.DataFrame:
    """Forecast all variables from an already-fitted VAR result."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    history = np.asarray(fitted.endog, dtype=float)[-fitted.k_ar :]
    forecast = fitted.forecast(y=history, steps=steps)
    return pd.DataFrame(forecast, index=_future_index(fitted, steps), columns=list(fitted.names))


def forecast_svar(
    data: pd.DataFrame,
    steps: int = max(DEFAULT_HORIZONS),
    lag_order: int = DEFAULT_SVAR_LAG_ORDER,
    ordering: Sequence[str] | None = None,
    trend: str = "c",
) -> pd.DataFrame:
    """Fit a levels VAR and return ``steps`` out-of-sample forecasts."""
    fitted = fit_svar(data=data, lag_order=lag_order, ordering=ordering, trend=trend)
    return forecast_from_fit(fitted=fitted, steps=steps)


def simulate_paths_from_fit(
    fitted,
    steps: int = max(DEFAULT_HORIZONS),
    n_sims: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    seed: int = 42,
) -> np.ndarray:
    """Simulate future paths from an already-fitted VAR.

    Returns an array with shape ``(n_sims, steps, n_variables)`` in the fitted
    column order.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1.")

    paths = fitted.simulate_var(
        steps=steps + fitted.k_ar,
        nsimulations=n_sims,
        seed=seed,
        initial_values=np.asarray(fitted.endog, dtype=float)[-fitted.k_ar :],
    )
    paths = np.asarray(paths, dtype=float)[:, fitted.k_ar :, :]
    expected_shape = (n_sims, steps, len(fitted.names))
    if paths.shape != expected_shape:
        raise ValueError(
            f"simulation paths have shape {paths.shape}, expected {expected_shape}."
        )
    return paths


def simulate_svar_paths(
    data: pd.DataFrame,
    steps: int = max(DEFAULT_HORIZONS),
    n_sims: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    lag_order: int = DEFAULT_SVAR_LAG_ORDER,
    ordering: Sequence[str] | None = None,
    trend: str = "c",
    seed: int = 42,
) -> np.ndarray:
    """Fit the levels VAR once and simulate future paths from the fitted result."""
    fitted = fit_svar(data=data, lag_order=lag_order, ordering=ordering, trend=trend)
    return simulate_paths_from_fit(fitted=fitted, steps=steps, n_sims=n_sims, seed=seed)


def recursive_cholesky_irfs(
    fitted,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Return orthogonalized recursive Cholesky IRFs in long form."""
    requested_horizons = _validate_positive_horizons(horizons)
    periods = max(requested_horizons)
    irfs = np.asarray(fitted.irf(periods=periods).orth_irfs, dtype=float)
    return _irf_array_to_frame(
        irfs=irfs,
        names=tuple(str(name) for name in fitted.names),
        horizons=requested_horizons,
        value_column="irf",
    )


def bootstrap_cholesky_irf_bands(
    fitted,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float = DEFAULT_UPPER_QUANTILE,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute residual-bootstrap 80% recursive Cholesky IRF bands.

    The original IRF is computed with ``orth=True`` through statsmodels'
    ``orth_irfs`` output. Bootstrap samples resample fitted residual vectors
    jointly, recursively simulate a pseudo history with the fitted VAR
    coefficients, refit the same lag-order VAR, and take pointwise quantiles.
    """
    requested_horizons = _validate_positive_horizons(horizons)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("lower_quantile and upper_quantile must satisfy 0 <= lower < upper <= 1.")

    original = recursive_cholesky_irfs(fitted=fitted, horizons=requested_horizons)
    boot_irfs = _bootstrap_orth_irf_array(
        fitted=fitted,
        horizons=requested_horizons,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    lower = np.quantile(boot_irfs, lower_quantile, axis=0)
    upper = np.quantile(boot_irfs, upper_quantile, axis=0)
    lower_frame = _irf_array_to_frame(
        irfs=lower,
        names=tuple(str(name) for name in fitted.names),
        horizons=requested_horizons,
        value_column="lower",
    )
    upper_frame = _irf_array_to_frame(
        irfs=upper,
        names=tuple(str(name) for name in fitted.names),
        horizons=requested_horizons,
        value_column="upper",
    )
    bands = original.merge(lower_frame, on=["response", "shock", "horizon"], how="left")
    bands = bands.merge(upper_frame, on=["response", "shock", "horizon"], how="left")
    bands["lower_quantile"] = float(lower_quantile)
    bands["upper_quantile"] = float(upper_quantile)
    bands["bootstrap_replications"] = int(n_bootstrap)
    return bands


def bootstrap_orth_irf_draws(
    fitted,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    seed: int = 42,
) -> np.ndarray:
    """Return raw residual-bootstrap orthogonalized IRF draws.

    The output has shape ``(n_bootstrap, max(horizons) + 1, n_vars, n_vars)``.
    The second axis keeps statsmodels' period-zero IRF at index 0, so a
    requested positive horizon ``h`` is read from ``draws[:, h, :, :]``.
    """
    requested_horizons = _validate_positive_horizons(horizons)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    return _bootstrap_orth_irf_array(
        fitted=fitted,
        horizons=requested_horizons,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def compare_system_a_to_b_irf_bands(
    system_a_bands: pd.DataFrame,
    system_b_bands: pd.DataFrame,
    shared_shocks: Sequence[str] = SHARED_ESCALATION_SHOCKS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    system_a_response: str = "cpi_yoy",
    system_b_response: str = "trimmed_mean_cpi_yoy",
) -> pd.DataFrame:
    """Compare 80% IRF bands for the fixed Phase 1b System A escalation rule."""
    requested_horizons = _validate_positive_horizons(horizons)
    a = _select_irf_band_rows(
        system_a_bands,
        response=system_a_response,
        shocks=shared_shocks,
        horizons=requested_horizons,
        system_label="system_a",
    )
    b = _select_irf_band_rows(
        system_b_bands,
        response=system_b_response,
        shocks=shared_shocks,
        horizons=requested_horizons,
        system_label="system_b",
    )
    comparison = a.merge(b, on=["shock", "horizon"], how="outer", validate="one_to_one")
    required = ["system_a_lower", "system_a_upper", "system_b_lower", "system_b_upper"]
    if comparison[required].isna().any(axis=None):
        missing = comparison.loc[comparison[required].isna().any(axis=1), ["shock", "horizon"]]
        raise ValueError(f"missing IRF band rows for escalation comparison: {missing.to_dict('records')}")

    comparison["non_overlapping"] = (
        comparison["system_a_upper"].astype(float) < comparison["system_b_lower"].astype(float)
    ) | (
        comparison["system_b_upper"].astype(float) < comparison["system_a_lower"].astype(float)
    )
    comparison["escalate_system_a"] = comparison["non_overlapping"].astype(bool)
    return comparison.sort_values(["shock", "horizon"]).reset_index(drop=True)


def system_a_requires_full_treatment(comparison: pd.DataFrame) -> bool:
    """Return whether any shared-shock horizon triggers System A escalation."""
    if "escalate_system_a" not in comparison:
        raise ValueError("comparison must contain an 'escalate_system_a' column.")
    return bool(comparison["escalate_system_a"].astype(bool).any())


def svar_diagnostics(
    fitted,
    whiteness_nlags: int = DEFAULT_WHITENESS_NLAGS,
    arch_lags: int = DEFAULT_ARCH_LAGS,
    significance_level: float = 0.05,
) -> SVARDiagnostics:
    """Run full VAR diagnostics for a fitted System B or escalated System A."""
    if whiteness_nlags <= fitted.k_ar:
        raise ValueError("whiteness_nlags must exceed the fitted VAR lag order.")
    if arch_lags < 1:
        raise ValueError("arch_lags must be at least 1.")

    whiteness = fitted.test_whiteness(nlags=whiteness_nlags, signif=significance_level)
    multivariate_rows = [_hypothesis_test_row("portmanteau_whiteness", whiteness)]
    normality = fitted.test_normality(signif=significance_level)
    multivariate_rows.append(_hypothesis_test_row("multivariate_normality", normality))

    equation_rows: list[dict[str, object]] = []
    residuals = pd.DataFrame(fitted.resid, columns=list(fitted.names)).astype(float)
    for variable in residuals.columns:
        jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals[variable])
        equation_rows.append(
            {
                "variable": variable,
                "test": "jarque_bera_normality",
                "statistic": float(jb_stat),
                "p_value": float(jb_pvalue),
                "skew": float(skew),
                "kurtosis": float(kurtosis),
                "lags": np.nan,
                "reject_5pct": bool(jb_pvalue < significance_level),
            }
        )
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals[variable], nlags=arch_lags)
        equation_rows.append(
            {
                "variable": variable,
                "test": "arch_lm_heteroskedasticity",
                "statistic": float(lm_stat),
                "p_value": float(lm_pvalue),
                "f_statistic": float(f_stat),
                "f_p_value": float(f_pvalue),
                "skew": np.nan,
                "kurtosis": np.nan,
                "lags": int(arch_lags),
                "reject_5pct": bool(lm_pvalue < significance_level),
            }
        )

    return SVARDiagnostics(
        multivariate_tests=pd.DataFrame(multivariate_rows),
        equation_tests=pd.DataFrame(equation_rows),
    )


def walk_forward_svar_backtest(
    data: pd.DataFrame,
    target_column: str,
    ordering: Sequence[str],
    lag_order: int = DEFAULT_SVAR_LAG_ORDER,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    trend: str = "c",
    model_name: str = "svar",
) -> SVARBacktestResult:
    """Run a diagnostic expanding-window point forecast backtest for one SVAR."""
    frame = _prepare_ordered_var_frame(data, ordering=ordering)
    if target_column not in frame.columns:
        raise ValueError(f"target_column {target_column!r} is not present in data.")
    exog_columns = tuple(column for column in frame.columns if column != target_column)
    predictions = walk_forward_backtest_direct_multihorizon(
        series=frame[target_column],
        exog=frame.loc[:, list(exog_columns)],
        forecast_func=lambda train_frame, steps: forecast_svar(
            train_frame,
            steps=steps,
            lag_order=lag_order,
            ordering=ordering,
            trend=trend,
        )[target_column],
        initial_train_size=initial_train_size,
        horizons=horizons,
        model_name=model_name,
        target_column=target_column,
    )
    return SVARBacktestResult(
        predictions=predictions,
        metrics=compute_metric_table(predictions),
    )


def _validate_var_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty:
        raise ValueError("SVAR gate checks require at least one complete observation.")
    if frame.shape[1] < 2:
        raise ValueError("VAR checks require at least two variables.")
    return frame


def _prepare_ordered_var_frame(
    data: pd.DataFrame,
    ordering: Sequence[str] | None = None,
) -> pd.DataFrame:
    frame = _validate_var_frame(pd.DataFrame(data))
    if ordering is None:
        return frame

    requested = tuple(str(column) for column in ordering)
    missing = sorted(set(requested).difference(frame.columns))
    if missing:
        raise ValueError(f"VAR frame missing requested ordering columns: {missing}")
    return frame.loc[:, list(requested)]


def _validate_positive_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    if not requested_horizons or min(requested_horizons) < 1:
        raise ValueError("horizons must contain positive integers.")
    return requested_horizons


def _future_index(fitted, steps: int) -> pd.Index:
    labels = getattr(fitted.model.data, "row_labels", None)
    if labels is None or len(labels) == 0:
        return pd.RangeIndex(start=0, stop=steps)
    index = pd.Index(labels)
    last = index[-1]
    if isinstance(index, pd.PeriodIndex):
        return pd.period_range(last + 1, periods=steps, freq=index.freq or "Q")
    if isinstance(index, pd.DatetimeIndex) and index.freq is not None:
        return pd.date_range(last + index.freq, periods=steps, freq=index.freq)
    return pd.RangeIndex(start=len(index), stop=len(index) + steps)


def _irf_array_to_frame(
    irfs: np.ndarray,
    names: Sequence[str],
    horizons: Sequence[int],
    value_column: str,
) -> pd.DataFrame:
    values = np.asarray(irfs, dtype=float)
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        if horizon >= values.shape[0]:
            raise ValueError(
                f"IRF array has only {values.shape[0] - 1} positive horizons; "
                f"horizon {horizon} was requested."
            )
        for response_index, response in enumerate(names):
            for shock_index, shock in enumerate(names):
                rows.append(
                    {
                        "response": response,
                        "shock": shock,
                        "horizon": int(horizon),
                        value_column: float(values[horizon, response_index, shock_index]),
                    }
                )
    return pd.DataFrame(rows)


def _bootstrap_orth_irf_array(
    fitted,
    horizons: Sequence[int],
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    residuals = np.asarray(fitted.resid, dtype=float)
    endog = np.asarray(fitted.endog, dtype=float)
    residual_draw_length = len(endog) - fitted.k_ar
    draws: list[np.ndarray] = []
    max_attempts = max(n_bootstrap * 3, n_bootstrap + 10)
    attempts = 0
    while len(draws) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        residual_draws = _draw_contiguous_residual_blocks(
            residuals=residuals,
            draw_length=residual_draw_length,
            block_length=DEFAULT_ARCH_LAGS,
            rng=rng,
        )
        simulated = _simulate_var_history_with_residuals(
            fitted=fitted,
            initial_values=endog[: fitted.k_ar],
            residual_draws=residual_draws,
        )
        try:
            refit = VAR(
                pd.DataFrame(
                    simulated,
                    columns=list(fitted.names),
                    index=getattr(fitted.model.data, "row_labels", None),
                )
            ).fit(fitted.k_ar, trend=fitted.trend)
            draws.append(np.asarray(refit.irf(periods=max(horizons)).orth_irfs, dtype=float))
        except (np.linalg.LinAlgError, ValueError):
            continue
    if len(draws) < n_bootstrap:
        raise RuntimeError(
            f"only {len(draws)} successful bootstrap VAR refits after {attempts} attempts."
        )
    return np.stack(draws, axis=0)


def _draw_contiguous_residual_blocks(
    residuals: np.ndarray,
    draw_length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw non-wrapping contiguous residual blocks and trim to exact length."""
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 2:
        raise ValueError("residuals must be a 2D array.")
    if draw_length < 1:
        raise ValueError("draw_length must be at least 1.")
    if block_length < 1:
        raise ValueError("block_length must be at least 1.")
    if len(values) < block_length:
        raise ValueError("block_length cannot exceed the residual history length.")

    max_start = len(values) - block_length
    n_blocks = int(np.ceil(draw_length / block_length))
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    blocks = [values[start : start + block_length] for start in starts]
    return np.vstack(blocks)[:draw_length]


def _simulate_var_history_with_residuals(
    fitted,
    initial_values: np.ndarray,
    residual_draws: np.ndarray,
) -> np.ndarray:
    initial = np.asarray(initial_values, dtype=float)
    residuals = np.asarray(residual_draws, dtype=float)
    n_obs = initial.shape[0] + residuals.shape[0]
    values = np.zeros((n_obs, initial.shape[1]), dtype=float)
    values[: fitted.k_ar] = initial
    intercept = np.asarray(fitted.intercept, dtype=float)
    for t in range(fitted.k_ar, n_obs):
        predicted = intercept.copy()
        for lag in range(1, fitted.k_ar + 1):
            predicted += np.asarray(fitted.coefs[lag - 1], dtype=float) @ values[t - lag]
        values[t] = predicted + residuals[t - fitted.k_ar]
    return values


def _select_irf_band_rows(
    bands: pd.DataFrame,
    response: str,
    shocks: Sequence[str],
    horizons: Sequence[int],
    system_label: str,
) -> pd.DataFrame:
    required = {"response", "shock", "horizon", "lower", "upper"}
    missing = required.difference(bands.columns)
    if missing:
        raise ValueError(f"{system_label} bands missing required columns: {sorted(missing)}")
    selected = bands.loc[
        bands["response"].eq(response)
        & bands["shock"].isin(tuple(shocks))
        & bands["horizon"].astype(int).isin(tuple(horizons)),
        ["shock", "horizon", "lower", "upper"],
    ].copy()
    selected["horizon"] = selected["horizon"].astype(int)
    selected = selected.rename(
        columns={
            "lower": f"{system_label}_lower",
            "upper": f"{system_label}_upper",
        }
    )
    return selected


def _hypothesis_test_row(test_name: str, result) -> dict[str, object]:
    p_value = float(getattr(result, "pvalue"))
    return {
        "test": test_name,
        "statistic": float(getattr(result, "test_statistic")),
        "critical_value": float(getattr(result, "crit_value")),
        "p_value": p_value,
        "df": int(getattr(result, "df")),
        "reject_5pct": bool(p_value < 0.05),
        "conclusion": str(getattr(result, "conclusion")),
    }


def _bic_values_by_positive_lag(
    frame: pd.DataFrame,
    selection,
    max_lag: int,
    trend: str,
) -> dict[int, float]:
    raw_bic = selection.ics.get("bic", []) if hasattr(selection, "ics") else []
    bic_by_lag = {
        lag: float(value)
        for lag, value in enumerate(raw_bic)
        if 1 <= lag <= max_lag and np.isfinite(value)
    }
    if bic_by_lag:
        return bic_by_lag

    return {
        lag: float(VAR(frame).fit(lag, trend=trend).bic)
        for lag in range(1, max_lag + 1)
    }


def _sequential_trace_rank(
    trace_statistics: Sequence[float],
    critical_values: Sequence[float],
) -> int:
    rank = 0
    for statistic, critical_value in zip(trace_statistics, critical_values):
        if statistic > critical_value:
            rank += 1
        else:
            break
    return rank


def _phase_1b_model_from_rank(rank: int, n_vars: int) -> str:
    if rank < 0 or rank > n_vars:
        raise ValueError(f"rank must be between 0 and n_vars; got rank={rank}, n_vars={n_vars}.")
    if 0 < rank < n_vars:
        return "vecm"
    return "levels_var_svar"
