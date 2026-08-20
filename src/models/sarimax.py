"""Reusable SARIMAX model helpers for quarterly Australian CPI inflation."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.evaluation import (
    CURATED_DATA_PATH,
    PROJECT_ROOT,
    TARGET_COLUMN,
    load_target_series,
)
from src.models.sarima import DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER
from src.models.sarima import _simulation_result_to_paths


# Selected by the AIC order search in `src.models.sarimax_order_search` on
# development-only data for Group D ("full core"); these fixed values are
# recorded here for serving registration, not re-derived at import/runtime.
GROUP_D_FEATURE_COLUMNS = (
    "cash_rate_change_lag1",
    "unemployment_rate_change_lag1",
    "inflation_expectations_business_lag1",
    "ppi_growth_lag2",
    "commodity_growth_lag1",
    "wti_growth_lag1",
)
GROUP_D_ORDER = (1, 0, 1)
GROUP_D_SEASONAL_ORDER = (0, 0, 1, 4)
GROUP_D_HORIZON_CAP_ERROR = "SARIMAX Group D is lag-safety capped at horizon 1"
SARIMAX_COMPARISON_REPORT_PATH = PROJECT_ROOT / "reports/model_comparison_sarimax.csv"


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


def _prepare_future_exog(exog: pd.DataFrame, future_exog: pd.DataFrame, steps: int) -> pd.DataFrame:
    future = pd.DataFrame(future_exog).iloc[:steps].astype(float)
    if len(future) < steps:
        raise ValueError("future_exog must contain at least ``steps`` rows.")
    if future.isna().any(axis=None):
        raise ValueError("future_exog must not contain missing values.")

    train_columns = list(pd.DataFrame(exog).columns)
    return future.loc[:, train_columns]


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

    future = _prepare_future_exog(exog=exog, future_exog=future_exog, steps=steps)

    fitted = fit_sarimax(
        series=series,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        maxiter=maxiter,
    )
    forecast = fitted.get_forecast(steps=steps, exog=future).predicted_mean
    return pd.Series(forecast, name="forecast")


def simulate_paths_from_fit(
    fitted,
    future_exog: pd.DataFrame,
    steps: int = 8,
    n_sims: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate future ``cpi_yoy`` paths from an already-fitted SARIMAX model."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1.")

    future = pd.DataFrame(future_exog).iloc[:steps].astype(float)
    if len(future) < steps:
        raise ValueError("future_exog must contain at least ``steps`` rows.")
    if future.isna().any(axis=None):
        raise ValueError("future_exog must not contain missing values.")

    # Innovation uncertainty only; this does not include parameter uncertainty.
    simulated = fitted.simulate(
        nsimulations=steps,
        anchor="end",
        repetitions=n_sims,
        exog=future,
        random_state=seed,
    )
    return _simulation_result_to_paths(simulated, n_sims=n_sims, steps=steps)


def simulate_sarimax_paths(
    series: pd.Series,
    exog: pd.DataFrame,
    future_exog: pd.DataFrame,
    steps: int = 8,
    n_sims: int = 1000,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
    trend: str = "n",
    maxiter: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Fit SARIMAX and simulate future ``cpi_yoy`` paths with supplied exog."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1.")

    future = _prepare_future_exog(exog=exog, future_exog=future_exog, steps=steps)
    fitted = fit_sarimax(
        series=series,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        maxiter=maxiter,
    )
    return simulate_paths_from_fit(fitted, future, steps=steps, n_sims=n_sims, seed=seed)


def group_d_future_exog(curated_frame: pd.DataFrame) -> pd.DataFrame:
    """One-row future exog for Group D's horizon-1 forecast.

    Every Group D feature is a fixed lag of an unlagged column already
    present in the curated dataset, so next quarter's lagged value is
    just this quarter's already-known reading -- shifted forward
    mechanically, not forecast.
    """
    sorted_frame = pd.DataFrame(curated_frame).sort_index()
    latest = sorted_frame.iloc[-1]
    return pd.DataFrame(
        [
            {
                "cash_rate_change_lag1": latest["cash_rate_change"],
                "unemployment_rate_change_lag1": latest["unemployment_rate_change"],
                "inflation_expectations_business_lag1": latest[
                    "inflation_expectations_business"
                ],
                "ppi_growth_lag2": latest["ppi_growth_lag1"],
                "commodity_growth_lag1": latest["commodity_growth"],
                "wti_growth_lag1": latest["wti_growth"],
            }
        ],
        index=[sorted_frame.index[-1] + 1],
    )


def _group_d_frame(series: pd.Series, exog: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [pd.Series(series).rename(TARGET_COLUMN), pd.DataFrame(exog)],
        axis=1,
    )


def forecast_sarimax_group_d(
    series: pd.Series,
    exog: pd.DataFrame,
    steps: int = 1,
) -> pd.Series:
    """Fit fixed SARIMAX Group D and return its horizon-1-only point forecast.

    ``exog`` must be the wide curated macro frame, not only
    ``GROUP_D_FEATURE_COLUMNS``, because the horizon-1 future row is built from
    the latest already-observed unlagged/base columns.
    """
    if steps != 1:
        raise ValueError(GROUP_D_HORIZON_CAP_ERROR)

    curated_frame = _group_d_frame(series, exog)
    future = group_d_future_exog(curated_frame).loc[:, list(GROUP_D_FEATURE_COLUMNS)]
    train_exog = pd.DataFrame(exog).loc[:, list(GROUP_D_FEATURE_COLUMNS)]
    return forecast_sarimax(
        series=series,
        exog=train_exog,
        future_exog=future,
        steps=steps,
        order=GROUP_D_ORDER,
        seasonal_order=GROUP_D_SEASONAL_ORDER,
    )


def simulate_sarimax_group_d_paths(
    series: pd.Series,
    exog: pd.DataFrame,
    steps: int = 1,
    n_sims: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Fit fixed SARIMAX Group D and simulate horizon-1-only paths.

    ``exog`` must be the wide curated macro frame, not only
    ``GROUP_D_FEATURE_COLUMNS``, because the horizon-1 future row is built from
    the latest already-observed unlagged/base columns.
    """
    if steps != 1:
        raise ValueError(GROUP_D_HORIZON_CAP_ERROR)

    curated_frame = _group_d_frame(series, exog)
    future = group_d_future_exog(curated_frame).loc[:, list(GROUP_D_FEATURE_COLUMNS)]
    train_exog = pd.DataFrame(exog).loc[:, list(GROUP_D_FEATURE_COLUMNS)]
    fitted = fit_sarimax(
        series=series,
        exog=train_exog,
        order=GROUP_D_ORDER,
        seasonal_order=GROUP_D_SEASONAL_ORDER,
    )
    return simulate_paths_from_fit(fitted, future, steps=steps, n_sims=n_sims, seed=seed)


def _load_existing_sarimax_d_metrics(path: Path = SARIMAX_COMPARISON_REPORT_PATH) -> pd.DataFrame:
    """Load canonical SARIMAX(D) metrics from the existing SARIMAX report, if present."""
    if not path.exists():
        return pd.DataFrame()
    report = pd.read_csv(path)
    required_columns = {"group_id", "model", "horizon", "n", "rmse", "mae"}
    if not required_columns.issubset(report.columns):
        return pd.DataFrame()
    rows = report.loc[report["group_id"].eq("D") & report["model"].eq("sarimax")].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["sample_size_note"] = (
        "Copied from reports/model_comparison_sarimax.csv Group D; not "
        "intersected with the Elastic Net 8-horizon grid."
    )
    return rows


def _load_group_d_training_exog(path: Path = CURATED_DATA_PATH) -> pd.DataFrame:
    """Load the narrow Group D exog matrix for full-sample fitting/registration."""
    df = pd.read_csv(path)
    required_columns = {"quarter", *GROUP_D_FEATURE_COLUMNS}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required Group D columns: {sorted(missing)}")
    df.index = pd.PeriodIndex(df.pop("quarter").astype(str), freq="Q")
    return df.loc[:, list(GROUP_D_FEATURE_COLUMNS)].sort_index()


def run_sarimax_group_d_registration(
    curated_path: Path = CURATED_DATA_PATH,
    verbose: bool = False,
) -> str:
    """Log the full-sample fixed Group D SARIMAX as a distinct servable artifact."""
    from src.models import tracking

    series = load_target_series(curated_path)
    exog = _load_group_d_training_exog(curated_path)
    metrics = _load_existing_sarimax_d_metrics(SARIMAX_COMPARISON_REPORT_PATH)
    if metrics.empty:
        raise FileNotFoundError(
            "Canonical SARIMAX Group D metrics were not found in "
            f"{SARIMAX_COMPARISON_REPORT_PATH}."
        )

    if verbose:
        print("Fitting final full-sample SARIMAX Group D for MLflow logging...", flush=True)
    final_fit = fit_sarimax(
        series=series,
        exog=exog,
        order=GROUP_D_ORDER,
        seasonal_order=GROUP_D_SEASONAL_ORDER,
    )
    return tracking.log_model_run(
        run_name="sarimax_group_d_registration",
        model_name="sarimax",
        metrics=metrics,
        params={
            "order": GROUP_D_ORDER,
            "seasonal_order": GROUP_D_SEASONAL_ORDER,
            "features": GROUP_D_FEATURE_COLUMNS,
            "horizon_cap": 1,
        },
        tags={
            "model_family": "sarimax_group_d",
            "group_id": "D",
            "horizon_cap": "1",
        },
        artifact_paths=[SARIMAX_COMPARISON_REPORT_PATH],
        model_logger=lambda: tracking.log_statsmodels_model(final_fit),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Register the fixed SARIMAX Group D model.")
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    run_id = run_sarimax_group_d_registration(curated_path=args.data, verbose=args.verbose)
    print(run_id)


if __name__ == "__main__":
    main()
