"""Regularized direct multi-horizon Elastic Net challenger for CPI year-ended inflation.

SARIMAX loses to plain SARIMA on every screened feature group
(``reports/model_comparison_sarimax.csv``), pointing at small-sample
overfitting on the exogenous macro block rather than underfitting. This model
tests the direct fix -- L1/L2-penalized coefficients -- on SARIMAX Group D's
lag-safe feature set, plus explicit CPI autoregressive lags (a linear model
has no built-in AR structure the way SARIMA/SARIMAX do).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from src.models.evaluation import (
    CURATED_DATA_PATH,
    DEFAULT_HORIZONS,
    PROJECT_ROOT,
    RBA_FORECAST_PATH,
    TrainWindowScaler,
    align_rba_forecasts_to_grid,
    assert_consecutive_quarters,
    compute_metric_table,
    fit_train_window_scaler,
    load_target_series,
    restrict_to_common_grid,
    seasonal_naive_backtest,
    walk_forward_backtest,
    walk_forward_backtest_direct_multihorizon,
)
from src.models.sarima import forecast_sarima


TARGET_COLUMN = "cpi_yoy"
ELASTIC_NET_MACRO_FEATURE_COLUMNS = (
    "cash_rate_change_lag1",
    "unemployment_rate_change_lag1",
    "inflation_expectations_business_lag1",
    "ppi_growth_lag2",
    "commodity_growth_lag1",
    "wti_growth_lag1",
)
ELASTIC_NET_FEATURE_COLUMNS = (
    "cpi_yoy_lag1",
    "cpi_yoy_lag4",
    *ELASTIC_NET_MACRO_FEATURE_COLUMNS,
)

FORECAST_HORIZON = 8
DEFAULT_INITIAL_TRAIN_SIZE = 40
DEFAULT_SEED = 42
DEFAULT_L1_RATIOS = (0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0)
DEFAULT_ALPHAS = tuple(np.logspace(-3, 1, 10))
DEFAULT_CV_SPLITS = 4
# joblib's process-pool backend re-spawns workers on every GridSearchCV.fit()
# call; since this runs once per horizon per walk-forward origin (hundreds of
# short-lived calls), n_jobs>1 is dominated by spawn overhead rather than
# compute -- sequential is faster in practice for this workload.
DEFAULT_GRID_SEARCH_N_JOBS = 1
DEFAULT_MAX_ITER = 10_000

ELASTIC_NET_COMPARISON_OUTPUT_PATH = PROJECT_ROOT / "reports/model_comparison_elastic_net.csv"
ELASTIC_NET_COEFFICIENT_OUTPUT_PATH = PROJECT_ROOT / "reports/elastic_net_coefficients.csv"
SARIMAX_COMPARISON_REPORT_PATH = PROJECT_ROOT / "reports/model_comparison_sarimax.csv"

ELASTIC_NET_MODEL_NOTE = (
    "Direct multi-horizon Elastic Net: one scaler+ElasticNet Pipeline per "
    "horizon (1-8), features = SARIMAX Group D macro block + cpi_yoy_lag1/"
    "lag4, GridSearchCV over alpha/l1_ratio with TimeSeriesSplit inner CV -- "
    "chronological (never shuffled) folds -- refitting the scaler inside "
    "each fold so no validation row leaks into that fold's own scaling."
)
SARIMAX_D_NOTE = (
    "SARIMAX(D) remains lag-safety capped at horizon 1 because Group D includes "
    "lag-1 regressors; Elastic Net horizons 1-8 use only the past origin row."
)


@dataclass
class ElasticNetDirectFit:
    models: dict[int, Any]
    scaler: TrainWindowScaler
    feature_columns: tuple[str, ...]
    target_column: str
    horizons: tuple[int, ...]

    def predict_next(self, train_frame: pd.DataFrame) -> np.ndarray:
        """Predict all fitted horizons from the most recent origin-time row.

        ``models[horizon]`` is a fitted ``scaler -> ElasticNet`` sklearn
        ``Pipeline`` (see ``fit_elastic_net_direct``), so this passes raw,
        unscaled feature values -- the pipeline's own scaler step (fit on
        that horizon's full training window during the final refit) applies
        the transform internally.
        """
        clean = pd.DataFrame(train_frame).loc[:, list(self.feature_columns)].dropna().astype(float)
        if clean.empty:
            raise ValueError("forecast frame has no complete feature rows.")
        latest = clean.iloc[[-1]]
        return np.asarray(
            [float(self.models[horizon].predict(latest)[0]) for horizon in self.horizons],
            dtype=float,
        )


def clean_elastic_net_frame(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: tuple[str, ...] = ELASTIC_NET_FEATURE_COLUMNS,
) -> pd.DataFrame:
    columns = (target_column, *feature_columns)
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Elastic Net frame missing required columns: {sorted(missing)}")
    clean = pd.DataFrame(frame).loc[:, list(columns)].dropna().astype(float).sort_index()
    if clean.index.has_duplicates:
        raise ValueError("Elastic Net frame index must not contain duplicate quarters.")
    assert_consecutive_quarters(clean.index, "Elastic Net frame")
    return clean


def load_elastic_net_feature_frame(path: Path = CURATED_DATA_PATH) -> pd.DataFrame:
    """Load the Elastic Net feature block with a quarterly index."""
    required_columns = ("quarter", *ELASTIC_NET_FEATURE_COLUMNS)
    df = pd.read_csv(path, usecols=list(required_columns))
    df.index = pd.PeriodIndex(df.pop("quarter").astype(str), freq="Q")
    return df.sort_index().astype(float)


def _safe_cv_splits(n_examples: int, requested: int) -> int:
    """Cap TimeSeriesSplit folds so every fold has at least one sample."""
    return max(2, min(requested, n_examples - 1))


def _elastic_net_horizon_training_xy(
    clean: pd.DataFrame,
    horizon: int,
    feature_columns: tuple[str, ...],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    x = clean.loc[:, list(feature_columns)].iloc[: len(clean) - horizon]
    y = clean[target_column].iloc[horizon:]
    return x, y


def fit_elastic_net_direct(
    train_frame: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    feature_columns: tuple[str, ...] = ELASTIC_NET_FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
    l1_ratios: tuple[float, ...] = DEFAULT_L1_RATIOS,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    cv_splits: int = DEFAULT_CV_SPLITS,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = DEFAULT_SEED,
    n_jobs: int = DEFAULT_GRID_SEARCH_N_JOBS,
) -> ElasticNetDirectFit:
    """Fit one scaler+ElasticNet pipeline per horizon on a chronological train window.

    Each horizon's alpha/l1_ratio is selected by ``GridSearchCV`` over a
    ``StandardScaler -> ElasticNet`` ``Pipeline`` with ``TimeSeriesSplit``
    folds -- chronological, not sklearn's default shuffled K-fold. Using
    ``ElasticNetCV`` directly (the previous approach) fit one scaler on the
    *entire* training window before any inner-CV split, so every fold's
    validation portion still leaked into the mean/std used to scale its own
    training portion. Wrapping the scaler inside the ``Pipeline`` that
    ``GridSearchCV`` cross-validates fixes this: for every candidate
    alpha/l1_ratio and every fold, the pipeline (scaler included) is cloned
    and fit on that fold's training rows only. ``refit=True`` (the default)
    then refits the winning pipeline -- scaler included -- on the *entire*
    training window for the final model, which is legitimate: that window is
    exactly what's available at this forecast origin.
    """
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clean = clean_elastic_net_frame(train_frame, target_column=target_column, feature_columns=feature_columns)

    models: dict[int, Any] = {}
    for horizon in horizons:
        if len(clean) <= horizon:
            raise ValueError(f"training frame is too short for horizon {horizon}.")
        x, y = _elastic_net_horizon_training_xy(
            clean=clean,
            horizon=horizon,
            feature_columns=feature_columns,
            target_column=target_column,
        )
        n_splits = _safe_cv_splits(len(x), cv_splits)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(max_iter=max_iter, random_state=seed)),
            ]
        )
        search = GridSearchCV(
            pipeline,
            param_grid={"model__alpha": list(alphas), "model__l1_ratio": list(l1_ratios)},
            cv=TimeSeriesSplit(n_splits=n_splits),
            scoring="neg_mean_squared_error",
            n_jobs=n_jobs,
        )
        search.fit(x, y.to_numpy())
        models[horizon] = search.best_estimator_

    # Full-training-window scaler kept only for MLflow logging metadata
    # (documenting the feature scale at this origin); prediction/coefficients
    # use each horizon's own pipeline-internal scaler above, not this one.
    scaler = fit_train_window_scaler(clean.loc[:, list(feature_columns)])

    return ElasticNetDirectFit(
        models=models,
        scaler=scaler,
        feature_columns=feature_columns,
        target_column=target_column,
        horizons=tuple(horizons),
    )


def forecast_from_fit(fitted: ElasticNetDirectFit, train_frame: pd.DataFrame) -> np.ndarray:
    return fitted.predict_next(train_frame)


def forecast_elastic_net_direct(
    train_frame: pd.DataFrame,
    steps: int = FORECAST_HORIZON,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Fit the Elastic Net and return a direct 8-quarter forecast vector."""
    if steps > FORECAST_HORIZON:
        raise ValueError("the Elastic Net baseline emits at most 8 horizons.")
    fitted = fit_elastic_net_direct(train_frame, seed=seed)
    return forecast_from_fit(fitted, train_frame)[:steps]


def _elastic_net_residual_matrix(fitted: ElasticNetDirectFit, clean: pd.DataFrame) -> pd.DataFrame:
    residuals = []
    for horizon in fitted.horizons:
        x, y = _elastic_net_horizon_training_xy(
            clean=clean,
            horizon=horizon,
            feature_columns=fitted.feature_columns,
            target_column=fitted.target_column,
        )
        predicted = fitted.models[horizon].predict(x)
        residuals.append(pd.Series(y.to_numpy() - predicted, index=x.index, name=horizon))

    matrix = pd.concat(residuals, axis=1, join="inner").dropna()
    if matrix.empty:
        raise ValueError("Elastic Net residual bootstrap has no common training origins.")
    return matrix


def simulate_paths_from_fit(
    fitted: ElasticNetDirectFit,
    train_frame: pd.DataFrame,
    steps: int = 8,
    n_sims: int = 1000,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Draw residual-bootstrap future ``cpi_yoy`` paths from an already-fitted model."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if steps > FORECAST_HORIZON:
        raise ValueError("the Elastic Net baseline emits at most 8 horizons.")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1.")

    horizons = tuple(range(1, steps + 1))
    missing_horizons = sorted(set(horizons).difference(fitted.horizons))
    if missing_horizons:
        available_horizons = sorted(fitted.horizons)
        raise ValueError(
            "already-fitted Elastic Net model does not include requested horizons: "
            f"{missing_horizons}; available horizons: {available_horizons}."
        )

    clean = clean_elastic_net_frame(
        train_frame,
        target_column=fitted.target_column,
        feature_columns=fitted.feature_columns,
    )
    point_forecast = fitted.predict_next(train_frame)[:steps]
    # In-sample residuals likely understate true forecast-error variance,
    # especially at longer horizons; not a calibrated out-of-sample interval.
    residual_matrix = _elastic_net_residual_matrix(fitted=fitted, clean=clean).loc[:, list(horizons)]

    rng = np.random.default_rng(seed)
    origin_indices = rng.integers(0, len(residual_matrix), size=n_sims)
    paths = point_forecast + residual_matrix.to_numpy()[origin_indices, :]
    return np.asarray(paths, dtype=float)


def simulate_elastic_net_paths(
    train_frame: pd.DataFrame,
    steps: int = 8,
    n_sims: int = 1000,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Fit Elastic Net once and draw residual-bootstrap future ``cpi_yoy`` paths."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if steps > FORECAST_HORIZON:
        raise ValueError("the Elastic Net baseline emits at most 8 horizons.")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1.")

    horizons = tuple(range(1, steps + 1))
    fitted = fit_elastic_net_direct(train_frame, horizons=horizons, seed=seed)
    return simulate_paths_from_fit(
        fitted=fitted,
        train_frame=train_frame,
        steps=steps,
        n_sims=n_sims,
        seed=seed,
    )


def coefficient_table(fitted: ElasticNetDirectFit) -> pd.DataFrame:
    """Extract per-horizon coefficients, intercept, and selected alpha/l1_ratio.

    ``fitted.models[horizon]`` is a ``scaler -> ElasticNet`` Pipeline
    (``fit_elastic_net_direct``), so coefficients/intercept/alpha/l1_ratio
    are read off its ``"model"`` step.
    """
    rows: list[dict[str, object]] = []
    for horizon, pipeline in fitted.models.items():
        model = pipeline.named_steps["model"]
        for feature, coef in zip(fitted.feature_columns, model.coef_):
            rows.append(
                {
                    "horizon": horizon,
                    "feature": feature,
                    "coef": float(coef),
                    "intercept": float(model.intercept_),
                    "selected_alpha": float(model.alpha),
                    "selected_l1_ratio": float(model.l1_ratio),
                }
            )
    return pd.DataFrame(rows)


def _comparison_metadata(
    metrics: pd.DataFrame,
    origin_n: int,
    note: str,
    group_id: str = "ELASTIC_NET",
    group_name: str = "regularized direct multihorizon",
    horizon_range: str = "1-8",
    horizon_cap: int = FORECAST_HORIZON,
    features: str = ";".join(ELASTIC_NET_FEATURE_COLUMNS),
    selected_order: str = "fixed_architecture",
    selected_seasonal_order: str = "",
    selected_by: str = "elastic_net_cv_per_horizon",
) -> pd.DataFrame:
    result = metrics.copy()
    result.insert(0, "group_id", group_id)
    result.insert(1, "group_name", group_name)
    result.insert(2, "horizon_range", horizon_range)
    result.insert(3, "horizon_cap", horizon_cap)
    result.insert(4, "features", features)
    result.insert(5, "selected_order", selected_order)
    result.insert(6, "selected_seasonal_order", selected_seasonal_order)
    result.insert(7, "selected_by", selected_by)
    result.insert(8, "group_origin_n", origin_n)
    result.insert(9, "group_overall_metric_n", int(metrics.loc[metrics["horizon"].eq("overall"), "n"].max()))
    result.insert(10, "sample_size_note", note)
    return result


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


def run_elastic_net_comparison(
    curated_path: Path = CURATED_DATA_PATH,
    rba_path: Path = RBA_FORECAST_PATH,
    comparison_output_path: Path = ELASTIC_NET_COMPARISON_OUTPUT_PATH,
    coefficient_output_path: Path = ELASTIC_NET_COEFFICIENT_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run the Elastic Net walk-forward backtest, comparison, and coefficient report."""
    from src.models import tracking

    started = time.perf_counter()
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    series = load_target_series(curated_path)
    exog = load_elastic_net_feature_frame(curated_path)

    if verbose:
        print("Running Elastic Net direct-multihorizon walk-forward backtest...", flush=True)
    elastic_net_predictions = walk_forward_backtest_direct_multihorizon(
        series=series,
        exog=exog,
        forecast_func=lambda train_frame, steps: forecast_elastic_net_direct(
            train_frame,
            steps=steps,
            seed=seed,
        ),
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        model_name="elastic_net",
        target_column=TARGET_COLUMN,
        max_origins=max_origins,
    )
    if elastic_net_predictions.empty:
        raise ValueError("Elastic Net backtest produced no forecast origins.")

    sarima = walk_forward_backtest(
        series=series,
        forecast_func=lambda train, steps: forecast_sarima(train, steps=steps),
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        model_name="sarima",
    )
    naive = seasonal_naive_backtest(
        series=series,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
    )
    frames = [elastic_net_predictions, sarima, naive]
    if rba_path.exists():
        rba = align_rba_forecasts_to_grid(
            pd.read_csv(rba_path),
            elastic_net_predictions,
            horizons=requested_horizons,
        )
        if not rba.empty:
            frames.append(rba)

    full_horizon_common = restrict_to_common_grid(frames)
    full_metrics = compute_metric_table(full_horizon_common)
    origin_n = int(
        full_horizon_common.loc[full_horizon_common["model"].eq("elastic_net"), "forecast_origin"].nunique()
    )
    comparison = _comparison_metadata(full_metrics, origin_n=origin_n, note=SARIMAX_D_NOTE)

    if max_origins is None:
        sarimax_comparison = _load_existing_sarimax_d_metrics()
        if not sarimax_comparison.empty:
            comparison = pd.concat([comparison, sarimax_comparison], ignore_index=True, sort=False)

    if verbose:
        print("Fitting final full-sample Elastic Net for coefficients/MLflow logging...", flush=True)
    full_frame = pd.concat([series.rename(TARGET_COLUMN), exog], axis=1)
    final_fit = fit_elastic_net_direct(full_frame, horizons=requested_horizons, seed=seed)
    coefficients = coefficient_table(final_fit)

    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    coefficient_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.round({"rmse": 6, "mae": 6}).to_csv(comparison_output_path, index=False)
    coefficients.round(
        {"coef": 6, "intercept": 6, "selected_alpha": 6, "selected_l1_ratio": 6}
    ).to_csv(coefficient_output_path, index=False)

    tracking.log_model_run(
        run_name="elastic_net_comparison",
        model_name="elastic_net",
        metrics=comparison,
        params={
            "order": "fixed_architecture",
            "seasonal_order": "",
            "features": ELASTIC_NET_FEATURE_COLUMNS,
            "selection_criterion": "elastic_net_cv_per_horizon",
            "initial_train_size": initial_train_size,
            "horizons": requested_horizons,
            "l1_ratios": DEFAULT_L1_RATIOS,
            "cv_splits": DEFAULT_CV_SPLITS,
            "seed": seed,
        },
        tags={
            "model_family": "elastic_net",
            "reused_feature_group_id": "D",
            "run_role": "comparison_with_full_sample_model",
        },
        artifact_paths=[comparison_output_path, coefficient_output_path],
        model_logger=lambda: tracking.log_elastic_net_model(final_fit, full_frame),
    )
    runtime_seconds = time.perf_counter() - started
    return comparison, coefficients, runtime_seconds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--rba-data", type=Path, default=RBA_FORECAST_PATH)
    parser.add_argument("--comparison-output", type=Path, default=ELASTIC_NET_COMPARISON_OUTPUT_PATH)
    parser.add_argument("--coefficients-output", type=Path, default=ELASTIC_NET_COEFFICIENT_OUTPUT_PATH)
    parser.add_argument("--initial-train-size", type=int, default=DEFAULT_INITIAL_TRAIN_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-origins", type=int, default=None)
    args = parser.parse_args(argv)

    comparison, coefficients, runtime_seconds = run_elastic_net_comparison(
        curated_path=args.data,
        rba_path=args.rba_data,
        comparison_output_path=args.comparison_output,
        coefficient_output_path=args.coefficients_output,
        initial_train_size=args.initial_train_size,
        seed=args.seed,
        max_origins=args.max_origins,
        verbose=True,
    )
    print(ELASTIC_NET_MODEL_NOTE)
    print(f"Runtime seconds: {runtime_seconds:.1f}")
    print("\nComparison:")
    print(comparison.to_string(index=False))
    print("\nCoefficients (overall horizon 1):")
    print(coefficients.loc[coefficients["horizon"] == 1].to_string(index=False))


if __name__ == "__main__":
    main()
