"""Regularized direct multi-horizon Elastic Net challenger for CPI year-ended inflation.

Uses L1/L2-penalized coefficients on a lag-safe macro feature block, plus
explicit CPI autoregressive lags (a linear model has no built-in AR structure
the way SARIMA does). Regularized coefficients also make this the model of
choice for exogenous-variable scenario/sensitivity analysis: every shipped
feature's coefficient is stable and well-behaved under cross-validated
selection, unlike an unregularized single-equation MLE fit on the same
small sample.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.evaluation import (
    CURATED_DATA_PATH,
    DEFAULT_HORIZONS,
    ELASTIC_NET_MODEL_PATH,
    PROJECT_ROOT,
    RBA_FORECAST_PATH,
    TrainWindowScaler,
    compute_baseline_predictions,
    assert_consecutive_quarters,
    compute_metric_table,
    fit_train_window_scaler,
    load_target_series,
    restrict_to_common_grid,
    walk_forward_backtest,
    walk_forward_backtest_direct_multihorizon,
)
from src.models.sarima import forecast_sarima


TARGET_COLUMN = "cpi_yoy"
# WHY these three: the quarterly-CPI screen in book appendix 5 finds a Granger signal only for business
# inflation expectations, the cash rate and (weakly) PPI, all at lag 1. The earlier eleven-term sets (oil,
# commodity, unemployment, squared and interaction terms) came from a YoY screen whose p-values were too small.
# Walk-forward RMSE on the same 53 origins was a tie with the old set for headline, so we use the smaller set.
# See reports/model_refit_phase1_decisions.md (addendum).
ELASTIC_NET_MACRO_FEATURE_COLUMNS = (
    "inflation_expectations_business_lag1",
    "cash_rate_change_lag1",
    "ppi_growth_lag1",
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
# Sequential is faster here: GridSearchCV runs hundreds of tiny fits, and starting worker
# processes for each one costs more than it saves.
DEFAULT_GRID_SEARCH_N_JOBS = 1
DEFAULT_MAX_ITER = 10_000

ELASTIC_NET_COMPARISON_OUTPUT_PATH = PROJECT_ROOT / "reports/model_comparison_elastic_net.csv"
ELASTIC_NET_COEFFICIENT_OUTPUT_PATH = PROJECT_ROOT / "reports/elastic_net_coefficients.csv"

ELASTIC_NET_MODEL_NOTE = (
    "Direct multi-horizon Elastic Net: one scaler+ElasticNet Pipeline per "
    "horizon (1-8), features = lag-safe macro block + cpi_yoy_lag1/"
    "lag4, GridSearchCV over alpha/l1_ratio with TimeSeriesSplit inner CV -- "
    "chronological (never shuffled) folds -- refitting the scaler inside "
    "each fold so no validation row leaks into that fold's own scaling."
)


@dataclass
class ElasticNetDirectFit:
    models: dict[int, Any]
    scaler: TrainWindowScaler
    feature_columns: tuple[str, ...]
    target_column: str
    horizons: tuple[int, ...]
    oos_residuals: dict[int, pd.Series]

    def predict_next(self, train_frame: pd.DataFrame) -> np.ndarray:
        """Predict every fitted horizon from the latest complete feature row (raw values; each pipeline scales them)."""
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
    """Keep the target and feature columns, drop incomplete rows, and check the quarters are consecutive."""
    columns = (target_column, *feature_columns)
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Elastic Net frame missing required columns: {sorted(missing)}")
    clean = pd.DataFrame(frame).loc[:, list(columns)].dropna().astype(float).sort_index()
    if clean.index.has_duplicates:
        raise ValueError("Elastic Net frame index must not contain duplicate quarters.")
    assert_consecutive_quarters(clean.index, "Elastic Net frame")
    return clean


def load_elastic_net_feature_frame(
    path: Path = CURATED_DATA_PATH,
    feature_columns: tuple[str, ...] = ELASTIC_NET_FEATURE_COLUMNS,
    max_quarter: pd.Period | None = None,
) -> pd.DataFrame:
    """Load the Elastic Net feature columns with a quarterly index, optionally cut at ``max_quarter``."""
    required_columns = ("quarter", *feature_columns)
    df = pd.read_csv(path, usecols=list(required_columns))
    df.index = pd.PeriodIndex(df.pop("quarter").astype(str), freq="Q")
    frame = df.sort_index().astype(float)
    if max_quarter is not None:
        frame = frame.loc[frame.index <= max_quarter]
    return frame


def _safe_cv_splits(n_examples: int, requested: int) -> int:
    """Cap TimeSeriesSplit folds so every fold has at least one sample."""
    return max(2, min(requested, n_examples - 1))


def _time_series_oos_residuals(
    x: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    l1_ratio: float,
    n_splits: int,
    max_iter: int,
    seed: int,
) -> pd.Series:
    """Out-of-sample residuals from TimeSeriesSplit folds, using the already-chosen alpha and l1_ratio.

    WHY a manual loop: cross_val_predict refuses TimeSeriesSplit because the first block is never a
    test fold. In-sample residuals from the final fit would understate the forecast error.
    """
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cv = TimeSeriesSplit(n_splits=n_splits)
    residual_frames = []
    for train_idx, test_idx in cv.split(x):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=seed)),
            ]
        )
        pipeline.fit(x.iloc[train_idx], y.iloc[train_idx].to_numpy())
        predicted = pipeline.predict(x.iloc[test_idx])
        residual_frames.append(
            pd.Series(y.iloc[test_idx].to_numpy() - predicted, index=x.index[test_idx])
        )
    residuals = pd.concat(residual_frames).sort_index()
    if residuals.empty:
        raise ValueError("no out-of-sample residuals were produced by the CV folds.")
    return residuals


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
    """Fit one scaler + ElasticNet pipeline per horizon, tuning alpha and l1_ratio with chronological CV.

    WHY the scaler sits inside the pipeline: GridSearchCV then refits it on each fold's training
    rows only, so no validation row leaks into the scaling. The winner is refit on the whole window.
    """
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clean = clean_elastic_net_frame(train_frame, target_column=target_column, feature_columns=feature_columns)

    models = {}
    oos_residuals = {}
    for horizon in horizons:
        if len(clean) <= horizon:
            raise ValueError(f"training frame is too short for horizon {horizon}.")
        # Direct forecasting: features at quarter t predict the target at t + horizon.
        x = clean.loc[:, list(feature_columns)].iloc[: len(clean) - horizon]
        y = clean[target_column].iloc[horizon:]
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
        oos_residuals[horizon] = _time_series_oos_residuals(
            x=x,
            y=y,
            alpha=search.best_params_["model__alpha"],
            l1_ratio=search.best_params_["model__l1_ratio"],
            n_splits=n_splits,
            max_iter=max_iter,
            seed=seed,
        )

    # Full-training-window scaler kept only to document the feature scale at this
    # origin; prediction/coefficients use each horizon's own pipeline-internal
    # scaler above, not this one.
    scaler = fit_train_window_scaler(clean.loc[:, list(feature_columns)])

    return ElasticNetDirectFit(
        models=models,
        scaler=scaler,
        feature_columns=feature_columns,
        target_column=target_column,
        horizons=tuple(horizons),
        oos_residuals=oos_residuals,
    )


def forecast_elastic_net_direct(
    train_frame: pd.DataFrame,
    steps: int = FORECAST_HORIZON,
    seed: int = DEFAULT_SEED,
    feature_columns: tuple[str, ...] = ELASTIC_NET_FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> np.ndarray:
    """Fit the Elastic Net and return a direct 8-quarter forecast vector."""
    if steps > FORECAST_HORIZON:
        raise ValueError("the Elastic Net baseline emits at most 8 horizons.")
    fitted = fit_elastic_net_direct(
        train_frame,
        seed=seed,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    return fitted.predict_next(train_frame)[:steps]


def _elastic_net_residual_matrix(fitted: ElasticNetDirectFit) -> pd.DataFrame:
    """Join each horizon's out-of-sample residuals on shared dates, so one bootstrap draw keeps the cross-horizon correlation."""
    residuals = [fitted.oos_residuals[horizon].rename(horizon) for horizon in fitted.horizons]
    matrix = pd.concat(residuals, axis=1, join="inner").dropna()
    if matrix.empty:
        raise ValueError("Elastic Net residual bootstrap has no common out-of-sample origins.")
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

    point_forecast = fitted.predict_next(train_frame)[:steps]
    residual_matrix = _elastic_net_residual_matrix(fitted=fitted).loc[:, list(horizons)]
    residual_matrix = residual_matrix - residual_matrix.median(axis=0)

    rng = np.random.default_rng(seed)
    origin_indices = rng.integers(0, len(residual_matrix), size=n_sims)
    paths = point_forecast + residual_matrix.to_numpy()[origin_indices, :]
    return np.asarray(paths, dtype=float)


def simulate_elastic_net_paths(
    train_frame: pd.DataFrame,
    steps: int = 8,
    n_sims: int = 1000,
    seed: int = DEFAULT_SEED,
    feature_columns: tuple[str, ...] = ELASTIC_NET_FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> np.ndarray:
    """Fit Elastic Net once and draw residual-bootstrap future ``cpi_yoy`` paths."""
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if steps > FORECAST_HORIZON:
        raise ValueError("the Elastic Net baseline emits at most 8 horizons.")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1.")

    horizons = tuple(range(1, steps + 1))
    fitted = fit_elastic_net_direct(
        train_frame,
        horizons=horizons,
        seed=seed,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    return simulate_paths_from_fit(
        fitted=fitted,
        train_frame=train_frame,
        steps=steps,
        n_sims=n_sims,
        seed=seed,
    )


def coefficient_table(fitted: ElasticNetDirectFit) -> pd.DataFrame:
    """One row per horizon and feature: coefficient (on scaled features), intercept and chosen alpha/l1_ratio."""
    rows = []
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
    """Add the description columns the comparison report expects in front of the metric table."""
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


def run_elastic_net_comparison(
    curated_path: Path = CURATED_DATA_PATH,
    rba_path: Path = RBA_FORECAST_PATH,
    comparison_output_path: Path = ELASTIC_NET_COMPARISON_OUTPUT_PATH,
    coefficient_output_path: Path = ELASTIC_NET_COEFFICIENT_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    include_rba: bool = True,
    model_output_path: Path = ELASTIC_NET_MODEL_PATH,
    verbose: bool = False,
    max_quarter: pd.Period | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run the Elastic Net walk-forward backtest, comparison, and coefficient report."""
    from src.models.sarima import DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER

    started = time.perf_counter()
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    series = load_target_series(curated_path, target_column=TARGET_COLUMN, max_quarter=max_quarter)
    exog = load_elastic_net_feature_frame(
        curated_path, feature_columns=ELASTIC_NET_FEATURE_COLUMNS, max_quarter=max_quarter
    )

    def forecast_elastic_net(train_frame, steps):
        return forecast_elastic_net_direct(
            train_frame,
            steps=steps,
            seed=seed,
            feature_columns=ELASTIC_NET_FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
        )

    def forecast_sarima_baseline(train, steps):
        return forecast_sarima(train, steps=steps, order=DEFAULT_ORDER, seasonal_order=DEFAULT_SEASONAL_ORDER)

    if verbose:
        print("Running Elastic Net direct-multihorizon walk-forward backtest...", flush=True)
    elastic_net_predictions = walk_forward_backtest_direct_multihorizon(
        series=series,
        exog=exog,
        forecast_func=forecast_elastic_net,
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
        forecast_func=forecast_sarima_baseline,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        model_name="sarima",
    )
    baselines = compute_baseline_predictions(
        series=series,
        rba_path=rba_path,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        include_rba=include_rba,
    )
    frames = [elastic_net_predictions, sarima, baselines]

    full_horizon_common = restrict_to_common_grid(frames)
    full_metrics = compute_metric_table(full_horizon_common)
    origin_n = int(
        full_horizon_common.loc[full_horizon_common["model"].eq("elastic_net"), "forecast_origin"].nunique()
    )
    comparison = _comparison_metadata(
        full_metrics,
        origin_n=origin_n,
        note=(
            "Elastic Net horizons 1-8 use only the past origin row (direct "
            "multi-horizon, lag-safe feature set)."
        ),
        features=";".join(ELASTIC_NET_FEATURE_COLUMNS),
    )

    if verbose:
        print("Fitting final full-sample Elastic Net for coefficients and the saved model...", flush=True)
    full_frame = pd.concat([series.rename(TARGET_COLUMN), exog], axis=1)
    final_fit = fit_elastic_net_direct(
        full_frame,
        horizons=requested_horizons,
        seed=seed,
        feature_columns=ELASTIC_NET_FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )
    coefficients = coefficient_table(final_fit)

    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    coefficient_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.round({"rmse": 6, "mae": 6}).to_csv(comparison_output_path, index=False)
    coefficients.round(
        {"coef": 6, "intercept": 6, "selected_alpha": 6, "selected_l1_ratio": 6}
    ).to_csv(coefficient_output_path, index=False)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_fit, model_output_path)
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

    from src.models import svar

    comparison, coefficients, runtime_seconds = run_elastic_net_comparison(
        curated_path=args.data,
        rba_path=args.rba_data,
        comparison_output_path=args.comparison_output,
        coefficient_output_path=args.coefficients_output,
        initial_train_size=args.initial_train_size,
        seed=args.seed,
        max_origins=args.max_origins,
        verbose=True,
        max_quarter=svar.FORECAST_ORIGIN_PIN,
    )
    print(ELASTIC_NET_MODEL_NOTE)
    print(f"Runtime seconds: {runtime_seconds:.1f}")
    print("\nComparison:")
    print(comparison.to_string(index=False))
    print("\nCoefficients (overall horizon 1):")
    print(coefficients.loc[coefficients["horizon"] == 1].to_string(index=False))


if __name__ == "__main__":
    # Re-import so the saved model's class is src.models.elastic_net.ElasticNetDirectFit.
    # Defined in __main__ it could not be loaded by the API.
    from src.models.elastic_net import main as main_from_module

    main_from_module()
