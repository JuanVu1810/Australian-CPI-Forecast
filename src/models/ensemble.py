"""SARIMA + Elastic Net ensemble for CPI year-ended inflation forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.elastic_net import (
    TARGET_COLUMN,
    forecast_elastic_net_direct,
    load_elastic_net_feature_frame,
    simulate_elastic_net_paths,
)
from src.models.evaluation import (
    CURATED_DATA_PATH,
    DEFAULT_HORIZONS,
    PROJECT_ROOT,
    RBA_FORECAST_PATH,
    align_rba_forecasts_to_grid,
    compute_metric_table,
    load_target_series,
    restrict_to_common_grid,
    seasonal_naive_backtest,
    walk_forward_backtest,
    walk_forward_backtest_direct_multihorizon,
)
from src.models.sarima import (
    DEFAULT_ORDER as SARIMA_DEFAULT_ORDER,
    DEFAULT_SEASONAL_ORDER as SARIMA_DEFAULT_SEASONAL_ORDER,
    forecast_sarima,
    simulate_sarima_paths,
)


DEFAULT_WEIGHTS = (0.5, 0.5)
DEFAULT_INITIAL_TRAIN_SIZE = 40
DEFAULT_SEED = 42
ENSEMBLE_COMPARISON_OUTPUT_PATH = PROJECT_ROOT / "reports/model_comparison_ensemble.csv"


def _validate_weights(weights: tuple[float, float]) -> tuple[float, float]:
    if len(weights) != 2:
        raise ValueError("weights must contain exactly two values: SARIMA and Elastic Net.")
    sarima_weight, elastic_net_weight = (float(weights[0]), float(weights[1]))
    if not np.isclose(sarima_weight + elastic_net_weight, 1.0):
        raise ValueError("weights must sum to 1.")
    return sarima_weight, elastic_net_weight


def combine_point_forecasts(
    sarima_forecast,
    elastic_net_forecast,
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
) -> np.ndarray:
    """Return the weighted average of equal-length one-dimensional forecasts."""
    sarima_weight, elastic_net_weight = _validate_weights(weights)
    sarima_values = np.asarray(sarima_forecast, dtype=float)
    elastic_net_values = np.asarray(elastic_net_forecast, dtype=float)
    if sarima_values.ndim != 1 or elastic_net_values.ndim != 1:
        raise ValueError("both forecasts must be one-dimensional arrays.")
    if sarima_values.shape != elastic_net_values.shape:
        raise ValueError("both forecasts must have the same length.")
    return sarima_weight * sarima_values + elastic_net_weight * elastic_net_values


def combine_paths(
    sarima_paths,
    elastic_net_paths,
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
) -> np.ndarray:
    """Combine paired Monte Carlo draws from independent component models."""
    sarima_weight, elastic_net_weight = _validate_weights(weights)
    sarima_values = np.asarray(sarima_paths, dtype=float)
    elastic_net_values = np.asarray(elastic_net_paths, dtype=float)
    if sarima_values.ndim != 2 or elastic_net_values.ndim != 2:
        raise ValueError("both path arrays must be two-dimensional with shape (n_sims, steps).")
    if sarima_values.shape != elastic_net_values.shape:
        raise ValueError("both path arrays must have the same shape.")
    return sarima_weight * sarima_values + elastic_net_weight * elastic_net_values


def forecast_ensemble(
    train_frame: pd.DataFrame,
    steps: int = 8,
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Fit both components on raw training data and combine their point forecasts."""
    sarima_forecast = forecast_sarima(train_frame[TARGET_COLUMN], steps=steps)
    elastic_net_forecast = forecast_elastic_net_direct(train_frame, steps=steps, seed=seed)
    return combine_point_forecasts(sarima_forecast, elastic_net_forecast, weights=weights)


def simulate_ensemble_paths(
    train_frame: pd.DataFrame,
    steps: int = 8,
    n_sims: int = 1000,
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Fit both components on raw training data and combine their predictive draws."""
    sarima_paths = simulate_sarima_paths(
        train_frame[TARGET_COLUMN],
        steps=steps,
        n_sims=n_sims,
        seed=seed,
    )
    elastic_net_paths = simulate_elastic_net_paths(
        train_frame,
        steps=steps,
        n_sims=n_sims,
        seed=seed + 1,
    )
    return combine_paths(sarima_paths, elastic_net_paths, weights=weights)


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
    weights: tuple[float, float],
) -> pd.DataFrame:
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
        weights=weights,
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


def _log_ensemble_config(weights: tuple[float, float]) -> None:
    from src.models import tracking

    mlflow = tracking.configure_mlflow()
    mlflow.log_dict(
        {"weights": list(weights), "components": ["sarima", "elastic_net"]},
        "model_preprocessing/ensemble_weights.json",
    )


def run_ensemble_comparison(
    curated_path: Path = CURATED_DATA_PATH,
    rba_path: Path = RBA_FORECAST_PATH,
    comparison_output_path: Path = ENSEMBLE_COMPARISON_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the ensemble comparison, write its report, and log an MLflow run."""
    from src.models import tracking

    weights = _validate_weights(weights)
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    series = load_target_series(curated_path)
    exog = load_elastic_net_feature_frame(curated_path)

    if verbose:
        print("Running SARIMA walk-forward backtest...", flush=True)
    sarima_predictions = walk_forward_backtest(
        series=series,
        forecast_func=lambda train, steps: forecast_sarima(train, steps=steps),
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        model_name="sarima",
    )

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

    ensemble_predictions = _ensemble_prediction_frame(
        sarima_predictions=sarima_predictions,
        elastic_net_predictions=elastic_net_predictions,
        weights=weights,
    )
    naive_predictions = seasonal_naive_backtest(
        series=series,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
    )

    frames = [ensemble_predictions, sarima_predictions, elastic_net_predictions, naive_predictions]
    if rba_path.exists():
        rba_predictions = align_rba_forecasts_to_grid(
            pd.read_csv(rba_path),
            ensemble_predictions,
            horizons=requested_horizons,
        )
        if not rba_predictions.empty:
            frames.append(rba_predictions)

    common_predictions = restrict_to_common_grid(frames)
    comparison = compute_metric_table(common_predictions)

    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.round({"rmse": 6, "mae": 6}).to_csv(comparison_output_path, index=False)

    tracking.log_model_run(
        run_name="ensemble_comparison",
        model_name="ensemble",
        metrics=comparison,
        params={
            "weights": weights,
            "sarima_order": SARIMA_DEFAULT_ORDER,
            "sarima_seasonal_order": SARIMA_DEFAULT_SEASONAL_ORDER,
            "seed": seed,
            "initial_train_size": initial_train_size,
            "horizons": requested_horizons,
        },
        tags={
            "model_family": "ensemble",
            "component_families": "sarima;elastic_net",
            "run_role": "comparison_with_full_sample_model",
        },
        artifact_paths=[comparison_output_path],
        model_logger=lambda: _log_ensemble_config(weights),
    )
    return comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--rba-data", type=Path, default=RBA_FORECAST_PATH)
    parser.add_argument("--comparison-output", type=Path, default=ENSEMBLE_COMPARISON_OUTPUT_PATH)
    parser.add_argument("--initial-train-size", type=int, default=DEFAULT_INITIAL_TRAIN_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-origins", type=int, default=None)
    args = parser.parse_args(argv)

    comparison = run_ensemble_comparison(
        curated_path=args.data,
        rba_path=args.rba_data,
        comparison_output_path=args.comparison_output,
        initial_train_size=args.initial_train_size,
        seed=args.seed,
        max_origins=args.max_origins,
        verbose=True,
    )
    print("\nComparison:")
    print(comparison.round({"rmse": 3, "mae": 3}).to_string(index=False))


if __name__ == "__main__":
    main()
