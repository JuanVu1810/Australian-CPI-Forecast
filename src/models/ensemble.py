"""SARIMA + Elastic Net ensemble for CPI year-ended inflation forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.elastic_net import (
    ELASTIC_NET_FEATURE_COLUMNS,
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
    compute_baseline_predictions,
    compute_metric_table,
    load_target_series,
    restrict_to_common_grid,
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
DYNAMIC_WEIGHTS_SOURCE_PATH = PROJECT_ROOT / "reports/model_comparison_elastic_net.csv"
ENSEMBLE_COMPARISON_OUTPUT_PATH = PROJECT_ROOT / "reports/model_comparison_ensemble.csv"


def _validate_weights(weights: tuple[float, float]) -> tuple[float, float]:
    if len(weights) != 2:
        raise ValueError("weights must contain exactly two values: SARIMA and Elastic Net.")
    sarima_weight, elastic_net_weight = (float(weights[0]), float(weights[1]))
    if not np.isclose(sarima_weight + elastic_net_weight, 1.0):
        raise ValueError("weights must sum to 1.")
    return sarima_weight, elastic_net_weight


def horizon_rmse_weights(
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    path: Path = DYNAMIC_WEIGHTS_SOURCE_PATH,
) -> dict[int, tuple[float, float]]:
    """Inverse-RMSE weights per horizon from the SARIMA vs Elastic Net report.

    weight_sarima(h) = rmse_elastic_net(h) / (rmse_sarima(h) + rmse_elastic_net(h)).
    This is a deterministic formula on already-computed numbers, not a separately
    fit or optimized parameter per horizon. It is still a mild in-sample choice:
    the weights come from the same full walk-forward sample whose accuracy
    ``run_ensemble_comparison`` then reports for the ensemble.
    """
    if not path.exists():
        raise RuntimeError(
            f"Dynamic ensemble weight source report is missing at {path}; cannot derive "
            "per-horizon inverse-RMSE weights without the SARIMA and Elastic Net comparison."
        )

    table = pd.read_csv(path)
    required_columns = {"group_id", "model", "horizon", "rmse"}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(
            f"Dynamic ensemble weight source report {path} is missing required columns: {missing}."
        )

    filtered = table.loc[
        table["group_id"].eq("ELASTIC_NET")
        & table["model"].isin(["sarima", "elastic_net"])
    ].copy()

    weights: dict[int, tuple[float, float]] = {}
    for horizon in horizons:
        horizon_key = int(horizon)
        per_horizon: dict[str, float] = {}
        for model in ("sarima", "elastic_net"):
            rows = filtered.loc[
                filtered["model"].eq(model)
                & filtered["horizon"].astype(str).eq(str(horizon_key))
            ]
            if rows.empty:
                raise RuntimeError(
                    f"Dynamic ensemble RMSE for model {model!r} at horizon {horizon_key} "
                    f"was not found in {path}; cannot derive per-horizon weights."
                )
            rmse = float(rows.iloc[0]["rmse"])
            if np.isnan(rmse):
                raise RuntimeError(
                    f"Dynamic ensemble RMSE for model {model!r} at horizon {horizon_key} "
                    f"in {path} is NaN."
                )
            per_horizon[model] = rmse

        denominator = per_horizon["sarima"] + per_horizon["elastic_net"]
        if np.isclose(denominator, 0.0):
            raise RuntimeError(
                f"Dynamic ensemble RMSE denominator at horizon {horizon_key} in {path} is zero."
            )
        pair = (
            per_horizon["elastic_net"] / denominator,
            per_horizon["sarima"] / denominator,
        )
        assert np.isclose(sum(pair), 1.0)
        weights[horizon_key] = pair
    return weights


def _position_horizons(length: int, horizons) -> tuple[int, ...]:
    if horizons is None:
        return tuple(range(1, length + 1))
    horizon_values = np.asarray(horizons)
    if horizon_values.ndim != 1:
        raise ValueError("horizons must be a one-dimensional array.")
    if len(horizon_values) != length:
        raise ValueError("horizons must have the same length as forecast positions.")
    return tuple(int(horizon) for horizon in horizon_values)


def _weights_by_position(
    weights: dict[int, tuple[float, float]],
    length: int,
    horizons,
) -> tuple[np.ndarray, np.ndarray]:
    sarima_weights = []
    elastic_net_weights = []
    for horizon in _position_horizons(length, horizons):
        if horizon not in weights:
            raise ValueError(f"weights are missing horizon {horizon}.")
        sarima_weight, elastic_net_weight = _validate_weights(weights[horizon])
        sarima_weights.append(sarima_weight)
        elastic_net_weights.append(elastic_net_weight)
    return np.asarray(sarima_weights), np.asarray(elastic_net_weights)


def combine_point_forecasts(
    sarima_forecast,
    elastic_net_forecast,
    weights: tuple[float, float] | dict[int, tuple[float, float]] = DEFAULT_WEIGHTS,
    horizons=None,
) -> np.ndarray:
    """Return the weighted average of equal-length one-dimensional forecasts."""
    if not isinstance(weights, dict):
        sarima_weight, elastic_net_weight = _validate_weights(weights)
    sarima_values = np.asarray(sarima_forecast, dtype=float)
    elastic_net_values = np.asarray(elastic_net_forecast, dtype=float)
    if sarima_values.ndim != 1 or elastic_net_values.ndim != 1:
        raise ValueError("both forecasts must be one-dimensional arrays.")
    if sarima_values.shape != elastic_net_values.shape:
        raise ValueError("both forecasts must have the same length.")
    if isinstance(weights, dict):
        sarima_weight, elastic_net_weight = _weights_by_position(
            weights,
            len(sarima_values),
            horizons,
        )
    return sarima_weight * sarima_values + elastic_net_weight * elastic_net_values


def combine_paths(
    sarima_paths,
    elastic_net_paths,
    weights: tuple[float, float] | dict[int, tuple[float, float]] = DEFAULT_WEIGHTS,
    horizons=None,
) -> np.ndarray:
    """Combine paired Monte Carlo draws from independent component models."""
    if not isinstance(weights, dict):
        sarima_weight, elastic_net_weight = _validate_weights(weights)
    sarima_values = np.asarray(sarima_paths, dtype=float)
    elastic_net_values = np.asarray(elastic_net_paths, dtype=float)
    if sarima_values.ndim != 2 or elastic_net_values.ndim != 2:
        raise ValueError("both path arrays must be two-dimensional with shape (n_sims, steps).")
    if sarima_values.shape != elastic_net_values.shape:
        raise ValueError("both path arrays must have the same shape.")
    if isinstance(weights, dict):
        sarima_weight, elastic_net_weight = _weights_by_position(
            weights,
            sarima_values.shape[1],
            horizons,
        )
    return sarima_weight * sarima_values + elastic_net_weight * elastic_net_values


def forecast_ensemble(
    train_frame: pd.DataFrame,
    steps: int = 8,
    weights: tuple[float, float] | dict[int, tuple[float, float]] | None = None,
    seed: int = DEFAULT_SEED,
    target_column: str = TARGET_COLUMN,
    sarima_order: tuple[int, int, int] = SARIMA_DEFAULT_ORDER,
    sarima_seasonal_order: tuple[int, int, int, int] = SARIMA_DEFAULT_SEASONAL_ORDER,
    elastic_net_feature_columns: tuple[str, ...] = ELASTIC_NET_FEATURE_COLUMNS,
) -> np.ndarray:
    """Fit both components on raw training data and combine their point forecasts."""
    requested_horizons = tuple(range(1, steps + 1))
    if weights is None:
        weights = horizon_rmse_weights(horizons=requested_horizons)
    sarima_forecast = forecast_sarima(
        train_frame[target_column],
        steps=steps,
        order=sarima_order,
        seasonal_order=sarima_seasonal_order,
    )
    elastic_net_forecast = forecast_elastic_net_direct(
        train_frame,
        steps=steps,
        seed=seed,
        feature_columns=elastic_net_feature_columns,
        target_column=target_column,
    )
    return combine_point_forecasts(
        sarima_forecast,
        elastic_net_forecast,
        weights=weights,
        horizons=requested_horizons,
    )


def simulate_ensemble_paths(
    train_frame: pd.DataFrame,
    steps: int = 8,
    n_sims: int = 1000,
    weights: tuple[float, float] | dict[int, tuple[float, float]] | None = None,
    seed: int = DEFAULT_SEED,
    target_column: str = TARGET_COLUMN,
    sarima_order: tuple[int, int, int] = SARIMA_DEFAULT_ORDER,
    sarima_seasonal_order: tuple[int, int, int, int] = SARIMA_DEFAULT_SEASONAL_ORDER,
    elastic_net_feature_columns: tuple[str, ...] = ELASTIC_NET_FEATURE_COLUMNS,
) -> np.ndarray:
    """Fit both components on raw training data and combine their predictive draws."""
    requested_horizons = tuple(range(1, steps + 1))
    if weights is None:
        weights = horizon_rmse_weights(horizons=requested_horizons)
    sarima_paths = simulate_sarima_paths(
        train_frame[target_column],
        steps=steps,
        n_sims=n_sims,
        order=sarima_order,
        seasonal_order=sarima_seasonal_order,
        seed=seed,
    )
    elastic_net_paths = simulate_elastic_net_paths(
        train_frame,
        steps=steps,
        n_sims=n_sims,
        seed=seed + 1,
        feature_columns=elastic_net_feature_columns,
        target_column=target_column,
    )
    return combine_paths(
        sarima_paths,
        elastic_net_paths,
        weights=weights,
        horizons=requested_horizons,
    )


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
    weights: tuple[float, float] | dict[int, tuple[float, float]],
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
        horizons=merged["horizon"].to_numpy(),
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


def _log_ensemble_config(weights: tuple[float, float] | dict[int, tuple[float, float]]) -> None:
    from src.models import tracking

    mlflow = tracking.configure_mlflow()
    weights_for_log = (
        {str(horizon): list(pair) for horizon, pair in weights.items()}
        if isinstance(weights, dict)
        else list(weights)
    )
    mlflow.log_dict(
        {"weights": weights_for_log, "components": ["sarima", "elastic_net"]},
        "model_preprocessing/ensemble_weights.json",
    )


def run_ensemble_comparison(
    curated_path: Path = CURATED_DATA_PATH,
    rba_path: Path = RBA_FORECAST_PATH,
    comparison_output_path: Path = ENSEMBLE_COMPARISON_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    weights: tuple[float, float] | dict[int, tuple[float, float]] | None = None,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    target_column: str = TARGET_COLUMN,
    sarima_order: tuple[int, int, int] = SARIMA_DEFAULT_ORDER,
    sarima_seasonal_order: tuple[int, int, int, int] = SARIMA_DEFAULT_SEASONAL_ORDER,
    elastic_net_feature_columns: tuple[str, ...] = ELASTIC_NET_FEATURE_COLUMNS,
    include_rba: bool = True,
    run_name: str = "ensemble_comparison",
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the ensemble comparison, write its report, and log an MLflow run."""
    from src.models import tracking

    requested_horizons = tuple(int(horizon) for horizon in horizons)
    weighting_scheme = "fixed"
    if weights is None:
        weights = horizon_rmse_weights(horizons=requested_horizons)
        weighting_scheme = "dynamic_inverse_rmse_per_horizon"
    elif not isinstance(weights, dict):
        weights = _validate_weights(weights)
    series = load_target_series(curated_path, target_column=target_column)
    exog = load_elastic_net_feature_frame(
        curated_path,
        feature_columns=elastic_net_feature_columns,
    )

    if verbose:
        print("Running SARIMA walk-forward backtest...", flush=True)
    sarima_predictions = walk_forward_backtest(
        series=series,
        forecast_func=lambda train, steps: forecast_sarima(
            train,
            steps=steps,
            order=sarima_order,
            seasonal_order=sarima_seasonal_order,
        ),
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
            feature_columns=elastic_net_feature_columns,
            target_column=target_column,
        ),
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        model_name="elastic_net",
        target_column=target_column,
        max_origins=max_origins,
    )
    if elastic_net_predictions.empty:
        raise ValueError("Elastic Net backtest produced no forecast origins.")

    ensemble_predictions = _ensemble_prediction_frame(
        sarima_predictions=sarima_predictions,
        elastic_net_predictions=elastic_net_predictions,
        weights=weights,
    )
    baseline_predictions = compute_baseline_predictions(
        series=series,
        rba_path=rba_path,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        include_rba=include_rba,
    )

    frames = [
        ensemble_predictions,
        sarima_predictions,
        elastic_net_predictions,
        baseline_predictions,
    ]

    common_predictions = restrict_to_common_grid(frames)
    comparison = compute_metric_table(common_predictions)

    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.round({"rmse": 6, "mae": 6}).to_csv(comparison_output_path, index=False)

    tracking.log_model_run(
        run_name=run_name,
        model_name="ensemble",
        metrics=comparison,
        params={
            "weights": weights,
            "weighting_scheme": weighting_scheme,
            "weighting_note": (
                "Dynamic weights are derived from the same full walk-forward comparison "
                "sample used to report the ensemble RMSE, so the ensemble metric is not "
                "a strictly out-of-sample validation of the weighting scheme itself."
                if weighting_scheme == "dynamic_inverse_rmse_per_horizon"
                else ""
            ),
            "sarima_order": sarima_order,
            "sarima_seasonal_order": sarima_seasonal_order,
            "elastic_net_feature_columns": elastic_net_feature_columns,
            "target_column": target_column,
            "seed": seed,
            "initial_train_size": initial_train_size,
            "horizons": requested_horizons,
        },
        tags={
            "model_family": "ensemble",
            "target_column": target_column,
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
