"""Joined row-level model comparison across CPI forecast families."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.models.elastic_net import (
    DEFAULT_INITIAL_TRAIN_SIZE,
    DEFAULT_SEED,
    ELASTIC_NET_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
    TRIMMED_MEAN_TARGET_COLUMN,
    forecast_elastic_net_direct,
    load_elastic_net_feature_frame,
    run_elastic_net_comparison,
)
from src.models.ensemble import (
    _ensemble_prediction_frame,
    dynamic_weights_source_path,
    horizon_rmse_weights,
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
    run_sarima_comparison,
    walk_forward_backtest,
    walk_forward_backtest_direct_multihorizon,
)
from src.models.sarima import forecast_sarima
from src.models.sarima import DEFAULT_ORDER as SARIMA_DEFAULT_ORDER
from src.models.sarima import DEFAULT_SEASONAL_ORDER as SARIMA_DEFAULT_SEASONAL_ORDER
from src.models.sarima import TRIMMED_MEAN_DEFAULT_ORDER, TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER
from src.models import svar


COMPARISON_ALL_OUTPUT_PATH = PROJECT_ROOT / "reports/model_comparison_all.csv"
TRIMMED_MEAN_COMPARISON_ALL_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/model_comparison_trimmed_mean_all.csv"
)
PREDICTIONS_OUTPUT_PATH = PROJECT_ROOT / "reports/backtest_predictions.csv"
TRIMMED_MEAN_PREDICTIONS_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/backtest_predictions_trimmed_mean.csv"
)
TRIMMED_MEAN_SARIMA_COMPARISON_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/model_comparison_sarima_trimmed_mean.csv"
)
TRIMMED_MEAN_ELASTIC_NET_COMPARISON_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/model_comparison_elastic_net_trimmed_mean.csv"
)
TRIMMED_MEAN_ELASTIC_NET_COEFFICIENT_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/elastic_net_coefficients_trimmed_mean.csv"
)
PREDICTIONS_COLUMNS = [
    "model",
    "forecast_origin",
    "target_quarter",
    "horizon",
    "actual",
    "forecast",
    "error",
    "rba_actual",
    "rba_reported_error",
    "horizon_cap",
]
KEY_COLUMNS = ["forecast_origin", "target_quarter", "horizon"]
MODEL_ORDER = {
    "sarima": 0,
    "elastic_net": 1,
    "ensemble": 2,
    "seasonal_naive": 4,
    "rba": 5,
}


def _horizon_label(horizon: object) -> str:
    return "overall" if str(horizon) == "overall" else str(int(horizon))


def _horizon_order(horizon: object) -> int:
    return 0 if str(horizon) == "overall" else int(horizon)


def _horizon_range_label(horizons: Iterable[int]) -> str:
    values = sorted({int(horizon) for horizon in horizons})
    if not values:
        return "none"
    if values == list(range(values[0], values[-1] + 1)):
        return str(values[0]) if values[0] == values[-1] else f"{values[0]}-{values[-1]}"
    return ",".join(str(value) for value in values)


def _origin_count(predictions: pd.DataFrame, model: str, horizon: object) -> int:
    rows = predictions.loc[predictions["model"].eq(model)]
    if str(horizon) != "overall":
        rows = rows.loc[rows["horizon"].astype(int).eq(int(horizon))]
    return int(rows["forecast_origin"].nunique())


def _with_grid_metadata(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    comparison_grid: str,
    evaluation_status: str,
    group_id: str = "",
) -> pd.DataFrame:
    result = metrics.copy()
    result["comparison_grid"] = comparison_grid
    result["evaluation_status"] = evaluation_status
    result["forecast_origin_n"] = [
        _origin_count(predictions, str(row.model), row.horizon)
        for row in result.itertuples(index=False)
    ]
    result["evaluated_horizons"] = [
        _horizon_range_label(
            predictions.loc[predictions["model"].eq(str(row.model)), "horizon"].unique()
        )
        for row in result.itertuples(index=False)
    ]
    result["group_id"] = group_id
    return result


def _assign_best_model(metrics: pd.DataFrame, full_horizon_label: str) -> pd.DataFrame:
    result = metrics.copy()
    result["best_model"] = pd.NA
    for horizon, rows in result.groupby(result["horizon"].map(_horizon_label), sort=False):
        real_rows = rows.dropna(subset=["rmse"])
        if horizon == "overall":
            real_rows = real_rows.loc[real_rows["evaluated_horizons"].eq(full_horizon_label)]
        if real_rows.empty:
            continue
        best_index = real_rows.sort_values(
            ["rmse", "model"],
            key=lambda column: column.map(MODEL_ORDER).fillna(99) if column.name == "model" else column,
        ).index[0]
        result.loc[rows.index, "best_model"] = result.loc[best_index, "model"]
    return result


def build_joined_metric_table(
    wide_prediction_frames: Sequence[pd.DataFrame],
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Build metrics from row-level predictions aligned by forecast keys.

    SARIMA, Elastic Net, Ensemble, seasonal-naive, and optional RBA are
    evaluated on their full shared 1-8 horizon grid.
    """
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    if not wide_prediction_frames:
        raise ValueError("wide_prediction_frames must contain at least one prediction frame.")

    wide_common = restrict_to_common_grid(wide_prediction_frames)
    if wide_common.empty:
        raise ValueError("wide model predictions have no common forecast grid.")
    wide_metrics = _with_grid_metadata(
        compute_metric_table(wide_common),
        predictions=wide_common,
        comparison_grid="shared_horizon_1_8_grid",
        evaluation_status="evaluated",
    )

    metrics = _assign_best_model(wide_metrics, full_horizon_label=_horizon_range_label(requested_horizons))
    metrics["_horizon_order"] = metrics["horizon"].map(_horizon_order)
    metrics["_model_order"] = metrics["model"].map(MODEL_ORDER).fillna(99)
    return metrics.sort_values(["_horizon_order", "_model_order"]).drop(
        columns=["_horizon_order", "_model_order"]
    )


def build_backtest_predictions_table(
    prediction_frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Stack row-level walk-forward predictions from every model into one table.

    Each source frame (SARIMA, Elastic Net, Ensemble, seasonal-naive, RBA)
    carries the same forecast/actual/error core, plus a few
    model-specific extras (``rba_actual``, ``rba_reported_error``,
    ``horizon_cap``). This is a plain outer-union stack of those origin-level
    rows -- unlike ``build_joined_metric_table``, it is not restricted to a
    common forecast grid, so every model's full evaluated history is kept
    for BI/downstream consumption (e.g. an actual-vs-forecast-over-time chart).
    """
    if not prediction_frames:
        raise ValueError("prediction_frames must contain at least one prediction frame.")

    combined = pd.concat(prediction_frames, ignore_index=True, sort=False)
    for column in PREDICTIONS_COLUMNS:
        if column not in combined.columns:
            combined[column] = pd.NA
    combined = combined.loc[:, PREDICTIONS_COLUMNS].copy()
    combined["_model_order"] = combined["model"].map(MODEL_ORDER).fillna(99)
    combined = combined.sort_values(
        ["_model_order", "forecast_origin", "horizon"]
    ).drop(columns="_model_order")
    return combined.reset_index(drop=True)


def run_model_comparison_all(
    curated_path: Path = CURATED_DATA_PATH,
    rba_path: Path = RBA_FORECAST_PATH,
    include_rba: bool = True,
    output_path: Path = COMPARISON_ALL_OUTPUT_PATH,
    predictions_output_path: Path | None = PREDICTIONS_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    weights: tuple[float, float] | dict[int, tuple[float, float]] | None = None,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    target_column: str = TARGET_COLUMN,
    elastic_net_feature_columns: tuple[str, ...] = ELASTIC_NET_FEATURE_COLUMNS,
    sarima_order: tuple[int, int, int] = SARIMA_DEFAULT_ORDER,
    sarima_seasonal_order: tuple[int, int, int, int] = SARIMA_DEFAULT_SEASONAL_ORDER,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute and save the joined all-model comparison report."""
    started = time.perf_counter()
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    if weights is None:
        weights = horizon_rmse_weights(
            horizons=requested_horizons,
            path=dynamic_weights_source_path(target_column),
        )
    series = load_target_series(
        curated_path,
        target_column=target_column,
        max_quarter=svar.FORECAST_ORIGIN_PIN,
    )
    exog = load_elastic_net_feature_frame(
        curated_path,
        feature_columns=elastic_net_feature_columns,
        max_quarter=svar.FORECAST_ORIGIN_PIN,
    )

    if verbose:
        print("Running SARIMA row-level backtest...", flush=True)
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
        print("Running Elastic Net row-level backtest...", flush=True)
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

    comparison = build_joined_metric_table(
        wide_prediction_frames=[
            sarima_predictions,
            elastic_net_predictions,
            ensemble_predictions,
            baseline_predictions,
        ],
        horizons=requested_horizons,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.round({"rmse": 6, "mae": 6}).to_csv(output_path, index=False)

    if predictions_output_path is not None:
        predictions = build_backtest_predictions_table(
            [
                sarima_predictions,
                elastic_net_predictions,
                ensemble_predictions,
                baseline_predictions,
            ]
        )
        predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.round({"actual": 6, "forecast": 6, "error": 6}).to_csv(
            predictions_output_path, index=False
        )
        if verbose:
            print(f"Wrote {predictions_output_path} in {time.perf_counter() - started:.1f}s", flush=True)

    if verbose:
        print(f"Wrote {output_path} in {time.perf_counter() - started:.1f}s", flush=True)
    return comparison


def refresh_trimmed_mean_served_model_runs(
    curated_path: Path = CURATED_DATA_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    verbose: bool = False,
) -> None:
    """Log full-sample trimmed-mean model artifacts used by FastAPI serving."""
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    if verbose:
        print("Refreshing trimmed-mean SARIMA served MLflow run...", flush=True)
    run_sarima_comparison(
        curated_path=curated_path,
        output_path=TRIMMED_MEAN_SARIMA_COMPARISON_OUTPUT_PATH,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        target_column=TRIMMED_MEAN_TARGET_COLUMN,
        order=TRIMMED_MEAN_DEFAULT_ORDER,
        seasonal_order=TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER,
        include_rba=False,
        run_name="trimmed_mean_sarima_comparison",
        model_family_tag="trimmed_mean_sarima",
        selection_criterion="fixed_trimmed_mean_default",
        max_quarter=svar.FORECAST_ORIGIN_PIN,
    )

    if verbose:
        print("Refreshing trimmed-mean Elastic Net served MLflow run...", flush=True)
    run_elastic_net_comparison(
        curated_path=curated_path,
        rba_path=RBA_FORECAST_PATH,
        comparison_output_path=TRIMMED_MEAN_ELASTIC_NET_COMPARISON_OUTPUT_PATH,
        coefficient_output_path=TRIMMED_MEAN_ELASTIC_NET_COEFFICIENT_OUTPUT_PATH,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        seed=seed,
        max_origins=max_origins,
        target_column=TRIMMED_MEAN_TARGET_COLUMN,
        feature_columns=TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
        sarima_order=TRIMMED_MEAN_DEFAULT_ORDER,
        sarima_seasonal_order=TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER,
        include_rba=False,
        run_name="trimmed_mean_elastic_net_comparison",
        model_family_tag="trimmed_mean_elastic_net",
        feature_set_label="trimmed_mean_primary_wti",
        verbose=verbose,
        max_quarter=svar.FORECAST_ORIGIN_PIN,
    )


def run_trimmed_mean_model_comparison_all(
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = TRIMMED_MEAN_COMPARISON_ALL_OUTPUT_PATH,
    predictions_output_path: Path | None = TRIMMED_MEAN_PREDICTIONS_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Reproduce the Phase 3 trimmed-mean all-model comparison report."""
    comparison = run_model_comparison_all(
        curated_path=curated_path,
        include_rba=False,
        output_path=output_path,
        predictions_output_path=predictions_output_path,
        initial_train_size=initial_train_size,
        horizons=horizons,
        weights=None,
        seed=seed,
        max_origins=max_origins,
        target_column=TRIMMED_MEAN_TARGET_COLUMN,
        elastic_net_feature_columns=TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
        sarima_order=TRIMMED_MEAN_DEFAULT_ORDER,
        sarima_seasonal_order=TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER,
        verbose=verbose,
    )
    refresh_trimmed_mean_served_model_runs(
        curated_path=curated_path,
        initial_train_size=initial_train_size,
        horizons=horizons,
        seed=seed,
        max_origins=max_origins,
        verbose=verbose,
    )
    return comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("headline", "trimmed_mean"), default="headline")
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--rba-data", type=Path, default=RBA_FORECAST_PATH)
    parser.add_argument(
        "--no-rba",
        action="store_true",
        help=(
            "Exclude RBA historical forecasts even when --rba-data exists. Use this "
            "for the SA-basis headline cpi_yoy refit because the RBA CPI forecast "
            "workbook is on the officially quoted NSA basis."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument(
        "--no-predictions-output",
        action="store_true",
        help="Skip writing the row-level backtest predictions table.",
    )
    parser.add_argument("--initial-train-size", type=int, default=DEFAULT_INITIAL_TRAIN_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-origins", type=int, default=None)
    args = parser.parse_args(argv)

    if args.target == "trimmed_mean":
        comparison = run_trimmed_mean_model_comparison_all(
            curated_path=args.data,
            output_path=args.output or TRIMMED_MEAN_COMPARISON_ALL_OUTPUT_PATH,
            predictions_output_path=(
                None
                if args.no_predictions_output
                else args.predictions_output or TRIMMED_MEAN_PREDICTIONS_OUTPUT_PATH
            ),
            initial_train_size=args.initial_train_size,
            seed=args.seed,
            max_origins=args.max_origins,
            verbose=True,
        )
    else:
        comparison = run_model_comparison_all(
            curated_path=args.data,
            rba_path=args.rba_data,
            include_rba=not args.no_rba,
            output_path=args.output or COMPARISON_ALL_OUTPUT_PATH,
            predictions_output_path=(
                None if args.no_predictions_output else args.predictions_output or PREDICTIONS_OUTPUT_PATH
            ),
            initial_train_size=args.initial_train_size,
            seed=args.seed,
            max_origins=args.max_origins,
            verbose=True,
        )
    print(comparison.round({"rmse": 3, "mae": 3}).to_string(index=False))


if __name__ == "__main__":
    main()
