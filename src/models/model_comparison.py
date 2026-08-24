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
    TARGET_COLUMN,
    forecast_elastic_net_direct,
    load_elastic_net_feature_frame,
)
from src.models.ensemble import _ensemble_prediction_frame, horizon_rmse_weights
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
from src.models.sarima import forecast_sarima
from src.models.sarimax_order_search import (
    DEFAULT_COMPARISON_MAX_ARMA_ORDER,
    ORDER_SELECTION_HOLDOUT_QUARTERS,
    _development_split,
    _group_predictions,
    _select_order,
    build_feature_groups,
    load_exog_frame,
    resolve_level_change_features,
    run_sarimax_order_search,
)


COMPARISON_ALL_OUTPUT_PATH = PROJECT_ROOT / "reports/model_comparison_all.csv"
PREDICTIONS_OUTPUT_PATH = PROJECT_ROOT / "reports/backtest_predictions.csv"
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
    "sarimax": 3,
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


def _sarimax_placeholder_rows(
    evaluated_horizons: set[int],
    requested_horizons: tuple[int, ...],
) -> pd.DataFrame:
    missing_horizons = [horizon for horizon in requested_horizons if horizon not in evaluated_horizons]
    return pd.DataFrame(
        [
            {
                "model": "sarimax",
                "horizon": horizon,
                "n": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "comparison_grid": "sarimax_group_d_shared_key_intersection",
                "evaluation_status": "not_evaluated_at_this_horizon",
                "forecast_origin_n": 0,
                "evaluated_horizons": _horizon_range_label(evaluated_horizons),
                "group_id": "D",
            }
            for horizon in missing_horizons
        ]
    )


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
    sarimax_group_d_predictions: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Build metrics from row-level predictions aligned by forecast keys.

    SARIMA, Elastic Net, Ensemble, seasonal-naive, and RBA are evaluated on
    their full shared 1-8 horizon grid. SARIMAX Group D is then intersected
    with that same key grid and reported only for horizons it genuinely emits.
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

    key_grid = wide_common.loc[:, KEY_COLUMNS].drop_duplicates()
    sarimax_aligned = sarimax_group_d_predictions.merge(key_grid, on=KEY_COLUMNS, how="inner")
    sarimax_frames: list[pd.DataFrame] = []
    evaluated_horizons: set[int] = set()
    if not sarimax_aligned.empty:
        evaluated_horizons = {int(value) for value in sarimax_aligned["horizon"].unique()}
        sarimax_metrics = _with_grid_metadata(
            compute_metric_table(sarimax_aligned),
            predictions=sarimax_aligned,
            comparison_grid="sarimax_group_d_shared_key_intersection",
            evaluation_status="evaluated_on_available_horizons",
            group_id="D",
        )
        sarimax_frames.append(sarimax_metrics)
    sarimax_placeholders = _sarimax_placeholder_rows(evaluated_horizons, requested_horizons)
    if not sarimax_placeholders.empty:
        sarimax_frames.append(sarimax_placeholders)

    metrics = pd.concat([wide_metrics, *sarimax_frames], ignore_index=True, sort=False)
    metrics = _assign_best_model(metrics, full_horizon_label=_horizon_range_label(requested_horizons))
    metrics["_horizon_order"] = metrics["horizon"].map(_horizon_order)
    metrics["_model_order"] = metrics["model"].map(MODEL_ORDER).fillna(99)
    return metrics.sort_values(["_horizon_order", "_model_order"]).drop(
        columns=["_horizon_order", "_model_order"]
    )


def build_backtest_predictions_table(
    prediction_frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Stack row-level walk-forward predictions from every model into one table.

    Each source frame (SARIMA, Elastic Net, Ensemble, seasonal-naive, RBA,
    SARIMAX Group D) carries the same forecast/actual/error core, plus a few
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


def _sarimax_group_d_predictions(
    curated_path: Path,
    initial_train_size: int,
    horizons: tuple[int, ...],
    criterion: str,
    max_p: int,
    max_q: int,
    max_p_seasonal: int,
    max_q_seasonal: int,
    d_values: Iterable[int],
    seasonal_d_values: Iterable[int],
    maxiter: int,
    order_selection_holdout_quarters: int,
) -> pd.DataFrame:
    series = load_target_series(curated_path)
    exog = load_exog_frame(curated_path)
    development_series, development_exog = _development_split(
        series,
        exog,
        holdout_quarters=order_selection_holdout_quarters,
    )
    choices = resolve_level_change_features(development_series, development_exog, maxiter=maxiter)
    group_d = next(group for group in build_feature_groups(choices) if group.group_id == "D")
    search_results = run_sarimax_order_search(
        series=development_series,
        exog=development_exog.loc[:, list(group_d.features)],
        max_p=max_p,
        max_q=max_q,
        max_p_seasonal=max_p_seasonal,
        max_q_seasonal=max_q_seasonal,
        d_values=d_values,
        seasonal_d_values=seasonal_d_values,
        trend="n",
        maxiter=maxiter,
    )
    best = _select_order(search_results, criterion)
    return _group_predictions(
        group=group_d,
        series=series,
        exog=exog,
        order=best["order"],
        seasonal_order=best["seasonal_order"],
        initial_train_size=initial_train_size,
        horizons=horizons,
        maxiter=maxiter,
    )


def run_model_comparison_all(
    curated_path: Path = CURATED_DATA_PATH,
    rba_path: Path = RBA_FORECAST_PATH,
    output_path: Path = COMPARISON_ALL_OUTPUT_PATH,
    predictions_output_path: Path | None = PREDICTIONS_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    weights: tuple[float, float] | dict[int, tuple[float, float]] | None = None,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    criterion: str = "aic",
    max_p: int = DEFAULT_COMPARISON_MAX_ARMA_ORDER,
    max_q: int = DEFAULT_COMPARISON_MAX_ARMA_ORDER,
    max_p_seasonal: int = DEFAULT_COMPARISON_MAX_ARMA_ORDER,
    max_q_seasonal: int = DEFAULT_COMPARISON_MAX_ARMA_ORDER,
    d_values: Iterable[int] = (0,),
    seasonal_d_values: Iterable[int] = (0,),
    maxiter: int = 100,
    order_selection_holdout_quarters: int = ORDER_SELECTION_HOLDOUT_QUARTERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute and save the joined all-model comparison report."""
    started = time.perf_counter()
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    if weights is None:
        weights = horizon_rmse_weights(horizons=requested_horizons)
    series = load_target_series(curated_path)
    exog = load_elastic_net_feature_frame(curated_path)

    if verbose:
        print("Running SARIMA row-level backtest...", flush=True)
    sarima_predictions = walk_forward_backtest(
        series=series,
        forecast_func=lambda train, steps: forecast_sarima(train, steps=steps),
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
    baseline_predictions = compute_baseline_predictions(
        series=series,
        rba_path=rba_path,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
    )

    if verbose:
        print("Running SARIMAX Group D row-level backtest...", flush=True)
    sarimax_group_d = _sarimax_group_d_predictions(
        curated_path=curated_path,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        criterion=criterion,
        max_p=max_p,
        max_q=max_q,
        max_p_seasonal=max_p_seasonal,
        max_q_seasonal=max_q_seasonal,
        d_values=d_values,
        seasonal_d_values=seasonal_d_values,
        maxiter=maxiter,
        order_selection_holdout_quarters=order_selection_holdout_quarters,
    )

    comparison = build_joined_metric_table(
        wide_prediction_frames=[
            sarima_predictions,
            elastic_net_predictions,
            ensemble_predictions,
            baseline_predictions,
        ],
        sarimax_group_d_predictions=sarimax_group_d,
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
                sarimax_group_d,
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


def _parse_int_values(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--rba-data", type=Path, default=RBA_FORECAST_PATH)
    parser.add_argument("--output", type=Path, default=COMPARISON_ALL_OUTPUT_PATH)
    parser.add_argument("--predictions-output", type=Path, default=PREDICTIONS_OUTPUT_PATH)
    parser.add_argument(
        "--no-predictions-output",
        action="store_true",
        help="Skip writing the row-level backtest predictions table.",
    )
    parser.add_argument("--initial-train-size", type=int, default=DEFAULT_INITIAL_TRAIN_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--criterion", choices=("aic", "bic"), default="aic")
    parser.add_argument("--d-values", default="0")
    parser.add_argument("--seasonal-d-values", default="0")
    parser.add_argument("--max-p", type=int, default=DEFAULT_COMPARISON_MAX_ARMA_ORDER)
    parser.add_argument("--max-q", type=int, default=DEFAULT_COMPARISON_MAX_ARMA_ORDER)
    parser.add_argument("--max-p-seasonal", type=int, default=DEFAULT_COMPARISON_MAX_ARMA_ORDER)
    parser.add_argument("--max-q-seasonal", type=int, default=DEFAULT_COMPARISON_MAX_ARMA_ORDER)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument(
        "--order-selection-holdout-quarters",
        type=int,
        default=ORDER_SELECTION_HOLDOUT_QUARTERS,
    )
    args = parser.parse_args(argv)

    comparison = run_model_comparison_all(
        curated_path=args.data,
        rba_path=args.rba_data,
        output_path=args.output,
        predictions_output_path=None if args.no_predictions_output else args.predictions_output,
        initial_train_size=args.initial_train_size,
        seed=args.seed,
        max_origins=args.max_origins,
        criterion=args.criterion,
        max_p=args.max_p,
        max_q=args.max_q,
        max_p_seasonal=args.max_p_seasonal,
        max_q_seasonal=args.max_q_seasonal,
        d_values=_parse_int_values(args.d_values),
        seasonal_d_values=_parse_int_values(args.seasonal_d_values),
        maxiter=args.maxiter,
        order_selection_holdout_quarters=args.order_selection_holdout_quarters,
        verbose=True,
    )
    print(comparison.round({"rmse": 3, "mae": 3}).to_string(index=False))


if __name__ == "__main__":
    main()
