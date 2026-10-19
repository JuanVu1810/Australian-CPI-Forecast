"""Walk-forward forecast interval coverage report for CPI model families."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.elastic_net import (
    DEFAULT_INITIAL_TRAIN_SIZE as ELASTIC_NET_INITIAL_TRAIN_SIZE,
    ELASTIC_NET_FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_elastic_net_feature_frame,
    simulate_elastic_net_paths,
)
from src.models.ensemble import simulate_ensemble_paths
from src.models.evaluation import (
    CURATED_DATA_PATH,
    DEFAULT_HORIZONS,
    PROJECT_ROOT,
    compute_interval_coverage_table,
    compute_rolling_conformal_scale_factors,
    load_target_series,
    walk_forward_interval_coverage_backtest,
)
from src.models.sarima import simulate_sarima_paths
from src.models.sarima import DEFAULT_ORDER as SARIMA_DEFAULT_ORDER
from src.models.sarima import DEFAULT_SEASONAL_ORDER as SARIMA_DEFAULT_SEASONAL_ORDER
from src.models import svar


INTERVAL_COVERAGE_OUTPUT_PATH = PROJECT_ROOT / "reports/model_interval_coverage.csv"
INTERVAL_CALIBRATION_FACTORS_PATH = (
    PROJECT_ROOT / "reports/model_interval_calibration_factors.csv"
)
SAMPLE_GRID_NOTE = (
    "Coverage is evaluated on each family's own achievable walk-forward origins; "
    "rows are not restricted to a cross-family common grid."
)
DEFAULT_N_SIMS = 1000
DEFAULT_SEED = 42
DEFAULT_LOWER_QUANTILE = 0.1
DEFAULT_UPPER_QUANTILE = 0.9
DEFAULT_INITIAL_TRAIN_SIZE = ELASTIC_NET_INITIAL_TRAIN_SIZE


def apply_interval_calibration(
    predictions: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Apply multiplicative per-side calibration around the point forecast.

    Factors are keyed by model and horizon, plus ``forecast_origin`` when the
    factor frame carries it (rolling, ex-ante factors). Rows with no matching
    factor keep their raw interval.
    """
    if predictions.empty or factors.empty:
        return predictions.copy()

    required_predictions = {
        "model",
        "horizon",
        "actual",
        "point_forecast_proxy",
        "interval_lower",
        "interval_upper",
    }
    missing_predictions = required_predictions.difference(predictions.columns)
    if missing_predictions:
        raise ValueError(
            f"interval predictions missing required columns: {sorted(missing_predictions)}"
        )
    required_factors = {"model", "horizon", "scale_factor"}
    missing_factors = required_factors.difference(factors.columns)
    if missing_factors:
        raise ValueError(f"calibration factors missing required columns: {sorted(missing_factors)}")

    result = predictions.copy()
    keys = ["model", "horizon"]
    if "forecast_origin" in factors.columns:
        keys = ["model", "forecast_origin", "horizon"]
        if "forecast_origin" not in predictions.columns:
            raise ValueError("origin-keyed calibration factors need forecast_origin in predictions.")
    factor_frame = pd.DataFrame(factors).loc[:, [*keys, "scale_factor"]].copy()
    factor_frame["horizon"] = factor_frame["horizon"].astype(int)
    result["horizon"] = result["horizon"].astype(int)
    result = result.merge(
        factor_frame,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    scale = result["scale_factor"].astype(float)
    apply_mask = scale.notna() & np.isfinite(scale) & scale.ge(0)
    center = result["point_forecast_proxy"].astype(float)
    lower_dist = center - result["interval_lower"].astype(float)
    upper_dist = result["interval_upper"].astype(float) - center

    result.loc[apply_mask, "interval_lower"] = (
        center.loc[apply_mask] - lower_dist.loc[apply_mask] * scale.loc[apply_mask]
    )
    result.loc[apply_mask, "interval_upper"] = (
        center.loc[apply_mask] + upper_dist.loc[apply_mask] * scale.loc[apply_mask]
    )
    result.loc[apply_mask, "hit"] = (
        result.loc[apply_mask, "interval_lower"].astype(float)
        <= result.loc[apply_mask, "actual"].astype(float)
    ) & (
        result.loc[apply_mask, "actual"].astype(float)
        <= result.loc[apply_mask, "interval_upper"].astype(float)
    )
    result.loc[apply_mask, "interval_width"] = (
        result.loc[apply_mask, "interval_upper"].astype(float)
        - result.loc[apply_mask, "interval_lower"].astype(float)
    )
    result["interval_calibrated"] = apply_mask.astype(bool)
    result["interval_scale_factor"] = result["scale_factor"]
    return result.drop(columns=["scale_factor"])


def run_family_interval_backtests(
    curated_path: Path = CURATED_DATA_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float = DEFAULT_UPPER_QUANTILE,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    skip_origins: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run raw interval walk-forward backtests for the three served model families."""
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    series = load_target_series(
        curated_path,
        target_column=TARGET_COLUMN,
        max_quarter=svar.FORECAST_ORIGIN_PIN,
    )
    elastic_net_exog = load_elastic_net_feature_frame(
        curated_path,
        feature_columns=ELASTIC_NET_FEATURE_COLUMNS,
        max_quarter=svar.FORECAST_ORIGIN_PIN,
    )

    # Each simulate function fits one family on the data available at a forecast origin
    # and returns simulated future paths; the walk-forward harness calls it once per origin.
    def simulate_sarima(train_series, steps, n_sims, seed):
        return simulate_sarima_paths(
            train_series,
            steps=steps,
            n_sims=n_sims,
            order=SARIMA_DEFAULT_ORDER,
            seasonal_order=SARIMA_DEFAULT_SEASONAL_ORDER,
            seed=seed,
        )

    def simulate_elastic_net(train_frame, steps, n_sims, seed):
        return simulate_elastic_net_paths(
            train_frame,
            steps=steps,
            n_sims=n_sims,
            seed=seed,
            feature_columns=ELASTIC_NET_FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
        )

    if verbose:
        print("Running SARIMA interval coverage backtest...", flush=True)
    sarima_predictions = walk_forward_interval_coverage_backtest(
        series=series,
        simulate_func=simulate_sarima,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        model_name="sarima",
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_sims=n_sims,
        seed=seed,
        max_origins=max_origins,
        skip_origins=skip_origins,
    )

    if verbose:
        print("Running Elastic Net interval coverage backtest...", flush=True)
    elastic_net_predictions = walk_forward_interval_coverage_backtest(
        series=series,
        exog=elastic_net_exog,
        simulate_func=simulate_elastic_net,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        model_name="elastic_net",
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_sims=n_sims,
        seed=seed + 10_000,
        target_column=TARGET_COLUMN,
        max_origins=max_origins,
        skip_origins=skip_origins,
    )

    if verbose:
        print("Running ensemble interval coverage backtest...", flush=True)
    # The ensemble gets the same target-plus-features frame as Elastic Net; its SARIMA half
    # uses that frame's target column.
    ensemble_predictions = walk_forward_interval_coverage_backtest(
        series=series,
        exog=elastic_net_exog,
        simulate_func=simulate_ensemble_paths,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        model_name="ensemble",
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_sims=n_sims,
        seed=seed + 30_000,
        target_column=TARGET_COLUMN,
        max_origins=max_origins,
        skip_origins=skip_origins,
    )

    return pd.concat(
        [sarima_predictions, elastic_net_predictions, ensemble_predictions],
        ignore_index=True,
        sort=False,
    )


def run_interval_coverage(
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = INTERVAL_COVERAGE_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float = DEFAULT_UPPER_QUANTILE,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    skip_origins: int = 0,
    apply_calibration: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the interval backtests, optionally calibrate them, and save the coverage table.

    Each family keeps its own origin grid (no common grid), so sample sizes can differ between models.
    With ``apply_calibration`` each origin's interval is widened by a rolling factor built only from
    earlier realised scores; origins without enough history stay raw.
    """
    predictions = run_family_interval_backtests(
        curated_path=curated_path,
        initial_train_size=initial_train_size,
        horizons=horizons,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_sims=n_sims,
        seed=seed,
        max_origins=max_origins,
        skip_origins=skip_origins,
        verbose=verbose,
    )

    if predictions.empty:
        raise ValueError("Interval coverage backtests produced no forecast origins.")
    if apply_calibration:
        factors = compute_rolling_conformal_scale_factors(
            predictions,
            target_coverage=float(upper_quantile - lower_quantile),
        )
        predictions = apply_interval_calibration(predictions, factors)

    coverage = compute_interval_coverage_table(
        predictions,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    coverage["sample_grid_note"] = SAMPLE_GRID_NOTE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.round(
        {
            "nominal_coverage": 6,
            "empirical_coverage": 6,
            "coverage_ci_lower": 6,
            "coverage_ci_upper": 6,
            "binom_p_value": 6,
            "mean_interval_width": 6,
        }
    ).to_csv(output_path, index=False)

    if verbose:
        print(f"Saved interval coverage report to {output_path}", flush=True)
    return coverage


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--output", type=Path, default=INTERVAL_COVERAGE_OUTPUT_PATH)
    parser.add_argument("--initial-train-size", type=int, default=DEFAULT_INITIAL_TRAIN_SIZE)
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lower-quantile", type=float, default=DEFAULT_LOWER_QUANTILE)
    parser.add_argument("--upper-quantile", type=float, default=DEFAULT_UPPER_QUANTILE)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--skip-origins", type=int, default=0)
    parser.add_argument("--raw", action="store_true", help="Report raw, uncalibrated intervals.")
    args = parser.parse_args(argv)

    coverage = run_interval_coverage(
        curated_path=args.data,
        output_path=args.output,
        initial_train_size=args.initial_train_size,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        n_sims=args.n_sims,
        seed=args.seed,
        max_origins=args.max_origins,
        skip_origins=args.skip_origins,
        apply_calibration=not args.raw,
        verbose=True,
    )
    print("\nInterval coverage:")
    print(
        coverage.round(
            {
                "nominal_coverage": 3,
                "empirical_coverage": 3,
                "coverage_ci_lower": 3,
                "coverage_ci_upper": 3,
                "binom_p_value": 3,
                "mean_interval_width": 3,
            }
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
