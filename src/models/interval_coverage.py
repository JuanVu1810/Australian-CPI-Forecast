"""Walk-forward forecast interval coverage report for CPI model families."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.models.elastic_net import (
    DEFAULT_INITIAL_TRAIN_SIZE as ELASTIC_NET_INITIAL_TRAIN_SIZE,
    TARGET_COLUMN,
    load_elastic_net_feature_frame,
    simulate_elastic_net_paths,
)
from src.models.ensemble import horizon_rmse_weights, simulate_ensemble_paths
from src.models.evaluation import (
    CURATED_DATA_PATH,
    DEFAULT_HORIZONS,
    PROJECT_ROOT,
    compute_interval_coverage_table,
    load_target_series,
    walk_forward_interval_coverage_backtest,
)
from src.models.sarima import simulate_sarima_paths
from src.models.sarimax import GROUP_D_FEATURE_COLUMNS, simulate_sarimax_group_d_paths


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
SARIMAX_GROUP_D_BASE_COLUMNS = (
    "cash_rate_change",
    "unemployment_rate_change",
    "inflation_expectations_business",
    "ppi_growth_lag1",
    "commodity_growth",
    "wti_growth",
)


def _normalise_families(families: tuple[str, ...] | None) -> tuple[str, ...]:
    available = ("sarima", "elastic_net", "sarimax_group_d", "ensemble")
    if families is None:
        return available
    requested = tuple(str(family) for family in families)
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError(f"unknown interval coverage families: {unknown}")
    return requested


def load_sarimax_group_d_interval_exog(path: Path = CURATED_DATA_PATH) -> pd.DataFrame:
    """Load the wide macro frame needed by SARIMAX Group D simulation."""
    required_columns = (
        "quarter",
        *GROUP_D_FEATURE_COLUMNS,
        *SARIMAX_GROUP_D_BASE_COLUMNS,
    )
    df = pd.read_csv(path, usecols=list(dict.fromkeys(required_columns)))
    df.index = pd.PeriodIndex(df.pop("quarter").astype(str), freq="Q")
    return df.sort_index().astype(float)


def load_interval_calibration_factors(
    path: Path = INTERVAL_CALIBRATION_FACTORS_PATH,
) -> pd.DataFrame:
    """Load interval calibration factors, returning an empty frame when absent."""
    columns = ["model", "horizon", "scale_factor", "calibration_n", "target_coverage"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    factors = pd.read_csv(path)
    required = {"model", "horizon", "scale_factor"}
    missing = required.difference(factors.columns)
    if missing:
        raise ValueError(f"{path} missing required calibration columns: {sorted(missing)}")
    factors = factors.copy()
    factors["horizon"] = pd.to_numeric(factors["horizon"], errors="coerce")
    factors["scale_factor"] = pd.to_numeric(factors["scale_factor"], errors="coerce")
    factors = factors.dropna(subset=["model", "horizon", "scale_factor"])
    factors = factors.loc[np.isfinite(factors["scale_factor"]) & factors["scale_factor"].ge(0)]
    factors["horizon"] = factors["horizon"].astype(int)
    return factors


def apply_interval_calibration(
    predictions: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Apply per-model/horizon multiplicative half-width calibration."""
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
    factor_frame = pd.DataFrame(factors).loc[:, ["model", "horizon", "scale_factor"]].copy()
    factor_frame["horizon"] = factor_frame["horizon"].astype(int)
    result["horizon"] = result["horizon"].astype(int)
    result = result.merge(
        factor_frame,
        on=["model", "horizon"],
        how="left",
        validate="many_to_one",
    )

    scale = result["scale_factor"].astype(float)
    apply_mask = scale.notna() & np.isfinite(scale) & scale.ge(0)
    raw_half_width = (
        result["interval_upper"].astype(float) - result["interval_lower"].astype(float)
    ) / 2.0
    center = result["point_forecast_proxy"].astype(float)
    calibrated_half_width = raw_half_width * scale

    result.loc[apply_mask, "interval_lower"] = (
        center.loc[apply_mask] - calibrated_half_width.loc[apply_mask]
    )
    result.loc[apply_mask, "interval_upper"] = (
        center.loc[apply_mask] + calibrated_half_width.loc[apply_mask]
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
    families: tuple[str, ...] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run raw interval walk-forward backtests for selected served model families."""
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    requested_families = _normalise_families(families)
    series = load_target_series(curated_path)
    elastic_net_exog = load_elastic_net_feature_frame(curated_path)
    sarimax_exog = load_sarimax_group_d_interval_exog(curated_path)
    weights = horizon_rmse_weights(horizons=requested_horizons)

    frames: list[pd.DataFrame] = []
    if "sarima" in requested_families:
        if verbose:
            print("Running SARIMA interval coverage backtest...", flush=True)
        frames.append(
            walk_forward_interval_coverage_backtest(
                series=series,
                simulate_func=simulate_sarima_paths,
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
        )

    if "elastic_net" in requested_families:
        if verbose:
            print("Running Elastic Net interval coverage backtest...", flush=True)
        frames.append(
            walk_forward_interval_coverage_backtest(
                series=series,
                exog=elastic_net_exog,
                simulate_func=simulate_elastic_net_paths,
                initial_train_size=initial_train_size,
                horizons=requested_horizons,
                model_name="elastic_net",
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                n_sims=n_sims,
                seed=seed + 10_000,
                training_data="frame",
                target_column=TARGET_COLUMN,
                max_origins=max_origins,
                skip_origins=skip_origins,
            )
        )

    if "sarimax_group_d" in requested_families:
        if verbose:
            print("Running SARIMAX Group D interval coverage backtest...", flush=True)
        frames.append(
            walk_forward_interval_coverage_backtest(
                series=series,
                exog=sarimax_exog,
                simulate_func=simulate_sarimax_group_d_paths,
                initial_train_size=initial_train_size,
                horizons=requested_horizons,
                model_name="sarimax_group_d",
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                n_sims=n_sims,
                seed=seed + 20_000,
                training_data="series_exog",
                target_column=TARGET_COLUMN,
                horizon_cap=1,
                max_origins=max_origins,
                skip_origins=skip_origins,
            )
        )

    if "ensemble" in requested_families:
        if verbose:
            print("Running ensemble interval coverage backtest...", flush=True)
        frames.append(
            walk_forward_interval_coverage_backtest(
                series=series,
                exog=elastic_net_exog,
                simulate_func=lambda train_frame, steps, n_sims, seed: simulate_ensemble_paths(
                    train_frame,
                    steps=steps,
                    n_sims=n_sims,
                    weights=weights,
                    seed=seed,
                ),
                initial_train_size=initial_train_size,
                horizons=requested_horizons,
                model_name="ensemble",
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
                n_sims=n_sims,
                seed=seed + 30_000,
                training_data="frame",
                target_column=TARGET_COLUMN,
                max_origins=max_origins,
                skip_origins=skip_origins,
            )
        )

    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


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
    families: tuple[str, ...] | None = None,
    apply_calibration: bool = True,
    calibration_factors_path: Path = INTERVAL_CALIBRATION_FACTORS_PATH,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run interval coverage backtests for served model families and save a report.

    This report intentionally keeps each family's achievable origin grid instead
    of intersecting all families onto a common grid. It judges each model's own
    interval calibration; cross-model sample sizes can therefore differ.
    """
    started = time.perf_counter()
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
        families=families,
        verbose=verbose,
    )

    if predictions.empty:
        raise ValueError("Interval coverage backtests produced no forecast origins.")
    if apply_calibration:
        factors = load_interval_calibration_factors(calibration_factors_path)
        if not factors.empty:
            predictions = apply_interval_calibration(predictions, factors)
        elif verbose:
            print(
                f"Calibration factors not found at {calibration_factors_path}; "
                "reporting raw interval coverage.",
                flush=True,
            )

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
        elapsed = time.perf_counter() - started
        print(f"Saved interval coverage report to {output_path}", flush=True)
        print(f"Runtime seconds: {elapsed:.1f}", flush=True)
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
    parser.add_argument(
        "--calibration-factors",
        type=Path,
        default=INTERVAL_CALIBRATION_FACTORS_PATH,
    )
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
        calibration_factors_path=args.calibration_factors,
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
