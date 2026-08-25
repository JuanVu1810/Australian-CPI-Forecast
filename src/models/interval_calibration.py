"""Split-conformal-style interval scale calibration for served CPI families."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from src.models.evaluation import (
    CURATED_DATA_PATH,
    DEFAULT_HORIZONS,
    PROJECT_ROOT,
    compute_conformal_scale_factors,
    compute_interval_coverage_table,
)
from src.models.interval_coverage import (
    DEFAULT_INITIAL_TRAIN_SIZE,
    DEFAULT_LOWER_QUANTILE,
    DEFAULT_N_SIMS,
    DEFAULT_SEED,
    DEFAULT_UPPER_QUANTILE,
    INTERVAL_CALIBRATION_FACTORS_PATH,
    INTERVAL_COVERAGE_OUTPUT_PATH,
    TARGET_COLUMN,
    TRIMMED_MEAN_INTERVAL_COVERAGE_OUTPUT_PATH,
    apply_interval_calibration,
    run_family_interval_backtests,
)
from src.models.elastic_net import (
    TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
    TRIMMED_MEAN_TARGET_COLUMN,
)
from src.models.sarima import TRIMMED_MEAN_DEFAULT_ORDER, TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER


INTERVAL_CALIBRATION_VALIDATION_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/model_interval_calibration_validation.csv"
)
TRIMMED_MEAN_INTERVAL_CALIBRATION_FACTORS_PATH = (
    PROJECT_ROOT / "reports/model_interval_calibration_factors_trimmed_mean.csv"
)
TRIMMED_MEAN_INTERVAL_CALIBRATION_VALIDATION_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/model_interval_calibration_validation_trimmed_mean.csv"
)
DEFAULT_CALIBRATION_FRACTION = 0.7
DEFAULT_FAMILIES = ("sarima", "elastic_net", "ensemble")


def _origin_counts_from_coverage_report(
    path: Path = INTERVAL_COVERAGE_OUTPUT_PATH,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
) -> dict[str, int]:
    """Read per-family origin counts from the existing interval coverage report."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required to choose calibration/validation split sizes. "
            "Run python -m src.models.interval_coverage first."
        )
    report = pd.read_csv(path)
    required = {"model", "horizon", "n"}
    missing = required.difference(report.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    counts: dict[str, int] = {}
    rows = report.loc[~report["horizon"].astype(str).eq("overall")].copy()
    for family in families:
        family_rows = rows.loc[rows["model"].eq(family)]
        if family_rows.empty:
            raise ValueError(f"{path} has no horizon rows for model {family!r}.")
        counts[family] = int(family_rows["n"].astype(int).max())
    return counts


def _calibration_origin_count(total_origins: int, calibration_fraction: float) -> int:
    if total_origins < 2:
        raise ValueError("at least two origins are required for calibration and validation.")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must satisfy 0 < fraction < 1.")
    return max(1, min(total_origins - 1, int(total_origins * calibration_fraction)))


def _calibrated_validation_predictions(
    validation_predictions: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    calibrated = apply_interval_calibration(validation_predictions, factors)
    if "interval_calibrated" not in calibrated:
        calibrated["interval_calibrated"] = False
    return calibrated


def _held_out_comparison(
    raw_validation: pd.DataFrame,
    calibrated_validation: pd.DataFrame,
    factors: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
) -> pd.DataFrame:
    raw = compute_interval_coverage_table(
        raw_validation,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    calibrated = compute_interval_coverage_table(
        calibrated_validation,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    raw = raw.rename(
        columns={
            "n": "validation_n",
            "nominal_coverage": "target_coverage",
            "empirical_coverage": "raw_empirical_coverage",
            "mean_interval_width": "raw_mean_interval_width",
            "coverage_ci_lower": "raw_coverage_ci_lower",
            "coverage_ci_upper": "raw_coverage_ci_upper",
            "binom_p_value": "raw_binom_p_value",
            "significantly_miscalibrated": "raw_significantly_miscalibrated",
        }
    )
    calibrated = calibrated.rename(
        columns={
            "n": "calibrated_validation_n",
            "empirical_coverage": "calibrated_empirical_coverage",
            "mean_interval_width": "calibrated_mean_interval_width",
            "coverage_ci_lower": "calibrated_coverage_ci_lower",
            "coverage_ci_upper": "calibrated_coverage_ci_upper",
            "binom_p_value": "calibrated_binom_p_value",
            "significantly_miscalibrated": "calibrated_significantly_miscalibrated",
        }
    )
    merged = raw.merge(
        calibrated[
            [
                "model",
                "horizon",
                "calibrated_validation_n",
                "calibrated_empirical_coverage",
                "calibrated_coverage_ci_lower",
                "calibrated_coverage_ci_upper",
                "calibrated_binom_p_value",
                "calibrated_significantly_miscalibrated",
                "calibrated_mean_interval_width",
            ]
        ],
        on=["model", "horizon"],
        how="inner",
        validate="one_to_one",
    )
    horizon_factors = factors.rename(
        columns={"n": "calibration_n"}
    ).loc[:, ["model", "horizon", "calibration_n", "scale_factor"]]
    merged = merged.merge(
        horizon_factors,
        on=["model", "horizon"],
        how="left",
        validate="many_to_one",
    )
    return merged[
        [
            "model",
            "horizon",
            "calibration_n",
            "validation_n",
            "target_coverage",
            "scale_factor",
            "raw_empirical_coverage",
            "calibrated_empirical_coverage",
            "raw_mean_interval_width",
            "calibrated_mean_interval_width",
            "raw_binom_p_value",
            "calibrated_binom_p_value",
            "raw_significantly_miscalibrated",
            "calibrated_significantly_miscalibrated",
        ]
    ]


def run_interval_calibration(
    curated_path: Path = CURATED_DATA_PATH,
    factors_output_path: Path = INTERVAL_CALIBRATION_FACTORS_PATH,
    validation_output_path: Path = INTERVAL_CALIBRATION_VALIDATION_OUTPUT_PATH,
    origin_count_report_path: Path = INTERVAL_COVERAGE_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float = DEFAULT_UPPER_QUANTILE,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    target_column: str = TARGET_COLUMN,
    sarima_order: tuple[int, int, int] | None = None,
    sarima_seasonal_order: tuple[int, int, int, int] | None = None,
    elastic_net_feature_columns: tuple[str, ...] | None = None,
    weights: tuple[float, float] | dict[int, tuple[float, float]] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit interval scale factors on early origins and validate on later origins."""
    target_coverage = float(upper_quantile - lower_quantile)
    origin_counts = _origin_counts_from_coverage_report(
        origin_count_report_path,
        families=families,
    )
    factor_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []
    started = time.perf_counter()

    for family in families:
        total_origins = origin_counts[family]
        calibration_origins = _calibration_origin_count(total_origins, calibration_fraction)
        validation_origins = total_origins - calibration_origins
        if verbose:
            print(
                f"Running {family} calibration split: "
                f"{calibration_origins} calibration origins, "
                f"{validation_origins} held-out origins...",
                flush=True,
            )

        calibration_predictions = run_family_interval_backtests(
            curated_path=curated_path,
            initial_train_size=initial_train_size,
            horizons=horizons,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            n_sims=n_sims,
            seed=seed,
            max_origins=calibration_origins,
            skip_origins=0,
            families=(family,),
            target_column=target_column,
            **({"sarima_order": sarima_order} if sarima_order is not None else {}),
            **(
                {"sarima_seasonal_order": sarima_seasonal_order}
                if sarima_seasonal_order is not None
                else {}
            ),
            **(
                {"elastic_net_feature_columns": elastic_net_feature_columns}
                if elastic_net_feature_columns is not None
                else {}
            ),
            weights=weights,
            verbose=verbose,
        )
        factors = compute_conformal_scale_factors(
            calibration_predictions,
            target_coverage=target_coverage,
        )
        factor_frames.append(factors)

        validation_predictions = run_family_interval_backtests(
            curated_path=curated_path,
            initial_train_size=initial_train_size,
            horizons=horizons,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            n_sims=n_sims,
            seed=seed,
            max_origins=None,
            skip_origins=calibration_origins,
            families=(family,),
            target_column=target_column,
            **({"sarima_order": sarima_order} if sarima_order is not None else {}),
            **(
                {"sarima_seasonal_order": sarima_seasonal_order}
                if sarima_seasonal_order is not None
                else {}
            ),
            **(
                {"elastic_net_feature_columns": elastic_net_feature_columns}
                if elastic_net_feature_columns is not None
                else {}
            ),
            weights=weights,
            verbose=verbose,
        )
        calibrated_validation = _calibrated_validation_predictions(
            validation_predictions,
            factors,
        )
        comparison_frames.append(
            _held_out_comparison(
                raw_validation=validation_predictions,
                calibrated_validation=calibrated_validation,
                factors=factors,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
        )

    all_factors = pd.concat(factor_frames, ignore_index=True, sort=False)
    all_factors = all_factors.rename(columns={"n": "calibration_n"})
    all_factors = all_factors[
        ["model", "horizon", "scale_factor", "calibration_n", "target_coverage"]
    ].sort_values(["model", "horizon"])

    validation_comparison = pd.concat(comparison_frames, ignore_index=True, sort=False)
    validation_comparison["_horizon_order"] = validation_comparison["horizon"].map(
        lambda horizon: 0 if horizon == "overall" else int(horizon)
    )
    validation_comparison = validation_comparison.sort_values(
        ["model", "_horizon_order"]
    ).drop(columns="_horizon_order")

    factors_output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_output_path.parent.mkdir(parents=True, exist_ok=True)
    all_factors.round({"scale_factor": 6, "target_coverage": 6}).to_csv(
        factors_output_path,
        index=False,
    )
    validation_comparison.round(
        {
            "target_coverage": 6,
            "scale_factor": 6,
            "raw_empirical_coverage": 6,
            "calibrated_empirical_coverage": 6,
            "raw_mean_interval_width": 6,
            "calibrated_mean_interval_width": 6,
            "raw_binom_p_value": 6,
            "calibrated_binom_p_value": 6,
        }
    ).to_csv(validation_output_path, index=False)

    if verbose:
        elapsed = time.perf_counter() - started
        print(f"Saved calibration factors to {factors_output_path}", flush=True)
        print(f"Saved held-out validation comparison to {validation_output_path}", flush=True)
        print(f"Runtime seconds: {elapsed:.1f}", flush=True)
    return all_factors, validation_comparison


def run_trimmed_mean_interval_calibration(
    curated_path: Path = CURATED_DATA_PATH,
    factors_output_path: Path = TRIMMED_MEAN_INTERVAL_CALIBRATION_FACTORS_PATH,
    validation_output_path: Path = TRIMMED_MEAN_INTERVAL_CALIBRATION_VALIDATION_OUTPUT_PATH,
    origin_count_report_path: Path = TRIMMED_MEAN_INTERVAL_COVERAGE_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float = DEFAULT_UPPER_QUANTILE,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the Phase 3 trimmed-mean interval calibration reports."""
    return run_interval_calibration(
        curated_path=curated_path,
        factors_output_path=factors_output_path,
        validation_output_path=validation_output_path,
        origin_count_report_path=origin_count_report_path,
        initial_train_size=initial_train_size,
        horizons=horizons,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        n_sims=n_sims,
        seed=seed,
        calibration_fraction=calibration_fraction,
        families=("sarima", "elastic_net", "ensemble"),
        target_column=TRIMMED_MEAN_TARGET_COLUMN,
        sarima_order=TRIMMED_MEAN_DEFAULT_ORDER,
        sarima_seasonal_order=TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER,
        elastic_net_feature_columns=TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
        weights=(0.5, 0.5),
        verbose=verbose,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("headline", "trimmed_mean"), default="headline")
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--factors-output", type=Path, default=None)
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--origin-count-report",
        type=Path,
        default=None,
    )
    parser.add_argument("--initial-train-size", type=int, default=DEFAULT_INITIAL_TRAIN_SIZE)
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lower-quantile", type=float, default=DEFAULT_LOWER_QUANTILE)
    parser.add_argument("--upper-quantile", type=float, default=DEFAULT_UPPER_QUANTILE)
    parser.add_argument("--calibration-fraction", type=float, default=DEFAULT_CALIBRATION_FRACTION)
    args = parser.parse_args(argv)

    if args.target == "trimmed_mean":
        factors, comparison = run_trimmed_mean_interval_calibration(
            curated_path=args.data,
            factors_output_path=args.factors_output or TRIMMED_MEAN_INTERVAL_CALIBRATION_FACTORS_PATH,
            validation_output_path=(
                args.validation_output or TRIMMED_MEAN_INTERVAL_CALIBRATION_VALIDATION_OUTPUT_PATH
            ),
            origin_count_report_path=args.origin_count_report or TRIMMED_MEAN_INTERVAL_COVERAGE_OUTPUT_PATH,
            initial_train_size=args.initial_train_size,
            lower_quantile=args.lower_quantile,
            upper_quantile=args.upper_quantile,
            n_sims=args.n_sims,
            seed=args.seed,
            calibration_fraction=args.calibration_fraction,
            verbose=True,
        )
    else:
        factors, comparison = run_interval_calibration(
            curated_path=args.data,
            factors_output_path=args.factors_output or INTERVAL_CALIBRATION_FACTORS_PATH,
            validation_output_path=args.validation_output or INTERVAL_CALIBRATION_VALIDATION_OUTPUT_PATH,
            origin_count_report_path=args.origin_count_report or INTERVAL_COVERAGE_OUTPUT_PATH,
            initial_train_size=args.initial_train_size,
            lower_quantile=args.lower_quantile,
            upper_quantile=args.upper_quantile,
            n_sims=args.n_sims,
            seed=args.seed,
            calibration_fraction=args.calibration_fraction,
            verbose=True,
        )
    print("\nCalibration factors:")
    print(factors.round({"scale_factor": 3}).to_string(index=False))
    print("\nHeld-out before/after interval coverage:")
    print(
        comparison.round(
            {
                "target_coverage": 3,
                "scale_factor": 3,
                "raw_empirical_coverage": 3,
                "calibrated_empirical_coverage": 3,
                "raw_mean_interval_width": 3,
                "calibrated_mean_interval_width": 3,
                "raw_binom_p_value": 3,
                "calibrated_binom_p_value": 3,
            }
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
