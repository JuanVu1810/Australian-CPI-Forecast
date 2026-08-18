"""Bounded, leakage-aware hyperparameter search for the compact LSTM.

The fixed LSTM (``src/models/lstm.py``) uses 16 units, dropout 0.2, an
8-quarter lookback, and Adam's default 1e-3 learning rate by design choice,
not a validated optimum. This module runs a staged, bounded search -- not an
exhaustive or Bayesian search over the full hyperparameter space, since this
project has only ~53 real walk-forward origins and an earlier, smaller
version of this search already demonstrated the failure mode a larger one
would only amplify: a config that won on a 20-origin screen (1.0462 vs
1.0949) lost to the fixed baseline at full 53-origin scale (1.8612 vs
1.8182). Runtime is not the constraint here; the sample size is.

Search protocol (see README/PROJECT_BRIEF for the fuller writeup):
1. Screen units, then dropout, then lookback, then learning rate --
   one coordinate at a time, each on a reduced-origin subset, at a single
   seed, for speed.
2. Re-validate the best-screened config against the fixed baseline on a
   chronologically DISJOINT set of later origins (never the screening
   origins), across multiple seeds each, reporting mean/std RMSE and MAE
   and a per-horizon breakdown pooled across seeds. A screening-stage win
   is not itself a reportable result -- only this step's outcome is.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.evaluation import (
    CURATED_DATA_PATH,
    DEFAULT_HORIZONS,
    PROJECT_ROOT,
    compute_metric_table,
    load_target_series,
    walk_forward_backtest_direct_multihorizon,
)
from src.models.lstm import (
    DEFAULT_INITIAL_TRAIN_SIZE,
    DEFAULT_SEED,
    DROPOUT,
    LEARNING_RATE,
    LOOKBACK_QUARTERS,
    LSTM_UNITS,
    forecast_lstm_direct,
    load_lstm_feature_frame,
)


SEARCH_OUTPUT_PATH = PROJECT_ROOT / "reports/lstm_hyperparameter_search.csv"
SCREEN_MAX_ORIGINS = 20
UNITS_CANDIDATES = (4, 8, 16, 32)
DROPOUT_CANDIDATES = (0.0, 0.1, 0.2, 0.3, 0.4)
LOOKBACK_CANDIDATES = (4, 6, 8, 12)
LEARNING_RATE_CANDIDATES = (1e-4, 3e-4, 1e-3, 3e-3)
FINALIST_SEEDS = (42, 7, 123, 2024, 99)


@dataclass(frozen=True)
class LSTMConfig:
    units: int = LSTM_UNITS
    dropout: float = DROPOUT
    lookback: int = LOOKBACK_QUARTERS
    learning_rate: float = LEARNING_RATE


def run_predictions(
    series: pd.Series,
    exog: pd.DataFrame,
    config: LSTMConfig,
    initial_train_size: int,
    horizons: tuple[int, ...],
    seed: int,
    max_origins: int | None = None,
    skip_origins: int = 0,
    forecast_func=None,
) -> pd.DataFrame:
    build_forecast_func = forecast_func or (
        lambda train_frame, steps: forecast_lstm_direct(
            train_frame,
            steps=steps,
            lookback=config.lookback,
            units=config.units,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            seed=seed,
        )
    )
    return walk_forward_backtest_direct_multihorizon(
        series=series,
        exog=exog,
        forecast_func=build_forecast_func,
        initial_train_size=initial_train_size,
        horizons=horizons,
        model_name="lstm",
        max_origins=max_origins,
        skip_origins=skip_origins,
    )


def evaluate_configuration(
    series: pd.Series,
    exog: pd.DataFrame,
    config: LSTMConfig,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    skip_origins: int = 0,
    forecast_func=None,
) -> tuple[float, float, int]:
    """Return (overall RMSE, overall MAE, origin count) for one configuration."""
    predictions = run_predictions(
        series, exog, config, initial_train_size, horizons, seed,
        max_origins, skip_origins, forecast_func,
    )
    if predictions.empty:
        raise ValueError("configuration produced no forecast origins.")
    errors = predictions["error"].astype(float)
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    mae = float(np.mean(np.abs(errors)))
    origin_n = int(predictions["forecast_origin"].nunique())
    return rmse, mae, origin_n


def bounded_coordinate_search(
    series: pd.Series,
    exog: pd.DataFrame,
    baseline: LSTMConfig = LSTMConfig(),
    units_candidates: tuple[int, ...] = UNITS_CANDIDATES,
    dropout_candidates: tuple[float, ...] = DROPOUT_CANDIDATES,
    lookback_candidates: tuple[int, ...] = LOOKBACK_CANDIDATES,
    learning_rate_candidates: tuple[float, ...] = LEARNING_RATE_CANDIDATES,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = SCREEN_MAX_ORIGINS,
    forecast_func=None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, LSTMConfig]:
    """Screen units, then dropout, then lookback, then learning rate.

    Coordinate-descent-style screen (not the full cross-product grid),
    chosen to bound runtime and, more importantly, to bound how many
    comparisons are drawn from a ~20-origin sample before it stops being
    statistically meaningful.
    """
    trials: list[dict[str, object]] = []
    best = baseline

    def _try(config: LSTMConfig, stage: str) -> float:
        rmse, mae, origin_n = evaluate_configuration(
            series=series,
            exog=exog,
            config=config,
            initial_train_size=initial_train_size,
            horizons=horizons,
            seed=seed,
            max_origins=max_origins,
            forecast_func=forecast_func,
        )
        trials.append(
            {
                "stage": stage,
                "units": config.units,
                "dropout": config.dropout,
                "lookback": config.lookback,
                "learning_rate": config.learning_rate,
                "origin_n": origin_n,
                "rmse": rmse,
                "mae": mae,
            }
        )
        if verbose:
            print(
                f"  [{stage}] units={config.units} dropout={config.dropout} "
                f"lookback={config.lookback} lr={config.learning_rate} -> rmse={rmse:.4f}",
                flush=True,
            )
        return rmse

    best_rmse = _try(best, "baseline")

    stages = (
        ("units", units_candidates, lambda c, v: LSTMConfig(v, c.dropout, c.lookback, c.learning_rate)),
        ("dropout", dropout_candidates, lambda c, v: LSTMConfig(c.units, v, c.lookback, c.learning_rate)),
        ("lookback", lookback_candidates, lambda c, v: LSTMConfig(c.units, c.dropout, v, c.learning_rate)),
        ("learning_rate", learning_rate_candidates, lambda c, v: LSTMConfig(c.units, c.dropout, c.lookback, v)),
    )
    for stage_name, candidates, make_config in stages:
        for value in candidates:
            candidate = make_config(best, value)
            if candidate == best:
                continue
            rmse = _try(candidate, stage_name)
            if rmse < best_rmse:
                best, best_rmse = candidate, rmse

    return pd.DataFrame(trials), best


def evaluate_across_seeds(
    series: pd.Series,
    exog: pd.DataFrame,
    config: LSTMConfig,
    seeds: tuple[int, ...],
    initial_train_size: int,
    horizons: tuple[int, ...],
    skip_origins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one config across several seeds on held-out origins.

    Returns (per_seed_summary, pooled_predictions). Pooling predictions
    across seeds before computing per-horizon metrics reflects both
    origin-to-origin and seed-to-seed variability in one table.
    """
    per_seed_rows: list[dict[str, object]] = []
    pooled_frames: list[pd.DataFrame] = []
    for seed in seeds:
        predictions = run_predictions(
            series, exog, config, initial_train_size, horizons, seed,
            max_origins=None, skip_origins=skip_origins,
        )
        errors = predictions["error"].astype(float)
        per_seed_rows.append(
            {
                "seed": seed,
                "origin_n": int(predictions["forecast_origin"].nunique()),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "mae": float(np.mean(np.abs(errors))),
            }
        )
        predictions = predictions.copy()
        predictions["model"] = "lstm"
        predictions["seed"] = seed
        pooled_frames.append(predictions)
    return pd.DataFrame(per_seed_rows), pd.concat(pooled_frames, ignore_index=True)


def run_search(
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = SEARCH_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    screen_seed: int = DEFAULT_SEED,
    screen_max_origins: int = SCREEN_MAX_ORIGINS,
    finalist_seeds: tuple[int, ...] = FINALIST_SEEDS,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the bounded search, re-validate on disjoint held-out origins across
    multiple seeds, and save a report distinguishing screen vs final-test."""
    series = load_target_series(curated_path)
    exog = load_lstm_feature_frame(curated_path)
    baseline = LSTMConfig()

    if verbose:
        print(f"Screening on the first {screen_max_origins} origins (seed={screen_seed})...", flush=True)
    trials, best = bounded_coordinate_search(
        series=series,
        exog=exog,
        baseline=baseline,
        initial_train_size=initial_train_size,
        horizons=horizons,
        seed=screen_seed,
        max_origins=screen_max_origins,
        verbose=verbose,
    )
    trials.insert(0, "phase", "screen")

    if verbose:
        print(
            f"Re-validating baseline and best-screened config on origins after "
            f"the first {screen_max_origins} (held out from screening), "
            f"across {len(finalist_seeds)} seeds each...",
            flush=True,
        )
    finalist_summary_rows: list[dict[str, object]] = []
    horizon_tables: list[pd.DataFrame] = []
    for label, config in (("baseline", baseline), ("best_screened", best)):
        per_seed, pooled = evaluate_across_seeds(
            series=series,
            exog=exog,
            config=config,
            seeds=finalist_seeds,
            initial_train_size=initial_train_size,
            horizons=horizons,
            skip_origins=screen_max_origins,
        )
        finalist_summary_rows.append(
            {
                "phase": "final_test",
                "stage": label,
                "units": config.units,
                "dropout": config.dropout,
                "lookback": config.lookback,
                "learning_rate": config.learning_rate,
                "origin_n": int(per_seed["origin_n"].iloc[0]),
                "n_seeds": len(finalist_seeds),
                "rmse_mean": float(per_seed["rmse"].mean()),
                "rmse_std": float(per_seed["rmse"].std(ddof=0)),
                "mae_mean": float(per_seed["mae"].mean()),
                "mae_std": float(per_seed["mae"].std(ddof=0)),
            }
        )
        by_horizon = compute_metric_table(pooled)
        by_horizon.insert(0, "config", label)
        horizon_tables.append(by_horizon)
        if verbose:
            print(
                f"  [final_test:{label}] rmse_mean={per_seed['rmse'].mean():.4f} "
                f"rmse_std={per_seed['rmse'].std(ddof=0):.4f} "
                f"(n_origins={int(per_seed['origin_n'].iloc[0])}, seeds={len(finalist_seeds)})",
                flush=True,
            )

    final_summary = pd.DataFrame(finalist_summary_rows)
    by_horizon = pd.concat(horizon_tables, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trials.round({"rmse": 6, "mae": 6}).to_csv(output_path, index=False)
    horizon_output_path = output_path.with_name(output_path.stem + "_by_horizon.csv")
    by_horizon.round({"rmse": 6, "mae": 6}).to_csv(horizon_output_path, index=False)
    summary_output_path = output_path.with_name(output_path.stem + "_final_summary.csv")
    final_summary.round(
        {"rmse_mean": 6, "rmse_std": 6, "mae_mean": 6, "mae_std": 6}
    ).to_csv(summary_output_path, index=False)

    return {"screen_trials": trials, "final_summary": final_summary, "by_horizon": by_horizon}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--output", type=Path, default=SEARCH_OUTPUT_PATH)
    parser.add_argument("--initial-train-size", type=int, default=DEFAULT_INITIAL_TRAIN_SIZE)
    parser.add_argument("--screen-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--screen-max-origins", type=int, default=SCREEN_MAX_ORIGINS)
    args = parser.parse_args(argv)

    results = run_search(
        curated_path=args.data,
        output_path=args.output,
        initial_train_size=args.initial_train_size,
        screen_seed=args.screen_seed,
        screen_max_origins=args.screen_max_origins,
        verbose=True,
    )
    final_summary = results["final_summary"]
    baseline_row = final_summary.loc[final_summary["stage"] == "baseline"].iloc[0]
    best_row = final_summary.loc[final_summary["stage"] == "best_screened"].iloc[0]

    print("\nFinal held-out test (disjoint from screening origins, multi-seed):")
    print(final_summary.to_string(index=False))
    print("\nBy horizon:")
    print(results["by_horizon"].to_string(index=False))

    if best_row["rmse_mean"] < baseline_row["rmse_mean"]:
        print(
            f"\nBest screened config improves held-out mean RMSE: "
            f"{baseline_row['rmse_mean']:.4f} -> {best_row['rmse_mean']:.4f} "
            f"(baseline std {baseline_row['rmse_std']:.4f}, best std {best_row['rmse_std']:.4f})."
        )
    else:
        print(
            f"\nNo screened config beats the fixed baseline on the held-out "
            f"multi-seed test (baseline {baseline_row['rmse_mean']:.4f} vs "
            f"best screened {best_row['rmse_mean']:.4f}); keeping the fixed "
            "compact LSTM as documented."
        )


if __name__ == "__main__":
    main()
