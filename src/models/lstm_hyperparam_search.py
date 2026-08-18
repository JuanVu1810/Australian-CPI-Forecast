"""Bounded coordinate-descent hyperparameter search for the compact LSTM.

The fixed LSTM (``src/models/lstm.py``) uses 16 units, dropout 0.2, and an
8-quarter lookback by design choice, not a validated optimum. This module
runs a small, bounded search -- not an exhaustive grid, to keep runtime and
overfitting risk in check on a ~53-origin walk-forward sample -- and reports
the result honestly, including if no configuration beats the fixed baseline
or SARIMA.

Search protocol:
1. Screen candidate values one hyperparameter at a time (units, then dropout,
   then lookback), each screened on a reduced-origin subset for speed.
2. Re-validate the best-found configuration against the fixed baseline on the
   full origin set, since a screening-stage win on a small subset is not by
   itself a reportable result.
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
    load_target_series,
    walk_forward_backtest_direct_multihorizon,
)
from src.models.lstm import (
    DEFAULT_INITIAL_TRAIN_SIZE,
    DEFAULT_SEED,
    DROPOUT,
    LOOKBACK_QUARTERS,
    LSTM_UNITS,
    forecast_lstm_direct,
    load_lstm_feature_frame,
)


SEARCH_OUTPUT_PATH = PROJECT_ROOT / "reports/lstm_hyperparameter_search.csv"
SCREEN_MAX_ORIGINS = 20
UNITS_CANDIDATES = (8, 16, 32)
DROPOUT_CANDIDATES = (0.1, 0.2, 0.3)
LOOKBACK_CANDIDATES = (4, 8, 12)


@dataclass(frozen=True)
class LSTMConfig:
    units: int = LSTM_UNITS
    dropout: float = DROPOUT
    lookback: int = LOOKBACK_QUARTERS


def evaluate_configuration(
    series: pd.Series,
    exog: pd.DataFrame,
    config: LSTMConfig,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    forecast_func=None,
) -> tuple[float, int]:
    """Return (overall RMSE, origin count) for one LSTM configuration."""
    build_forecast_func = forecast_func or (
        lambda train_frame, steps: forecast_lstm_direct(
            train_frame,
            steps=steps,
            lookback=config.lookback,
            units=config.units,
            dropout=config.dropout,
            seed=seed,
        )
    )
    predictions = walk_forward_backtest_direct_multihorizon(
        series=series,
        exog=exog,
        forecast_func=build_forecast_func,
        initial_train_size=initial_train_size,
        horizons=horizons,
        model_name="lstm",
        max_origins=max_origins,
    )
    if predictions.empty:
        raise ValueError("configuration produced no forecast origins.")
    rmse = float(np.sqrt(np.mean(np.square(predictions["error"].astype(float)))))
    origin_n = int(predictions["forecast_origin"].nunique())
    return rmse, origin_n


def bounded_coordinate_search(
    series: pd.Series,
    exog: pd.DataFrame,
    baseline: LSTMConfig = LSTMConfig(),
    units_candidates: tuple[int, ...] = UNITS_CANDIDATES,
    dropout_candidates: tuple[float, ...] = DROPOUT_CANDIDATES,
    lookback_candidates: tuple[int, ...] = LOOKBACK_CANDIDATES,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = SCREEN_MAX_ORIGINS,
    forecast_func=None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, LSTMConfig]:
    """Screen units, then dropout, then lookback, keeping the running best.

    This is a coordinate-descent-style screen (9 fits, not the full 27-point
    factorial grid) chosen to bound runtime on a small quarterly sample.
    """
    trials: list[dict[str, object]] = []
    best = baseline

    def _try(config: LSTMConfig, stage: str) -> float:
        rmse, origin_n = evaluate_configuration(
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
                "origin_n": origin_n,
                "rmse": rmse,
            }
        )
        if verbose:
            print(f"  [{stage}] units={config.units} dropout={config.dropout} "
                  f"lookback={config.lookback} -> rmse={rmse:.4f}", flush=True)
        return rmse

    best_rmse = _try(best, "baseline")

    for units in units_candidates:
        candidate = LSTMConfig(units=units, dropout=best.dropout, lookback=best.lookback)
        if candidate == best:
            continue
        rmse = _try(candidate, "units")
        if rmse < best_rmse:
            best, best_rmse = candidate, rmse

    for dropout in dropout_candidates:
        candidate = LSTMConfig(units=best.units, dropout=dropout, lookback=best.lookback)
        if candidate == best:
            continue
        rmse = _try(candidate, "dropout")
        if rmse < best_rmse:
            best, best_rmse = candidate, rmse

    for lookback in lookback_candidates:
        candidate = LSTMConfig(units=best.units, dropout=best.dropout, lookback=lookback)
        if candidate == best:
            continue
        rmse = _try(candidate, "lookback")
        if rmse < best_rmse:
            best, best_rmse = candidate, rmse

    return pd.DataFrame(trials), best


def run_search(
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = SEARCH_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    seed: int = DEFAULT_SEED,
    screen_max_origins: int = SCREEN_MAX_ORIGINS,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the bounded search, re-validate the winner at full scale, and save a report."""
    series = load_target_series(curated_path)
    exog = load_lstm_feature_frame(curated_path)
    baseline = LSTMConfig()

    if verbose:
        print(f"Screening candidates on {screen_max_origins} origins...", flush=True)
    trials, best = bounded_coordinate_search(
        series=series,
        exog=exog,
        baseline=baseline,
        initial_train_size=initial_train_size,
        horizons=horizons,
        seed=seed,
        max_origins=screen_max_origins,
        verbose=verbose,
    )
    trials.insert(0, "phase", "screen")

    if verbose:
        print("Re-validating baseline and best-screened config at full scale...", flush=True)
    full_rows: list[dict[str, object]] = []
    for label, config in (("baseline", baseline), ("best_screened", best)):
        rmse, origin_n = evaluate_configuration(
            series=series,
            exog=exog,
            config=config,
            initial_train_size=initial_train_size,
            horizons=horizons,
            seed=seed,
            max_origins=None,
        )
        full_rows.append(
            {
                "phase": "full_validation",
                "stage": label,
                "units": config.units,
                "dropout": config.dropout,
                "lookback": config.lookback,
                "origin_n": origin_n,
                "rmse": rmse,
            }
        )
        if verbose:
            print(f"  [full_validation:{label}] units={config.units} "
                  f"dropout={config.dropout} lookback={config.lookback} "
                  f"-> rmse={rmse:.4f} (n={origin_n})", flush=True)

    report = pd.concat([trials, pd.DataFrame(full_rows)], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.round({"rmse": 6}).to_csv(output_path, index=False)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--output", type=Path, default=SEARCH_OUTPUT_PATH)
    parser.add_argument("--initial-train-size", type=int, default=DEFAULT_INITIAL_TRAIN_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--screen-max-origins", type=int, default=SCREEN_MAX_ORIGINS)
    args = parser.parse_args(argv)

    report = run_search(
        curated_path=args.data,
        output_path=args.output,
        initial_train_size=args.initial_train_size,
        seed=args.seed,
        screen_max_origins=args.screen_max_origins,
        verbose=True,
    )
    full = report.loc[report["phase"] == "full_validation"]
    baseline_rmse = float(full.loc[full["stage"] == "baseline", "rmse"].iloc[0])
    best_rmse = float(full.loc[full["stage"] == "best_screened", "rmse"].iloc[0])
    print("\nFull-scale result:")
    print(full.to_string(index=False))
    if best_rmse < baseline_rmse:
        print(
            f"\nBest screened config improves full-scale RMSE: "
            f"{baseline_rmse:.4f} -> {best_rmse:.4f}."
        )
    else:
        print(
            f"\nNo screened config beats the fixed baseline at full scale "
            f"(baseline {baseline_rmse:.4f} vs best screened {best_rmse:.4f}); "
            "keeping the fixed compact LSTM as documented."
        )


if __name__ == "__main__":
    main()
