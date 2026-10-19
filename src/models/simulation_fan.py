"""Forward simulated-path percentile fan-chart reports for CPI model families."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import ensemble, svar
from src.models.elastic_net import (
    ELASTIC_NET_FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_elastic_net_feature_frame,
    simulate_elastic_net_paths,
)
from src.models.evaluation import (
    CURATED_DATA_PATH,
    DEFAULT_HORIZONS,
    PROJECT_ROOT,
    load_target_series,
)
from src.models.interval_coverage import DEFAULT_N_SIMS, DEFAULT_SEED
from src.models.sarima import (
    DEFAULT_ORDER as SARIMA_DEFAULT_ORDER,
    DEFAULT_SEASONAL_ORDER as SARIMA_DEFAULT_SEASONAL_ORDER,
    simulate_sarima_paths,
)


SIMULATION_FAN_SARIMA_OUTPUT_PATH = PROJECT_ROOT / "reports/simulation_fan_sarima.csv"
SIMULATION_FAN_ELASTIC_NET_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/simulation_fan_elastic_net.csv"
)
SIMULATION_FAN_ENSEMBLE_OUTPUT_PATH = PROJECT_ROOT / "reports/simulation_fan_ensemble.csv"
SIMULATION_FAN_SVAR_HEADLINE_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/simulation_fan_svar_headline.csv"
)
SIMULATION_PATHS_SAMPLE_ENSEMBLE_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/simulation_paths_sample_ensemble.csv"
)
FAN_PERCENTILES = (10, 25, 50, 75, 90)
FAN_OUTPUT_COLUMNS = (
    "model_family",
    "target_column",
    "forecast_origin",
    "horizon",
    "target_quarter",
    "p10",
    "p25",
    "median",
    "p75",
    "p90",
)


def _future_quarter(forecast_origin: str, horizon: int) -> str:
    """Quarter label ``horizon`` quarters after the origin, e.g. ("2025Q4", 2) -> "2026Q2"."""
    return str(pd.Period(str(forecast_origin), freq="Q") + int(horizon))


def fan_frame_from_paths(
    paths: np.ndarray,
    *,
    model_family: str,
    target_column: str,
    forecast_origin: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Summarise simulated paths into 10/25/50/75/90 percentile bands."""
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    values = np.asarray(paths, dtype=float)
    if values.ndim != 2:
        raise ValueError("fan chart paths must have shape (n_sims, steps).")
    if not requested_horizons:
        raise ValueError("horizons must contain at least one horizon.")
    if min(requested_horizons) < 1 or max(requested_horizons) > values.shape[1]:
        raise ValueError(
            "fan chart paths do not cover requested horizons "
            f"{requested_horizons}; path shape is {values.shape}."
        )

    selected = values[:, [horizon - 1 for horizon in requested_horizons]]
    percentile_values = np.percentile(selected, FAN_PERCENTILES, axis=0)
    rows = []
    for position, horizon in enumerate(requested_horizons):
        rows.append(
            {
                "model_family": model_family,
                "target_column": target_column,
                "forecast_origin": str(forecast_origin),
                "horizon": int(horizon),
                "target_quarter": _future_quarter(forecast_origin, horizon),
                "p10": float(percentile_values[0, position]),
                "p25": float(percentile_values[1, position]),
                "median": float(percentile_values[2, position]),
                "p75": float(percentile_values[3, position]),
                "p90": float(percentile_values[4, position]),
            }
        )
    return pd.DataFrame(rows, columns=list(FAN_OUTPUT_COLUMNS))


def _write_fan(frame: pd.DataFrame, output_path: Path, *, verbose: bool) -> pd.DataFrame:
    """Save the table as CSV and return it unchanged."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved {output_path} ({len(frame)} rows).", flush=True)
    return frame


def _elastic_net_training_frame(curated_path: Path) -> pd.DataFrame:
    """Target plus Elastic Net features up to the pinned forecast origin, complete rows only."""
    target = load_target_series(
        curated_path,
        target_column=TARGET_COLUMN,
        max_quarter=svar.FORECAST_ORIGIN_PIN,
    )
    features = load_elastic_net_feature_frame(
        curated_path,
        feature_columns=ELASTIC_NET_FEATURE_COLUMNS,
        max_quarter=svar.FORECAST_ORIGIN_PIN,
    )
    frame = pd.concat([target.rename(TARGET_COLUMN), features], axis=1).dropna()
    if frame.empty:
        raise ValueError(f"No complete Elastic Net rows for target {TARGET_COLUMN!r}.")
    return frame.sort_index()


def _load_svar_frame(curated_path: Path) -> pd.DataFrame:
    """System A SVAR level columns up to the pinned forecast origin, complete rows only."""
    columns = svar.SYSTEM_A_COLUMNS
    required_columns = ["quarter", *columns]
    data = pd.read_csv(curated_path, usecols=required_columns)
    data.index = pd.PeriodIndex(data.pop("quarter").astype(str), freq="Q")
    frame = data.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame.loc[frame.index <= svar.FORECAST_ORIGIN_PIN]
    if frame.empty:
        raise ValueError(f"No complete SVAR rows for columns {tuple(columns)}.")
    return frame.sort_index()


def _simulate_pinned_ensemble_paths(
    curated_path: Path,
    *,
    horizons: Sequence[int],
    n_sims: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Simulate ensemble paths from the pinned origin; return the paths and the origin quarter."""
    series = load_target_series(
        curated_path,
        target_column=TARGET_COLUMN,
        max_quarter=svar.FORECAST_ORIGIN_PIN,
    )
    frame = _elastic_net_training_frame(curated_path)
    if series.index[-1] != frame.index[-1]:
        raise ValueError(
            "Ensemble components end on different quarters: "
            f"SARIMA={series.index[-1]}, Elastic Net={frame.index[-1]}."
        )
    paths = ensemble.simulate_ensemble_paths(
        frame,
        steps=max(horizons),
        n_sims=n_sims,
        seed=seed,
        sarima_series=series,
    )
    return paths, str(frame.index[-1])


def run_sarima_fan(
    *,
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = SIMULATION_FAN_SARIMA_OUTPUT_PATH,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Save the SARIMA percentile fan from the pinned forecast origin."""
    series = load_target_series(
        curated_path,
        target_column=TARGET_COLUMN,
        max_quarter=svar.FORECAST_ORIGIN_PIN,
    )
    paths = simulate_sarima_paths(
        series,
        steps=max(horizons),
        n_sims=n_sims,
        order=SARIMA_DEFAULT_ORDER,
        seasonal_order=SARIMA_DEFAULT_SEASONAL_ORDER,
        seed=seed,
    )
    frame = fan_frame_from_paths(
        paths,
        model_family="sarima",
        target_column=TARGET_COLUMN,
        forecast_origin=str(series.index[-1]),
        horizons=horizons,
    )
    return _write_fan(frame, output_path, verbose=verbose)


def run_elastic_net_fan(
    *,
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = SIMULATION_FAN_ELASTIC_NET_OUTPUT_PATH,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Save the Elastic Net percentile fan from the pinned forecast origin."""
    frame = _elastic_net_training_frame(curated_path)
    paths = simulate_elastic_net_paths(
        frame,
        steps=max(horizons),
        n_sims=n_sims,
        seed=seed,
        feature_columns=ELASTIC_NET_FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
    )
    fan = fan_frame_from_paths(
        paths,
        model_family="elastic_net",
        target_column=TARGET_COLUMN,
        forecast_origin=str(frame.index[-1]),
        horizons=horizons,
    )
    return _write_fan(fan, output_path, verbose=verbose)


def run_ensemble_fan(
    *,
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = SIMULATION_FAN_ENSEMBLE_OUTPUT_PATH,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Save the ensemble percentile fan from the pinned forecast origin."""
    paths, forecast_origin = _simulate_pinned_ensemble_paths(
        curated_path,
        horizons=horizons,
        n_sims=n_sims,
        seed=seed,
    )
    fan = fan_frame_from_paths(
        paths,
        model_family="ensemble",
        target_column=TARGET_COLUMN,
        forecast_origin=forecast_origin,
        horizons=horizons,
    )
    return _write_fan(fan, output_path, verbose=verbose)


def sample_paths_frame(
    paths: np.ndarray,
    *,
    model_family: str,
    target_column: str,
    forecast_origin: str,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_draws: int = 120,
    sample_seed: int = 7,
) -> pd.DataFrame:
    """Pick ``n_draws`` random simulated paths and return them in long format, one row per draw and horizon."""
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    values = np.asarray(paths, dtype=float)
    if values.ndim != 2:
        raise ValueError("paths must have shape (n_sims, steps).")
    if min(requested_horizons) < 1 or max(requested_horizons) > values.shape[1]:
        raise ValueError(
            f"sample paths do not cover requested horizons {requested_horizons}; "
            f"path shape is {values.shape}."
        )
    n_draws = min(n_draws, values.shape[0])
    rng = np.random.default_rng(sample_seed)
    draw_indices = rng.choice(values.shape[0], size=n_draws, replace=False)
    sample = values[np.sort(draw_indices)][:, [horizon - 1 for horizon in requested_horizons]]

    rows = []
    for draw_id, draw_values in enumerate(sample):
        for position, horizon in enumerate(requested_horizons):
            rows.append(
                {
                    "model_family": model_family,
                    "target_column": target_column,
                    "forecast_origin": str(forecast_origin),
                    "draw_id": draw_id,
                    "horizon": int(horizon),
                    "target_quarter": _future_quarter(forecast_origin, horizon),
                    "value": float(draw_values[position]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "model_family",
            "target_column",
            "forecast_origin",
            "draw_id",
            "horizon",
            "target_quarter",
            "value",
        ],
    )


def run_ensemble_path_sample(
    *,
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = SIMULATION_PATHS_SAMPLE_ENSEMBLE_OUTPUT_PATH,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_sims: int = DEFAULT_N_SIMS,
    n_draws: int = 120,
    seed: int = DEFAULT_SEED,
    sample_seed: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    """Save a sample of the same ensemble draws ``run_ensemble_fan`` summarises, as individual paths."""
    paths, forecast_origin = _simulate_pinned_ensemble_paths(
        curated_path,
        horizons=horizons,
        n_sims=n_sims,
        seed=seed,
    )
    sample = sample_paths_frame(
        paths,
        model_family="ensemble",
        target_column=TARGET_COLUMN,
        forecast_origin=forecast_origin,
        horizons=horizons,
        n_draws=n_draws,
        sample_seed=sample_seed,
    )
    return _write_fan(sample, output_path, verbose=verbose)


def run_svar_fan(
    *,
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = SIMULATION_FAN_SVAR_HEADLINE_OUTPUT_PATH,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_sims: int = svar.DEFAULT_BOOTSTRAP_REPLICATIONS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Save the System A SVAR percentile fan for headline CPI from the pinned forecast origin."""
    frame = _load_svar_frame(curated_path)
    ordering = svar.SYSTEM_A_CHOLESKY_ORDER
    paths = svar.simulate_svar_paths(
        frame,
        steps=max(horizons),
        n_sims=n_sims,
        lag_order=svar.DEFAULT_SVAR_LAG_ORDER,
        ordering=ordering,
        seed=seed,
    )
    # paths has shape (n_sims, steps, variables) in ``ordering``; keep only the target variable.
    target_position = list(ordering).index(TARGET_COLUMN)
    target_paths = np.asarray(paths, dtype=float)[:, :, target_position]
    fan = fan_frame_from_paths(
        target_paths,
        model_family="svar_headline",
        target_column=TARGET_COLUMN,
        forecast_origin=str(frame.index[-1]),
        horizons=horizons,
    )
    return _write_fan(fan, output_path, verbose=verbose)


def run_all_simulation_fans(
    *,
    curated_path: Path = CURATED_DATA_PATH,
    n_sims: int = DEFAULT_N_SIMS,
    svar_n_sims: int = svar.DEFAULT_BOOTSTRAP_REPLICATIONS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Regenerate every forward simulation fan-chart report."""
    return {
        "sarima": run_sarima_fan(
            curated_path=curated_path,
            n_sims=n_sims,
            seed=seed,
            verbose=verbose,
        ),
        "elastic_net": run_elastic_net_fan(
            curated_path=curated_path,
            n_sims=n_sims,
            seed=seed,
            verbose=verbose,
        ),
        "ensemble": run_ensemble_fan(
            curated_path=curated_path,
            n_sims=n_sims,
            seed=seed,
            verbose=verbose,
        ),
        "svar_headline": run_svar_fan(
            curated_path=curated_path,
            n_sims=svar_n_sims,
            seed=seed,
            verbose=verbose,
        ),
        "ensemble_path_sample": run_ensemble_path_sample(
            curated_path=curated_path,
            n_sims=n_sims,
            seed=seed,
            verbose=verbose,
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument(
        "--svar-n-sims",
        type=int,
        default=svar.DEFAULT_BOOTSTRAP_REPLICATIONS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    run_all_simulation_fans(
        curated_path=args.data,
        n_sims=args.n_sims,
        svar_n_sims=args.svar_n_sims,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
