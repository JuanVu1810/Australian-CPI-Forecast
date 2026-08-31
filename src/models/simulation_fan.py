"""Forward simulated-path percentile fan-chart reports for CPI model families."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.models import ensemble, svar
from src.models.elastic_net import (
    ELASTIC_NET_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
    TRIMMED_MEAN_TARGET_COLUMN,
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
    TRIMMED_MEAN_DEFAULT_ORDER,
    TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER,
    simulate_sarima_paths,
)


SIMULATION_FAN_SARIMA_OUTPUT_PATH = PROJECT_ROOT / "reports/simulation_fan_sarima.csv"
SIMULATION_FAN_SARIMA_TRIMMED_MEAN_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/simulation_fan_sarima_trimmed_mean.csv"
)
SIMULATION_FAN_ELASTIC_NET_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/simulation_fan_elastic_net.csv"
)
SIMULATION_FAN_ELASTIC_NET_TRIMMED_MEAN_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/simulation_fan_elastic_net_trimmed_mean.csv"
)
SIMULATION_FAN_ENSEMBLE_OUTPUT_PATH = PROJECT_ROOT / "reports/simulation_fan_ensemble.csv"
SIMULATION_FAN_ENSEMBLE_TRIMMED_MEAN_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/simulation_fan_ensemble_trimmed_mean.csv"
)
SIMULATION_FAN_SVAR_HEADLINE_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/simulation_fan_svar_headline.csv"
)
SIMULATION_FAN_SVAR_TRIMMED_MEAN_OUTPUT_PATH = (
    PROJECT_ROOT / "reports/simulation_fan_svar_trimmed_mean.csv"
)
SIMULATION_FAN_OUTPUT_PATHS = {
    "sarima": SIMULATION_FAN_SARIMA_OUTPUT_PATH,
    "sarima_trimmed_mean": SIMULATION_FAN_SARIMA_TRIMMED_MEAN_OUTPUT_PATH,
    "elastic_net": SIMULATION_FAN_ELASTIC_NET_OUTPUT_PATH,
    "elastic_net_trimmed_mean": SIMULATION_FAN_ELASTIC_NET_TRIMMED_MEAN_OUTPUT_PATH,
    "ensemble": SIMULATION_FAN_ENSEMBLE_OUTPUT_PATH,
    "ensemble_trimmed_mean": SIMULATION_FAN_ENSEMBLE_TRIMMED_MEAN_OUTPUT_PATH,
    "svar_headline": SIMULATION_FAN_SVAR_HEADLINE_OUTPUT_PATH,
    "svar_trimmed_mean": SIMULATION_FAN_SVAR_TRIMMED_MEAN_OUTPUT_PATH,
}
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    if verbose:
        family = (
            str(frame["model_family"].iloc[0])
            if not frame.empty
            else "simulation fan"
        )
        print(
            f"Saved {family} simulation fan to {output_path} ({len(frame)} rows).",
            flush=True,
        )
    return frame


def _elastic_net_training_frame(
    curated_path: Path,
    *,
    target_column: str,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    target = load_target_series(curated_path, target_column=target_column)
    features = load_elastic_net_feature_frame(
        curated_path,
        feature_columns=feature_columns,
    )
    frame = pd.concat([target.rename(target_column), features], axis=1).dropna()
    if frame.empty:
        raise ValueError(f"No complete Elastic Net rows for target {target_column!r}.")
    return frame.sort_index()


def _load_svar_frame(curated_path: Path, columns: Sequence[str]) -> pd.DataFrame:
    required_columns = ["quarter", *columns]
    data = pd.read_csv(curated_path, usecols=required_columns)
    data.index = pd.PeriodIndex(data.pop("quarter").astype(str), freq="Q")
    frame = data.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty:
        raise ValueError(f"No complete SVAR rows for columns {tuple(columns)}.")
    return frame.sort_index()


def run_sarima_fan(
    *,
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = SIMULATION_FAN_SARIMA_OUTPUT_PATH,
    target_column: str = TARGET_COLUMN,
    model_family: str = "sarima",
    order: tuple[int, int, int] = SARIMA_DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = SARIMA_DEFAULT_SEASONAL_ORDER,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    series = load_target_series(curated_path, target_column=target_column)
    paths = simulate_sarima_paths(
        series,
        steps=max(horizons),
        n_sims=n_sims,
        order=order,
        seasonal_order=seasonal_order,
        seed=seed,
    )
    frame = fan_frame_from_paths(
        paths,
        model_family=model_family,
        target_column=target_column,
        forecast_origin=str(series.index[-1]),
        horizons=horizons,
    )
    return _write_fan(frame, output_path, verbose=verbose)


def run_elastic_net_fan(
    *,
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = SIMULATION_FAN_ELASTIC_NET_OUTPUT_PATH,
    target_column: str = TARGET_COLUMN,
    model_family: str = "elastic_net",
    feature_columns: tuple[str, ...] = ELASTIC_NET_FEATURE_COLUMNS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    frame = _elastic_net_training_frame(
        curated_path,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    paths = simulate_elastic_net_paths(
        frame,
        steps=max(horizons),
        n_sims=n_sims,
        seed=seed,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    fan = fan_frame_from_paths(
        paths,
        model_family=model_family,
        target_column=target_column,
        forecast_origin=str(frame.index[-1]),
        horizons=horizons,
    )
    return _write_fan(fan, output_path, verbose=verbose)


def run_ensemble_fan(
    *,
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = SIMULATION_FAN_ENSEMBLE_OUTPUT_PATH,
    target_column: str = TARGET_COLUMN,
    model_family: str = "ensemble",
    sarima_order: tuple[int, int, int] = SARIMA_DEFAULT_ORDER,
    sarima_seasonal_order: tuple[int, int, int, int] = SARIMA_DEFAULT_SEASONAL_ORDER,
    elastic_net_feature_columns: tuple[str, ...] = ELASTIC_NET_FEATURE_COLUMNS,
    weights: tuple[float, float] | dict[int, tuple[float, float]] | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    series = load_target_series(curated_path, target_column=target_column)
    frame = _elastic_net_training_frame(
        curated_path,
        target_column=target_column,
        feature_columns=elastic_net_feature_columns,
    )
    if series.index[-1] != frame.index[-1]:
        raise ValueError(
            f"Ensemble {target_column!r} components end on different quarters: "
            f"SARIMA={series.index[-1]}, Elastic Net={frame.index[-1]}."
        )
    if weights is None:
        weights = ensemble.horizon_rmse_weights(
            horizons=tuple(int(h) for h in horizons)
        )
    paths = ensemble.simulate_ensemble_paths(
        frame,
        steps=max(horizons),
        n_sims=n_sims,
        weights=weights,
        seed=seed,
        target_column=target_column,
        sarima_order=sarima_order,
        sarima_seasonal_order=sarima_seasonal_order,
        elastic_net_feature_columns=elastic_net_feature_columns,
        sarima_series=series,
    )
    fan = fan_frame_from_paths(
        paths,
        model_family=model_family,
        target_column=target_column,
        forecast_origin=str(frame.index[-1]),
        horizons=horizons,
    )
    return _write_fan(fan, output_path, verbose=verbose)


def run_svar_fan(
    *,
    curated_path: Path = CURATED_DATA_PATH,
    output_path: Path = SIMULATION_FAN_SVAR_HEADLINE_OUTPUT_PATH,
    target_column: str = TARGET_COLUMN,
    model_family: str = "svar_headline",
    columns: Sequence[str] = svar.SYSTEM_A_COLUMNS,
    ordering: Sequence[str] = svar.SYSTEM_A_CHOLESKY_ORDER,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_sims: int = svar.DEFAULT_BOOTSTRAP_REPLICATIONS,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    frame = _load_svar_frame(curated_path, columns)
    ordered_names = tuple(str(name) for name in ordering)
    paths = svar.simulate_svar_paths(
        frame,
        steps=max(horizons),
        n_sims=n_sims,
        lag_order=svar.DEFAULT_SVAR_LAG_ORDER,
        ordering=ordering,
        seed=seed,
    )
    target_paths = np.asarray(paths, dtype=float)[
        :,
        :,
        ordered_names.index(target_column),
    ]
    fan = fan_frame_from_paths(
        target_paths,
        model_family=model_family,
        target_column=target_column,
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
            output_path=SIMULATION_FAN_SARIMA_OUTPUT_PATH,
            n_sims=n_sims,
            seed=seed,
            verbose=verbose,
        ),
        "sarima_trimmed_mean": run_sarima_fan(
            curated_path=curated_path,
            output_path=SIMULATION_FAN_SARIMA_TRIMMED_MEAN_OUTPUT_PATH,
            target_column=TRIMMED_MEAN_TARGET_COLUMN,
            model_family="sarima_trimmed_mean",
            order=TRIMMED_MEAN_DEFAULT_ORDER,
            seasonal_order=TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER,
            n_sims=n_sims,
            seed=seed,
            verbose=verbose,
        ),
        "elastic_net": run_elastic_net_fan(
            curated_path=curated_path,
            output_path=SIMULATION_FAN_ELASTIC_NET_OUTPUT_PATH,
            n_sims=n_sims,
            seed=seed,
            verbose=verbose,
        ),
        "elastic_net_trimmed_mean": run_elastic_net_fan(
            curated_path=curated_path,
            output_path=SIMULATION_FAN_ELASTIC_NET_TRIMMED_MEAN_OUTPUT_PATH,
            target_column=TRIMMED_MEAN_TARGET_COLUMN,
            model_family="elastic_net_trimmed_mean",
            feature_columns=TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
            n_sims=n_sims,
            seed=seed,
            verbose=verbose,
        ),
        "ensemble": run_ensemble_fan(
            curated_path=curated_path,
            output_path=SIMULATION_FAN_ENSEMBLE_OUTPUT_PATH,
            n_sims=n_sims,
            seed=seed,
            verbose=verbose,
        ),
        "ensemble_trimmed_mean": run_ensemble_fan(
            curated_path=curated_path,
            output_path=SIMULATION_FAN_ENSEMBLE_TRIMMED_MEAN_OUTPUT_PATH,
            target_column=TRIMMED_MEAN_TARGET_COLUMN,
            model_family="ensemble_trimmed_mean",
            sarima_order=TRIMMED_MEAN_DEFAULT_ORDER,
            sarima_seasonal_order=TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER,
            elastic_net_feature_columns=(
                TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS
            ),
            weights={
                horizon: ensemble.DEFAULT_WEIGHTS
                for horizon in DEFAULT_HORIZONS
            },
            n_sims=n_sims,
            seed=seed,
            verbose=verbose,
        ),
        "svar_headline": run_svar_fan(
            curated_path=curated_path,
            output_path=SIMULATION_FAN_SVAR_HEADLINE_OUTPUT_PATH,
            n_sims=svar_n_sims,
            seed=seed,
            verbose=verbose,
        ),
        "svar_trimmed_mean": run_svar_fan(
            curated_path=curated_path,
            output_path=SIMULATION_FAN_SVAR_TRIMMED_MEAN_OUTPUT_PATH,
            target_column=TRIMMED_MEAN_TARGET_COLUMN,
            model_family="svar_trimmed_mean",
            columns=svar.SYSTEM_B_COLUMNS,
            ordering=svar.SYSTEM_B_CHOLESKY_ORDER,
            n_sims=svar_n_sims,
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
