"""Export the SVAR systems' forecast path for the unemployment rate.

Writes ``reports/svar_unemployment_forecast.csv``. For each SVAR system it holds the mean-path forecast of the
unemployment rate over eight quarters from the pinned forecast origin, the 10th and 90th percentiles of the
block-bootstrap simulation (the same machinery behind the IRF bands and the credit stress test), and the cumulative
change from the last observed rate.

    python -m src.models.svar_unemployment_export
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import svar
from src.models.tableau_export import SVAR_SYSTEMS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "svar_unemployment_forecast.csv"
HORIZONS = 8
DECIMALS = 3
NUMERIC_COLUMNS = [
    "unemployment_rate_origin",
    "unemployment_rate_forecast",
    "p10",
    "p90",
    "cumulative_change_vs_origin",
]


def build_unemployment_forecast(
    steps: int = HORIZONS,
    n_sims: int = svar.DEFAULT_BOOTSTRAP_REPLICATIONS,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per system and horizon; values are unrounded until written."""
    rows = []
    for name, config in SVAR_SYSTEMS.items():
        frame = svar.load_svar_level_frame(columns=config["columns"])
        frame = frame.loc[frame.index <= svar.FORECAST_ORIGIN_PIN]
        origin_quarter = frame.index[-1]
        origin_rate = float(frame["unemployment_rate"].iloc[-1])

        fitted = svar.fit_svar(frame, ordering=config["ordering"])
        mean_path = svar.forecast_from_fit(fitted, steps=steps)["unemployment_rate"].to_numpy()
        paths = svar.simulate_paths_from_fit(fitted, steps=steps, n_sims=n_sims, seed=seed)
        simulated = paths[:, :, fitted.names.index("unemployment_rate")]
        p10, p90 = np.percentile(
            simulated,
            [100 * svar.DEFAULT_LOWER_QUANTILE, 100 * svar.DEFAULT_UPPER_QUANTILE],
            axis=0,
        )
        for horizon in range(1, steps + 1):
            forecast = float(mean_path[horizon - 1])
            rows.append(
                {
                    "system": name.split()[-1],
                    "system_target": config["target"],
                    "forecast_origin": str(origin_quarter),
                    "quarter": str(origin_quarter + horizon),
                    "horizon": horizon,
                    "unemployment_rate_origin": origin_rate,
                    "unemployment_rate_forecast": forecast,
                    "p10": float(p10[horizon - 1]),
                    "p90": float(p90[horizon - 1]),
                    "cumulative_change_vs_origin": forecast - origin_rate,
                }
            )
    return pd.DataFrame(rows)


def write_unemployment_forecast(path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    table = build_unemployment_forecast()
    table[NUMERIC_COLUMNS] = table[NUMERIC_COLUMNS].round(DECIMALS)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the SVAR unemployment-rate forecast path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    table = write_unemployment_forecast(args.output)
    print(f"Wrote {len(table)} rows to {args.output}")


if __name__ == "__main__":
    main()
