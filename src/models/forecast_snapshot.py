"""Persist each family's next-quarter forecast to Supabase/PostgreSQL and compare
previously persisted snapshots against published actuals.

Snapshots only the horizon-1 ("next quarter") forecast per family so that
``sql/schema_app_metadata.sql``'s ``forecast_results`` table (quarter, forecast,
lower_ci, upper_ci -- no forecast_origin/horizon columns) needs no migration.
Both entry points are optional credential-dependent hooks, gated on
``DATABASE_URL``/``sqlalchemy`` exactly like ``load_postgres_quality_report`` in
``src/platform_loads.py``: they return a summary rather than raising when
Postgres isn't configured.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import pandas as pd

from api import main as api_main
from src.models.evaluation import CURATED_DATA_PATH, PROJECT_ROOT, load_target_series
from src.platform_loads import package_available


SNAPSHOT_HORIZON = 1
DEFAULT_N_SIMS = 1000
COMPARISON_REPORT_PATH = PROJECT_ROOT / "reports/forecast_snapshot_accuracy.csv"
COMPARISON_COLUMNS = [
    "model_name",
    "quarter",
    "forecast",
    "lower_ci",
    "upper_ci",
    "actual",
    "error",
    "hit",
    "status",
]


@dataclass(frozen=True)
class SnapshotSummary:
    """Outcome of one ``snapshot_forecasts()`` call."""

    status: str  # "unconfigured" or "completed"
    inserted: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    notes: str = ""


def snapshot_forecasts(
    n_sims: int = DEFAULT_N_SIMS,
) -> SnapshotSummary:
    """Serve each family's next-quarter forecast and persist it to Postgres.

    Calls ``POST /forecast/all`` (via ``api_main.forecast_all`` directly, not the
    HTTP layer) once with ``horizon=1``, then inserts one ``model_metrics`` row and
    one ``forecast_results`` row per available family, skipping families whose
    ``(model_name, quarter)`` key is already present so re-running the snapshot for
    an already-snapshotted quarter is a safe no-op.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return SnapshotSummary(
            status="unconfigured",
            notes="DATABASE_URL is not configured; skipped forecast snapshot.",
        )
    if not package_available("sqlalchemy"):
        return SnapshotSummary(
            status="unconfigured",
            notes="sqlalchemy is not installed.",
        )

    from sqlalchemy import create_engine, text

    engine = create_engine(database_url)
    response = api_main.forecast_all(
        api_main.AllForecastsRequest(horizon=SNAPSHOT_HORIZON, n_sims=n_sims)
    )
    run_date = datetime.now(timezone.utc).isoformat()

    inserted: list[str] = []
    skipped: list[str] = []

    with engine.begin() as conn:
        for family_forecast in response.models:
            family = family_forecast.model_family
            quarter = family_forecast.quarters[0]
            run_id = f"{family}:{quarter}"

            already_snapshotted = conn.execute(
                text("SELECT 1 FROM model_metrics WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).first()
            if already_snapshotted is not None:
                skipped.append(family)
                continue

            conn.execute(
                text(
                    "INSERT INTO model_metrics "
                    "(run_id, model_name, run_date, forecast_horizon, selected_model) "
                    "VALUES (:run_id, :model_name, :run_date, :forecast_horizon, :selected_model)"
                ),
                {
                    "run_id": run_id,
                    "model_name": family,
                    "run_date": run_date,
                    "forecast_horizon": SNAPSHOT_HORIZON,
                    # Always False: this project has no promoted "champion" model,
                    # so no family is singled out as selected. Column kept for
                    # schema compatibility with the deployed Postgres table.
                    "selected_model": False,
                },
            )
            conn.execute(
                text(
                    "INSERT INTO forecast_results "
                    "(run_id, quarter, forecast, lower_ci, upper_ci) "
                    "VALUES (:run_id, :quarter, :forecast, :lower_ci, :upper_ci)"
                ),
                {
                    "run_id": run_id,
                    "quarter": quarter,
                    "forecast": float(family_forecast.forecast[0]),
                    "lower_ci": float(family_forecast.interval_lower[0]),
                    "upper_ci": float(family_forecast.interval_upper[0]),
                },
            )
            inserted.append(family)

    return SnapshotSummary(
        status="completed",
        inserted=inserted,
        skipped=skipped,
        unavailable=[item.model_family for item in response.unavailable],
        notes=f"Inserted {len(inserted)} family snapshot(s); skipped {len(skipped)} already present.",
    )


def _observed_row(row: pd.Series, actuals_by_quarter: dict[str, float]) -> pd.Series:
    actual = actuals_by_quarter.get(str(row["quarter"]))
    if actual is None:
        return pd.Series({"actual": np.nan, "error": np.nan, "hit": pd.NA, "status": "pending"})
    error = actual - float(row["forecast"])
    hit = bool(float(row["lower_ci"]) <= actual <= float(row["upper_ci"]))
    return pd.Series({"actual": actual, "error": error, "hit": hit, "status": "observed"})


def compare_forecast_snapshots(
    output_path: Path = COMPARISON_REPORT_PATH,
    curated_path: Path = CURATED_DATA_PATH,
) -> pd.DataFrame:
    """Compare persisted forecast snapshots against observed actuals.

    Reads ``forecast_results`` joined to ``model_metrics`` from Postgres and joins
    it, at read time only, against the curated CPI series -- actuals are never
    written back into Postgres. Quarters without a published actual yet are
    reported as ``pending``, not dropped.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url or not package_available("sqlalchemy"):
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    snapshots = pd.read_sql(
        "SELECT mm.model_name AS model_name, fr.quarter AS quarter, "
        "fr.forecast AS forecast, fr.lower_ci AS lower_ci, fr.upper_ci AS upper_ci "
        "FROM forecast_results fr JOIN model_metrics mm ON mm.run_id = fr.run_id",
        engine,
    )

    if snapshots.empty:
        accuracy = snapshots.reindex(columns=COMPARISON_COLUMNS)
    else:
        actuals = load_target_series(curated_path)
        actuals_by_quarter = {str(period): float(value) for period, value in actuals.items()}
        extra = snapshots.apply(lambda row: _observed_row(row, actuals_by_quarter), axis=1)
        accuracy = pd.concat([snapshots, extra], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    accuracy.to_csv(output_path, index=False)
    return accuracy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser(
        "snapshot", help="Persist each family's next-quarter forecast to Postgres."
    )
    snapshot_parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare persisted forecast snapshots against actuals."
    )
    compare_parser.add_argument("--output", type=Path, default=COMPARISON_REPORT_PATH)
    compare_parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)

    args = parser.parse_args(argv)

    if args.command == "snapshot":
        summary = snapshot_forecasts(n_sims=args.n_sims)
        print(f"Status: {summary.status}")
        print(f"Inserted: {summary.inserted}")
        print(f"Skipped: {summary.skipped}")
        print(f"Unavailable: {summary.unavailable}")
        print(summary.notes)
    else:
        accuracy = compare_forecast_snapshots(output_path=args.output, curated_path=args.data)
        if accuracy.empty:
            print(
                "No forecast snapshot accuracy rows available "
                "(Postgres not configured, sqlalchemy missing, or no snapshots yet)."
            )
        else:
            print(accuracy.to_string(index=False))


if __name__ == "__main__":
    main()
