"""Persist each family's next-quarter forecast to Supabase/PostgreSQL and compare
previously persisted snapshots against published actuals.

Snapshots only the horizon-1 ("next quarter") forecast per family so that
``sql/schema_app_metadata.sql``'s ``forecast_results`` table (quarter, forecast,
lower_ci, upper_ci -- no forecast_origin/horizon columns) needs no migration.
Headline rows keep the plain family name (``sarima``); trimmed-mean rows carry a
``_trimmed_mean`` suffix on ``model_name`` and ``run_id``, so no target column is
needed either. ``model_metrics`` rmse/mae/mse are the family's horizon-1
walk-forward figures from the committed comparison reports (mse is rmse squared),
and ``feature_set`` lists the inputs the served model uses. Both entry points are
optional credential-dependent hooks, gated on ``DATABASE_URL``/``sqlalchemy``
exactly like ``load_postgres_quality_report`` in ``src/platform_loads.py``: they
return a summary rather than raising when Postgres isn't configured.
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
from src.models.elastic_net import (
    ELASTIC_NET_FEATURE_COLUMNS,
    TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
    TRIMMED_MEAN_TARGET_COLUMN,
)
from src.models.evaluation import (
    CURATED_DATA_PATH,
    PROJECT_ROOT,
    TARGET_COLUMN,
    load_target_series,
)
from src.platform_loads import package_available


SNAPSHOT_HORIZON = 1
DEFAULT_N_SIMS = 1000
TRIMMED_MEAN_SUFFIX = "_trimmed_mean"
TARGET_COLUMNS = {"headline": TARGET_COLUMN, "trimmed_mean": TRIMMED_MEAN_TARGET_COLUMN}
METRICS_REPORT_PATHS = {
    "": PROJECT_ROOT / "reports/model_comparison_all.csv",
    TRIMMED_MEAN_SUFFIX: PROJECT_ROOT / "reports/model_comparison_trimmed_mean_all.csv",
}
COMPARISON_REPORT_PATH = PROJECT_ROOT / "reports/forecast_snapshot_accuracy.csv"
COMPARISON_COLUMNS = [
    "model_name",
    "target",
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
    backfilled: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    notes: str = ""


def snapshot_forecasts(
    n_sims: int = DEFAULT_N_SIMS,
) -> SnapshotSummary:
    """Serve each family's next-quarter forecast and persist it to Postgres.

    Calls ``POST /forecast/all`` and ``POST /forecast/trimmed-mean/all`` (via
    ``api_main`` directly, not the HTTP layer) once each with ``horizon=1``, then
    inserts one ``model_metrics`` row and one ``forecast_results`` row per available
    family and target, skipping any whose ``(model_name, quarter)`` key is already
    present so re-running the snapshot for an already-snapshotted quarter is a safe
    no-op.
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

    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    request = api_main.AllForecastsRequest(horizon=SNAPSHOT_HORIZON, n_sims=n_sims)
    responses = [
        ("", api_main.forecast_all(request)),
        (TRIMMED_MEAN_SUFFIX, api_main.forecast_trimmed_mean_all(request)),
    ]
    run_date = datetime.now(timezone.utc).isoformat()

    inserted: list[str] = []
    skipped: list[str] = []
    backfilled: list[str] = []
    unavailable: list[str] = []

    with engine.begin() as conn:
        for suffix, response in responses:
            unavailable.extend(f"{item.model_family}{suffix}" for item in response.unavailable)
            horizon_one = _load_horizon_one_metrics(METRICS_REPORT_PATHS[suffix])
            for family_forecast in response.models:
                family = family_forecast.model_family
                rmse, mae = horizon_one.get(family, (None, None))
                _insert_snapshot(
                    conn,
                    model_name=f"{family}{suffix}",
                    family_forecast=family_forecast,
                    run_date=run_date,
                    metrics={
                        "feature_set": _feature_set(family, suffix),
                        "rmse": rmse,
                        "mae": mae,
                        "mse": None if rmse is None else rmse**2,
                    },
                    inserted=inserted,
                    skipped=skipped,
                    backfilled=backfilled,
                )

    return SnapshotSummary(
        status="completed",
        inserted=inserted,
        skipped=skipped,
        backfilled=backfilled,
        unavailable=unavailable,
        notes=(
            f"Inserted {len(inserted)} family snapshot(s); skipped {len(skipped)} already "
            f"present; filled in missing metrics on {len(backfilled)}."
        ),
    )


def _load_horizon_one_metrics(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        return {}
    report = pd.read_csv(path, usecols=["model", "horizon", "rmse", "mae"])
    # The report's horizon column also holds an "overall" row, so it is read as text.
    report = report[report["horizon"].astype(str) == str(SNAPSHOT_HORIZON)]
    return {row.model: (float(row.rmse), float(row.mae)) for row in report.itertuples()}


def _feature_set(family: str, suffix: str) -> str | None:
    if family == "sarima":
        return "univariate (own lags only)"
    if family == "elastic_net":
        columns = (
            TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS
            if suffix
            else ELASTIC_NET_FEATURE_COLUMNS
        )
        return ", ".join(columns)
    if family == "ensemble":
        return "sarima + elastic_net (inverse-RMSE weights)"
    return None


def _insert_snapshot(
    conn,
    *,
    model_name: str,
    family_forecast: api_main.FamilyForecast,
    run_date: str,
    metrics: dict[str, object],
    inserted: list[str],
    skipped: list[str],
    backfilled: list[str],
) -> None:
    from sqlalchemy import text

    quarter = family_forecast.quarters[0]
    run_id = f"{model_name}:{quarter}"

    already_snapshotted = conn.execute(
        text("SELECT 1 FROM model_metrics WHERE run_id = :run_id"),
        {"run_id": run_id},
    ).first()
    if already_snapshotted is not None:
        # Rows written before metrics were recorded get them filled in once.
        if metrics["rmse"] is not None:
            updated = conn.execute(
                text(
                    "UPDATE model_metrics SET feature_set = :feature_set, rmse = :rmse, "
                    "mae = :mae, mse = :mse WHERE run_id = :run_id AND rmse IS NULL"
                ),
                {"run_id": run_id, **metrics},
            )
            if updated.rowcount:
                backfilled.append(model_name)
        skipped.append(model_name)
        return

    conn.execute(
        text(
            "INSERT INTO model_metrics "
            "(run_id, model_name, run_date, forecast_horizon, feature_set, rmse, mae, mse, "
            "selected_model) VALUES (:run_id, :model_name, :run_date, :forecast_horizon, "
            ":feature_set, :rmse, :mae, :mse, :selected_model)"
        ),
        {
            "run_id": run_id,
            "model_name": model_name,
            "run_date": run_date,
            "forecast_horizon": SNAPSHOT_HORIZON,
            **metrics,
            # Always False: this project has no promoted "champion" model,
            # so no family is singled out as selected. Column kept to
            # match sql/schema_app_metadata.sql.
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
    inserted.append(model_name)


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
    it, at read time only, against the curated CPI series for each row's target
    (headline, or trimmed mean when ``model_name`` ends in ``_trimmed_mean``) --
    actuals are never written back into Postgres. Quarters without a published actual yet are
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
        snapshots.insert(
            1,
            "target",
            snapshots["model_name"].map(
                lambda name: "trimmed_mean" if name.endswith(TRIMMED_MEAN_SUFFIX) else "headline"
            ),
        )
        actuals_by_target = {}
        for target in snapshots["target"].unique():
            actuals = load_target_series(curated_path, target_column=TARGET_COLUMNS[target])
            actuals_by_target[target] = {
                str(period): float(value) for period, value in actuals.items()
            }
        extra = snapshots.apply(
            lambda row: _observed_row(row, actuals_by_target[row["target"]]), axis=1
        )
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
        print(f"Metrics filled in: {summary.backfilled}")
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
