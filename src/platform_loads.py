"""Optional platform loaders for ETL and validation outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path

import pandas as pd

from src.validation import QualityRecord


def package_available(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def write_duckdb(
    curated: pd.DataFrame,
    quality_report: pd.DataFrame,
    db_path: Path,
) -> QualityRecord:
    """Write curated data and validation results to a local DuckDB database."""
    if not package_available("duckdb"):
        return QualityRecord(
            dataset="platform:duckdb",
            status="WARNING",
            rows=len(curated),
            columns=len(curated.columns),
            start_date=str(curated["quarter"].iloc[0]) if len(curated) else "",
            end_date=str(curated["quarter"].iloc[-1]) if len(curated) else "",
            missing_values=int(curated.isna().sum().sum()),
            duplicate_dates=int(curated["quarter"].duplicated().sum()) if "quarter" in curated else 0,
            notes="DuckDB is not installed; skipped local analytical database load.",
        )

    import duckdb

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as conn:
        conn.register("curated_df", curated)
        conn.register("quality_df", quality_report)
        conn.execute(
            "CREATE OR REPLACE TABLE quarterly_macro_features AS SELECT * FROM curated_df"
        )
        conn.execute(
            "CREATE OR REPLACE TABLE data_quality_results AS SELECT * FROM quality_df"
        )

    return QualityRecord(
        dataset="platform:duckdb",
        status="PASS",
        rows=len(curated),
        columns=len(curated.columns),
        start_date=str(curated["quarter"].iloc[0]) if len(curated) else "",
        end_date=str(curated["quarter"].iloc[-1]) if len(curated) else "",
        missing_values=int(curated.isna().sum().sum()),
        duplicate_dates=int(curated["quarter"].duplicated().sum()) if "quarter" in curated else 0,
        notes=f"Loaded curated data and quality report to {db_path}.",
    )


def load_postgres_quality_report(
    quality_report: pd.DataFrame,
    curated: pd.DataFrame | None = None,
    started_at: datetime | None = None,
) -> QualityRecord:
    """Load validation results to Supabase/PostgreSQL when configured.

    When ``curated`` is given, one ``pipeline_runs`` row describing this ETL run is
    written first and every quality row is linked to it through ``run_id``.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return QualityRecord(
            dataset="platform:postgres_data_quality_results",
            status="WARNING",
            rows=len(quality_report),
            columns=len(quality_report.columns),
            start_date="",
            end_date="",
            missing_values=int(quality_report.isna().sum().sum()),
            duplicate_dates=0,
            notes="DATABASE_URL is not configured; skipped PostgreSQL load.",
        )

    if not package_available("sqlalchemy"):
        return QualityRecord(
            dataset="platform:postgres_data_quality_results",
            status="FAIL",
            rows=len(quality_report),
            columns=len(quality_report.columns),
            start_date="",
            end_date="",
            missing_values=int(quality_report.isna().sum().sum()),
            duplicate_dates=0,
            notes="sqlalchemy is not installed.",
        )

    from sqlalchemy import create_engine

    run_id = None
    try:
        engine = create_engine(database_url)
        with engine.begin() as conn:
            to_load = quality_report
            if curated is not None:
                run_id = _insert_pipeline_run(conn, curated, quality_report, started_at)
                to_load = quality_report.assign(run_id=run_id)
            to_load.to_sql("data_quality_results", conn, if_exists="append", index=False)
    except Exception as exc:
        # Only the exception type is kept: driver messages can carry the database
        # host, and this note reaches the report and CI logs. The load is rolled
        # back, so a paused or unreachable database cannot stop the ETL.
        return QualityRecord(
            dataset="platform:postgres_data_quality_results",
            status="FAIL",
            rows=len(quality_report),
            columns=len(quality_report.columns),
            start_date="",
            end_date="",
            missing_values=int(quality_report.isna().sum().sum()),
            duplicate_dates=0,
            notes=f"PostgreSQL load failed ({type(exc).__name__}); nothing was written.",
        )

    return QualityRecord(
        dataset="platform:postgres_data_quality_results",
        status="PASS",
        rows=len(quality_report),
        columns=len(quality_report.columns),
        start_date="",
        end_date="",
        missing_values=int(quality_report.isna().sum().sum()),
        duplicate_dates=0,
        notes=(
            "Loaded data-quality report to PostgreSQL."
            if run_id is None
            else f"Loaded data-quality report to PostgreSQL under pipeline run {run_id}."
        ),
    )


def _insert_pipeline_run(
    conn,
    curated: pd.DataFrame,
    quality_report: pd.DataFrame,
    started_at: datetime | None,
) -> str:
    from sqlalchemy import text

    finished_at = datetime.now(timezone.utc)
    started_at = started_at or finished_at
    run_id = f"etl-{started_at:%Y%m%dT%H%M%SZ}"
    counts = quality_report["status"].value_counts()
    quarters = curated["quarter"].astype(str)
    conn.execute(
        text(
            "INSERT INTO pipeline_runs (run_id, started_at, finished_at, status, "
            "source_start_year, source_end_year, rows_curated, notes) VALUES "
            "(:run_id, :started_at, :finished_at, :status, :source_start_year, "
            ":source_end_year, :rows_curated, :notes)"
        ),
        {
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "status": "completed",
            "source_start_year": int(quarters.iloc[0][:4]),
            "source_end_year": int(quarters.iloc[-1][:4]),
            "rows_curated": len(curated),
            "notes": (
                f"Data quality: {counts.get('PASS', 0)} PASS, "
                f"{counts.get('WARNING', 0)} WARNING, {counts.get('FAIL', 0)} FAIL."
            ),
        },
    )
    return run_id
