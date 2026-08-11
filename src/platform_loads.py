"""Optional platform loaders for ETL and validation outputs."""

from __future__ import annotations

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


def load_bigquery(curated: pd.DataFrame) -> QualityRecord:
    """Upload curated data to BigQuery when credentials are configured."""
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset = os.getenv("BIGQUERY_DATASET", "cpi_forecast")
    table = os.getenv("BIGQUERY_TABLE", "quarterly_macro_features")

    if not project_id:
        return QualityRecord(
            dataset="platform:bigquery",
            status="WARNING",
            rows=len(curated),
            columns=len(curated.columns),
            start_date=str(curated["quarter"].iloc[0]) if len(curated) else "",
            end_date=str(curated["quarter"].iloc[-1]) if len(curated) else "",
            missing_values=int(curated.isna().sum().sum()),
            duplicate_dates=int(curated["quarter"].duplicated().sum()) if "quarter" in curated else 0,
            notes="GCP_PROJECT_ID is not configured; skipped BigQuery load.",
        )

    if not package_available("google.cloud.bigquery"):
        return QualityRecord(
            dataset="platform:bigquery",
            status="FAIL",
            rows=len(curated),
            columns=len(curated.columns),
            start_date=str(curated["quarter"].iloc[0]) if len(curated) else "",
            end_date=str(curated["quarter"].iloc[-1]) if len(curated) else "",
            missing_values=int(curated.isna().sum().sum()),
            duplicate_dates=int(curated["quarter"].duplicated().sum()) if "quarter" in curated else 0,
            notes="google-cloud-bigquery is not installed.",
        )

    from google.cloud import bigquery

    table_id = f"{project_id}.{dataset}.{table}"
    client = bigquery.Client(project=project_id)
    job = client.load_table_from_dataframe(
        curated,
        table_id,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()

    return QualityRecord(
        dataset="platform:bigquery",
        status="PASS",
        rows=len(curated),
        columns=len(curated.columns),
        start_date=str(curated["quarter"].iloc[0]) if len(curated) else "",
        end_date=str(curated["quarter"].iloc[-1]) if len(curated) else "",
        missing_values=int(curated.isna().sum().sum()),
        duplicate_dates=int(curated["quarter"].duplicated().sum()) if "quarter" in curated else 0,
        notes=f"Loaded curated data to {table_id}.",
    )


def load_postgres_quality_report(quality_report: pd.DataFrame) -> QualityRecord:
    """Load validation results to Supabase/PostgreSQL when configured."""
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

    engine = create_engine(database_url)
    quality_report.to_sql(
        "data_quality_results",
        engine,
        if_exists="append",
        index=False,
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
        notes="Loaded data-quality report to PostgreSQL.",
    )
