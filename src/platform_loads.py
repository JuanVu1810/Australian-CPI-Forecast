"""Load the curated data and its quality report into a local DuckDB database."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.validation import QualityRecord


def write_duckdb(
    curated: pd.DataFrame,
    quality_report: pd.DataFrame,
    db_path: Path,
) -> QualityRecord:
    """Write curated data and validation results to a local DuckDB database."""
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
