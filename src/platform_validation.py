"""Pandera schema check for the curated dataset."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from src.validation import QualityRecord


# The first rows of cpi_qoq and cpi_yoy are missing on purpose (no earlier quarter
# to compare with), so those two columns allow nulls. Types are checked, not coerced.
CURATED_SCHEMA = pa.DataFrameSchema(
    {
        "quarter": pa.Column(str, nullable=False),
        "cpi_qoq": pa.Column(float, nullable=True),
        "cpi_yoy": pa.Column(float, nullable=True),
    },
    coerce=False,
    strict=False,
)


def validate_curated_with_pandera(df: pd.DataFrame) -> QualityRecord:
    """Check the curated columns and types; return a PASS or FAIL quality record."""
    try:
        CURATED_SCHEMA.validate(df, lazy=True)
        status = "PASS"
        notes = "Pandera schema validation passed."
    except pa.errors.SchemaErrors as exc:
        status = "FAIL"
        notes = f"Pandera schema failed with {len(exc.failure_cases)} failure cases."

    has_quarter = "quarter" in df and len(df) > 0
    return QualityRecord(
        dataset="pandera:curated_quarterly_macro_features",
        status=status,
        rows=len(df),
        columns=len(df.columns),
        start_date=str(df["quarter"].iloc[0]) if has_quarter else "",
        end_date=str(df["quarter"].iloc[-1]) if has_quarter else "",
        missing_values=int(df.isna().sum().sum()),
        duplicate_dates=int(df["quarter"].duplicated().sum()) if "quarter" in df else 0,
        notes=notes,
    )
