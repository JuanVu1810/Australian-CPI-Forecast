"""Optional platform-backed validation for ETL outputs."""

from __future__ import annotations

import importlib.util

import pandas as pd

from src.validation import QualityRecord


def pandera_available() -> bool:
    return importlib.util.find_spec("pandera") is not None


def validate_curated_with_pandera(df: pd.DataFrame) -> QualityRecord:
    """Validate the curated dataset with Pandera when the dependency exists.

    The custom validation module remains the local fallback. This function makes
    the target validation platform explicit without forcing Pandera to be
    installed before the basic ETL can run.
    """
    if not pandera_available():
        return QualityRecord(
            dataset="pandera:curated_quarterly_macro_features",
            status="WARNING",
            rows=len(df),
            columns=len(df.columns),
            start_date=str(df["quarter"].iloc[0]) if "quarter" in df and len(df) else "",
            end_date=str(df["quarter"].iloc[-1]) if "quarter" in df and len(df) else "",
            missing_values=int(df.isna().sum().sum()),
            duplicate_dates=int(df["quarter"].duplicated().sum()) if "quarter" in df else 0,
            notes="Pandera is not installed; custom validation was used as fallback.",
        )

    import pandera.pandas as pa
    from pandera import Check, Column

    schema = pa.DataFrameSchema(
        {
            "quarter": Column(str, checks=Check(lambda s: s.is_unique)),
            "cpi_qoq": Column(float, nullable=True),
            "cpi_yoy": Column(float, nullable=True),
        },
        coerce=True,
        strict=False,
    )

    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return QualityRecord(
            dataset="pandera:curated_quarterly_macro_features",
            status="FAIL",
            rows=len(df),
            columns=len(df.columns),
            start_date=str(df["quarter"].iloc[0]) if len(df) else "",
            end_date=str(df["quarter"].iloc[-1]) if len(df) else "",
            missing_values=int(df.isna().sum().sum()),
            duplicate_dates=int(df["quarter"].duplicated().sum()) if "quarter" in df else 0,
            notes=f"Pandera schema failed with {len(exc.failure_cases)} failure cases.",
        )

    return QualityRecord(
        dataset="pandera:curated_quarterly_macro_features",
        status="PASS",
        rows=len(df),
        columns=len(df.columns),
        start_date=str(df["quarter"].iloc[0]) if len(df) else "",
        end_date=str(df["quarter"].iloc[-1]) if len(df) else "",
        missing_values=int(df.isna().sum().sum()),
        duplicate_dates=int(df["quarter"].duplicated().sum()) if "quarter" in df else 0,
        notes="Pandera schema validation passed.",
    )
