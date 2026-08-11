"""Data validation helpers for CPI forecasting ETL outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class QualityRecord:
    """A compact validation result suitable for a CSV quality report."""

    dataset: str
    status: str
    rows: int
    columns: int
    start_date: str
    end_date: str
    missing_values: int
    duplicate_dates: int
    notes: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "status": self.status,
            "rows": self.rows,
            "columns": self.columns,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "missing_values": self.missing_values,
            "duplicate_dates": self.duplicate_dates,
            "notes": self.notes,
        }


def find_date_column(df: pd.DataFrame) -> str:
    """Return the likely date column used by downloaded ABS/RBA/yfinance files."""
    for column in df.columns:
        if str(column).strip().lower() in {"date", "time", "period", "quarter"}:
            return str(column)
    raise ValueError("No date-like column found. Expected date, time, period, or quarter.")


def parse_temporal_values(values: pd.Series) -> pd.Series:
    """Parse regular dates or quarterly labels such as 2020Q1."""
    text_values = values.dropna().astype(str).str.strip()
    if not text_values.empty and text_values.str.match(r"^\d{4}Q[1-4]$").all():
        parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        parsed.loc[text_values.index] = pd.PeriodIndex(text_values, freq="Q").to_timestamp(
            how="start"
        )
        return parsed
    return pd.to_datetime(values, errors="coerce")


def validate_time_series(
    df: pd.DataFrame,
    dataset: str,
    date_col: str,
    value_cols: Iterable[str],
    min_rows: int = 1,
) -> QualityRecord:
    """Validate one source or processed time-series dataset.

    The checks are intentionally lightweight and transparent for a portfolio
    project: date parsing, duplicates, ordering, numeric values, missingness,
    and unexpected empty inputs.
    """
    notes: list[str] = []
    status = "PASS"

    value_cols = list(value_cols)
    if df.empty:
        status = "FAIL"
        notes.append("dataset is empty")

    if len(df) < min_rows:
        status = "FAIL"
        notes.append(f"row count below minimum {min_rows}")

    if date_col not in df.columns:
        return QualityRecord(
            dataset=dataset,
            status="FAIL",
            rows=len(df),
            columns=len(df.columns),
            start_date="",
            end_date="",
            missing_values=int(df.isna().sum().sum()),
            duplicate_dates=0,
            notes=f"missing date column: {date_col}",
        )

    dates = parse_temporal_values(df[date_col])
    invalid_dates = int(dates.isna().sum())
    duplicate_dates = int(dates.duplicated().sum())

    if invalid_dates:
        status = "FAIL"
        notes.append(f"{invalid_dates} invalid date values")
    if duplicate_dates:
        status = "FAIL"
        notes.append(f"{duplicate_dates} duplicate dates")
    if dates.notna().any() and not dates.dropna().is_monotonic_increasing:
        status = "WARNING" if status == "PASS" else status
        notes.append("dates are not sorted")

    for column in value_cols:
        if column not in df.columns:
            status = "FAIL"
            notes.append(f"missing value column: {column}")
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        non_numeric = int(numeric.isna().sum() - df[column].isna().sum())
        if non_numeric:
            status = "FAIL"
            notes.append(f"{column} has {non_numeric} non-numeric values")

    missing_values = int(df[value_cols].isna().sum().sum()) if value_cols else 0
    if missing_values and status == "PASS":
        status = "WARNING"
        notes.append(f"{missing_values} missing values")

    return QualityRecord(
        dataset=dataset,
        status=status,
        rows=len(df),
        columns=len(df.columns),
        start_date=str(dates.min().date()) if dates.notna().any() else "",
        end_date=str(dates.max().date()) if dates.notna().any() else "",
        missing_values=missing_values,
        duplicate_dates=duplicate_dates,
        notes="; ".join(notes) if notes else "ok",
    )


def validate_curated_dataset(
    df: pd.DataFrame,
    min_rows: int = 80,
    target_col: str = "cpi_yoy",
) -> list[QualityRecord]:
    """Validate the final quarterly modelling table."""
    records: list[QualityRecord] = []

    required_cols = ["quarter", "cpi_index", "cpi_qoq", "cpi_yoy"]
    notes: list[str] = []
    status = "PASS"

    missing_required = [column for column in required_cols if column not in df.columns]
    if missing_required:
        status = "FAIL"
        notes.append(f"missing required columns: {', '.join(missing_required)}")

    duplicate_quarters = int(df["quarter"].duplicated().sum()) if "quarter" in df.columns else 0
    if duplicate_quarters:
        status = "FAIL"
        notes.append(f"{duplicate_quarters} duplicate quarters")

    usable_rows = int(df[target_col].notna().sum()) if target_col in df.columns else 0
    if usable_rows < min_rows:
        status = "FAIL"
        notes.append(f"{target_col} usable rows below minimum {min_rows}")

    if "quarter" in df.columns:
        quarters = pd.PeriodIndex(df["quarter"], freq="Q")
        full_range = pd.period_range(quarters.min(), quarters.max(), freq="Q")
        missing_quarters = len(full_range.difference(quarters))
        if missing_quarters:
            status = "WARNING" if status == "PASS" else status
            notes.append(f"{missing_quarters} missing quarters in date range")
        start_date = str(quarters.min())
        end_date = str(quarters.max())
    else:
        missing_quarters = 0
        start_date = ""
        end_date = ""

    records.append(
        QualityRecord(
            dataset="curated_quarterly_macro_features",
            status=status,
            rows=len(df),
            columns=len(df.columns),
            start_date=start_date,
            end_date=end_date,
            missing_values=int(df.isna().sum().sum()),
            duplicate_dates=duplicate_quarters,
            notes="; ".join(notes) if notes else "ok",
        )
    )

    for column in df.columns:
        if column == "quarter":
            continue
        missing = int(df[column].isna().sum())
        missing_share = missing / len(df) if len(df) else 1.0
        column_status = "PASS"
        column_notes = "ok"
        if missing_share > 0.50:
            column_status = "WARNING"
            column_notes = f"{missing_share:.1%} missing after quarterly merge"
        records.append(
            QualityRecord(
                dataset=f"curated_column:{column}",
                status=column_status,
                rows=int(df[column].notna().sum()),
                columns=1,
                start_date=start_date,
                end_date=end_date,
                missing_values=missing,
                duplicate_dates=0,
                notes=column_notes,
            )
        )

    return records
