"""Transformation helpers for aligning economic indicators to quarters."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from src.validation import find_date_column


Aggregation = Literal["mean", "last", "sum"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def to_quarterly(
    df: pd.DataFrame,
    value_col: str,
    output_col: str,
    agg: Aggregation = "mean",
    date_col: str | None = None,
) -> pd.DataFrame:
    """Convert a source time series to one row per quarter."""
    date_col = date_col or find_date_column(df)
    if value_col not in df.columns:
        raise ValueError(f"Missing value column {value_col}")

    result = df[[date_col, value_col]].copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="raise")
    result[value_col] = pd.to_numeric(result[value_col], errors="coerce")
    result["quarter"] = result[date_col].dt.to_period("Q")

    if agg == "mean":
        quarterly = result.groupby("quarter", as_index=False)[value_col].mean()
    elif agg == "last":
        quarterly = (
            result.sort_values(date_col)
            .groupby("quarter", as_index=False)[value_col]
            .last()
        )
    elif agg == "sum":
        quarterly = result.groupby("quarter", as_index=False)[value_col].sum()
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")

    quarterly = quarterly.rename(columns={value_col: output_col})
    quarterly["quarter"] = quarterly["quarter"].astype(str)
    return quarterly.sort_values("quarter").reset_index(drop=True)


def merge_quarterly_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join quarterly frames and preserve chronological order."""
    if not frames:
        raise ValueError("No quarterly frames were provided.")

    merged = frames[0].copy()
    for frame in frames[1:]:
        merged = merged.merge(frame, on="quarter", how="outer")

    merged["quarter_period"] = pd.PeriodIndex(merged["quarter"], freq="Q")
    merged = merged.sort_values("quarter_period").drop(columns="quarter_period")
    return merged.reset_index(drop=True)


def save_processed_frame(df: pd.DataFrame, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path
