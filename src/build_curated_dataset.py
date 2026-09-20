"""Build the quarterly macroeconomic modelling dataset for CPI forecasting.

Run from the project root:

    python -m src.build_curated_dataset

Outputs:

    data/processed/*.csv
    data/curated/quarterly_macro_features.csv
    reports/data_quality_report.csv

Parquet output is attempted when an optional Parquet engine is installed.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.features import (
    add_growth_rates,
    add_intervention_dummies,
    add_lag_features,
    add_nonlinear_elastic_net_terms,
    order_feature_columns,
)
from src.platform_loads import load_bigquery, load_postgres_quality_report, write_duckdb
from src.platform_validation import validate_curated_with_pandera
from src.transform import (
    latest_dataset_path,
    merge_quarterly_frames,
    read_csv,
    save_processed_frame,
    to_quarterly,
)
from src.validation import (
    QualityRecord,
    find_date_column,
    validate_curated_dataset,
    validate_time_series,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    path: Path
    value_col: str
    output_col: str
    agg: str


SERIES_SPECS = [
    SeriesSpec(
        name="cpi_qoq",
        path=Path("dataset/abs/cpi_qoq_1995_2026.csv"),
        value_col="cpi_qoq",
        output_col="cpi_qoq",
        agg="last",
    ),
    SeriesSpec(
        name="cpi_yoy",
        path=Path("dataset/abs/cpi_yoy_1995_2026.csv"),
        value_col="cpi_yoy",
        output_col="cpi_yoy",
        agg="last",
    ),
    SeriesSpec(
        name="trimmed_mean_cpi_qoq",
        path=Path("dataset/abs/trimmed_mean_cpi_qoq_1995_2026.csv"),
        value_col="trimmed_mean_cpi_qoq",
        output_col="trimmed_mean_cpi_qoq",
        agg="last",
    ),
    SeriesSpec(
        name="trimmed_mean_cpi_yoy",
        path=Path("dataset/abs/trimmed_mean_cpi_yoy_1995_2026.csv"),
        value_col="trimmed_mean_cpi_yoy",
        output_col="trimmed_mean_cpi_yoy",
        agg="last",
    ),
    SeriesSpec(
        name="unemployment_rate",
        path=Path("dataset/abs/unemployment_rate_1995_2026.csv"),
        value_col="unemployment_rate",
        output_col="unemployment_rate",
        agg="mean",
    ),
    SeriesSpec(
        name="cash_rate",
        path=Path("dataset/rba/cash_rate_1995_2026.csv"),
        value_col="cash_rate",
        output_col="cash_rate",
        agg="mean",
    ),
    SeriesSpec(
        name="wage_price_index",
        path=Path("dataset/abs/wage_price_index_1995_2026.csv"),
        value_col="wage_price_index",
        output_col="wage_price_index",
        agg="last",
    ),
    SeriesSpec(
        name="producer_price_index",
        path=Path("dataset/abs/producer_price_index_1995_2026.csv"),
        value_col="producer_price_index",
        output_col="producer_price_index",
        agg="last",
    ),
    SeriesSpec(
        name="commodity_price_index",
        path=Path("dataset/rba/commodity_prices_1995_2026.csv"),
        value_col="GRCPAIAD",
        output_col="commodity_price_index",
        agg="mean",
    ),
    SeriesSpec(
        name="wti_price",
        path=Path("dataset/market/wti_crude_oil_1995_2025.csv"),
        value_col="Close",
        output_col="wti_price",
        agg="mean",
    ),
    SeriesSpec(
        name="brent_price",
        path=Path("dataset/market/brent_crude_oil_1995_2025.csv"),
        value_col="Close",
        output_col="brent_price",
        agg="mean",
    ),
    SeriesSpec(
        name="aud_usd",
        path=Path("dataset/rba/aud_usd_exchange_rate_1995_2026.csv"),
        value_col="FXRUSD",
        output_col="aud_usd",
        agg="mean",
    ),
    SeriesSpec(
        name="household_spending",
        path=Path("dataset/abs/household_spending_1995_2026.csv"),
        value_col="household_spending",
        output_col="household_spending",
        agg="mean",
    ),
    SeriesSpec(
        name="inflation_expectations_business",
        path=Path("dataset/rba/inflation_expectations_1995_2026.csv"),
        value_col="GBUSEXP",
        output_col="inflation_expectations_business",
        agg="mean",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the curated quarterly CPI macro-feature dataset."
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--duckdb-path", default="data/analytics/cpi_forecast.duckdb")
    parser.add_argument(
        "--skip-duckdb",
        action="store_true",
        help="Skip the DuckDB analytical load attempted by default.",
    )
    parser.add_argument(
        "--load-bigquery",
        action="store_true",
        help="Upload the curated dataset to BigQuery using environment variables.",
    )
    parser.add_argument(
        "--load-postgres",
        action="store_true",
        help="Load the data-quality report to PostgreSQL/Supabase using DATABASE_URL.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def build_quarterly_frames(
    specs: list[SeriesSpec],
    processed_dir: Path,
) -> tuple[list[pd.DataFrame], list[QualityRecord]]:
    frames: list[pd.DataFrame] = []
    quality_records: list[QualityRecord] = []

    for spec in specs:
        source_path = latest_dataset_path(spec.path)
        LOGGER.info("Processing %s from %s", spec.name, source_path)
        source = read_csv(source_path)
        date_col = find_date_column(source)
        quality_records.append(
            validate_time_series(
                source,
                dataset=f"raw:{spec.name}",
                date_col=date_col,
                value_cols=[spec.value_col],
            )
        )

        quarterly = to_quarterly(
            source,
            value_col=spec.value_col,
            output_col=spec.output_col,
            agg=spec.agg,  # type: ignore[arg-type]
            date_col=date_col,
        )
        save_processed_frame(quarterly, processed_dir, spec.name)
        quality_records.append(
            validate_time_series(
                quarterly,
                dataset=f"processed:{spec.name}",
                date_col="quarter",
                value_cols=[spec.output_col],
            )
        )
        frames.append(quarterly)

    return frames, quality_records


def save_curated_outputs(df: pd.DataFrame, curated_dir: Path) -> None:
    curated_dir.mkdir(parents=True, exist_ok=True)
    csv_path = curated_dir / "quarterly_macro_features.csv"
    df.to_csv(csv_path, index=False)
    LOGGER.info("Saved curated CSV: %s", csv_path)

    parquet_path = curated_dir / "quarterly_macro_features.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
    except ImportError:
        LOGGER.warning("Parquet engine not installed; skipped %s", parquet_path)
    else:
        LOGGER.info("Saved curated Parquet: %s", parquet_path)


def save_quality_report(records: list[QualityRecord], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame([record.as_dict() for record in records])
    path = reports_dir / "data_quality_report.csv"
    report.to_csv(path, index=False)
    LOGGER.info("Saved data quality report: %s", path)


def records_to_frame(records: list[QualityRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.as_dict() for record in records])


def main() -> int:
    configure_logging()
    args = parse_args()

    processed_dir = Path(args.processed_dir)
    curated_dir = Path(args.curated_dir)
    reports_dir = Path(args.reports_dir)

    frames, quality_records = build_quarterly_frames(SERIES_SPECS, processed_dir)
    curated = merge_quarterly_frames(frames)
    curated = add_growth_rates(curated)
    curated = add_lag_features(curated)
    curated = add_nonlinear_elastic_net_terms(curated)
    curated = add_intervention_dummies(curated)
    curated = order_feature_columns(curated)

    quality_records.extend(validate_curated_dataset(curated))
    quality_records.append(validate_curated_with_pandera(curated))

    if not args.skip_duckdb:
        quality_records.append(
            write_duckdb(
                curated=curated,
                quality_report=records_to_frame(quality_records),
                db_path=Path(args.duckdb_path),
            )
        )

    if args.load_bigquery:
        quality_records.append(load_bigquery(curated))

    if args.load_postgres:
        quality_records.append(
            load_postgres_quality_report(records_to_frame(quality_records))
        )

    save_curated_outputs(curated, curated_dir)
    save_quality_report(quality_records, reports_dir)

    LOGGER.info(
        "Curated dataset complete: %s rows, %s columns",
        len(curated),
        len(curated.columns),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
