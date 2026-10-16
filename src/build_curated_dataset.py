"""Build the quarterly macroeconomic modelling dataset for CPI forecasting.

Run from the project root:

    python -m src.build_curated_dataset

Outputs:

    data/processed/*.csv
    data/curated/quarterly_macro_features.csv
    reports/data_quality_report.csv

Parquet output needs pyarrow, which requirements.txt installs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features import (
    add_growth_rates,
    add_intervention_dummies,
    add_lag_features,
    add_nonlinear_elastic_net_terms,
    order_feature_columns,
)
from src.platform_loads import write_duckdb
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


# One row per source series: (output column name, source file, value column in that file,
# how to turn it into one value per quarter).
SERIES_SPECS = [
    ("cpi_qoq", "dataset/abs/cpi_qoq_1995_2026.csv", "cpi_qoq", "last"),
    ("cpi_yoy", "dataset/abs/cpi_yoy_1995_2026.csv", "cpi_yoy", "last"),
    ("trimmed_mean_cpi_qoq", "dataset/abs/trimmed_mean_cpi_qoq_1995_2026.csv", "trimmed_mean_cpi_qoq", "last"),
    ("trimmed_mean_cpi_yoy", "dataset/abs/trimmed_mean_cpi_yoy_1995_2026.csv", "trimmed_mean_cpi_yoy", "last"),
    ("unemployment_rate", "dataset/abs/unemployment_rate_1995_2026.csv", "unemployment_rate", "mean"),
    ("cash_rate", "dataset/rba/cash_rate_1995_2026.csv", "cash_rate", "mean"),
    ("wage_price_index", "dataset/abs/wage_price_index_1995_2026.csv", "wage_price_index", "last"),
    ("producer_price_index", "dataset/abs/producer_price_index_1995_2026.csv", "producer_price_index", "last"),
    ("commodity_price_index", "dataset/rba/commodity_prices_1995_2026.csv", "GRCPAIAD", "mean"),
    ("wti_price", "dataset/market/wti_crude_oil_1995_2025.csv", "Close", "mean"),
    ("brent_price", "dataset/market/brent_crude_oil_1995_2025.csv", "Close", "mean"),
    ("aud_usd", "dataset/rba/aud_usd_exchange_rate_1995_2026.csv", "FXRUSD", "mean"),
    ("household_spending", "dataset/abs/household_spending_1995_2026.csv", "household_spending", "mean"),
    ("inflation_expectations_business", "dataset/rba/inflation_expectations_1995_2026.csv", "GBUSEXP", "mean"),
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
    return parser.parse_args()


def build_quarterly_frames(
    specs: list[tuple[str, str, str, str]],
    processed_dir: Path,
) -> tuple[list[pd.DataFrame], list[QualityRecord]]:
    """Read each source file, check it, convert it to quarters and save it; return the frames and checks."""
    frames = []
    quality_records = []

    for name, path, value_col, agg in specs:
        source_path = latest_dataset_path(Path(path))
        print(f"Processing {name} from {source_path}")
        source = read_csv(source_path)
        date_col = find_date_column(source)
        quality_records.append(
            validate_time_series(
                source,
                dataset=f"raw:{name}",
                date_col=date_col,
                value_cols=[value_col],
            )
        )

        quarterly = to_quarterly(source, value_col=value_col, output_col=name, agg=agg, date_col=date_col)
        save_processed_frame(quarterly, processed_dir, name)
        quality_records.append(
            validate_time_series(
                quarterly,
                dataset=f"processed:{name}",
                date_col="quarter",
                value_cols=[name],
            )
        )
        frames.append(quarterly)

    return frames, quality_records


def save_curated_outputs(df: pd.DataFrame, curated_dir: Path) -> None:
    curated_dir.mkdir(parents=True, exist_ok=True)
    csv_path = curated_dir / "quarterly_macro_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved curated CSV: {csv_path}")

    # SVAR and the RBA classifier read the Parquet copy, so a missing engine must stop the build
    # instead of leaving a stale file behind.
    parquet_path = curated_dir / "quarterly_macro_features.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved curated Parquet: {parquet_path}")


def records_to_frame(records: list[QualityRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.as_dict() for record in records])


def main() -> int:
    args = parse_args()

    frames, quality_records = build_quarterly_frames(SERIES_SPECS, Path(args.processed_dir))
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

    # Save the report first, then refuse to overwrite the curated data if any check failed.
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "data_quality_report.csv"
    records_to_frame(quality_records).to_csv(report_path, index=False)
    print(f"Saved data quality report: {report_path}")

    failed = [record.dataset for record in quality_records if record.status == "FAIL"]
    if failed:
        print(
            f"Data quality checks failed, curated data not written: {', '.join(failed)}. "
            "data/processed was already rewritten; run `git checkout data/processed` to undo it."
        )
        return 1
    save_curated_outputs(curated, Path(args.curated_dir))

    print(f"Curated dataset complete: {len(curated)} rows, {len(curated.columns)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
