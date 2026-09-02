"""Download Australian macroeconomic indicators for a chosen year range.

Required packages
-----------------
    python -m pip install "readabs>=0.2.5" pandas yfinance

Example
-------
    python data_retrieval.py 1995 2025
    python data_retrieval.py 2005 2024 --output-dir dataset

The start and end years are inclusive.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

import pandas as pd
import readabs as ra
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent

# Exact ABS series are used so the script does not download and save every
# series in each catalogue. These identify the intended Australia-wide series.
ABS_SERIES: dict[str, dict[str, str]] = {
    "cpi_qoq": {
        "catalogue": "6401.0",
        "series_id": "A3604507J",
        "single_excel_only": "64010Appendix1a",
        "title": (
            "Consumer Price Index: All groups CPI, seasonally adjusted, "
            "quarterly percentage change"
        ),
    },
    "cpi_yoy": {
        "catalogue": "6401.0",
        "series_id": "A3604508K",
        "single_excel_only": "64010Appendix1a",
        "title": (
            "Consumer Price Index: All groups CPI, seasonally adjusted, "
            "year-ended percentage change"
        ),
    },
    # Trimmed mean quarters near the October 2025 CPI compilation transition
    # may differ from contemporaneous ABS headline prints as the continuing
    # pre-October-2025-basis analytical series is revised.
    "trimmed_mean_cpi_qoq": {
        "catalogue": "6401.0",
        "series_id": "A3604510W",
        "single_excel_only": "64010Appendix1a",
        "title": "Consumer Price Index: Trimmed Mean, quarterly percentage change",
    },
    "trimmed_mean_cpi_yoy": {
        "catalogue": "6401.0",
        "series_id": "A3604511X",
        "single_excel_only": "64010Appendix1a",
        "title": "Consumer Price Index: Trimmed Mean, year-ended percentage change",
    },
    "unemployment_rate": {
        "catalogue": "6202.0",
        "series_id": "A84423050A",
        "single_excel_only": "62020001",
        "title": "Unemployment rate: Persons, Australia, seasonally adjusted",
    },
    "wage_price_index": {
        "catalogue": "6345.0",
        "series_id": "A2603609J",
        "single_excel_only": "634501",
        "title": (
            "Wage Price Index: Total hourly rates of pay excluding bonuses, "
            "private and public, all industries, Australia"
        ),
    },
    "producer_price_index": {
        "catalogue": "6427.0",
        "series_id": "A2314865F",
        "single_excel_only": "642701",
        "title": "Producer Price Index: Final demand, index number",
    },
    "household_spending": {
        "catalogue": "5682.0",
        "series_id": "A130200584T",
        "single_excel_only": "5682001",
        "title": (
            "Monthly Household Spending Indicator: Total household spending, "
            "Australia, current price, seasonally adjusted"
        ),
    },
}

# RBA table retrieval remains appropriate because G3 and I2 contain several
# related measures. The selector narrows F11 to AUD/USD and retains all useful
# matching measures from G3 and I2. If metadata labels change, the script saves
# the full requested RBA table rather than silently returning no data.
RBA_TABLES: dict[str, dict[str, Any]] = {
    "aud_usd_exchange_rate": {
        "table": "Z:F11.1-Monthly",  # Monthly exchange rates; F11.1 is the daily table.
        "title": "AUD/USD exchange rate",
        "series_ids": ("FXRUSD",),
        "keyword_groups": (
            ("united states", "dollar"),
            ("usd",),
        ),
        "allow_full_table_fallback": False,
    },
    "inflation_expectations": {
        "table": "G3",
        "title": "Inflation expectations",
        "series_ids": (),
        "keyword_groups": (
            ("inflation", "expect"),
            ("consumer", "inflation"),
            ("market", "inflation"),
        ),
        "allow_full_table_fallback": True,
    },
    "commodity_prices": {
        "table": "I2",
        "title": "RBA commodity price indexes",
        "series_ids": (),
        "keyword_groups": (
            ("commodity", "price"),
            ("index of commodity prices",),
        ),
        "allow_full_table_fallback": True,
    },
}

YFINANCE_TICKERS: dict[str, dict[str, str]] = {
    "wti_crude_oil": {
        "ticker": "CL=F",
        "title": "WTI crude oil futures",
    },
    "brent_crude_oil": {
        "ticker": "BZ=F",
        "title": "Brent crude oil futures",
    },
}

RBA_HISTORICAL_FORECASTS: dict[str, str] = {
    "cpi_by_horizon": "https://www.rba.gov.au/statistics/xls/cpi-by-horizon.xls",
}

APRA_QADIP_SOURCE_PAGE = (
    "https://www.apra.gov.au/news-and-publications/"
    "quarterly-authorised-deposit-taking-institution-statistics"
)
APRA_QADIP_WORKBOOK_URL = (
    "https://www.apra.gov.au/system/files/2026-06/"
    "Quarterly%20authorised%20deposit-taking%20institution%20performance-"
    "September%202004%20to%20March%202026.xlsx"
)
APRA_QADIP_RATIO_SHEET = "Tab 1g"
APRA_QADIP_RATIO_ROWS = (
    "Impaired facilities to loans and advances",
    "Non-performing to loans and advances",
)
APRA_CREDIT_QUALITY_START = pd.Period("2004Q3", freq="Q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download ABS, RBA and oil-market data using readabs, pandas "
            "and yfinance. The start and end years are inclusive."
        )
    )
    parser.add_argument("start_year", type=int, help="First year to retain.")
    parser.add_argument("end_year", type=int, help="Last year to retain.")
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Directory for downloaded CSV files. Default: dataset",
    )
    parser.add_argument(
        "--oil-tickers",
        nargs="*",
        choices=tuple(YFINANCE_TICKERS),
        default=list(YFINANCE_TICKERS),
        metavar="NAME",
        help=(
            "Oil series downloaded with yfinance. Choices: "
            f"{', '.join(YFINANCE_TICKERS)}. Default: both."
        ),
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not save ABS/RBA metadata CSV files.",
    )
    return parser.parse_args()


def validate_years(start_year: int, end_year: int) -> None:
    current_year = datetime.now().year

    if start_year < 1900:
        raise ValueError("start_year must be 1900 or later.")
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year.")
    if end_year > current_year:
        raise ValueError(
            f"end_year cannot be later than the current year ({current_year})."
        )


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MultiIndex columns, including yfinance output, to plain strings."""
    result = df.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            "_".join(str(part) for part in column if str(part) not in {"", "None"})
            for column in result.columns.to_flat_index()
        ]
    else:
        result.columns = [str(column) for column in result.columns]

    return result


def filter_years(
    data: pd.DataFrame | pd.Series,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Filter a time series by inclusive calendar years.

    The function deliberately examines only a PeriodIndex, DatetimeIndex, or a
    clearly named date column. It never attempts to parse arbitrary numeric
    value columns as dates.
    """
    result = data.to_frame(name=data.name or "value") if isinstance(data, pd.Series) else data.copy()

    if isinstance(result.index, pd.PeriodIndex):
        mask = (result.index.year >= start_year) & (result.index.year <= end_year)
        result = result.loc[mask]
        result.index = result.index.to_timestamp(how="start")
        result.index.name = "date"
        return flatten_columns(result.reset_index())

    if isinstance(result.index, pd.DatetimeIndex):
        mask = (result.index.year >= start_year) & (result.index.year <= end_year)
        result = result.loc[mask]
        result.index.name = result.index.name or "date"
        return flatten_columns(result.reset_index())

    date_column = next(
        (
            column
            for column in result.columns
            if str(column).strip().lower() in {"date", "time", "period"}
        ),
        None,
    )
    if date_column is None:
        raise ValueError(
            "No PeriodIndex, DatetimeIndex, or clearly named date column was found."
        )

    dates = pd.to_datetime(result[date_column], errors="raise")
    mask = dates.dt.year.between(start_year, end_year, inclusive="both")
    result = result.loc[mask].copy()
    result[date_column] = dates.loc[mask]
    return flatten_columns(result.reset_index(drop=True))


def save_csv(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return {"file": str(path), "rows": len(df), "columns": list(df.columns)}


def save_metadata(metadata: Any, path: Path) -> dict[str, Any] | None:
    """Save metadata without applying the requested observation-year filter."""
    if metadata is None:
        return None

    if isinstance(metadata, pd.Series):
        metadata = metadata.to_frame(name=metadata.name or "value")
    if not isinstance(metadata, pd.DataFrame):
        return None

    return save_csv(flatten_columns(metadata.reset_index(drop=True)), path)


def download_binary(url: str, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        payload = response.read()
    path.write_bytes(payload)
    return {"file": str(path), "bytes": len(payload), "url": url}


def standardise_single_series(
    data: pd.DataFrame | pd.Series,
    variable_name: str,
) -> pd.DataFrame | pd.Series:
    if isinstance(data, pd.Series):
        return data.rename(variable_name)

    result = data.copy()
    if result.shape[1] == 1:
        result.columns = [variable_name]
    return result


def horizon_label_to_quarters(label: Any) -> int:
    text = str(label).strip().lower()
    if text == "t":
        return 0
    if text.startswith("t+"):
        return int(text.removeprefix("t+"))
    if text.startswith("t-"):
        return -int(text.removeprefix("t-"))
    raise ValueError(f"Unrecognised forecast horizon label: {label}")


def parse_rba_cpi_by_horizon_workbook(path: Path) -> pd.DataFrame:
    """Return tidy RBA CPI inflation forecasts, actuals and errors by horizon."""

    def parse_sheet(sheet_name: str, value_col: str) -> pd.DataFrame:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        header_matches = raw.index[
            raw.iloc[:, 0].astype(str).str.strip().str.lower().eq("forecast date")
        ]
        if header_matches.empty:
            raise ValueError(f"Could not find 'Forecast date' header in {sheet_name}.")

        header_row = int(header_matches[0])
        headers = raw.iloc[header_row].tolist()
        data = raw.iloc[header_row + 1 :].copy()
        data.columns = headers
        data = data.dropna(how="all")

        first_col = data.columns[0]
        data = data.rename(columns={first_col: "forecast_date"})
        data["forecast_date"] = pd.to_datetime(data["forecast_date"], errors="coerce")
        data = data.loc[data["forecast_date"].notna()].copy()

        if "Source" not in data.columns:
            data["Source"] = pd.NA

        horizon_cols = [
            column
            for column in data.columns
            if str(column).strip().lower().startswith("t")
        ]
        tidy = data.melt(
            id_vars=["forecast_date", "Source"],
            value_vars=horizon_cols,
            var_name="horizon_label",
            value_name=value_col,
        )
        tidy = tidy.rename(columns={"Source": "source"})
        tidy["horizon_quarters"] = tidy["horizon_label"].map(horizon_label_to_quarters)
        tidy[value_col] = pd.to_numeric(tidy[value_col], errors="coerce")
        return tidy

    forecasts = parse_sheet("Forecasts", "rba_forecast_cpi_yoy")
    actuals = parse_sheet("Actuals", "rba_actual_cpi_yoy").drop(columns="source")
    errors = parse_sheet("Errors", "rba_forecast_error_cpi_yoy").drop(columns="source")

    keys = ["forecast_date", "horizon_label", "horizon_quarters"]
    tidy = forecasts.merge(actuals, on=keys, how="outer").merge(errors, on=keys, how="outer")
    tidy = tidy.sort_values(["forecast_date", "horizon_quarters"]).reset_index(drop=True)
    tidy["forecast_date"] = tidy["forecast_date"].dt.date.astype(str)
    return tidy[
        [
            "forecast_date",
            "source",
            "horizon_label",
            "horizon_quarters",
            "rba_forecast_cpi_yoy",
            "rba_actual_cpi_yoy",
            "rba_forecast_error_cpi_yoy",
        ]
    ]


def parse_apra_credit_quality_workbook(path: Path) -> pd.DataFrame:
    """Return APRA's aggregate ADI credit-quality ratio as a quarterly series."""

    def parse_quarter_header(value: Any) -> pd.Timestamp:
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, (datetime, pd.Timestamp)):
            return pd.Timestamp(value)
        return pd.to_datetime(str(value).strip(), format="%b %Y", errors="coerce")

    raw = pd.read_excel(path, sheet_name=APRA_QADIP_RATIO_SHEET, header=None)
    quarter_header_matches = raw.index[
        raw.iloc[:, 1].astype(str).str.strip().str.lower().eq("quarter end")
    ]
    if quarter_header_matches.empty:
        raise ValueError(
            f"Could not find 'Quarter end' header in {APRA_QADIP_RATIO_SHEET}."
        )

    date_row = int(quarter_header_matches[0]) + 1
    quarter_dates = raw.iloc[date_row, 1:].map(parse_quarter_header)
    if quarter_dates.isna().all():
        raise ValueError(
            f"Could not parse quarter dates in {APRA_QADIP_RATIO_SHEET}."
        )

    row_labels = raw.iloc[:, 0].astype(str).str.strip()
    frames: list[pd.DataFrame] = []
    for label in APRA_QADIP_RATIO_ROWS:
        row_matches = row_labels.str.lower().eq(label.lower())
        match_count = int(row_matches.sum())
        if not match_count:
            raise ValueError(
                f"Could not find '{label}' row in {APRA_QADIP_RATIO_SHEET}."
            )
        if match_count > 1:
            raise ValueError(
                f"Found {match_count} '{label}' rows in {APRA_QADIP_RATIO_SHEET}; "
                "expected a unique aggregate ADI ratio row."
            )

        row_number = int(row_matches[row_matches].index[0])
        values = pd.to_numeric(raw.iloc[row_number, 1:], errors="coerce")
        frame = pd.DataFrame(
            {
                "quarter_date": quarter_dates,
                "credit_quality_ratio_percent": values.to_numpy() * 100,
                "source_measure": label,
                "source_sheet": APRA_QADIP_RATIO_SHEET,
            }
        ).dropna(subset=["quarter_date", "credit_quality_ratio_percent"])
        frames.append(frame)

    tidy = pd.concat(frames, ignore_index=True)
    tidy["quarter_period"] = tidy["quarter_date"].dt.to_period("Q")
    tidy = tidy.loc[tidy["quarter_period"] >= APRA_CREDIT_QUALITY_START].copy()
    tidy["quarter"] = tidy["quarter_period"].astype(str)
    tidy["measurement_basis"] = tidy["source_measure"].map(
        {
            "Impaired facilities to loans and advances": (
                "impaired facilities to loans and advances"
            ),
            "Non-performing to loans and advances": (
                "non-performing exposures to loans and advances"
            ),
        }
    )
    tidy["measurement_note"] = tidy["source_measure"].map(
        {
            "Impaired facilities to loans and advances": (
                "APRA impaired-facilities measure used before the March 2022 "
                "APS 220 publication update."
            ),
            "Non-performing to loans and advances": (
                "APRA non-performing measure used from the March 2022 APS 220 "
                "publication update; compare with earlier impaired-facilities "
                "values with care."
            ),
        }
    )
    tidy = tidy.sort_values(["quarter_period", "source_measure"]).reset_index(
        drop=True
    )
    duplicate_quarters = tidy["quarter"].duplicated(keep=False)
    if duplicate_quarters.any():
        duplicated = ", ".join(tidy.loc[duplicate_quarters, "quarter"].unique())
        raise ValueError(f"Multiple APRA credit-quality measures for: {duplicated}.")

    return tidy[
        [
            "quarter",
            "credit_quality_ratio_percent",
            "source_measure",
            "measurement_basis",
            "measurement_note",
            "source_sheet",
        ]
    ]


def describe_apra_credit_quality_breaks(tidy: pd.DataFrame) -> pd.DataFrame:
    """Create plain-language metadata notes for known and checked breaks."""
    quarters = pd.PeriodIndex(tidy["quarter"], freq="Q")
    values = pd.Series(tidy["credit_quality_ratio_percent"].to_numpy(), index=quarters)
    aasb9_window = values.loc["2017Q1":"2018Q4"]
    qoq_changes = aasb9_window.diff().abs().dropna()
    max_qoq_change = float(qoq_changes.max()) if not qoq_changes.empty else 0.0
    y2017 = values.loc["2017Q1":"2017Q4"]
    y2018 = values.loc["2018Q1":"2018Q4"]
    mean_shift = (
        abs(float(y2018.mean() - y2017.mean()))
        if len(y2017) and len(y2018)
        else 0.0
    )
    visible_aasb9_break = max_qoq_change >= 0.25 or mean_shift >= 0.25
    aasb9_note = (
        "A visible level shift is present around the 2018 AASB 9 expected-credit-loss "
        f"transition: maximum quarter-to-quarter change was {max_qoq_change:.2f} "
        f"percentage points and the 2018 mean differed from 2017 by {mean_shift:.2f} "
        "percentage points."
        if visible_aasb9_break
        else (
            "No visible level shift was found around the 2018 AASB 9 "
            f"expected-credit-loss transition: maximum quarter-to-quarter change "
            f"was {max_qoq_change:.2f} percentage points and the 2018 mean "
            f"differed from 2017 by {mean_shift:.2f} percentage points."
        )
    )

    return pd.DataFrame(
        [
            {
                "item": "confirmed_workbook",
                "note": (
                    "Quarterly authorised deposit-taking institution performance-"
                    "September 2004 to March 2026.xlsx"
                ),
                "source_url": APRA_QADIP_WORKBOOK_URL,
            },
            {
                "item": "confirmed_sheet_and_rows",
                "note": (
                    f"Sheet '{APRA_QADIP_RATIO_SHEET}' uses row labels "
                    f"'{APRA_QADIP_RATIO_ROWS[0]}' and "
                    f"'{APRA_QADIP_RATIO_ROWS[1]}' for the aggregate ADI ratio."
                ),
                "source_url": APRA_QADIP_WORKBOOK_URL,
            },
            {
                "item": "aasb9_2018_check",
                "note": aasb9_note,
                "source_url": APRA_QADIP_WORKBOOK_URL,
            },
            {
                "item": "aps_220_2022_definition_change",
                "note": (
                    "APRA's explanatory notes say asset-quality data were updated "
                    "from the March 2022 reference period, replacing the concept "
                    "of impaired with non-performing; this output therefore marks "
                    "the measurement basis for each quarter."
                ),
                "source_url": APRA_QADIP_WORKBOOK_URL,
            },
        ]
    )


def download_apra_credit_quality(
    output_dir: Path,
    start_year: int,
    end_year: int,
    save_meta: bool = True,
) -> dict[str, Any]:
    """Download APRA QADIP and save the aggregate ADI credit-quality ratio."""
    apra_dir = output_dir / "apra"
    workbook_path = apra_dir / "apra_qadip_performance_sep2004_mar2026.xlsx"

    print("APRA: Quarterly ADI Performance credit quality", flush=True)
    try:
        workbook_record = download_binary(APRA_QADIP_WORKBOOK_URL, workbook_path)
        tidy = parse_apra_credit_quality_workbook(workbook_path)
    except Exception as exc:
        raise RuntimeError(f"Could not retrieve APRA credit-quality data: {exc}") from exc

    quarters = pd.PeriodIndex(tidy["quarter"], freq="Q")
    filtered = tidy.loc[
        (quarters.year >= start_year) & (quarters.year <= end_year)
    ].copy()
    curated_dir = PROJECT_ROOT / "data" / "curated"
    metadata_dir = PROJECT_ROOT / "data" / "metadata"

    data_record = save_csv(filtered, curated_dir / "credit_quality_quarterly.csv")
    metadata_record = None
    if save_meta:
        notes = describe_apra_credit_quality_breaks(tidy)
        metadata_record = save_csv(
            notes,
            metadata_dir / "credit_quality_quarterly_notes.csv",
        )

    return {
        "variable": "credit_quality_quarterly",
        "title": "APRA aggregate ADI impaired/non-performing loan ratio",
        "source_page": APRA_QADIP_SOURCE_PAGE,
        "source_url": APRA_QADIP_WORKBOOK_URL,
        "raw_workbook": workbook_record,
        "data": data_record,
        "metadata": metadata_record,
    }


def download_rba_historical_forecasts(
    output_dir: Path,
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    """Download and tidy RBA historical CPI inflation forecasts by horizon."""
    rba_dir = output_dir / "rba"
    url = RBA_HISTORICAL_FORECASTS["cpi_by_horizon"]
    workbook_path = rba_dir / "rba_historical_cpi_forecasts_by_horizon.xls"

    print("RBA Historical Forecasts: CPI inflation (year-ended)", flush=True)
    try:
        workbook_record = download_binary(url, workbook_path)
        tidy = parse_rba_cpi_by_horizon_workbook(workbook_path)
    except Exception as exc:
        raise RuntimeError(f"Could not retrieve RBA historical CPI forecasts: {exc}") from exc

    forecast_dates = pd.to_datetime(tidy["forecast_date"], errors="raise")
    filtered = tidy.loc[forecast_dates.dt.year.between(start_year, end_year)].copy()
    data_record = save_csv(
        filtered,
        rba_dir / f"rba_historical_cpi_forecasts_by_horizon_{start_year}_{end_year}.csv",
    )

    return {
        "variable": "rba_historical_cpi_forecasts_by_horizon",
        "title": "RBA historical forecasts for CPI inflation, year-ended",
        "source_url": url,
        "raw_workbook": workbook_record,
        "data": data_record,
    }


def download_abs_series(
    output_dir: Path,
    start_year: int,
    end_year: int,
    save_meta: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    abs_dir = output_dir / "abs"

    for variable_name, source in ABS_SERIES.items():
        print(f"ABS: {source['title']}", flush=True)
        try:
            read_kwargs: dict[str, Any] = {
                # Continue past non-time-series auxiliary workbooks that may
                # appear on an ABS publication page.
                "ignore_errors": True,
            }
            if source.get("single_excel_only"):
                read_kwargs["single_excel_only"] = source["single_excel_only"]

            data, metadata = ra.read_abs_series(
                cat=source["catalogue"],
                series_id=source["series_id"],
                **read_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not retrieve ABS series {source['series_id']} "
                f"from catalogue {source['catalogue']}: {exc}"
            ) from exc

        data = standardise_single_series(data, variable_name)
        filtered = filter_years(data, start_year, end_year)
        data_record = save_csv(
            filtered,
            abs_dir / f"{variable_name}_{start_year}_{end_year}.csv",
        )

        metadata_record = None
        if save_meta:
            metadata_record = save_metadata(
                metadata,
                abs_dir / f"{variable_name}_metadata.csv",
            )

        records.append(
            {
                "variable": variable_name,
                **source,
                "data": data_record,
                "metadata": metadata_record,
            }
        )

    return records


def _normalise(value: Any) -> str:
    return " ".join(str(value).lower().replace("_", " ").split())


def _contains_keyword_group(text: str, groups: Iterable[Iterable[str]]) -> bool:
    return any(all(_normalise(term) in text for term in group) for group in groups)


def _column_lookup(df: pd.DataFrame) -> dict[str, Any]:
    return {_normalise(column): column for column in df.columns}


def select_rba_columns(
    data: pd.DataFrame | pd.Series,
    metadata: Any,
    series_ids: Iterable[str],
    keyword_groups: Iterable[Iterable[str]],
    allow_full_table_fallback: bool,
) -> pd.DataFrame:
    """Select relevant RBA columns using IDs, metadata text, then labels.

    readabs/RBA metadata layouts can vary by table. This function does not rely
    on a hard-coded metadata column name; instead, it detects metadata values
    that match actual data-column IDs.
    """
    frame = data.to_frame(name=data.name or "value") if isinstance(data, pd.Series) else data.copy()
    frame = flatten_columns(frame)
    lookup = _column_lookup(frame)
    selected: list[Any] = []

    # First preference: known exact RBA series identifiers.
    requested_ids = {_normalise(series_id) for series_id in series_ids}
    selected.extend(lookup[series_id] for series_id in requested_ids if series_id in lookup)

    # Second preference: metadata rows whose text matches the requested concept.
    if isinstance(metadata, pd.DataFrame) and not metadata.empty:
        meta = flatten_columns(metadata.reset_index(drop=True))
        for _, row in meta.iterrows():
            row_values = [_normalise(value) for value in row.tolist()]
            row_text = " | ".join(row_values)
            if not _contains_keyword_group(row_text, keyword_groups):
                continue

            for value in row_values:
                if value in lookup:
                    selected.append(lookup[value])

    # Third preference: descriptive column labels.
    for normalised, original in lookup.items():
        if _contains_keyword_group(normalised, keyword_groups):
            selected.append(original)

    # Preserve source order and remove duplicates.
    selected_set = set(selected)
    selected = [column for column in frame.columns if column in selected_set]

    if selected:
        return frame[selected]
    if allow_full_table_fallback:
        print(
            "  Warning: no unique RBA columns matched; saving the full table.",
            flush=True,
        )
        return frame

    raise ValueError(
        "No matching RBA series was found. Inspect the saved/returned RBA "
        "metadata or run readabs.print_rba_catalogue() to confirm the table."
    )


def download_rba_data(
    output_dir: Path,
    start_year: int,
    end_year: int,
    save_meta: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rba_dir = output_dir / "rba"

    print("RBA: Official Cash Rate", flush=True)
    try:
        cash_rate = ra.read_rba_ocr(monthly=True)
    except Exception as exc:
        raise RuntimeError(f"Could not retrieve the RBA cash rate: {exc}") from exc

    cash_rate = standardise_single_series(cash_rate, "cash_rate")
    cash_filtered = filter_years(cash_rate, start_year, end_year)
    records.append(
        {
            "variable": "cash_rate",
            "table": "OCR",
            "title": "Official Cash Rate, monthly",
            "data": save_csv(
                cash_filtered,
                rba_dir / f"cash_rate_{start_year}_{end_year}.csv",
            ),
            "metadata": None,
        }
    )

    for variable_name, source in RBA_TABLES.items():
        print(f"RBA {source['table']}: {source['title']}", flush=True)
        try:
            data, metadata = ra.read_rba_table(source["table"])
            selected = select_rba_columns(
                data=data,
                metadata=metadata,
                series_ids=source["series_ids"],
                keyword_groups=source["keyword_groups"],
                allow_full_table_fallback=source["allow_full_table_fallback"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not retrieve/select RBA table {source['table']}: {exc}"
            ) from exc

        filtered = filter_years(selected, start_year, end_year)
        data_record = save_csv(
            filtered,
            rba_dir / f"{variable_name}_{start_year}_{end_year}.csv",
        )

        metadata_record = None
        if save_meta:
            metadata_record = save_metadata(
                metadata,
                rba_dir / f"{variable_name}_metadata.csv",
            )

        records.append(
            {
                "variable": variable_name,
                "table": source["table"],
                "title": source["title"],
                "data": data_record,
                "metadata": metadata_record,
            }
        )

    records.append(
        download_rba_historical_forecasts(
            output_dir=output_dir,
            start_year=start_year,
            end_year=end_year,
        )
    )

    return records


def download_yfinance_data(
    output_dir: Path,
    start_year: int,
    end_year: int,
    selected_names: Iterable[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    market_dir = output_dir / "market"

    # yfinance start is inclusive and end is exclusive, so using 1 January of
    # the following year includes every observation from end_year.
    start_date = f"{start_year}-01-01"
    end_exclusive = f"{end_year + 1}-01-01"

    for variable_name in selected_names:
        source = YFINANCE_TICKERS[variable_name]
        print(f"yfinance {source['ticker']}: {source['title']}", flush=True)
        try:
            data = yf.download(
                source["ticker"],
                start=start_date,
                end=end_exclusive,
                interval="1d",
                auto_adjust=False,
                progress=False,
                multi_level_index=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not retrieve yfinance ticker {source['ticker']}: {exc}"
            ) from exc

        if data is None or data.empty:
            raise RuntimeError(
                f"yfinance returned no observations for {source['ticker']} "
                f"between {start_date} and {end_exclusive}."
            )

        # The API dates already use the requested bounds, but filtering again
        # provides a consistent inclusive-year guarantee.
        filtered = filter_years(data, start_year, end_year)
        records.append(
            {
                "variable": variable_name,
                **source,
                "data": save_csv(
                    filtered,
                    market_dir / f"{variable_name}_{start_year}_{end_year}.csv",
                ),
            }
        )

    return records


def main() -> int:
    args = parse_args()

    try:
        validate_years(args.start_year, args.end_year)
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "packages": {
            "pandas": pd.__version__,
            "readabs": getattr(ra, "__version__", "unknown"),
            "yfinance": getattr(yf, "__version__", "unknown"),
        },
        "start_year": args.start_year,
        "end_year": args.end_year,
        "abs": [],
        "rba": [],
        "apra": [],
        "yfinance": [],
    }

    try:
        manifest["abs"] = download_abs_series(
            output_dir=output_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            save_meta=not args.no_metadata,
        )
        manifest["rba"] = download_rba_data(
            output_dir=output_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            save_meta=not args.no_metadata,
        )
        manifest["apra"] = download_apra_credit_quality(
            output_dir=output_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            save_meta=not args.no_metadata,
        )
        manifest["yfinance"] = download_yfinance_data(
            output_dir=output_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            selected_names=args.oil_tickers,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nFinished. Manifest saved to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
