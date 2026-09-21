"""The saved SQL queries run against the DuckDB database the ETL builds.

Nothing else in the project executes sql/queries/, so a renamed or dropped curated column would break them
silently. These tests build the database with the same loader the ETL uses, run every saved query, and check the
annual summary against an independent pandas calculation.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.models import svar
from src.platform_loads import write_duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_FILES = sorted((PROJECT_ROOT / "sql" / "queries").glob("*.sql"))

ANNUAL_COLUMNS = {
    "avg_cpi_yoy": "cpi_yoy",
    "avg_unemployment_rate": "unemployment_rate",
    "avg_cash_rate": "cash_rate",
    "avg_wpi_growth": "wpi_growth",
    "avg_ppi_growth": "ppi_growth",
}


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    curated = pd.read_csv(PROJECT_ROOT / "data" / "curated" / "quarterly_macro_features.csv")
    quality_report = pd.read_csv(PROJECT_ROOT / "reports" / "data_quality_report.csv")
    path = tmp_path_factory.mktemp("duckdb") / "cpi_forecast.duckdb"
    record = write_duckdb(curated=curated, quality_report=quality_report, db_path=path)
    assert record.status == "PASS"
    return path, curated


def _run(path: Path, sql: str) -> pd.DataFrame:
    with duckdb.connect(str(path), read_only=True) as connection:
        return connection.sql(sql).df()


def test_there_is_at_least_one_saved_query():
    assert QUERY_FILES, "sql/queries/ has no .sql files"


@pytest.mark.parametrize("query_file", QUERY_FILES, ids=lambda path: path.name)
def test_every_saved_query_runs_and_returns_rows(database, query_file):
    path, _ = database
    result = _run(path, query_file.read_text(encoding="utf-8"))
    assert len(result) > 0


def test_annual_inflation_summary_matches_a_pandas_groupby(database):
    path, curated = database
    result = _run(path, (PROJECT_ROOT / "sql" / "queries" / "annual_inflation_summary.sql").read_text(encoding="utf-8"))
    assert list(result.columns) == ["year", *ANNUAL_COLUMNS]
    assert result["year"].is_monotonic_increasing and result["year"].is_unique

    expected = (
        curated.assign(year=curated["quarter"].str[:4].astype(int))
        .groupby("year")[list(ANNUAL_COLUMNS.values())]
        .mean()
        .rename(columns={source: name for name, source in ANNUAL_COLUMNS.items()})
    )
    pd.testing.assert_frame_equal(result.set_index("year"), expected, check_dtype=False, check_index_type=False, rtol=1e-9)


PIN_FILTERED_QUERIES = ["surge_vs_before_and_after.sql", "yearly_change_in_inflation.sql"]


def _within_pin(curated: pd.DataFrame) -> pd.DataFrame:
    return curated[curated["quarter"] <= str(svar.FORECAST_ORIGIN_PIN)].reset_index(drop=True)


@pytest.mark.parametrize("name", PIN_FILTERED_QUERIES)
def test_pin_filtered_queries_use_the_real_forecast_origin_pin(name):
    sql = (PROJECT_ROOT / "sql" / "queries" / name).read_text(encoding="utf-8")
    assert f"quarter <= '{svar.FORECAST_ORIGIN_PIN}'" in sql, f"{name} no longer stops at the forecast-origin pin"


def test_surge_vs_before_and_after_matches_a_pandas_groupby(database):
    path, curated = database
    result = _run(path, (PROJECT_ROOT / "sql" / "queries" / "surge_vs_before_and_after.sql").read_text(encoding="utf-8"))
    assert result["period"].tolist() == ["before", "surge", "after"]
    assert result["quarters"].tolist() == [100, 16, 8]

    frame = _within_pin(curated)
    period = pd.Series("after", index=frame.index)
    period[frame["quarter"] <= "2023Q4"] = "surge"
    period[frame["quarter"] <= "2019Q4"] = "before"
    expected = frame.groupby(period)[
        ["cpi_yoy", "cash_rate", "unemployment_rate", "inflation_expectations_business"]
    ].mean()
    got = result.set_index("period")[
        ["avg_cpi_yoy", "avg_cash_rate", "avg_unemployment_rate", "avg_business_inflation_expectations"]
    ]
    assert got.to_numpy() == pytest.approx(expected.loc[["before", "surge", "after"]].to_numpy(), rel=1e-9)


def test_yearly_change_in_inflation_matches_a_pandas_diff(database):
    path, curated = database
    result = _run(path, (PROJECT_ROOT / "sql" / "queries" / "yearly_change_in_inflation.sql").read_text(encoding="utf-8"))
    frame = _within_pin(curated).sort_values("quarter").reset_index(drop=True)
    assert result["quarter"].tolist() == frame["quarter"].tolist()
    pd.testing.assert_series_equal(
        result["cpi_yoy_change_1y"], frame["cpi_yoy"].diff(4), check_names=False, check_dtype=False, rtol=1e-9
    )
    pd.testing.assert_series_equal(
        result["cash_rate_change_1y"], frame["cash_rate"].diff(4), check_names=False, check_dtype=False, rtol=1e-9
    )
