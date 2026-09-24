"""Tests for the optional Postgres load of the ETL data-quality report.

Postgres isn't available here, so ``sqlalchemy.create_engine`` is monkeypatched to
return one in-memory SQLite engine seeded with a simplified copy of
``sql/schema_app_metadata.sql``'s ``pipeline_runs``/``data_quality_results`` tables.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy import create_engine, text

from src.platform_loads import load_postgres_quality_report
from src.validation import QualityRecord


@pytest.fixture
def sqlite_engine(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    source_start_year INTEGER,
                    source_end_year INTEGER,
                    rows_curated INTEGER,
                    notes TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE data_quality_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT REFERENCES pipeline_runs(run_id),
                    dataset TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rows INTEGER,
                    columns INTEGER,
                    start_date TEXT,
                    end_date TEXT,
                    missing_values INTEGER,
                    duplicate_dates INTEGER,
                    notes TEXT
                )
                """
            )
        )
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *args, **kwargs: engine)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    return engine


def _quality_report() -> pd.DataFrame:
    def _record(dataset: str, status: str) -> dict[str, object]:
        return QualityRecord(
            dataset=dataset,
            status=status,
            rows=10,
            columns=2,
            start_date="1995Q1",
            end_date="2026Q4",
            missing_values=0,
            duplicate_dates=0,
            notes="ok",
        ).as_dict()

    return pd.DataFrame(
        [_record("raw:a", "PASS"), _record("raw:b", "PASS"), _record("raw:c", "WARNING")]
    )


def _curated() -> pd.DataFrame:
    return pd.DataFrame({"quarter": ["1995Q1", "1995Q2", "2026Q4"], "cpi_yoy": [1.0, 2.0, None]})


def test_load_with_curated_writes_a_pipeline_run_and_links_quality_rows(sqlite_engine):
    started_at = datetime(2026, 9, 24, 7, 0, 0, tzinfo=timezone.utc)

    record = load_postgres_quality_report(
        _quality_report(), curated=_curated(), started_at=started_at
    )

    assert record.status == "PASS"
    assert "etl-20260924T070000Z" in record.notes
    with sqlite_engine.connect() as conn:
        runs = pd.read_sql("SELECT * FROM pipeline_runs", conn)
        results = pd.read_sql("SELECT * FROM data_quality_results", conn)

    assert list(runs["run_id"]) == ["etl-20260924T070000Z"]
    assert runs.loc[0, "status"] == "completed"
    assert runs.loc[0, "source_start_year"] == 1995
    assert runs.loc[0, "source_end_year"] == 2026
    assert runs.loc[0, "rows_curated"] == 3
    assert runs.loc[0, "notes"] == "Data quality: 2 PASS, 1 WARNING, 0 FAIL."
    assert len(results) == 3
    assert set(results["run_id"]) == {"etl-20260924T070000Z"}


def test_load_without_curated_keeps_quality_rows_unlinked(sqlite_engine):
    load_postgres_quality_report(_quality_report())

    with sqlite_engine.connect() as conn:
        runs = pd.read_sql("SELECT * FROM pipeline_runs", conn)
        results = pd.read_sql("SELECT * FROM data_quality_results", conn)

    assert runs.empty
    assert len(results) == 3
    assert results["run_id"].isna().all()


def test_load_skips_when_database_url_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    record = load_postgres_quality_report(_quality_report(), curated=_curated())
    assert record.status == "WARNING"


def test_load_failure_is_recorded_not_raised_and_hides_the_error_detail(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secretpw@db.example.com/postgres")

    def _fail(*args, **kwargs):
        raise RuntimeError("connection to db.example.com failed for user:secretpw")

    monkeypatch.setattr(sqlalchemy, "create_engine", _fail)

    record = load_postgres_quality_report(_quality_report(), curated=_curated())

    assert record.status == "FAIL"
    assert "RuntimeError" in record.notes
    assert "secretpw" not in record.notes
    assert "example.com" not in record.notes


def test_load_failure_rolls_back_the_pipeline_run(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE pipeline_runs (run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, "
                "finished_at TEXT, status TEXT NOT NULL, source_start_year INTEGER, "
                "source_end_year INTEGER, rows_curated INTEGER, notes TEXT)"
            )
        )
        # Incompatible on purpose: the quality-row insert fails after the run row is written.
        conn.execute(text("CREATE TABLE data_quality_results (dataset TEXT)"))
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *args, **kwargs: engine)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")

    record = load_postgres_quality_report(_quality_report(), curated=_curated())

    assert record.status == "FAIL"
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM pipeline_runs")).scalar() == 0
