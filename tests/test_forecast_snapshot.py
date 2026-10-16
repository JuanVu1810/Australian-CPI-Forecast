"""Tests for the optional forecast snapshot example, using SQLite instead of Supabase."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

from src.models import forecast_snapshot


def _rows(actual=None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": "headline",
                "model": "sarima",
                "forecast_origin": "2025Q4",
                "target_quarter": "2026Q1",
                "point_forecast": 3.5,
                "actual": actual,
            }
        ]
    )


def test_saving_the_same_snapshot_twice_keeps_one_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")

    forecast_snapshot.save_snapshots(engine, _rows())
    forecast_snapshot.save_snapshots(engine, _rows())

    saved = pd.read_sql("SELECT * FROM forecast_snapshots", engine)
    assert len(saved) == 1
    assert saved.loc[0, "point_forecast"] == 3.5


def test_forecast_error_query_compares_with_the_known_actual(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    forecast_snapshot.save_snapshots(engine, _rows())  # actual not published yet
    assert forecast_snapshot.read_forecast_errors(engine).empty

    forecast_snapshot.save_snapshots(engine, _rows(actual=4.0))  # a rerun fills in the actual
    errors = forecast_snapshot.read_forecast_errors(engine)

    assert len(errors) == 1
    assert errors.loc[0, "point_forecast"] == 3.5  # the saved forecast is not overwritten
    assert errors.loc[0, "error"] == 0.5
    assert errors.loc[0, "abs_error"] == 0.5


def test_main_without_credentials_saves_nothing_and_explains(monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    forecast_snapshot.main()

    assert "DATABASE_URL is not set" in capsys.readouterr().out
