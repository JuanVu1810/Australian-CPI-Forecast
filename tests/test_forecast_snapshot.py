"""Tests for forecast snapshot persistence and actuals comparison.

Postgres isn't available in this environment, so these tests build a small
SQLite test double: ``sqlalchemy.create_engine`` is monkeypatched to always
return one in-memory engine pre-seeded with a simplified version of
``sql/schema_app_metadata.sql``'s ``model_metrics``/``forecast_results`` tables,
so INSERT/SELECT round-trip logic is actually exercised.
"""

from __future__ import annotations

import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy import create_engine, text

from api import main as api_main
from src.models import forecast_snapshot


def _fake_response(
    quarter: str = "2026Q3", shift: float = 0.0
) -> api_main.AllForecastsResponse:
    def _family(name: str, forecast: float, lower: float, upper: float) -> api_main.FamilyForecast:
        return api_main.FamilyForecast(
            model_family=name,
            run_id=f"mlflow-run-{name}",
            horizon=1,
            horizon_cap=None,
            forecast=[forecast + shift],
            interval_lower=[lower + shift],
            interval_upper=[upper + shift],
            empirical_coverage=[None],
            coverage_n=[None],
            significantly_miscalibrated=[None],
            quarters=[quarter],
            forecast_origin="2026Q2",
        )

    return api_main.AllForecastsResponse(
        requested_horizon=1,
        models=[
            _family("sarima", 5.0, 4.0, 6.0),
            _family("elastic_net", 5.2, 4.2, 6.2),
        ],
        unavailable=[
            api_main.UnavailableFamily(
                model_family="ensemble", reason="No finished MLflow run found"
            ),
        ],
    )


TRIMMED_MEAN_SHIFT = -1.0


def _patch_api(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "forecast_all", lambda request: _fake_response())
    monkeypatch.setattr(
        api_main,
        "forecast_trimmed_mean_all",
        lambda request: _fake_response(shift=TRIMMED_MEAN_SHIFT),
    )


@pytest.fixture
def sqlite_engine(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE model_metrics (
                    run_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    run_date TEXT NOT NULL,
                    forecast_horizon INTEGER NOT NULL,
                    feature_set TEXT,
                    rmse REAL,
                    mae REAL,
                    mse REAL,
                    selected_model BOOLEAN DEFAULT 0
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE forecast_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT REFERENCES model_metrics(run_id),
                    quarter TEXT NOT NULL,
                    forecast REAL NOT NULL,
                    lower_ci REAL,
                    upper_ci REAL
                )
                """
            )
        )
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *args, **kwargs: engine)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    return engine


def test_snapshot_forecasts_skips_gracefully_when_database_url_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    summary = forecast_snapshot.snapshot_forecasts()
    assert summary.status == "unconfigured"
    assert summary.inserted == []
    assert summary.skipped == []


def test_compare_forecast_snapshots_skips_gracefully_when_database_url_unset(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    output_path = tmp_path / "accuracy.csv"
    accuracy = forecast_snapshot.compare_forecast_snapshots(output_path=output_path)
    assert isinstance(accuracy, pd.DataFrame)
    assert accuracy.empty
    assert list(accuracy.columns) == forecast_snapshot.COMPARISON_COLUMNS
    assert not output_path.exists()


def test_snapshot_forecasts_inserts_available_families(monkeypatch, sqlite_engine):
    _patch_api(monkeypatch)

    summary = forecast_snapshot.snapshot_forecasts()

    assert summary.status == "completed"
    assert sorted(summary.inserted) == [
        "elastic_net",
        "elastic_net_trimmed_mean",
        "sarima",
        "sarima_trimmed_mean",
    ]
    assert summary.skipped == []
    assert sorted(summary.unavailable) == ["ensemble", "ensemble_trimmed_mean"]

    with sqlite_engine.connect() as conn:
        metrics = pd.read_sql("SELECT * FROM model_metrics ORDER BY model_name", conn)
        results = pd.read_sql("SELECT * FROM forecast_results ORDER BY run_id", conn)

    assert sorted(metrics["run_id"]) == [
        "elastic_net:2026Q3",
        "elastic_net_trimmed_mean:2026Q3",
        "sarima:2026Q3",
        "sarima_trimmed_mean:2026Q3",
    ]
    selected = dict(zip(metrics["model_name"], metrics["selected_model"]))
    assert bool(selected["sarima"]) is False
    assert bool(selected["elastic_net"]) is False
    assert set(metrics["forecast_horizon"]) == {1}

    forecasts = dict(zip(results["run_id"], results["forecast"]))
    assert forecasts["sarima:2026Q3"] == pytest.approx(5.0)
    assert forecasts["elastic_net:2026Q3"] == pytest.approx(5.2)
    assert forecasts["sarima_trimmed_mean:2026Q3"] == pytest.approx(5.0 + TRIMMED_MEAN_SHIFT)
    assert set(results["quarter"]) == {"2026Q3"}


def test_snapshot_forecasts_is_idempotent_for_same_quarter(monkeypatch, sqlite_engine):
    _patch_api(monkeypatch)

    first = forecast_snapshot.snapshot_forecasts()
    second = forecast_snapshot.snapshot_forecasts()

    assert len(first.inserted) == 4
    assert second.inserted == []
    assert sorted(second.skipped) == sorted(first.inserted)

    with sqlite_engine.connect() as conn:
        metrics_count = conn.execute(text("SELECT COUNT(*) FROM model_metrics")).scalar()
        results_count = conn.execute(text("SELECT COUNT(*) FROM forecast_results")).scalar()
    assert metrics_count == 4
    assert results_count == 4


def test_compare_forecast_snapshots_splits_observed_and_pending(
    monkeypatch, sqlite_engine, tmp_path
):
    _patch_api(monkeypatch)
    forecast_snapshot.snapshot_forecasts()

    with sqlite_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO model_metrics "
                "(run_id, model_name, run_date, forecast_horizon, selected_model) "
                "VALUES ('sarima:2099Q1', 'sarima', '2026-01-01T00:00:00', 1, 0)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO forecast_results (run_id, quarter, forecast, lower_ci, upper_ci) "
                "VALUES ('sarima:2099Q1', '2099Q1', 5.5, 4.5, 6.5)"
            )
        )

    quarters = pd.PeriodIndex(["2026Q3"], freq="Q")
    fake_actuals = {
        "cpi_yoy": pd.Series([5.1], index=quarters, name="cpi_yoy"),
        "trimmed_mean_cpi_yoy": pd.Series([4.3], index=quarters, name="trimmed_mean_cpi_yoy"),
    }
    monkeypatch.setattr(
        forecast_snapshot,
        "load_target_series",
        lambda path, target_column: fake_actuals[target_column],
    )

    output_path = tmp_path / "accuracy.csv"
    accuracy = forecast_snapshot.compare_forecast_snapshots(output_path=output_path)

    assert output_path.exists()
    assert len(accuracy) == 5
    assert list(accuracy.columns) == forecast_snapshot.COMPARISON_COLUMNS

    by_key = {
        (row.model_name, row.quarter): row for row in accuracy.itertuples()
    }

    sarima_row = by_key[("sarima", "2026Q3")]
    assert sarima_row.status == "observed"
    assert sarima_row.actual == pytest.approx(5.1)
    assert sarima_row.error == pytest.approx(5.1 - 5.0)
    assert bool(sarima_row.hit) is True
    assert sarima_row.target == "headline"

    elastic_net_row = by_key[("elastic_net", "2026Q3")]
    assert elastic_net_row.status == "observed"
    assert elastic_net_row.error == pytest.approx(5.1 - 5.2)
    assert bool(elastic_net_row.hit) is True

    pending_row = by_key[("sarima", "2099Q1")]
    assert pending_row.status == "pending"
    assert pd.isna(pending_row.actual)
    assert pd.isna(pending_row.error)

    trimmed_row = by_key[("sarima_trimmed_mean", "2026Q3")]
    assert trimmed_row.target == "trimmed_mean"
    assert trimmed_row.status == "observed"
    assert trimmed_row.actual == pytest.approx(4.3)
    assert trimmed_row.error == pytest.approx(4.3 - (5.0 + TRIMMED_MEAN_SHIFT))
    assert bool(trimmed_row.hit) is True


def _write_metrics_reports(tmp_path, monkeypatch) -> None:
    headline = tmp_path / "headline.csv"
    trimmed = tmp_path / "trimmed.csv"
    header = "model,horizon,n,rmse,mae\n"
    headline.write_text(
        header
        + "sarima,overall,424,7.7,7.7\nsarima,1,53,0.5,0.4\nsarima,2,53,9.9,9.9\n"
        + "elastic_net,1,53,1.0,0.8\n"
    )
    trimmed.write_text(
        header + "sarima,overall,424,7.7,7.7\nsarima,1,53,0.25,0.2\nelastic_net,1,53,0.5,0.4\n"
    )
    monkeypatch.setattr(
        forecast_snapshot,
        "METRICS_REPORT_PATHS",
        {"": headline, forecast_snapshot.TRIMMED_MEAN_SUFFIX: trimmed},
    )


def test_snapshot_forecasts_records_horizon_one_metrics_and_feature_set(
    monkeypatch, sqlite_engine, tmp_path
):
    _patch_api(monkeypatch)
    _write_metrics_reports(tmp_path, monkeypatch)

    forecast_snapshot.snapshot_forecasts()

    with sqlite_engine.connect() as conn:
        metrics = pd.read_sql("SELECT * FROM model_metrics", conn).set_index("model_name")

    assert metrics.loc["sarima", "rmse"] == pytest.approx(0.5)
    assert metrics.loc["sarima", "mae"] == pytest.approx(0.4)
    assert metrics.loc["sarima", "mse"] == pytest.approx(0.25)
    assert metrics.loc["sarima", "feature_set"] == "univariate (own lags only)"
    assert metrics.loc["elastic_net", "mse"] == pytest.approx(1.0)
    assert metrics.loc["sarima_trimmed_mean", "rmse"] == pytest.approx(0.25)
    assert metrics.loc["sarima_trimmed_mean", "mse"] == pytest.approx(0.0625)
    assert "cash_rate" in metrics.loc["elastic_net", "feature_set"]


def test_snapshot_forecasts_backfills_metrics_on_rows_written_without_them(
    monkeypatch, sqlite_engine, tmp_path
):
    _patch_api(monkeypatch)
    monkeypatch.setattr(
        forecast_snapshot,
        "METRICS_REPORT_PATHS",
        {"": tmp_path / "missing.csv", forecast_snapshot.TRIMMED_MEAN_SUFFIX: tmp_path / "missing.csv"},
    )
    forecast_snapshot.snapshot_forecasts()
    with sqlite_engine.connect() as conn:
        assert pd.read_sql("SELECT rmse FROM model_metrics", conn)["rmse"].isna().all()

    _write_metrics_reports(tmp_path, monkeypatch)
    second = forecast_snapshot.snapshot_forecasts()

    assert second.inserted == []
    assert len(second.backfilled) == 4
    third = forecast_snapshot.snapshot_forecasts()
    assert third.backfilled == []
    with sqlite_engine.connect() as conn:
        metrics = pd.read_sql("SELECT * FROM model_metrics", conn).set_index("model_name")
        results_count = conn.execute(text("SELECT COUNT(*) FROM forecast_results")).scalar()
    assert metrics.loc["sarima", "rmse"] == pytest.approx(0.5)
    assert metrics.loc["sarima_trimmed_mean", "mae"] == pytest.approx(0.2)
    assert results_count == 4


def test_horizon_one_metrics_load_from_the_committed_reports():
    for path in forecast_snapshot.METRICS_REPORT_PATHS.values():
        metrics = forecast_snapshot._load_horizon_one_metrics(path)
        assert {"sarima", "elastic_net", "ensemble"} <= set(metrics)
        assert all(rmse > 0 and mae > 0 for rmse, mae in metrics.values())
