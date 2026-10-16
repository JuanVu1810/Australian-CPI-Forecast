import sys

import pandas as pd

from src import build_curated_dataset as build
from src.validation import QualityRecord


def _record(dataset, status):
    return QualityRecord(
        dataset=dataset, status=status, rows=1, columns=2, start_date="2020Q1", end_date="2020Q1",
        missing_values=0, duplicate_dates=0, notes="",
    )


def _run_build(monkeypatch, tmp_path, curated_status):
    frame = pd.DataFrame({"quarter": ["2020Q1"], "cpi_yoy": [2.0]})
    monkeypatch.setattr(build, "build_quarterly_frames", lambda specs, processed_dir: ([frame], []))
    for name in (
        "add_growth_rates", "add_lag_features", "add_nonlinear_elastic_net_terms",
        "add_intervention_dummies", "order_feature_columns",
    ):
        monkeypatch.setattr(build, name, lambda df: df)
    monkeypatch.setattr(build, "validate_curated_dataset", lambda df: [_record("curated", curated_status)])
    monkeypatch.setattr(build, "validate_curated_with_pandera", lambda df: _record("pandera", "PASS"))
    monkeypatch.setattr(
        sys, "argv",
        [
            "build", "--skip-duckdb", "--processed-dir", str(tmp_path / "processed"),
            "--curated-dir", str(tmp_path / "curated"), "--reports-dir", str(tmp_path / "reports"),
        ],
    )
    return build.main()


def test_a_failed_quality_check_stops_the_build_before_curated_data_is_written(monkeypatch, tmp_path):
    assert _run_build(monkeypatch, tmp_path, "FAIL") == 1

    report = pd.read_csv(tmp_path / "reports" / "data_quality_report.csv")
    assert "FAIL" in set(report["status"])
    assert not (tmp_path / "curated").exists()


def test_a_clean_build_writes_the_csv_and_parquet(monkeypatch, tmp_path):
    assert _run_build(monkeypatch, tmp_path, "WARNING") == 0

    assert (tmp_path / "curated" / "quarterly_macro_features.csv").exists()
    assert (tmp_path / "curated" / "quarterly_macro_features.parquet").exists()
