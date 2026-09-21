"""The SVAR unemployment-rate export reproduces the checked-in report.

reports/svar_unemployment_forecast.csv was once exported by a script that was never committed. These tests keep the
command that now writes it in step with the file, so a change to the SVAR code that moves those numbers fails here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models import svar_unemployment_export as export

CHECKED_IN = Path(__file__).resolve().parents[1] / "reports" / "svar_unemployment_forecast.csv"


def test_export_reproduces_the_checked_in_report_byte_for_byte(tmp_path):
    output = tmp_path / "svar_unemployment_forecast.csv"
    export.write_unemployment_forecast(output)
    assert output.read_bytes() == CHECKED_IN.read_bytes()


def test_export_has_both_systems_and_eight_horizons_from_the_pinned_origin():
    table = export.build_unemployment_forecast()
    assert table["system"].tolist() == ["A"] * 8 + ["B"] * 8
    assert table["horizon"].tolist() == list(range(1, 9)) * 2
    assert set(table["forecast_origin"]) == {"2025Q4"}
    assert table["quarter"].iloc[0] == "2026Q1" and table["quarter"].iloc[7] == "2027Q4"


def test_export_values_are_internally_consistent():
    table = export.build_unemployment_forecast()
    assert (table["p10"] < table["unemployment_rate_forecast"]).all()
    assert (table["unemployment_rate_forecast"] < table["p90"]).all()
    pd.testing.assert_series_equal(
        table["cumulative_change_vs_origin"],
        table["unemployment_rate_forecast"] - table["unemployment_rate_origin"],
        check_names=False,
    )
