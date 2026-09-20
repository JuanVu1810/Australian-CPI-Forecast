import pandas as pd
import pytest

from app.lib.report_frames import (
    MACRO_DRIVER,
    OWN_SHOCK,
    decomposition_for_quarter,
    decomposition_totals,
    drift_summary,
    flagged_quarters,
    strongest_flagged_quarter,
    system_shock_events,
    with_quarter_date,
    z_axis_bound,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "system": ["System A"] * 3 + ["System B"] * 2,
            "quarter": ["2001Q1", "2000Q3", "2000Q4", "2000Q3", "2000Q4"],
            "shock_z": [0.5, -2.2, 2.6, 1.0, -0.3],
            "flagged": [False, True, True, False, False],
        }
    )


def test_with_quarter_date_uses_quarter_end():
    frame = with_quarter_date(pd.DataFrame({"quarter": ["2000Q1", "2000Q4"]}))

    assert frame["quarter_date"].dt.strftime("%Y-%m-%d").tolist() == ["2000-03-31", "2000-12-31"]


def test_system_shock_events_filters_and_sorts_by_time():
    result = system_shock_events(_events(), "System A")

    assert result["quarter"].tolist() == ["2000Q3", "2000Q4", "2001Q1"]
    assert set(result["system"]) == {"System A"}


def test_flagged_quarters_and_strongest_pick_largest_absolute_shock():
    result = system_shock_events(_events(), "System A")

    assert flagged_quarters(result) == ["2000Q3", "2000Q4"]
    assert strongest_flagged_quarter(result) == "2000Q4"


def test_strongest_flagged_quarter_is_none_when_nothing_flagged():
    result = system_shock_events(_events(), "System B")

    assert flagged_quarters(result) == []
    assert strongest_flagged_quarter(result) is None


def test_flagged_column_read_from_csv_strings_still_works():
    events = _events().assign(flagged=lambda f: f["flagged"].astype(str))

    assert flagged_quarters(events[events["system"] == "System A"]) == ["2000Q3", "2000Q4"]


def _decomposition() -> pd.DataFrame:
    components = {
        "baseline": 2.5,
        "commodity_growth": 0.4,
        "unemployment_rate": -0.1,
        "cpi_yoy": -0.3,
        "inflation_expectations_business": 0.2,
        "cash_rate": -0.05,
    }
    rows = [
        {"system": "System A", "target": "cpi_yoy", "quarter": "2008Q4", "component": c, "contribution": v}
        for c, v in components.items()
    ]
    rows.append(
        {"system": "System A", "target": "cpi_yoy", "quarter": "2009Q1", "component": "baseline", "contribution": 9.9}
    )
    return pd.DataFrame(rows)


def test_decomposition_for_quarter_tags_baseline_own_shock_and_macro_drivers():
    parts = decomposition_for_quarter(_decomposition(), "System A", "2008Q4")
    kinds = dict(zip(parts["component"], parts["kind"]))

    assert len(parts) == 6
    assert kinds["baseline"] == "baseline"
    assert kinds["cpi_yoy"] == OWN_SHOCK
    assert {kinds[c] for c in ["commodity_growth", "unemployment_rate", "cash_rate"]} == {MACRO_DRIVER}


def test_decomposition_totals_split_baseline_from_shocks_and_sum_to_actual():
    parts = decomposition_for_quarter(_decomposition(), "System A", "2008Q4")

    totals = decomposition_totals(parts)

    assert totals["baseline"] == pytest.approx(2.5)
    assert totals["shocks"] == pytest.approx(0.4 - 0.1 - 0.3 + 0.2 - 0.05)
    assert totals["actual"] == pytest.approx(totals["baseline"] + totals["shocks"])


def test_decomposition_for_unknown_quarter_is_empty():
    assert decomposition_for_quarter(_decomposition(), "System A", "1900Q1").empty


def test_drift_summary_counts_unusual_misses_undershoots_and_inputs():
    error_check = pd.DataFrame(
        {
            "target_quarter": ["2026Q1", "2026Q1", "2026Q2"],
            "z_score": [1.3, -2.4, 0.2],
            "error": [0.9, -0.5, 0.1],
        }
    )
    covariate = pd.DataFrame(
        {
            "variable_label": ["Cash Rate", "Business Inflation Expectations"],
            "value_standardized": [0.2, -2.05],
        }
    )

    summary = drift_summary(error_check, covariate)

    assert summary["graded_forecasts"] == 3
    assert summary["graded_quarters"] == ["2026Q1", "2026Q2"]
    assert summary["unusual_misses"] == 1
    assert summary["largest_abs_z"] == pytest.approx(2.4)
    assert summary["undershot"] == 2
    assert summary["unusual_inputs"] == ["Business Inflation Expectations"]


def test_drift_summary_handles_no_graded_forecasts():
    empty_errors = pd.DataFrame({"target_quarter": [], "z_score": [], "error": []})
    empty_covariates = pd.DataFrame({"variable_label": [], "value_standardized": []})

    summary = drift_summary(empty_errors, empty_covariates)

    assert summary["graded_forecasts"] == 0
    assert summary["unusual_misses"] == 0
    assert pd.isna(summary["largest_abs_z"])


def test_z_axis_bound_always_shows_the_two_sigma_zone_and_every_point():
    assert z_axis_bound(pd.Series([0.4, -1.3])) == 3.0
    assert z_axis_bound(pd.Series([0.4, -4.2])) == 5.5
    assert z_axis_bound(pd.Series([], dtype=float)) == 3.0
