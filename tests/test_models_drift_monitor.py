import pandas as pd

from src.models import drift_monitor


def _forecast_frame():
    return pd.DataFrame(
        [
            {
                "target": "Headline",
                "model_family": "sarima",
                "forecast_origin": "2025Q4",
                "quarter": "2026Q1",
                "horizon": 1,
                "forecast": 3.4,
                "interval_lower": 2.6,
                "interval_upper": 4.2,
            },
            {
                "target": "Headline",
                "model_family": "ensemble",
                "forecast_origin": "2025Q4",
                "quarter": "2026Q1",
                "horizon": 1,
                "forecast": 3.15,
                "interval_lower": 2.4,
                "interval_upper": 3.9,
            },
            {
                # No realized actual exists for this quarter yet -- must be dropped,
                # not treated as a graded row.
                "target": "Headline",
                "model_family": "sarima",
                "forecast_origin": "2025Q4",
                "quarter": "2027Q4",
                "horizon": 8,
                "forecast": 3.0,
                "interval_lower": 1.5,
                "interval_upper": 4.5,
            },
        ]
    )


def _historical_frame():
    return pd.DataFrame(
        [
            {
                "quarter": "2026Q1",
                "variable": "cpi_yoy",
                "value": 4.0,
                "variable_label": "Headline CPI YoY",
                "value_standardized": 0.81,
            },
            {
                "quarter": "2025Q4",
                "variable": "cpi_yoy",
                "value": 3.7,
                "variable_label": "Headline CPI YoY",
                "value_standardized": 0.62,
            },
            {
                "quarter": "2026Q2",
                "variable": "inflation_expectations_business",
                "value": 3.59,
                "variable_label": "Business Inflation Expectations",
                "value_standardized": 2.05,
            },
            {
                "quarter": "2025Q4",
                "variable": "inflation_expectations_business",
                "value": 1.91,
                "variable_label": "Business Inflation Expectations",
                "value_standardized": 0.16,
            },
        ]
    )


def _write_backtest_reports(tmp_path, monkeypatch):
    headline_path = tmp_path / "backtest_predictions.csv"
    trimmed_path = tmp_path / "backtest_predictions_trimmed_mean.csv"

    headline = pd.DataFrame(
        [
            {"model": "sarima", "forecast_origin": "2024Q1", "target_quarter": "2024Q2", "horizon": 1, "error": -0.1},
            {"model": "sarima", "forecast_origin": "2024Q2", "target_quarter": "2024Q3", "horizon": 1, "error": 0.2},
            {"model": "sarima", "forecast_origin": "2024Q3", "target_quarter": "2024Q4", "horizon": 1, "error": -0.3},
            {"model": "ensemble", "forecast_origin": "2024Q1", "target_quarter": "2024Q2", "horizon": 1, "error": -0.2},
            {"model": "ensemble", "forecast_origin": "2024Q2", "target_quarter": "2024Q3", "horizon": 1, "error": 0.1},
            {"model": "ensemble", "forecast_origin": "2024Q3", "target_quarter": "2024Q4", "horizon": 1, "error": -0.2},
            # Models never served in forecast.csv should be ignored entirely.
            {"model": "seasonal_naive", "forecast_origin": "2024Q1", "target_quarter": "2024Q2", "horizon": 1, "error": 5.0},
        ]
    )
    headline.to_csv(headline_path, index=False)
    pd.DataFrame(columns=headline.columns).to_csv(trimmed_path, index=False)

    monkeypatch.setattr(drift_monitor, "BACKTEST_REPORTS", {"Headline": headline_path, "Trimmed mean": trimmed_path})
    return headline


def test_build_drift_error_check_frame_only_grades_quarters_with_a_real_actual(tmp_path, monkeypatch):
    _write_backtest_reports(tmp_path, monkeypatch)

    result = drift_monitor.build_drift_error_check_frame(_forecast_frame(), _historical_frame())

    # 2027Q4 has no actual yet, so only the two 2026Q1 rows should be graded.
    assert set(result["target_quarter"]) == {"2026Q1"}
    assert len(result) == 2

    sarima_row = result.loc[result["model"] == "sarima"].iloc[0]
    assert sarima_row["actual"] == 4.0
    assert round(sarima_row["error"], 2) == round(4.0 - 3.4, 2)
    assert sarima_row["hist_n"] == 3
    # seasonal_naive's error must never leak into sarima's historical distribution.
    assert round(sarima_row["hist_mean_error"], 4) == round((-0.1 + 0.2 - 0.3) / 3, 4)


def test_build_drift_error_check_frame_empty_when_nothing_graded(tmp_path, monkeypatch):
    _write_backtest_reports(tmp_path, monkeypatch)
    forecast_only_future = _forecast_frame().iloc[[2]]  # the 2027Q4 row only

    result = drift_monitor.build_drift_error_check_frame(forecast_only_future, _historical_frame())

    assert result.empty
    assert list(result.columns) == drift_monitor.ERROR_CHECK_COLUMNS


def test_build_drift_error_history_frame_flags_recent_rows_and_keeps_horizon_one_only(tmp_path, monkeypatch):
    _write_backtest_reports(tmp_path, monkeypatch)

    result = drift_monitor.build_drift_error_history_frame(_forecast_frame(), _historical_frame())

    assert "horizon" not in result.columns  # already filtered to horizon 1, no need to carry it
    recent = result.loc[result["is_recent"]]
    historical = result.loc[~result["is_recent"]]

    # Both sarima and ensemble are graded at horizon 1 for 2026Q1 (the 2027Q4/horizon-8
    # row has no actual yet, so it's excluded, same as in the check-frame test).
    assert len(recent) == 2
    assert set(recent["target_quarter"]) == {"2026Q1"}
    assert set(recent["forecast_origin"]) == {"2025Q4"}
    assert len(historical) == 6  # the three sarima + three ensemble horizon-1 rows on disk


def test_build_drift_covariate_frame_ranks_the_sharpest_recent_move_first():
    result = drift_monitor.build_drift_covariate_frame(_historical_frame())

    assert list(result["variable"]) == ["inflation_expectations_business", "cpi_yoy"]
    top = result.iloc[0]
    assert top["latest_quarter"] == "2026Q2"
    assert top["abs_z_rank"] == 1
    assert top["n_obs"] == 2
