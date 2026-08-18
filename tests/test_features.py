import pandas as pd
import pytest

from src.features import add_growth_rates, add_intervention_dummies, add_lag_features


def test_growth_rates_create_cpi_qoq_and_yoy_without_backfill():
    source = pd.DataFrame(
        {
            "quarter": ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1"],
            "cpi_index": [100.0, 101.0, 102.0, 103.0, 108.0],
        }
    )

    result = add_growth_rates(source)

    assert pd.isna(result.loc[0, "cpi_qoq"])
    assert round(result.loc[1, "cpi_qoq"], 2) == 1.00
    assert pd.isna(result.loc[3, "cpi_yoy"])
    assert round(result.loc[4, "cpi_yoy"], 2) == 8.00


def test_growth_rates_create_rate_changes_as_percentage_point_differences():
    source = pd.DataFrame(
        {
            "quarter": ["2020Q1", "2020Q2", "2020Q3"],
            "cash_rate": [0.25, 0.50, 0.10],
            "unemployment_rate": [5.1, 5.4, 5.2],
        }
    )

    result = add_growth_rates(source)

    assert pd.isna(result.loc[0, "cash_rate_change"])
    assert result.loc[1, "cash_rate_change"] == 0.25
    assert result.loc[2, "cash_rate_change"] == -0.40
    assert pd.isna(result.loc[0, "unemployment_rate_change"])
    assert round(result.loc[1, "unemployment_rate_change"], 2) == 0.30
    assert round(result.loc[2, "unemployment_rate_change"], 2) == -0.20


def test_lag_features_shift_values_forward_in_time():
    source = pd.DataFrame(
        {
            "quarter": ["2020Q1", "2020Q2", "2020Q3"],
            "cash_rate": [1.0, 1.5, 2.0],
        }
    )

    result = add_lag_features(source, lag_map={"cash_rate": [1]})

    assert pd.isna(result.loc[0, "cash_rate_lag1"])
    assert result.loc[1, "cash_rate_lag1"] == 1.0
    assert result.loc[2, "cash_rate_lag1"] == 1.5


def test_default_lag_features_include_rate_changes_and_household_spending_growth():
    source = pd.DataFrame(
        {
            "quarter": ["2020Q1", "2020Q2", "2020Q3"],
            "cash_rate_change": [None, 0.25, -0.40],
            "unemployment_rate_change": [None, 0.30, -0.20],
            "household_spending_growth": [None, 1.0, 1.5],
        }
    )

    result = add_lag_features(source)

    assert "cash_rate_change_lag1" in result
    assert "unemployment_rate_change_lag1" in result
    assert "unemployment_rate_change_lag2" in result
    assert "household_spending_growth_lag1" in result
    assert result.loc[2, "cash_rate_change_lag1"] == 0.25
    assert result.loc[2, "unemployment_rate_change_lag1"] == 0.30
    assert result.loc[2, "household_spending_growth_lag1"] == 1.0


def test_intervention_dummies_flag_only_documented_quarters(tmp_path):
    table_path = tmp_path / "intervention_quarters.csv"
    table_path.write_text(
        "quarter,dummy_name,lead_quarters,reason,source_url\n"
        "2020Q2,covid_shock_down,0,test reason,http://example.com\n"
        "2020Q3,covid_shock_rebound,1,test reason,http://example.com\n"
    )
    source = pd.DataFrame(
        {
            "quarter": ["2020Q1", "2020Q2", "2020Q3", "2020Q4"],
            "cpi_yoy": [1.0, -0.3, 0.6, 0.9],
        }
    )

    result = add_intervention_dummies(source, table_path=table_path)

    assert "covid_shock_down_lag0" in result
    assert "covid_shock_rebound_lag1" in result
    assert result["covid_shock_down_lag0"].tolist() == [0, 1, 0, 0]
    assert result["covid_shock_rebound_lag1"].tolist() == [0, 0, 1, 0]
    assert not result[["covid_shock_down_lag0", "covid_shock_rebound_lag1"]].isna().any().any()


def test_intervention_dummies_support_multiple_quarters_per_dummy(tmp_path):
    table_path = tmp_path / "intervention_quarters.csv"
    table_path.write_text(
        "quarter,dummy_name,lead_quarters,reason,source_url\n"
        "2020Q2,covid_shock,0,test reason,http://example.com\n"
        "2020Q3,covid_shock,0,test reason,http://example.com\n"
    )
    source = pd.DataFrame({"quarter": ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]})

    result = add_intervention_dummies(source, table_path=table_path)

    assert result["covid_shock_lag0"].tolist() == [0, 1, 1, 0]


def test_intervention_dummies_reject_inconsistent_lead_quarters(tmp_path):
    table_path = tmp_path / "intervention_quarters.csv"
    table_path.write_text(
        "quarter,dummy_name,lead_quarters,reason,source_url\n"
        "2020Q2,covid_shock,0,test reason,http://example.com\n"
        "2020Q3,covid_shock,1,test reason,http://example.com\n"
    )
    source = pd.DataFrame({"quarter": ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]})

    with pytest.raises(ValueError, match="consistent lead_quarters"):
        add_intervention_dummies(source, table_path=table_path)
