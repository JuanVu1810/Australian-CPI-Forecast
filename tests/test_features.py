import pandas as pd

from src.features import add_growth_rates, add_lag_features


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
