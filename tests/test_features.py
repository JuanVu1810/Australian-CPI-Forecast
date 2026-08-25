import pandas as pd
import pytest

from src.features import (
    add_growth_rates,
    add_intervention_dummies,
    add_lag_features,
    add_nonlinear_elastic_net_terms,
    order_feature_columns,
)


def test_growth_rates_preserve_abs_cpi_percentage_change_columns():
    source = pd.DataFrame(
        {
            "quarter": ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1"],
            "cpi_qoq": [0.2, 0.4, 0.3, 0.5, 0.6],
            "cpi_yoy": [1.1, 1.3, 1.5, 1.7, 2.0],
            "trimmed_mean_cpi_qoq": [0.3, 0.5, 0.4, 0.6, 0.7],
            "trimmed_mean_cpi_yoy": [1.4, 1.6, 1.8, 2.0, 2.2],
        }
    )

    result = add_growth_rates(source)

    assert result[
        ["cpi_qoq", "cpi_yoy", "trimmed_mean_cpi_qoq", "trimmed_mean_cpi_yoy"]
    ].equals(source[["cpi_qoq", "cpi_yoy", "trimmed_mean_cpi_qoq", "trimmed_mean_cpi_yoy"]])


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


def test_growth_rates_create_external_growth_and_exchange_rate_changes():
    source = pd.DataFrame(
        {
            "quarter": ["2020Q1", "2020Q2", "2020Q3"],
            "producer_price_index": [100.0, 105.0, 94.5],
            "aud_usd": [0.7000, 0.7350, 0.6615],
        }
    )

    result = add_growth_rates(source)

    assert pd.isna(result.loc[0, "ppi_growth"])
    assert round(result.loc[1, "ppi_growth"], 2) == 5.00
    assert round(result.loc[2, "ppi_growth"], 2) == -10.00
    assert pd.isna(result.loc[0, "aud_usd_change"])
    assert round(result.loc[1, "aud_usd_change"], 2) == 5.00
    assert round(result.loc[2, "aud_usd_change"], 2) == -10.00


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


def test_order_feature_columns_places_trimmed_mean_cpi_next_to_cpi_columns():
    source = pd.DataFrame(
        {
            "unemployment_rate": [5.0],
            "trimmed_mean_cpi_yoy": [3.0],
            "cpi_yoy": [2.5],
            "trimmed_mean_cpi_qoq": [0.7],
            "quarter": ["2020Q1"],
            "cpi_qoq": [0.5],
        }
    )

    result = order_feature_columns(source)

    assert result.columns.tolist()[:5] == [
        "quarter",
        "cpi_qoq",
        "cpi_yoy",
        "trimmed_mean_cpi_qoq",
        "trimmed_mean_cpi_yoy",
    ]


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


def test_default_lag_features_include_trimmed_mean_cpi_yoy_lags():
    source = pd.DataFrame(
        {
            "quarter": ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1"],
            "trimmed_mean_cpi_yoy": [1.1, 1.2, 1.3, 1.4, 1.5],
        }
    )

    result = add_lag_features(source)

    assert "trimmed_mean_cpi_yoy_lag1" in result
    assert "trimmed_mean_cpi_yoy_lag4" in result
    assert result.loc[4, "trimmed_mean_cpi_yoy_lag1"] == 1.4
    assert result.loc[4, "trimmed_mean_cpi_yoy_lag4"] == 1.1


def test_nonlinear_elastic_net_terms_use_lagged_origin_available_inputs():
    source = pd.DataFrame(
        {
            "ppi_growth_lag2": [None, 2.0, -3.0],
            "commodity_growth_lag1": [1.0, -4.0, 0.5],
            "wti_growth_lag1": [1.5, -2.0, 0.0],
            "brent_growth_lag1": [2.0, -3.0, 4.0],
            "cash_rate_change_lag1": [0.25, -0.10, 0.00],
            "unemployment_rate_change_lag1": [0.20, 0.30, -0.40],
        }
    )

    result = add_nonlinear_elastic_net_terms(source)

    assert pd.isna(result.loc[0, "ppi_growth_lag2_sq"])
    assert result.loc[1, "ppi_growth_lag2_sq"] == 4.0
    assert result.loc[2, "ppi_growth_lag2_sq"] == 9.0
    assert result["commodity_growth_lag1_sq"].tolist() == [1.0, 16.0, 0.25]
    assert result["wti_growth_lag1_sq"].tolist() == [2.25, 4.0, 0.0]
    assert result["brent_growth_lag1_sq"].tolist() == [4.0, 9.0, 16.0]
    assert result["cash_rate_change_lag1_x_unemployment_rate_change_lag1"].round(3).tolist() == [
        0.05,
        -0.03,
        -0.0,
    ]


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
