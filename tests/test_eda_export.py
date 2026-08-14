"""Tests for static EDA dashboard export helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import eda_export


def test_stationarity_summary_handles_low_variance_and_valid_series(monkeypatch):
    monkeypatch.setattr(eda_export, "STATIONARITY_COLUMNS", ["constant_feature", "valid_feature"])
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "constant_feature": np.ones(40),
            "valid_feature": rng.normal(size=40),
        }
    )

    summary = eda_export.build_stationarity_summary(df)

    constant_rows = summary.loc[summary["variable"] == "constant_feature"]
    assert constant_rows["test"].tolist() == ["not_run"]
    assert constant_rows["notes"].iloc[0] == "insufficient non-missing variation"

    valid_rows = summary.loc[summary["variable"] == "valid_feature"]
    assert set(valid_rows["test"]) == {"ADF", "KPSS"}
    assert valid_rows["observations"].tolist() == [40, 40]


def test_correlation_summary_uses_lagged_features_and_derives_missing_lag(monkeypatch):
    monkeypatch.setattr(
        eda_export,
        "CORRELATION_COLUMNS",
        ["existing_feature", "derived_feature", "missing_feature"],
    )
    cpi_yoy = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    existing_lag = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0])
    same_quarter_existing = pd.Series([99.0, -99.0, 99.0, -99.0, 99.0])
    derived_source = pd.Series([5.0, 15.0, 25.0, 35.0, 45.0])
    df = pd.DataFrame(
        {
            "cpi_yoy": cpi_yoy,
            "existing_feature": same_quarter_existing,
            "existing_feature_lag1": existing_lag,
            "derived_feature": derived_source,
        }
    )

    summary = eda_export.build_correlation_summary(df)

    existing = summary.loc[summary["variable"] == "existing_feature_lag1"].iloc[0]
    assert existing["source_variable"] == "existing_feature"
    assert existing["notes"] == "used existing lag1 feature"
    assert existing["correlation_with_cpi_yoy"] == 1.0

    derived = summary.loc[summary["variable"] == "derived_feature_lag1"].iloc[0]
    expected_derived_corr = cpi_yoy.iloc[1:].corr(derived_source.shift(1).iloc[1:])
    assert derived["notes"] == "derived lag1 for dashboard export"
    assert derived["correlation_with_cpi_yoy"] == expected_derived_corr
    assert "missing_feature_lag1" not in set(summary["variable"])


def test_vif_summary_handles_insufficient_input_and_valid_multicollinearity(monkeypatch):
    monkeypatch.setattr(eda_export, "VIF_COLUMNS", ["feature_a_lag1"])
    insufficient = pd.DataFrame({"feature_a_lag1": [1.0, 2.0, 3.0, 4.0]})

    insufficient_summary = eda_export.build_vif_summary(insufficient)

    assert insufficient_summary["variable"].tolist() == ["feature_a_lag1"]
    assert np.isnan(insufficient_summary["vif"].iloc[0])
    assert insufficient_summary["notes"].iloc[0] == "insufficient rows or variables"

    monkeypatch.setattr(
        eda_export,
        "VIF_COLUMNS",
        ["feature_a_lag1", "feature_b_lag1", "feature_c_lag1"],
    )
    valid = pd.DataFrame(
        {
            "feature_a_lag1": [1, 2, 3, 4, 5, 6],
            "feature_b_lag1": [2.1, 3.9, 6.2, 7.8, 10.1, 11.9],
            "feature_c_lag1": [6.0, 5.2, 3.8, 3.1, 2.0, 1.1],
        }
    )

    valid_summary = eda_export.build_vif_summary(valid)

    assert set(valid_summary["variable"]) == {
        "feature_a_lag1",
        "feature_b_lag1",
        "feature_c_lag1",
    }
    assert valid_summary["vif"].notna().all()
