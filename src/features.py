"""Feature engineering for the curated quarterly CPI modelling dataset."""

from __future__ import annotations

import pandas as pd


def add_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add CPI inflation, external growth-rate, and rate-change features."""
    result = df.copy()

    if "cpi_index" in result:
        result["cpi_qoq"] = result["cpi_index"].pct_change(1) * 100
        result["cpi_yoy"] = result["cpi_index"].pct_change(4) * 100

    growth_specs = {
        "wage_price_index": "wpi_growth",
        "producer_price_index": "ppi_growth",
        "commodity_price_index": "commodity_growth",
        "wti_price": "wti_growth",
        "brent_price": "brent_growth",
        "household_spending": "household_spending_growth",
    }
    for source_col, output_col in growth_specs.items():
        if source_col in result:
            result[output_col] = result[source_col].pct_change(1) * 100

    if "aud_usd" in result:
        result["aud_usd_change"] = result["aud_usd"].pct_change(1) * 100

    rate_change_specs = {
        "cash_rate": "cash_rate_change",
        "unemployment_rate": "unemployment_rate_change",
    }
    for source_col, output_col in rate_change_specs.items():
        if source_col in result:
            result[output_col] = result[source_col].diff()

    return result


def add_lag_features(
    df: pd.DataFrame,
    lag_map: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """Create lagged features without back-filling missing values."""
    result = df.copy()
    lag_map = lag_map or {
        "cpi_yoy": [1, 4],
        "cash_rate": [1, 2, 4],
        "unemployment_rate": [1, 2, 4],
        "wpi_growth": [1, 2],
        "ppi_growth": [1, 2],
        "commodity_growth": [1],
        "wti_growth": [1],
        "brent_growth": [1],
        "aud_usd_change": [1],
        "cash_rate_change": [1],
        "unemployment_rate_change": [1, 2],
        "household_spending_growth": [1],
        "inflation_expectations_business": [1],
    }

    for column, lags in lag_map.items():
        if column not in result:
            continue
        for lag in lags:
            result[f"{column}_lag{lag}"] = result[column].shift(lag)

    return result


def order_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place target and high-signal modelling columns before optional extras."""
    preferred = [
        "quarter",
        "cpi_index",
        "cpi_qoq",
        "cpi_yoy",
        "unemployment_rate",
        "unemployment_rate_change",
        "cash_rate",
        "cash_rate_change",
        "wage_price_index",
        "wpi_growth",
        "producer_price_index",
        "ppi_growth",
        "commodity_price_index",
        "commodity_growth",
        "wti_price",
        "wti_growth",
        "brent_price",
        "brent_growth",
        "aud_usd",
        "aud_usd_change",
        "household_spending",
        "household_spending_growth",
        "inflation_expectations_business",
    ]
    lagged = [column for column in df.columns if "_lag" in column]
    ordered = [column for column in preferred + lagged if column in df.columns]
    remaining = [column for column in df.columns if column not in ordered]
    return df[ordered + remaining]
