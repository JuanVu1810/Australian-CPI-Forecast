"""Feature engineering for the curated quarterly CPI modelling dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_INTERVENTION_TABLE_PATH = Path("data/metadata/intervention_quarters.csv")


def add_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add external growth-rate and rate-change features."""
    result = df.copy()

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
        "trimmed_mean_cpi_yoy": [1, 4],
        # SARIMAX order search Group F intentionally uses lag-4 raw rate levels
        # for long-horizon comparability, separate from change-based groups.
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


def add_nonlinear_elastic_net_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Add the small approved nonlinear Elastic Net terms.

    These columns are deterministic transforms of lagged, origin-available
    inputs. They keep Elastic Net linear in parameters while allowing a few
    screened squared/interaction effects.
    """
    result = df.copy()

    if "ppi_growth_lag2" in result:
        result["ppi_growth_lag2_sq"] = result["ppi_growth_lag2"] ** 2
    if "commodity_growth_lag1" in result:
        result["commodity_growth_lag1_sq"] = result["commodity_growth_lag1"] ** 2
    if "wti_growth_lag1" in result:
        result["wti_growth_lag1_sq"] = result["wti_growth_lag1"] ** 2
    if "brent_growth_lag1" in result:
        result["brent_growth_lag1_sq"] = result["brent_growth_lag1"] ** 2
    if {"cash_rate_change_lag1", "unemployment_rate_change_lag1"}.issubset(result.columns):
        result["cash_rate_change_lag1_x_unemployment_rate_change_lag1"] = (
            result["cash_rate_change_lag1"] * result["unemployment_rate_change_lag1"]
        )

    return result


def add_intervention_dummies(
    df: pd.DataFrame,
    table_path: Path = DEFAULT_INTERVENTION_TABLE_PATH,
) -> pd.DataFrame:
    """Add 0/1 pulse dummies for hand-curated, externally documented shock quarters.

    Column names carry a ``_lag{N}`` suffix where ``N`` is the table's
    ``lead_quarters`` value: how many quarters *before* the shock quarter the
    underlying real-world event (a policy announcement, not the CPI print
    itself) was genuinely public knowledge. The shared walk-forward harness's
    ``infer_min_lag_from_columns`` (``src/models/evaluation.py``) then caps
    each dummy's usable forecast horizon at ``N``, so a forecast origin that
    predates the event's real announcement never sees it encoded in its
    future exogenous inputs. This matters because the shock quarters
    themselves were selected by inspecting the historical CPI series for its
    largest swings -- that selection is legitimate for *explaining* those
    quarters in-sample, but only genuinely forecast-safe for origins on or
    after the real announcement date, which ``lead_quarters`` records
    per-dummy in ``table_path``.
    """
    result = df.copy()
    if "quarter" not in result:
        raise ValueError("add_intervention_dummies requires a 'quarter' column.")

    table = pd.read_csv(table_path)
    required = {"quarter", "dummy_name", "lead_quarters"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{table_path} missing required columns: {sorted(missing)}")

    quarter_strings = result["quarter"].astype(str)
    for dummy_name, rows in table.groupby("dummy_name"):
        shock_quarters = set(rows["quarter"].astype(str))
        lead_values = rows["lead_quarters"].unique()
        if len(lead_values) != 1:
            raise ValueError(
                f"{dummy_name!r} rows in {table_path} must share one consistent "
                "lead_quarters value."
            )
        lag = int(lead_values[0])
        result[f"{dummy_name}_lag{lag}"] = quarter_strings.isin(shock_quarters).astype(int)

    return result


def order_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place target and high-signal modelling columns before optional extras."""
    preferred = [
        "quarter",
        "cpi_qoq",
        "cpi_yoy",
        "trimmed_mean_cpi_qoq",
        "trimmed_mean_cpi_yoy",
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
        "ppi_growth_lag2_sq",
        "commodity_growth_lag1_sq",
        "wti_growth_lag1_sq",
        "brent_growth_lag1_sq",
        "cash_rate_change_lag1_x_unemployment_rate_change_lag1",
    ]
    lagged = [column for column in df.columns if "_lag" in column]
    ordered = list(dict.fromkeys(column for column in preferred + lagged if column in df.columns))
    remaining = [column for column in df.columns if column not in ordered]
    return df[ordered + remaining]
