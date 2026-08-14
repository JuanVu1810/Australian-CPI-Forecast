"""Export static EDA summaries for the Streamlit dashboard."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, kpss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_DATA_PATH = PROJECT_ROOT / "data/curated/quarterly_macro_features.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
STATIONARITY_PATH = REPORTS_DIR / "eda_stationarity.csv"
CORRELATIONS_PATH = REPORTS_DIR / "eda_correlations.csv"
VIF_PATH = REPORTS_DIR / "eda_vif.csv"

STATIONARITY_COLUMNS = [
    "cpi_yoy",
    "cpi_qoq",
    "unemployment_rate",
    "cash_rate",
    "wpi_growth",
    "ppi_growth",
    "commodity_growth",
    "wti_growth",
    "brent_growth",
    "aud_usd_change",
    "household_spending_growth",
    "inflation_expectations_business",
]
CORRELATION_COLUMNS = [
    "unemployment_rate",
    "cash_rate",
    "wpi_growth",
    "ppi_growth",
    "commodity_growth",
    "wti_growth",
    "brent_growth",
    "aud_usd_change",
    "household_spending_growth",
    "inflation_expectations_business",
]
CORRELATION_SAFE_LAG = 1
VIF_COLUMNS = [
    "cpi_yoy_lag1",
    "cpi_yoy_lag4",
    "cash_rate_lag1",
    "unemployment_rate_lag1",
    "wpi_growth_lag1",
    "ppi_growth_lag1",
    "commodity_growth_lag1",
    "wti_growth_lag1",
    "brent_growth_lag1",
    "aud_usd_change_lag1",
    "inflation_expectations_business_lag1",
]


def load_curated_data(path: Path = CURATED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _clean_series(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def build_stationarity_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in _available_columns(df, STATIONARITY_COLUMNS):
        values = _clean_series(df[variable])
        if len(values) < 12 or values.nunique() < 2:
            rows.append(
                {
                    "variable": variable,
                    "test": "not_run",
                    "observations": len(values),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "lags": np.nan,
                    "stationary_at_5pct": np.nan,
                    "notes": "insufficient non-missing variation",
                }
            )
            continue

        try:
            adf_stat, adf_p, adf_lags, *_ = adfuller(values, autolag="AIC")
            rows.append(
                {
                    "variable": variable,
                    "test": "ADF",
                    "observations": len(values),
                    "statistic": adf_stat,
                    "p_value": adf_p,
                    "lags": adf_lags,
                    "stationary_at_5pct": adf_p < 0.05,
                    "notes": "rejects unit root when p < 0.05",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive export guard
            rows.append(
                {
                    "variable": variable,
                    "test": "ADF",
                    "observations": len(values),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "lags": np.nan,
                    "stationary_at_5pct": np.nan,
                    "notes": f"failed: {exc}",
                }
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InterpolationWarning)
                kpss_stat, kpss_p, kpss_lags, *_ = kpss(
                    values,
                    regression="c",
                    nlags="auto",
                )
            rows.append(
                {
                    "variable": variable,
                    "test": "KPSS",
                    "observations": len(values),
                    "statistic": kpss_stat,
                    "p_value": kpss_p,
                    "lags": kpss_lags,
                    "stationary_at_5pct": kpss_p >= 0.05,
                    "notes": "fails to reject stationarity when p >= 0.05",
                }
            )
        except Exception as exc:  # pragma: no cover - defensive export guard
            rows.append(
                {
                    "variable": variable,
                    "test": "KPSS",
                    "observations": len(values),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "lags": np.nan,
                    "stationary_at_5pct": np.nan,
                    "notes": f"failed: {exc}",
                }
            )

    return pd.DataFrame(rows)


def build_correlation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target = "cpi_yoy"

    if target not in df.columns:
        return pd.DataFrame(
            columns=[
                "variable",
                "source_variable",
                "target",
                "observations",
                "correlation_with_cpi_yoy",
                "abs_correlation",
                "notes",
            ]
        )

    for source_variable in CORRELATION_COLUMNS:
        lagged_variable = f"{source_variable}_lag{CORRELATION_SAFE_LAG}"
        if lagged_variable in df.columns:
            variable = lagged_variable
            predictor = df[lagged_variable]
            notes = "used existing lag1 feature"
        elif source_variable in df.columns:
            variable = lagged_variable
            predictor = df[source_variable].shift(CORRELATION_SAFE_LAG)
            notes = "derived lag1 for dashboard export"
        else:
            continue

        pair = (
            pd.DataFrame({target: df[target], variable: predictor})
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        correlation = pair[target].corr(pair[variable]) if len(pair) >= 3 else np.nan
        rows.append(
            {
                "variable": variable,
                "source_variable": source_variable,
                "target": target,
                "observations": len(pair),
                "correlation_with_cpi_yoy": correlation,
                "abs_correlation": abs(correlation) if pd.notna(correlation) else np.nan,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def build_vif_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = _available_columns(df, VIF_COLUMNS)
    matrix = df[columns].replace([np.inf, -np.inf], np.nan).dropna()
    rows: list[dict[str, object]] = []

    if len(matrix) < 3 or len(columns) < 2:
        return pd.DataFrame(
            [
                {
                    "variable": column,
                    "observations": len(matrix),
                    "vif": np.nan,
                    "notes": "insufficient rows or variables",
                }
                for column in columns
            ]
        )

    standardized = (matrix - matrix.mean()) / matrix.std(ddof=0)
    standardized = standardized.loc[:, standardized.nunique(dropna=True) > 1]
    vif_matrix = add_constant(standardized, has_constant="add")
    for index, variable in enumerate(standardized.columns):
        try:
            vif = variance_inflation_factor(vif_matrix.to_numpy(), index + 1)
            note = "above 10 threshold" if vif > 10 else "within 10 threshold"
        except Exception as exc:  # pragma: no cover - defensive export guard
            vif = np.nan
            note = f"failed: {exc}"

        rows.append(
            {
                "variable": variable,
                "observations": len(standardized),
                "vif": vif,
                "notes": note,
            }
        )

    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def export_eda_reports() -> None:
    df = load_curated_data()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    build_stationarity_summary(df).round(6).to_csv(STATIONARITY_PATH, index=False)
    build_correlation_summary(df).round(6).to_csv(CORRELATIONS_PATH, index=False)
    build_vif_summary(df).round(6).to_csv(VIF_PATH, index=False)


if __name__ == "__main__":
    export_eda_reports()
    print(f"Wrote {STATIONARITY_PATH}")
    print(f"Wrote {CORRELATIONS_PATH}")
    print(f"Wrote {VIF_PATH}")
