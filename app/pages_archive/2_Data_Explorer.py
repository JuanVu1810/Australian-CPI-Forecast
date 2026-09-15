"""Interactive data explorer for the curated CPI macro dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURATED_DATA_PATH = PROJECT_ROOT / "data/curated/quarterly_macro_features.csv"
QUALITY_REPORT_PATH = PROJECT_ROOT / "reports/data_quality_report.csv"


@st.cache_data
def load_curated_data() -> pd.DataFrame:
    df = pd.read_csv(CURATED_DATA_PATH)
    df["quarter_date"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp(how="end").normalize()
    return df


@st.cache_data
def load_quality_report() -> pd.DataFrame:
    return pd.read_csv(QUALITY_REPORT_PATH)


st.set_page_config(page_title="Data Explorer | Australian CPI Forecast", layout="wide")
st.title("Data Explorer")

if not CURATED_DATA_PATH.exists():
    st.error("Curated dataset not found. Run `python -m src.build_curated_dataset` first.")
    st.stop()

df = load_curated_data()

min_date = df["quarter_date"].min().date()
max_date = df["quarter_date"].max().date()

filters = st.columns([1.2, 1.8])
selected_range = filters[0].date_input(
    "Quarter range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date, end_date = min_date, max_date

available_columns = [column for column in df.columns if column != "quarter_date"]
default_columns = [
    column
    for column in [
        "quarter",
        "cpi_qoq",
        "cpi_yoy",
        "trimmed_mean_cpi_qoq",
        "trimmed_mean_cpi_yoy",
        "unemployment_rate",
        "cash_rate",
        "wpi_growth",
        "ppi_growth",
        "commodity_growth",
    ]
    if column in available_columns
]
selected_columns = filters[1].multiselect(
    "Table columns",
    options=available_columns,
    default=default_columns,
)

mask = (df["quarter_date"].dt.date >= start_date) & (df["quarter_date"].dt.date <= end_date)
filtered_df = df.loc[mask].copy()

st.metric("Filtered rows", f"{len(filtered_df):,}")

st.subheader("Macro Indicators")
candidate_columns = [
    "unemployment_rate",
    "cash_rate",
    "wpi_growth",
    "ppi_growth",
    "commodity_growth",
    "wti_growth",
    "inflation_expectations_business",
]
indicator_options = [column for column in candidate_columns if column in df.columns]
default_indicators = [
    column for column in ["unemployment_rate", "cash_rate", "wpi_growth"] if column in indicator_options
]
selected_indicators = st.multiselect(
    "Select indicators",
    options=indicator_options,
    default=default_indicators,
)
if selected_indicators:
    st.line_chart(filtered_df.set_index("quarter_date")[selected_indicators], height=320)
else:
    st.info("Select one or more indicators to plot.")

st.subheader("Data Quality")
if QUALITY_REPORT_PATH.exists():
    quality = load_quality_report()
    st.dataframe(quality, width="stretch")
else:
    st.info("Data quality report not found yet.")

st.subheader("Curated Dataset")
if selected_columns:
    st.dataframe(filtered_df[selected_columns].round(4), width="stretch")
else:
    st.info("Select at least one column to display.")
