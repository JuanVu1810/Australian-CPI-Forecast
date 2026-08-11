"""Streamlit dashboard for exploring the curated CPI macro dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


CURATED_DATA_PATH = Path("data/curated/quarterly_macro_features.csv")
QUALITY_REPORT_PATH = Path("reports/data_quality_report.csv")


st.set_page_config(page_title="Australian CPI Forecast", layout="wide")
st.title("Australian CPI Forecast")

if not CURATED_DATA_PATH.exists():
    st.error("Curated dataset not found. Run `python -m src.build_curated_dataset` first.")
    st.stop()

df = pd.read_csv(CURATED_DATA_PATH)
df["quarter_date"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp(how="end")

st.subheader("CPI Overview")
metric_cols = st.columns(4)
metric_cols[0].metric("Rows", f"{len(df):,}")
metric_cols[1].metric("Start", df["quarter"].iloc[0])
metric_cols[2].metric("End", df["quarter"].iloc[-1])
metric_cols[3].metric("Latest CPI YoY", f"{df['cpi_yoy'].dropna().iloc[-1]:.2f}%")

st.line_chart(
    df.set_index("quarter_date")[["cpi_index", "cpi_yoy"]],
    height=320,
)

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
selected = st.multiselect(
    "Select indicators",
    options=[column for column in candidate_columns if column in df.columns],
    default=["unemployment_rate", "cash_rate", "wpi_growth"],
)
if selected:
    st.line_chart(df.set_index("quarter_date")[selected], height=320)

st.subheader("Data Quality")
if QUALITY_REPORT_PATH.exists():
    quality = pd.read_csv(QUALITY_REPORT_PATH)
    st.dataframe(quality, use_container_width=True)
else:
    st.info("Data quality report not found yet.")

st.subheader("Curated Dataset")
st.dataframe(df.round(4), use_container_width=True)
