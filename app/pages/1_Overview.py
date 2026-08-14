"""Overview page for the curated CPI macro dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURATED_DATA_PATH = PROJECT_ROOT / "data/curated/quarterly_macro_features.csv"


@st.cache_data
def load_curated_data() -> pd.DataFrame:
    df = pd.read_csv(CURATED_DATA_PATH)
    df["quarter_date"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp(how="end")
    return df


st.set_page_config(page_title="Overview | Australian CPI Forecast", layout="wide")
st.title("Overview")

if not CURATED_DATA_PATH.exists():
    st.error("Curated dataset not found. Run `python -m src.build_curated_dataset` first.")
    st.stop()

df = load_curated_data()

metric_cols = st.columns(4)
metric_cols[0].metric("Rows", f"{len(df):,}")
metric_cols[1].metric("Start", df["quarter"].iloc[0])
metric_cols[2].metric("End", df["quarter"].iloc[-1])
latest_cpi_yoy = df["cpi_yoy"].dropna().iloc[-1]
metric_cols[3].metric("Latest CPI YoY", f"{latest_cpi_yoy:.2f}%")

st.subheader("CPI Index And YoY Inflation")
st.line_chart(
    df.set_index("quarter_date")[["cpi_index", "cpi_yoy"]],
    height=340,
)
