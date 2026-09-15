"""Dashboard for precomputed EDA report artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATHS = {
    "stationarity": PROJECT_ROOT / "reports/eda_stationarity.csv",
    "correlations": PROJECT_ROOT / "reports/eda_correlations.csv",
    "vif": PROJECT_ROOT / "reports/eda_vif.csv",
}


@st.cache_data
def load_report(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def missing_reports() -> list[str]:
    return [name for name, path in REPORT_PATHS.items() if not path.exists()]


st.set_page_config(page_title="EDA Dashboard | Australian CPI Forecast", layout="wide")
st.title("EDA Dashboard")
st.caption(
    "Summaries below are computed directly from the curated dataset by "
    "`src/eda_export.py` as a lightweight dashboard view. See "
    "`notebooks/EDA.ipynb` for the full leakage-aware exploratory analysis."
)

missing = missing_reports()
if missing:
    st.error(
        "Missing EDA report artifacts: "
        + ", ".join(f"`reports/eda_{name}.csv`" for name in missing)
        + ". Run `python -m src.eda_export` first."
    )
    st.stop()

stationarity = load_report(REPORT_PATHS["stationarity"])
correlations = load_report(REPORT_PATHS["correlations"])
vif = load_report(REPORT_PATHS["vif"])

st.subheader("Stationarity Summary")
test_filter = st.multiselect(
    "Stationarity tests",
    options=sorted(stationarity["test"].dropna().unique()),
    default=sorted(stationarity["test"].dropna().unique()),
)
stationarity_view = stationarity.loc[stationarity["test"].isin(test_filter)].copy()
st.dataframe(stationarity_view, width="stretch")

chart_cols = st.columns(2)

with chart_cols[0]:
    st.subheader("Correlation With CPI YoY")
    correlation_view = correlations.dropna(subset=["correlation_with_cpi_yoy"]).copy()
    correlation_view = correlation_view.sort_values("correlation_with_cpi_yoy")
    if not correlation_view.empty:
        st.bar_chart(
            correlation_view.set_index("variable")["correlation_with_cpi_yoy"],
            height=360,
        )
    st.dataframe(correlations, width="stretch")

with chart_cols[1]:
    st.subheader("Variance Inflation Factors")
    vif_view = vif.dropna(subset=["vif"]).copy().sort_values("vif")
    if not vif_view.empty:
        st.bar_chart(vif_view.set_index("variable")["vif"], height=360)
    st.dataframe(vif, width="stretch")
