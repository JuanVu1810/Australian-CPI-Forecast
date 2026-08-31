"""Streamlit multipage dashboard entry point."""

from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Australian CPI Forecast", layout="wide")
st.title("Australian CPI Forecast")

st.markdown(
    """
    Use the pages in the sidebar to explore the local CPI forecasting data:

    - Overview
    - Data Explorer
    - EDA Dashboard
    - Forecasts
    - Scenario Explorer
    - RBA Policy
    - Methodology
    - Diagnostics
    """
)
