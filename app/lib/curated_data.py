"""Load the curated quarterly dataset for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURATED_DATA_PATH = PROJECT_ROOT / "data/curated/quarterly_macro_features.csv"


@st.cache_data
def load_curated_data() -> pd.DataFrame:
    """Load the curated quarterly macro dataset with a derived ``quarter_date`` column."""
    df = pd.read_csv(CURATED_DATA_PATH)
    df["quarter_date"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp(how="end").normalize()
    return df
