"""Scenario explorer: live SVAR-adjusted Ensemble forecast fan.

Calls the FastAPI service's ``POST /forecast/scenario`` over HTTP rather than
loading or fitting models in-process, per this project's Streamlit/FastAPI
split. The page uses the local curated target series only to anchor the live
scenario forecast at the latest known forecast origin.
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURATED_DATA_PATH = PROJECT_ROOT / "data/curated/quarterly_macro_features.csv"
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MAX_FORECAST_HORIZON = 8
SHOCK_VARIABLES = (
    "unemployment_rate",
    "cash_rate",
    "commodity_growth",
    "inflation_expectations_business",
)
TARGET_CONFIG = {
    "Headline (cpi_yoy)": {
        "request_value": "headline",
        "history_column": "cpi_yoy",
        "axis_title": "CPI YoY (%)",
    },
    "Trimmed mean (trimmed_mean_cpi_yoy)": {
        "request_value": "trimmed_mean",
        "history_column": "trimmed_mean_cpi_yoy",
        "axis_title": "Trimmed Mean CPI YoY (%)",
    },
}

PALETTE = {
    "light": {"forecast": "#eb6834", "muted": "#898781", "grid": "#e1e0d9"},
    "dark": {"forecast": "#d95926", "muted": "#898781", "grid": "#2c2c2a"},
}


@st.cache_data
def load_curated_data() -> pd.DataFrame:
    return pd.read_csv(CURATED_DATA_PATH)


def _current_theme() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def _request_error_detail(exc: requests.RequestException) -> str | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        detail = response.json().get("detail")
    except ValueError:
        return None
    return str(detail) if detail else None


def _anchor_value(curated: pd.DataFrame, target_column: str, forecast_origin: str) -> float:
    target = curated.set_index("quarter")[target_column].dropna()
    if forecast_origin in target.index:
        return float(target.loc[forecast_origin])
    return float(target.iloc[-1])


def _scenario_frame(payload: dict, anchor_quarter_date, anchor_value: float) -> pd.DataFrame:
    quarters = pd.PeriodIndex(payload["quarters"], freq="Q").to_timestamp(how="end").normalize()
    rows = pd.DataFrame(
        {
            "quarter_date": quarters,
            "forecast": payload["forecast"],
            "lower_ci": payload["interval_lower"],
            "upper_ci": payload["interval_upper"],
        }
    )
    bridge = pd.DataFrame(
        {
            "quarter_date": [anchor_quarter_date],
            "forecast": [anchor_value],
            "lower_ci": [anchor_value],
            "upper_ci": [anchor_value],
        }
    )
    return pd.concat([bridge, rows], ignore_index=True)


def build_scenario_chart(forecast: pd.DataFrame, theme_type: str, y_axis_title: str) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    band = (
        alt.Chart(forecast)
        .mark_area(opacity=0.18, color=colors["forecast"])
        .encode(
            x=alt.X("quarter_date:T", title="Quarter"),
            y=alt.Y("lower_ci:Q", title=y_axis_title),
            y2="upper_ci:Q",
        )
    )
    forecast_line = (
        alt.Chart(forecast)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45), color=colors["forecast"])
        .encode(x="quarter_date:T", y=alt.Y("forecast:Q", title=y_axis_title))
    )
    origin_rule = (
        alt.Chart(forecast.iloc[[0]])
        .mark_rule(color=colors["muted"], strokeDash=[2, 2])
        .encode(x="quarter_date:T")
    )
    return (
        (band + forecast_line + origin_rule)
        .properties(height=360)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


st.set_page_config(page_title="Scenario Explorer | Australian CPI Forecast", layout="wide")
st.title("Scenario Explorer")
st.caption(
    "Served live from `POST /forecast/scenario` on the FastAPI service -- this page "
    "calls the API rather than loading or fitting models directly."
)

if not CURATED_DATA_PATH.exists():
    st.error("Curated dataset not found. Run `python -m src.build_curated_dataset` first.")
    st.stop()

with st.sidebar:
    st.subheader("API connection")
    api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL).rstrip("/")
    target_label = st.radio("Target", options=list(TARGET_CONFIG))
    shock_variable = st.selectbox(
        "Shock variable",
        options=SHOCK_VARIABLES,
        format_func=lambda value: value.replace("_", " "),
    )
    shock_value = st.number_input("Shock value", value=0.0, step=0.25)
    max_horizon = st.slider(
        "Max horizon (quarters)",
        min_value=1,
        max_value=MAX_FORECAST_HORIZON,
        value=MAX_FORECAST_HORIZON,
    )

target = TARGET_CONFIG[target_label]
curated = load_curated_data()
horizons = list(range(1, max_horizon + 1))

try:
    response = requests.post(
        f"{api_base_url}/forecast/scenario",
        json={
            "target": target["request_value"],
            "shock_variable": shock_variable,
            "shock_value": shock_value,
            "horizons": horizons,
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
except requests.RequestException as exc:
    detail = _request_error_detail(exc)
    detail_text = f"\n\nAPI detail: {detail}" if detail else ""
    st.error(
        f"Could not reach the FastAPI service at `{api_base_url}`: {exc}{detail_text}\n\n"
        "Start it locally with `uvicorn api.main:app --reload`, or point the "
        "API base URL in the sidebar at a running deployment."
    )
    st.stop()

metrics = st.columns(4)
metrics[0].metric("Forecast origin", payload["forecast_origin"])
metrics[1].metric("Target", payload["target"])
metrics[2].metric("Shock value", f"{payload['shock_value']:.3f}")
metrics[3].metric("Shock size", f"{payload['shock_size']:.3f}")

anchor_quarter_date = (
    pd.PeriodIndex([payload["forecast_origin"]], freq="Q").to_timestamp(how="end").normalize()[0]
)
anchor_value = _anchor_value(curated, target["history_column"], payload["forecast_origin"])
forecast_frame = _scenario_frame(payload, anchor_quarter_date, anchor_value)

st.altair_chart(
    build_scenario_chart(forecast_frame, _current_theme(), target["axis_title"]),
    width="stretch",
)
st.warning(payload["caveat"])

with st.expander("Scenario forecast table"):
    table = pd.DataFrame(
        {
            "horizon": payload["horizons"],
            "quarter": payload["quarters"],
            "forecast": payload["forecast"],
            "lower_ci": payload["interval_lower"],
            "upper_ci": payload["interval_upper"],
        }
    ).round(4)
    st.dataframe(table, width="stretch")
