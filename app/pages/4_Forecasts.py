"""Forecast dashboard: model selector, calibrated interval bands, and
forecast-vs-actual accuracy once snapshotted quarters publish.

Calls the FastAPI service's ``POST /forecast/all`` over HTTP rather than
loading or fitting models in-process, per this project's Streamlit/FastAPI
split. The accuracy section reads ``reports/forecast_snapshot_accuracy.csv``
directly, the same way every other report in this app is read, rather than
opening a live Postgres connection from the dashboard.
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
ACCURACY_REPORT_PATH = PROJECT_ROOT / "reports/forecast_snapshot_accuracy.csv"
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MAX_FORECAST_HORIZON = 8
HISTORY_QUARTERS_SHOWN = 16

# Blue/orange pair from this project's categorical palette (slots 1-2), which
# already clears the adjacent colorblind-safety gate in both light and dark
# mode, so no separate validation pass is needed for a two-series chart.
PALETTE = {
    "light": {"actual": "#2a78d6", "forecast": "#eb6834", "muted": "#898781", "grid": "#e1e0d9"},
    "dark": {"actual": "#3987e5", "forecast": "#d95926", "muted": "#898781", "grid": "#2c2c2a"},
}


@st.cache_data
def load_curated_data() -> pd.DataFrame:
    df = pd.read_csv(CURATED_DATA_PATH)
    df["quarter_date"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp(how="end").normalize()
    return df


def _current_theme() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def _forecast_frame(family_forecast: dict, anchor_quarter_date, anchor_value: float) -> pd.DataFrame:
    quarters = pd.PeriodIndex(family_forecast["quarters"], freq="Q").to_timestamp(how="end").normalize()
    rows = pd.DataFrame(
        {
            "quarter_date": quarters,
            "series": "Forecast",
            "forecast": family_forecast["forecast"],
            "lower_ci": family_forecast["interval_lower"],
            "upper_ci": family_forecast["interval_upper"],
        }
    )
    bridge = pd.DataFrame(
        {
            "quarter_date": [anchor_quarter_date],
            "series": ["Forecast"],
            "forecast": [anchor_value],
            "lower_ci": [anchor_value],
            "upper_ci": [anchor_value],
        }
    )
    return pd.concat([bridge, rows], ignore_index=True)


def build_forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, theme_type: str) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    color_scale = alt.Scale(
        domain=["Actual", "Forecast"], range=[colors["actual"], colors["forecast"]]
    )
    dash_scale = alt.Scale(domain=["Actual", "Forecast"], range=[[1, 0], [6, 3]])

    band = (
        alt.Chart(forecast)
        .mark_area(opacity=0.18, color=colors["forecast"])
        .encode(
            x=alt.X("quarter_date:T", title="Quarter"),
            y=alt.Y("lower_ci:Q", title="CPI YoY (%)"),
            y2="upper_ci:Q",
        )
    )
    actual_line = (
        alt.Chart(history)
        .mark_line(strokeWidth=2)
        .encode(
            x="quarter_date:T",
            y=alt.Y("value:Q", title="CPI YoY (%)"),
            color=alt.Color(
                "series:N", scale=color_scale, title=None, legend=alt.Legend(orient="bottom")
            ),
            strokeDash=alt.StrokeDash("series:N", scale=dash_scale, legend=None),
        )
    )
    forecast_line = (
        alt.Chart(forecast)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45))
        .encode(
            x="quarter_date:T",
            y="forecast:Q",
            color=alt.Color(
                "series:N", scale=color_scale, title=None, legend=alt.Legend(orient="bottom")
            ),
            strokeDash=alt.StrokeDash("series:N", scale=dash_scale, legend=None),
        )
    )
    origin_rule = (
        alt.Chart(forecast.iloc[[0]])
        .mark_rule(color=colors["muted"], strokeDash=[2, 2])
        .encode(x="quarter_date:T")
    )

    return (
        (band + actual_line + forecast_line + origin_rule)
        .properties(height=360)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


st.set_page_config(page_title="Forecasts | Australian CPI Forecast", layout="wide")
st.title("Forecasts")
st.caption(
    "Served live from `POST /forecast/all` on the FastAPI service -- this page "
    "calls the API rather than loading or fitting models directly."
)

if not CURATED_DATA_PATH.exists():
    st.error("Curated dataset not found. Run `python -m src.build_curated_dataset` first.")
    st.stop()

with st.sidebar:
    st.subheader("API connection")
    api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL).rstrip("/")
    horizon = st.slider("Forecast horizon (quarters)", min_value=1, max_value=MAX_FORECAST_HORIZON, value=MAX_FORECAST_HORIZON)

curated = load_curated_data()

try:
    response = requests.post(
        f"{api_base_url}/forecast/all",
        json={"horizon": horizon},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
except requests.RequestException as exc:
    st.error(
        f"Could not reach the FastAPI service at `{api_base_url}`: {exc}\n\n"
        "Start it locally with `uvicorn api.main:app --reload`, or point the "
        "API base URL in the sidebar at a running deployment."
    )
    st.stop()

models = payload.get("models", [])
unavailable = payload.get("unavailable", [])

if not models:
    st.warning("No model families are currently servable by `/forecast/all`.")
else:
    family_names = [model["model_family"] for model in models]
    selected_family = st.selectbox("Model family", options=family_names)
    family_forecast = next(model for model in models if model["model_family"] == selected_family)

    origin_cols = st.columns(4)
    origin_cols[0].metric("Forecast origin", family_forecast["forecast_origin"])
    origin_cols[1].metric("Horizon served", family_forecast["horizon"])
    origin_cols[2].metric("Next quarter forecast", f"{family_forecast['forecast'][0]:.2f}%")
    origin_cols[3].metric(
        "Next quarter interval",
        f"[{family_forecast['interval_lower'][0]:.2f}, {family_forecast['interval_upper'][0]:.2f}]",
    )

    if any(family_forecast.get("significantly_miscalibrated") or []):
        st.warning(
            f"`{selected_family}`'s forecast interval is significantly miscalibrated at one or "
            "more horizons per `reports/model_interval_coverage.csv` -- treat the shaded band "
            "as indicative, not a validated confidence interval."
        )

    history = curated.tail(HISTORY_QUARTERS_SHOWN)[["quarter_date", "cpi_yoy"]].rename(
        columns={"cpi_yoy": "value"}
    )
    history["series"] = "Actual"
    anchor_quarter_date = (
        pd.PeriodIndex([family_forecast["forecast_origin"]], freq="Q")
        .to_timestamp(how="end")
        .normalize()[0]
    )
    anchor_value = float(curated.set_index("quarter")["cpi_yoy"].get(family_forecast["forecast_origin"], history["value"].iloc[-1]))
    forecast_frame = _forecast_frame(family_forecast, anchor_quarter_date, anchor_value)

    st.subheader(f"{selected_family}: actual vs. forecast CPI YoY")
    st.altair_chart(
        build_forecast_chart(history, forecast_frame, _current_theme()),
        width="stretch",
    )

    with st.expander("Forecast table"):
        table = pd.DataFrame(
            {
                "quarter": family_forecast["quarters"],
                "forecast": family_forecast["forecast"],
                "lower_ci": family_forecast["interval_lower"],
                "upper_ci": family_forecast["interval_upper"],
            }
        ).round(4)
        st.dataframe(table, width="stretch")

if unavailable:
    with st.expander(f"Unavailable families ({len(unavailable)})"):
        for item in unavailable:
            st.write(f"**{item['model_family']}**: {item['reason']}")

st.divider()
st.subheader("Forecast vs. actual (published quarters)")
st.caption(
    "From `reports/forecast_snapshot_accuracy.csv`, generated by "
    "`python -m src.models.forecast_snapshot snapshot` then `compare` once "
    "`DATABASE_URL` is configured. Actuals are joined at read time only -- "
    "Postgres never stores a copy of the curated CPI series."
)

if not ACCURACY_REPORT_PATH.exists():
    st.info(
        "No forecast snapshot accuracy report yet. Configure `DATABASE_URL`, then run "
        "`python -m src.models.forecast_snapshot snapshot` and "
        "`python -m src.models.forecast_snapshot compare`."
    )
else:
    accuracy = pd.read_csv(ACCURACY_REPORT_PATH)
    observed = accuracy["status"].eq("observed")
    hit_mask = accuracy.loc[observed, "hit"].astype(str).eq("True")

    accuracy_cols = st.columns(3)
    accuracy_cols[0].metric("Snapshots observed", f"{int(observed.sum())}")
    accuracy_cols[1].metric("Snapshots pending", f"{int((~observed).sum())}")
    accuracy_cols[2].metric(
        "Interval hit rate",
        f"{hit_mask.mean() * 100:.0f}%" if len(hit_mask) else "n/a",
    )

    st.dataframe(accuracy.round(4), width="stretch")
