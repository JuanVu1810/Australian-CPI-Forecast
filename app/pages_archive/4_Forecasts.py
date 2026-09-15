"""Forecast dashboard: macro inputs, Step 1 controls, and CPI forecast bands.

Calls the FastAPI service's all-forecasts endpoints over HTTP rather than
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
INTERVAL_LABEL = "calibrated simulation interval (80% nominal target)"
TARGET_CONFIG = {
    "Headline": {
        "endpoint": "/forecast/all",
        "history_column": "cpi_yoy",
        "display_name": "Headline CPI YoY",
        "coverage_report": "reports/model_interval_coverage.csv",
    },
    "Trimmed mean": {
        "endpoint": "/forecast/trimmed-mean/all",
        "history_column": "trimmed_mean_cpi_yoy",
        "display_name": "Trimmed Mean CPI YoY",
        "coverage_report": "reports/model_interval_coverage_trimmed_mean.csv",
    },
}
MODEL_LABELS = {
    "ensemble": "Ensemble (SARIMA + Elastic Net)",
    "sarima": "SARIMA",
    "elastic_net": "Elastic Net",
}
MODEL_ORDER = ("ensemble", "sarima", "elastic_net")
MACRO_INPUT_COLUMNS = {
    "cpi_yoy": "Headline CPI YoY",
    "trimmed_mean_cpi_yoy": "Trimmed Mean CPI YoY",
    "unemployment_rate": "Unemployment Rate",
    "cash_rate": "Cash Rate",
    "commodity_growth": "Commodity Growth",
    "inflation_expectations_business": "Business Inflation Expectations",
}

PALETTE = {
    "light": {
        "headline": "#2a78d6",
        "trimmed": "#eb6834",
        "target": "#2f8f5b",
        "midpoint": "#4d6b57",
        "muted": "#898781",
        "grid": "#e1e0d9",
    },
    "dark": {
        "headline": "#3987e5",
        "trimmed": "#d95926",
        "target": "#61b983",
        "midpoint": "#8eb89b",
        "muted": "#898781",
        "grid": "#2c2c2a",
    },
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


def _request_error_detail(exc: requests.RequestException) -> str | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        detail = response.json().get("detail")
    except ValueError:
        return None
    return str(detail) if detail else None


def _family_map(payload: dict) -> dict[str, dict]:
    return {model["model_family"]: model for model in payload.get("models", [])}


def _ordered_families(families: set[str]) -> list[str]:
    ordered = [family for family in MODEL_ORDER if family in families]
    return ordered + sorted(families.difference(ordered))


def _model_label(model_family: str) -> str:
    return MODEL_LABELS.get(model_family, model_family.replace("_", " ").title())


def _latest_macro_inputs(curated: pd.DataFrame) -> pd.Series:
    available_columns = [column for column in MACRO_INPUT_COLUMNS if column in curated.columns]
    rows = curated.dropna(subset=available_columns, how="all")
    return rows.iloc[-1]


def _history_frame(curated: pd.DataFrame) -> pd.DataFrame:
    frames = []
    recent = curated.tail(HISTORY_QUARTERS_SHOWN)
    for target_label, config in TARGET_CONFIG.items():
        frame = recent[["quarter_date", config["history_column"]]].rename(
            columns={config["history_column"]: "value"}
        )
        frame["target"] = target_label
        frame["series"] = "Actual"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _anchor_value(curated: pd.DataFrame, target_column: str, forecast_origin: str, fallback: float) -> float:
    target = curated.set_index("quarter")[target_column].dropna()
    if forecast_origin in target.index:
        return float(target.loc[forecast_origin])
    return fallback


def _forecast_frame(
    family_forecast: dict,
    target_label: str,
    anchor_quarter_date,
    anchor_value: float,
) -> pd.DataFrame:
    quarters = pd.PeriodIndex(family_forecast["quarters"], freq="Q").to_timestamp(how="end").normalize()
    rows = pd.DataFrame(
        {
            "quarter_date": quarters,
            "series": "Forecast",
            "target": target_label,
            "forecast": family_forecast["forecast"],
            "lower_ci": family_forecast["interval_lower"],
            "upper_ci": family_forecast["interval_upper"],
        }
    )
    bridge = pd.DataFrame(
        {
            "quarter_date": [anchor_quarter_date],
            "series": ["Forecast"],
            "target": [target_label],
            "forecast": [anchor_value],
            "lower_ci": [anchor_value],
            "upper_ci": [anchor_value],
        }
    )
    return pd.concat([bridge, rows], ignore_index=True)


def _combined_forecast_frame(
    curated: pd.DataFrame,
    selected_family: str,
    payloads_by_target: dict[str, dict],
    family_maps_by_target: dict[str, dict[str, dict]],
) -> pd.DataFrame:
    frames = []
    for target_label, payload in payloads_by_target.items():
        family_forecast = family_maps_by_target[target_label][selected_family]
        anchor_quarter_date = (
            pd.PeriodIndex([family_forecast["forecast_origin"]], freq="Q")
            .to_timestamp(how="end")
            .normalize()[0]
        )
        target_column = TARGET_CONFIG[target_label]["history_column"]
        fallback = float(curated[target_column].dropna().iloc[-1])
        anchor_value = _anchor_value(curated, target_column, family_forecast["forecast_origin"], fallback)
        frames.append(_forecast_frame(family_forecast, target_label, anchor_quarter_date, anchor_value))
    return pd.concat(frames, ignore_index=True)


def build_forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, theme_type: str) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    color_scale = alt.Scale(
        domain=["Headline", "Trimmed mean"], range=[colors["headline"], colors["trimmed"]]
    )
    dash_scale = alt.Scale(domain=["Actual", "Forecast"], range=[[1, 0], [6, 3]])
    start = min(history["quarter_date"].min(), forecast["quarter_date"].min())
    end = max(history["quarter_date"].max(), forecast["quarter_date"].max())
    rba_band = pd.DataFrame({"start": [start], "end": [end], "lower": [2.0], "upper": [3.0]})
    midpoint = pd.DataFrame({"midpoint": [2.5]})

    target_band = (
        alt.Chart(rba_band)
        .mark_rect(opacity=0.08, color=colors["target"])
        .encode(
            x=alt.X("start:T", title="Quarter"),
            x2="end:T",
            y=alt.Y("lower:Q", title="CPI YoY (%)"),
            y2="upper:Q",
        )
    )
    midpoint_rule = (
        alt.Chart(midpoint)
        .mark_rule(color=colors["midpoint"], strokeDash=[6, 3], strokeWidth=1.4)
        .encode(y="midpoint:Q")
    )

    band = (
        alt.Chart(forecast)
        .mark_area(opacity=0.16)
        .encode(
            x=alt.X("quarter_date:T", title="Quarter"),
            y=alt.Y("lower_ci:Q", title="CPI YoY (%)"),
            y2="upper_ci:Q",
            color=alt.Color("target:N", scale=color_scale, title="Series"),
            tooltip=[
                "target:N",
                "quarter_date:T",
                alt.Tooltip("lower_ci:Q", title=f"Lower {INTERVAL_LABEL}", format=".2f"),
                alt.Tooltip("upper_ci:Q", title=f"Upper {INTERVAL_LABEL}", format=".2f"),
            ],
        )
    )
    actual_line = (
        alt.Chart(history)
        .mark_line(strokeWidth=2)
        .encode(
            x="quarter_date:T",
            y=alt.Y("value:Q", title="CPI YoY (%)"),
            color=alt.Color("target:N", scale=color_scale, title="Series"),
            strokeDash=alt.StrokeDash("series:N", scale=dash_scale, legend=None),
            tooltip=["target:N", "series:N", "quarter_date:T", alt.Tooltip("value:Q", format=".2f")],
        )
    )
    forecast_line = (
        alt.Chart(forecast)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45))
        .encode(
            x="quarter_date:T",
            y="forecast:Q",
            color=alt.Color("target:N", scale=color_scale, title="Series", legend=alt.Legend(orient="bottom")),
            strokeDash=alt.StrokeDash("series:N", scale=dash_scale, legend=None),
            tooltip=[
                "target:N",
                "series:N",
                "quarter_date:T",
                alt.Tooltip("forecast:Q", title="Forecast", format=".2f"),
            ],
        )
    )
    origin_rule = (
        alt.Chart(forecast.sort_values("quarter_date").iloc[[0]])
        .mark_rule(color=colors["muted"], strokeDash=[2, 2])
        .encode(x="quarter_date:T")
    )

    return (
        (target_band + midpoint_rule + band + actual_line + forecast_line + origin_rule)
        .properties(height=360)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


st.set_page_config(page_title="Forecasts | Australian CPI Forecast", layout="wide")
st.title("Executive Forecast Dashboard")
st.caption(
    "Served live from `POST /forecast/all` and `POST /forecast/trimmed-mean/all` "
    "on the FastAPI service -- this page calls the API rather than loading or "
    "fitting models directly."
)

if not CURATED_DATA_PATH.exists():
    st.error("Curated dataset not found. Run `python -m src.build_curated_dataset` first.")
    st.stop()

curated = load_curated_data()
latest_inputs = _latest_macro_inputs(curated)

with st.sidebar:
    st.subheader("API connection")
    api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL).rstrip("/")

st.subheader("Raw macro inputs")
st.caption(f"Latest curated quarter: `{latest_inputs['quarter']}`.")
input_cols = st.columns(3)
for index, (column, label) in enumerate(MACRO_INPUT_COLUMNS.items()):
    value = latest_inputs.get(column)
    display = "n/a" if pd.isna(value) else f"{float(value):.2f}"
    input_cols[index % 3].metric(label, display)

st.subheader("Step 1 forecast controls")
control_cols = st.columns(2)
with control_cols[0]:
    target_label = st.radio("Target detail", options=list(TARGET_CONFIG), horizontal=True)
with control_cols[1]:
    horizon = st.slider(
        "Forecast horizon (quarters)",
        min_value=1,
        max_value=MAX_FORECAST_HORIZON,
        value=MAX_FORECAST_HORIZON,
    )

try:
    payloads_by_target = {}
    for label, config in TARGET_CONFIG.items():
        response = requests.post(
            f"{api_base_url}{config['endpoint']}",
            json={"horizon": horizon},
            timeout=60,
        )
        response.raise_for_status()
        payloads_by_target[label] = response.json()
except requests.RequestException as exc:
    detail = _request_error_detail(exc)
    detail_text = f"\n\nAPI detail: {detail}" if detail else ""
    st.error(
        f"Could not reach the FastAPI service at `{api_base_url}`: {exc}{detail_text}\n\n"
        "Start it locally with `uvicorn api.main:app --reload`, or point the "
        "API base URL in the sidebar at a running deployment."
    )
    st.stop()

family_maps_by_target = {
    label: _family_map(payload) for label, payload in payloads_by_target.items()
}
common_families = set.intersection(
    *(set(family_map) for family_map in family_maps_by_target.values())
)
unavailable = [
    {"target": label, **item}
    for label, payload in payloads_by_target.items()
    for item in payload.get("unavailable", [])
]

if not common_families:
    st.warning("No model family is currently servable for both headline and trimmed-mean targets.")
else:
    family_names = _ordered_families(common_families)
    selected_family = st.selectbox(
        "Model family",
        options=family_names,
        format_func=_model_label,
        index=0,
    )

    selected_target_config = TARGET_CONFIG[target_label]
    family_forecast = family_maps_by_target[target_label][selected_family]
    origin_cols = st.columns(4)
    origin_cols[0].metric("Forecast origin", family_forecast["forecast_origin"])
    origin_cols[1].metric("Horizon served", family_forecast["horizon"])
    origin_cols[2].metric(f"Next quarter {target_label.lower()}", f"{family_forecast['forecast'][0]:.2f}%")
    origin_cols[3].metric(
        f"Next quarter {INTERVAL_LABEL}",
        f"[{family_forecast['interval_lower'][0]:.2f}, {family_forecast['interval_upper'][0]:.2f}]",
    )

    if any(family_forecast.get("significantly_miscalibrated") or []):
        st.warning(
            f"`{_model_label(selected_family)}` has at least one significantly miscalibrated "
            f"horizon for `{target_label}` per `{selected_target_config['coverage_report']}`. "
            "Treat the shaded band as an indicative calibrated simulation interval, not a "
            "validated probability guarantee."
        )

    history = _history_frame(curated)
    forecast_frame = _combined_forecast_frame(
        curated,
        selected_family,
        payloads_by_target,
        family_maps_by_target,
    )

    st.subheader(f"{_model_label(selected_family)}: headline and trimmed-mean CPI")
    st.caption(
        "The RBA target band is fixed at 2.0-3.0% with a dashed 2.5% midpoint. "
        f"Forecast uncertainty is shown as a shaded {INTERVAL_LABEL}."
    )
    st.altair_chart(
        build_forecast_chart(history, forecast_frame, _current_theme()),
        width="stretch",
    )
    st.caption(
        "Ensemble combines SARIMA and Elastic Net forecasts only. SVAR is kept separate as "
        "structural scenario evidence and is not blended into this combiner."
    )

    with st.expander(f"{target_label} forecast table"):
        table = pd.DataFrame(
            {
                "quarter": family_forecast["quarters"],
                "forecast": family_forecast["forecast"],
                f"lower {INTERVAL_LABEL}": family_forecast["interval_lower"],
                f"upper {INTERVAL_LABEL}": family_forecast["interval_upper"],
            }
        ).round(4)
        st.dataframe(table, width="stretch")

if unavailable:
    with st.expander(f"Unavailable families ({len(unavailable)})"):
        for item in unavailable:
            st.write(f"**{item['target']} / {item['model_family']}**: {item['reason']}")

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
        "Simulation interval hit rate",
        f"{hit_mask.mean() * 100:.0f}%" if len(hit_mask) else "n/a",
    )

    st.dataframe(accuracy.round(4), width="stretch")
