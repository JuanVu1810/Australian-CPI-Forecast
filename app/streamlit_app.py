"""Australian CPI Forecast: interactive demo.

A hands-on companion to the Jupyter Book (``book/australian_cpi_forecasting/``). The
book explains the methods and reports the results; this app only holds the parts a
static page cannot show: a live forecast, the Ensemble's Monte-Carlo path reveal, a
what-if shock scenario, and the RBA policy-action call. Tab names match the book's
chapter names so its "open 4.3 Ensemble" style pointers land on the right tab.

Three tabs call the running FastAPI service (live forecast, scenario, RBA) and show a
soft banner if it is unreachable; the Ensemble tab reads pinned report CSVs. Nothing
here fits a model in-process.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.lib.curated_data import load_curated_data  # noqa: E402


DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
# Set CPI_DEMO_HOSTED=1 on a public deployment (for example a Streamlit Cloud secret). It
# locks the API address to API_BASE_URL, so visitors cannot point this server at any URL,
# and swaps the local "start uvicorn" messages for ones that make sense to a visitor.
HOSTED_DEMO = os.getenv("CPI_DEMO_HOSTED") == "1"
MAX_FORECAST_HORIZON = 8
HISTORY_QUARTERS_SHOWN = 16
INTERVAL_LABEL = "calibrated simulation interval (80% nominal target)"
LIVE_API_TIMEOUT_SECONDS = 60  # Cloud Run's first call after idle took ~14s
# The scenario endpoint takes ~40s and /rba-action ~10s on a laptop, so they get longer.
SLOW_API_TIMEOUT_SECONDS = 120

ENSEMBLE_FLAGSHIP_CONFIG = {
    "Headline": {
        "sample_path": PROJECT_ROOT / "reports/simulation_paths_sample_ensemble.csv",
        "fan_path": PROJECT_ROOT / "reports/simulation_fan_ensemble.csv",
        "value_label": "Simulated headline CPI YoY (%)",
        "reported_target": "Headline",
    },
    "Trimmed mean": {
        "sample_path": PROJECT_ROOT / "reports/simulation_paths_sample_ensemble_trimmed_mean.csv",
        "fan_path": PROJECT_ROOT / "reports/simulation_fan_ensemble_trimmed_mean.csv",
        "value_label": "Simulated trimmed-mean CPI YoY (%)",
        "reported_target": "Trimmed mean",
    },
}

FORECAST_TARGET_CONFIG = {
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
SCENARIO_SHOCK_VARIABLES = (
    "unemployment_rate",
    "cash_rate",
    "commodity_growth",
    "inflation_expectations_business",
)
SCENARIO_TARGET_CONFIG = {
    "Headline (cpi_yoy)": {
        "request_value": "headline",
        "baseline_endpoint": "/forecast/all",
        "history_column": "cpi_yoy",
        "axis_title": "CPI YoY (%)",
    },
    "Trimmed mean (trimmed_mean_cpi_yoy)": {
        "request_value": "trimmed_mean",
        "baseline_endpoint": "/forecast/trimmed-mean/all",
        "history_column": "trimmed_mean_cpi_yoy",
        "axis_title": "Trimmed Mean CPI YoY (%)",
    },
}

SECONDARY_MODELS = (
    "taylor_rule",
    "taylor_rule_estimated",
    "ordered_logit",
    "ordered_probit",
    "frank_hall_xgboost",
    "majority_vote_ensemble",
)
ACTION_LABELS = {"cut": "P(Cut)", "hold": "P(Hold)", "hike": "P(Hike)"}
ACTION_COLORS = {"cut": "#2a78d6", "hold": "#898781", "hike": "#eb6834"}

PALETTE = {
    "light": {
        "fan": "#eb6834", "median": "#a33d19", "muted": "#898781", "grid": "#e1e0d9",
        "ink": "#1c1b1a", "paper": "#fbfaf8", "border": "#e5e2dc", "accent2": "#3a6b63",
        "headline": "#2a78d6", "trimmed": "#eb6834", "target": "#2f8f5b", "midpoint": "#4d6b57",
        "scenario": "#eb6834", "baseline": "#2a78d6",
        "actual": "#2a78d6", "forecast": "#eb6834", "miscalibrated": "#b83232", "nominal": "#898781",
    },
    "dark": {
        "fan": "#d95926", "median": "#ffb08f", "muted": "#a8a49c", "grid": "#2c2c2a",
        "ink": "#ece9e4", "paper": "#15140f", "border": "#33312b", "accent2": "#7fc4b8",
        "headline": "#3987e5", "trimmed": "#d95926", "target": "#61b983", "midpoint": "#8eb89b",
        "scenario": "#d95926", "baseline": "#3987e5",
        "actual": "#3987e5", "forecast": "#d95926", "miscalibrated": "#ff8a8a", "nominal": "#898781",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_report(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _current_theme() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def _display_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _request_error_detail(exc: requests.RequestException) -> str | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        detail = response.json().get("detail")
    except ValueError:
        return None
    return str(detail) if detail else None


def _get_json(method: str, base_url: str, path: str, timeout: int, **kwargs) -> dict:
    response = requests.request(method, f"{base_url}{path}", timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


# Both wrappers cache successes only: they raise on failure, and ``st.cache_data`` never
# caches an exception, so an API that was down a moment ago is retried on the next click
# instead of being remembered as down. The cache is shared by every visitor of the app.
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_json(method: str, base_url: str, path: str, timeout: int, **kwargs) -> dict:
    return _get_json(method, base_url, path, timeout, **kwargs)


# For answers that do not change until the API is redeployed (seeded simulations at a
# pinned forecast origin: the same request always returns the same numbers). Keeping them
# for an hour means the first visitor pays the ~45s and everyone after gets it instantly.
# The catch: after a redeploy with new models, the app can show the old answer for up to
# an hour, so keep this modest.
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_json_stable(method: str, base_url: str, path: str, timeout: int, **kwargs) -> dict:
    return _get_json(method, base_url, path, timeout, **kwargs)


def api_request(method: str, base_url: str, path: str, **kwargs) -> tuple[dict | None, str | None]:
    """Call the FastAPI service; return ``(payload, None)`` or ``(None, error_banner_text)``.

    One shared helper for the three live tabs, so a problem with the API produces one
    consistent soft banner instead of blanking the page. Pass ``timeout=`` for slow
    calls (the scenario endpoint takes about 40 seconds), and ``stable=True`` for answers
    that only change on a redeploy so they are remembered for an hour instead of five
    minutes.
    """
    timeout = kwargs.pop("timeout", LIVE_API_TIMEOUT_SECONDS)
    fetch = _fetch_json_stable if kwargs.pop("stable", False) else _fetch_json
    try:
        return fetch(method, base_url, path, timeout, **kwargs), None
    except requests.Timeout:
        if HOSTED_DEMO:
            return None, (
                f"The forecast service did not answer within {timeout} seconds. "
                "It may be waking up, so try again in a moment."
            )
        return None, (
            f"The API at `{base_url}{path}` did not answer within {timeout} seconds. "
            "It may still be working, so try again in a moment."
        )
    except requests.RequestException as exc:
        if HOSTED_DEMO:
            return None, "The forecast service is not responding right now. Please try again in a minute."
        detail = _request_error_detail(exc)
        detail_text = f" API detail: {detail}" if detail else ""
        return None, (
            f"Could not reach the FastAPI service at `{base_url}{path}`: {exc}.{detail_text} "
            "Start it locally with `uvicorn api.main:app --reload`, or point the API base URL "
            "in the sidebar at a running deployment."
        )


def _anchor_value(
    curated: pd.DataFrame, target_column: str, forecast_origin: str, fallback: float | None = None
) -> float:
    target = curated.set_index("quarter")[target_column].dropna()
    if forecast_origin in target.index:
        return float(target.loc[forecast_origin])
    return float(fallback) if fallback is not None else float(target.iloc[-1])


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def build_ensemble_path_chart(
    sample: pd.DataFrame, fan: pd.DataFrame, n_reveal: int, theme_type: str, value_label: str
) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    # One shared x definition so every layer agrees on the title and on whole-quarter ticks.
    horizon_axis = alt.X("horizon:Q", title="Horizon (quarters ahead)", axis=alt.Axis(tickMinStep=1))
    revealed = sample[sample["draw_id"] < n_reveal]
    final_step = n_reveal >= int(sample["draw_id"].max()) + 1
    layers = [
        alt.Chart(revealed)
        .mark_line(strokeWidth=0.8, opacity=0.35, color=colors["fan"])
        .encode(x=horizon_axis, y=alt.Y("value:Q", title=value_label), detail="draw_id:N")
    ]
    if final_step:
        outer_band = (
            alt.Chart(fan)
            .mark_area(opacity=0.14, color=colors["fan"])
            .encode(x=horizon_axis, y="p10:Q", y2="p90:Q")
        )
        inner_band = (
            alt.Chart(fan)
            .mark_area(opacity=0.28, color=colors["fan"])
            .encode(x=horizon_axis, y="p25:Q", y2="p75:Q")
        )
        median_line = (
            alt.Chart(fan)
            .mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=40), color=colors["median"])
            .encode(
                x=horizon_axis,
                y="median:Q",
                tooltip=["target_quarter:N", "horizon:Q", alt.Tooltip("median:Q", title="Forecast", format=".2f")],
            )
        )
        layers = [outer_band, inner_band, *layers, median_line]
    return (
        alt.layer(*layers)
        .properties(height=300)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


def build_static_rba_probability_chart(rba: pd.DataFrame, theme_type: str) -> alt.Chart:
    colors = PALETTE[theme_type]
    probs = rba.dropna(subset=["p_cut"])[["model", "p_cut", "p_hold", "p_hike"]]
    long = probs.melt(id_vars="model", var_name="action", value_name="probability")
    action_colors = {"p_cut": colors["accent2"], "p_hold": colors["muted"], "p_hike": colors["fan"]}
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            y=alt.Y("model:N", title=None, sort="-x"),
            x=alt.X("probability:Q", stack="normalize", title="Predicted probability"),
            color=alt.Color(
                "action:N",
                scale=alt.Scale(domain=list(action_colors), range=list(action_colors.values())),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["model:N", "action:N", "probability:Q"],
        )
        .properties(height=200)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


def _threshold_probability_frame(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action": ["cut", "hold", "hike"],
            "label": [ACTION_LABELS[action] for action in ("cut", "hold", "hike")],
            "order": [0, 1, 2],
            "probability": [
                float(row["p_cut"] or 0.0),
                float(row["p_hold"] or 0.0),
                float(row["p_hike"] or 0.0),
            ],
        }
    )


def build_threshold_probability_bar(probabilities: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(probabilities)
        .mark_bar(height=42)
        .encode(
            x=alt.X("probability:Q", stack="normalize", axis=alt.Axis(format="%", tickCount=5), title=None),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=list(ACTION_LABELS.values()), range=list(ACTION_COLORS.values())),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            order=alt.Order("order:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("label:N", title="Action"),
                alt.Tooltip("probability:Q", title="Probability", format=".1%"),
            ],
        )
        .properties(height=150)
        .configure_view(strokeWidth=0)
    )


def build_forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, theme_type: str) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    color_scale = alt.Scale(domain=["Headline", "Trimmed mean"], range=[colors["headline"], colors["trimmed"]])
    dash_scale = alt.Scale(domain=["Actual", "Forecast"], range=[[1, 0], [6, 3]])
    start = min(history["quarter_date"].min(), forecast["quarter_date"].min())
    end = max(history["quarter_date"].max(), forecast["quarter_date"].max())
    rba_band = pd.DataFrame({"start": [start], "end": [end], "lower": [2.0], "upper": [3.0]})
    midpoint = pd.DataFrame({"midpoint": [2.5]})

    target_band = (
        alt.Chart(rba_band)
        .mark_rect(opacity=0.08, color=colors["target"])
        .encode(x=alt.X("start:T", title="Quarter"), x2="end:T", y=alt.Y("lower:Q", title="CPI YoY (%)"), y2="upper:Q")
    )
    midpoint_rule = alt.Chart(midpoint).mark_rule(color=colors["midpoint"], strokeDash=[6, 3], strokeWidth=1.4).encode(y="midpoint:Q")
    band = (
        alt.Chart(forecast)
        .mark_area(opacity=0.16)
        .encode(
            x=alt.X("quarter_date:T", title="Quarter"),
            y=alt.Y("lower_ci:Q", title="CPI YoY (%)"),
            y2="upper_ci:Q",
            color=alt.Color("target:N", scale=color_scale, title="Series"),
            tooltip=[
                "target:N", "quarter_date:T",
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
            tooltip=["target:N", "series:N", "quarter_date:T", alt.Tooltip("forecast:Q", title="Forecast", format=".2f")],
        )
    )
    origin_rule = (
        alt.Chart(forecast.sort_values("quarter_date").iloc[[0]])
        .mark_rule(color=colors["muted"], strokeDash=[2, 2])
        .encode(x="quarter_date:T")
    )
    return (
        (target_band + midpoint_rule + band + actual_line + forecast_line + origin_rule)
        .properties(height=300)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


def build_scenario_chart(forecast: pd.DataFrame, theme_type: str, y_axis_title: str) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    color_scale = alt.Scale(domain=["Ensemble baseline", "SVAR shock scenario"], range=[colors["baseline"], colors["scenario"]])
    dash_scale = alt.Scale(domain=["Ensemble baseline", "SVAR shock scenario"], range=[[5, 3], [1, 0]])
    start = forecast["quarter_date"].min()
    end = forecast["quarter_date"].max()
    rba_band = pd.DataFrame({"start": [start], "end": [end], "lower": [2.0], "upper": [3.0]})
    midpoint = pd.DataFrame({"midpoint": [2.5]})
    target_band = (
        alt.Chart(rba_band)
        .mark_rect(opacity=0.08, color=colors["target"])
        .encode(x=alt.X("start:T", title="Quarter"), x2="end:T", y=alt.Y("lower:Q", title=y_axis_title), y2="upper:Q")
    )
    midpoint_rule = alt.Chart(midpoint).mark_rule(color=colors["midpoint"], strokeDash=[6, 3], strokeWidth=1.4).encode(y="midpoint:Q")
    band = (
        alt.Chart(forecast)
        .mark_area(opacity=0.14)
        .encode(
            x=alt.X("quarter_date:T", title="Quarter"),
            y=alt.Y("lower_ci:Q", title=y_axis_title),
            y2="upper_ci:Q",
            color=alt.Color("series:N", scale=color_scale, title=None),
            tooltip=[
                "series:N", "quarter_date:T",
                alt.Tooltip("lower_ci:Q", title=f"Lower {INTERVAL_LABEL}", format=".2f"),
                alt.Tooltip("upper_ci:Q", title=f"Upper {INTERVAL_LABEL}", format=".2f"),
            ],
        )
    )
    forecast_line = (
        alt.Chart(forecast)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45))
        .encode(
            x="quarter_date:T",
            y=alt.Y("forecast:Q", title=y_axis_title),
            color=alt.Color("series:N", scale=color_scale, title=None, legend=alt.Legend(orient="bottom")),
            strokeDash=alt.StrokeDash("series:N", scale=dash_scale, legend=None),
            tooltip=["series:N", "quarter_date:T", alt.Tooltip("forecast:Q", title="Forecast", format=".2f")],
        )
    )
    origin_rule = alt.Chart(forecast.iloc[[0]]).mark_rule(color=colors["muted"], strokeDash=[2, 2]).encode(x="quarter_date:T")
    return (
        (target_band + midpoint_rule + band + forecast_line + origin_rule)
        .properties(height=300)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


def _family_map(payload: dict) -> dict[str, dict]:
    return {model["model_family"]: model for model in payload.get("models", [])}


def _ordered_families(families: set[str]) -> list[str]:
    ordered = [family for family in MODEL_ORDER if family in families]
    return ordered + sorted(families.difference(ordered))


def _model_label(model_family: str) -> str:
    return MODEL_LABELS.get(model_family, model_family.replace("_", " ").title())


def _history_frame(curated: pd.DataFrame) -> pd.DataFrame:
    frames = []
    recent = curated.tail(HISTORY_QUARTERS_SHOWN)
    for target_label, config in FORECAST_TARGET_CONFIG.items():
        frame = recent[["quarter_date", config["history_column"]]].rename(columns={config["history_column"]: "value"})
        frame["target"] = target_label
        frame["series"] = "Actual"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _forecast_frame(family_forecast: dict, target_label: str, anchor_quarter_date, anchor_value: float) -> pd.DataFrame:
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
            pd.PeriodIndex([family_forecast["forecast_origin"]], freq="Q").to_timestamp(how="end").normalize()[0]
        )
        target_column = FORECAST_TARGET_CONFIG[target_label]["history_column"]
        fallback = float(curated[target_column].dropna().iloc[-1])
        anchor_value = _anchor_value(curated, target_column, family_forecast["forecast_origin"], fallback)
        frames.append(_forecast_frame(family_forecast, target_label, anchor_quarter_date, anchor_value))
    return pd.concat(frames, ignore_index=True)


def _baseline_ensemble(payload: dict) -> dict | None:
    for model in payload.get("models", []):
        if model.get("model_family") == "ensemble":
            return model
    return None


def _scenario_frame(payload: dict, anchor_quarter_date, anchor_value: float, series: str) -> pd.DataFrame:
    quarters = pd.PeriodIndex(payload["quarters"], freq="Q").to_timestamp(how="end").normalize()
    rows = pd.DataFrame(
        {
            "quarter_date": quarters,
            "series": series,
            "forecast": payload["forecast"],
            "lower_ci": payload["interval_lower"],
            "upper_ci": payload["interval_upper"],
        }
    )
    bridge = pd.DataFrame(
        {
            "quarter_date": [anchor_quarter_date],
            "series": [series],
            "forecast": [anchor_value],
            "lower_ci": [anchor_value],
            "upper_ci": [anchor_value],
        }
    )
    return pd.concat([bridge, rows], ignore_index=True)


# ===========================================================================
# Page
# ===========================================================================
st.set_page_config(page_title="Australian CPI Forecast Demo", layout="wide")
theme_type = _current_theme()

st.title("Australian CPI forecast demo")
st.write(
    "Try the live parts of the project. The Jupyter Book explains how each one works; "
    "this app is only for playing with them."
)

with st.sidebar:
    st.subheader("API connection")
    if HOSTED_DEMO:
        api_base_url = DEFAULT_API_BASE_URL.rstrip("/")
        if not os.getenv("API_BASE_URL"):
            st.error("This deployment is missing its API_BASE_URL setting, so the live tabs cannot work.")
        st.caption(
            "Live forecast, 4.5 Scenario Engine and 4.6 RBA Policy Classifier use this project's "
            "forecast API on Google Cloud Run. 4.3 Ensemble reads pinned files and needs no API. "
            "The scenario and RBA calls can take up to a minute."
        )
    else:
        api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL, key="api_base_url").rstrip("/")
        st.caption(
            "Live forecast, 4.5 Scenario Engine and 4.6 RBA Policy Classifier call this service. "
            "Start it with `uvicorn api.main:app --reload`. 4.3 Ensemble reads local files and "
            "needs no API."
        )

curated = load_curated_data()

forecast_tab, ensemble_tab, scenario_tab, rba_tab = st.tabs(
    ["Live forecast", "4.3 Ensemble", "4.5 Scenario Engine", "4.6 RBA Policy Classifier"]
)

with forecast_tab:
    st.caption(
        "Pick a horizon and load the forecasts served by the API. Book: section 6, Deployment."
    )
    live_control_cols = st.columns(2)
    with live_control_cols[0]:
        live_target_label = st.radio("Target detail", options=list(FORECAST_TARGET_CONFIG), horizontal=True, key="live_forecast_target")
    with live_control_cols[1]:
        live_horizon = st.slider("Forecast horizon (quarters)", min_value=1, max_value=MAX_FORECAST_HORIZON, value=MAX_FORECAST_HORIZON, key="live_forecast_horizon")

    if st.button("Load live forecasts", key="load_live_forecasts"):
        live_payloads_by_target: dict[str, dict] = {}
        live_fetch_error = None
        with st.spinner("Loading forecasts..."):
            for label, config in FORECAST_TARGET_CONFIG.items():
                payload, error = api_request("POST", api_base_url, config["endpoint"], json={"horizon": live_horizon})
                if error:
                    live_fetch_error = error
                    break
                live_payloads_by_target[label] = payload
        st.session_state["live_forecast_result"] = {
            "payloads_by_target": live_payloads_by_target,
            "error": live_fetch_error,
            "horizon": live_horizon,
        }

    live_result = st.session_state.get("live_forecast_result")
    if live_result is None:
        st.info("Load live forecasts to query the configured FastAPI service.")
    elif live_result["error"]:
        st.info(live_result["error"])
    else:
        live_payloads_by_target = live_result["payloads_by_target"]
        if live_result["horizon"] != live_horizon:
            st.info("Loaded forecasts use the previous horizon. Reload to refresh this view.")
        live_family_maps = {label: _family_map(payload) for label, payload in live_payloads_by_target.items()}
        common_families = set.intersection(*(set(fm) for fm in live_family_maps.values()))
        live_unavailable = [
            {"target": label, **item} for label, payload in live_payloads_by_target.items() for item in payload.get("unavailable", [])
        ]
        if not common_families:
            st.warning("No model family is currently servable for both headline and trimmed-mean targets.")
        else:
            family_names = _ordered_families(common_families)
            live_selected_family = st.selectbox(
                "Model family", options=family_names, format_func=_model_label, index=0, key="live_forecast_family"
            )
            live_target_config = FORECAST_TARGET_CONFIG[live_target_label]
            live_family_forecast = live_family_maps[live_target_label][live_selected_family]
            origin_cols = st.columns(4)
            origin_cols[0].metric("Forecast origin", live_family_forecast["forecast_origin"])
            origin_cols[1].metric("Horizon served", live_family_forecast["horizon"])
            origin_cols[2].metric(f"Next quarter {live_target_label.lower()}", f"{live_family_forecast['forecast'][0]:.2f}%")
            origin_cols[3].metric(
                f"Next quarter {INTERVAL_LABEL}",
                f"[{live_family_forecast['interval_lower'][0]:.2f}, {live_family_forecast['interval_upper'][0]:.2f}]",
            )
            if any(live_family_forecast.get("significantly_miscalibrated") or []):
                st.warning(
                    f"`{_model_label(live_selected_family)}` has at least one significantly miscalibrated "
                    f"horizon for `{live_target_label}` per `{live_target_config['coverage_report']}`. Treat "
                    "the shaded band as an indicative calibrated simulation interval, not a validated "
                    "probability guarantee."
                )
            live_history = _history_frame(curated)
            live_forecast_frame = _combined_forecast_frame(curated, live_selected_family, live_payloads_by_target, live_family_maps)
            st.markdown(f"**{_model_label(live_selected_family)}: headline and trimmed-mean CPI**")
            st.caption(
                "The RBA target band is fixed at 2.0-3.0% with a dashed 2.5% midpoint. Forecast "
                f"uncertainty is shown as a shaded {INTERVAL_LABEL}."
            )
            st.altair_chart(build_forecast_chart(live_history, live_forecast_frame, theme_type), width="stretch")
            st.caption(
                "Ensemble combines SARIMA and Elastic Net forecasts only. SVAR is kept separate as "
                "structural scenario evidence and is not blended into this combiner."
            )
            with st.expander(f"{live_target_label} forecast table"):
                live_table = pd.DataFrame(
                    {
                        "quarter": live_family_forecast["quarters"],
                        "forecast": live_family_forecast["forecast"],
                        f"lower {INTERVAL_LABEL}": live_family_forecast["interval_lower"],
                        f"upper {INTERVAL_LABEL}": live_family_forecast["interval_upper"],
                    }
                ).round(4)
                st.dataframe(live_table, width="stretch")
        if live_unavailable:
            with st.expander(f"Unavailable families ({len(live_unavailable)})"):
                for item in live_unavailable:
                    st.write(f"**{item['target']} / {item['model_family']}**: {item['reason']}")

with ensemble_tab:
    st.caption(
        "Pick a target, then drag the slider or press Play to reveal a sample of the Monte-Carlo "
        "draws behind the Ensemble forecast. Book: section 4.3."
    )
    ensemble_target = st.radio(
        "Target", options=list(ENSEMBLE_FLAGSHIP_CONFIG), horizontal=True, key="ensemble_target"
    )
    ensemble_config = ENSEMBLE_FLAGSHIP_CONFIG[ensemble_target]
    sample_path = ensemble_config["sample_path"]
    fan_path = ensemble_config["fan_path"]
    if sample_path.exists() and fan_path.exists():
        sample = load_report(sample_path)
        fan = load_report(fan_path)
        n_draws = int(sample["draw_id"].max()) + 1
        forecast_origin = str(fan["forecast_origin"].iloc[0])

        metric_cols = st.columns(3)
        metric_cols[0].metric("Forecast origin", forecast_origin)
        metric_cols[1].metric("Draws shown", f"{n_draws} of 1,000")
        metric_cols[2].metric("Horizons", f"{int(fan['horizon'].min())}–{int(fan['horizon'].max())}")

        slider_col, play_col = st.columns([5, 1])
        with slider_col:
            n_reveal = st.slider("Paths revealed", 1, n_draws, value=1, key=f"ensemble_reveal_{ensemble_target}")
        chart_placeholder = st.empty()
        with play_col:
            st.write("")
            play = st.button("▶ Play", key=f"ensemble_play_{ensemble_target}")
        if play:
            for step in range(1, n_draws + 1, max(1, n_draws // 60)):
                chart_placeholder.altair_chart(
                    build_ensemble_path_chart(sample, fan, step, theme_type, ensemble_config["value_label"]),
                    width="stretch",
                )
                time.sleep(0.03)
            chart_placeholder.altair_chart(
                build_ensemble_path_chart(sample, fan, n_draws, theme_type, ensemble_config["value_label"]),
                width="stretch",
            )
        else:
            chart_placeholder.altair_chart(
                build_ensemble_path_chart(sample, fan, n_reveal, theme_type, ensemble_config["value_label"]),
                width="stretch",
            )

        reported_path = PROJECT_ROOT / "reports/tableau/forecast.csv"
        if reported_path.exists():
            reported = load_report(reported_path)
            reported = reported[
                (reported["model_family"] == "ensemble") & (reported["target"] == ensemble_config["reported_target"])
            ].sort_values("horizon")
            if not reported.empty:
                h1_reported = float(reported["forecast"].iloc[0])
                h1_median = float(fan.sort_values("horizon")["median"].iloc[0])
                h1_width = float(reported["interval_upper"].iloc[0] - reported["interval_lower"].iloc[0])
                raw_width = float(
                    fan.sort_values("horizon")["p90"].iloc[0] - fan.sort_values("horizon")["p10"].iloc[0]
                )
                st.info(
                    "**The shaded band is the raw simulated interval, not what the Forecast tab serves.** "
                    f"At horizon 1 the raw 80% band is {raw_width:.2f} points wide; the calibrated interval "
                    f"that is served is {h1_width:.2f} points wide, because the raw interval historically "
                    f"under-covers. The reported point forecast ({h1_reported:.2f}%) and this simulation's "
                    f"median ({h1_median:.2f}%) agree at horizon 1 by construction."
                )
    else:
        st.error(
            "Missing simulation reports. Run `python -m src.models.simulation_fan` to regenerate them."
        )

with scenario_tab:
    st.caption(
        "Pick a shock variable and size, then run it against the live API. Only the surprise "
        "relative to the SVAR's own forecast counts. Book: section 4.5."
    )
    scenario_control_cols = st.columns(4)
    scenario_target_label = scenario_control_cols[0].radio(
        "Target", options=list(SCENARIO_TARGET_CONFIG), key="scenario_target"
    )
    scenario_shock_variable = scenario_control_cols[1].selectbox(
        "Shock variable", options=SCENARIO_SHOCK_VARIABLES, format_func=lambda v: v.replace("_", " "), key="scenario_shock_variable"
    )
    scenario_shock_value = scenario_control_cols[2].number_input(
        "Shock value", value=0.0, step=0.25, key="scenario_shock_value"
    )
    scenario_max_horizon = scenario_control_cols[3].slider(
        "Max horizon (quarters)", min_value=1, max_value=MAX_FORECAST_HORIZON, value=MAX_FORECAST_HORIZON, key="scenario_max_horizon"
    )
    scenario_target = SCENARIO_TARGET_CONFIG[scenario_target_label]
    scenario_horizons = list(range(1, scenario_max_horizon + 1))

    if st.button("Run scenario", key="run_scenario"):
        with st.spinner("Running the scenario. This can take up to a minute..."):
            scenario_payload, scenario_error = api_request(
                "POST",
                api_base_url,
                "/forecast/scenario",
                json={
                    "target": scenario_target["request_value"],
                    "shock_variable": scenario_shock_variable,
                    "shock_value": scenario_shock_value,
                    "horizons": scenario_horizons,
                },
                timeout=SLOW_API_TIMEOUT_SECONDS,
                stable=True,
            )
            baseline_payload, baseline_error = (None, None)
            if scenario_payload is not None:
                baseline_payload, baseline_error = api_request(
                    "POST",
                    api_base_url,
                    scenario_target["baseline_endpoint"],
                    json={"horizon": scenario_max_horizon},
                )
        st.session_state["scenario_result"] = {
            "payload": scenario_payload,
            "error": scenario_error,
            "baseline_payload": baseline_payload,
            "baseline_error": baseline_error,
            "target": scenario_target,
        }

    scenario_result = st.session_state.get("scenario_result")
    if scenario_result is None:
        st.info("Run the scenario to fetch the live SVAR-adjusted forecast.")
    else:
        scenario_payload = scenario_result["payload"]
        scenario_error = scenario_result["error"]
        baseline_payload = scenario_result["baseline_payload"]
        scenario_target = scenario_result["target"]

    if scenario_result is not None and scenario_error:
        st.info(scenario_error)
    elif scenario_result is not None and scenario_payload is not None:
        scenario_metric_cols = st.columns(4)
        scenario_metric_cols[0].metric("Forecast origin", scenario_payload["forecast_origin"])
        scenario_metric_cols[1].metric("Target", scenario_payload["target"])
        scenario_metric_cols[2].metric("Shock value", f"{scenario_payload['shock_value']:.3f}")
        scenario_metric_cols[3].metric("Shock size", f"{scenario_payload['shock_size']:.3f}")

        anchor_quarter_date = (
            pd.PeriodIndex([scenario_payload["forecast_origin"]], freq="Q").to_timestamp(how="end").normalize()[0]
        )
        anchor_value = _anchor_value(curated, scenario_target["history_column"], scenario_payload["forecast_origin"])
        scenario_frame = _scenario_frame(scenario_payload, anchor_quarter_date, anchor_value, "SVAR shock scenario")
        baseline_ensemble = _baseline_ensemble(baseline_payload) if baseline_payload else None
        if baseline_ensemble is not None:
            baseline_anchor_quarter_date = (
                pd.PeriodIndex([baseline_ensemble["forecast_origin"]], freq="Q").to_timestamp(how="end").normalize()[0]
            )
            baseline_anchor_value = _anchor_value(curated, scenario_target["history_column"], baseline_ensemble["forecast_origin"])
            baseline_frame = _scenario_frame(baseline_ensemble, baseline_anchor_quarter_date, baseline_anchor_value, "Ensemble baseline")
            scenario_chart_frame = pd.concat([baseline_frame, scenario_frame], ignore_index=True)
        else:
            scenario_chart_frame = scenario_frame
            st.info("The Ensemble baseline is unavailable, so the chart shows only the SVAR shock scenario.")

        st.caption(
            "Read-only overlay: the dashed Ensemble baseline is SARIMA + Elastic Net only; the "
            "solid line is the scenario endpoint's SVAR-adjusted Ensemble draw."
        )
        st.altair_chart(build_scenario_chart(scenario_chart_frame, theme_type, scenario_target["axis_title"]), width="stretch")
        st.warning(scenario_payload["caveat"])
        with st.expander("Scenario forecast table"):
            scenario_table = pd.DataFrame(
                {
                    "horizon": scenario_payload["horizons"],
                    "quarter": scenario_payload["quarters"],
                    "forecast": scenario_payload["forecast"],
                    f"lower {INTERVAL_LABEL}": scenario_payload["interval_lower"],
                    f"upper {INTERVAL_LABEL}": scenario_payload["interval_upper"],
                }
            ).round(4)
            st.dataframe(scenario_table, width="stretch")

with rba_tab:
    st.caption(
        "Seven readings of the RBA's next cut, hold or hike call from the same forecasts. "
        "Book: section 4.6."
    )
    if st.button("Refresh RBA action", key="refresh_rba_action"):
        with st.spinner("Asking the seven classifiers..."):
            rba_payload, rba_error = api_request(
                "GET", api_base_url, "/rba-action", timeout=SLOW_API_TIMEOUT_SECONDS, stable=True
            )
        st.session_state["rba_action_result"] = {"payload": rba_payload, "error": rba_error}

    rba_result = st.session_state.get("rba_action_result")
    if rba_result and rba_result["error"]:
        st.info(rba_result["error"])

    if rba_result and rba_result["payload"] is not None:
        rba_payload = rba_result["payload"]
        rba_headline_cols = st.columns(5)
        rba_headline_cols[0].metric("Threshold action", rba_payload["reportable_action"])
        rba_headline_cols[1].metric("Target quarter", rba_payload["target_quarter"])
        rba_headline_cols[2].metric("Forecast origin", rba_payload["forecast_origin"])
        rba_headline_cols[3].metric("Headline forecast", f"{rba_payload['headline_forecast']:.2f}%")
        rba_headline_cols[4].metric("Trimmed mean forecast", f"{rba_payload['trimmed_mean_forecast']:.2f}%")
        st.warning(rba_payload["caveat"])

        rba_models = pd.DataFrame(rba_payload.get("models", []))
        if rba_models.empty:
            st.warning("No RBA policy-action model predictions were returned by `/rba-action`.")
        else:
            reportable_model = rba_payload.get("reportable_model", "threshold")
            threshold_rows = rba_models.loc[rba_models["model"].eq(reportable_model)].copy()
            if threshold_rows.empty:
                threshold_rows = rba_models.loc[rba_models["reportable"].astype(bool)].copy()
            if threshold_rows.empty:
                st.error("The `/rba-action` response did not include the threshold reportable row.")
            else:
                threshold = threshold_rows.iloc[0]
                probabilities = _threshold_probability_frame(threshold)
                st.success(f"Winning class: {str(threshold['predicted_action']).upper()}")
                st.altair_chart(build_threshold_probability_bar(probabilities), width="stretch")
                probability_cols = st.columns(3)
                for index, row in probabilities.iterrows():
                    probability_cols[index].metric(row["label"], f"{row['probability']:.1%}")

                comparison = rba_models.loc[rba_models["model"].ne(reportable_model)].copy()
                comparison["model_order"] = comparison["model"].apply(
                    lambda value: SECONDARY_MODELS.index(value) if value in SECONDARY_MODELS else len(SECONDARY_MODELS)
                )
                comparison = comparison.sort_values(["model_order", "model"])
                secondary_table = comparison[
                    ["model", "predicted_action", "confidence", "p_cut", "p_hold", "p_hike", "reportable", "majority_vote_tie_break"]
                ].copy()
                for column in ["confidence", "p_cut", "p_hold", "p_hike"]:
                    secondary_table[column] = secondary_table[column].astype(float).round(4)
                st.markdown("**Secondary classifier comparison**")
                st.dataframe(secondary_table, width="stretch", hide_index=True)
    else:
        rba_path = PROJECT_ROOT / "reports/tableau/rba_action.csv"
        if rba_path.exists():
            rba_static = load_report(rba_path)
            origin = str(rba_static["forecast_origin"].iloc[0])
            target_q = str(rba_static["target_quarter"].iloc[0])
            call = str(rba_static["reportable_action"].iloc[0])
            st.metric(f"Reportable call for {target_q} (origin {origin})", call.upper())
            st.altair_chart(build_static_rba_probability_chart(rba_static, theme_type), width="stretch")
        else:
            st.error(f"Missing: `{_display_path(rba_path)}`")
