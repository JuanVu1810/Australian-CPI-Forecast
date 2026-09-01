"""RBA policy-action classifier: live model breakdown.

Calls the FastAPI service's ``GET /rba-action`` over HTTP rather than loading
or fitting classifiers in-process, per this project's Streamlit/FastAPI split.
The endpoint returns all classifier rows and marks the threshold baseline as
the reportable result.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.lib.rba_reports import render_historical_backtest_panel


DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
RBA_CLASSIFIER_REPORT_PATH = PROJECT_ROOT / "reports/rba_classifier_evaluation.md"
HEADLINE_COVERAGE_REPORT_PATH = PROJECT_ROOT / "reports/model_interval_coverage.csv"
TRIMMED_MEAN_COVERAGE_REPORT_PATH = PROJECT_ROOT / "reports/model_interval_coverage_trimmed_mean.csv"
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


def _request_error_detail(exc: requests.RequestException) -> str | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        detail = response.json().get("detail")
    except ValueError:
        return None
    return str(detail) if detail else None


def _probability_frame(row: pd.Series) -> pd.DataFrame:
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


def build_probability_bar(probabilities: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(probabilities)
        .mark_bar(height=42)
        .encode(
            x=alt.X(
                "probability:Q",
                stack="normalize",
                axis=alt.Axis(format="%"),
                title=None,
            ),
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
        .properties(height=86)
        .configure_view(strokeWidth=0)
    )


st.set_page_config(page_title="RBA Policy | Australian CPI Forecast", layout="wide")
st.title("Step 2: RBA Policy Action")
st.caption(
    "Served live from `GET /rba-action` on the FastAPI service -- this page calls "
    "the API rather than loading or fitting classifiers directly."
)

with st.sidebar:
    st.subheader("API connection")
    api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL).rstrip("/")

try:
    response = requests.get(f"{api_base_url}/rba-action", timeout=120)
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

headline_cols = st.columns(5)
headline_cols[0].metric("Threshold action", payload["reportable_action"])
headline_cols[1].metric("Target quarter", payload["target_quarter"])
headline_cols[2].metric("Forecast origin", payload["forecast_origin"])
headline_cols[3].metric("Headline forecast", f"{payload['headline_forecast']:.2f}%")
headline_cols[4].metric("Trimmed mean forecast", f"{payload['trimmed_mean_forecast']:.2f}%")

st.warning(payload["caveat"])

models = pd.DataFrame(payload.get("models", []))
if models.empty:
    st.warning("No RBA policy-action model predictions were returned by `/rba-action`.")
else:
    reportable_model = payload.get("reportable_model", "threshold")
    threshold_rows = models.loc[models["model"].eq(reportable_model)].copy()
    if threshold_rows.empty:
        threshold_rows = models.loc[models["reportable"].astype(bool)].copy()
    if threshold_rows.empty:
        st.error("The `/rba-action` response did not include the threshold reportable row.")
        st.stop()

    threshold = threshold_rows.iloc[0]
    probabilities = _probability_frame(threshold)
    winning_action = str(threshold["predicted_action"]).upper()

    st.subheader("Threshold model probabilities")
    st.success(f"Winning class: {winning_action}")
    st.altair_chart(build_probability_bar(probabilities), width="stretch")
    probability_cols = st.columns(3)
    for index, row in probabilities.iterrows():
        probability_cols[index].metric(row["label"], f"{row['probability']:.1%}")

    comparison = models.loc[models["model"].ne(reportable_model)].copy()
    comparison["model_order"] = comparison["model"].apply(
        lambda value: SECONDARY_MODELS.index(value) if value in SECONDARY_MODELS else len(SECONDARY_MODELS)
    )
    comparison = comparison.sort_values(["model_order", "model"])
    table = comparison[
        [
            "model",
            "predicted_action",
            "confidence",
            "p_cut",
            "p_hold",
            "p_hike",
            "reportable",
            "majority_vote_tie_break",
        ]
    ].copy()
    for column in ["confidence", "p_cut", "p_hold", "p_hike"]:
        table[column] = table[column].astype(float).round(4)

    st.subheader("Secondary classifier comparison")
    st.dataframe(table, width="stretch", hide_index=True)

    render_historical_backtest_panel(
        rba_classifier_report_path=RBA_CLASSIFIER_REPORT_PATH,
        missing_report_label="reports/rba_classifier_evaluation.md",
        coverage_reports=(
            ("Headline", HEADLINE_COVERAGE_REPORT_PATH),
            ("Trimmed mean", TRIMMED_MEAN_COVERAGE_REPORT_PATH),
        ),
        coverage_columns=(
            "target",
            "model",
            "n",
            "nominal_coverage",
            "empirical_coverage",
            "significantly_miscalibrated",
        ),
        coverage_metrics=(
            {
                "label": "Headline coverage",
                "target": "Headline",
                "model": "ensemble",
                "column": "empirical_coverage",
                "format": "percent",
            },
            {
                "label": "Trimmed mean coverage",
                "target": "Trimmed mean",
                "model": "ensemble",
                "column": "empirical_coverage",
                "format": "percent",
            },
        ),
        caption=(
            "The classifier report does not publish a Brier-style score, so this page does "
            "not recompute one. Coverage rates are from the committed calibrated simulation "
            "interval reports with an 80% nominal target."
        ),
    )
