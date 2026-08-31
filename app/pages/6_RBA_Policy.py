"""RBA policy-action classifier: live model breakdown.

Calls the FastAPI service's ``GET /rba-action`` over HTTP rather than loading
or fitting classifiers in-process, per this project's Streamlit/FastAPI split.
The endpoint returns all classifier rows and marks the threshold baseline as
the reportable result.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _request_error_detail(exc: requests.RequestException) -> str | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        detail = response.json().get("detail")
    except ValueError:
        return None
    return str(detail) if detail else None


st.set_page_config(page_title="RBA Policy | Australian CPI Forecast", layout="wide")
st.title("RBA Policy")
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
headline_cols[0].metric("Threshold/reportable action", payload["reportable_action"])
headline_cols[1].metric("Target quarter", payload["target_quarter"])
headline_cols[2].metric("Forecast origin", payload["forecast_origin"])
headline_cols[3].metric("Headline forecast", f"{payload['headline_forecast']:.2f}%")
headline_cols[4].metric("Trimmed mean forecast", f"{payload['trimmed_mean_forecast']:.2f}%")

st.warning(payload["caveat"])

models = pd.DataFrame(payload.get("models", []))
if models.empty:
    st.warning("No RBA policy-action model predictions were returned by `/rba-action`.")
else:
    table = models[
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
    table["model"] = table.apply(
        lambda row: f"* {row['model']}" if row["reportable"] else row["model"],
        axis=1,
    )
    for column in ["confidence", "p_cut", "p_hold", "p_hike"]:
        table[column] = table[column].astype(float).round(4)

    st.subheader("Classifier Breakdown")
    st.dataframe(table, width="stretch", hide_index=True)
