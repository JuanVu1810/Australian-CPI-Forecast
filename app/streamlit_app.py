"""Australian CPI Forecast -- consolidated Methodology report.

Single-page Streamlit report, sized on screen like a wide institutional site
(after opdi.aero) but paginating to true A4 pages on export, that folds in
every other page previously under ``app/pages/`` (now archived at
``app/pages_archive/`` -- moved, not deleted, and no longer auto-discovered
since Streamlit only scans a directory literally named ``pages``). This is
the technical-audience counterpart to the Tableau executive dashboard
(``.ai/TABLEAU_DASHBOARD_GUIDE.md``, fixed at 6 short stakeholder-facing
tabs): this report is deliberately longer and carries the math, the
diagnostics, and every live interactive tool the project has, not a
one-tab-per-page grand tour.

Static sections read local report CSVs/metadata only and never fit models
in-process. Three sections (the Ensemble flagship demo aside) call the
running FastAPI service directly -- the Scenario Engine, the RBA Classifier,
and the "live forecast tool" in Deployment -- and degrade to a soft banner
(never a page-blanking ``st.stop()``) if the API is unreachable, since one
section failing must not blank the rest of the report.
"""

from __future__ import annotations

import os
import re
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
from app.lib.rba_reports import render_historical_backtest_panel  # noqa: E402
from src.eda_export import restrict_to_eda_window  # noqa: E402


DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MAX_FORECAST_HORIZON = 8
HISTORY_QUARTERS_SHOWN = 16
INTERVAL_LABEL = "calibrated simulation interval (80% nominal target)"
LIVE_API_TIMEOUT_SECONDS = 8
REPORT_SECTIONS = (
    "1. Business Understanding",
    "2. Data Understanding",
    "3. Data Preparation",
    "4. Modeling",
    "5. Evaluation",
    "6. Deployment",
    "7. Conclusion",
)

REPORT_PATHS = {
    "SARIMA (headline)": PROJECT_ROOT / "reports/simulation_fan_sarima.csv",
    "SARIMA (trimmed mean)": PROJECT_ROOT / "reports/simulation_fan_sarima_trimmed_mean.csv",
    "Elastic Net (headline)": PROJECT_ROOT / "reports/simulation_fan_elastic_net.csv",
    "Elastic Net (trimmed mean)": (
        PROJECT_ROOT / "reports/simulation_fan_elastic_net_trimmed_mean.csv"
    ),
    "Ensemble (headline)": PROJECT_ROOT / "reports/simulation_fan_ensemble.csv",
    "Ensemble (trimmed mean)": PROJECT_ROOT / "reports/simulation_fan_ensemble_trimmed_mean.csv",
    "SVAR (headline)": PROJECT_ROOT / "reports/simulation_fan_svar_headline.csv",
    "SVAR (trimmed mean)": PROJECT_ROOT / "reports/simulation_fan_svar_trimmed_mean.csv",
}

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
MACRO_INPUT_COLUMNS = {
    "cpi_yoy": "Headline CPI YoY",
    "trimmed_mean_cpi_yoy": "Trimmed Mean CPI YoY",
    "unemployment_rate": "Unemployment Rate",
    "cash_rate": "Cash Rate",
    "commodity_growth": "Commodity Growth",
    "inflation_expectations_business": "Business Inflation Expectations",
}
ACCURACY_REPORT_PATH = PROJECT_ROOT / "reports/forecast_snapshot_accuracy.csv"

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

RBA_CLASSIFIER_REPORT_PATH = PROJECT_ROOT / "reports/rba_classifier_evaluation.md"
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

DIAGNOSTICS_TARGET_CONFIG = {
    "Headline": {
        "comparison": PROJECT_ROOT / "reports/model_comparison_all.csv",
        "coverage": PROJECT_ROOT / "reports/model_interval_coverage.csv",
        "backtests": PROJECT_ROOT / "reports/backtest_predictions.csv",
        "caption": (
            "Headline comparisons include the RBA benchmark where available, "
            "but the SA-basis model comparison excludes incompatible NSA rows."
        ),
    },
    "Trimmed mean": {
        "comparison": PROJECT_ROOT / "reports/model_comparison_trimmed_mean_all.csv",
        "coverage": PROJECT_ROOT / "reports/model_interval_coverage_trimmed_mean.csv",
        "backtests": PROJECT_ROOT / "reports/backtest_predictions_trimmed_mean.csv",
        "caption": (
            "Trimmed-mean diagnostics exclude RBA rows because the RBA workbook "
            "is an NSA headline CPI forecast series, not a trimmed-mean target."
        ),
    },
}

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
# Generic helpers
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


@st.cache_data(ttl=300, show_spinner=False)
def api_request(method: str, base_url: str, path: str, **kwargs) -> tuple[dict | None, str | None]:
    """Call the FastAPI service; return ``(payload, None)`` or ``(None, error_banner_text)``.

    One shared helper for every live section (Scenario Engine, RBA Classifier,
    live forecast tool) so a single unreachable API produces one consistent
    soft-fallback banner instead of three different error paths -- and so it
    never blanks the rest of this single-page report the way ``st.stop()``
    would. Cached briefly so unrelated widget reruns elsewhere on the page
    don't refire every live call.
    """
    timeout = kwargs.pop("timeout", LIVE_API_TIMEOUT_SECONDS)
    try:
        response = requests.request(method, f"{base_url}{path}", timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        detail = _request_error_detail(exc)
        detail_text = f" API detail: {detail}" if detail else ""
        return None, (
            f"Could not reach the FastAPI service at `{base_url}{path}`: {exc}.{detail_text} "
            "Start it locally with `uvicorn api.main:app --reload`, or point the API base URL "
            "in the sidebar at a running deployment. Showing the static fallback below instead."
        )


def _anchor_value(
    curated: pd.DataFrame, target_column: str, forecast_origin: str, fallback: float | None = None
) -> float:
    target = curated.set_index("quarter")[target_column].dropna()
    if forecast_origin in target.index:
        return float(target.loc[forecast_origin])
    return float(fallback) if fallback is not None else float(target.iloc[-1])


def _fan_frame(report: pd.DataFrame) -> pd.DataFrame:
    frame = report.copy()
    frame["quarter_date"] = (
        pd.PeriodIndex(frame["target_quarter"], freq="Q").to_timestamp(how="end").normalize()
    )
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    return frame.sort_values("horizon")


def caveat_for(label: str) -> tuple[str, str]:
    if label.startswith("SVAR"):
        return (
            "warning",
            "SVAR simulation fans come from Phase 1b VAR(2)-in-levels systems "
            "that still fail multivariate residual whiteness and normality "
            "diagnostics after COVID treatment checks and the block-bootstrap "
            "IRF fix; treat them as illustrative structural simulations, not "
            "precise causal estimates.",
        )
    return (
        "info",
        "This fan is an unconditional forward simulation from a model fit fresh "
        "on the full available history. It is distinct from the walk-forward "
        "accuracy and interval-coverage diagnostics shown in §5 Evaluation.",
    )


# ---------------------------------------------------------------------------
# Article + A4 print presentation layer
# ---------------------------------------------------------------------------
def slugify(text: str) -> str:
    """Match Streamlit's own header-anchor slug exactly (verified against the
    rendered DOM: lowercase, non-alphanumeric runs collapsed to one hyphen)."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# NOTE: every string below is passed to ``st.html()``, never ``st.markdown()``.
# ``st.markdown(..., unsafe_allow_html=True)`` runs the content through a
# CommonMark HTML-block parser first; a blank line inside a large <style>
# block can make that parser bail out of "raw HTML" mode partway through and
# dump the remainder as literal visible text (this is exactly what happened
# before this fix -- the tail of this CSS was rendering as text at the top of
# the page). ``st.html()`` (Streamlit >= 1.36) inserts the string as-is, no
# markdown parsing, so this class of bug can't recur.
def inject_article_css(theme_type: str) -> None:
    colors = PALETTE[theme_type]
    st.html(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap" rel="stylesheet">
        <style>
          [data-testid="stMarkdownContainer"] p,
          [data-testid="stMarkdownContainer"] li,
          [data-testid="stMarkdownContainer"] h1,
          [data-testid="stMarkdownContainer"] h2,
          [data-testid="stMarkdownContainer"] h3,
          [data-testid="stMarkdownContainer"] h4,
          .cpi-kicker, .cpi-dek, .cpi-source {{
            font-family: 'Source Serif 4', Georgia, serif;
          }}
          [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
            font-size: 1.05rem;
            line-height: 1.72;
            max-width: 760px;
          }}
          [data-testid="stMarkdownContainer"] h2 {{
            font-weight: 700;
            border-top: 1px solid {colors['border']};
            padding-top: 1.6rem;
            margin-top: 0.6rem;
            scroll-margin-top: 90px;
          }}
          [data-testid="stMarkdownContainer"] h3 {{
            font-weight: 600;
            scroll-margin-top: 90px;
          }}
          .cpi-kicker {{
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.8rem;
            color: {colors['fan']};
            font-style: italic;
            margin-bottom: -0.4rem;
          }}
          .cpi-dek {{
            font-size: 1.2rem;
            line-height: 1.6;
            color: {colors['muted']};
            font-style: italic;
            max-width: 760px;
          }}
          .cpi-source {{
            font-size: 0.85rem;
            color: {colors['muted']};
            font-style: italic;
          }}
          .cpi-callout {{
            border-left: 3px solid {colors['fan']};
            background: color-mix(in srgb, {colors['fan']} 8%, transparent);
            padding: 0.9rem 1.1rem;
            border-radius: 4px;
            font-family: 'Source Serif 4', Georgia, serif;
            max-width: 760px;
            margin: 0.5rem 0 1rem 0;
          }}
          .cpi-eq-caption {{
            font-size: 0.85rem;
            color: {colors['muted']};
            font-style: italic;
            max-width: 760px;
          }}

          /* --- Wide institutional-site report shell, after opdi.aero --------
             The page uses Streamlit's sidebar as the persistent side rail for
             the table of contents and live API control. The report body stays
             capped and centered so charts never sit under navigation chrome.
             The print pass further down still renders true A4 pages for the
             PDF export regardless of this on-screen width. ---------------- */
          [data-testid="stAppViewContainer"] .block-container {{
            max-width: 1120px !important;
            margin: 1.5rem auto !important;
            padding: 2.5rem 3rem 4rem 3rem !important;
            background: {colors['paper']};
            box-shadow: 0 0 0 1px {colors['border']};
            border-radius: 2px;
            position: relative;
          }}

          .cpi-print-btn {{
            font-family: 'Source Serif 4', Georgia, serif;
            background: {colors['fan']};
            color: #fff;
            border: none;
            padding: 0.55rem 1.15rem;
            border-radius: 6px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
          }}
          .cpi-print-btn:hover {{ filter: brightness(1.08); }}

          /* --- Print / "Download PDF" pass --------------------------------- */
          @media print {{
            @page {{ size: A4; margin: 2cm; }}
            [data-testid="stSidebar"], [data-testid="stHeader"], [data-testid="stToolbar"],
            [data-testid="stDecoration"], [data-testid="collapsedControl"],
            #MainMenu, footer, .cpi-print-hide {{
              display: none !important;
            }}
            [data-testid="stAppViewContainer"] .block-container {{
              max-width: 100% !important;
              box-shadow: none !important;
              border: none !important;
              margin: 0 !important;
              padding: 0 !important;
            }}
          }}
        </style>
        """
    )


def kicker(text: str) -> None:
    st.html(f"<div class='cpi-kicker'>{text}</div>")


def dek(text: str) -> None:
    st.html(f"<div class='cpi-dek'>{text}</div>")


def source_line(text: str) -> None:
    st.html(f"<div class='cpi-source'>Source: {text}</div>")


def callout(text: str) -> None:
    st.html(f"<div class='cpi-callout'>{text}</div>")


def download_pdf_button() -> None:
    st.html(
        """
        <div class="cpi-print-hide" style="margin: 0.75rem 0 1.5rem 0;">
          <button class="cpi-print-btn" onclick="window.print()">Download as PDF</button>
          <div class="cpi-source" style="margin-top: 0.5rem;">
            Opens your browser's print dialog &mdash; choose &ldquo;Save as PDF&rdquo; as the
            destination. The export paginates to true A4 pages, with every chart, table,
            and control rendered exactly as shown on screen.
          </div>
        </div>
        """
    )


def render_sidebar_table_of_contents(sections: tuple[str, ...]) -> None:
    st.markdown("### On this page")
    for section in sections:
        st.markdown(f"- [{section}](#{slugify(section)})")
    st.divider()


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def build_fan_chart(report: pd.DataFrame, theme_type: str) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    outer_band = (
        alt.Chart(report)
        .mark_area(opacity=0.14, color=colors["fan"])
        .encode(
            x=alt.X("quarter_date:T", title="Target Quarter"),
            y=alt.Y("p10:Q", title="Simulated CPI YoY (%)"),
            y2="p90:Q",
            tooltip=["target_quarter:N", "horizon:Q", "p10:Q", "p90:Q"],
        )
    )
    inner_band = (
        alt.Chart(report)
        .mark_area(opacity=0.28, color=colors["fan"])
        .encode(
            x=alt.X("quarter_date:T", title="Target Quarter"),
            y=alt.Y("p25:Q", title="Simulated CPI YoY (%)"),
            y2="p75:Q",
            tooltip=["target_quarter:N", "horizon:Q", "p25:Q", "p75:Q"],
        )
    )
    median_line = (
        alt.Chart(report)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=45), color=colors["median"])
        .encode(
            x=alt.X("quarter_date:T", title="Target Quarter"),
            y=alt.Y("median:Q", title="Simulated CPI YoY (%)"),
            tooltip=["target_quarter:N", "horizon:Q", "median:Q"],
        )
    )
    return (
        (outer_band + inner_band + median_line)
        .properties(height=300)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


def build_cpi_history_chart(curated: pd.DataFrame, theme_type: str) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    history = curated[["quarter", "quarter_date", "cpi_yoy", "trimmed_mean_cpi_yoy"]].dropna(
        subset=["cpi_yoy"], how="all"
    )
    band = (
        alt.Chart(pd.DataFrame({"lo": [2.0], "hi": [3.0]}))
        .mark_rect(opacity=0.12, color=colors["accent2"])
        .encode(y="lo:Q", y2="hi:Q")
    )
    headline = (
        alt.Chart(history)
        .mark_line(strokeWidth=1.8, color=colors["ink"])
        .encode(
            x=alt.X("quarter_date:T", title="Quarter"),
            y=alt.Y("cpi_yoy:Q", title="YoY inflation (%)"),
            tooltip=["quarter:N", "cpi_yoy:Q"],
        )
    )
    trimmed = (
        alt.Chart(history)
        .mark_line(strokeWidth=1.8, color=colors["fan"])
        .encode(x="quarter_date:T", y="trimmed_mean_cpi_yoy:Q", tooltip=["quarter:N", "trimmed_mean_cpi_yoy:Q"])
    )
    return (
        (band + headline + trimmed)
        .properties(height=280)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


def build_ensemble_path_chart(
    sample: pd.DataFrame, fan: pd.DataFrame, n_reveal: int, theme_type: str, value_label: str
) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    revealed = sample[sample["draw_id"] < n_reveal]
    final_step = n_reveal >= int(sample["draw_id"].max()) + 1
    layers = [
        alt.Chart(revealed)
        .mark_line(strokeWidth=0.8, opacity=0.35, color=colors["fan"])
        .encode(x="horizon:Q", y=alt.Y("value:Q", title=value_label), detail="draw_id:N")
    ]
    if final_step:
        outer_band = (
            alt.Chart(fan)
            .mark_area(opacity=0.14, color=colors["fan"])
            .encode(x=alt.X("horizon:Q", title="Horizon (quarters ahead)"), y="p10:Q", y2="p90:Q")
        )
        inner_band = (
            alt.Chart(fan)
            .mark_area(opacity=0.28, color=colors["fan"])
            .encode(x="horizon:Q", y="p25:Q", y2="p75:Q")
        )
        median_line = (
            alt.Chart(fan)
            .mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=40), color=colors["median"])
            .encode(
                x="horizon:Q",
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


def build_svar_irf_chart(irf: pd.DataFrame, max_horizon: int, theme_type: str) -> alt.Chart:
    colors = PALETTE[theme_type]
    subset = irf[irf["horizon"] <= max_horizon]
    zero_line = alt.Chart().mark_rule(color=colors["muted"], strokeDash=[3, 3]).encode(y=alt.datum(0))
    band = (
        alt.Chart()
        .mark_area(opacity=0.22, color=colors["fan"])
        .encode(x=alt.X("horizon:Q", title="Horizon"), y=alt.Y("lower:Q", title="IRF"), y2="upper:Q")
    )
    line = (
        alt.Chart()
        .mark_line(strokeWidth=2, point=True, color=colors["median"])
        .encode(x="horizon:Q", y="irf:Q", tooltip=["shock:N", "horizon:Q", "irf:Q"])
    )
    layered = alt.layer(zero_line, band, line, data=subset).properties(height=150, width=300)
    return (
        layered.facet(facet=alt.Facet("shock:N", title=None), columns=2)
        .resolve_scale(y="independent")
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_header(labelColor=colors["ink"], labelFontWeight="bold")
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
            x=alt.X("probability:Q", stack="normalize", axis=alt.Axis(format="%"), title=None),
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


def build_credit_stress_chart(credit: pd.DataFrame, scenario: str, theme_type: str) -> alt.Chart:
    colors = PALETTE[theme_type]
    subset = credit[credit["scenario"] == scenario]
    return (
        alt.Chart(subset)
        .mark_bar(color=colors["fan"])
        .encode(
            x=alt.X("segment:N", title=None),
            y=alt.Y("ecl_aud_m:Q", title="12-month ECL ($AUDm)"),
            tooltip=["segment:N", "ecl_aud_m:Q", "pd_stressed:Q"],
        )
        .properties(height=240)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


def build_rmse_bar_chart(comparison: pd.DataFrame, theme_type: str) -> alt.Chart:
    colors = PALETTE[theme_type]
    overall = comparison[comparison["horizon"] == "overall"].sort_values("rmse")
    overall = overall.assign(
        highlight=overall["model"].apply(lambda m: "ensemble" if m == "ensemble" else "other")
    )
    return (
        alt.Chart(overall)
        .mark_bar()
        .encode(
            y=alt.Y("model:N", title=None, sort="x"),
            x=alt.X("rmse:Q", title="Overall RMSE (horizons 1-8, shared grid)"),
            color=alt.Color(
                "highlight:N",
                scale=alt.Scale(domain=["ensemble", "other"], range=[colors["fan"], colors["muted"]]),
                legend=None,
            ),
            tooltip=["model:N", "rmse:Q", "n:Q"],
        )
        .properties(height=200)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


def build_coverage_bar_chart(coverage: pd.DataFrame, theme_type: str) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    overall = coverage[coverage["horizon"] == "overall"]
    bars = (
        alt.Chart(overall)
        .mark_bar(color=colors["fan"], opacity=0.75)
        .encode(
            x=alt.X("model:N", title=None),
            y=alt.Y("empirical_coverage:Q", title="Empirical coverage", scale=alt.Scale(domain=[0, 1])),
            tooltip=["model:N", "empirical_coverage:Q", "n:Q"],
        )
    )
    nominal = (
        alt.Chart(pd.DataFrame({"y": [0.8]}))
        .mark_rule(color=colors["ink"], strokeDash=[4, 2])
        .encode(y="y:Q")
    )
    return (
        (bars + nominal)
        .properties(height=220)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
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


def _horizon_rows(report: pd.DataFrame) -> pd.DataFrame:
    rows = report.loc[report["horizon"].astype(str).ne("overall")].copy()
    rows["horizon_num"] = pd.to_numeric(rows["horizon"], errors="coerce")
    return rows.dropna(subset=["horizon_num"]).sort_values(["model", "horizon_num"])


def build_accuracy_chart(report: pd.DataFrame, theme_type: str) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    return (
        alt.Chart(report)
        .mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2)
        .encode(
            x=alt.X("horizon_num:O", title="Horizon"),
            y=alt.Y("rmse:Q", title="RMSE"),
            color=alt.Color("model:N", title="Model"),
            tooltip=["model:N", "horizon:N", "rmse:Q", "mae:Q", "n:Q"],
        )
        .properties(height=300)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


def build_coverage_chart(report: pd.DataFrame, theme_type: str) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    nominal = (
        alt.Chart(report)
        .mark_line(strokeDash=[4, 3], color=colors["nominal"], strokeWidth=1.5)
        .encode(x=alt.X("horizon_num:O", title="Horizon"), y=alt.Y("nominal_coverage:Q"))
    )
    empirical = (
        alt.Chart(report)
        .mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2)
        .encode(
            x=alt.X("horizon_num:O", title="Horizon"),
            y=alt.Y("empirical_coverage:Q", title="Coverage"),
            color=alt.Color("model:N", title="Model"),
            tooltip=["model:N", "horizon:N", "empirical_coverage:Q", "nominal_coverage:Q", "significantly_miscalibrated:N"],
        )
    )
    flagged = (
        alt.Chart(report.loc[report["significantly_miscalibrated"].astype(str).eq("True")])
        .mark_point(filled=True, size=110, color=colors["miscalibrated"], shape="diamond")
        .encode(x="horizon_num:O", y="empirical_coverage:Q")
    )
    return (
        (nominal + empirical + flagged)
        .properties(height=300)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


def build_scatter_chart(report: pd.DataFrame, theme_type: str) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    lower = float(min(report["actual"].min(), report["forecast"].min()))
    upper = float(max(report["actual"].max(), report["forecast"].max()))
    diagonal = pd.DataFrame({"actual": [lower, upper], "forecast": [lower, upper]})
    points = (
        alt.Chart(report)
        .mark_circle(size=42, opacity=0.55)
        .encode(
            x=alt.X("actual:Q", title="Actual"),
            y=alt.Y("forecast:Q", title="Forecast"),
            color=alt.Color("model:N", title="Model"),
            tooltip=["model:N", "forecast_origin:N", "target_quarter:N", "horizon:Q", "actual:Q", "forecast:Q"],
        )
    )
    reference = alt.Chart(diagonal).mark_line(strokeDash=[4, 3], color=colors["nominal"], strokeWidth=1.5).encode(x="actual:Q", y="forecast:Q")
    return (
        (points + reference)
        .properties(height=300)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


# ---------------------------------------------------------------------------
# Forecasts-page helpers (live "6.1 Live forecast tool" section)
# ---------------------------------------------------------------------------
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
st.set_page_config(page_title="Methodology | Australian CPI Forecast", layout="wide")
theme_type = _current_theme()
inject_article_css(theme_type)

kicker("Australian CPI Forecasting Project · Methodology Report")
st.title("How this project forecasts Australian inflation")
dek(
    "Seven methodologies, one CRISP-DM story — the math behind every model, every "
    "interactive tool the project has, and a live look at what “simulating a "
    "forecast” actually means. This report is the technical-audience counterpart to "
    "the six-tab Tableau executive dashboard, and is deliberately longer."
)
download_pdf_button()

with st.sidebar:
    render_sidebar_table_of_contents(REPORT_SECTIONS)
    st.subheader("Live API connection")
    api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL, key="api_base_url").rstrip("/")
    st.caption(
        "Used by §4.5 Scenario Engine, §4.6 RBA Classifier, and §6 Deployment's "
        "live forecast tool. Every other section reads local report CSVs and needs no API."
    )

curated = load_curated_data()

# --- 1. Business Understanding -------------------------------------------------
st.header("1. Business Understanding")
st.markdown(
    """
    **Core question.** Can external macroeconomic indicators — unemployment, the cash
    rate, producer prices, commodity/oil prices, and business inflation expectations —
    improve Australian CPI forecasts relative to a seasonal-naive baseline and the RBA's
    own published forecasts, for both headline (`cpi_yoy`) and trimmed-mean
    (`trimmed_mean_cpi_yoy`) inflation?
    """
)

st.subheader("Current state at a glance")
overview_cols = st.columns(4)
overview_cols[0].metric("Curated quarters", f"{len(curated):,}")
overview_cols[1].metric("Start", curated["quarter"].iloc[0])
overview_cols[2].metric("End", curated["quarter"].iloc[-1])
latest_cpi_yoy = curated["cpi_yoy"].dropna().iloc[-1]
overview_cols[3].metric("Latest headline CPI YoY", f"{latest_cpi_yoy:.2f}%")
st.line_chart(curated.set_index("quarter_date")[["cpi_yoy"]], height=220)

st.markdown("**Success criteria** — defined precisely in §5 Evaluation, previewed here:")
criteria_cols = st.columns(3)
with criteria_cols[0]:
    st.latex(r"\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat y_i)^2}")
    st.caption("Point-forecast accuracy, per horizon, vs. seasonal-naive and RBA")
with criteria_cols[1]:
    st.latex(r"P(\text{lower} \le y \le \text{upper}) \approx 0.80")
    st.caption("Nominal 80% prediction-interval coverage")
with criteria_cols[2]:
    st.latex(r"\text{macro-F1} = \tfrac{1}{3}\sum_{c \in \{cut,hold,hike\}} F1_c")
    st.caption("RBA policy-action classifier, class-balanced")
st.warning(
    "**Out of scope.** The SVAR/scenario engine produces *illustrative* structural "
    "simulations, not causal point estimates — both systems fail multivariate residual "
    "whiteness and normality diagnostics even after COVID treatment and a block-bootstrap "
    "fix. Every scenario output downstream is labeled accordingly."
)

# --- 2. Data Understanding -------------------------------------------------
st.header("2. Data Understanding")
st.markdown(
    "A single quarterly table built from ABS CPI, ABS trimmed-mean CPI, RBA cash-rate "
    "decisions, RBA/ABS labour-force and producer-price series, WTI/Brent crude, and the "
    "RBA's own published CPI forecast workbook — joined on quarter, no manual pasting."
)
# EDA only sees quarters up to the forecast origin; later quarters are held out
# for benchmarking the pinned forecasts.
eda_curated = restrict_to_eda_window(curated)
st.altair_chart(build_cpi_history_chart(eda_curated, theme_type), width="stretch")
st.caption(
    f"{len(eda_curated)} quarters, {eda_curated['quarter'].iloc[0]}–{eda_curated['quarter'].iloc[-1]}. "
    "Later quarters are held out for benchmarking only. "
    "Shaded band is the RBA's 2–3% target range."
)

st.subheader("2.1 Explore the macro indicators")
st.caption("Filter the curated table and browse any combination of indicators.")
min_date = eda_curated["quarter_date"].min().date()
max_date = eda_curated["quarter_date"].max().date()
explorer_cols = st.columns([1.2, 1.8])
selected_range = explorer_cols[0].date_input(
    "Quarter range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="data_explorer_range"
)
if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date, end_date = min_date, max_date

available_columns = [column for column in eda_curated.columns if column != "quarter_date"]
default_columns = [
    column
    for column in [
        "quarter", "cpi_qoq", "cpi_yoy", "trimmed_mean_cpi_qoq", "trimmed_mean_cpi_yoy",
        "unemployment_rate", "cash_rate", "wpi_growth", "ppi_growth", "commodity_growth",
    ]
    if column in available_columns
]
selected_columns = explorer_cols[1].multiselect(
    "Table columns", options=available_columns, default=default_columns, key="data_explorer_columns"
)

mask = (eda_curated["quarter_date"].dt.date >= start_date) & (eda_curated["quarter_date"].dt.date <= end_date)
filtered_curated = eda_curated.loc[mask].copy()
st.metric("Filtered rows", f"{len(filtered_curated):,}")

candidate_columns = [
    "unemployment_rate", "cash_rate", "wpi_growth", "ppi_growth",
    "commodity_growth", "wti_growth", "inflation_expectations_business",
]
indicator_options = [column for column in candidate_columns if column in eda_curated.columns]
default_indicators = [column for column in ["unemployment_rate", "cash_rate", "wpi_growth"] if column in indicator_options]
selected_indicators = st.multiselect(
    "Select indicators to plot", options=indicator_options, default=default_indicators, key="data_explorer_indicators"
)
if selected_indicators:
    st.line_chart(filtered_curated.set_index("quarter_date")[selected_indicators], height=300)
else:
    st.info("Select one or more indicators to plot.")

quality_report_path = PROJECT_ROOT / "reports/data_quality_report.csv"
if quality_report_path.exists():
    with st.expander("Data quality report"):
        st.dataframe(load_report(quality_report_path), width="stretch")

with st.expander("Filtered curated dataset table"):
    if selected_columns:
        st.dataframe(filtered_curated[selected_columns].round(4), width="stretch")
    else:
        st.info("Select at least one column to display.")
source_line("`src/build_curated_dataset.py`, `data/curated/quarterly_macro_features.csv`")

st.subheader("2.2 Exploratory diagnostics")
st.caption(
    "Computed directly from the curated dataset by `src/eda_export.py`. See "
    "`notebooks/EDA.ipynb` for the full leakage-aware exploratory analysis."
)
eda_paths = {
    "stationarity": PROJECT_ROOT / "reports/eda_stationarity.csv",
    "correlations": PROJECT_ROOT / "reports/eda_correlations.csv",
    "vif": PROJECT_ROOT / "reports/eda_vif.csv",
}
eda_missing = [name for name, path in eda_paths.items() if not path.exists()]
if eda_missing:
    st.error(
        "Missing EDA report artifacts: " + ", ".join(f"`reports/eda_{name}.csv`" for name in eda_missing)
        + ". Run `python -m src.eda_export` first."
    )
else:
    stationarity = load_report(eda_paths["stationarity"])
    correlations = load_report(eda_paths["correlations"])
    vif = load_report(eda_paths["vif"])

    st.markdown("**Stationarity summary**")
    test_filter = st.multiselect(
        "Stationarity tests",
        options=sorted(stationarity["test"].dropna().unique()),
        default=sorted(stationarity["test"].dropna().unique()),
        key="eda_stationarity_tests",
    )
    st.dataframe(stationarity.loc[stationarity["test"].isin(test_filter)].copy(), width="stretch")

    eda_chart_cols = st.columns(2)
    with eda_chart_cols[0]:
        st.markdown("**Correlation with CPI YoY**")
        correlation_view = correlations.dropna(subset=["correlation_with_cpi_yoy"]).copy().sort_values("correlation_with_cpi_yoy")
        if not correlation_view.empty:
            st.bar_chart(correlation_view.set_index("variable")["correlation_with_cpi_yoy"], height=320)
        with st.expander("Correlation table"):
            st.dataframe(correlations, width="stretch")
    with eda_chart_cols[1]:
        st.markdown("**Variance inflation factors**")
        vif_view = vif.dropna(subset=["vif"]).copy().sort_values("vif")
        if not vif_view.empty:
            st.bar_chart(vif_view.set_index("variable")["vif"], height=320)
        with st.expander("VIF table"):
            st.dataframe(vif, width="stretch")
source_line("`src/eda_export.py`, `reports/eda_*.csv`")

# --- 3. Data Preparation -------------------------------------------------
st.header("3. Data Preparation")
st.markdown(
    "Two rules govern every feature: **leakage-aware lags** (every macro predictor "
    "enters lagged, never contemporaneously — a forecast made “as of” quarter "
    "$t$ only ever sees data that would genuinely have been published by then), and "
    "**documented interventions, not silent dummies** (structural breaks get a named "
    "column with a cited reason and a genuine forecast-lead-time count)."
)
interventions_path = PROJECT_ROOT / "data/metadata/intervention_quarters.csv"
if interventions_path.exists():
    interventions = pd.read_csv(interventions_path)
    display_cols = [c for c in ["quarter", "dummy_name", "lead_quarters", "reason"] if c in interventions.columns]
    table = interventions[display_cols].copy()
    if "reason" in table.columns:
        table["reason"] = table["reason"].str.slice(0, 110) + "…"
    st.dataframe(table, width="stretch", hide_index=True)
source_line("`src/features.py`, `data/metadata/intervention_quarters.csv`")

# --- 4. Modeling -------------------------------------------------
st.header("4. Modeling")
st.markdown(
    """
    Three levels of ambition, in one project: **accuracy** (SARIMA, Elastic Net,
    Ensemble — forecast the number), **structure** (SVAR, Scenario Engine — explain a
    shock), and **decisions** (RBA Classifier, Credit Stress — classify policy / stress-test
    credit losses). Every equation below is taken from the code that actually runs.
    """
)

st.subheader("4.1 SARIMA — the univariate baseline")
st.latex(r"\varphi(B)\Phi(B^4)(1-B)^d(1-B^4)^D y_t = \theta(B)\Theta(B^4)\,\varepsilon_t")
st.markdown(
    "The shipped headline order is $(1,0,2)\\times(1,0,2,4)$: last quarter's value and "
    "shock, an MA(2) memory, and the same pattern echoed four quarters back — the "
    "seasonal term. Zero exogenous inputs, by design: it answers *how far CPI's own "
    "history carries a forecast*, before any macro predictor is added."
)
source_line("`src/models/sarima.py`")

st.subheader("4.2 Elastic Net — regularized, direct multi-horizon")
st.latex(
    r"\hat\beta_h = \operatorname*{argmin}_{\beta}\ \frac{1}{2n}\lVert y_{t+h}-X_t\beta\rVert_2^2"
    r"+ \alpha\rho\lVert\beta\rVert_1 + \alpha\tfrac{1-\rho}{2}\lVert\beta\rVert_2^2 ,\qquad h=1,\dots,8"
)
st.markdown(
    "One fitted $\\beta_h$ *per horizon* — not one model rolled forward eight times. "
    "$\\alpha$ and $\\rho$ (`l1_ratio`) are chosen by `GridSearchCV` over chronological "
    "`TimeSeriesSplit` folds. The $\\ell_1$ term can zero out a weak macro feature "
    "entirely; the $\\ell_2$ term keeps correlated survivors from cancelling each other out."
)
callout(
    "<b>This argmin does not minimize RMSE, deliberately.</b> The <code>(1/2n)‖·‖²</code> "
    "term is proportional to MSE, but the ℓ₁/ℓ₂ penalty terms bias β̂ <i>away</i> from the "
    "MSE-minimizing (OLS) solution on purpose — that trade-off is the entire mechanism of "
    "regularization. <code>GridSearchCV</code>'s own <code>scoring=\"neg_mean_squared_error\"</code> "
    "isn't RMSE either, though for <i>ranking</i> candidate (α, l1_ratio) pairs it's "
    "equivalent — RMSE = √MSE, and √ is monotonic. The project's headline RMSE success "
    "criterion (§1) only re-enters once β̂ is fixed: it's the walk-forward, out-of-sample "
    "metric §5 Evaluation uses to compare Elastic Net's <i>resulting forecasts</i> against "
    "SARIMA, Ensemble, seasonal-naive, and the RBA — not a claim that every family's own "
    "training loss literally is RMSE."
)
coef_path = PROJECT_ROOT / "reports/elastic_net_coefficients.csv"
if coef_path.exists():
    coefs = load_report(coef_path)
    h1 = coefs[coefs["horizon"] == 1].copy()
    h1["abs_coef"] = h1["coef"].abs()
    zero_count = int((h1["coef"] == 0).sum())
    st.caption(
        f"Horizon 1: α={h1['selected_alpha'].iloc[0]:.4f}, l1_ratio="
        f"{h1['selected_l1_ratio'].iloc[0]:.2f} ({zero_count} of {len(h1)} features shrunk to zero)."
    )
    with st.expander("Horizon-1 coefficients"):
        st.dataframe(
            h1.sort_values("abs_coef", ascending=False)[["feature", "coef"]].reset_index(drop=True),
            width="stretch",
        )
source_line("`src/models/elastic_net.py`")

st.subheader("4.3 Ensemble — inverse-RMSE blend, and the flagship demo")
st.latex(
    r"\hat y_{ens,h} = w_h\,\hat y_{SARIMA,h} + (1-w_h)\,\hat y_{EN,h}, \qquad "
    r"w_h \propto \frac{1}{\mathrm{RMSE}_{SARIMA,h}}"
)
st.markdown(
    "Whichever family won at *that specific horizon* in the walk-forward comparison gets "
    "more say. This is also the project's reportable forecast, so it's the one worth "
    "watching get built: pick a target, then drag the slider (or press Play) to reveal, "
    "one by one, a sample of the 1,000 Monte-Carlo draws behind it."
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
            callout(
                f"<b>The shaded band above is the raw, uncalibrated simulated interval</b> — "
                f"not what <code>/forecast/all</code> actually serves. At horizon 1, this raw "
                f"80% band is {raw_width:.2f} points wide; the calibrated interval actually "
                f"served is {h1_width:.2f} points wide. <code>interval_calibration.py</code> "
                f"found the raw simulated interval under-covers historically, so serving "
                f"widens each side of it by a factor learned from the last 12 quarters of "
                f"observed forecast errors. The reported point forecast ({h1_reported:.2f}%) and "
                f"this simulation's median ({h1_median:.2f}%) agree exactly at horizon 1 — "
                f"<code>ensemble.recenter_paths_to_median</code> guarantees that by construction."
            )
    with st.expander("Compare other model families' simulation fans"):
        selected_label = st.selectbox("Simulation fan", options=list(REPORT_PATHS), key="other_fans")
        selected_path = REPORT_PATHS[selected_label]
        if selected_path.exists():
            other_fan = _fan_frame(load_report(selected_path))
            st.altair_chart(build_fan_chart(other_fan, theme_type), width="stretch")
            caveat_level, caveat_text = caveat_for(selected_label)
            (st.warning if caveat_level == "warning" else st.info)(caveat_text)
        else:
            st.error(f"Missing: `{_display_path(selected_path)}`")
else:
    st.error(
        "Missing simulation reports. Run `python -m src.models.simulation_fan` to regenerate them."
    )
source_line("`src/models/ensemble.py`")

st.subheader("4.4 SVAR — structural systems for both CPI measures")
st.latex(
    r"y_t = c + A_1y_{t-1} + A_2y_{t-2} + u_t,\quad u_t\sim(0,\Sigma) \qquad "
    r"\Sigma = PP',\ \varepsilon_t = P^{-1}u_t \qquad \Theta_h = \Phi_h P"
)
st.markdown(
    "Two five-variable VAR(2)-in-levels systems (System A → `cpi_yoy`, System B → "
    "`trimmed_mean_cpi_yoy`), recursive Cholesky ordering commodity growth → "
    "unemployment → CPI target → business inflation expectations → cash rate, 80% "
    "block-bootstrap IRF bands. **Read every panel as illustrative, not causal** — both "
    "systems still reject multivariate residual whiteness and normality."
)
irf_path = PROJECT_ROOT / "reports/tableau/svar_irf.csv"
if irf_path.exists():
    irf = load_report(irf_path)
    irf = irf[(irf["system"] == "System A") & (irf["response"] == "cpi_yoy")]
    max_horizon = st.slider("Horizons shown", 1, 8, value=8, key="svar_horizon")
    st.altair_chart(build_svar_irf_chart(irf, max_horizon, theme_type), width="stretch")
else:
    st.error(f"Missing: `{_display_path(irf_path)}`")
source_line("`src/models/svar.py`, exported to `reports/tableau/svar_irf.csv`")

st.subheader("4.5 Scenario Engine — SVAR shocks meet the Ensemble, live")
st.latex(
    r"\Delta_h = \text{shock}\times \mathrm{IRF}_h(\text{driver}\to\text{target}),\quad"
    r"\text{shock} = \text{input} - \mathrm{SVAR}_{h=1} \qquad"
    r"\hat y^{*}_{h,i} = \hat y_{ens,h,i} + \Delta_{h,i}"
)
st.markdown(
    "Only the *surprise* over SVAR's own horizon-1 forecast counts, so the expected macro "
    "path is never double-counted; draws are paired index-for-index, never randomly "
    "matched. Served live from `POST /forecast/scenario` — pick a target, a shock "
    "variable, and a shock size below."
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
source_line("`src/models/scenario.py`, served at `POST /forecast/scenario`")

st.subheader("4.6 RBA Policy Classifier — seven readings of one decision")
st.latex(
    r"\text{action} = \begin{cases}\text{cut} & \hat y < 2.0 \\ \text{hike} & \hat y > 3.0 \\"
    r"\text{hold} & \text{otherwise}\end{cases} \qquad "
    r"i^{*} = r^{*} + \pi + \varphi_\pi(\pi-2.5) + \varphi_u\,\Delta u_{t-1} \qquad "
    r"P(Y\le j) = F(\theta_j - x'\beta)"
)
st.markdown(
    "Left to right: the transparent **threshold** baseline (reportable, macro-F1 0.775); "
    "the **Taylor rule** (fixed and estimated variants); **ordered logit/probit** "
    "($F$ = logistic or $\\Phi$). Frank–Hall XGBoost decomposes the same ordinal target "
    "into $K{-}1$ cumulative binary classifiers; majority vote takes the mode of the "
    "first four, tie-broken by threshold's own call. The threshold rule takes the "
    "**headline** CPI forecast because the 2–3% band is a target for CPI inflation; "
    "trimmed mean enters ordered logit/probit and XGBoost as a feature. Run through the "
    "same rule instead, the trimmed-mean forecast scores lower on the same 41 test "
    "quarters (macro-F1 0.696 vs 0.775), mainly by over-calling cuts — a sensitivity "
    "check on that sample, not a model-selection step."
)
if st.button("Refresh RBA action", key="refresh_rba_action"):
    rba_payload, rba_error = api_request("GET", api_base_url, "/rba-action")
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
source_line("`src/models/rba_classifier.py`, served at `GET /rba-action`")

st.subheader("4.7 Credit Stress & Illustrative ECL")
st.latex(
    r"PD_{stressed} = \operatorname{clip}\!\big(PD_{base} + s\cdot\Delta u_{cum}/100,\ PD_{base},\ 1.0\big)"
    r"\qquad ECL = \dfrac{PD_{stressed}\times LGD\times EAD}{(1+r)^{0.5}} \qquad "
    r"ECL_{weighted} = \!\!\sum_{scenario}\!\! w_{scenario}\,ECL_{scenario}"
)
st.markdown(
    "NAB's own disclosed FY2025 Pillar 3 PD/LGD/EAD, stressed with generic RBA "
    "sensitivity coefficients, mid-year discounted at RBA's published lending rates, and "
    "combined with NAB's own 55/42.5/2.5 base/downside/upside scenario weights. **This is "
    "12-month, Stage-1-only and is not a lower bound on NAB's real provision.**"
)
credit_path = PROJECT_ROOT / "reports/tableau/credit_stress.csv"
if credit_path.exists():
    credit = load_report(credit_path)
    scenario = st.radio("Scenario", options=["upside", "base", "downside"], index=1, horizontal=True, key="credit_scenario")
    st.altair_chart(build_credit_stress_chart(credit, scenario, theme_type), width="stretch")
    weighted = credit.drop_duplicates("segment")[["segment", "ecl_aud_m_12m_probability_weighted"]]
    weighted.columns = ["Segment", "Probability-weighted 12m ECL ($AUDm)"]
    st.dataframe(weighted, width="stretch", hide_index=True)
else:
    st.error(f"Missing: `{_display_path(credit_path)}`")
source_line("`src/models/credit_stress.py`, served at `GET /credit-risk/stress-test`")

# --- 5. Evaluation -------------------------------------------------
st.header("5. Evaluation")
st.markdown(
    "Walk-forward comparison on the shared horizon-1-to-8 grid, against seasonal-naive "
    "and the RBA's own published forecasts."
)
comparison_path = PROJECT_ROOT / "reports/model_comparison_all.csv"
coverage_path = PROJECT_ROOT / "reports/model_interval_coverage.csv"
eval_cols = st.columns(2)
if comparison_path.exists():
    with eval_cols[0]:
        st.altair_chart(build_rmse_bar_chart(load_report(comparison_path), theme_type), width="stretch")
        st.caption("Walk-forward accuracy — headline CPI YoY")
if coverage_path.exists():
    with eval_cols[1]:
        st.altair_chart(build_coverage_bar_chart(load_report(coverage_path), theme_type), width="stretch")
        st.caption("80% interval coverage, overall — all three families sit below nominal")
st.markdown(
    "Every family sits below its 80% nominal interval coverage — a known, tracked gap, "
    "not something papered over here. SARIMA's prediction intervals are closest to "
    "nominal; the Ensemble wins on point-forecast RMSE."
)

st.subheader("5.1 Diagnostics by horizon")
st.caption(
    "Reads `reports/model_comparison*.csv`, `reports/model_interval_coverage*.csv`, and "
    "`reports/backtest_predictions*.csv`, generated by the walk-forward evaluation pipeline."
)
diagnostics_target_label = st.radio("Target", options=list(DIAGNOSTICS_TARGET_CONFIG), key="diagnostics_target")
diagnostics_target = DIAGNOSTICS_TARGET_CONFIG[diagnostics_target_label]
diagnostics_paths = {
    "model comparison": diagnostics_target["comparison"],
    "interval coverage": diagnostics_target["coverage"],
    "backtest predictions": diagnostics_target["backtests"],
}
diagnostics_missing = [_display_path(path) for path in diagnostics_paths.values() if not path.exists()]
if diagnostics_missing:
    st.error(
        "Missing diagnostic report artifacts: " + ", ".join(f"`{path}`" for path in diagnostics_missing)
        + ". Regenerate the model comparison, interval coverage, and walk-forward reports first."
    )
else:
    diag_comparison = load_report(diagnostics_target["comparison"])
    diag_coverage = load_report(diagnostics_target["coverage"])
    diag_backtests = load_report(diagnostics_target["backtests"])
    st.caption(diagnostics_target["caption"])

    st.markdown("**Accuracy by horizon**")
    accuracy_rows = _horizon_rows(diag_comparison)
    accuracy_models = sorted(accuracy_rows["model"].dropna().unique())
    selected_accuracy_models = st.multiselect(
        "Accuracy models", options=accuracy_models, default=accuracy_models, key="diagnostics_accuracy_models"
    )
    accuracy_view = accuracy_rows.loc[accuracy_rows["model"].isin(selected_accuracy_models)].copy()
    if accuracy_view.empty:
        st.info("Select at least one model to show the accuracy chart.")
    else:
        st.altair_chart(build_accuracy_chart(accuracy_view, theme_type), width="stretch")
    overall = diag_comparison.loc[
        diag_comparison["horizon"].astype(str).eq("overall") & diag_comparison["model"].isin(selected_accuracy_models)
    ].copy()
    with st.expander("Overall accuracy table"):
        st.dataframe(overall[["model", "n", "rmse", "mae", "best_model", "evaluation_status"]].round(4), width="stretch", hide_index=True)

    st.markdown("**Calibrated simulation interval coverage**")
    st.caption(
        "Empirical coverage vs. the calibrated simulation interval (80% nominal target); "
        "flagged points mark rows the report's binomial test labels significantly miscalibrated."
    )
    coverage_rows = _horizon_rows(diag_coverage)
    coverage_models = sorted(coverage_rows["model"].dropna().unique())
    selected_coverage_models = st.multiselect(
        "Coverage models", options=coverage_models, default=coverage_models, key="diagnostics_coverage_models"
    )
    coverage_view = coverage_rows.loc[coverage_rows["model"].isin(selected_coverage_models)].copy()
    if coverage_view.empty:
        st.info("Select at least one model to show the coverage chart.")
    else:
        st.altair_chart(build_coverage_chart(coverage_view, theme_type), width="stretch")
    if coverage_models:
        detail_model = st.selectbox("Coverage detail model", options=coverage_models, key="diagnostics_coverage_detail")
        detail = coverage_rows.loc[coverage_rows["model"].eq(detail_model)].copy()
        detail_columns = [
            c for c in [
                "horizon", "n", "nominal_coverage", "empirical_coverage", "coverage_ci_lower",
                "coverage_ci_upper", "significantly_miscalibrated", "mean_interval_width",
            ] if c in detail.columns
        ]
        st.dataframe(detail[detail_columns].round(4), width="stretch", hide_index=True)

    st.markdown("**Forecast vs. actual (walk-forward backtests)**")
    st.caption(
        "Walk-forward backtest predictions — distinct from §4.3's unconditional "
        "full-history simulation fans."
    )
    scatter_rows = diag_backtests.dropna(subset=["actual", "forecast"]).copy()
    scatter_models = sorted(scatter_rows["model"].dropna().unique())
    selected_scatter_models = st.multiselect(
        "Forecast-vs-actual models", options=scatter_models, default=scatter_models, key="diagnostics_scatter_models"
    )
    scatter_view = scatter_rows.loc[scatter_rows["model"].isin(selected_scatter_models)].copy()
    if scatter_view.empty:
        st.info("Select at least one model to show the forecast-vs-actual scatter.")
    else:
        st.altair_chart(build_scatter_chart(scatter_view, theme_type), width="stretch")
    with st.expander("Backtest prediction sample"):
        st.dataframe(scatter_view.head(500).round(4), width="stretch", hide_index=True)

render_historical_backtest_panel(
    rba_classifier_report_path=RBA_CLASSIFIER_REPORT_PATH,
    missing_report_label="reports/rba_classifier_evaluation.md",
    coverage_reports=tuple((label, cfg["coverage"]) for label, cfg in DIAGNOSTICS_TARGET_CONFIG.items()),
    coverage_columns=(
        "target", "model", "n", "nominal_coverage", "empirical_coverage",
        "significantly_miscalibrated", "mean_interval_width",
    ),
    coverage_metrics=(
        {"label": "Headline coverage", "target": "Headline", "model": "ensemble", "column": "empirical_coverage", "format": "percent"},
        {"label": "Trimmed mean coverage", "target": "Trimmed mean", "model": "ensemble", "column": "empirical_coverage", "format": "percent"},
    ),
    static_metrics=(("Nominal target", "80%"),),
    caption=(
        "The classifier report does not publish a Brier-style score, so this page does "
        "not recompute one. Coverage rates are from the committed calibrated simulation "
        "interval reports with an 80% nominal target."
    ),
)

# --- 6. Deployment -------------------------------------------------
st.header("6. Deployment")
st.markdown(
    """
    - **FastAPI on Cloud Run** — redeployed and verified live 2026-08-29, serving
      `/forecast/all`, `/forecast/trimmed-mean/all`, and `/forecast/scenario` from baked
      local MLflow runs. Pushes do not auto-redeploy; manual rebuild required after
      serving changes.
    - **`GET /rba-action`** and **`GET /credit-risk/stress-test`** — implemented locally
      in `api/main.py` after that verified image; not yet part of the deployed Cloud Run
      image.
    - **Streamlit** — this single report calls FastAPI live in three sections above and
      reads local reports everywhere else; nothing fits models in-process.
    - **No MLflow champion/registry promotion** — `/forecast/all` serves every family's
      latest finished run directly.
    """
)

st.subheader("6.1 Live forecast tool")
st.caption(
    "Served live from `POST /forecast/all` and `POST /forecast/trimmed-mean/all` on the "
    "FastAPI service configured in the sidebar."
)
latest_inputs = _latest_macro_inputs(curated)
st.markdown(f"**Raw macro inputs** — latest curated quarter: `{latest_inputs['quarter']}`.")
input_cols = st.columns(3)
for index, (column, label) in enumerate(MACRO_INPUT_COLUMNS.items()):
    value = latest_inputs.get(column)
    display = "n/a" if pd.isna(value) else f"{float(value):.2f}"
    input_cols[index % 3].metric(label, display)

live_control_cols = st.columns(2)
with live_control_cols[0]:
    live_target_label = st.radio("Target detail", options=list(FORECAST_TARGET_CONFIG), horizontal=True, key="live_forecast_target")
with live_control_cols[1]:
    live_horizon = st.slider("Forecast horizon (quarters)", min_value=1, max_value=MAX_FORECAST_HORIZON, value=MAX_FORECAST_HORIZON, key="live_forecast_horizon")

if st.button("Load live forecasts", key="load_live_forecasts"):
    live_payloads_by_target: dict[str, dict] = {}
    live_fetch_error = None
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

st.subheader("6.2 Forecast vs. actual (published quarters)")
st.caption(
    "From `reports/forecast_snapshot_accuracy.csv`, generated by "
    "`python -m src.models.forecast_snapshot snapshot` then `compare` once "
    "`DATABASE_URL` is configured. Actuals are joined at read time only — "
    "Postgres never stores a copy of the curated CPI series."
)
if not ACCURACY_REPORT_PATH.exists():
    st.info(
        "No forecast snapshot accuracy report yet. Configure `DATABASE_URL`, then run "
        "`python -m src.models.forecast_snapshot snapshot` and `python -m src.models.forecast_snapshot compare`."
    )
else:
    accuracy = pd.read_csv(ACCURACY_REPORT_PATH)
    observed = accuracy["status"].eq("observed")
    hit_mask = accuracy.loc[observed, "hit"].astype(str).eq("True")
    accuracy_cols = st.columns(3)
    accuracy_cols[0].metric("Snapshots observed", f"{int(observed.sum())}")
    accuracy_cols[1].metric("Snapshots pending", f"{int((~observed).sum())}")
    accuracy_cols[2].metric("Simulation interval hit rate", f"{hit_mask.mean() * 100:.0f}%" if len(hit_mask) else "n/a")
    st.dataframe(accuracy.round(4), width="stretch")

# --- 7. Conclusion -------------------------------------------------
st.header("7. Conclusion")
st.markdown(
    """
    Blending SARIMA with the regularized macro model (the Ensemble) edges out either
    one alone on the shared grid, while Elastic Net on its own does not beat plain
    SARIMA in pooled RMSE — real, if modest, evidence that structure helps at this
    sample size without over-engineering the model. Against the RBA's own published
    headline forecast, scored on the same origins and horizons (each against its own
    actual), the models are comparable rather than demonstrably better or worse: the
    Ensemble's pooled RMSE is 1.640 against 1.690, and the gap is within sampling
    noise. The SVAR/scenario
    layer adds a second, separate kind of value — *why* a
    shock might move CPI — at the honestly-stated cost of two systems that still fail
    their own residual diagnostics.

    **Everything above is live in this one report:** the API base URL is configurable in
    the sidebar; every live section (§4.5, §4.6, §6.1) degrades to a static fallback or
    a clear banner if that API isn't reachable, rather than blanking the page. The live
    API's own `/docs` endpoint (see `README.md`) exposes every endpoint directly.
    """
)
