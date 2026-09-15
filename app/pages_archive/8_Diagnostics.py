"""Backtesting and diagnostics dashboard for precomputed report artifacts.

Reads committed CSV reports from ``reports/model_comparison*.csv``,
``reports/model_interval_coverage*.csv``, and
``reports/backtest_predictions*.csv``. These are read-only visualizations of
the walk-forward evaluation pipeline, not live API calls or report generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.lib.rba_reports import render_historical_backtest_panel


RBA_CLASSIFIER_REPORT_PATH = PROJECT_ROOT / "reports/rba_classifier_evaluation.md"
TARGET_CONFIG = {
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
        "actual": "#2a78d6",
        "forecast": "#eb6834",
        "miscalibrated": "#b83232",
        "nominal": "#898781",
        "grid": "#e1e0d9",
        "muted": "#898781",
    },
    "dark": {
        "actual": "#3987e5",
        "forecast": "#d95926",
        "miscalibrated": "#ff8a8a",
        "nominal": "#898781",
        "grid": "#2c2c2a",
        "muted": "#898781",
    },
}


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


def missing_reports(paths: dict[str, Path]) -> list[str]:
    return [_display_path(path) for path in paths.values() if not path.exists()]


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
        .properties(height=340)
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
            tooltip=[
                "model:N",
                "horizon:N",
                "empirical_coverage:Q",
                "nominal_coverage:Q",
                "significantly_miscalibrated:N",
            ],
        )
    )
    flagged = (
        alt.Chart(report.loc[report["significantly_miscalibrated"].astype(str).eq("True")])
        .mark_point(filled=True, size=110, color=colors["miscalibrated"], shape="diamond")
        .encode(x="horizon_num:O", y="empirical_coverage:Q")
    )
    return (
        (nominal + empirical + flagged)
        .properties(height=340)
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
    reference = (
        alt.Chart(diagonal)
        .mark_line(strokeDash=[4, 3], color=colors["nominal"], strokeWidth=1.5)
        .encode(x="actual:Q", y="forecast:Q")
    )
    return (
        (points + reference)
        .properties(height=380)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


st.set_page_config(page_title="Diagnostics | Australian CPI Forecast", layout="wide")
st.title("Diagnostics")
st.caption(
    "Reads `reports/model_comparison*.csv`, `reports/model_interval_coverage*.csv`, "
    "`reports/backtest_predictions*.csv`, and `reports/rba_classifier_evaluation.md`, "
    "generated by the model comparison, interval coverage, and walk-forward evaluation pipeline."
)

with st.sidebar:
    target_label = st.radio("Target", options=list(TARGET_CONFIG))

target = TARGET_CONFIG[target_label]
paths = {
    "model comparison": target["comparison"],
    "interval coverage": target["coverage"],
    "backtest predictions": target["backtests"],
}
missing = missing_reports(paths)
if missing:
    st.error(
        "Missing diagnostic report artifacts: "
        + ", ".join(f"`{path}`" for path in missing)
        + ". Regenerate the model comparison, interval coverage, and walk-forward reports first."
    )
    st.stop()

comparison = load_report(target["comparison"])
coverage = load_report(target["coverage"])
backtests = load_report(target["backtests"])
st.caption(target["caption"])

st.subheader("Accuracy by horizon")
accuracy_rows = _horizon_rows(comparison)
accuracy_models = sorted(accuracy_rows["model"].dropna().unique())
selected_accuracy_models = st.multiselect(
    "Accuracy models",
    options=accuracy_models,
    default=accuracy_models,
)
accuracy_view = accuracy_rows.loc[accuracy_rows["model"].isin(selected_accuracy_models)].copy()
if accuracy_view.empty:
    st.info("Select at least one model to show the accuracy chart.")
else:
    st.altair_chart(build_accuracy_chart(accuracy_view, _current_theme()), width="stretch")

overall = comparison.loc[
    comparison["horizon"].astype(str).eq("overall") & comparison["model"].isin(selected_accuracy_models)
].copy()
with st.expander("Overall accuracy table", expanded=True):
    st.dataframe(
        overall[["model", "n", "rmse", "mae", "best_model", "evaluation_status"]].round(4),
        width="stretch",
        hide_index=True,
    )

st.subheader("Calibrated simulation interval coverage")
st.caption(
    "Empirical coverage is compared with the calibrated simulation interval "
    "(80% nominal target); flagged points mark rows where the report's binomial "
    "test labels coverage as significantly miscalibrated. Current committed "
    "evidence remains below nominal overall."
)
coverage_rows = _horizon_rows(coverage)
coverage_models = sorted(coverage_rows["model"].dropna().unique())
selected_coverage_models = st.multiselect(
    "Coverage models",
    options=coverage_models,
    default=coverage_models,
)
coverage_view = coverage_rows.loc[coverage_rows["model"].isin(selected_coverage_models)].copy()
if coverage_view.empty:
    st.info("Select at least one model to show the coverage chart.")
else:
    st.altair_chart(build_coverage_chart(coverage_view, _current_theme()), width="stretch")

detail_model = st.selectbox("Coverage detail model", options=coverage_models)
detail = coverage_rows.loc[coverage_rows["model"].eq(detail_model)].copy()
st.dataframe(
    detail[
        [
            "horizon",
            "n",
            "nominal_coverage",
            "empirical_coverage",
            "coverage_ci_lower",
            "coverage_ci_upper",
            "significantly_miscalibrated",
            "mean_interval_width",
        ]
    ].round(4),
    width="stretch",
    hide_index=True,
)

st.subheader("Forecast vs actual")
st.caption(
    "This scatter uses walk-forward backtest predictions and is distinct from "
    "the Methodology page's unconditional full-history simulation fans."
)
scatter_rows = backtests.dropna(subset=["actual", "forecast"]).copy()
scatter_models = sorted(scatter_rows["model"].dropna().unique())
selected_scatter_models = st.multiselect(
    "Forecast-vs-actual models",
    options=scatter_models,
    default=scatter_models,
)
scatter_view = scatter_rows.loc[scatter_rows["model"].isin(selected_scatter_models)].copy()
if scatter_view.empty:
    st.info("Select at least one model to show the forecast-vs-actual scatter.")
else:
    st.altair_chart(build_scatter_chart(scatter_view, _current_theme()), width="stretch")

with st.expander("Backtest prediction sample"):
    st.dataframe(scatter_view.head(500).round(4), width="stretch", hide_index=True)

render_historical_backtest_panel(
    rba_classifier_report_path=RBA_CLASSIFIER_REPORT_PATH,
    missing_report_label="reports/rba_classifier_evaluation.md",
    coverage_reports=tuple((target_label, config["coverage"]) for target_label, config in TARGET_CONFIG.items()),
    coverage_columns=(
        "target",
        "model",
        "n",
        "nominal_coverage",
        "empirical_coverage",
        "significantly_miscalibrated",
        "mean_interval_width",
    ),
    coverage_metrics=(
        {
            "label": "Headline Ensemble coverage",
            "target": "Headline",
            "model": "ensemble",
            "column": "empirical_coverage",
            "format": "percent",
        },
    ),
    static_metrics=(("Nominal target", "80%"),),
    caption=(
        "The classifier report does not publish a Brier-style score, so this page does "
        "not recompute one. Coverage rates are read from the committed calibrated "
        "simulation interval reports."
    ),
)
