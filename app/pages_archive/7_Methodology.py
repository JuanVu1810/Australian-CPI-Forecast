"""Methodology — full CRISP-DM writeup, math, and interactive demos.

Reads local report CSVs and curated/metadata files only; never fits models
in-process, matching every other page in this app. Ported from
``notebooks/methodology_report.ipynb`` (same numbers, same equations), styled
as a long-form article rather than a notebook.
"""

from __future__ import annotations

import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

PALETTE = {
    "light": {
        "fan": "#eb6834", "median": "#a33d19", "muted": "#898781", "grid": "#e1e0d9",
        "ink": "#1c1b1a", "paper": "#fbfaf8", "border": "#e5e2dc", "accent2": "#3a6b63",
    },
    "dark": {
        "fan": "#d95926", "median": "#ffb08f", "muted": "#a8a49c", "grid": "#2c2c2a",
        "ink": "#ece9e4", "paper": "#15140f", "border": "#33312b", "accent2": "#7fc4b8",
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


def _fan_frame(report: pd.DataFrame) -> pd.DataFrame:
    frame = report.copy()
    frame["quarter_date"] = (
        pd.PeriodIndex(frame["target_quarter"], freq="Q").to_timestamp(how="end").normalize()
    )
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    return frame.sort_values("horizon")


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
        .properties(height=380)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


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
        "accuracy and interval-coverage diagnostics shown on the Diagnostics page.",
    )


# ---------------------------------------------------------------------------
# Medium-style presentation layer
# ---------------------------------------------------------------------------
def inject_article_css(theme_type: str) -> None:
    colors = PALETTE[theme_type]
    st.markdown(
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
            font-size: 1.08rem;
            line-height: 1.72;
            max-width: 740px;
          }}
          [data-testid="stMarkdownContainer"] h2 {{
            font-weight: 700;
            border-top: 1px solid {colors['border']};
            padding-top: 1.6rem;
            margin-top: 0.6rem;
          }}
          [data-testid="stMarkdownContainer"] h3 {{
            font-weight: 600;
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
            font-size: 1.25rem;
            line-height: 1.6;
            color: {colors['muted']};
            font-style: italic;
            max-width: 740px;
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
            max-width: 740px;
            margin: 0.5rem 0 1rem 0;
          }}
          .cpi-eq-caption {{
            font-size: 0.85rem;
            color: {colors['muted']};
            font-style: italic;
            max-width: 740px;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kicker(text: str) -> None:
    st.markdown(f"<div class='cpi-kicker'>{text}</div>", unsafe_allow_html=True)


def dek(text: str) -> None:
    st.markdown(f"<div class='cpi-dek'>{text}</div>", unsafe_allow_html=True)


def source_line(text: str) -> None:
    st.markdown(f"<div class='cpi-source'>Source: {text}</div>", unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f"<div class='cpi-callout'>{text}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section-specific chart builders (each reads a local report only)
# ---------------------------------------------------------------------------
@st.cache_data
def load_curated_history() -> pd.DataFrame:
    df = pd.read_csv(
        PROJECT_ROOT / "data/curated/quarterly_macro_features.csv",
        usecols=["quarter", "cpi_yoy", "trimmed_mean_cpi_yoy"],
    )
    df["quarter_date"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp(how="end")
    return df


def build_cpi_history_chart(history: pd.DataFrame, theme_type: str) -> alt.LayerChart:
    colors = PALETTE[theme_type]
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
        .encode(
            x="quarter_date:T",
            y="trimmed_mean_cpi_yoy:Q",
            tooltip=["quarter:N", "trimmed_mean_cpi_yoy:Q"],
        )
    )
    return (
        (band + headline + trimmed)
        .properties(height=340)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


def build_ensemble_path_chart(
    sample: pd.DataFrame,
    fan: pd.DataFrame,
    n_reveal: int,
    theme_type: str,
) -> alt.LayerChart:
    colors = PALETTE[theme_type]
    revealed = sample[sample["draw_id"] < n_reveal]

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
    lines = (
        alt.Chart(revealed)
        .mark_line(strokeWidth=0.8, opacity=0.35, color=colors["fan"])
        .encode(
            x="horizon:Q",
            y=alt.Y("value:Q", title="Simulated headline CPI YoY (%)"),
            detail="draw_id:N",
        )
    )
    median_line = (
        alt.Chart(fan)
        .mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=40), color=colors["median"])
        .encode(x="horizon:Q", y="median:Q")
    )
    return (
        (outer_band + inner_band + lines + median_line)
        .properties(height=380)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


def build_svar_irf_chart(irf: pd.DataFrame, max_horizon: int, theme_type: str) -> alt.Chart:
    colors = PALETTE[theme_type]
    subset = irf[irf["horizon"] <= max_horizon]
    zero_line = (
        alt.Chart()
        .mark_rule(color=colors["muted"], strokeDash=[3, 3])
        .encode(y=alt.datum(0))
    )
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
    layered = alt.layer(zero_line, band, line, data=subset).properties(height=180, width=320)
    return (
        layered.facet(facet=alt.Facet("shock:N", title=None), columns=2)
        .resolve_scale(y="independent")
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_header(labelColor=colors["ink"], labelFontWeight="bold")
    )


def build_rba_probability_chart(rba: pd.DataFrame, theme_type: str) -> alt.Chart:
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
        .properties(height=220)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
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
        .properties(height=280)
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
        .properties(height=220)
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
        .properties(height=260)
        .configure_axis(gridColor=colors["grid"], labelColor=colors["muted"], titleColor=colors["muted"])
        .configure_view(strokeWidth=0)
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Methodology | Australian CPI Forecast", layout="wide")
theme_type = _current_theme()
inject_article_css(theme_type)

kicker("Australian CPI Forecasting Project · Methodology")
st.title("How this project forecasts Australian inflation")
dek(
    "Seven methodologies, one CRISP-DM story — the math behind every model, and a live "
    "look at what “simulating a forecast” actually means."
)
st.write("")

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
history = load_curated_history()
st.altair_chart(build_cpi_history_chart(history, theme_type), width="stretch")
st.caption(
    f"{len(history)} quarters, {history['quarter'].iloc[0]}–{history['quarter'].iloc[-1]}. "
    "Shaded band is the RBA's 2–3% target range."
)

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
    display_cols = ["quarter", "dummy_name", "lead_quarters", "reason"]
    display_cols = [c for c in display_cols if c in interventions.columns]
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
    "watching get built: drag the slider (or press Play) to reveal, one by one, a sample "
    "of the 1,000 Monte-Carlo draws behind it."
)

sample_path = PROJECT_ROOT / "reports/simulation_paths_sample_ensemble.csv"
fan_path = PROJECT_ROOT / "reports/simulation_fan_ensemble.csv"
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
        n_reveal = st.slider("Paths revealed", 1, n_draws, value=n_draws, key="ensemble_reveal")
    chart_placeholder = st.empty()
    with play_col:
        st.write("")
        play = st.button("▶ Play", key="ensemble_play")
    if play:
        for step in range(1, n_draws + 1, max(1, n_draws // 60)):
            chart_placeholder.altair_chart(
                build_ensemble_path_chart(sample, fan, step, theme_type), width="stretch"
            )
            time.sleep(0.03)
        chart_placeholder.altair_chart(
            build_ensemble_path_chart(sample, fan, n_draws, theme_type), width="stretch"
        )
    else:
        chart_placeholder.altair_chart(
            build_ensemble_path_chart(sample, fan, n_reveal, theme_type), width="stretch"
        )

    reported_path = PROJECT_ROOT / "reports/tableau/forecast.csv"
    if reported_path.exists():
        reported = load_report(reported_path)
        reported = reported[
            (reported["model_family"] == "ensemble") & (reported["target"] == "Headline")
        ].sort_values("horizon")
        if not reported.empty:
            h1_reported = float(reported["forecast"].iloc[0])
            h1_median = float(fan.sort_values("horizon")["median"].iloc[0])
            h1_width = float(
                reported["interval_upper"].iloc[0] - reported["interval_lower"].iloc[0]
            )
            raw_width = float(fan.sort_values("horizon")["p90"].iloc[0] - fan.sort_values("horizon")["p10"].iloc[0])
            callout(
                f"<b>The shaded band above is the raw, uncalibrated simulated interval</b> — "
                f"not what <code>/forecast/all</code> actually serves. At horizon 1, this raw "
                f"80% band is {raw_width:.2f} points wide; the calibrated interval actually "
                f"served is {h1_width:.2f} points wide. <code>interval_calibration.py</code> "
                f"found the raw simulated interval under-covers historically, so serving "
                f"rescales it per horizon. The reported point forecast ({h1_reported:.2f}%) and "
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
        "Missing simulation reports. Run `python -m src.models.simulation_fan` to "
        "regenerate them."
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

st.subheader("4.5 Scenario Engine — SVAR shocks meet the Ensemble")
st.latex(
    r"\Delta_h = \text{shock}\times \mathrm{IRF}_h(\text{driver}\to\text{target}),\quad"
    r"\text{shock} = \text{input} - \mathrm{SVAR}_{h=1} \qquad"
    r"\hat y^{*}_{h,i} = \hat y_{ens,h,i} + \Delta_{h,i}"
)
st.markdown(
    "Only the *surprise* over SVAR's own horizon-1 forecast counts, so the expected macro "
    "path is never double-counted; draws are paired index-for-index, never randomly "
    "matched. Try it interactively on the **Scenario Explorer** page in the sidebar."
)
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
    "first four, tie-broken by threshold's own call."
)
rba_path = PROJECT_ROOT / "reports/tableau/rba_action.csv"
if rba_path.exists():
    rba = load_report(rba_path)
    origin = str(rba["forecast_origin"].iloc[0])
    target_q = str(rba["target_quarter"].iloc[0])
    call = str(rba["reportable_action"].iloc[0])
    st.metric(f"Reportable call for {target_q} (origin {origin})", call.upper())
    st.altair_chart(build_rba_probability_chart(rba, theme_type), width="stretch")
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
    scenario = st.radio(
        "Scenario", options=["upside", "base", "downside"], index=1, horizontal=True, key="credit_scenario"
    )
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
    - **Streamlit** — every page here calls FastAPI or reads a local report; none fit
      models in-process, this page included.
    - **No MLflow champion/registry promotion** — `/forecast/all` serves every family's
      latest finished run directly.
    """
)

# --- 7. Conclusion -------------------------------------------------
st.header("7. Conclusion")
st.markdown(
    """
    Regularizing the macro block (Elastic Net) edges out plain SARIMA on the shared
    grid, and blending the two (Ensemble) edges out both — real, if modest, evidence
    that structure helps at this sample size without over-engineering the model. None
    of the three forecast-accuracy families beat the RBA's own published forecast
    outright. The SVAR/scenario layer adds a second, separate kind of value — *why* a
    shock might move CPI — at the honestly-stated cost of two systems that still fail
    their own residual diagnostics.

    **Explore further:** the **Scenario Explorer** and **RBA Policy** pages in the
    sidebar make the scenario engine and policy classifier interactive; the live API's
    `/docs` endpoint (see `README.md`) exposes every endpoint directly.
    """
)
