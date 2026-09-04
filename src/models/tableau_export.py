"""Export a bounded set of flat CSVs for the Tableau executive dashboard.

Calls the running FastAPI service over HTTP (same contract Streamlit uses)
rather than loading or fitting models in-process, per this project's
Streamlit/FastAPI serving separation. Reads only the small, already-generated
comparison/coverage reports and metadata -- never the raw curated feature
matrix or dataset/ downloads.

Usage:
    uvicorn api.main:app --reload   # in one terminal
    python -m src.models.tableau_export   # in another

Writes to reports/tableau/:
    forecast.csv              -- point + interval, all families, both targets
    rba_action.csv             -- all classifier rows from GET /rba-action
    credit_stress.csv          -- PD stress-test rows from GET /credit-risk/stress-test
    model_comparison.csv       -- overall RMSE/MAE per family, both targets
    model_interval_coverage.csv -- overall coverage per family, both targets
    dataset_overview.csv       -- one-row summary of the curated dataset
    historical_indicators.csv  -- tidy full-history series for every distinct
                                   exogenous macro variable plus both CPI
                                   targets (raw levels, not the lag/interaction
                                   engineering copies), with a per-variable
                                   z-scored `value_standardized` column
                                   alongside the raw `value`; includes actual
                                   CPI, so it blends with forecast.csv on
                                   `quarter` to lead a trend chart into the
                                   forecast
    elastic_net_feature_importance.csv -- tidy per-horizon Elastic Net
                                   coefficients for both targets from
                                   reports/elastic_net_coefficients*.csv, with
                                   a readable feature_label and an
                                   is_exogenous flag separating macro drivers
                                   from the model's own autoregressive lags
    svar_irf.csv             -- target responses to one-SD recursive Cholesky
                                structural shocks in the four SVAR exogenous
                                variables, with bootstrap bands
    drift_error_check.csv     -- every forecast that now has a real outcome,
                                compared against that model's own historical
                                walk-forward error distribution at that horizon
    drift_error_history.csv   -- horizon-1 error time series per target x
                                model, historical origins plus any newly
                                graded origin, flagged by `is_recent`
    drift_covariate_check.csv -- latest reading per macro variable ranked
                                against its own full history, flagging inputs
                                sitting at unusual levels right now

    The three drift_*.csv exports never fit or refit a model -- they only
    compare already-generated forecasts and reports, per the guardrail in
    .ai/TABLEAU_DASHBOARD_GUIDE.md against refitting as a side effect of a
    routine data refresh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from src.models import drift_monitor, svar

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURATED_DATA_PATH = PROJECT_ROOT / "data/curated/quarterly_macro_features.csv"
SERIES_AVAILABILITY_PATH = PROJECT_ROOT / "data/metadata/series_availability.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/tableau"
DEFAULT_API_BASE_URL = "http://localhost:8000"
SVAR_EXPORT_FORECAST_ORIGIN = svar.FORECAST_ORIGIN_PIN
SVAR_EXPORT_HORIZONS = tuple(range(1, 9))

HISTORICAL_INDICATOR_COLUMNS = {
    "cpi_yoy": "Headline CPI YoY",
    "trimmed_mean_cpi_yoy": "Trimmed Mean CPI YoY",
    "unemployment_rate": "Unemployment Rate",
    "cash_rate": "Cash Rate",
    "wpi_growth": "Wage Price Growth",
    "ppi_growth": "Producer Price Growth",
    "commodity_growth": "Commodity Price Growth",
    "wti_growth": "WTI Oil Price Growth",
    "brent_growth": "Brent Oil Price Growth",
    "aud_usd_change": "AUD/USD Exchange Rate Change",
    "household_spending_growth": "Household Spending Growth",
    "inflation_expectations_business": "Business Inflation Expectations",
}

ELASTIC_NET_COEFFICIENT_REPORTS = {
    "Headline": PROJECT_ROOT / "reports/elastic_net_coefficients.csv",
    "Trimmed mean": PROJECT_ROOT / "reports/elastic_net_coefficients_trimmed_mean.csv",
}

# Readable labels for every feature name that appears in either coefficient
# report. Own-target lags are flagged separately (AUTOREGRESSIVE_FEATURES)
# rather than dropped, so Tableau can show how much of the forecast is
# inertia versus exogenous macro drivers.
FEATURE_LABELS = {
    "cash_rate_change_lag1": "Cash Rate Change (lag 1)",
    "cash_rate_change_lag1_x_unemployment_rate_change_lag1": (
        "Cash Rate Change x Unemployment Rate Change Interaction (lag 1)"
    ),
    "commodity_growth_lag1": "Commodity Price Growth (lag 1)",
    "commodity_growth_lag1_sq": "Commodity Price Growth Squared (lag 1)",
    "cpi_yoy_lag1": "Headline CPI YoY (lag 1, autoregressive)",
    "cpi_yoy_lag4": "Headline CPI YoY (lag 4, autoregressive)",
    "inflation_expectations_business_lag1": "Business Inflation Expectations (lag 1)",
    "ppi_growth_lag2": "Producer Price Growth (lag 2)",
    "ppi_growth_lag2_sq": "Producer Price Growth Squared (lag 2)",
    "trimmed_mean_cpi_yoy_lag1": "Trimmed Mean CPI YoY (lag 1, autoregressive)",
    "trimmed_mean_cpi_yoy_lag4": "Trimmed Mean CPI YoY (lag 4, autoregressive)",
    "unemployment_rate_change_lag1": "Unemployment Rate Change (lag 1)",
    "wti_growth_lag1": "WTI Oil Price Growth (lag 1)",
    "wti_growth_lag1_sq": "WTI Oil Price Growth Squared (lag 1)",
}

AUTOREGRESSIVE_FEATURES = {
    "cpi_yoy_lag1",
    "cpi_yoy_lag4",
    "trimmed_mean_cpi_yoy_lag1",
    "trimmed_mean_cpi_yoy_lag4",
}

FORECAST_TARGETS = {
    "Headline": {
        "endpoint": "/forecast/all",
        "comparison_report": PROJECT_ROOT / "reports/model_comparison_all.csv",
        "coverage_report": PROJECT_ROOT / "reports/model_interval_coverage.csv",
    },
    "Trimmed mean": {
        "endpoint": "/forecast/trimmed-mean/all",
        "comparison_report": PROJECT_ROOT / "reports/model_comparison_trimmed_mean_all.csv",
        "coverage_report": PROJECT_ROOT / "reports/model_interval_coverage_trimmed_mean.csv",
    },
}

SVAR_SYSTEMS = {
    "System A": {
        "target": "cpi_yoy",
        "columns": svar.SYSTEM_A_COLUMNS,
        "ordering": svar.SYSTEM_A_CHOLESKY_ORDER,
    },
    "System B": {
        "target": "trimmed_mean_cpi_yoy",
        "columns": svar.SYSTEM_B_COLUMNS,
        "ordering": svar.SYSTEM_B_CHOLESKY_ORDER,
    },
}


def _fetch_forecast_payload(api_base_url: str, endpoint: str, horizon: int) -> dict:
    response = requests.post(f"{api_base_url}{endpoint}", json={"horizon": horizon}, timeout=60)
    response.raise_for_status()
    return response.json()


def _fetch_rba_action_payload(api_base_url: str) -> dict:
    response = requests.get(f"{api_base_url}/rba-action", timeout=60)
    response.raise_for_status()
    return response.json()


def _fetch_credit_risk_stress_test_payload(api_base_url: str) -> dict:
    response = requests.get(f"{api_base_url}/credit-risk/stress-test", timeout=60)
    response.raise_for_status()
    return response.json()


def build_forecast_frame(payloads_by_target: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for target_label, payload in payloads_by_target.items():
        for model in payload.get("models", []):
            for horizon, (quarter, forecast, lower, upper) in enumerate(
                zip(
                    model["quarters"],
                    model["forecast"],
                    model["interval_lower"],
                    model["interval_upper"],
                    strict=True,
                ),
                start=1,
            ):
                rows.append(
                    {
                        "target": target_label,
                        "model_family": model["model_family"],
                        "forecast_origin": model["forecast_origin"],
                        "quarter": quarter,
                        "horizon": horizon,
                        "forecast": forecast,
                        "interval_lower": lower,
                        "interval_upper": upper,
                    }
                )
    return pd.DataFrame(rows)


def build_rba_action_frame(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame(payload.get("models", []))
    if frame.empty:
        return frame
    frame["target_quarter"] = payload["target_quarter"]
    frame["forecast_origin"] = payload["forecast_origin"]
    frame["headline_forecast"] = payload["headline_forecast"]
    frame["trimmed_mean_forecast"] = payload["trimmed_mean_forecast"]
    frame["reportable_model"] = payload["reportable_model"]
    frame["reportable_action"] = payload["reportable_action"]
    return frame


def build_credit_stress_frame(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame(payload.get("segments", []))
    if frame.empty:
        return frame
    frame["forecast_origin"] = payload["forecast_origin"]
    frame["target_quarter"] = payload["target_quarter"]
    frame["horizon"] = payload["horizon"]
    frame["delta_unemployment_cumulative"] = payload["delta_unemployment_cumulative"]
    return frame


def build_model_comparison_frame() -> pd.DataFrame:
    frames = []
    for target_label, config in FORECAST_TARGETS.items():
        path = config["comparison_report"]
        if not path.exists():
            continue
        report = pd.read_csv(path)
        overall = report.loc[report["horizon"].astype(str).eq("overall"), ["model", "n", "rmse", "mae"]].copy()
        # rba mixes an NSA-basis benchmark against this SA-basis target; STATUS_RULES.md
        # requires --no-rba for SA-sourced comparisons, so it never belongs in this export.
        overall = overall.loc[overall["model"].ne("rba")]
        overall["target"] = target_label
        frames.append(overall)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)[["target", "model", "n", "rmse", "mae"]]


def build_coverage_frame() -> pd.DataFrame:
    frames = []
    for target_label, config in FORECAST_TARGETS.items():
        path = config["coverage_report"]
        if not path.exists():
            continue
        report = pd.read_csv(path)
        overall = report.loc[
            report["horizon"].astype(str).eq("overall"),
            ["model", "n", "nominal_coverage", "empirical_coverage", "significantly_miscalibrated"],
        ].copy()
        overall["target"] = target_label
        frames.append(overall)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)[
        ["target", "model", "n", "nominal_coverage", "empirical_coverage", "significantly_miscalibrated"]
    ]


def build_dataset_overview_frame() -> pd.DataFrame:
    indicator_columns = list(HISTORICAL_INDICATOR_COLUMNS)
    curated = pd.read_csv(CURATED_DATA_PATH, usecols=["quarter", *indicator_columns])
    observed = curated.loc[curated[indicator_columns].notna().any(axis=1)]
    quarters = observed["quarter"].astype(str)
    series_availability = pd.read_csv(SERIES_AVAILABILITY_PATH)
    return pd.DataFrame(
        [
            {
                "quarter_range_start": quarters.min(),
                "quarter_range_end": quarters.max(),
                "n_quarters": len(quarters),
                "n_source_series": len(series_availability),
                "n_source_families": series_availability["source_family"].nunique(),
                "curated_dataset_last_modified": pd.Timestamp(
                    CURATED_DATA_PATH.stat().st_mtime, unit="s"
                ).strftime("%Y-%m-%d"),
            }
        ]
    )


def build_historical_indicators_frame() -> pd.DataFrame:
    columns = list(HISTORICAL_INDICATOR_COLUMNS)
    curated = pd.read_csv(CURATED_DATA_PATH, usecols=["quarter", *columns])
    long_frame = curated.melt(id_vars="quarter", value_vars=columns, var_name="variable", value_name="value")
    long_frame["variable_label"] = long_frame["variable"].map(HISTORICAL_INDICATOR_COLUMNS)
    long_frame = long_frame.dropna(subset=["value"]).sort_values(["variable", "quarter"]).reset_index(drop=True)
    # Z-score per variable over its own full available history -- puts series with very
    # different natural units (%, index points, exchange rate) on one comparable axis
    # without depending on an arbitrary rebase quarter, while `value` keeps the raw units.
    stats = long_frame.groupby("variable")["value"].agg(["mean", "std"])
    long_frame = long_frame.join(stats, on="variable")
    long_frame["value_standardized"] = (long_frame["value"] - long_frame["mean"]) / long_frame["std"]
    return long_frame.drop(columns=["mean", "std"])


def build_feature_importance_frame() -> pd.DataFrame:
    frames = []
    for target_label, path in ELASTIC_NET_COEFFICIENT_REPORTS.items():
        if not path.exists():
            continue
        report = pd.read_csv(path, usecols=["horizon", "feature", "coef"])
        report["target"] = target_label
        frames.append(report)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # coef is read off a StandardScaler -> ElasticNet pipeline (elastic_net.py),
    # so magnitudes are already on a comparable standardized scale across
    # features -- abs_coef is a valid cross-feature importance ranking as-is.
    combined["feature_label"] = combined["feature"].map(FEATURE_LABELS).fillna(combined["feature"])
    combined["is_exogenous"] = ~combined["feature"].isin(AUTOREGRESSIVE_FEATURES)
    combined["abs_coef"] = combined["coef"].abs()
    return combined[["target", "horizon", "feature", "feature_label", "is_exogenous", "coef", "abs_coef"]]


def build_svar_irf_frame() -> pd.DataFrame:
    frames = []
    for system, config in SVAR_SYSTEMS.items():
        target = str(config["target"])
        frame = svar.load_svar_level_frame(columns=config["columns"])
        frame = frame.loc[frame.index <= SVAR_EXPORT_FORECAST_ORIGIN]
        fitted = svar.fit_svar(frame, ordering=config["ordering"])
        point = svar.recursive_cholesky_irfs(
            fitted,
            horizons=SVAR_EXPORT_HORIZONS,
        )
        bands = svar.bootstrap_cholesky_irf_bands(
            fitted,
            horizons=SVAR_EXPORT_HORIZONS,
        )
        combined = point.merge(
            bands.drop(columns=["irf"]),
            on=["response", "shock", "horizon"],
            how="left",
        )
        filtered = combined.loc[
            combined["response"].eq(target)
            & combined["shock"].isin(svar.COMMON_MACRO_COLUMNS)
        ].copy()
        filtered["system"] = system
        filtered["target"] = target
        filtered["forecast_origin"] = str(SVAR_EXPORT_FORECAST_ORIGIN)
        frames.append(filtered)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)[
        [
            "system",
            "target",
            "forecast_origin",
            "response",
            "shock",
            "horizon",
            "irf",
            "lower",
            "upper",
            "lower_quantile",
            "upper_quantile",
            "bootstrap_replications",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        forecast_payloads = {
            target_label: _fetch_forecast_payload(args.api_base_url, config["endpoint"], args.horizon)
            for target_label, config in FORECAST_TARGETS.items()
        }
        rba_action_payload = _fetch_rba_action_payload(args.api_base_url)
        credit_stress_payload = _fetch_credit_risk_stress_test_payload(args.api_base_url)
    except requests.RequestException as exc:
        raise SystemExit(
            f"Could not reach the FastAPI service at {args.api_base_url}: {exc}\n"
            "Start it first with: uvicorn api.main:app --reload"
        ) from exc

    forecast_frame = build_forecast_frame(forecast_payloads)
    historical_frame = build_historical_indicators_frame()

    forecast_frame.to_csv(args.output_dir / "forecast.csv", index=False)
    build_rba_action_frame(rba_action_payload).to_csv(args.output_dir / "rba_action.csv", index=False)
    build_credit_stress_frame(credit_stress_payload).to_csv(args.output_dir / "credit_stress.csv", index=False)
    build_model_comparison_frame().to_csv(args.output_dir / "model_comparison.csv", index=False)
    build_coverage_frame().to_csv(args.output_dir / "model_interval_coverage.csv", index=False)
    build_dataset_overview_frame().to_csv(args.output_dir / "dataset_overview.csv", index=False)
    historical_frame.to_csv(args.output_dir / "historical_indicators.csv", index=False)
    build_feature_importance_frame().to_csv(
        args.output_dir / "elastic_net_feature_importance.csv", index=False
    )
    build_svar_irf_frame().to_csv(args.output_dir / "svar_irf.csv", index=False)
    drift_monitor.build_drift_error_check_frame(forecast_frame, historical_frame).to_csv(
        args.output_dir / "drift_error_check.csv", index=False
    )
    drift_monitor.build_drift_error_history_frame(forecast_frame, historical_frame).to_csv(
        args.output_dir / "drift_error_history.csv", index=False
    )
    drift_monitor.build_drift_covariate_frame(historical_frame).to_csv(
        args.output_dir / "drift_covariate_check.csv", index=False
    )

    print(f"Wrote 12 CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
