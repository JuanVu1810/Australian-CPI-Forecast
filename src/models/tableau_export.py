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
    credit_stress.csv          -- illustrative 12-month ECL rows (segment x
                                   scenario) from GET /credit-risk/stress-test
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
    svar_shock_events.csv    -- located historical own-target structural
                                shocks, standardized within each system's own
                                history
    svar_historical_decomposition.csv -- each target's actual fitted-history
                                path decomposed into the zero-shock baseline,
                                target own-shock, and four exogenous-shock
                                contributions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.models import svar

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


def build_forecast_frame(payloads_by_target: dict[str, dict]) -> pd.DataFrame:
    """One row per target, model and forecast quarter from the /forecast responses."""
    rows = []
    for target_label, payload in payloads_by_target.items():
        for model in payload.get("models", []):
            for index in range(len(model["quarters"])):
                rows.append(
                    {
                        "target": target_label,
                        "model_family": model["model_family"],
                        "forecast_origin": model["forecast_origin"],
                        "quarter": model["quarters"][index],
                        "horizon": index + 1,
                        "forecast": model["forecast"][index],
                        "interval_lower": model["interval_lower"][index],
                        "interval_upper": model["interval_upper"][index],
                    }
                )
    return pd.DataFrame(rows)


def build_rba_action_frame(payload: dict) -> pd.DataFrame:
    """One row per classifier model, with the shared forecast context repeated on each row."""
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
    """Flatten the credit stress payload into one row per segment and scenario."""
    segments = payload.get("segments", [])
    if not segments:
        return pd.DataFrame()

    scenario_weights = {}
    scenario_deltas = {}
    for scenario in payload.get("scenarios", []):
        scenario_weights[scenario["name"]] = scenario["probability_weight"]
        scenario_deltas[scenario["name"]] = scenario["delta_unemployment_cumulative"]

    rows = []
    for segment in segments:
        for scenario_name, pd_stressed in segment["pd_stressed_by_scenario"].items():
            rows.append(
                {
                    "segment": segment["segment"],
                    "scenario": scenario_name,
                    "probability_weight": scenario_weights.get(scenario_name),
                    "delta_unemployment_cumulative": scenario_deltas.get(scenario_name),
                    "pd_base": segment["pd_base"],
                    "ur_sensitivity": segment["ur_sensitivity"],
                    "lgd": segment["lgd"],
                    "ead_aud_m": segment["ead_aud_m"],
                    "discount_rate": segment["discount_rate"],
                    "pd_stressed": pd_stressed,
                    "ecl_aud_m": segment["ecl_aud_m_by_scenario"][scenario_name],
                    "ecl_aud_m_12m_probability_weighted": segment[
                        "ecl_aud_m_12m_probability_weighted"
                    ],
                    "forecast_origin": payload["forecast_origin"],
                    "target_quarter": payload["target_quarter"],
                    "horizon": payload["horizon"],
                }
            )
    return pd.DataFrame(rows)


def build_model_comparison_frame() -> pd.DataFrame:
    """Overall RMSE/MAE per model for both targets, without the RBA row."""
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
    """Overall interval coverage per model for both targets."""
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
    """One-row summary of the curated dataset's date range and sources."""
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
    """Long table of every indicator's full history, raw and z-scored."""
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
    """Elastic Net coefficients per target, horizon and feature, with readable labels."""
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


def _fit_pinned_svar(config: dict):
    """Fit one SVAR system on data up to the export forecast origin."""
    frame = svar.load_svar_level_frame(columns=config["columns"])
    frame = frame.loc[frame.index <= SVAR_EXPORT_FORECAST_ORIGIN]
    return svar.fit_svar(frame, ordering=config["ordering"])


def build_svar_irf_frame() -> pd.DataFrame:
    """Each target's response to the four macro shocks, with bootstrap bands."""
    frames = []
    for system, config in SVAR_SYSTEMS.items():
        target = str(config["target"])
        fitted = _fit_pinned_svar(config)
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


def build_svar_shock_events_frame() -> pd.DataFrame:
    """Each target's own structural shock per quarter, z-scored and flagged when |z| > 2."""
    frames = []
    for system, config in SVAR_SYSTEMS.items():
        target = str(config["target"])
        fitted = _fit_pinned_svar(config)
        shocks = svar.structural_shocks(fitted)
        # Locate quarters where the system target itself has an unusual
        # structural shock; macro-driver contributions are shown separately
        # in svar_historical_decomposition.csv.
        filtered = shocks.loc[shocks["shock"].eq(target)].copy()
        stats = filtered.groupby("shock")["shock_value"].agg(["mean", "std"])
        filtered = filtered.join(stats, on="shock")
        shock_std = filtered["std"].replace(0, np.nan)
        filtered["shock_z"] = (filtered["shock_value"] - filtered["mean"]) / shock_std
        filtered["flagged"] = filtered["shock_z"].abs().gt(2).fillna(False).astype(bool)
        filtered["system"] = system
        filtered["target"] = target
        filtered["forecast_origin"] = str(SVAR_EXPORT_FORECAST_ORIGIN)
        frames.append(filtered)

    return pd.concat(frames, ignore_index=True)[
        [
            "system",
            "target",
            "forecast_origin",
            "quarter",
            "shock",
            "shock_z",
            "flagged",
        ]
    ]


def build_svar_historical_decomposition_frame() -> pd.DataFrame:
    """Each target's fitted history split into baseline and per-shock contributions."""
    frames = []
    for system, config in SVAR_SYSTEMS.items():
        target = str(config["target"])
        fitted = _fit_pinned_svar(config)
        decomposition = svar.historical_decomposition(fitted)
        filtered = decomposition.loc[decomposition["response"].eq(target)].copy()
        filtered["system"] = system
        filtered["target"] = target
        filtered["forecast_origin"] = str(SVAR_EXPORT_FORECAST_ORIGIN)
        frames.append(filtered)

    return pd.concat(frames, ignore_index=True)[
        [
            "system",
            "target",
            "forecast_origin",
            "quarter",
            "component",
            "contribution",
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
        forecast_payloads = {}
        for target_label, config in FORECAST_TARGETS.items():
            response = requests.post(
                args.api_base_url + config["endpoint"], json={"horizon": args.horizon}, timeout=60
            )
            response.raise_for_status()
            forecast_payloads[target_label] = response.json()
        response = requests.get(args.api_base_url + "/rba-action", timeout=60)
        response.raise_for_status()
        rba_action_payload = response.json()
        response = requests.get(args.api_base_url + "/credit-risk/stress-test", timeout=60)
        response.raise_for_status()
        credit_stress_payload = response.json()
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
    build_svar_shock_events_frame().to_csv(args.output_dir / "svar_shock_events.csv", index=False)
    build_svar_historical_decomposition_frame().to_csv(
        args.output_dir / "svar_historical_decomposition.csv", index=False
    )

    print(f"Wrote 11 CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
