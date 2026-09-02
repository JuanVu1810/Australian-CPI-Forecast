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
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURATED_DATA_PATH = PROJECT_ROOT / "data/curated/quarterly_macro_features.csv"
SERIES_AVAILABILITY_PATH = PROJECT_ROOT / "data/metadata/series_availability.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/tableau"
DEFAULT_API_BASE_URL = "http://localhost:8000"

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
    curated = pd.read_csv(CURATED_DATA_PATH, usecols=["quarter"])
    quarters = curated["quarter"].astype(str)
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

    build_forecast_frame(forecast_payloads).to_csv(args.output_dir / "forecast.csv", index=False)
    build_rba_action_frame(rba_action_payload).to_csv(args.output_dir / "rba_action.csv", index=False)
    build_credit_stress_frame(credit_stress_payload).to_csv(args.output_dir / "credit_stress.csv", index=False)
    build_model_comparison_frame().to_csv(args.output_dir / "model_comparison.csv", index=False)
    build_coverage_frame().to_csv(args.output_dir / "model_interval_coverage.csv", index=False)
    build_dataset_overview_frame().to_csv(args.output_dir / "dataset_overview.csv", index=False)
    build_historical_indicators_frame().to_csv(args.output_dir / "historical_indicators.csv", index=False)

    print(f"Wrote 7 CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
