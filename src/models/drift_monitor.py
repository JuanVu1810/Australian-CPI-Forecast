"""Read-only checks for forecast drift and covariate drift.

Compares forecasts that now have a real outcome against each model's own
historical walk-forward error distribution (``reports/backtest_predictions*.csv``)
and flags exogenous indicators sitting at unusual levels relative to their own
history (the ``historical_indicators`` export). Every function here reads
already-generated reports or takes an in-memory frame -- nothing in this module
fits, refits, or calls a model, per this project's guardrail against
refitting as a side effect of a routine data refresh (see
``.ai/TABLEAU_DASHBOARD_GUIDE.md``, "2026 actuals refresh").

Grading a quarter (comparing its forecast to a historical error distribution)
never depends on a hardcoded cutoff date -- it's driven entirely by which
quarters have both a forecast row and a real actual, so this keeps working
unattended as later quarters' actuals arrive in future refreshes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Deliberately not imported from model_comparison.py: that module pulls in the
# full elastic_net/sarima/evaluation fitting chain (sklearn, statsmodels), which
# this module has no other reason to depend on -- everything here only ever
# reads already-generated reports. These two paths mirror model_comparison.py's
# PREDICTIONS_OUTPUT_PATH / TRIMMED_MEAN_PREDICTIONS_OUTPUT_PATH; if either
# moves there, update it here too.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEADLINE_BACKTEST_PATH = PROJECT_ROOT / "reports/backtest_predictions.csv"
TRIMMED_MEAN_BACKTEST_PATH = PROJECT_ROOT / "reports/backtest_predictions_trimmed_mean.csv"

# forecast.csv's model set (see the Tableau dashboard guide's data-shape traps):
# seasonal_naive and rba appear in the backtest reports but never in forecast.csv,
# so a drift check against a forecast that was never actually served would be
# comparing against a model nobody is looking at.
FORECAST_MODEL_FAMILIES = ("sarima", "elastic_net", "ensemble")

BACKTEST_REPORTS = {
    "Headline": HEADLINE_BACKTEST_PATH,
    "Trimmed mean": TRIMMED_MEAN_BACKTEST_PATH,
}

HISTORICAL_TARGET_VARIABLES = {
    "cpi_yoy": "Headline",
    "trimmed_mean_cpi_yoy": "Trimmed mean",
}

ERROR_CHECK_COLUMNS = [
    "target",
    "model",
    "target_quarter",
    "horizon",
    "actual",
    "forecast",
    "error",
    "hist_n",
    "hist_mean_error",
    "hist_std_error",
    "z_score",
    "error_percentile",
]


def load_backtest_error_frame() -> pd.DataFrame:
    """Concatenate both targets' walk-forward backtest predictions into one tidy frame."""
    frames = []
    for target_label, path in BACKTEST_REPORTS.items():
        if not path.exists():
            continue
        report = pd.read_csv(
            path, usecols=["model", "forecast_origin", "target_quarter", "horizon", "error"]
        )
        report = report.loc[report["model"].isin(FORECAST_MODEL_FAMILIES)].copy()
        if report.empty:
            continue
        report["horizon"] = report["horizon"].astype(int)
        report["target"] = target_label
        frames.append(report)
    if not frames:
        return pd.DataFrame(
            columns=["target", "model", "forecast_origin", "target_quarter", "horizon", "error"]
        )
    return pd.concat(frames, ignore_index=True)


def _realized_actuals(historical_frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce the historical_indicators export to one row per (target, quarter) actual."""
    actuals = historical_frame.loc[historical_frame["variable"].isin(HISTORICAL_TARGET_VARIABLES)].copy()
    actuals["target"] = actuals["variable"].map(HISTORICAL_TARGET_VARIABLES)
    return actuals[["target", "quarter", "value"]].rename(
        columns={"value": "actual", "quarter": "target_quarter"}
    )


def build_drift_error_check_frame(
    forecast_frame: pd.DataFrame, historical_frame: pd.DataFrame
) -> pd.DataFrame:
    """For every forecast that now has a real outcome, compare its error against
    the same model's own historical walk-forward error distribution at that
    horizon.

    Grain: target x model x target_quarter. Answers "is this miss unusual for
    this model," not "is this miss large in absolute terms" -- a model with
    naturally noisy errors gets a wider bar to clear than a tight one.
    """
    forecasts = forecast_frame.rename(columns={"model_family": "model", "quarter": "target_quarter"})
    forecasts = forecasts.loc[forecasts["model"].isin(FORECAST_MODEL_FAMILIES)]
    actuals = _realized_actuals(historical_frame)

    graded = forecasts.merge(actuals, on=["target", "target_quarter"], how="inner")
    if graded.empty:
        return pd.DataFrame(columns=ERROR_CHECK_COLUMNS)
    graded = graded.copy()
    graded["error"] = graded["actual"] - graded["forecast"]

    backtest = load_backtest_error_frame()
    rows = []
    for _, row in graded.iterrows():
        hist = backtest.loc[
            (backtest["target"] == row["target"])
            & (backtest["model"] == row["model"])
            & (backtest["horizon"] == row["horizon"]),
            "error",
        ]
        n = int(len(hist))
        mean = float(hist.mean()) if n else float("nan")
        std = float(hist.std(ddof=1)) if n > 1 else float("nan")
        z = (row["error"] - mean) / std if std and std > 0 else float("nan")
        percentile = float((hist <= row["error"]).mean() * 100) if n else float("nan")
        rows.append(
            {
                "target": row["target"],
                "model": row["model"],
                "target_quarter": row["target_quarter"],
                "horizon": int(row["horizon"]),
                "actual": row["actual"],
                "forecast": row["forecast"],
                "error": row["error"],
                "hist_n": n,
                "hist_mean_error": mean,
                "hist_std_error": std,
                "z_score": z,
                "error_percentile": percentile,
            }
        )
    return (
        pd.DataFrame(rows, columns=ERROR_CHECK_COLUMNS)
        .sort_values(["target", "model", "horizon"])
        .reset_index(drop=True)
    )


def build_drift_error_history_frame(
    forecast_frame: pd.DataFrame, historical_frame: pd.DataFrame
) -> pd.DataFrame:
    """One-step-ahead (horizon 1) error time series per target x model: every
    historical walk-forward origin plus any newly-graded origin, flagged by
    `is_recent`.

    Lets a viewer check a fresh miss against precedent -- e.g. whether the
    model has produced same-direction misses this size, or larger, before.
    """
    backtest = load_backtest_error_frame()
    historical = backtest.loc[backtest["horizon"] == 1].copy()
    historical["is_recent"] = False

    check = build_drift_error_check_frame(forecast_frame, historical_frame)
    recent = check.loc[check["horizon"] == 1, ["target", "model", "target_quarter", "error"]].copy()
    if not recent.empty:
        origin = forecast_frame["forecast_origin"].iloc[0] if not forecast_frame.empty else None
        recent["forecast_origin"] = origin
    else:
        recent["forecast_origin"] = pd.Series(dtype="object")
    recent["is_recent"] = True

    combined = pd.concat(
        [
            historical[["target", "model", "forecast_origin", "target_quarter", "error", "is_recent"]],
            recent[["target", "model", "forecast_origin", "target_quarter", "error", "is_recent"]],
        ],
        ignore_index=True,
    )
    return combined.sort_values(["target", "model", "target_quarter"]).reset_index(drop=True)


def build_drift_covariate_frame(historical_frame: pd.DataFrame) -> pd.DataFrame:
    """Latest reading per variable against its own full history, ranked by how
    extreme that reading is -- flags inputs sitting at unusual levels right
    now, independent of whether any forecast has been graded yet.

    Grain: one row per variable. `abs_z_rank` of 1 is that variable's most
    extreme reading on record; a low rank on a variable that feeds a model
    directly (e.g. an Elastic Net feature) is a plausible, checkable reason
    for that model's forecast to be off, distinct from the model itself
    having gone stale.
    """
    frame = historical_frame.copy()
    frame["abs_z"] = frame["value_standardized"].abs()
    frame["abs_z_rank"] = frame.groupby("variable")["abs_z"].rank(ascending=False, method="min")
    frame["n_obs"] = frame.groupby("variable")["variable"].transform("count")

    # `quarter` strings are fixed-width "YYYYQ#", so a plain lexicographic sort already
    # matches chronological order -- sort + take the last row per group rather than
    # idxmax, which isn't reliable on object-dtype columns across pandas versions.
    latest = (
        frame.sort_values("quarter")
        .groupby("variable", as_index=False)
        .tail(1)[["variable", "variable_label", "quarter", "value", "value_standardized", "abs_z_rank", "n_obs"]]
        .rename(columns={"quarter": "latest_quarter"})
    )
    # Sort by the z-score itself, not by abs_z_rank: rank is scoped to each variable's
    # own history, so a rank of 1 on a naturally placid series and a rank of 1 on a
    # naturally volatile one aren't comparable -- the z-score is what's actually
    # standardized across variables, which is the whole point of using it here.
    return latest.sort_values(
        by="value_standardized", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)
