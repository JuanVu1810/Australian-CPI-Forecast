"""Pure data preparation for the SVAR shock-attribution and drift-monitor sections.

Everything here takes already-exported report frames (``reports/tableau/svar_*.csv``
and ``reports/tableau/drift_*.csv``) and returns another frame or a small dict.
Nothing reads files, imports Streamlit, or fits a model, so the logic can be
unit-tested without running the report script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# The drift exports and the SVAR shock export both flag |z| > 2, the same
# threshold ``src/models/drift_monitor.py`` documents for "unusual".
UNUSUAL_Z = 2.0

# Label -> (system, target column) for the two SVAR systems in the exports.
SVAR_SYSTEMS = {
    "System A (headline CPI)": ("System A", "cpi_yoy"),
    "System B (trimmed-mean CPI)": ("System B", "trimmed_mean_cpi_yoy"),
}

MACRO_DRIVER = "Macro driver"
OWN_SHOCK = "CPI's own shock"


def with_quarter_date(frame: pd.DataFrame, column: str = "quarter") -> pd.DataFrame:
    """Add a ``quarter_date`` column (quarter end), matching ``load_curated_data``."""
    result = frame.copy()
    result["quarter_date"] = pd.PeriodIndex(result[column], freq="Q").to_timestamp(how="end").normalize()
    return result


def system_shock_events(events: pd.DataFrame, system: str) -> pd.DataFrame:
    """One system's target-shock z-scores in time order, with a ``quarter_date`` column."""
    subset = events.loc[events["system"] == system]
    return with_quarter_date(subset).sort_values("quarter_date").reset_index(drop=True)


def flagged_quarters(events: pd.DataFrame) -> list[str]:
    """Flagged quarters in time order."""
    return events.loc[events["flagged"].astype(str).eq("True"), "quarter"].tolist()


def strongest_flagged_quarter(events: pd.DataFrame) -> str | None:
    """The flagged quarter with the largest absolute shock, or ``None`` if none flagged."""
    flagged = events.loc[events["flagged"].astype(str).eq("True")]
    if flagged.empty:
        return None
    return str(flagged.loc[flagged["shock_z"].abs().idxmax(), "quarter"])


def decomposition_for_quarter(decomposition: pd.DataFrame, system: str, quarter: str) -> pd.DataFrame:
    """Components of one system's target level in one quarter, tagged by kind.

    ``kind`` is ``baseline`` for the zero-shock path, ``CPI's own shock`` for the
    component named after the system's own target, and ``Macro driver`` otherwise.
    Baseline plus every other component sums to the actual target value.
    """
    rows = decomposition.loc[(decomposition["system"] == system) & (decomposition["quarter"] == quarter)].copy()
    if rows.empty:
        return rows.assign(kind=pd.Series(dtype=object))
    target = rows["target"].iloc[0]

    def kind(component: str) -> str:
        if component == "baseline":
            return "baseline"
        return OWN_SHOCK if component == target else MACRO_DRIVER

    rows["kind"] = rows["component"].map(kind)
    return rows.reset_index(drop=True)


def decomposition_totals(parts: pd.DataFrame) -> dict[str, float]:
    """Actual level, zero-shock baseline, and the combined effect of all shocks."""
    baseline = float(parts.loc[parts["kind"] == "baseline", "contribution"].sum())
    shocks = float(parts.loc[parts["kind"] != "baseline", "contribution"].sum())
    return {"actual": baseline + shocks, "baseline": baseline, "shocks": shocks}


def drift_summary(error_check: pd.DataFrame, covariate_check: pd.DataFrame, threshold: float = UNUSUAL_Z) -> dict:
    """Headline counts for the drift monitor, computed from the exports.

    ``error`` in the export is ``actual - forecast``, so a positive error means the
    forecast came in below the outcome (an undershoot).
    """
    abs_z = error_check["z_score"].abs()
    return {
        "graded_forecasts": int(len(error_check)),
        "graded_quarters": sorted(error_check["target_quarter"].unique().tolist()),
        "unusual_misses": int((abs_z > threshold).sum()),
        "largest_abs_z": float(abs_z.max()) if len(abs_z) else float("nan"),
        "undershot": int((error_check["error"] > 0).sum()),
        "unusual_inputs": covariate_check.loc[
            covariate_check["value_standardized"].abs() > threshold, "variable_label"
        ].tolist(),
    }


def z_axis_bound(z_scores: pd.Series, minimum: float = 3.0) -> float:
    """Symmetric x-axis limit that always shows the +-2 zone and every point."""
    largest = float(np.nanmax(np.abs(z_scores))) if len(z_scores) else 0.0
    return max(minimum, float(np.ceil(largest)) + 0.5)
