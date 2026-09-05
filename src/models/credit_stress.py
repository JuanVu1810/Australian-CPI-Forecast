"""Fit-free credit-risk stress-test formulas based on RBA RDP 2022-03."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.evaluation import PROJECT_ROOT


# Garvin et al (2022), Section 3.3, p.15; Table A2, p.39: a 1 percentage
# point higher unemployment rate raises the personal-loan PD by 0.4 points.
PERSONAL_LOAN_UR_SENSITIVITY = 0.4
# Garvin et al (2022), Equation 1, p.7; coefficient calibration discussion,
# p.8: beta_Mort,UR is set to 0.6.
MORTGAGE_UR_SENSITIVITY = 0.6

PD_BASE_ASSUMPTIONS_PATH = PROJECT_ROOT / "data/metadata/pd_base_assumptions.csv"
SEGMENT_ORDER = ("personal_loans", "mortgages")

CREDIT_STRESS_CAVEAT = (
    "Credit-stress PD base values are NAB's own disclosed FY2025 Pillar 3 "
    "weighted-average PDs (Table CR6, as at 30 Sep 2025): 2.07% for the "
    "residential mortgage exposure class and 8.78% for other retail. The "
    "personal_loans segment here uses the 'other retail' figure as its "
    "closest disclosed proxy -- that Basel exposure class is broader than "
    "unsecured personal lending alone (it can include small-business retail "
    "exposures), so the category mapping is not exact. The unemployment-"
    "sensitivity coefficients, 0.4 for personal loans and 0.6 for "
    "mortgages, are still the RBA's published values from Garvin et al "
    "(2022), RBA Research Discussion Paper No 2022-03, applied uniformly "
    "across the nine banks covered in that paper (including NAB), not "
    "adapted or derived for NAB specifically. The mortgage coefficient is "
    "applied in a simplified form without the paper's LVR-bucket tiering, "
    "and the floor used here is the segment's own PD_base rather than the "
    "paper's separate lower natural-default-rate floor PD-bar, which this "
    "project does not model as a distinct parameter. This combines a real "
    "disclosed starting PD with a generic, illustrative stress mechanism -- "
    "it is not NAB's own stress-testing methodology and must not be used "
    "for actual credit or regulatory decisions. For context, APRA's "
    "aggregate non-performing/impaired-loan ratio is available in this repo "
    "at data/curated/credit_quality_quarterly.csv as corroborating context "
    "only; it does not calibrate the PD base values or sensitivity "
    "coefficients."
)


def _stressed_pd(
    pd_base: float,
    delta_unemployment_cumulative: float,
    ur_sensitivity_percentage_points: float,
) -> float:
    """Apply a percentage-point PD sensitivity and return decimal annual PD."""
    pd_effect_decimal = (
        ur_sensitivity_percentage_points * delta_unemployment_cumulative / 100
    )
    return min(1.0, max(float(pd_base), float(pd_base) + pd_effect_decimal))


def personal_loan_stressed_pd(
    pd_base: float,
    delta_unemployment_cumulative: float,
) -> float:
    """Return stressed annual personal-loan PD as a decimal."""
    return _stressed_pd(
        pd_base=pd_base,
        delta_unemployment_cumulative=delta_unemployment_cumulative,
        ur_sensitivity_percentage_points=PERSONAL_LOAN_UR_SENSITIVITY,
    )


def mortgage_stressed_pd(
    pd_base: float,
    delta_unemployment_cumulative: float,
) -> float:
    """Return stressed annual mortgage PD as a decimal."""
    return _stressed_pd(
        pd_base=pd_base,
        delta_unemployment_cumulative=delta_unemployment_cumulative,
        ur_sensitivity_percentage_points=MORTGAGE_UR_SENSITIVITY,
    )


def load_pd_base_assumptions(
    path: Path = PD_BASE_ASSUMPTIONS_PATH,
) -> pd.DataFrame:
    """Load illustrative segment-level annual PD assumptions."""
    frame = pd.read_csv(path)
    required = {"segment", "pd_base", "note"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = frame.copy()
    frame["pd_base"] = pd.to_numeric(frame["pd_base"], errors="raise")
    return frame


def run_credit_stress_test(
    delta_unemployment_cumulative: float,
    pd_base_path: Path = PD_BASE_ASSUMPTIONS_PATH,
) -> pd.DataFrame:
    """Apply the illustrative PD stress formula to supported loan segments."""
    assumptions = load_pd_base_assumptions(pd_base_path).set_index("segment")
    missing_segments = [
        segment for segment in SEGMENT_ORDER if segment not in assumptions.index
    ]
    if missing_segments:
        raise ValueError(f"PD base assumptions missing segments: {missing_segments}")

    rows = []
    for segment in SEGMENT_ORDER:
        pd_base = float(assumptions.loc[segment, "pd_base"])
        if segment == "personal_loans":
            sensitivity = PERSONAL_LOAN_UR_SENSITIVITY
            stressed_pd = personal_loan_stressed_pd(
                pd_base,
                delta_unemployment_cumulative,
            )
        else:
            sensitivity = MORTGAGE_UR_SENSITIVITY
            stressed_pd = mortgage_stressed_pd(
                pd_base,
                delta_unemployment_cumulative,
            )
        rows.append(
            {
                "segment": segment,
                "pd_base": pd_base,
                "ur_sensitivity": sensitivity,
                "delta_unemployment_cumulative": float(delta_unemployment_cumulative),
                "pd_stressed": stressed_pd,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "segment",
            "pd_base",
            "ur_sensitivity",
            "delta_unemployment_cumulative",
            "pd_stressed",
        ],
    )
