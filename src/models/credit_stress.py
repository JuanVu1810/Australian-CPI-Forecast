"""Fit-free credit-risk stress-test formulas based on RBA RDP 2022-03."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from src.models.evaluation import PROJECT_ROOT


# Garvin et al (2022), Section 3.3, p.15; Table A2, p.39: a 1 percentage
# point higher unemployment rate raises the personal-loan PD by 0.4 points.
PERSONAL_LOAN_UR_SENSITIVITY = 0.4
# Garvin et al (2022), Equation 1, p.7; coefficient calibration discussion,
# p.8: beta_Mort,UR is set to 0.6.
MORTGAGE_UR_SENSITIVITY = 0.6

PD_BASE_ASSUMPTIONS_PATH = PROJECT_ROOT / "data/metadata/pd_base_assumptions.csv"
LGD_EAD_ASSUMPTIONS_PATH = PROJECT_ROOT / "data/metadata/lgd_ead_assumptions.csv"
DISCOUNT_RATE_ASSUMPTIONS_PATH = (
    PROJECT_ROOT / "data/metadata/discount_rate_assumptions.csv"
)
SEGMENT_ORDER = ("personal_loans", "mortgages")

# Scenario quantiles taken from the cumulative-unemployment-change draws
# returned by svar.forecast_cumulative_unemployment_change_quantiles. "downside"
# (weaker economy) maps to the 90th percentile of the unemployment-change
# distribution -- the largest simulated rise in unemployment, worse for
# credit -- and "upside" (stronger economy) to the 10th percentile -- the
# smallest rise or a fall in unemployment, better for credit. This is the
# standard direction convention for credit-risk/ECL scenario naming (a
# "downside" scenario means a weaker economy), and is why it's the reverse
# of how the same 10th/90th split is used for this project's existing 80%
# SVAR bootstrap IRF bands (src/models/svar.py's DEFAULT_LOWER_QUANTILE /
# DEFAULT_UPPER_QUANTILE), kept as an independent constant here so changing
# the IRF band width elsewhere does not silently change this scenario
# definition.
SCENARIO_QUANTILES = {"downside": 0.9, "base": 0.5, "upside": 0.1}
# NAB's own disclosed FY2025 Annual Report macroeconomic scenario
# probability weightings (base case 55%, downside 42.5%, upside 2.5%),
# applied here to this project's own generic PD-stress mechanism, not to
# NAB's own ECL model outputs.
SCENARIO_PROBABILITY_WEIGHTS = {"downside": 0.425, "base": 0.55, "upside": 0.025}

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
    "project does not model as a distinct parameter. LGD and EaD post-CCF "
    "and post-CRM (in $AUD millions) are from the same NAB FY2025 Pillar 3 "
    "Table CR6 disclosure: 16% LGD / $429,996m EAD for residential mortgage, "
    "73% LGD / $1,663m EAD for other retail; EAD is a portfolio-level "
    "aggregate for NAB's whole book in that exposure class, not a per-loan "
    "figure, so the resulting ECL is an illustrative aggregate dollar "
    "amount, not a per-customer estimate. "
    "Present-value ECL = PD_stressed x LGD x EAD / (1 + discount_rate)^0.5 "
    "is computed under three scenarios "
    "(downside/base/upside, at the 90th/50th/10th percentile of the SVAR "
    "system's simulated cumulative unemployment-change draws, since a "
    "downside/weaker-economy scenario means a larger rise in unemployment) "
    "and "
    "combined using NAB's own disclosed FY2025 Annual Report macroeconomic "
    "scenario probability weightings (55% base, 42.5% downside, 2.5% "
    "upside) -- those weights are NAB's own, but applied here to this "
    "project's generic PD-stress mechanism, not to NAB's own ECL model. "
    "Only PD is stressed across scenarios; LGD and EAD are held fixed at "
    "their latest disclosed values in every scenario, so the downside case "
    "likely understates the true loss a real downturn would cause (e.g. "
    "falling house prices would also raise mortgage LGD via reduced "
    "collateral recovery, which this project does not model). This is a "
    "12-month, Stage-1-only estimate under AASB 9: it assumes the entire "
    "segment sits in Stage 1 with no significant-increase-in-credit-risk "
    "(SICR) or default-staging logic, because no loan-level "
    "origination-vs-current credit risk or delinquency data exists in this "
    "project to detect SICR. A real AASB 9 provision would add lifetime ECL "
    "for whatever sits in Stage 2/3, which this project does not compute "
    "and which typically contributes disproportionately more of the total "
    "provision in a downturn. Separately, pd_base is NAB's disclosed "
    "average PD blended across whatever mix of Stage 1/2/3 exposures "
    "currently sits in that Basel exposure class -- not a genuinely low, "
    "Stage-1-only PD -- and it is applied here to the full EAD rather than "
    "a Stage-1-only exposure subset, so this illustrative Stage-1 ECL is "
    "not a lower bound on NAB's real provision and should not be read as "
    "one. Per NAB's FY2025 Annual Report (Note 17, Provision for credit "
    "impairment), NAB's real Stage 1 (12-month ECL) provision was only "
    "$646m of a $6,165m Group total across all stages (about 10.5%), and "
    "NAB's real total provision (all three stages combined) for its "
    "Housing portfolio was $1,296m -- both smaller than this project's "
    "illustrative Stage-1-only, probability-weighted mortgages ECL of "
    "roughly $1.59bn (discounted; see below). These figures are not "
    "measuring the same thing and "
    "must not be compared as if they were. Expected cash shortfalls are "
    "discounted using segment-level effective-interest-rate proxies from "
    "published RBA lending-rate tables rather than loan-level EIRs: 6.80% "
    "for mortgages from RBA Statistical Table F5, series FILRHLBVD "
    "'Lending rates; Housing loans; Banks; Variable; Discounted; "
    "Owner-occupier', observed 31 Aug 2026 and published 07 Sep 2026; and "
    "8.86% for personal_loans from RBA Statistical Table F8, series "
    "FLRPFOFTT 'Lending rates; Personal credit; Outstanding; Fixed-term "
    "loans; Total', observed 31 Jul 2026 and published 07 Sep 2026. Table "
    "F8 is used for personal_loans because RBA's Table F5 personal "
    "term-loan series were discontinued from the April 2020 release and "
    "RBA states alternative personal lending-rate series are published in "
    "Table F8. The discounting convention is a separate modelling choice: "
    "the 12-month expected cash shortfall is discounted by (1 + rate)^0.5 "
    "under a mid-year convention, following the standard general DCF "
    "valuation technique used when cash flows are assumed to be spread "
    "evenly across a period in Aswath Damodaran's NYU Stern valuation "
    "materials and McKinsey & Company's Valuation reference. This is "
    "applied here by analogy -- treating a 12-month flow of expected "
    "credit losses like a 12-month flow of DCF cash flows -- not because "
    "AASB 9 or an ECL-specific source prescribes this exact within-year "
    "loss-timing assumption. Combining these real disclosed inputs with this "
    "generic, simplified stress-and-ECL mechanism is still not NAB's own "
    "stress-testing or ECL methodology and must not be used for actual "
    "credit, regulatory, or accounting-provision decisions. For context, "
    "APRA's aggregate non-performing/impaired-loan ratio is available in "
    "this repo at data/curated/credit_quality_quarterly.csv as corroborating "
    "context only; it does not calibrate the PD/LGD/EAD base values, "
    "sensitivity coefficients, or scenario weights."
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


def load_lgd_ead_assumptions(
    path: Path = LGD_EAD_ASSUMPTIONS_PATH,
) -> pd.DataFrame:
    """Load illustrative segment-level LGD and EAD (post-CCF/post-CRM) assumptions."""
    frame = pd.read_csv(path)
    required = {"segment", "lgd", "ead_aud_m", "note"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = frame.copy()
    frame["lgd"] = pd.to_numeric(frame["lgd"], errors="raise")
    frame["ead_aud_m"] = pd.to_numeric(frame["ead_aud_m"], errors="raise")
    return frame


def load_discount_rate_assumptions(
    path: Path = DISCOUNT_RATE_ASSUMPTIONS_PATH,
) -> pd.DataFrame:
    """Load illustrative segment-level EIR proxy assumptions."""
    frame = pd.read_csv(path)
    required = {"segment", "discount_rate", "note"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = frame.copy()
    frame["discount_rate"] = pd.to_numeric(frame["discount_rate"], errors="raise")
    return frame


def run_credit_stress_test(
    scenario_deltas: Mapping[str, float],
    pd_base_path: Path = PD_BASE_ASSUMPTIONS_PATH,
    lgd_ead_path: Path = LGD_EAD_ASSUMPTIONS_PATH,
    discount_rate_path: Path = DISCOUNT_RATE_ASSUMPTIONS_PATH,
    scenario_weights: Mapping[str, float] = SCENARIO_PROBABILITY_WEIGHTS,
) -> pd.DataFrame:
    """Apply the illustrative PD/LGD/EAD 12-month ECL formula per segment and scenario.

    ``scenario_deltas`` maps a scenario name (e.g. "downside"/"base"/"upside")
    to that scenario's cumulative unemployment-rate change. Returns one row
    per segment per scenario with the stressed PD and present-value ECL:
    ``ecl = pd_stressed * lgd * ead / (1 + discount_rate) ** 0.5`` (in $AUD
    millions, since ``ead`` is).
    """
    missing_weights = sorted(set(scenario_deltas) - set(scenario_weights))
    if missing_weights:
        raise ValueError(f"Scenario weights missing for scenarios: {missing_weights}")

    pd_assumptions = load_pd_base_assumptions(pd_base_path).set_index("segment")
    lgd_ead_assumptions = load_lgd_ead_assumptions(lgd_ead_path).set_index("segment")
    discount_rate_assumptions = load_discount_rate_assumptions(
        discount_rate_path
    ).set_index("segment")
    missing_segments = [
        segment
        for segment in SEGMENT_ORDER
        if segment not in pd_assumptions.index
        or segment not in lgd_ead_assumptions.index
        or segment not in discount_rate_assumptions.index
    ]
    if missing_segments:
        raise ValueError(f"Credit-stress assumptions missing segments: {missing_segments}")

    rows = []
    for segment in SEGMENT_ORDER:
        pd_base = float(pd_assumptions.loc[segment, "pd_base"])
        lgd = float(lgd_ead_assumptions.loc[segment, "lgd"])
        ead = float(lgd_ead_assumptions.loc[segment, "ead_aud_m"])
        discount_rate = float(discount_rate_assumptions.loc[segment, "discount_rate"])
        if segment == "personal_loans":
            sensitivity = PERSONAL_LOAN_UR_SENSITIVITY
            stressed_pd_fn = personal_loan_stressed_pd
        else:
            sensitivity = MORTGAGE_UR_SENSITIVITY
            stressed_pd_fn = mortgage_stressed_pd

        for scenario, delta_unemployment_cumulative in scenario_deltas.items():
            stressed_pd = stressed_pd_fn(pd_base, delta_unemployment_cumulative)
            rows.append(
                {
                    "segment": segment,
                    "scenario": scenario,
                    "probability_weight": float(scenario_weights[scenario]),
                    "delta_unemployment_cumulative": float(delta_unemployment_cumulative),
                    "pd_base": pd_base,
                    "ur_sensitivity": sensitivity,
                    "lgd": lgd,
                    "ead_aud_m": ead,
                    "discount_rate": discount_rate,
                    "pd_stressed": stressed_pd,
                    "ecl_aud_m": stressed_pd
                    * lgd
                    * ead
                    / (1 + discount_rate) ** 0.5,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "segment",
            "scenario",
            "probability_weight",
            "delta_unemployment_cumulative",
            "pd_base",
            "ur_sensitivity",
            "lgd",
            "ead_aud_m",
            "discount_rate",
            "pd_stressed",
            "ecl_aud_m",
        ],
    )
