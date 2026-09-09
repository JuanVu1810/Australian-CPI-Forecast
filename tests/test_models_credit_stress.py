import pytest

from src.models import credit_stress


def test_personal_loan_stressed_pd_matches_formula_cases():
    pd_base = 0.03

    assert credit_stress.personal_loan_stressed_pd(pd_base, 0.0) == pytest.approx(
        pd_base
    )
    assert credit_stress.personal_loan_stressed_pd(pd_base, 2.0) == pytest.approx(
        0.038
    )
    assert credit_stress.personal_loan_stressed_pd(pd_base, -1.0) == pytest.approx(
        pd_base
    )


def test_mortgage_stressed_pd_matches_formula_cases():
    pd_base = 0.005

    assert credit_stress.mortgage_stressed_pd(pd_base, 0.0) == pytest.approx(pd_base)
    assert credit_stress.mortgage_stressed_pd(pd_base, 2.0) == pytest.approx(0.017)
    assert credit_stress.mortgage_stressed_pd(pd_base, -1.0) == pytest.approx(pd_base)


def test_stressed_pd_is_capped_at_one():
    assert credit_stress.personal_loan_stressed_pd(
        pd_base=0.03,
        delta_unemployment_cumulative=300.0,
    ) == pytest.approx(1.0)


def test_segments_use_distinct_sensitivity_constants():
    pd_base = 0.01
    delta_unemployment_cumulative = 1.5

    personal = credit_stress.personal_loan_stressed_pd(
        pd_base,
        delta_unemployment_cumulative,
    )
    mortgage = credit_stress.mortgage_stressed_pd(
        pd_base,
        delta_unemployment_cumulative,
    )

    assert credit_stress.PERSONAL_LOAN_UR_SENSITIVITY == 0.4
    assert credit_stress.MORTGAGE_UR_SENSITIVITY == 0.6
    assert mortgage - personal == pytest.approx(0.003)


def test_load_pd_base_assumptions_reads_real_metadata_file():
    assumptions = credit_stress.load_pd_base_assumptions()

    assert set(assumptions["segment"]) == {"personal_loans", "mortgages"}
    personal = assumptions.set_index("segment").loc["personal_loans"]
    mortgage = assumptions.set_index("segment").loc["mortgages"]
    assert personal["pd_base"] == pytest.approx(0.0878)
    assert mortgage["pd_base"] == pytest.approx(0.0207)
    assert "NAB" in personal["note"]
    assert "Pillar 3" in personal["note"]
    assert "not an exact category match" in personal["note"]
    assert "residential mortgage" in mortgage["note"]


def test_load_lgd_ead_assumptions_reads_real_metadata_file():
    assumptions = credit_stress.load_lgd_ead_assumptions()

    assert set(assumptions["segment"]) == {"personal_loans", "mortgages"}
    personal = assumptions.set_index("segment").loc["personal_loans"]
    mortgage = assumptions.set_index("segment").loc["mortgages"]
    assert personal["lgd"] == pytest.approx(0.73)
    assert personal["ead_aud_m"] == pytest.approx(1663.0)
    assert mortgage["lgd"] == pytest.approx(0.16)
    assert mortgage["ead_aud_m"] == pytest.approx(429996.0)
    assert "NAB" in personal["note"]
    assert "Table CR6" in personal["note"]
    assert "not an exact category match" in personal["note"]
    assert "residential mortgage" in mortgage["note"]


def test_load_discount_rate_assumptions_reads_real_metadata_file():
    assumptions = credit_stress.load_discount_rate_assumptions()

    assert set(assumptions["segment"]) == {"personal_loans", "mortgages"}
    personal = assumptions.set_index("segment").loc["personal_loans"]
    mortgage = assumptions.set_index("segment").loc["mortgages"]
    assert personal["discount_rate"] == pytest.approx(0.0886)
    assert mortgage["discount_rate"] == pytest.approx(0.0680)
    assert "RBA Statistical Table F8" in personal["note"]
    assert "FLRPFOFTT" in personal["note"]
    assert "31 Jul 2026" in personal["note"]
    assert "RBA Statistical Table F5" in mortgage["note"]
    assert "FILRHLBVD" in mortgage["note"]
    assert "31 Aug 2026" in mortgage["note"]


def test_run_credit_stress_test_returns_segment_by_scenario_frame():
    scenario_deltas = {"downside": -1.0, "base": 0.0, "upside": 2.0}

    result = credit_stress.run_credit_stress_test(scenario_deltas=scenario_deltas)

    assert list(result.columns) == [
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
    ]
    assert set(result["segment"]) == {"personal_loans", "mortgages"}
    assert set(result["scenario"]) == set(scenario_deltas)
    assert len(result) == 6

    personal_upside = result.loc[
        (result["segment"] == "personal_loans") & (result["scenario"] == "upside")
    ].iloc[0]
    expected_personal_pd = credit_stress.personal_loan_stressed_pd(0.0878, 2.0)
    assert personal_upside["pd_stressed"] == pytest.approx(expected_personal_pd)
    assert personal_upside["lgd"] == pytest.approx(0.73)
    assert personal_upside["ead_aud_m"] == pytest.approx(1663.0)
    assert personal_upside["discount_rate"] == pytest.approx(0.0886)
    assert personal_upside["ecl_aud_m"] == pytest.approx(
        expected_personal_pd * 0.73 * 1663.0 / (1 + 0.0886) ** 0.5
    )
    assert personal_upside["probability_weight"] == pytest.approx(
        credit_stress.SCENARIO_PROBABILITY_WEIGHTS["upside"]
    )

    mortgage_downside = result.loc[
        (result["segment"] == "mortgages") & (result["scenario"] == "downside")
    ].iloc[0]
    expected_mortgage_pd = credit_stress.mortgage_stressed_pd(0.0207, -1.0)
    assert mortgage_downside["pd_stressed"] == pytest.approx(expected_mortgage_pd)
    assert mortgage_downside["lgd"] == pytest.approx(0.16)
    assert mortgage_downside["ead_aud_m"] == pytest.approx(429996.0)
    assert mortgage_downside["discount_rate"] == pytest.approx(0.0680)
    assert mortgage_downside["ecl_aud_m"] == pytest.approx(
        expected_mortgage_pd * 0.16 * 429996.0 / (1 + 0.0680) ** 0.5
    )


def test_run_credit_stress_test_requires_weight_for_every_scenario():
    with pytest.raises(ValueError, match="Scenario weights missing"):
        credit_stress.run_credit_stress_test(scenario_deltas={"unknown_scenario": 1.0})
