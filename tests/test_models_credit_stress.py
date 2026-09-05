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


def test_run_credit_stress_test_returns_two_segment_frame():
    result = credit_stress.run_credit_stress_test(delta_unemployment_cumulative=2.0)

    assert result["segment"].tolist() == ["personal_loans", "mortgages"]
    assert list(result.columns) == [
        "segment",
        "pd_base",
        "ur_sensitivity",
        "delta_unemployment_cumulative",
        "pd_stressed",
    ]
    assert result["ur_sensitivity"].tolist() == [0.4, 0.6]
    assert result["delta_unemployment_cumulative"].tolist() == [2.0, 2.0]
    assert result["pd_stressed"].tolist() == pytest.approx([0.0958, 0.0327])
