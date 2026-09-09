import numpy as np
import pandas as pd

from src.models import tableau_export


def test_build_svar_shock_events_frame_filters_to_target_own_shock(monkeypatch):
    target = "cpi_yoy"
    systems = {
        "Synthetic": {
            "target": target,
            "columns": (target, *tableau_export.svar.COMMON_MACRO_COLUMNS),
            "ordering": (target, *tableau_export.svar.COMMON_MACRO_COLUMNS),
        }
    }
    index = pd.period_range("2023Q2", periods=12, freq="Q")
    frame = pd.DataFrame(1.0, index=index, columns=list(systems["Synthetic"]["columns"]))

    def fake_load_svar_level_frame(columns):
        assert columns == systems["Synthetic"]["columns"]
        return frame

    def fake_fit_svar(data, ordering):
        assert data.index.max() == tableau_export.SVAR_EXPORT_FORECAST_ORIGIN
        assert ordering == systems["Synthetic"]["ordering"]
        return object()

    def fake_structural_shocks(fitted):
        quarters = pd.period_range("2023Q3", periods=10, freq="Q")
        rows = []
        for shock in (*tableau_export.svar.COMMON_MACRO_COLUMNS, target):
            values = np.zeros(len(quarters))
            if shock == target:
                values[-1] = 10.0
            for quarter, value in zip(quarters, values, strict=True):
                rows.append({"quarter": quarter, "shock": shock, "shock_value": value})
        return pd.DataFrame(rows)

    monkeypatch.setattr(tableau_export, "SVAR_SYSTEMS", systems)
    monkeypatch.setattr(tableau_export.svar, "load_svar_level_frame", fake_load_svar_level_frame)
    monkeypatch.setattr(tableau_export.svar, "fit_svar", fake_fit_svar)
    monkeypatch.setattr(tableau_export.svar, "structural_shocks", fake_structural_shocks)

    events = tableau_export.build_svar_shock_events_frame()

    assert list(events.columns) == [
        "system",
        "target",
        "forecast_origin",
        "quarter",
        "shock",
        "shock_z",
        "flagged",
    ]
    assert set(events["shock"]) == {target}
    assert len(events) == 10
    flagged = events.loc[events["flagged"]]
    assert len(flagged) == 1
    assert flagged.iloc[0]["shock"] == target
    assert flagged.iloc[0]["shock_z"] > 2


def test_build_svar_historical_decomposition_frame_keeps_target_baseline_and_own_shock(
    monkeypatch,
):
    target = "trimmed_mean_cpi_yoy"
    systems = {
        "Synthetic": {
            "target": target,
            "columns": (target, *tableau_export.svar.COMMON_MACRO_COLUMNS),
            "ordering": (target, *tableau_export.svar.COMMON_MACRO_COLUMNS),
        }
    }
    index = pd.period_range("2023Q2", periods=12, freq="Q")
    frame = pd.DataFrame(1.0, index=index, columns=list(systems["Synthetic"]["columns"]))

    def fake_load_svar_level_frame(columns):
        assert columns == systems["Synthetic"]["columns"]
        return frame

    def fake_fit_svar(data, ordering):
        assert data.index.max() == tableau_export.SVAR_EXPORT_FORECAST_ORIGIN
        assert ordering == systems["Synthetic"]["ordering"]
        return object()

    def fake_historical_decomposition(fitted):
        rows = []
        for response in [target, "cash_rate"]:
            for component in ["baseline", target, *tableau_export.svar.COMMON_MACRO_COLUMNS]:
                rows.append(
                    {
                        "quarter": pd.Period("2025Q4", freq="Q"),
                        "response": response,
                        "component": component,
                        "contribution": 1.0,
                    }
                )
        return pd.DataFrame(rows)

    monkeypatch.setattr(tableau_export, "SVAR_SYSTEMS", systems)
    monkeypatch.setattr(tableau_export.svar, "load_svar_level_frame", fake_load_svar_level_frame)
    monkeypatch.setattr(tableau_export.svar, "fit_svar", fake_fit_svar)
    monkeypatch.setattr(
        tableau_export.svar,
        "historical_decomposition",
        fake_historical_decomposition,
    )

    decomposition = tableau_export.build_svar_historical_decomposition_frame()

    assert list(decomposition.columns) == [
        "system",
        "target",
        "forecast_origin",
        "quarter",
        "component",
        "contribution",
    ]
    assert set(decomposition["component"]) == {
        "baseline",
        target,
        *tableau_export.svar.COMMON_MACRO_COLUMNS,
    }
    assert len(decomposition) == 1 + 1 + len(tableau_export.svar.COMMON_MACRO_COLUMNS)


def test_build_credit_stress_frame_flattens_segment_by_scenario_payload():
    payload = {
        "forecast_origin": "2025Q4",
        "target_quarter": "2026Q4",
        "horizon": 4,
        "scenarios": [
            {"name": "downside", "probability_weight": 0.425, "delta_unemployment_cumulative": 0.5},
            {"name": "base", "probability_weight": 0.55, "delta_unemployment_cumulative": 0.2},
            {"name": "upside", "probability_weight": 0.025, "delta_unemployment_cumulative": -0.1},
        ],
        "segments": [
            {
                "segment": "personal_loans",
                "pd_base": 0.0878,
                "ur_sensitivity": 0.4,
                "lgd": 0.73,
                "ead_aud_m": 1663.0,
                "discount_rate": 0.0886,
                "pd_stressed_by_scenario": {
                    "downside": 0.0898,
                    "base": 0.0886,
                    "upside": 0.0878,
                },
                "ecl_aud_m_by_scenario": {
                    "downside": 109.0,
                    "base": 107.6,
                    "upside": 106.6,
                },
                "ecl_aud_m_12m_probability_weighted": 107.9,
            },
            {
                "segment": "mortgages",
                "pd_base": 0.0207,
                "ur_sensitivity": 0.6,
                "lgd": 0.16,
                "ead_aud_m": 429996.0,
                "discount_rate": 0.068,
                "pd_stressed_by_scenario": {
                    "downside": 0.0237,
                    "base": 0.0219,
                    "upside": 0.0207,
                },
                "ecl_aud_m_by_scenario": {
                    "downside": 1630.0,
                    "base": 1507.0,
                    "upside": 1424.0,
                },
                "ecl_aud_m_12m_probability_weighted": 1553.0,
            },
        ],
    }

    frame = tableau_export.build_credit_stress_frame(payload)

    assert list(frame.columns) == [
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
        "ecl_aud_m_12m_probability_weighted",
        "forecast_origin",
        "target_quarter",
        "horizon",
    ]
    assert len(frame) == 6
    assert set(frame["segment"]) == {"personal_loans", "mortgages"}
    assert set(frame["scenario"]) == {"downside", "base", "upside"}

    personal_downside = frame.loc[
        (frame["segment"] == "personal_loans") & (frame["scenario"] == "downside")
    ].iloc[0]
    assert personal_downside["probability_weight"] == 0.425
    assert personal_downside["delta_unemployment_cumulative"] == 0.5
    assert personal_downside["pd_stressed"] == 0.0898
    assert personal_downside["discount_rate"] == 0.0886
    assert personal_downside["ecl_aud_m"] == 109.0
    assert personal_downside["ecl_aud_m_12m_probability_weighted"] == 107.9
    assert personal_downside["forecast_origin"] == "2025Q4"
    assert personal_downside["target_quarter"] == "2026Q4"
    assert personal_downside["horizon"] == 4


def test_build_credit_stress_frame_returns_empty_frame_for_no_segments():
    assert tableau_export.build_credit_stress_frame({"segments": []}).empty
