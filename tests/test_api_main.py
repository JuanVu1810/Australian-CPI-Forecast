from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from api import main as api_main


@pytest.fixture(autouse=True)
def _clear_api_report_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api_main,
        "INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH",
        tmp_path / "missing_interval_calibration_validation.csv",
    )
    api_main._load_interval_coverage_report.cache_clear()
    api_main._load_interval_calibration_factors.cache_clear()
    yield
    api_main._load_interval_coverage_report.cache_clear()
    api_main._load_interval_calibration_factors.cache_clear()


def _use_tmp_models(monkeypatch, tmp_path, with_elastic_net=False):
    """Point the API at a tiny saved SARIMA (and optionally a placeholder Elastic Net file)."""
    paths = {"sarima": tmp_path / "sarima.joblib", "elastic_net": tmp_path / "elastic_net.joblib"}
    joblib.dump(_fit_tiny_statsmodels(), paths["sarima"])
    if with_elastic_net:
        joblib.dump({}, paths["elastic_net"])
    monkeypatch.setattr(api_main, "HEADLINE_MODEL_PATHS", paths)


def _patch_elastic_net_fit(monkeypatch, fit):
    """Serve ``fit`` for the Elastic Net file; every other model loads for real."""
    real_load_model = api_main._load_model
    monkeypatch.setattr(
        api_main,
        "_load_model",
        lambda path: fit if path.name == "elastic_net.joblib" else real_load_model(path),
    )


def _fit_tiny_statsmodels(train_end="2021Q4"):
    series = pd.Series(
        np.linspace(2.0, 4.0, 24),
        index=pd.period_range(end=pd.Period(train_end, freq="Q"), periods=24, freq="Q"),
    )
    return sm.tsa.SARIMAX(
        series,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)


def _write_curated_frame(path, end_quarter="2021Q4"):
    quarters = pd.period_range(end=pd.Period(end_quarter, freq="Q"), periods=16, freq="Q")
    pd.DataFrame(
        {
            "quarter": quarters.astype(str),
            "cpi_yoy": np.linspace(2.0, 4.0, len(quarters)),
        }
    ).to_csv(path, index=False)


def _write_rba_action_curated_frame(path, end_quarter="2021Q4"):
    quarters = pd.period_range(
        end=pd.Period(end_quarter, freq="Q"),
        periods=4,
        freq="Q",
    )
    pd.DataFrame(
        {
            "quarter": quarters.astype(str),
            "cpi_yoy": np.linspace(2.0, 3.0, len(quarters)),
            "cash_rate": [2.5, 2.75, 3.0, 3.25],
            "cash_rate_lag1": [2.25, 2.5, 2.75, 3.0],
            "unemployment_rate_change_lag1": [0.1, -0.1, 0.0, 0.2],
        }
    ).to_csv(path, index=False)


def _write_interval_calibration_validation_report(path, target_coverage=0.8):
    pd.DataFrame(
        {
            "model": ["sarima", "sarima", "elastic_net"],
            "horizon": [1, 2, 1],
            "calibration_n": [51, 51, 37],
            "validation_n": [22, 22, 16],
            "target_coverage": [target_coverage, target_coverage, target_coverage],
            "scale_factor": [0.740229, 0.941796, 1.01613],
            "raw_empirical_coverage": [0.681818, 0.590909, 0.3125],
            "calibrated_empirical_coverage": [0.681818, 0.590909, 0.375],
            "raw_mean_interval_width": [1.359118, 1.965402, 1.69655],
            "calibrated_mean_interval_width": [1.006058, 1.851007, 1.723916],
            "raw_binom_p_value": [0.180912, 0.02752, 0.000033],
            "calibrated_binom_p_value": [0.180912, 0.02752, 0.000248],
            "raw_significantly_miscalibrated": [False, True, True],
            "calibrated_significantly_miscalibrated": [False, True, True],
        }
    ).to_csv(path, index=False)


def _write_interval_calibration_factors(path, target_coverage=0.8):
    pd.DataFrame(
        {
            "model": ["sarima", "sarima"],
            "horizon": [1, 2],
            "scale_factor": [2.0, 0.5],
            "calibration_n": [51, 51],
            "target_coverage": [target_coverage, target_coverage],
        }
    ).to_csv(path, index=False)


def test_forecast_all_returns_saved_sarima_and_marks_missing_families(
    monkeypatch,
    tmp_path,
):
    _use_tmp_models(monkeypatch, tmp_path)
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(horizon=2, n_sims=100)
    ).model_dump()

    assert payload["requested_horizon"] == 2
    assert [model["model_family"] for model in payload["models"]] == ["sarima"]
    assert len(payload["models"][0]["forecast"]) == 2
    unavailable = {item["model_family"]: item["reason"] for item in payload["unavailable"]}
    assert set(unavailable) == {"elastic_net", "ensemble"}
    assert "elastic_net.joblib not found" in unavailable["elastic_net"]
    assert "elastic_net.joblib not found" in unavailable["ensemble"]


def test_rba_action_returns_live_policy_breakdown_and_marks_threshold(
    monkeypatch,
    tmp_path,
):
    curated_path = tmp_path / "curated.csv"
    _write_rba_action_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(
        api_main.rba_classifier,
        "assemble_policy_sample",
        lambda: pd.DataFrame({"policy_action": ["cut", "hold", "hike"]}),
    )
    monkeypatch.setattr(
        api_main.rba_classifier,
        "_load_threshold_simulation_frame",
        lambda curated_path: pd.DataFrame({"cpi_yoy": [2.0]}),
    )

    def fake_headline_ensemble(model_paths, requested_horizon, n_sims, seed):
        return api_main.FamilyForecastData(
            forecast=[3.2],
            draws=np.zeros((n_sims, 1)),
            quarters=["2022Q1"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    def fake_predict_single_quarter(train, test, **kwargs):
        assert test.iloc[0]["target_quarter"] == "2022Q1"
        assert test.iloc[0]["headline_forecast"] == pytest.approx(3.2)
        assert test.iloc[0]["cash_rate"] == pytest.approx(3.25)
        return pd.DataFrame(
            {
                "model": list(api_main.rba_classifier.MODEL_ORDER),
                "predicted_action": [
                    "hike",
                    "hold",
                    "hike",
                    "hold",
                    "hike",
                    "hold",
                    "hike",
                ],
                "confidence": [0.7, np.nan, 0.6, 0.5, 0.8, np.nan, np.nan],
                "p_cut": [0.1, np.nan, 0.1, 0.2, 0.1, np.nan, np.nan],
                "p_hold": [0.2, np.nan, 0.3, 0.5, 0.1, np.nan, np.nan],
                "p_hike": [0.7, np.nan, 0.6, 0.3, 0.8, np.nan, np.nan],
                "majority_vote_tie_break": [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                ],
            }
        )

    monkeypatch.setitem(api_main.FAMILY_HANDLERS, "ensemble", fake_headline_ensemble)
    predict_calls = []

    def counting_predict_single_quarter(*args, **kwargs):
        predict_calls.append(1)
        return fake_predict_single_quarter(*args, **kwargs)

    monkeypatch.setattr(
        api_main.rba_classifier,
        "predict_single_quarter",
        counting_predict_single_quarter,
    )

    payload = api_main.rba_action().model_dump()
    api_main.rba_action()
    assert len(predict_calls) == 1  # the second request is served from the cache

    assert payload["target_quarter"] == "2022Q1"
    assert payload["forecast_origin"] == "2021Q4"
    assert payload["reportable_model"] == "threshold"
    assert payload["reportable_action"] == "hike"
    assert payload["headline_forecast"] == pytest.approx(3.2)
    assert len(payload["models"]) == 7
    threshold = payload["models"][0]
    assert threshold["model"] == "threshold"
    assert threshold["reportable"] is True
    assert threshold["confidence"] == pytest.approx(0.7)
    assert payload["models"][1]["confidence"] is None
    assert payload["models"][-1]["majority_vote_tie_break"] is True
    assert "documented comparison exercise" in payload["caveat"]


def test_rba_action_converts_a_missing_model_to_503(monkeypatch, tmp_path):
    curated_path = tmp_path / "curated.csv"
    _write_rba_action_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(
        api_main.rba_classifier,
        "assemble_policy_sample",
        lambda: pd.DataFrame({"policy_action": ["cut", "hold", "hike"]}),
    )

    def raise_missing_model(model_paths, requested_horizon, n_sims, seed):
        raise RuntimeError("Saved model sarima.joblib not found. Train the models first (see README).")

    monkeypatch.setitem(api_main.FAMILY_HANDLERS, "ensemble", raise_missing_model)

    with pytest.raises(api_main.HTTPException) as exc_info:
        api_main.rba_action()

    assert exc_info.value.status_code == 503
    assert "Saved model sarima.joblib not found" in exc_info.value.detail


def test_credit_risk_stress_test_returns_svar_ecl_segments(monkeypatch, tmp_path):
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)

    scenario_names = list(api_main.credit_stress.SCENARIO_QUANTILES)
    fake_deltas = [-1.0, 0.0, 2.0]

    def fake_unemployment_change_quantiles(horizon, quantiles):
        assert horizon == 2
        assert list(quantiles) == [
            api_main.credit_stress.SCENARIO_QUANTILES[name] for name in scenario_names
        ]
        return fake_deltas, "2021Q3"

    monkeypatch.setattr(
        api_main.svar,
        "forecast_cumulative_unemployment_change_quantiles",
        fake_unemployment_change_quantiles,
    )

    payload = api_main.credit_risk_stress_test(horizon=2).model_dump()

    assert payload["forecast_origin"] == "2021Q3"
    assert payload["target_quarter"] == "2022Q1"
    assert payload["horizon"] == 2
    assert payload["caveat"] == api_main.credit_stress.CREDIT_STRESS_CAVEAT

    scenario_deltas = dict(zip(scenario_names, fake_deltas))
    weights = api_main.credit_stress.SCENARIO_PROBABILITY_WEIGHTS
    assert payload["scenarios"] == [
        {
            "name": name,
            "probability_weight": pytest.approx(weights[name]),
            "delta_unemployment_cumulative": pytest.approx(scenario_deltas[name]),
        }
        for name in scenario_names
    ]

    personal_pd = {
        name: api_main.credit_stress.personal_loan_stressed_pd(0.0878, delta)
        for name, delta in scenario_deltas.items()
    }
    personal_ecl = {
        name: pd_stressed * 0.73 * 1663.0 / (1 + 0.0886)
        for name, pd_stressed in personal_pd.items()
    }
    mortgage_pd = {
        name: api_main.credit_stress.mortgage_stressed_pd(0.0207, delta)
        for name, delta in scenario_deltas.items()
    }
    mortgage_ecl = {
        name: pd_stressed * 0.16 * 429996.0 / (1 + 0.0680)
        for name, pd_stressed in mortgage_pd.items()
    }

    assert payload["segments"] == [
        {
            "segment": "personal_loans",
            "pd_base": pytest.approx(0.0878),
            "ur_sensitivity": pytest.approx(0.4),
            "lgd": pytest.approx(0.73),
            "ead_aud_m": pytest.approx(1663.0),
            "discount_rate": pytest.approx(0.0886),
            "pd_stressed_by_scenario": {
                name: pytest.approx(value) for name, value in personal_pd.items()
            },
            "ecl_aud_m_by_scenario": {
                name: pytest.approx(value) for name, value in personal_ecl.items()
            },
            "ecl_aud_m_12m_probability_weighted": pytest.approx(
                sum(weights[name] * value for name, value in personal_ecl.items())
            ),
        },
        {
            "segment": "mortgages",
            "pd_base": pytest.approx(0.0207),
            "ur_sensitivity": pytest.approx(0.6),
            "lgd": pytest.approx(0.16),
            "ead_aud_m": pytest.approx(429996.0),
            "discount_rate": pytest.approx(0.0680),
            "pd_stressed_by_scenario": {
                name: pytest.approx(value) for name, value in mortgage_pd.items()
            },
            "ecl_aud_m_by_scenario": {
                name: pytest.approx(value) for name, value in mortgage_ecl.items()
            },
            "ecl_aud_m_12m_probability_weighted": pytest.approx(
                sum(weights[name] * value for name, value in mortgage_ecl.items())
            ),
        },
    ]


def test_credit_risk_stress_test_converts_value_errors_to_503(monkeypatch, tmp_path):
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    bad_pd_base_path = tmp_path / "pd_base_assumptions.csv"
    pd.DataFrame(
        {
            "segment": ["personal_loans", "mortgages"],
            "pd_base": [0.03, 0.005],
        }
    ).to_csv(bad_pd_base_path, index=False)
    real_run_credit_stress_test = api_main.credit_stress.run_credit_stress_test

    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(
        api_main.svar,
        "forecast_cumulative_unemployment_change_quantiles",
        lambda horizon, quantiles: ([1.0] * len(quantiles), "2021Q4"),
    )

    def run_with_malformed_assumptions(scenario_deltas):
        return real_run_credit_stress_test(
            scenario_deltas=scenario_deltas,
            pd_base_path=bad_pd_base_path,
        )

    monkeypatch.setattr(
        api_main.credit_stress,
        "run_credit_stress_test",
        run_with_malformed_assumptions,
    )

    with pytest.raises(api_main.HTTPException) as exc_info:
        api_main.credit_risk_stress_test()

    assert exc_info.value.status_code == 503
    assert "missing required columns" in exc_info.value.detail


def test_forecast_all_returns_sarima_and_elastic_net_with_draw_intervals(
    monkeypatch,
    tmp_path,
):
    class FakeElasticNetFit:
        feature_columns = ("cpi_yoy_lag1",)
        horizons = (1, 2, 3)

        def predict_next(self, train_frame):
            return [10.0, 20.0, 30.0]

    _use_tmp_models(monkeypatch, tmp_path, with_elastic_net=True)
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    _patch_elastic_net_fit(monkeypatch, FakeElasticNetFit())
    monkeypatch.setattr(  # the ensemble reads real report CSVs; it has its own tests
        api_main,
        "FAMILY_HANDLERS",
        {name: api_main.FAMILY_HANDLERS[name] for name in ("sarima", "elastic_net")},
    )
    monkeypatch.setattr(
        api_main,
        "_current_elastic_net_frame",
        lambda **kwargs: pd.DataFrame(
            {"cpi_yoy_lag1": [1.0, 2.0]},
            index=pd.period_range("2021Q3", periods=2, freq="Q"),
        ),
    )
    monkeypatch.setattr(
        api_main.elastic_net,
        "simulate_paths_from_fit",
        lambda fitted, train_frame, steps, n_sims, seed: np.tile(
            np.arange(1, steps + 1, dtype=float),
            (n_sims, 1),
        ),
    )

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(horizon=3, n_sims=100)
    ).model_dump()

    models = {model["model_family"]: model for model in payload["models"]}
    assert set(models) == {"sarima", "elastic_net"}
    for model in models.values():
        assert len(model["interval_lower"]) == 3
        assert len(model["interval_upper"]) == 3
        assert len(model["forecast"]) == 3
    assert models["elastic_net"]["forecast"] == [10.0, 20.0, 30.0]


def test_forecast_all_attaches_held_out_interval_coverage_for_validated_bounds(
    monkeypatch,
    tmp_path,
):
    _use_tmp_models(monkeypatch, tmp_path)
    curated_path = tmp_path / "curated.csv"
    validation_path = tmp_path / "held_out_validation.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_calibration_validation_report(validation_path)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(
        api_main,
        "INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH",
        validation_path,
    )
    api_main._load_interval_coverage_report.cache_clear()

    def fake_sarima_family_forecast(model_paths, requested_horizon, n_sims, seed):
        return api_main.FamilyForecastData(
            forecast=[1.0, 2.0],
            draws=np.tile(np.array([0.5, 2.5]), (n_sims, 1)),
            quarters=["2022Q1", "2022Q2"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    monkeypatch.setattr(api_main, "FAMILY_HANDLERS", {"sarima": fake_sarima_family_forecast})

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(horizon=2, n_sims=100)
    ).model_dump()

    model = payload["models"][0]
    assert model["model_family"] == "sarima"
    assert model["empirical_coverage"] == [0.681818, 0.590909]
    assert model["coverage_n"] == [22, 22]
    assert model["significantly_miscalibrated"] == [False, True]


def test_forecast_all_leaves_interval_coverage_null_for_unvalidated_bounds(
    monkeypatch,
    tmp_path,
):
    _use_tmp_models(monkeypatch, tmp_path)
    curated_path = tmp_path / "curated.csv"
    validation_path = tmp_path / "held_out_validation.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_calibration_validation_report(validation_path)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(
        api_main,
        "INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH",
        validation_path,
    )
    api_main._load_interval_coverage_report.cache_clear()

    def fake_sarima_family_forecast(model_paths, requested_horizon, n_sims, seed):
        return api_main.FamilyForecastData(
            forecast=[1.0, 2.0],
            draws=np.tile(np.array([0.5, 2.5]), (n_sims, 1)),
            quarters=["2022Q1", "2022Q2"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    monkeypatch.setattr(api_main, "FAMILY_HANDLERS", {"sarima": fake_sarima_family_forecast})

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(
            horizon=2,
            n_sims=100,
            interval_lower=0.2,
            interval_upper=0.8,
        )
    ).model_dump()

    model = payload["models"][0]
    assert model["empirical_coverage"] == [None, None]
    assert model["coverage_n"] == [None, None]
    assert model["significantly_miscalibrated"] == [None, None]


def test_forecast_all_leaves_interval_coverage_null_for_validation_nominal_mismatch(
    monkeypatch,
    tmp_path,
):
    _use_tmp_models(monkeypatch, tmp_path)
    curated_path = tmp_path / "curated.csv"
    validation_path = tmp_path / "held_out_validation.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_calibration_validation_report(validation_path, target_coverage=0.7)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(
        api_main,
        "INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH",
        validation_path,
    )
    api_main._load_interval_coverage_report.cache_clear()

    def fake_sarima_family_forecast(model_paths, requested_horizon, n_sims, seed):
        return api_main.FamilyForecastData(
            forecast=[1.0, 2.0],
            draws=np.tile(np.array([0.5, 2.5]), (n_sims, 1)),
            quarters=["2022Q1", "2022Q2"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    monkeypatch.setattr(api_main, "FAMILY_HANDLERS", {"sarima": fake_sarima_family_forecast})

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(horizon=2, n_sims=100)
    ).model_dump()

    model = payload["models"][0]
    assert model["empirical_coverage"] == [None, None]
    assert model["coverage_n"] == [None, None]
    assert model["significantly_miscalibrated"] == [None, None]


def test_forecast_all_leaves_interval_coverage_null_when_report_missing(
    monkeypatch,
    tmp_path,
):
    _use_tmp_models(monkeypatch, tmp_path)
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    api_main._load_interval_coverage_report.cache_clear()  # the validation report is missing in this test

    def fake_sarima_family_forecast(model_paths, requested_horizon, n_sims, seed):
        return api_main.FamilyForecastData(
            forecast=[1.0, 2.0],
            draws=np.tile(np.array([0.5, 2.5]), (n_sims, 1)),
            quarters=["2022Q1", "2022Q2"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    monkeypatch.setattr(api_main, "FAMILY_HANDLERS", {"sarima": fake_sarima_family_forecast})

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(horizon=2, n_sims=100)
    ).model_dump()

    assert payload["unavailable"] == []
    assert payload["models"][0]["empirical_coverage"] == [None, None]
    assert payload["models"][0]["coverage_n"] == [None, None]
    assert payload["models"][0]["significantly_miscalibrated"] == [None, None]


def test_forecast_all_applies_interval_calibration_for_validated_bounds(
    monkeypatch,
    tmp_path,
):
    _use_tmp_models(monkeypatch, tmp_path)
    curated_path = tmp_path / "curated.csv"
    factors_path = tmp_path / "calibration_factors.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_calibration_factors(factors_path)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(api_main, "INTERVAL_CALIBRATION_FACTORS_PATH", factors_path)
    api_main._load_interval_calibration_factors.cache_clear()

    def fake_sarima_family_forecast(model_paths, requested_horizon, n_sims, seed):
        return api_main.FamilyForecastData(
            forecast=[10.0, 20.0],
            draws=np.column_stack(
                [
                    np.linspace(9.0, 11.0, n_sims),
                    np.linspace(18.0, 22.0, n_sims),
                ]
            ),
            quarters=["2022Q1", "2022Q2"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    monkeypatch.setattr(api_main, "FAMILY_HANDLERS", {"sarima": fake_sarima_family_forecast})

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(horizon=2, n_sims=100)
    ).model_dump()

    model = payload["models"][0]
    np.testing.assert_allclose(model["interval_lower"], [8.4, 19.2])
    np.testing.assert_allclose(model["interval_upper"], [11.6, 20.8])


def test_forecast_all_scales_a_skewed_interval_on_each_side_around_the_point(
    monkeypatch,
    tmp_path,
):
    _use_tmp_models(monkeypatch, tmp_path)
    curated_path = tmp_path / "curated.csv"
    factors_path = tmp_path / "calibration_factors.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_calibration_factors(factors_path)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(api_main, "INTERVAL_CALIBRATION_FACTORS_PATH", factors_path)
    api_main._load_interval_calibration_factors.cache_clear()

    def fake_sarima_family_forecast(model_paths, requested_horizon, n_sims, seed):
        # raw 10th/90th percentiles land at about 9.8 and 12.0 around a point of 10.0
        return api_main.FamilyForecastData(
            forecast=[10.0],
            draws=np.linspace(9.75, 12.25, n_sims).reshape(n_sims, 1),
            quarters=["2022Q1"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    monkeypatch.setattr(api_main, "FAMILY_HANDLERS", {"sarima": fake_sarima_family_forecast})

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(horizon=1, n_sims=1000)
    ).model_dump()

    model = payload["models"][0]
    raw_lower, raw_upper = np.quantile(np.linspace(9.75, 12.25, 1000), [0.1, 0.9])
    np.testing.assert_allclose(model["interval_lower"], [10.0 - 2.0 * (10.0 - raw_lower)])
    np.testing.assert_allclose(model["interval_upper"], [10.0 + 2.0 * (raw_upper - 10.0)])
    lower_dist = 10.0 - model["interval_lower"][0]
    upper_dist = model["interval_upper"][0] - 10.0
    assert upper_dist > 3 * lower_dist


def test_forecast_all_refuses_to_serve_when_calibration_factors_are_missing(
    monkeypatch,
    tmp_path,
):
    _use_tmp_models(monkeypatch, tmp_path)
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(
        api_main, "INTERVAL_CALIBRATION_FACTORS_PATH", tmp_path / "missing_factors.csv"
    )
    api_main._load_interval_calibration_factors.cache_clear()

    with pytest.raises(api_main.HTTPException) as exc_info:
        api_main.forecast_all(api_main.AllForecastsRequest(horizon=1, n_sims=100))

    assert exc_info.value.status_code == 503
    assert "missing_factors.csv" in exc_info.value.detail


def test_forecast_all_keeps_raw_intervals_when_calibration_bounds_do_not_match(
    monkeypatch,
    tmp_path,
):
    _use_tmp_models(monkeypatch, tmp_path)
    curated_path = tmp_path / "curated.csv"
    factors_path = tmp_path / "calibration_factors.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_calibration_factors(factors_path)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(api_main, "INTERVAL_CALIBRATION_FACTORS_PATH", factors_path)
    api_main._load_interval_calibration_factors.cache_clear()

    def fake_sarima_family_forecast(model_paths, requested_horizon, n_sims, seed):
        return api_main.FamilyForecastData(
            forecast=[10.0],
            draws=np.linspace(9.0, 11.0, n_sims).reshape(n_sims, 1),
            quarters=["2022Q1"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    monkeypatch.setattr(api_main, "FAMILY_HANDLERS", {"sarima": fake_sarima_family_forecast})

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(
            horizon=1,
            n_sims=100,
            interval_lower=0.2,
            interval_upper=0.8,
        )
    ).model_dump()

    model = payload["models"][0]
    np.testing.assert_allclose(model["interval_lower"], [9.4])
    np.testing.assert_allclose(model["interval_upper"], [10.6])


def test_forecast_all_marks_elastic_net_horizon_mismatch_unavailable(
    monkeypatch,
    tmp_path,
):
    class FakeElasticNetFit:
        feature_columns = ("cpi_yoy_lag1",)
        horizons = (1, 2)

        def predict_next(self, train_frame):
            return [10.0, 20.0]

    _use_tmp_models(monkeypatch, tmp_path, with_elastic_net=True)
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    _patch_elastic_net_fit(monkeypatch, FakeElasticNetFit())
    monkeypatch.setattr(
        api_main,
        "_current_elastic_net_frame",
        lambda **kwargs: pd.DataFrame(
            {"cpi_yoy_lag1": [1.0, 2.0]},
            index=pd.period_range("2022Q3", periods=2, freq="Q"),
        ),
    )

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(horizon=3, n_sims=100)
    ).model_dump()

    assert [model["model_family"] for model in payload["models"]] == ["sarima"]
    unavailable = {item["model_family"]: item["reason"] for item in payload["unavailable"]}
    assert "elastic_net" in unavailable
    assert "does not have fitted horizons [3]" in unavailable["elastic_net"]


def _fake_ensemble_components(monkeypatch, seen):
    """Replace the two component forecasts so the ensemble blend can be checked by hand."""

    def fake_sarima_family_forecast(model_paths, requested_horizon, n_sims, seed):
        seen["sarima_model_paths"] = model_paths
        return api_main.FamilyForecastData(
            forecast=[1.0, 2.0, 3.0],
            draws=np.tile(np.array([1.0, 2.0, 3.0]), (n_sims, 1)),
            quarters=["2022Q1", "2022Q2", "2022Q3"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    def fake_elastic_net_family_forecast(
        model_paths,
        requested_horizon,
        n_sims,
        seed,
        forecast_origin=None,
    ):
        seen["elastic_net_model_paths"] = model_paths
        seen["elastic_forecast_origin"] = forecast_origin
        return api_main.FamilyForecastData(
            forecast=[3.0, 4.0, 5.0],
            draws=np.tile(np.array([3.0, 4.0, 5.0]), (n_sims, 1)),
            quarters=["2022Q1", "2022Q2", "2022Q3"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    monkeypatch.setattr(api_main, "_sarima_family_forecast", fake_sarima_family_forecast)
    monkeypatch.setattr(api_main, "_elastic_net_family_forecast", fake_elastic_net_family_forecast)


def test_forecast_all_ensemble_averages_sarima_and_elastic_net_50_50(
    monkeypatch,
    tmp_path,
):
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(api_main, "FAMILY_HANDLERS", {"ensemble": api_main._ensemble_family_forecast})
    seen = {}
    _fake_ensemble_components(monkeypatch, seen)

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(horizon=3, n_sims=100)
    ).model_dump()

    ensemble = payload["models"][0]
    assert ensemble["model_family"] == "ensemble"
    assert ensemble["horizon"] == 3
    assert ensemble["horizon_cap"] is None
    assert ensemble["forecast"] == [2.0, 3.0, 4.0]
    assert ensemble["quarters"] == ["2022Q1", "2022Q2", "2022Q3"]
    assert len(ensemble["interval_lower"]) == 3
    assert len(ensemble["interval_upper"]) == 3
    assert seen["sarima_model_paths"] is api_main.HEADLINE_MODEL_PATHS
    assert seen["elastic_net_model_paths"] is api_main.HEADLINE_MODEL_PATHS
    assert seen["elastic_forecast_origin"] == "2021Q4"


def test_credit_stress_horizon_is_limited_to_the_forecast_horizons():
    openapi = api_main.app.openapi()
    horizon = openapi["paths"]["/credit-risk/stress-test"]["get"]["parameters"][0]["schema"]
    assert horizon["minimum"] == 1
    assert horizon["maximum"] == api_main.MAX_FORECAST_HORIZON


def test_interval_from_paths_matches_numpy_percentile():
    draws = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ]
    )

    lower, upper = api_main._interval_from_paths(draws, lower=0.25, upper=0.75)

    expected = np.percentile(draws, [25.0, 75.0], axis=0)
    np.testing.assert_allclose(lower, expected[0])
    np.testing.assert_allclose(upper, expected[1])


def test_public_message_shows_project_errors_but_hides_file_paths():
    assert api_main._public_message(RuntimeError("Saved model sarima.joblib not found.")) == (
        "Saved model sarima.joblib not found."
    )
    hidden = api_main._public_message(FileNotFoundError(2, "No such file", "/app/secret/path.csv"))
    assert "/app" not in hidden
    assert "FileNotFoundError" in hidden


def test_shipped_models_and_reports_serve_a_forecast():
    """Smoke test on the committed models/, data/ and reports/ files the Docker image copies."""
    api_main._load_interval_calibration_factors.cache_clear()
    response = api_main.forecast_all(api_main.AllForecastsRequest(horizon=2, n_sims=100))

    assert response.unavailable == []
    assert {model.model_family for model in response.models} == {"sarima", "elastic_net", "ensemble"}
    for model in response.models:
        assert len(model.forecast) == 2
        assert np.isfinite(model.forecast).all()


def test_health_reports_missing_files_with_503(monkeypatch, tmp_path):
    ok = api_main.Response()
    assert api_main.health(ok)["status"] == "ok"
    assert ok.status_code == 200

    monkeypatch.setitem(api_main.HEADLINE_MODEL_PATHS, "sarima", tmp_path / "gone.joblib")
    broken = api_main.Response()
    payload = api_main.health(broken)

    assert broken.status_code == 503
    assert payload["status"] == "degraded"
    assert payload["missing_files"] == [str(tmp_path / "gone.joblib")]


def test_load_model_turns_an_unreadable_pickle_into_a_runtime_error(tmp_path):
    bad = tmp_path / "sarima.joblib"
    bad.write_bytes(b"not a pickle")

    with pytest.raises(RuntimeError, match="sarima.joblib could not be loaded"):
        api_main._load_model(bad)


@pytest.mark.parametrize("shock_value", [float("nan"), float("inf"), 1e308])
def test_scenario_request_rejects_non_finite_or_huge_shocks(shock_value):
    with pytest.raises(ValueError):
        api_main.ScenarioForecastRequest(shock_variable="cash_rate", shock_value=shock_value)


def test_scenario_request_caps_the_number_of_simulations():
    with pytest.raises(ValueError):
        api_main.ScenarioForecastRequest(
            shock_variable="cash_rate", shock_value=1.0, n_sims=api_main.MAX_SCENARIO_SIMS + 1
        )


def test_validation_error_response_survives_a_nan_input():
    import asyncio
    import json

    from fastapi.exceptions import RequestValidationError

    try:
        api_main.ScenarioForecastRequest.model_validate_json(
            '{"shock_variable":"cash_rate","shock_value":NaN}'
        )
    except ValueError as exc:
        errors = exc.errors()
    response = asyncio.run(api_main._validation_error(None, RequestValidationError(errors)))

    assert response.status_code == 422
    assert json.loads(response.body)["detail"][0]["loc"] == ["shock_value"]
