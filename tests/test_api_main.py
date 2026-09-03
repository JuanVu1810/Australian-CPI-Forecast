from types import SimpleNamespace

import mlflow
import mlflow.statsmodels
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from api import main as api_main
from src.models import registry, tracking


@pytest.fixture(autouse=True)
def _clear_api_report_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api_main,
        "INTERVAL_COVERAGE_REPORT_PATH",
        tmp_path / "missing_interval_coverage.csv",
    )
    monkeypatch.setattr(
        api_main,
        "INTERVAL_CALIBRATION_FACTORS_PATH",
        tmp_path / "missing_interval_calibration_factors.csv",
    )
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


def _configure_tmp_mlflow(monkeypatch, tmp_path, experiment_name="pytest-api-registry"):
    mlflow.end_run()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", experiment_name)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking.configure_mlflow()


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


def _log_run_with_model(family, rmse_overall, train_end="2021Q4"):
    with mlflow.start_run(run_name=f"{family}_candidate") as run:
        mlflow.set_tag("model_family", family)
        mlflow.log_metric("rmse_overall", rmse_overall)
        mlflow.log_metric("mae_overall", rmse_overall / 2)
        for horizon in range(1, 9):
            mlflow.log_metric(f"rmse_h{horizon}", rmse_overall + horizon / 100)
            mlflow.log_metric(f"mae_h{horizon}", rmse_overall / 2 + horizon / 100)
        tracking.log_statsmodels_model(_fit_tiny_statsmodels(train_end=train_end))
        return run.info.run_id


def _log_run_without_model(family, rmse_overall):
    with mlflow.start_run(run_name=f"{family}_candidate") as run:
        mlflow.set_tag("model_family", family)
        mlflow.log_metric("rmse_overall", rmse_overall)
        mlflow.log_metric("mae_overall", rmse_overall / 2)
        return run.info.run_id


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


def _write_interval_coverage_report(path, nominal_coverage=0.8):
    pd.DataFrame(
        {
            "model": ["sarima", "sarima", "elastic_net"],
            "horizon": [1, 2, 1],
            "n": [73, 73, 53],
            "nominal_coverage": [nominal_coverage, nominal_coverage, nominal_coverage],
            "empirical_coverage": [0.794521, 0.739726, 0.566038],
            "coverage_ci_lower": [0.684, 0.624, 0.423],
            "coverage_ci_upper": [0.880, 0.835, 0.702],
            "binom_p_value": [0.884, 0.190, 0.0001],
            "significantly_miscalibrated": [False, False, True],
            "mean_interval_width": [1.416, 1.940, 1.854],
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


def test_forecast_all_returns_registered_sarima_and_marks_missing_families(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-sarima-only")
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    _log_run_with_model("sarima", 1.25)

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(horizon=2, n_sims=100)
    ).model_dump()

    assert payload["requested_horizon"] == 2
    assert [model["model_family"] for model in payload["models"]] == ["sarima"]
    assert len(payload["models"][0]["forecast"]) == 2
    unavailable = {item["model_family"]: item["reason"] for item in payload["unavailable"]}
    assert set(unavailable) == {"elastic_net", "ensemble"}
    assert "No finished MLflow run found" in unavailable["elastic_net"]
    assert "No finished MLflow run found" in unavailable["ensemble"]


def test_trimmed_mean_forecast_all_uses_trimmed_handlers_and_calibration(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-trimmed-forecast-all")
    curated_path = tmp_path / "curated.csv"
    factors_path = tmp_path / "trimmed_calibration_factors.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_calibration_factors(factors_path)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(
        api_main,
        "TRIMMED_MEAN_INTERVAL_CALIBRATION_FACTORS_PATH",
        factors_path,
    )
    api_main._load_interval_calibration_factors.cache_clear()
    run_id = _log_run_with_model("trimmed_mean_sarima", 1.05)

    def fake_trimmed_sarima_family_forecast(model_uri, requested_horizon, n_sims, seed):
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

    monkeypatch.setattr(
        api_main,
        "TRIMMED_MEAN_FAMILY_HANDLERS",
        {"sarima": fake_trimmed_sarima_family_forecast},
    )

    payload = api_main.forecast_trimmed_mean_all(
        api_main.AllForecastsRequest(horizon=2, n_sims=100)
    ).model_dump()

    assert payload["requested_horizon"] == 2
    assert payload["unavailable"] == []
    model = payload["models"][0]
    assert model["model_family"] == "sarima"
    assert model["run_id"] == run_id
    assert model["forecast"] == [10.0, 20.0]
    np.testing.assert_allclose(model["interval_lower"], [8.4, 19.2])
    np.testing.assert_allclose(model["interval_upper"], [11.6, 20.8])


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

    def fake_headline_ensemble(model_uri, requested_horizon, n_sims, seed):
        return api_main.FamilyForecastData(
            forecast=[3.2],
            draws=np.zeros((n_sims, 1)),
            quarters=["2022Q1"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    def fake_trimmed_ensemble(model_uri, requested_horizon, n_sims, seed):
        return api_main.FamilyForecastData(
            forecast=[2.7],
            draws=np.zeros((n_sims, 1)),
            quarters=["2022Q1"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    def fake_predict_single_quarter(train, test, **kwargs):
        assert test.iloc[0]["target_quarter"] == "2022Q1"
        assert test.iloc[0]["headline_forecast"] == pytest.approx(3.2)
        assert test.iloc[0]["trimmed_mean_forecast"] == pytest.approx(2.7)
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

    monkeypatch.setattr(api_main, "_ensemble_family_forecast", fake_headline_ensemble)
    monkeypatch.setattr(
        api_main,
        "_ensemble_trimmed_mean_family_forecast",
        fake_trimmed_ensemble,
    )
    monkeypatch.setattr(
        api_main.rba_classifier,
        "predict_single_quarter",
        fake_predict_single_quarter,
    )

    payload = api_main.rba_action().model_dump()

    assert payload["target_quarter"] == "2022Q1"
    assert payload["forecast_origin"] == "2021Q4"
    assert payload["reportable_model"] == "threshold"
    assert payload["reportable_action"] == "hike"
    assert payload["headline_forecast"] == pytest.approx(3.2)
    assert payload["trimmed_mean_forecast"] == pytest.approx(2.7)
    assert len(payload["models"]) == 7
    threshold = payload["models"][0]
    assert threshold["model"] == "threshold"
    assert threshold["reportable"] is True
    assert threshold["confidence"] == pytest.approx(0.7)
    assert payload["models"][1]["confidence"] is None
    assert payload["models"][-1]["majority_vote_tie_break"] is True
    assert "documented comparison exercise" in payload["caveat"]


def test_rba_action_converts_mlflow_errors_to_503(monkeypatch, tmp_path):
    curated_path = tmp_path / "curated.csv"
    _write_rba_action_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(
        api_main.rba_classifier,
        "assemble_policy_sample",
        lambda: pd.DataFrame({"policy_action": ["cut", "hold", "hike"]}),
    )

    def raise_mlflow_exception(model_uri, requested_horizon, n_sims, seed):
        raise api_main.MlflowException("No finished MLflow run found.")

    monkeypatch.setattr(api_main, "_ensemble_family_forecast", raise_mlflow_exception)

    with pytest.raises(api_main.HTTPException) as exc_info:
        api_main.rba_action()

    assert exc_info.value.status_code == 503
    assert "No finished MLflow run found" in exc_info.value.detail


def test_credit_risk_stress_test_returns_svar_pd_segments(monkeypatch, tmp_path):
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)

    def fake_unemployment_change(horizon):
        assert horizon == 2
        return 2.0

    monkeypatch.setattr(
        api_main.svar,
        "forecast_cumulative_unemployment_change",
        fake_unemployment_change,
    )

    payload = api_main.credit_risk_stress_test(horizon=2).model_dump()

    assert payload["forecast_origin"] == "2021Q4"
    assert payload["target_quarter"] == "2022Q2"
    assert payload["horizon"] == 2
    assert payload["delta_unemployment_cumulative"] == pytest.approx(2.0)
    assert payload["caveat"] == api_main.credit_stress.CREDIT_STRESS_CAVEAT
    assert payload["segments"] == [
        {
            "segment": "personal_loans",
            "pd_base": pytest.approx(0.03),
            "ur_sensitivity": pytest.approx(0.4),
            "pd_stressed": pytest.approx(0.038),
        },
        {
            "segment": "mortgages",
            "pd_base": pytest.approx(0.005),
            "ur_sensitivity": pytest.approx(0.6),
            "pd_stressed": pytest.approx(0.017),
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
        "forecast_cumulative_unemployment_change",
        lambda horizon: 1.0,
    )

    def run_with_malformed_assumptions(delta_unemployment_cumulative):
        return real_run_credit_stress_test(
            delta_unemployment_cumulative=delta_unemployment_cumulative,
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

    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-sarima-elastic")
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    _log_run_with_model("sarima", 1.25)
    _log_run_with_model("elastic_net", 1.15)
    monkeypatch.setattr(api_main, "_load_elastic_net_fit", lambda model_uri: FakeElasticNetFit())
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


def test_forecast_all_attaches_interval_coverage_for_validated_bounds(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-coverage")
    curated_path = tmp_path / "curated.csv"
    coverage_path = tmp_path / "coverage.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_coverage_report(coverage_path)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(api_main, "INTERVAL_COVERAGE_REPORT_PATH", coverage_path)
    api_main._load_interval_coverage_report.cache_clear()
    sarima_run_id = _log_run_with_model("sarima", 1.25)

    def fake_sarima_family_forecast(model_uri, requested_horizon, n_sims, seed):
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
    model = payload["models"][0]
    assert model["model_family"] == "sarima"
    assert model["run_id"] == sarima_run_id
    assert model["empirical_coverage"] == [0.794521, 0.739726]
    assert model["coverage_n"] == [73, 73]
    assert model["significantly_miscalibrated"] == [False, False]


def test_forecast_all_prefers_held_out_calibration_validation_for_disclosure(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-validation-coverage")
    curated_path = tmp_path / "curated.csv"
    coverage_path = tmp_path / "full_sample_coverage.csv"
    validation_path = tmp_path / "held_out_validation.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_coverage_report(coverage_path)
    _write_interval_calibration_validation_report(validation_path)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(api_main, "INTERVAL_COVERAGE_REPORT_PATH", coverage_path)
    monkeypatch.setattr(
        api_main,
        "INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH",
        validation_path,
    )
    api_main._load_interval_coverage_report.cache_clear()
    sarima_run_id = _log_run_with_model("sarima", 1.25)

    def fake_sarima_family_forecast(model_uri, requested_horizon, n_sims, seed):
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
    assert model["run_id"] == sarima_run_id
    assert model["empirical_coverage"] == [0.681818, 0.590909]
    assert model["coverage_n"] == [22, 22]
    assert model["significantly_miscalibrated"] == [False, True]


def test_forecast_all_leaves_interval_coverage_null_for_unvalidated_bounds(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-coverage-bounds")
    curated_path = tmp_path / "curated.csv"
    coverage_path = tmp_path / "coverage.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_coverage_report(coverage_path)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(api_main, "INTERVAL_COVERAGE_REPORT_PATH", coverage_path)
    api_main._load_interval_coverage_report.cache_clear()
    _log_run_with_model("sarima", 1.25)

    def fake_sarima_family_forecast(model_uri, requested_horizon, n_sims, seed):
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


def test_forecast_all_leaves_interval_coverage_null_for_report_nominal_mismatch(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-coverage-nominal")
    curated_path = tmp_path / "curated.csv"
    coverage_path = tmp_path / "coverage.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_coverage_report(coverage_path, nominal_coverage=0.7)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(api_main, "INTERVAL_COVERAGE_REPORT_PATH", coverage_path)
    api_main._load_interval_coverage_report.cache_clear()
    _log_run_with_model("sarima", 1.25)

    def fake_sarima_family_forecast(model_uri, requested_horizon, n_sims, seed):
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


def test_forecast_all_leaves_interval_coverage_null_for_validation_nominal_mismatch(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-validation-nominal")
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
    _log_run_with_model("sarima", 1.25)

    def fake_sarima_family_forecast(model_uri, requested_horizon, n_sims, seed):
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
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-coverage-missing")
    curated_path = tmp_path / "curated.csv"
    coverage_path = tmp_path / "missing_coverage.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(api_main, "INTERVAL_COVERAGE_REPORT_PATH", coverage_path)
    api_main._load_interval_coverage_report.cache_clear()
    sarima_run_id = _log_run_with_model("sarima", 1.25)

    def fake_sarima_family_forecast(model_uri, requested_horizon, n_sims, seed):
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
    assert payload["models"][0]["run_id"] == sarima_run_id
    assert payload["models"][0]["empirical_coverage"] == [None, None]
    assert payload["models"][0]["coverage_n"] == [None, None]
    assert payload["models"][0]["significantly_miscalibrated"] == [None, None]


def test_forecast_all_applies_interval_calibration_for_validated_bounds(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-calibration")
    curated_path = tmp_path / "curated.csv"
    factors_path = tmp_path / "calibration_factors.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_calibration_factors(factors_path)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(api_main, "INTERVAL_CALIBRATION_FACTORS_PATH", factors_path)
    api_main._load_interval_calibration_factors.cache_clear()
    _log_run_with_model("sarima", 1.25)

    def fake_sarima_family_forecast(model_uri, requested_horizon, n_sims, seed):
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


def test_forecast_all_keeps_raw_intervals_when_calibration_report_missing(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-calibration-missing")
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    _log_run_with_model("sarima", 1.25)

    def fake_sarima_family_forecast(model_uri, requested_horizon, n_sims, seed):
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
        api_main.AllForecastsRequest(horizon=1, n_sims=100)
    ).model_dump()

    model = payload["models"][0]
    np.testing.assert_allclose(model["interval_lower"], [9.2])
    np.testing.assert_allclose(model["interval_upper"], [10.8])


def test_forecast_all_keeps_raw_intervals_when_calibration_bounds_do_not_match(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-calibration-bounds")
    curated_path = tmp_path / "curated.csv"
    factors_path = tmp_path / "calibration_factors.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    _write_interval_calibration_factors(factors_path)
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    monkeypatch.setattr(api_main, "INTERVAL_CALIBRATION_FACTORS_PATH", factors_path)
    api_main._load_interval_calibration_factors.cache_clear()
    _log_run_with_model("sarima", 1.25)

    def fake_sarima_family_forecast(model_uri, requested_horizon, n_sims, seed):
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

    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-elastic-missing")
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    _log_run_with_model("sarima", 1.25)
    _log_run_with_model("elastic_net", 1.15)
    monkeypatch.setattr(api_main, "_load_elastic_net_fit", lambda model_uri: FakeElasticNetFit())
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


def test_forecast_all_ensemble_uses_component_model_uris_without_own_model_artifact(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-forecast-all-ensemble")
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    sarima_run_id = _log_run_with_model("sarima", 1.25)
    elastic_net_run_id = _log_run_with_model("elastic_net", 1.15)
    ensemble_run_id = _log_run_without_model("ensemble", 1.05)
    seen_component_uris = {}

    def fake_sarima_family_forecast(model_uri, requested_horizon, n_sims, seed):
        seen_component_uris["sarima"] = model_uri
        return api_main.FamilyForecastData(
            forecast=[1.0, 2.0, 3.0],
            draws=np.tile(np.array([1.0, 2.0, 3.0]), (n_sims, 1)),
            quarters=["2022Q1", "2022Q2", "2022Q3"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    def fake_elastic_net_family_forecast(
        model_uri,
        requested_horizon,
        n_sims,
        seed,
        forecast_origin=None,
    ):
        seen_component_uris["elastic_net"] = model_uri
        seen_component_uris["elastic_forecast_origin"] = forecast_origin
        return api_main.FamilyForecastData(
            forecast=[3.0, 4.0, 5.0],
            draws=np.tile(np.array([3.0, 4.0, 5.0]), (n_sims, 1)),
            quarters=["2022Q1", "2022Q2", "2022Q3"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    monkeypatch.setattr(api_main, "_sarima_family_forecast", fake_sarima_family_forecast)
    monkeypatch.setattr(
        api_main,
        "_elastic_net_family_forecast",
        fake_elastic_net_family_forecast,
    )
    monkeypatch.setattr(
        api_main.ensemble,
        "horizon_rmse_weights",
        lambda horizons: {horizon: (0.25, 0.75) for horizon in horizons},
    )

    payload = api_main.forecast_all(
        api_main.AllForecastsRequest(horizon=3, n_sims=100)
    ).model_dump()

    models = {model["model_family"]: model for model in payload["models"]}
    assert "ensemble" in models
    assert models["ensemble"]["run_id"] == ensemble_run_id
    assert models["ensemble"]["horizon"] == 3
    assert models["ensemble"]["horizon_cap"] is None
    assert models["ensemble"]["forecast"] == [2.5, 3.5, 4.5]
    assert models["ensemble"]["quarters"] == ["2022Q1", "2022Q2", "2022Q3"]
    assert len(models["ensemble"]["interval_lower"]) == 3
    assert len(models["ensemble"]["interval_upper"]) == 3
    client = MlflowClient()
    assert seen_component_uris["sarima"] == registry._logged_model_uri(
        client,
        sarima_run_id,
    )
    assert seen_component_uris["elastic_net"] == registry._logged_model_uri(
        client,
        elastic_net_run_id,
    )
    assert seen_component_uris["elastic_forecast_origin"] == "2021Q4"
    assert ensemble_run_id not in seen_component_uris["sarima"]
    assert ensemble_run_id not in seen_component_uris["elastic_net"]


def test_trimmed_mean_ensemble_uses_trimmed_components_and_fixed_weights(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-trimmed-ensemble-components")
    sarima_run_id = _log_run_with_model("trimmed_mean_sarima", 1.25)
    elastic_net_run_id = _log_run_with_model("trimmed_mean_elastic_net", 1.15)
    seen_component_uris = {}

    def fake_sarima_family_forecast(model_uri, requested_horizon, n_sims, seed):
        seen_component_uris["sarima"] = model_uri
        return api_main.FamilyForecastData(
            forecast=[1.0, 2.0, 3.0],
            draws=np.tile(np.array([1.0, 2.0, 3.0]), (n_sims, 1)),
            quarters=["2022Q1", "2022Q2", "2022Q3"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    def fake_elastic_net_family_forecast(
        model_uri,
        requested_horizon,
        n_sims,
        seed,
        target_column,
        feature_columns,
        forecast_origin=None,
    ):
        seen_component_uris["elastic_net"] = model_uri
        seen_component_uris["elastic_target"] = target_column
        seen_component_uris["elastic_features"] = feature_columns
        seen_component_uris["elastic_forecast_origin"] = forecast_origin
        return api_main.FamilyForecastData(
            forecast=[3.0, 4.0, 5.0],
            draws=np.tile(np.array([3.0, 4.0, 5.0]), (n_sims, 1)),
            quarters=["2022Q1", "2022Q2", "2022Q3"],
            forecast_origin="2021Q4",
            horizon_served=requested_horizon,
            horizon_cap=None,
        )

    monkeypatch.setattr(api_main, "_sarima_family_forecast", fake_sarima_family_forecast)
    monkeypatch.setattr(
        api_main,
        "_elastic_net_family_forecast",
        fake_elastic_net_family_forecast,
    )
    monkeypatch.setattr(
        api_main.ensemble,
        "horizon_rmse_weights",
        lambda horizons: pytest.fail("trimmed-mean ensemble should use fixed weights"),
    )

    result = api_main._ensemble_trimmed_mean_family_forecast(
        "ignored",
        requested_horizon=3,
        n_sims=100,
        seed=42,
    )

    assert result.forecast == [2.0, 3.0, 4.0]
    assert result.quarters == ["2022Q1", "2022Q2", "2022Q3"]
    assert result.horizon_cap is None
    client = MlflowClient()
    assert seen_component_uris["sarima"] == registry._logged_model_uri(client, sarima_run_id)
    assert seen_component_uris["elastic_net"] == registry._logged_model_uri(
        client,
        elastic_net_run_id,
    )
    assert seen_component_uris["elastic_target"] == "trimmed_mean_cpi_yoy"
    assert seen_component_uris["elastic_features"][0] == "trimmed_mean_cpi_yoy_lag1"
    assert seen_component_uris["elastic_forecast_origin"] == "2021Q4"


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
