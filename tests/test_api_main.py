from types import SimpleNamespace

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import statsmodels.api as sm

from api import main as api_main
from src.models import registry, tracking
from src.models.lstm import LSTMDirectFit, TrainWindowScaler


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


def _register_champion(run_id, family="sarima"):
    version = mlflow.register_model(
        registry._logged_model_uri(MlflowClient(), run_id),
        registry.REGISTERED_MODEL_NAME,
        tags={"model_family": family, "source_run_id": run_id},
    )
    MlflowClient().set_registered_model_alias(
        registry.REGISTERED_MODEL_NAME,
        registry.CHAMPION_ALIAS,
        version.version,
    )
    return version


def _write_curated_frame(path, end_quarter="2021Q4"):
    quarters = pd.period_range(end=pd.Period(end_quarter, freq="Q"), periods=16, freq="Q")
    pd.DataFrame(
        {
            "quarter": quarters.astype(str),
            "cpi_yoy": np.linspace(2.0, 4.0, len(quarters)),
        }
    ).to_csv(path, index=False)


def test_promote_champion_registers_lowest_eligible_mlflow_run(monkeypatch, tmp_path):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-promote-champion")
    sarima_run_id = _log_run_with_model("sarima", 1.2)
    _log_run_with_model("lstm", 1.8)

    result = registry.promote_champion()

    assert result.family == "sarima"
    assert result.run_id == sarima_run_id
    assert result.rmse_overall == 1.2
    run = MlflowClient().get_run(sarima_run_id)
    assert run.data.tags["logged_model_uri"].startswith("models:/")
    version = MlflowClient().get_model_version_by_alias(
        registry.REGISTERED_MODEL_NAME,
        registry.CHAMPION_ALIAS,
    )
    assert version.run_id == sarima_run_id
    assert version.tags["model_family"] == "sarima"


def test_models_metrics_and_sarima_forecast_use_champion_alias(monkeypatch, tmp_path):
    _configure_tmp_mlflow(monkeypatch, tmp_path)
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2021Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    run_id = _log_run_with_model("sarima", 1.25)
    version = _register_champion(run_id, family="sarima")

    registered = api_main.models()["registered_models"]
    champion = next(model for model in registered if model["name"] == registry.REGISTERED_MODEL_NAME)
    assert champion["versions"][0]["version"] == str(version.version)
    assert registry.CHAMPION_ALIAS in champion["versions"][0]["aliases"]

    metrics_payload = api_main.metrics()
    assert metrics_payload["model_family"] == "sarima"
    assert metrics_payload["run_id"] == run_id
    assert metrics_payload["metrics"]["rmse_overall"] == 1.25
    assert metrics_payload["metrics"]["rmse_h8"] == 1.33

    forecast_payload = api_main.forecast(api_main.ForecastRequest(horizon=3)).model_dump()
    assert forecast_payload["model_family"] == "sarima"
    assert forecast_payload["model_version"] == str(version.version)
    assert forecast_payload["horizon"] == 3
    assert len(forecast_payload["forecast"]) == 3
    assert forecast_payload["forecast_origin"] == "2021Q4"
    assert forecast_payload["quarters"] == ["2022Q1", "2022Q2", "2022Q3"]


def test_sarima_forecast_labels_come_from_model_when_curated_data_is_ahead(
    monkeypatch,
    tmp_path,
):
    _configure_tmp_mlflow(monkeypatch, tmp_path, "pytest-api-sarima-stale-curated")
    curated_path = tmp_path / "curated.csv"
    _write_curated_frame(curated_path, end_quarter="2022Q4")
    monkeypatch.setattr(api_main, "CURATED_DATA_PATH", curated_path)
    run_id = _log_run_with_model("sarima", 1.25, train_end="2021Q4")
    _register_champion(run_id, family="sarima")

    forecast_payload = api_main.forecast(api_main.ForecastRequest(horizon=2)).model_dump()

    assert forecast_payload["forecast_origin"] == "2021Q4"
    assert forecast_payload["quarters"] == ["2022Q1", "2022Q2"]


def test_lstm_forecast_branch_reuses_forecast_from_fit(monkeypatch):
    class FakeModel:
        def predict(self, values, verbose=0):
            assert values.shape == (1, 2, 1)
            return np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

    scaler = TrainWindowScaler(
        columns=("cpi_yoy",),
        mean_=pd.Series({"cpi_yoy": 10.0}),
        scale_=pd.Series({"cpi_yoy": 2.0}),
    )
    fitted = LSTMDirectFit(
        model=FakeModel(),
        scaler=scaler,
        feature_columns=("cpi_yoy",),
        target_column="cpi_yoy",
        lookback=2,
        horizon=4,
    )
    monkeypatch.setattr(api_main, "_load_lstm_fit", lambda *args, **kwargs: fitted)
    monkeypatch.setattr(
        api_main,
        "_current_lstm_frame",
        lambda: pd.DataFrame(
            {"cpi_yoy": [11.0, 12.0]},
            index=pd.period_range("2022Q3", periods=2, freq="Q"),
        ),
    )

    forecast = api_main._forecast_values(
        client=SimpleNamespace(),
        version=SimpleNamespace(run_id="run-id"),
        family="lstm",
        horizon=3,
    )

    assert forecast.values == [12.0, 14.0, 16.0]
    assert forecast.forecast_origin == "2022Q4"
    assert forecast.quarters == ["2023Q1", "2023Q2", "2023Q3"]
