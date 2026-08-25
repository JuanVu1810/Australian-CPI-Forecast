import json
from types import SimpleNamespace

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from src.models import evaluation, tracking
from src.models.sarima import DEFAULT_ORDER, fit_sarima as real_fit_sarima


def _configure_tmp_mlflow(monkeypatch, tmp_path, experiment_name):
    mlflow.end_run()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", experiment_name)
    tracking.configure_mlflow()


def _runs_for_experiment(experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None
    return client.search_runs([experiment.experiment_id], order_by=["attributes.start_time ASC"])


def _prediction_frame(model_name, error_by_horizon):
    rows = []
    origins = pd.period_range("2020Q4", periods=2, freq="Q")
    for origin in origins:
        for horizon, error in error_by_horizon.items():
            actual = float(10 + horizon)
            rows.append(
                {
                    "model": model_name,
                    "forecast_origin": origin,
                    "target_quarter": origin + horizon,
                    "horizon": horizon,
                    "actual": actual,
                    "forecast": actual - float(error),
                    "error": float(error),
                }
            )
    return pd.DataFrame(rows)


def test_sarima_orchestrator_logs_own_metrics_artifact_and_reloadable_model(
    monkeypatch,
    tmp_path,
):
    experiment_name = "pytest-sarima-tracking"
    _configure_tmp_mlflow(monkeypatch, tmp_path, experiment_name)

    quarters = pd.period_range("2014Q1", periods=36, freq="Q")
    curated_path = tmp_path / "curated.csv"
    pd.DataFrame(
        {
            "quarter": quarters.astype(str),
            "cpi_yoy": np.linspace(2.0, 4.0, len(quarters)),
        }
    ).to_csv(curated_path, index=False)
    rba_path = tmp_path / "rba.csv"
    pd.DataFrame(
        {
            "forecast_date": ["2020-12-31", "2020-12-31"] * 2,
            "horizon_quarters": [1, 2] * 2,
            "rba_forecast_cpi_yoy": [8.0, 8.0] * 2,
            "rba_actual_cpi_yoy": [11.0, 12.0] * 2,
            "rba_forecast_error_cpi_yoy": [3.0, 4.0] * 2,
        }
    ).to_csv(rba_path, index=False)
    output_path = tmp_path / "reports/model_comparison_sarima.csv"

    def fake_walk_forward_backtest(*args, model_name, **kwargs):
        return _prediction_frame(model_name, {1: 1.0, 2: 2.0})

    def fake_seasonal_naive_backtest(*args, **kwargs):
        return _prediction_frame("seasonal_naive", {1: 9.0, 2: 9.0})

    def fast_final_fit(series, order, seasonal_order, trend="n", maxiter=100):
        return real_fit_sarima(
            series,
            order=(1, 0, 0),
            seasonal_order=(0, 0, 0, 0),
            trend=trend,
            maxiter=20,
        )

    monkeypatch.setattr(evaluation, "walk_forward_backtest", fake_walk_forward_backtest)
    monkeypatch.setattr(evaluation, "seasonal_naive_backtest", fake_seasonal_naive_backtest)
    import src.models.sarima as sarima_module

    monkeypatch.setattr(sarima_module, "fit_sarima", fast_final_fit)

    evaluation.run_sarima_comparison(
        curated_path=curated_path,
        rba_path=rba_path,
        output_path=output_path,
        initial_train_size=8,
        horizons=(1, 2),
    )

    runs = _runs_for_experiment(experiment_name)
    assert len(runs) == 1
    run = runs[0]
    assert run.data.tags["model_family"] == "sarima"
    assert json.loads(run.data.params["order"]) == list(DEFAULT_ORDER)
    assert json.loads(run.data.params["horizons"]) == [1, 2]
    assert run.data.metrics["rmse_h1"] == 1.0
    assert run.data.metrics["mae_h2"] == 2.0
    assert run.data.metrics["rmse_overall"] == np.sqrt(2.5)
    assert run.data.tags["logged_model_uri"].startswith("models:/")

    client = MlflowClient()
    report_artifacts = client.list_artifacts(run.info.run_id, "reports")
    assert [artifact.path for artifact in report_artifacts] == [
        "reports/model_comparison_sarima.csv"
    ]

    loaded = mlflow.statsmodels.load_model(run.data.tags["logged_model_uri"])
    forecast = loaded.forecast(steps=3)
    assert len(forecast) == 3


