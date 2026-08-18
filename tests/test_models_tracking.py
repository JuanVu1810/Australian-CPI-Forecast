import json
from types import SimpleNamespace

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from src.models import evaluation, lstm, sarimax_order_search, tracking
from src.models.sarima import fit_sarima as real_fit_sarima


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
    assert json.loads(run.data.params["order"]) == [2, 0, 2]
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


def test_sarimax_orchestrator_logs_parent_and_group_child_runs(monkeypatch, tmp_path):
    experiment_name = "pytest-sarimax-tracking"
    _configure_tmp_mlflow(monkeypatch, tmp_path, experiment_name)

    quarters = pd.period_range("2015Q1", periods=16, freq="Q")
    feature_columns = {
        "cash_rate_lag2",
        "cash_rate_change_lag1",
        "unemployment_rate_lag2",
        "unemployment_rate_change_lag1",
        "inflation_expectations_business_lag1",
        "ppi_growth_lag2",
        "commodity_growth_lag1",
        "wti_growth_lag1",
        "wpi_growth_lag1",
        "aud_usd_change_lag1",
        "brent_growth_lag1",
        "cash_rate_lag4",
        "unemployment_rate_lag4",
        "ppi_growth_lag1",
        "household_spending_growth_lag1",
        "covid_shock_down_lag0",
        "covid_shock_rebound_lag1",
    }
    curated_path = tmp_path / "curated.csv"
    frame = pd.DataFrame({"quarter": quarters.astype(str), "cpi_yoy": np.linspace(2, 3, 16)})
    for column in feature_columns:
        frame[column] = np.linspace(0.1, 1.0, 16)
    frame.to_csv(curated_path, index=False)

    choices = {
        "cash_rate": sarimax_order_search.LevelChangeChoice(
            family="cash_rate",
            winner="change",
            winner_features=("cash_rate_change_lag1",),
            loser="level",
            winner_aic=1.0,
            loser_aic=2.0,
            aic_winner="change",
            aic_winner_aic=1.0,
            aic_runner_up="level",
            aic_runner_up_aic=2.0,
            aic_gap_abs=1.0,
            selection_note="selected change",
        ),
        "unemployment_rate": sarimax_order_search.LevelChangeChoice(
            family="unemployment_rate",
            winner="change",
            winner_features=("unemployment_rate_change_lag1",),
            loser="level",
            winner_aic=1.0,
            loser_aic=2.0,
            aic_winner="change",
            aic_winner_aic=1.0,
            aic_runner_up="level",
            aic_runner_up_aic=2.0,
            aic_gap_abs=1.0,
            selection_note="selected change",
        ),
    }

    class FakeFit:
        def __init__(self, columns):
            self.params = pd.Series({column: 0.1 for column in columns})
            self.bse = pd.Series({column: 0.01 for column in columns})
            self.pvalues = pd.Series({column: 0.5 for column in columns})
            self.aic = 1.0
            self.bic = 2.0
            self.mle_retvals = {"converged": True}

    monkeypatch.setattr(
        sarimax_order_search,
        "resolve_level_change_features",
        lambda *args, **kwargs: choices,
    )
    monkeypatch.setattr(
        sarimax_order_search,
        "run_sarimax_order_search",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "order": [(1, 0, 0)],
                "seasonal_order": [(0, 0, 0, 0)],
                "trend": ["n"],
                "converged": [True],
                "aic": [1.0],
                "bic": [2.0],
                "error": [""],
            }
        ),
    )
    monkeypatch.setattr(
        sarimax_order_search,
        "fit_sarimax",
        lambda series, exog, **kwargs: FakeFit(exog.columns),
    )
    monkeypatch.setattr(
        sarimax_order_search,
        "_group_predictions",
        lambda *args, **kwargs: _prediction_frame("sarimax", {1: 0.5}),
    )
    monkeypatch.setattr(
        sarimax_order_search,
        "_baseline_predictions",
        lambda series, sarimax_predictions, **kwargs: [sarimax_predictions],
    )
    monkeypatch.setattr(
        tracking,
        "log_statsmodels_model",
        lambda fitted: mlflow.log_dict({"fake_model": True}, "model/fake.json"),
    )

    sarimax_order_search.run_sarimax_comparison(
        curated_path=curated_path,
        rba_path=tmp_path / "missing_rba.csv",
        comparison_output_path=tmp_path / "reports/model_comparison_sarimax.csv",
        coefficient_output_path=tmp_path / "reports/sarimax_coefficients.csv",
        initial_train_size=8,
        horizons=(1, 2),
        max_p=1,
        max_q=0,
        max_p_seasonal=0,
        max_q_seasonal=0,
    )

    runs = _runs_for_experiment(experiment_name)
    assert len(runs) == 11
    parent = [run for run in runs if run.data.tags["run_role"] == "comparison_parent"]
    children = [
        run
        for run in runs
        if run.data.tags["run_role"] == "comparison_with_full_sample_model"
    ]
    assert len(parent) == 1
    assert len(children) == 10
    assert {run.data.tags["sarimax_feature_group_id"] for run in children} == set("ABCDEFGHIJ")
    group_a = next(run for run in children if run.data.tags["sarimax_feature_group_id"] == "A")
    assert json.loads(group_a.data.params["features"]) == [
        "cash_rate_change_lag1",
        "unemployment_rate_change_lag1",
    ]
    assert group_a.data.metrics["rmse_h1"] == 0.5
    assert "rmse_h2" not in group_a.data.metrics


def test_lstm_orchestrator_logs_lstm_metrics_artifacts_and_permutation_metrics(
    monkeypatch,
    tmp_path,
):
    experiment_name = "pytest-lstm-tracking"
    _configure_tmp_mlflow(monkeypatch, tmp_path, experiment_name)

    quarters = pd.period_range("2015Q1", periods=30, freq="Q")
    curated_path = tmp_path / "curated.csv"
    frame = pd.DataFrame({"quarter": quarters.astype(str), "cpi_yoy": np.linspace(2, 3, 30)})
    for column in lstm.LSTM_FEATURE_COLUMNS:
        frame[column] = np.linspace(0.1, 1.0, 30)
    frame.to_csv(curated_path, index=False)

    monkeypatch.setattr(
        lstm,
        "walk_forward_backtest_direct_multihorizon",
        lambda *args, **kwargs: _prediction_frame("lstm", {1: 0.25, 2: 0.75}),
    )
    monkeypatch.setattr(
        lstm,
        "walk_forward_backtest",
        lambda *args, **kwargs: _prediction_frame("sarima", {1: 5.0, 2: 5.0}),
    )
    monkeypatch.setattr(
        lstm,
        "seasonal_naive_backtest",
        lambda *args, **kwargs: _prediction_frame("seasonal_naive", {1: 6.0, 2: 6.0}),
    )

    def fake_permutation_importance(full_frame, seed, output_path):
        importance = pd.DataFrame(
            {
                "feature": ["cpi_yoy", "cash_rate_change_lag1"],
                "baseline_rmse": [1.0, 1.0],
                "permuted_rmse": [1.2, 0.9],
                "rmse_increase": [0.2, -0.1],
                "n_windows": [2, 2],
                "n_values": [16, 16],
                "n_repeats": [10, 10],
                "interpretation_note": ["test", "test"],
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(output_path, index=False)
        return importance

    class FakeScaler:
        columns = lstm.LSTM_INPUT_COLUMNS
        mean_ = pd.Series({column: 0.0 for column in columns})
        scale_ = pd.Series({column: 1.0 for column in columns})

        def transform(self, values):
            return np.zeros((len(values), len(self.columns)), dtype=np.float32)

    fake_fit = SimpleNamespace(
        model=object(),
        scaler=FakeScaler(),
        feature_columns=lstm.LSTM_INPUT_COLUMNS,
        target_column="cpi_yoy",
        lookback=8,
        horizon=8,
    )

    monkeypatch.setattr(lstm, "permutation_importance", fake_permutation_importance)
    monkeypatch.setattr(lstm, "fit_lstm_direct", lambda *args, **kwargs: fake_fit)
    monkeypatch.setattr(
        tracking,
        "log_lstm_keras_model",
        lambda fitted, train_frame: mlflow.log_dict({"fake_model": True}, "model/fake.json"),
    )

    lstm.run_lstm_comparison(
        curated_path=curated_path,
        rba_path=tmp_path / "missing_rba.csv",
        comparison_output_path=tmp_path / "reports/model_comparison_lstm.csv",
        permutation_output_path=tmp_path / "reports/lstm_permutation_importance.csv",
        initial_train_size=12,
        horizons=(1, 2),
        seed=123,
        max_origins=1,
    )

    runs = _runs_for_experiment(experiment_name)
    assert len(runs) == 1
    run = runs[0]
    assert run.data.tags["model_family"] == "lstm"
    assert run.data.tags["reused_feature_group_id"] == "D"
    assert run.data.params["lookback"] == "8"
    assert run.data.params["seed"] == "123"
    assert run.data.metrics["rmse_h1"] == 0.25
    assert run.data.metrics["mae_h2"] == 0.75
    assert run.data.metrics["permutation_importance_cpi_yoy"] == 0.2
    assert run.data.metrics["permutation_importance_cash_rate_change_lag1"] == -0.1
