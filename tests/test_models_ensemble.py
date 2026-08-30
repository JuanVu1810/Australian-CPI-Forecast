import numpy as np
import pandas as pd
import pytest

from src.models import ensemble
from src.models.elastic_net import ELASTIC_NET_FEATURE_COLUMNS


ENSEMBLE_COMPARISON_SOURCE = ensemble.DYNAMIC_WEIGHTS_SOURCE_PATH


def _synthetic_frame(n: int = 20) -> pd.DataFrame:
    index = pd.period_range("2018Q1", periods=n, freq="Q")
    signal = np.arange(n, dtype=float)
    frame = pd.DataFrame(
        {
            "cpi_yoy": 2.0 + 0.1 * signal + 0.2 * np.sin(signal / 2),
        },
        index=index,
    )
    for i, column in enumerate(ELASTIC_NET_FEATURE_COLUMNS, start=1):
        frame[column] = 0.05 * signal + i
    return frame


def _write_curated_csv(path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    output.insert(0, "quarter", output.index.astype(str))
    output.to_csv(path, index=False)


def _write_dynamic_weight_report(path, rows=None) -> None:
    if rows is None:
        rows = [
            {"group_id": "ELASTIC_NET", "model": "sarima", "horizon": 1, "rmse": 2.0},
            {"group_id": "ELASTIC_NET", "model": "elastic_net", "horizon": 1, "rmse": 1.0},
            {"group_id": "ELASTIC_NET", "model": "sarima", "horizon": 2, "rmse": 3.0},
            {"group_id": "ELASTIC_NET", "model": "elastic_net", "horizon": 2, "rmse": 6.0},
            {"group_id": "OTHER", "model": "sarima", "horizon": 1, "rmse": 99.0},
        ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_horizon_rmse_weights_reads_synthetic_report(tmp_path):
    report_path = tmp_path / "model_comparison_elastic_net.csv"
    _write_dynamic_weight_report(report_path)

    weights = ensemble.horizon_rmse_weights(horizons=(1, 2), path=report_path)

    assert weights[1][0] == pytest.approx(1.0 / 3.0)
    assert weights[1][1] == pytest.approx(2.0 / 3.0)
    assert weights[2][0] == pytest.approx(6.0 / 9.0)
    assert weights[2][1] == pytest.approx(3.0 / 9.0)
    for pair in weights.values():
        assert sum(pair) == pytest.approx(1.0)


def test_horizon_rmse_weights_missing_file_raises_clear_error(tmp_path):
    missing_report = tmp_path / "missing_model_comparison_elastic_net.csv"

    with pytest.raises(RuntimeError, match="Dynamic ensemble weight source report is missing"):
        ensemble.horizon_rmse_weights(horizons=(1,), path=missing_report)


def test_horizon_rmse_weights_missing_horizon_row_raises_clear_error(tmp_path):
    report_path = tmp_path / "model_comparison_elastic_net.csv"
    _write_dynamic_weight_report(
        report_path,
        rows=[
            {"group_id": "ELASTIC_NET", "model": "sarima", "horizon": 1, "rmse": 2.0},
            {"group_id": "ELASTIC_NET", "model": "elastic_net", "horizon": 1, "rmse": 1.0},
            {"group_id": "ELASTIC_NET", "model": "sarima", "horizon": 2, "rmse": 3.0},
        ],
    )

    with pytest.raises(RuntimeError, match="model 'elastic_net' at horizon 2"):
        ensemble.horizon_rmse_weights(horizons=(1, 2), path=report_path)


@pytest.mark.skipif(
    not ENSEMBLE_COMPARISON_SOURCE.exists(),
    reason="real Elastic Net comparison report is not available",
)
def test_horizon_rmse_weights_real_report_sanity_check():
    weights = ensemble.horizon_rmse_weights(
        horizons=(1, 4),
        path=ENSEMBLE_COMPARISON_SOURCE,
    )
    report = pd.read_csv(ENSEMBLE_COMPARISON_SOURCE)

    for horizon in (1, 4):
        rows = report.loc[
            report["group_id"].eq("ELASTIC_NET")
            & report["model"].isin(["sarima", "elastic_net"])
            & report["horizon"].astype(str).eq(str(horizon))
        ]
        rmse = dict(zip(rows["model"], rows["rmse"]))
        expected_sarima_weight = rmse["elastic_net"] / (rmse["sarima"] + rmse["elastic_net"])
        assert weights[horizon][0] == pytest.approx(expected_sarima_weight)


def test_combine_point_forecasts_weighted_average_and_validation():
    sarima_forecast = np.array([1.0, 2.0, 3.0])
    elastic_net_forecast = np.array([5.0, 6.0, 7.0])

    combined = ensemble.combine_point_forecasts(
        sarima_forecast,
        elastic_net_forecast,
        weights=(0.25, 0.75),
    )

    np.testing.assert_array_equal(combined, np.array([4.0, 5.0, 6.0]))
    with pytest.raises(ValueError, match="sum to 1"):
        ensemble.combine_point_forecasts(sarima_forecast, elastic_net_forecast, weights=(0.2, 0.2))
    with pytest.raises(ValueError, match="same length"):
        ensemble.combine_point_forecasts(sarima_forecast, elastic_net_forecast[:2])


def test_combine_point_forecasts_accepts_dict_weights_with_explicit_horizons():
    sarima_forecast = np.array([10.0, 20.0, 30.0])
    elastic_net_forecast = np.array([0.0, 100.0, 200.0])
    weights = {
        2: (0.25, 0.75),
        4: (0.75, 0.25),
        8: (0.5, 0.5),
    }

    combined = ensemble.combine_point_forecasts(
        sarima_forecast,
        elastic_net_forecast,
        weights=weights,
        horizons=np.array([4, 2, 8]),
    )

    np.testing.assert_array_equal(combined, np.array([7.5, 80.0, 115.0]))


def test_combine_paths_weighted_sum_and_validation():
    sarima_paths = np.array([[1.0, 2.0], [3.0, 4.0]])
    elastic_net_paths = np.array([[10.0, 20.0], [30.0, 40.0]])

    combined = ensemble.combine_paths(sarima_paths, elastic_net_paths, weights=(0.4, 0.6))

    np.testing.assert_array_equal(combined, np.array([[6.4, 12.8], [19.2, 25.6]]))
    with pytest.raises(ValueError, match="same shape"):
        ensemble.combine_paths(sarima_paths, elastic_net_paths[:, :1])
    with pytest.raises(ValueError, match="sum to 1"):
        ensemble.combine_paths(sarima_paths, elastic_net_paths, weights=(0.8, 0.8))


def test_combine_paths_accepts_dict_weights_with_explicit_horizons():
    sarima_paths = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    elastic_net_paths = np.array([[0.0, 100.0, 200.0], [10.0, 80.0, 120.0]])
    weights = {
        2: (0.25, 0.75),
        4: (0.75, 0.25),
        8: (0.5, 0.5),
    }

    combined = ensemble.combine_paths(
        sarima_paths,
        elastic_net_paths,
        weights=weights,
        horizons=np.array([4, 2, 8]),
    )

    np.testing.assert_array_equal(
        combined,
        np.array([[7.5, 80.0, 115.0], [32.5, 72.5, 90.0]]),
    )


def test_ensemble_interval_from_paths_uses_combined_draw_quantiles():
    paths = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    lower, upper = ensemble.ensemble_interval_from_paths(paths, lower=0.0, upper=1.0)

    np.testing.assert_array_equal(lower, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(upper, np.array([5.0, 6.0]))


def test_forecast_ensemble_matches_underlying_point_combination(monkeypatch):
    frame = _synthetic_frame()
    weights = (0.3, 0.7)
    sarima_point = np.array([1.0, 2.0, 3.0])
    elastic_net_point = np.array([4.0, 5.0, 6.0])

    monkeypatch.setattr(
        ensemble,
        "forecast_sarima",
        lambda series, steps, **kwargs: sarima_point[:steps],
    )
    monkeypatch.setattr(
        ensemble,
        "forecast_elastic_net_direct",
        lambda train_frame, steps, seed, **kwargs: elastic_net_point[:steps],
    )

    forecast = ensemble.forecast_ensemble(frame, steps=3, weights=weights, seed=123)
    expected = ensemble.combine_point_forecasts(
        ensemble.forecast_sarima(frame["cpi_yoy"], steps=3),
        ensemble.forecast_elastic_net_direct(frame, steps=3, seed=123),
        weights=weights,
    )

    np.testing.assert_array_equal(forecast, expected)


def test_forecast_ensemble_default_resolves_dynamic_weights(monkeypatch):
    frame = _synthetic_frame()
    sarima_point = np.array([10.0, 20.0, 30.0])
    elastic_net_point = np.array([0.0, 100.0, 200.0])
    requested = {}

    monkeypatch.setattr(
        ensemble,
        "forecast_sarima",
        lambda series, steps, **kwargs: sarima_point[:steps],
    )
    monkeypatch.setattr(
        ensemble,
        "forecast_elastic_net_direct",
        lambda train_frame, steps, seed, **kwargs: elastic_net_point[:steps],
    )

    def fake_horizon_rmse_weights(horizons, path=ensemble.DYNAMIC_WEIGHTS_SOURCE_PATH):
        requested["horizons"] = horizons
        return {
            1: (0.8, 0.2),
            2: (0.25, 0.75),
            3: (0.5, 0.5),
        }

    monkeypatch.setattr(ensemble, "horizon_rmse_weights", fake_horizon_rmse_weights)

    forecast = ensemble.forecast_ensemble(frame, steps=3, weights=None, seed=123)

    assert requested["horizons"] == (1, 2, 3)
    np.testing.assert_array_equal(forecast, np.array([8.0, 80.0, 115.0]))


def test_forecast_ensemble_passes_target_specific_component_configuration(monkeypatch):
    frame = _synthetic_frame()
    frame["trimmed_mean_cpi_yoy"] = frame["cpi_yoy"] - 0.2
    feature_columns = ("trimmed_mean_cpi_yoy_lag1", "commodity_growth_lag1")
    for column in feature_columns:
        frame[column] = np.arange(len(frame), dtype=float)
    calls = {}

    def fake_sarima(series, steps, order, seasonal_order):
        calls["sarima_name"] = series.name
        calls["sarima_order"] = order
        calls["sarima_seasonal_order"] = seasonal_order
        return np.array([1.0, 2.0])

    def fake_elastic_net(train_frame, steps, seed, feature_columns, target_column):
        calls["elastic_target"] = target_column
        calls["elastic_features"] = feature_columns
        return np.array([3.0, 4.0])

    monkeypatch.setattr(ensemble, "forecast_sarima", fake_sarima)
    monkeypatch.setattr(ensemble, "forecast_elastic_net_direct", fake_elastic_net)

    forecast = ensemble.forecast_ensemble(
        frame,
        steps=2,
        weights=(0.5, 0.5),
        target_column="trimmed_mean_cpi_yoy",
        sarima_order=(1, 1, 1),
        sarima_seasonal_order=(0, 0, 1, 4),
        elastic_net_feature_columns=feature_columns,
    )

    np.testing.assert_array_equal(forecast, np.array([2.0, 3.0]))
    assert calls == {
        "sarima_name": "trimmed_mean_cpi_yoy",
        "sarima_order": (1, 1, 1),
        "sarima_seasonal_order": (0, 0, 1, 4),
        "elastic_target": "trimmed_mean_cpi_yoy",
        "elastic_features": feature_columns,
    }


def test_simulate_ensemble_paths_shape_seed_divergence_and_forecast_mean(monkeypatch):
    frame = _synthetic_frame()
    weights = (0.25, 0.75)
    sarima_point = np.array([2.0, 3.0])
    elastic_net_point = np.array([6.0, 7.0])

    def fake_sarima_paths(series, steps, n_sims, seed, **kwargs):
        rng = np.random.default_rng(seed)
        return sarima_point[:steps] + rng.normal(0.0, 0.05, size=(n_sims, steps))

    def fake_elastic_net_paths(train_frame, steps, n_sims, seed, **kwargs):
        rng = np.random.default_rng(seed)
        return elastic_net_point[:steps] + rng.normal(0.0, 0.05, size=(n_sims, steps))

    monkeypatch.setattr(ensemble, "simulate_sarima_paths", fake_sarima_paths)
    monkeypatch.setattr(ensemble, "simulate_elastic_net_paths", fake_elastic_net_paths)
    monkeypatch.setattr(ensemble, "forecast_sarima", lambda series, steps, **kwargs: sarima_point[:steps])
    monkeypatch.setattr(
        ensemble,
        "forecast_elastic_net_direct",
        lambda train_frame, steps, seed, **kwargs: elastic_net_point[:steps],
    )

    paths = ensemble.simulate_ensemble_paths(frame, steps=2, n_sims=200, weights=weights, seed=123)
    repeat = ensemble.simulate_ensemble_paths(frame, steps=2, n_sims=200, weights=weights, seed=123)
    different = ensemble.simulate_ensemble_paths(
        frame,
        steps=2,
        n_sims=200,
        weights=weights,
        seed=456,
    )
    forecast = ensemble.forecast_ensemble(frame, steps=2, weights=weights, seed=123)

    assert paths.shape == (200, 2)
    np.testing.assert_array_equal(paths, repeat)
    assert not np.array_equal(paths, different)
    assert np.allclose(paths.mean(axis=0), forecast, atol=0.02)


def test_simulate_ensemble_paths_accepts_full_sarima_series(monkeypatch):
    frame = _synthetic_frame(n=4)
    full_sarima_series = pd.Series(
        [1.5, 1.6, 1.7, 1.8, 1.9],
        index=pd.period_range("2017Q4", periods=5, freq="Q"),
        name="cpi_yoy",
    )
    calls = {}

    def fake_sarima_paths(series, steps, n_sims, seed, **kwargs):
        calls["sarima_start"] = str(series.index[0])
        calls["sarima_n"] = len(series)
        return np.ones((n_sims, steps))

    def fake_elastic_net_paths(train_frame, steps, n_sims, seed, **kwargs):
        calls["elastic_net_start"] = str(train_frame.index[0])
        calls["elastic_net_n"] = len(train_frame)
        return np.ones((n_sims, steps)) * 3.0

    monkeypatch.setattr(ensemble, "simulate_sarima_paths", fake_sarima_paths)
    monkeypatch.setattr(ensemble, "simulate_elastic_net_paths", fake_elastic_net_paths)

    ensemble.simulate_ensemble_paths(
        frame,
        steps=1,
        n_sims=2,
        weights=(0.5, 0.5),
        seed=123,
        sarima_series=full_sarima_series,
    )

    assert calls == {
        "sarima_start": "2017Q4",
        "sarima_n": 5,
        "elastic_net_start": "2018Q1",
        "elastic_net_n": 4,
    }


def test_run_ensemble_comparison_smoke_on_tiny_synthetic_frame(tmp_path, monkeypatch):
    frame = _synthetic_frame(n=16)
    curated_path = tmp_path / "curated.csv"
    output_path = tmp_path / "reports/model_comparison_ensemble.csv"
    _write_curated_csv(curated_path, frame)
    logged = {}

    monkeypatch.setattr(
        ensemble,
        "forecast_sarima",
        lambda series, steps, **kwargs: np.linspace(
            float(series.iloc[-1]),
            float(series.iloc[-1]) + 0.1,
            steps,
        ),
    )
    monkeypatch.setattr(
        ensemble,
        "forecast_elastic_net_direct",
        lambda train_frame, steps, seed, **kwargs: np.linspace(
            float(train_frame["cpi_yoy"].iloc[-1]) + 0.2,
            float(train_frame["cpi_yoy"].iloc[-1]) + 0.3,
            steps,
        ),
    )

    def fake_log_model_run(**kwargs):
        logged.update(kwargs)
        return "run-id"

    from src.models import tracking

    monkeypatch.setattr(tracking, "log_model_run", fake_log_model_run)

    result = ensemble.run_ensemble_comparison(
        curated_path=curated_path,
        rba_path=tmp_path / "missing_rba.csv",
        comparison_output_path=output_path,
        initial_train_size=10,
        horizons=(1, 2),
        weights=(0.5, 0.5),
        seed=123,
        max_origins=2,
    )

    assert not result.empty
    assert not result.loc[result["model"].eq("ensemble")].empty
    assert output_path.exists()
    assert logged["run_name"] == "ensemble_comparison"
    assert logged["model_name"] == "ensemble"
    assert logged["tags"]["model_family"] == "ensemble"
    assert logged["tags"]["component_families"] == "sarima;elastic_net"
    assert callable(logged["model_logger"])


def test_run_trimmed_mean_ensemble_comparison_uses_trimmed_logging_config(monkeypatch):
    calls = {}

    def fake_run_ensemble_comparison(**kwargs):
        calls.update(kwargs)
        return pd.DataFrame({"model": ["ensemble"], "horizon": ["overall"]})

    monkeypatch.setattr(ensemble, "run_ensemble_comparison", fake_run_ensemble_comparison)

    result = ensemble.run_trimmed_mean_ensemble_comparison(verbose=True)

    assert not result.empty
    assert calls["target_column"] == "trimmed_mean_cpi_yoy"
    assert calls["elastic_net_feature_columns"][0] == "trimmed_mean_cpi_yoy_lag1"
    assert calls["include_rba"] is False
    assert calls["weights"] == ensemble.DEFAULT_WEIGHTS
    assert calls["run_name"] == "trimmed_mean_ensemble_comparison"
    assert calls["model_family_tag"] == "trimmed_mean_ensemble"
