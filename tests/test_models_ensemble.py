import numpy as np
import pandas as pd
import pytest

from src.models import ensemble
from src.models.elastic_net import ELASTIC_NET_FEATURE_COLUMNS


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


def test_combine_paths_weighted_sum_and_validation():
    sarima_paths = np.array([[1.0, 2.0], [3.0, 4.0]])
    elastic_net_paths = np.array([[10.0, 20.0], [30.0, 40.0]])

    combined = ensemble.combine_paths(sarima_paths, elastic_net_paths, weights=(0.4, 0.6))

    np.testing.assert_array_equal(combined, np.array([[6.4, 12.8], [19.2, 25.6]]))
    with pytest.raises(ValueError, match="same shape"):
        ensemble.combine_paths(sarima_paths, elastic_net_paths[:, :1])
    with pytest.raises(ValueError, match="sum to 1"):
        ensemble.combine_paths(sarima_paths, elastic_net_paths, weights=(0.8, 0.8))


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
        lambda series, steps: sarima_point[:steps],
    )
    monkeypatch.setattr(
        ensemble,
        "forecast_elastic_net_direct",
        lambda train_frame, steps, seed: elastic_net_point[:steps],
    )

    forecast = ensemble.forecast_ensemble(frame, steps=3, weights=weights, seed=123)
    expected = ensemble.combine_point_forecasts(
        ensemble.forecast_sarima(frame["cpi_yoy"], steps=3),
        ensemble.forecast_elastic_net_direct(frame, steps=3, seed=123),
        weights=weights,
    )

    np.testing.assert_array_equal(forecast, expected)


def test_simulate_ensemble_paths_shape_seed_divergence_and_forecast_mean(monkeypatch):
    frame = _synthetic_frame()
    weights = (0.25, 0.75)
    sarima_point = np.array([2.0, 3.0])
    elastic_net_point = np.array([6.0, 7.0])

    def fake_sarima_paths(series, steps, n_sims, seed):
        rng = np.random.default_rng(seed)
        return sarima_point[:steps] + rng.normal(0.0, 0.05, size=(n_sims, steps))

    def fake_elastic_net_paths(train_frame, steps, n_sims, seed):
        rng = np.random.default_rng(seed)
        return elastic_net_point[:steps] + rng.normal(0.0, 0.05, size=(n_sims, steps))

    monkeypatch.setattr(ensemble, "simulate_sarima_paths", fake_sarima_paths)
    monkeypatch.setattr(ensemble, "simulate_elastic_net_paths", fake_elastic_net_paths)
    monkeypatch.setattr(ensemble, "forecast_sarima", lambda series, steps: sarima_point[:steps])
    monkeypatch.setattr(
        ensemble,
        "forecast_elastic_net_direct",
        lambda train_frame, steps, seed: elastic_net_point[:steps],
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


def test_run_ensemble_comparison_smoke_on_tiny_synthetic_frame(tmp_path, monkeypatch):
    frame = _synthetic_frame(n=16)
    curated_path = tmp_path / "curated.csv"
    output_path = tmp_path / "reports/model_comparison_ensemble.csv"
    _write_curated_csv(curated_path, frame)
    logged = {}

    monkeypatch.setattr(
        ensemble,
        "forecast_sarima",
        lambda series, steps: np.linspace(float(series.iloc[-1]), float(series.iloc[-1]) + 0.1, steps),
    )
    monkeypatch.setattr(
        ensemble,
        "forecast_elastic_net_direct",
        lambda train_frame, steps, seed: np.linspace(
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
