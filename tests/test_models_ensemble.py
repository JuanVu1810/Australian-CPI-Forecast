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


def test_combine_point_forecasts_is_a_plain_average_and_validates_shape():
    sarima_forecast = np.array([1.0, 2.0, 3.0])
    elastic_net_forecast = np.array([5.0, 6.0, 7.0])

    combined = ensemble.combine_point_forecasts(sarima_forecast, elastic_net_forecast)

    np.testing.assert_array_equal(combined, np.array([3.0, 4.0, 5.0]))
    with pytest.raises(ValueError, match="same length"):
        ensemble.combine_point_forecasts(sarima_forecast, elastic_net_forecast[:2])


def test_combine_paths_is_a_plain_average_and_validates_shape():
    sarima_paths = np.array([[1.0, 2.0], [3.0, 4.0]])
    elastic_net_paths = np.array([[10.0, 20.0], [30.0, 40.0]])

    combined = ensemble.combine_paths(sarima_paths, elastic_net_paths)

    np.testing.assert_array_equal(combined, np.array([[5.5, 11.0], [16.5, 22.0]]))
    with pytest.raises(ValueError, match="same shape"):
        ensemble.combine_paths(sarima_paths, elastic_net_paths[:, :1])


def test_recenter_paths_to_median_shifts_without_changing_spread():
    # Deliberately skewed pool: median (2.0) and mean (2.33) differ, exercising
    # exactly the case recentering is meant to fix.
    paths = np.array([[1.0, 10.0], [2.0, 20.0], [2.0, 20.0], [4.0, 40.0]])
    target_median = np.array([5.0, 50.0])

    recentered = ensemble.recenter_paths_to_median(paths, target_median)

    np.testing.assert_allclose(np.percentile(recentered, 50, axis=0), target_median)
    # A pure additive shift preserves spread exactly.
    np.testing.assert_allclose(recentered.std(axis=0), paths.std(axis=0))
    with pytest.raises(ValueError, match="two-dimensional"):
        ensemble.recenter_paths_to_median(paths[:, 0], target_median)
    with pytest.raises(ValueError, match="one value per step"):
        ensemble.recenter_paths_to_median(paths, target_median[:1])


def test_ensemble_interval_from_paths_uses_combined_draw_quantiles():
    paths = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    lower, upper = ensemble.ensemble_interval_from_paths(paths, lower=0.0, upper=1.0)

    np.testing.assert_array_equal(lower, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(upper, np.array([5.0, 6.0]))


def test_forecast_ensemble_averages_the_two_component_forecasts(monkeypatch):
    frame = _synthetic_frame()
    monkeypatch.setattr(
        ensemble,
        "forecast_sarima",
        lambda series, steps, **kwargs: np.array([10.0, 20.0, 30.0])[:steps],
    )
    monkeypatch.setattr(
        ensemble,
        "forecast_elastic_net_direct",
        lambda train_frame, steps, seed, **kwargs: np.array([0.0, 100.0, 200.0])[:steps],
    )

    forecast = ensemble.forecast_ensemble(frame, steps=3, seed=123)

    np.testing.assert_array_equal(forecast, np.array([5.0, 60.0, 115.0]))


def test_forecast_ensemble_uses_the_headline_configuration(monkeypatch):
    frame = _synthetic_frame()
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

    forecast = ensemble.forecast_ensemble(frame, steps=2)

    np.testing.assert_array_equal(forecast, np.array([2.0, 3.0]))
    assert calls == {
        "sarima_name": "cpi_yoy",
        "sarima_order": ensemble.SARIMA_DEFAULT_ORDER,
        "sarima_seasonal_order": ensemble.SARIMA_DEFAULT_SEASONAL_ORDER,
        "elastic_target": "cpi_yoy",
        "elastic_features": ELASTIC_NET_FEATURE_COLUMNS,
    }


def test_simulate_ensemble_paths_shape_seed_divergence_and_forecast_mean(monkeypatch):
    frame = _synthetic_frame()
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

    paths = ensemble.simulate_ensemble_paths(frame, steps=2, n_sims=200, seed=123)
    repeat = ensemble.simulate_ensemble_paths(frame, steps=2, n_sims=200, seed=123)
    different = ensemble.simulate_ensemble_paths(
        frame,
        steps=2,
        n_sims=200,
        seed=456,
    )
    forecast = ensemble.forecast_ensemble(frame, steps=2, seed=123)

    assert paths.shape == (200, 2)
    np.testing.assert_array_equal(paths, repeat)
    assert not np.array_equal(paths, different)
    assert np.allclose(paths.mean(axis=0), forecast, atol=0.02)
    # Recentering guarantees this exactly (up to floating point), not just approximately.
    np.testing.assert_allclose(np.percentile(paths, 50, axis=0), forecast, atol=1e-10)


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
    # Recentering also needs the two deterministic point forecasts; mock these too
    # so this stays a fast, isolated wiring test rather than fitting real models
    # on 4-5 rows of synthetic data.
    monkeypatch.setattr(ensemble, "forecast_sarima", lambda series, steps, **kwargs: np.ones(steps))
    monkeypatch.setattr(
        ensemble,
        "forecast_elastic_net_direct",
        lambda train_frame, steps, seed, **kwargs: np.ones(steps) * 3.0,
    )

    ensemble.simulate_ensemble_paths(
        frame,
        steps=1,
        n_sims=2,
        seed=123,
        sarima_series=full_sarima_series,
    )

    assert calls == {
        "sarima_start": "2017Q4",
        "sarima_n": 5,
        "elastic_net_start": "2018Q1",
        "elastic_net_n": 4,
    }
