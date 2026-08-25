import numpy as np
import pandas as pd

from src.models.sarima import (
    CPI_INDEX_NOTEBOOK_ORDER,
    CPI_INDEX_NOTEBOOK_SEASONAL_ORDER,
    DEFAULT_ORDER,
    DEFAULT_SEASONAL_ORDER,
    TRIMMED_MEAN_DEFAULT_ORDER,
    TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER,
    fit_sarima,
    forecast_sarima,
    simulate_paths_from_fit,
    simulate_sarima_paths,
)
from src.models.sarima_order_search import iter_candidate_orders


def test_sarima_defaults_reflect_cpi_yoy_selected_specification():
    assert CPI_INDEX_NOTEBOOK_ORDER == (0, 1, 1)
    assert CPI_INDEX_NOTEBOOK_SEASONAL_ORDER == (0, 1, 1, 4)
    assert DEFAULT_ORDER == (1, 0, 2)
    assert DEFAULT_SEASONAL_ORDER == (1, 0, 2, 4)
    assert TRIMMED_MEAN_DEFAULT_ORDER == (1, 1, 1)
    assert TRIMMED_MEAN_DEFAULT_SEASONAL_ORDER == (0, 0, 1, 4)


def test_order_search_default_grid_is_non_differenced_for_cpi_yoy():
    candidates = list(
        iter_candidate_orders(
            max_p=1,
            max_q=1,
            max_p_seasonal=1,
            max_q_seasonal=1,
        )
    )

    assert len(candidates) == 16
    assert all(order[1] == 0 for order, _ in candidates)
    assert all(seasonal_order[1] == 0 for _, seasonal_order in candidates)


def test_forecast_sarima_returns_requested_number_of_forecasts():
    index = pd.period_range("2015Q1", periods=24, freq="Q")
    trend = np.linspace(2.0, 4.0, len(index))
    seasonal = np.tile([0.1, -0.1, 0.2, -0.2], 6)
    series = pd.Series(trend + seasonal, index=index, name="cpi_yoy")

    forecast = forecast_sarima(
        series,
        steps=3,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        maxiter=25,
    )

    assert len(forecast) == 3
    assert forecast.notna().all()


def test_simulate_sarima_paths_shape_seed_and_forecast_mean():
    index = pd.period_range("2015Q1", periods=24, freq="Q")
    trend = np.linspace(2.0, 4.0, len(index))
    seasonal = np.tile([0.1, -0.1, 0.2, -0.2], 6)
    series = pd.Series(trend + seasonal, index=index, name="cpi_yoy")

    kwargs = {
        "steps": 3,
        "n_sims": 80,
        "order": (1, 0, 0),
        "seasonal_order": (0, 0, 0, 0),
        "maxiter": 25,
    }
    paths = simulate_sarima_paths(series, seed=123, **kwargs)
    repeat = simulate_sarima_paths(series, seed=123, **kwargs)
    different = simulate_sarima_paths(series, seed=456, **kwargs)
    forecast = forecast_sarima(
        series,
        steps=3,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        maxiter=25,
    )

    assert paths.shape == (80, 3)
    np.testing.assert_array_equal(paths, repeat)
    assert not np.array_equal(paths, different)
    assert np.allclose(paths.mean(axis=0), forecast.to_numpy(), atol=0.5)


def test_simulate_sarima_paths_from_fit_reuses_fitted_model_without_state_leak():
    index = pd.period_range("2015Q1", periods=24, freq="Q")
    trend = np.linspace(2.0, 4.0, len(index))
    seasonal = np.tile([0.1, -0.1, 0.2, -0.2], 6)
    series = pd.Series(trend + seasonal, index=index, name="cpi_yoy")
    fitted = fit_sarima(
        series,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        maxiter=25,
    )

    paths = simulate_paths_from_fit(fitted, steps=3, n_sims=80, seed=123)
    different = simulate_paths_from_fit(fitted, steps=3, n_sims=80, seed=456)
    repeat = simulate_paths_from_fit(fitted, steps=3, n_sims=80, seed=123)

    assert not np.array_equal(paths, different)
    np.testing.assert_array_equal(paths, repeat)


def test_simulate_sarima_paths_wrapper_matches_manual_fit_simulation():
    index = pd.period_range("2015Q1", periods=24, freq="Q")
    trend = np.linspace(2.0, 4.0, len(index))
    seasonal = np.tile([0.1, -0.1, 0.2, -0.2], 6)
    series = pd.Series(trend + seasonal, index=index, name="cpi_yoy")
    kwargs = {
        "steps": 3,
        "n_sims": 80,
        "order": (1, 0, 0),
        "seasonal_order": (0, 0, 0, 0),
        "maxiter": 25,
    }

    wrapper_paths = simulate_sarima_paths(series, seed=123, **kwargs)
    fitted = fit_sarima(
        series,
        order=kwargs["order"],
        seasonal_order=kwargs["seasonal_order"],
        maxiter=kwargs["maxiter"],
    )
    manual_paths = simulate_paths_from_fit(
        fitted,
        steps=kwargs["steps"],
        n_sims=kwargs["n_sims"],
        seed=123,
    )

    np.testing.assert_array_equal(wrapper_paths, manual_paths)
