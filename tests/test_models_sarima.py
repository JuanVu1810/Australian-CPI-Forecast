import numpy as np
import pandas as pd

from src.models.sarima import (
    CPI_INDEX_NOTEBOOK_ORDER,
    CPI_INDEX_NOTEBOOK_SEASONAL_ORDER,
    DEFAULT_ORDER,
    DEFAULT_SEASONAL_ORDER,
    forecast_sarima,
)
from src.models.sarima_order_search import iter_candidate_orders


def test_sarima_defaults_reflect_cpi_yoy_selected_specification():
    assert CPI_INDEX_NOTEBOOK_ORDER == (0, 1, 1)
    assert CPI_INDEX_NOTEBOOK_SEASONAL_ORDER == (0, 1, 1, 4)
    assert DEFAULT_ORDER == (2, 0, 2)
    assert DEFAULT_SEASONAL_ORDER == (0, 0, 2, 4)


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
