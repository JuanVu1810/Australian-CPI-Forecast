import numpy as np
import pandas as pd
import pytest

from src.models import lstm_hyperparam_search as search


def test_evaluate_configuration_computes_rmse_from_forecast_errors():
    index = pd.period_range("2015Q1", periods=20, freq="Q")
    series = pd.Series(np.full(20, 2.0), index=index, name="cpi_yoy")
    exog = pd.DataFrame({"x_lag1": np.linspace(0.0, 1.0, 20)}, index=index)

    def fixed_offset_forecast(train_frame, steps):
        return np.repeat(float(train_frame["cpi_yoy"].iloc[-1]) + 0.5, steps)

    rmse, origin_n = search.evaluate_configuration(
        series,
        exog,
        search.LSTMConfig(),
        initial_train_size=10,
        horizons=(1,),
        max_origins=5,
        forecast_func=fixed_offset_forecast,
    )

    assert origin_n == 5
    assert rmse == pytest.approx(0.5)


def test_bounded_coordinate_search_converges_to_the_synthetic_minimum(monkeypatch):
    # Synthetic RMSE surface minimised at units=32, dropout=0.3, lookback=12 --
    # all at the edge of their candidate ranges, so a coordinate search only
    # finds it if each stage correctly keeps the running-best config.
    def fake_evaluate_configuration(series, exog, config, **kwargs):
        rmse = (
            abs(config.units - 32) * 0.01
            + abs(config.dropout - 0.3) * 1.0
            + abs(config.lookback - 12) * 0.01
        )
        return rmse, 5

    monkeypatch.setattr(search, "evaluate_configuration", fake_evaluate_configuration)

    trials, best = search.bounded_coordinate_search(
        series=pd.Series(dtype=float),
        exog=pd.DataFrame(),
    )

    assert best == search.LSTMConfig(units=32, dropout=0.3, lookback=12)
    assert len(trials) == 1 + 2 + 2 + 2  # baseline + 2 new units + 2 new dropout + 2 new lookback
