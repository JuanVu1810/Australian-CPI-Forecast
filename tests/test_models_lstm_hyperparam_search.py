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

    rmse, mae, origin_n = search.evaluate_configuration(
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
    assert mae == pytest.approx(0.5)


def test_bounded_coordinate_search_converges_to_the_synthetic_minimum(monkeypatch):
    # Synthetic RMSE surface minimised at units=32, dropout=0.3, lookback=12 --
    # all at the edge of their candidate ranges, so a coordinate search only
    # finds it if each stage correctly keeps the running-best config.
    # learning_rate is intentionally left out of the surface (flat/tied), so
    # it should never win the search away from the default.
    def fake_evaluate_configuration(series, exog, config, **kwargs):
        rmse = (
            abs(config.units - 32) * 0.01
            + abs(config.dropout - 0.3) * 1.0
            + abs(config.lookback - 12) * 0.01
        )
        return rmse, rmse, 5

    monkeypatch.setattr(search, "evaluate_configuration", fake_evaluate_configuration)

    trials, best = search.bounded_coordinate_search(
        series=pd.Series(dtype=float),
        exog=pd.DataFrame(),
    )

    assert best == search.LSTMConfig(units=32, dropout=0.3, lookback=12, learning_rate=1e-3)
    # baseline + 3 new units + 4 new dropout + 3 new lookback + 3 new learning_rate
    # (one candidate in each of the 4 candidate lists duplicates the running-best value)
    assert len(trials) == 1 + 3 + 4 + 3 + 3


def test_evaluate_across_seeds_reports_seed_variability_and_pools_predictions(monkeypatch):
    index = pd.period_range("2015Q1", periods=20, freq="Q")
    series = pd.Series(np.full(20, 2.0), index=index, name="cpi_yoy")
    exog = pd.DataFrame({"x_lag1": np.linspace(0.0, 1.0, 20)}, index=index)
    seeds = (1, 2, 3)

    def fake_forecast_lstm_direct(train_frame, steps, seed, **kwargs):
        # Forecast error grows with seed, so seeds are distinguishable.
        offset = float(seed)
        return np.repeat(float(train_frame["cpi_yoy"].iloc[-1]) + offset, steps)

    monkeypatch.setattr(search, "forecast_lstm_direct", fake_forecast_lstm_direct)

    per_seed, pooled = search.evaluate_across_seeds(
        series=series,
        exog=exog,
        config=search.LSTMConfig(),
        seeds=seeds,
        initial_train_size=10,
        horizons=(1,),
        skip_origins=0,
    )

    assert per_seed["seed"].tolist() == list(seeds)
    np.testing.assert_allclose(per_seed["rmse"], [1.0, 2.0, 3.0])
    assert set(pooled["seed"]) == set(seeds)
    assert (pooled["model"] == "lstm").all()
    assert len(pooled) == len(seeds) * per_seed["origin_n"].iloc[0]
