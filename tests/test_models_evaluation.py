import numpy as np
import pandas as pd

from src.models.evaluation import (
    align_rba_forecasts_to_grid,
    compute_metric_table,
    seasonal_naive_forecast,
    walk_forward_backtest,
)


def test_walk_forward_backtest_uses_expanding_refit_and_aligns_horizons():
    series = pd.Series(
        np.arange(1, 13, dtype=float),
        index=pd.period_range("2020Q1", periods=12, freq="Q"),
        name="cpi_yoy",
    )
    calls = []

    def recorder(train, steps):
        calls.append((len(train), train.index[-1], steps))
        return pd.Series([train.iloc[-1] + horizon for horizon in range(1, steps + 1)])

    result = walk_forward_backtest(
        series,
        recorder,
        initial_train_size=5,
        horizons=(1, 3),
        model_name="synthetic",
    )

    assert calls == [
        (5, pd.Period("2021Q1", freq="Q"), 3),
        (6, pd.Period("2021Q2", freq="Q"), 3),
        (7, pd.Period("2021Q3", freq="Q"), 3),
        (8, pd.Period("2021Q4", freq="Q"), 3),
        (9, pd.Period("2022Q1", freq="Q"), 3),
    ]
    assert result["forecast_origin"].tolist()[:2] == [
        pd.Period("2021Q1", freq="Q"),
        pd.Period("2021Q1", freq="Q"),
    ]
    assert result["target_quarter"].tolist()[:2] == [
        pd.Period("2021Q2", freq="Q"),
        pd.Period("2021Q4", freq="Q"),
    ]
    assert result["horizon"].tolist()[:4] == [1, 3, 1, 3]
    assert len(result) == 10


def test_seasonal_naive_repeats_same_quarter_year_ended_inflation():
    train = pd.Series(
        [2.0, 3.0, 4.0, 5.0, 6.0],
        index=pd.period_range("2020Q1", periods=5, freq="Q"),
    )

    forecast = seasonal_naive_forecast(train, steps=6, seasonal_period=4)

    assert forecast.index.tolist() == list(pd.period_range("2021Q2", periods=6, freq="Q"))
    assert forecast.tolist() == [3.0, 4.0, 5.0, 6.0, 3.0, 4.0]


def test_rba_alignment_joins_to_same_origin_horizon_grid_and_recomputes_error():
    grid = pd.DataFrame(
        {
            "model": ["sarima", "sarima", "sarima"],
            "forecast_origin": ["2021Q1", "2021Q1", "2021Q2"],
            "target_quarter": ["2021Q2", "2021Q3", "2021Q3"],
            "horizon": [1, 2, 1],
            "actual": [2.5, 3.0, 3.0],
            "forecast": [2.4, 2.9, 2.8],
            "error": [0.1, 0.1, 0.2],
        }
    )
    rba = pd.DataFrame(
        {
            "forecast_date": ["2021-03-01", "2021-03-01", "2021-06-01", "2020-12-01"],
            "horizon_quarters": [1, 2, 1, 1],
            "rba_forecast_cpi_yoy": [2.0, 2.7, np.nan, 9.9],
            "rba_actual_cpi_yoy": [2.6, 3.1, 3.0, 1.0],
            "rba_forecast_error_cpi_yoy": [0.6, 0.4, np.nan, -8.9],
        }
    )

    aligned = align_rba_forecasts_to_grid(rba, grid, horizons=(1, 2))

    assert aligned["model"].tolist() == ["rba", "rba"]
    assert aligned["forecast_origin"].tolist() == [
        pd.Period("2021Q1", freq="Q"),
        pd.Period("2021Q1", freq="Q"),
    ]
    assert aligned["horizon"].tolist() == [1, 2]
    assert aligned["forecast"].tolist() == [2.0, 2.7]
    np.testing.assert_allclose(aligned["error"], [0.5, 0.3])


def test_compute_metric_table_returns_overall_and_horizon_rows():
    predictions = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "horizon": [1, 2, 1, 2],
            "error": [1.0, -3.0, 2.0, -4.0],
        }
    )

    table = compute_metric_table(predictions)

    a_overall = table.loc[(table["model"] == "a") & (table["horizon"] == "overall")].iloc[0]
    assert a_overall["n"] == 2
    assert round(a_overall["rmse"], 6) == round(np.sqrt(5), 6)
    assert a_overall["mae"] == 2.0
    assert set(table["horizon"]) == {"overall", 1, 2}
