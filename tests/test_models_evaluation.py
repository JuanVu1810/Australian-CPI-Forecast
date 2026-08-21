import numpy as np
import pandas as pd

from src.models import model_comparison
from src.models.evaluation import (
    align_rba_forecasts_to_grid,
    compute_baseline_predictions,
    compute_metric_table,
    seasonal_naive_forecast,
    walk_forward_backtest,
    walk_forward_backtest_direct_multihorizon,
)


def _comparison_prediction_frame(model: str, errors_by_horizon: dict[int, float]) -> pd.DataFrame:
    rows = []
    origins = [pd.Period("2020Q4", freq="Q"), pd.Period("2021Q1", freq="Q")]
    for origin in origins:
        for horizon, error in errors_by_horizon.items():
            target = origin + horizon
            actual = 10.0 + horizon
            forecast = actual - error
            rows.append(
                {
                    "model": model,
                    "forecast_origin": origin,
                    "target_quarter": target,
                    "horizon": horizon,
                    "actual": actual,
                    "forecast": forecast,
                    "error": error,
                }
            )
    return pd.DataFrame(rows)


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


def test_compute_baseline_predictions_returns_shared_seasonal_naive_and_rba_grid(tmp_path):
    series = pd.Series(
        np.arange(1, 13, dtype=float),
        index=pd.period_range("2020Q1", periods=12, freq="Q"),
        name="cpi_yoy",
    )
    rba_path = tmp_path / "rba.csv"
    pd.DataFrame(
        {
            "forecast_date": ["2021-03-31", "2021-03-31", "2021-06-30"],
            "horizon_quarters": [1, 2, 1],
            "rba_forecast_cpi_yoy": [4.0, 5.0, 6.0],
            "rba_actual_cpi_yoy": [6.0, 7.0, 7.0],
            "rba_forecast_error_cpi_yoy": [2.0, 2.0, 1.0],
        }
    ).to_csv(rba_path, index=False)

    baselines = compute_baseline_predictions(
        series=series,
        rba_path=rba_path,
        initial_train_size=5,
        horizons=(1, 2),
    )

    assert set(baselines["model"]) == {"seasonal_naive", "rba"}
    seasonal = baselines.loc[baselines["model"].eq("seasonal_naive")]
    assert len(seasonal) == 12
    rba = baselines.loc[baselines["model"].eq("rba")]
    assert rba["forecast_origin"].tolist() == [
        pd.Period("2021Q1", freq="Q"),
        pd.Period("2021Q1", freq="Q"),
        pd.Period("2021Q2", freq="Q"),
    ]
    assert rba["target_quarter"].tolist() == [
        pd.Period("2021Q2", freq="Q"),
        pd.Period("2021Q3", freq="Q"),
        pd.Period("2021Q3", freq="Q"),
    ]
    np.testing.assert_allclose(rba["error"], [2.0, 2.0, 1.0])


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


def test_joined_model_comparison_keeps_sarimax_on_real_horizon_only():
    sarima = _comparison_prediction_frame("sarima", {1: 1.0, 2: 2.0})
    elastic_net = _comparison_prediction_frame("elastic_net", {1: 0.5, 2: 1.0})
    ensemble = _comparison_prediction_frame("ensemble", {1: 0.25, 2: 1.5})
    seasonal_naive = _comparison_prediction_frame("seasonal_naive", {1: 3.0, 2: 4.0})
    sarimax = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "model": "sarimax",
                        "forecast_origin": pd.Period("2020Q3", freq="Q"),
                        "target_quarter": pd.Period("2020Q4", freq="Q"),
                        "horizon": 1,
                        "actual": 10.0,
                        "forecast": 9.9,
                        "error": 0.1,
                    }
                ]
            ),
            _comparison_prediction_frame("sarimax", {1: 0.1}),
        ],
        ignore_index=True,
    )

    table = model_comparison.build_joined_metric_table(
        wide_prediction_frames=[sarima, elastic_net, ensemble, seasonal_naive],
        sarimax_group_d_predictions=sarimax,
        horizons=(1, 2),
    )

    sarimax_h1 = table.loc[(table["model"].eq("sarimax")) & (table["horizon"].eq(1))].iloc[0]
    assert sarimax_h1["n"] == 2
    assert sarimax_h1["forecast_origin_n"] == 2
    assert sarimax_h1["group_id"] == "D"

    sarimax_h2 = table.loc[(table["model"].eq("sarimax")) & (table["horizon"].eq(2))].iloc[0]
    assert sarimax_h2["n"] == 0
    assert pd.isna(sarimax_h2["rmse"])
    assert sarimax_h2["evaluation_status"] == "not_evaluated_at_this_horizon"

    h2_rows = table.loc[table["horizon"].eq(2)]
    assert set(h2_rows["best_model"]) == {"elastic_net"}
    overall_rows = table.loc[table["horizon"].eq("overall")]
    assert set(overall_rows["best_model"]) == {"elastic_net"}


def test_skip_origins_produces_a_chronologically_disjoint_origin_set():
    series = pd.Series(
        np.arange(1, 26, dtype=float),
        index=pd.period_range("2015Q1", periods=25, freq="Q"),
        name="cpi_yoy",
    )
    exog = pd.DataFrame({"x_lag1": np.arange(25, dtype=float)}, index=series.index)

    def recorder(train_frame, steps):
        return np.repeat(float(train_frame["cpi_yoy"].iloc[-1]), steps)

    screen = walk_forward_backtest_direct_multihorizon(
        series=series,
        exog=exog,
        forecast_func=recorder,
        initial_train_size=10,
        horizons=(1,),
        max_origins=5,
    )
    held_out = walk_forward_backtest_direct_multihorizon(
        series=series,
        exog=exog,
        forecast_func=recorder,
        initial_train_size=10,
        horizons=(1,),
        skip_origins=5,
    )

    screen_origins = set(screen["forecast_origin"])
    held_out_origins = set(held_out["forecast_origin"])
    assert len(screen_origins) == 5
    assert screen_origins.isdisjoint(held_out_origins)
    assert max(screen_origins) < min(held_out_origins)
