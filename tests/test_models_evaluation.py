import numpy as np
import pandas as pd

from src.models import model_comparison
from src.models.evaluation import (
    align_rba_forecasts_to_grid,
    compute_baseline_predictions,
    compute_conformal_scale_factors,
    compute_interval_coverage_table,
    compute_metric_table,
    seasonal_naive_forecast,
    walk_forward_backtest,
    walk_forward_backtest_direct_multihorizon,
    walk_forward_interval_coverage_backtest,
)
from src.models.interval_coverage import apply_interval_calibration


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


def test_compute_baseline_predictions_can_exclude_existing_rba_file(tmp_path):
    series = pd.Series(
        np.arange(1, 13, dtype=float),
        index=pd.period_range("2020Q1", periods=12, freq="Q"),
        name="cpi_yoy",
    )
    rba_path = tmp_path / "rba.csv"
    pd.DataFrame(
        {
            "forecast_date": ["2021-03-31"],
            "horizon_quarters": [1],
            "rba_forecast_cpi_yoy": [4.0],
        }
    ).to_csv(rba_path, index=False)

    baselines = compute_baseline_predictions(
        series=series,
        rba_path=rba_path,
        initial_train_size=5,
        horizons=(1, 2),
        include_rba=False,
    )

    assert set(baselines["model"]) == {"seasonal_naive"}
    assert len(baselines) == 12


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


def test_walk_forward_interval_coverage_records_hits_and_widths_deterministically():
    series = pd.Series(
        np.arange(1, 9, dtype=float),
        index=pd.period_range("2020Q1", periods=8, freq="Q"),
        name="cpi_yoy",
    )
    calls = []

    def simulator(train, steps, n_sims, seed):
        calls.append((len(train), train.index[-1], steps, n_sims, seed))
        latest = float(train.iloc[-1])
        columns = []
        for horizon in range(1, steps + 1):
            actual = latest + horizon
            if horizon == 1:
                columns.append(np.linspace(actual - 1.0, actual + 1.0, n_sims))
            else:
                columns.append(np.linspace(actual + 10.0, actual + 12.0, n_sims))
        return np.column_stack(columns)

    result = walk_forward_interval_coverage_backtest(
        series=series,
        simulate_func=simulator,
        initial_train_size=4,
        horizons=(1, 2),
        model_name="synthetic",
        lower_quantile=0.0,
        upper_quantile=1.0,
        n_sims=5,
        seed=99,
    )

    assert calls == [
        (4, pd.Period("2020Q4", freq="Q"), 2, 5, 99),
        (5, pd.Period("2021Q1", freq="Q"), 2, 5, 100),
        (6, pd.Period("2021Q2", freq="Q"), 2, 5, 101),
    ]
    assert result["horizon"].tolist() == [1, 2, 1, 2, 1, 2]
    assert result["hit"].tolist() == [True, False, True, False, True, False]
    np.testing.assert_allclose(result["interval_width"], np.repeat(2.0, 6))
    np.testing.assert_allclose(result["point_forecast_proxy"], [5.0, 17.0, 6.0, 18.0, 7.0, 19.0])
    np.testing.assert_allclose(result["nonconformity_score"], [0.0, 11.0, 0.0, 11.0, 0.0, 11.0])
    assert result["forecast_origin"].iloc[0] == pd.Period("2020Q4", freq="Q")
    assert result["target_quarter"].iloc[1] == pd.Period("2021Q2", freq="Q")


def test_compute_conformal_scale_factors_returns_target_quantile_by_model_horizon():
    predictions = pd.DataFrame(
        {
            "model": ["a", "a", "a", "a", "b", "b"],
            "horizon": [1, 1, 1, 2, 1, 1],
            "nonconformity_score": [0.5, 1.0, 1.5, 3.0, 2.0, 4.0],
        }
    )

    factors = compute_conformal_scale_factors(predictions, target_coverage=0.8)

    a_h1 = factors.loc[(factors["model"].eq("a")) & (factors["horizon"].eq(1))].iloc[0]
    assert a_h1["n"] == 3
    assert a_h1["target_coverage"] == 0.8
    assert a_h1["scale_factor"] == 1.3

    a_h2 = factors.loc[(factors["model"].eq("a")) & (factors["horizon"].eq(2))].iloc[0]
    assert a_h2["scale_factor"] == 3.0

    b_h1 = factors.loc[(factors["model"].eq("b")) & (factors["horizon"].eq(1))].iloc[0]
    assert b_h1["scale_factor"] == 3.6


def test_compute_conformal_scale_factors_floors_scale_at_one():
    predictions = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "horizon": [1, 1, 1, 1],
            "nonconformity_score": [0.1, 0.2, 1.2, 1.4],
        }
    )

    factors = compute_conformal_scale_factors(predictions, target_coverage=0.8)

    assert factors["scale_factor"].ge(1.0).all()
    a_h1 = factors.loc[(factors["model"].eq("a")) & (factors["horizon"].eq(1))].iloc[0]
    assert a_h1["scale_factor"] == 1.0
    b_h1 = factors.loc[(factors["model"].eq("b")) & (factors["horizon"].eq(1))].iloc[0]
    assert np.isclose(b_h1["scale_factor"], 1.36)


def test_apply_interval_calibration_preserves_asymmetric_bounds_at_scale_one():
    predictions = pd.DataFrame(
        {
            "model": ["synthetic"],
            "horizon": [1],
            "actual": [11.0],
            "point_forecast_proxy": [10.0],
            "interval_lower": [8.0],
            "interval_upper": [13.0],
            "hit": [True],
            "interval_width": [5.0],
        }
    )
    factors = pd.DataFrame(
        {
            "model": ["synthetic"],
            "horizon": [1],
            "scale_factor": [1.0],
        }
    )

    calibrated = apply_interval_calibration(predictions, factors)

    assert calibrated["interval_lower"].iloc[0] == 8.0
    assert calibrated["interval_upper"].iloc[0] == 13.0
    assert calibrated["interval_width"].iloc[0] == 5.0
    assert calibrated["hit"].iloc[0]


def test_compute_interval_coverage_table_returns_overall_and_horizon_rows():
    predictions = pd.DataFrame(
        {
            "model": ["a", "a", "a", "a", "b", "b"],
            "horizon": [1, 1, 2, 2, 1, 2],
            "hit": [True, False, False, False, True, True],
            "interval_width": [1.0, 3.0, 5.0, 7.0, 2.0, 4.0],
        }
    )

    table = compute_interval_coverage_table(predictions)

    a_overall = table.loc[(table["model"] == "a") & (table["horizon"] == "overall")].iloc[0]
    assert a_overall["n"] == 4
    assert a_overall["nominal_coverage"] == 0.8
    assert a_overall["empirical_coverage"] == 0.25
    assert a_overall["mean_interval_width"] == 4.0

    a_h1 = table.loc[(table["model"] == "a") & (table["horizon"] == 1)].iloc[0]
    assert a_h1["empirical_coverage"] == 0.5
    assert a_h1["mean_interval_width"] == 2.0
    assert table["horizon"].tolist() == ["overall", 1, 2, "overall", 1, 2]
    assert {
        "coverage_ci_lower",
        "coverage_ci_upper",
        "binom_p_value",
        "significantly_miscalibrated",
    }.issubset(table.columns)

    exact_case = pd.DataFrame(
        {
            "model": ["c", "c", "c", "c"],
            "horizon": [1, 1, 1, 1],
            "hit": [False, False, False, False],
            "interval_width": [1.0, 1.0, 1.0, 1.0],
        }
    )
    exact_table = compute_interval_coverage_table(
        exact_case,
        lower_quantile=0.25,
        upper_quantile=0.75,
    )
    exact_overall = exact_table.loc[exact_table["horizon"].eq("overall")].iloc[0]
    expected_upper = 1.0 - (0.05 / 2.0) ** (1.0 / 4.0)
    assert exact_overall["nominal_coverage"] == 0.5
    assert exact_overall["empirical_coverage"] == 0.0
    assert exact_overall["coverage_ci_lower"] == 0.0
    assert round(exact_overall["coverage_ci_upper"], 12) == round(expected_upper, 12)
    assert exact_overall["binom_p_value"] == 0.125
    assert not exact_overall["significantly_miscalibrated"]


def test_joined_model_comparison_assigns_best_model_on_shared_grid():
    sarima = _comparison_prediction_frame("sarima", {1: 1.0, 2: 2.0})
    elastic_net = _comparison_prediction_frame("elastic_net", {1: 0.5, 2: 1.0})
    ensemble = _comparison_prediction_frame("ensemble", {1: 0.25, 2: 1.5})
    seasonal_naive = _comparison_prediction_frame("seasonal_naive", {1: 3.0, 2: 4.0})

    table = model_comparison.build_joined_metric_table(
        wide_prediction_frames=[sarima, elastic_net, ensemble, seasonal_naive],
        horizons=(1, 2),
    )

    h1_rows = table.loc[table["horizon"].eq(1)]
    assert set(h1_rows["best_model"]) == {"ensemble"}
    h2_rows = table.loc[table["horizon"].eq(2)]
    assert set(h2_rows["best_model"]) == {"elastic_net"}
    overall_rows = table.loc[table["horizon"].eq("overall")]
    assert set(overall_rows["best_model"]) == {"elastic_net"}


def test_build_backtest_predictions_table_stacks_all_models_with_extras_preserved():
    sarima = _comparison_prediction_frame("sarima", {1: 1.0, 2: 2.0})
    elastic_net = _comparison_prediction_frame("elastic_net", {1: 0.5})
    rba = _comparison_prediction_frame("rba", {1: 0.2})
    rba["rba_actual"] = rba["actual"]
    rba["rba_reported_error"] = rba["error"]
    other_model = _comparison_prediction_frame("other_model", {1: 0.1})
    other_model["horizon_cap"] = 1

    table = model_comparison.build_backtest_predictions_table(
        [sarima, elastic_net, rba, other_model]
    )

    assert list(table.columns) == model_comparison.PREDICTIONS_COLUMNS
    assert len(table) == len(sarima) + len(elastic_net) + len(rba) + len(other_model)
    # Rows are ordered by MODEL_ORDER (sarima, elastic_net, ..., rba, then
    # anything unlisted, like "other_model", sorted last).
    assert list(table["model"].unique()) == ["sarima", "elastic_net", "rba", "other_model"]

    elastic_net_rows = table.loc[table["model"].eq("elastic_net")]
    assert elastic_net_rows["rba_actual"].isna().all()
    assert elastic_net_rows["horizon_cap"].isna().all()

    rba_rows = table.loc[table["model"].eq("rba")]
    assert (rba_rows["rba_actual"] == rba_rows["actual"]).all()

    other_model_rows = table.loc[table["model"].eq("other_model")]
    assert (other_model_rows["horizon_cap"] == 1).all()


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
