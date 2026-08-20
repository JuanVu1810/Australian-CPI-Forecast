import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import sarimax_order_search as order_search
from src.models.evaluation import walk_forward_backtest_with_exog
from src.models.sarimax import forecast_sarimax, simulate_sarimax_paths
from src.models.sarimax_order_search import run_sarimax_order_search


def test_forecast_sarimax_returns_requested_number_of_forecasts():
    index = pd.period_range("2015Q1", periods=28, freq="Q")
    trend = np.linspace(2.0, 4.0, len(index))
    signal = np.linspace(0.0, 1.0, len(index))
    series = pd.Series(trend + 0.2 * signal, index=index, name="cpi_yoy")
    exog = pd.DataFrame({"signal_lag1": signal}, index=index)

    forecast = forecast_sarimax(
        series.iloc[:-2],
        exog.iloc[:-2],
        exog.iloc[-2:],
        steps=2,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        maxiter=25,
    )

    assert len(forecast) == 2
    assert forecast.notna().all()


def test_simulate_sarimax_paths_shape_seed_and_forecast_mean():
    index = pd.period_range("2015Q1", periods=28, freq="Q")
    trend = np.linspace(2.0, 4.0, len(index))
    signal = np.linspace(0.0, 1.0, len(index))
    series = pd.Series(trend + 0.2 * signal, index=index, name="cpi_yoy")
    exog = pd.DataFrame({"signal_lag1": signal}, index=index)

    train_y = series.iloc[:-3]
    train_x = exog.iloc[:-3]
    future_x = exog.iloc[-3:]
    kwargs = {
        "steps": 3,
        "n_sims": 80,
        "order": (1, 0, 0),
        "seasonal_order": (0, 0, 0, 0),
        "maxiter": 25,
    }
    paths = simulate_sarimax_paths(train_y, train_x, future_x, seed=123, **kwargs)
    repeat = simulate_sarimax_paths(train_y, train_x, future_x, seed=123, **kwargs)
    different = simulate_sarimax_paths(train_y, train_x, future_x, seed=456, **kwargs)
    forecast = forecast_sarimax(
        train_y,
        train_x,
        future_x,
        steps=3,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        maxiter=25,
    )

    assert paths.shape == (80, 3)
    np.testing.assert_array_equal(paths, repeat)
    assert not np.array_equal(paths, different)
    assert np.allclose(paths.mean(axis=0), forecast.to_numpy(), atol=0.5)


def test_walk_forward_with_exog_aligns_future_exog_and_caps_lag1_horizons():
    index = pd.period_range("2020Q1", periods=10, freq="Q")
    series = pd.Series(np.arange(10, dtype=float), index=index, name="cpi_yoy")
    exog = pd.DataFrame(
        {
            "signal_lag1": np.arange(100, 110, dtype=float),
        },
        index=index,
    )
    calls = []

    def recorder(train_y, train_x, future_x, steps):
        calls.append(
            {
                "last_train_quarter": train_y.index[-1],
                "train_exog_columns": tuple(train_x.columns),
                "future_index": tuple(future_x.index),
                "future_values": tuple(future_x["signal_lag1"]),
                "steps": steps,
            }
        )
        return future_x["signal_lag1"].to_numpy()

    result = walk_forward_backtest_with_exog(
        series=series,
        exog=exog,
        forecast_func=recorder,
        initial_train_size=4,
        horizons=(1, 2, 3),
        model_name="synthetic_sarimax",
    )

    assert set(result["horizon"]) == {1}
    assert result["horizon"].max() == 1
    assert result["horizon_cap"].unique().tolist() == [1]
    assert calls[0] == {
        "last_train_quarter": pd.Period("2020Q4", freq="Q"),
        "train_exog_columns": ("signal_lag1",),
        "future_index": (pd.Period("2021Q1", freq="Q"),),
        "future_values": (104.0,),
        "steps": 1,
    }


def test_sarimax_order_search_is_aic_bic_only_for_small_grid():
    index = pd.period_range("2014Q1", periods=28, freq="Q")
    signal = np.sin(np.arange(len(index)) / 3)
    series = pd.Series(2.0 + 0.4 * signal, index=index, name="cpi_yoy")
    exog = pd.DataFrame({"signal_lag1": signal}, index=index)

    result = run_sarimax_order_search(
        series=series,
        exog=exog,
        max_p=1,
        max_q=0,
        max_p_seasonal=0,
        max_q_seasonal=0,
        seasonal_period=4,
        maxiter=25,
    )

    assert list(result.columns) == [
        "order",
        "seasonal_order",
        "trend",
        "converged",
        "aic",
        "bic",
        "error",
    ]
    assert len(result) == 2
    assert result["aic"].notna().all()
    assert result["bic"].notna().all()


def test_sarimax_comparison_defaults_to_conservative_arma_grid():
    signature = inspect.signature(order_search.run_sarimax_comparison)

    assert signature.parameters["max_p"].default == 1
    assert signature.parameters["max_q"].default == 1
    assert signature.parameters["max_p_seasonal"].default == 1
    assert signature.parameters["max_q_seasonal"].default == 1


def test_level_change_resolution_compares_one_representative_lag_per_side(monkeypatch):
    index = pd.period_range("2020Q1", periods=8, freq="Q")
    series = pd.Series(np.arange(8, dtype=float), index=index, name="cpi_yoy")
    exog = pd.DataFrame(
        {
            "cash_rate_lag2": np.arange(8, dtype=float),
            "cash_rate_change_lag1": np.arange(8, dtype=float),
            "unemployment_rate_lag2": np.arange(8, dtype=float),
            "unemployment_rate_change_lag1": np.arange(8, dtype=float),
        },
        index=index,
    )
    calls = []

    class FakeFit:
        def __init__(self, aic):
            self.aic = aic

    def fake_fit_sarimax(series, exog, **kwargs):
        columns = tuple(exog.columns)
        calls.append(columns)
        return FakeFit(1.0 if columns[0].endswith("change_lag1") else 2.0)

    monkeypatch.setattr(order_search, "fit_sarimax", fake_fit_sarimax)

    choices = order_search.resolve_level_change_features(series, exog)

    assert calls == [
        ("cash_rate_lag2",),
        ("cash_rate_change_lag1",),
        ("unemployment_rate_lag2",),
        ("unemployment_rate_change_lag1",),
    ]
    assert choices["cash_rate"].winner_features == ("cash_rate_change_lag1",)
    assert choices["unemployment_rate"].winner_features == (
        "unemployment_rate_change_lag1",
    )
    assert choices["cash_rate"].aic_winner == "change"
    assert "AIC selected change" in choices["cash_rate"].selection_note


def test_level_change_resolution_defaults_to_stationary_change_when_level_aic_edge_is_small(
    monkeypatch,
):
    index = pd.period_range("2020Q1", periods=8, freq="Q")
    series = pd.Series(np.arange(8, dtype=float), index=index, name="cpi_yoy")
    exog = pd.DataFrame(
        {
            "cash_rate_lag2": np.arange(8, dtype=float),
            "cash_rate_change_lag1": np.arange(8, dtype=float),
            "unemployment_rate_lag2": np.arange(8, dtype=float),
            "unemployment_rate_change_lag1": np.arange(8, dtype=float),
        },
        index=index,
    )
    aic_by_feature = {
        "cash_rate_lag2": 10.0,
        "cash_rate_change_lag1": 10.8,
        "unemployment_rate_lag2": 20.0,
        "unemployment_rate_change_lag1": 21.5,
    }

    class FakeFit:
        def __init__(self, aic):
            self.aic = aic

    def fake_fit_sarimax(series, exog, **kwargs):
        return FakeFit(aic_by_feature[exog.columns[0]])

    monkeypatch.setattr(order_search, "fit_sarimax", fake_fit_sarimax)

    choices = order_search.resolve_level_change_features(series, exog)

    assert choices["cash_rate"].aic_winner == "level"
    assert choices["cash_rate"].aic_gap_abs == pytest.approx(0.8)
    assert choices["cash_rate"].winner == "change"
    assert choices["cash_rate"].winner_features == ("cash_rate_change_lag1",)
    assert "AIC was inconclusive" in choices["cash_rate"].selection_note
    assert "I(1)" in choices["cash_rate"].selection_note

    assert choices["unemployment_rate"].aic_winner == "level"
    assert choices["unemployment_rate"].aic_gap_abs == pytest.approx(1.5)
    assert choices["unemployment_rate"].winner == "change"
    assert choices["unemployment_rate"].winner_features == (
        "unemployment_rate_change_lag1",
    )


def test_build_feature_groups_includes_covid_intervention_variants(monkeypatch):
    class FakeChoice:
        def __init__(self, winner_features):
            self.winner_features = winner_features

    choices = {
        "cash_rate": FakeChoice(("cash_rate_change_lag1",)),
        "unemployment_rate": FakeChoice(("unemployment_rate_change_lag1",)),
    }

    groups = order_search.build_feature_groups(choices)
    by_id = {group.group_id: group for group in groups}

    assert by_id["I"].features == (
        "covid_shock_down_lag0",
        "covid_shock_rebound_lag1",
    )
    assert by_id["J"].features == (
        *by_id["D"].features,
        "covid_shock_down_lag0",
        "covid_shock_rebound_lag1",
    )
    assert "I" not in order_search.NESTED_GROUP_IDS
    assert "J" not in order_search.NESTED_GROUP_IDS


def test_expected_sign_for_covid_intervention_features():
    assert order_search.expected_sign_for_feature("covid_shock_down_lag0") == "-"
    assert order_search.expected_sign_for_feature("covid_shock_rebound_lag1") == "+"


def test_intervention_groups_have_zero_horizon_cap_and_skip_walk_forward():
    horizon_cap = order_search.infer_min_lag_from_columns(order_search.INTERVENTION_FEATURES)
    assert horizon_cap == 0


def test_not_forecast_viable_metrics_has_no_crash_shape():
    group = order_search.FeatureGroup("I", "COVID intervention dummies only", order_search.INTERVENTION_FEATURES)

    metrics = order_search._not_forecast_viable_metrics(
        group=group,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 4),
        criterion="aic",
        selection_aic=100.0,
        selection_bic=110.0,
        level_change_note="",
        horizon_cap=0,
    )

    assert metrics["model"].tolist() == ["sarimax"]
    assert metrics["horizon"].tolist() == ["overall"]
    assert metrics["n"].iloc[0] == 0
    assert pd.isna(metrics["rmse"].iloc[0])
    assert metrics["group_id"].iloc[0] == "I"


def test_sarimax_selection_functions_do_not_reference_walk_forward_helpers():
    path = Path("src/models/sarimax_order_search.py")
    tree = ast.parse(path.read_text())
    forbidden = {"walk_forward_backtest", "walk_forward_backtest_with_exog"}
    selection_functions = {
        "run_sarimax_order_search",
        "resolve_level_change_features",
        "build_feature_groups",
        "_select_order",
    }

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)

    assert forbidden.isdisjoint(imported_names)

    function_bodies = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for function_name in selection_functions:
        body = function_bodies[function_name]
        for node in ast.walk(body):
            if isinstance(node, ast.Name):
                assert node.id not in forbidden
                assert node.id != "model_evaluation"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in forbidden
