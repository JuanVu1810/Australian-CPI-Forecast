import numpy as np
import pandas as pd
import pytest

from src.models import elastic_net
from src.models.evaluation import walk_forward_backtest_direct_multihorizon


def _synthetic_frame(n: int = 40) -> pd.DataFrame:
    index = pd.period_range("2010Q1", periods=n, freq="Q")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "cpi_yoy": 2.0 + np.sin(np.arange(n) / 4) + 0.05 * rng.standard_normal(n),
            "cpi_yoy_lag1": 2.0 + np.sin((np.arange(n) - 1) / 4),
            "cpi_yoy_lag4": 2.0 + np.sin((np.arange(n) - 4) / 4),
            "cash_rate_change_lag1": 0.1 * np.cos(np.arange(n) / 5),
            "unemployment_rate_change_lag1": -0.05 * np.sin(np.arange(n) / 6),
            "inflation_expectations_business_lag1": 2.5 + 0.1 * np.cos(np.arange(n) / 3),
            "ppi_growth_lag2": 0.5 * np.sin(np.arange(n) / 7),
            "commodity_growth_lag1": 0.3 * np.cos(np.arange(n) / 2),
            "wti_growth_lag1": 0.2 * np.sin(np.arange(n) / 8),
        },
        index=index,
    )


def test_forecast_elastic_net_direct_returns_exactly_8_finite_values():
    frame = _synthetic_frame(n=40)

    forecast = elastic_net.forecast_elastic_net_direct(frame, steps=8)

    assert len(forecast) == 8
    assert np.isfinite(forecast).all()


def test_forecast_elastic_net_direct_rejects_more_than_8_steps():
    frame = _synthetic_frame(n=40)

    with pytest.raises(ValueError, match="at most 8 horizons"):
        elastic_net.forecast_elastic_net_direct(frame, steps=9)


def test_walk_forward_direct_multihorizon_smoke():
    frame = _synthetic_frame(n=50)
    series = frame["cpi_yoy"]
    exog = frame.drop(columns="cpi_yoy")

    result = walk_forward_backtest_direct_multihorizon(
        series=series,
        exog=exog,
        forecast_func=lambda train_frame, steps: elastic_net.forecast_elastic_net_direct(
            train_frame, steps=steps
        ),
        initial_train_size=40,
        horizons=(1, 2),
        model_name="elastic_net",
    )

    assert not result.empty
    assert set(result["horizon"]) == {1, 2}
    assert result["forecast"].notna().all()


def test_scaler_is_fit_on_feature_columns_only_not_target():
    frame = _synthetic_frame(n=40)

    fitted = elastic_net.fit_elastic_net_direct(frame, horizons=(1, 2))

    assert set(fitted.scaler.columns) == set(elastic_net.ELASTIC_NET_FEATURE_COLUMNS)
    assert elastic_net.TARGET_COLUMN not in fitted.scaler.columns


def test_coefficient_table_has_one_row_per_horizon_and_feature():
    frame = _synthetic_frame(n=40)
    fitted = elastic_net.fit_elastic_net_direct(frame, horizons=(1, 2, 3))

    table = elastic_net.coefficient_table(fitted)

    assert set(table["horizon"]) == {1, 2, 3}
    for horizon in (1, 2, 3):
        features = set(table.loc[table["horizon"] == horizon, "feature"])
        assert features == set(elastic_net.ELASTIC_NET_FEATURE_COLUMNS)


def test_safe_cv_splits_never_exceeds_available_examples():
    assert elastic_net._safe_cv_splits(n_examples=100, requested=4) == 4
    assert elastic_net._safe_cv_splits(n_examples=5, requested=4) == 4
    assert elastic_net._safe_cv_splits(n_examples=2, requested=4) == 2
