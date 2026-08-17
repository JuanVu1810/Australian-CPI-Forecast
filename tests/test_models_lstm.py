from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import walk_forward_backtest_direct_multihorizon
from src.models import lstm


def test_direct_harness_scaler_is_fit_on_origin_training_slice_only():
    index = pd.period_range("2020Q1", periods=14, freq="Q")
    series = pd.Series(np.arange(1, 15, dtype=float), index=index, name="cpi_yoy")
    exog = pd.DataFrame(
        {
            "signal_lag1": np.arange(101, 115, dtype=float),
            "other_lag1": np.linspace(2.0, 5.0, len(index)),
        },
        index=index,
    )
    captured = []

    def recorder(train_frame, steps):
        scaler = lstm.fit_train_window_scaler(train_frame)
        captured.append(
            {
                "last_train_quarter": train_frame.index[-1],
                "mean": scaler.mean_,
                "scale": scaler.scale_,
            }
        )
        return np.repeat(float(train_frame["cpi_yoy"].iloc[-1]), steps)

    walk_forward_backtest_direct_multihorizon(
        series=series,
        exog=exog,
        forecast_func=recorder,
        initial_train_size=6,
        horizons=(1, 2),
        model_name="synthetic_lstm",
    )

    first_origin = pd.Period("2021Q2", freq="Q")
    manual = pd.concat(
        [
            series.loc[:first_origin].rename("cpi_yoy"),
            exog.loc[:first_origin],
        ],
        axis=1,
    )
    assert captured[0]["last_train_quarter"] == first_origin
    pd.testing.assert_series_equal(captured[0]["mean"], manual.mean(axis=0))
    pd.testing.assert_series_equal(captured[0]["scale"], manual.std(axis=0, ddof=0))


def test_direct_harness_never_passes_post_origin_exog():
    index = pd.period_range("2020Q1", periods=12, freq="Q")
    series = pd.Series(np.arange(12, dtype=float), index=index, name="cpi_yoy")
    exog = pd.DataFrame({"signal_lag1": np.arange(12, dtype=float)}, index=index)
    exog.loc[index[5]:, "signal_lag1"] = 9999.0
    calls = []

    def recorder(train_frame, steps):
        calls.append(
            {
                "last_index": train_frame.index[-1],
                "max_signal": float(train_frame["signal_lag1"].max()),
                "steps": steps,
            }
        )
        return np.arange(steps, dtype=float)

    result = walk_forward_backtest_direct_multihorizon(
        series=series,
        exog=exog,
        forecast_func=recorder,
        initial_train_size=5,
        horizons=(1, 3),
        model_name="synthetic_lstm",
    )

    assert calls[0] == {
        "last_index": pd.Period("2021Q1", freq="Q"),
        "max_signal": 4.0,
        "steps": 3,
    }
    assert calls[1]["last_index"] == pd.Period("2021Q2", freq="Q")
    assert calls[1]["max_signal"] == 9999.0
    assert set(result["horizon"]) == {1, 3}


def test_forecast_lstm_direct_returns_exactly_8_non_nan_values(monkeypatch):
    index = pd.period_range("2018Q1", periods=24, freq="Q")
    frame = pd.DataFrame(
        {
            "cpi_yoy": np.linspace(2.0, 4.0, len(index)),
            "cash_rate_change_lag1": np.linspace(-0.2, 0.2, len(index)),
            "unemployment_rate_change_lag1": np.linspace(0.1, -0.1, len(index)),
            "inflation_expectations_business_lag1": np.linspace(2.5, 3.0, len(index)),
            "ppi_growth_lag2": np.linspace(0.0, 1.0, len(index)),
            "commodity_growth_lag1": np.linspace(-1.0, 1.0, len(index)),
            "wti_growth_lag1": np.linspace(1.0, -1.0, len(index)),
        },
        index=index,
    )

    class FakeScaler:
        columns = lstm.LSTM_INPUT_COLUMNS

        def transform(self, values):
            return np.zeros((len(values), len(self.columns)), dtype=np.float32)

        def inverse_transform_target(self, values, target_column="cpi_yoy"):
            return np.asarray(values, dtype=float) + 2.0

    class FakeModel:
        def predict(self, values, verbose=0):
            return np.arange(8, dtype=np.float32).reshape(1, 8)

    def fake_fit_lstm_direct(train_frame, seed=42, verbose=0):
        return SimpleNamespace(
            model=FakeModel(),
            scaler=FakeScaler(),
            feature_columns=lstm.LSTM_INPUT_COLUMNS,
            target_column="cpi_yoy",
            lookback=8,
            horizon=8,
        )

    monkeypatch.setattr(lstm, "fit_lstm_direct", fake_fit_lstm_direct)

    forecast = lstm.forecast_lstm_direct(frame, steps=8)

    assert len(forecast) == 8
    assert np.isfinite(forecast).all()


def test_sequence_builder_rejects_gapped_quarterly_index():
    index = pd.PeriodIndex(
        ["2018Q1", "2018Q2", "2018Q4", "2019Q1", "2019Q2", "2019Q3"],
        freq="Q",
    )
    frame = pd.DataFrame(
        {
            "cpi_yoy": np.arange(len(index), dtype=float),
            "cash_rate_change_lag1": np.arange(len(index), dtype=float),
            "unemployment_rate_change_lag1": np.arange(len(index), dtype=float),
            "inflation_expectations_business_lag1": np.arange(len(index), dtype=float),
            "ppi_growth_lag2": np.arange(len(index), dtype=float),
            "commodity_growth_lag1": np.arange(len(index), dtype=float),
            "wti_growth_lag1": np.arange(len(index), dtype=float),
        },
        index=index,
    )

    with pytest.raises(ValueError, match="consecutive quarterly"):
        lstm.clean_lstm_frame(frame)


def test_permutation_importance_no_effect_model_has_zero_rmse_increase(monkeypatch):
    index = pd.period_range("2015Q1", periods=30, freq="Q")
    frame = pd.DataFrame(
        {
            "cpi_yoy": np.linspace(1.0, 3.0, len(index)),
            "cash_rate_change_lag1": np.linspace(0.0, 1.0, len(index)),
            "unemployment_rate_change_lag1": np.linspace(1.0, 0.0, len(index)),
            "inflation_expectations_business_lag1": np.linspace(2.0, 2.5, len(index)),
            "ppi_growth_lag2": np.linspace(-1.0, 1.0, len(index)),
            "commodity_growth_lag1": np.sin(np.arange(len(index))),
            "wti_growth_lag1": np.cos(np.arange(len(index))),
        },
        index=index,
    )

    class IdentityScaler:
        columns = lstm.LSTM_INPUT_COLUMNS

        def transform(self, values):
            return pd.DataFrame(values).loc[:, list(self.columns)].to_numpy(dtype=np.float32)

        def inverse_transform_target(self, values, target_column="cpi_yoy"):
            return np.asarray(values, dtype=float)

    class NoEffectModel:
        def predict(self, values, verbose=0):
            return np.zeros((len(values), 8), dtype=float)

    def fake_fit_lstm_direct(train_frame, seed=42):
        return SimpleNamespace(
            model=NoEffectModel(),
            scaler=IdentityScaler(),
            feature_columns=lstm.LSTM_INPUT_COLUMNS,
            target_column="cpi_yoy",
            lookback=8,
            horizon=8,
        )

    monkeypatch.setattr(lstm, "fit_lstm_direct", fake_fit_lstm_direct)

    importance = lstm.permutation_importance(
        full_frame=frame,
        holdout_size=10,
        n_repeats=3,
        seed=123,
    )

    assert set(importance["feature"]) == set(lstm.LSTM_INPUT_COLUMNS)
    assert np.allclose(importance["rmse_increase"], 0.0)
    assert importance["interpretation_note"].str.contains("Small-sample").all()


def test_existing_sarimax_d_metrics_loader_uses_canonical_group_d_rows(tmp_path):
    report_path = tmp_path / "model_comparison_sarimax.csv"
    pd.DataFrame(
        {
            "group_id": ["C", "D", "D"],
            "model": ["sarimax", "sarimax", "sarima"],
            "horizon": ["overall", "overall", "overall"],
            "n": [10, 66, 66],
            "rmse": [9.0, 0.860526, 0.5],
            "mae": [8.0, 0.649623, 0.4],
        }
    ).to_csv(report_path, index=False)

    rows = lstm._load_existing_sarimax_d_metrics(report_path)

    assert rows["group_id"].tolist() == ["D"]
    assert rows["model"].tolist() == ["sarimax"]
    assert rows["n"].tolist() == [66]
    assert rows["rmse"].tolist() == [0.860526]
    assert rows["sample_size_note"].str.contains("not intersected").all()
