import numpy as np
import pandas as pd
import pytest

from src.models import rba_classifier


def _sample(actions):
    quarters = pd.period_range("2020Q1", periods=len(actions), freq="Q").astype(str)
    return pd.DataFrame(
        {
            "target_quarter": quarters,
            "headline_forecast": [2.0 + i * 0.1 for i in range(len(actions))],
            "trimmed_mean_forecast": [2.1 + i * 0.1 for i in range(len(actions))],
            "unemployment_rate_change_lag1": [0.1 if i % 2 else -0.1 for i in range(len(actions))],
            "cash_rate": [3.0 + i * 0.05 for i in range(len(actions))],
            "cash_rate_lag1": [2.9 + i * 0.05 for i in range(len(actions))],
            "cash_rate_change": [
                0.1 if action == "hike" else -0.1 if action == "cut" else 0.0
                for action in actions
            ],
            "policy_action": pd.Series(actions, dtype=rba_classifier.ACTION_DTYPE),
        }
    )


def test_discretize_cash_rate_change_preserves_exact_zero_hold_boundary():
    changes = pd.Series([-0.25, -0.0001, 0.0, 0.0001, 0.25])

    labels = rba_classifier.discretize_cash_rate_change(changes)

    assert labels.tolist() == ["cut", "cut", "hold", "hike", "hike"]
    assert labels.dtype == rba_classifier.ACTION_DTYPE
    assert labels.cat.ordered


def test_class_count_gate_flags_degenerate_training_folds_and_selects_valid_split():
    sample = _sample(["hike", "hold", "cut", "cut", "hold", "hike"])

    audit = rba_classifier.split_class_count_audit(
        sample,
        candidate_initial_train_sizes=(1, 2, 3),
    )
    choice = rba_classifier.choose_walk_forward_split(
        sample,
        candidate_initial_train_sizes=(1, 2, 3),
        min_initial_train_size=1,
    )

    assert audit.loc[audit["initial_train_size"].eq(1), "degenerate_classes"].iloc[0] == "cut, hold"
    assert (
        audit.loc[audit["initial_train_size"].eq(1), "degenerate_binary_subproblems"].iloc[0]
        == "Y>cut, Y>hold"
    )
    assert audit.loc[audit["initial_train_size"].eq(2), "degenerate_classes"].iloc[0] == "cut"
    assert audit.loc[audit["initial_train_size"].eq(2), "degenerate_binary_subproblems"].iloc[0] == "Y>cut"
    assert audit.loc[audit["initial_train_size"].eq(3), "degenerate_classes"].iloc[0] == "none"
    assert (
        audit.loc[audit["initial_train_size"].eq(3), "degenerate_binary_subproblems"].iloc[0]
        == "none"
    )
    assert choice.initial_train_size == 3


def test_class_count_gate_raises_when_all_candidates_are_degenerate():
    sample = _sample(["hold", "hold", "cut", "cut"])

    with pytest.raises(ValueError, match="No candidate walk-forward split avoids degenerate"):
        rba_classifier.choose_walk_forward_split(
            sample,
            candidate_initial_train_sizes=(1, 2, 3),
            min_initial_train_size=1,
        )


def test_threshold_baseline_uses_open_cut_hike_thresholds_and_closed_hold_band():
    forecasts = pd.Series([1.99, 2.0, 2.5, 3.0, 3.01])

    predictions = rba_classifier.threshold_baseline_predict(forecasts)

    assert predictions.tolist() == ["cut", "hold", "hold", "hold", "hike"]
    assert predictions.dtype == rba_classifier.ACTION_DTYPE


def test_majority_vote_ensemble_uses_four_voters_and_threshold_tie_break():
    predictions, tie_breaks = rba_classifier.majority_vote_ensemble_predict(
        {
            "threshold": pd.Series(["cut", "hike", "hold", "cut"]),
            "taylor_rule_estimated": pd.Series(["cut", "hold", "hold", "hike"]),
            "ordered_logit": pd.Series(["hold", "hold", "cut", "hike"]),
            "ordered_probit": pd.Series(["cut", "hike", "hike", "cut"]),
        }
    )

    assert predictions.tolist() == ["cut", "hike", "hold", "cut"]
    assert tie_breaks.tolist() == [False, True, False, True]
    assert predictions.dtype == rba_classifier.ACTION_DTYPE
    assert tie_breaks.dtype == bool

    with pytest.raises(ValueError, match="requires exactly these voters"):
        rba_classifier.majority_vote_ensemble_predict(
            {
                "threshold": pd.Series(["hold"]),
                "ordered_logit": pd.Series(["hold"]),
                "ordered_probit": pd.Series(["hold"]),
                "frank_hall_xgboost": pd.Series(["hold"]),
            }
        )


def test_taylor_rule_baseline_uses_fixed_coefficients_and_lagged_cash_rate_status_quo():
    predictions = rba_classifier.taylor_rule_baseline_predict(
        headline_forecast=pd.Series([2.5, 2.5, 2.5]),
        unemployment_rate_change_lag1=pd.Series([0.0, 0.4, -0.4]),
        current_cash_rate=pd.Series([3.5, 3.5, 3.5]),
        r_star=1.0,
    )

    assert predictions.tolist() == ["hold", "hike", "cut"]
    assert predictions.dtype == rba_classifier.ACTION_DTYPE


def test_estimated_taylor_probabilities_sum_to_one():
    probabilities = rba_classifier._estimated_taylor_probability_frame(
        pd.Series([-0.5, 0.0, 0.5]),
        sigma=0.2,
    )

    assert probabilities.columns.tolist() == ["p_cut", "p_hold", "p_hike"]
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))


def test_threshold_probabilities_sum_to_one_for_simulated_draws():
    probabilities = rba_classifier._threshold_probability_frame(
        np.asarray([1.8, 2.0, 2.5, 3.0, 3.2]),
        index=pd.Index([0]),
    )

    assert probabilities.loc[0, "p_cut"] == pytest.approx(0.2)
    assert probabilities.loc[0, "p_hold"] == pytest.approx(0.6)
    assert probabilities.loc[0, "p_hike"] == pytest.approx(0.2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(1))


def test_predict_single_quarter_returns_all_model_rows_without_actual_label(monkeypatch):
    train = _sample(["cut", "hold", "hike", "hold"])
    test = pd.DataFrame(
        {
            "target_quarter": ["2021Q1"],
            "headline_forecast": [3.2],
            "trimmed_mean_forecast": [2.8],
            "unemployment_rate_change_lag1": [0.1],
            "cash_rate": [4.0],
            "cash_rate_lag1": [3.9],
        }
    )

    monkeypatch.setattr(
        rba_classifier,
        "_predict_threshold_confidence",
        lambda *args, **kwargs: (
            pd.DataFrame({"p_cut": [0.1], "p_hold": [0.2], "p_hike": [0.7]}),
            {
                "threshold_forecast_origin": "2020Q4",
                "threshold_simulated_median": 3.2,
                "threshold_simulated_action": "hike",
            },
        ),
    )
    monkeypatch.setattr(
        rba_classifier,
        "_predict_estimated_taylor_rule",
        lambda train, test: (
            pd.Series(
                pd.Categorical(
                    ["hike"],
                    categories=rba_classifier.ACTION_ORDER,
                    ordered=True,
                )
            ),
            pd.DataFrame({"p_cut": [0.1], "p_hold": [0.3], "p_hike": [0.6]}),
            {
                "intercept": 0.0,
                "coef_inflation_gap": 0.1,
                "coef_unemployment_rate_change_lag1": 0.2,
                "predicted_change": 0.3,
                "residual_se": 0.1,
                "unstable_coefficients": "",
            },
        ),
    )

    def fake_ordered_model(train, test, link):
        label = "hold" if link == "logit" else "hike"
        return (
            pd.Series(
                pd.Categorical(
                    [label],
                    categories=rba_classifier.ACTION_ORDER,
                    ordered=True,
                )
            ),
            pd.DataFrame({"p_cut": [0.2], "p_hold": [0.5], "p_hike": [0.3]}),
            [],
            {
                "ordered_alpha_cut_hold": -1.0,
                "ordered_alpha_hold_hike": 1.0,
                "train_rows": len(train),
            },
        )

    monkeypatch.setattr(rba_classifier, "_predict_ordered_model", fake_ordered_model)
    monkeypatch.setattr(
        rba_classifier,
        "_predict_frank_hall_xgboost",
        lambda train, test: pd.Series(
            pd.Categorical(
                ["hold"],
                categories=rba_classifier.ACTION_ORDER,
                ordered=True,
            )
        ),
    )

    predictions = rba_classifier.predict_single_quarter(
        train,
        test,
        r_star=1.0,
        threshold_simulation_frame=pd.DataFrame(
            {"cpi_yoy": [2.0]},
            index=pd.period_range("2020Q4", periods=1, freq="Q"),
        ),
        sarima_series=pd.Series(
            [2.0],
            index=pd.period_range("2020Q4", periods=1, freq="Q"),
        ),
    )

    assert predictions["model"].tolist() == list(rba_classifier.MODEL_ORDER)
    assert predictions["target_quarter"].unique().tolist() == ["2021Q1"]
    assert predictions["actual_action"].isna().all()
    threshold = predictions.loc[predictions["model"].eq("threshold")].iloc[0]
    assert threshold["predicted_action"] == "hike"
    assert threshold["confidence"] == pytest.approx(0.7)
    majority = predictions.loc[predictions["model"].eq("majority_vote_ensemble")].iloc[0]
    assert majority["predicted_action"] == "hike"


def test_predict_single_quarter_uses_independent_sarima_series_for_threshold(
    monkeypatch,
):
    train = _sample(["cut", "hold", "hike", "hold"])
    test = pd.DataFrame(
        {
            "target_quarter": ["2020Q4"],
            "headline_forecast": [2.4],
            "trimmed_mean_forecast": [2.5],
            "unemployment_rate_change_lag1": [0.0],
            "cash_rate": [3.0],
            "cash_rate_lag1": [2.9],
        }
    )
    raw_sarima_series = pd.Series(
        [1.8, 2.0, 2.2],
        index=pd.period_range("2020Q1", periods=3, freq="Q"),
        name=rba_classifier.TARGET_COLUMN,
    )
    threshold_simulation_frame = pd.DataFrame(
        {
            rba_classifier.TARGET_COLUMN: [2.2],
            "cpi_yoy_lag1": [2.0],
        },
        index=pd.period_range("2020Q3", periods=1, freq="Q"),
    )
    captured = {}

    def fake_simulate_ensemble_paths(train_frame, **kwargs):
        captured["train_frame_index"] = train_frame.index.copy()
        captured["sarima_series"] = kwargs["sarima_series"].copy()
        return np.asarray([[2.4], [2.6], [3.2]], dtype=float)

    monkeypatch.setattr(
        rba_classifier,
        "load_target_series",
        lambda curated_path, target_column: raw_sarima_series,
    )
    monkeypatch.setattr(
        rba_classifier,
        "simulate_ensemble_paths",
        fake_simulate_ensemble_paths,
    )
    monkeypatch.setattr(
        rba_classifier,
        "_predict_estimated_taylor_rule",
        lambda train, test: (
            pd.Series(
                pd.Categorical(
                    ["hold"],
                    categories=rba_classifier.ACTION_ORDER,
                    ordered=True,
                )
            ),
            pd.DataFrame({"p_cut": [0.1], "p_hold": [0.8], "p_hike": [0.1]}),
            {
                "intercept": 0.0,
                "coef_inflation_gap": 0.1,
                "coef_unemployment_rate_change_lag1": 0.2,
                "predicted_change": 0.0,
                "residual_se": 0.1,
                "unstable_coefficients": "",
            },
        ),
    )
    monkeypatch.setattr(
        rba_classifier,
        "_predict_ordered_model",
        lambda train, test, link: (
            pd.Series(
                pd.Categorical(
                    ["hold"],
                    categories=rba_classifier.ACTION_ORDER,
                    ordered=True,
                )
            ),
            pd.DataFrame({"p_cut": [0.1], "p_hold": [0.8], "p_hike": [0.1]}),
            [],
            {
                "ordered_alpha_cut_hold": -1.0,
                "ordered_alpha_hold_hike": 1.0,
                "train_rows": len(train),
            },
        ),
    )
    monkeypatch.setattr(
        rba_classifier,
        "_predict_frank_hall_xgboost",
        lambda train, test: pd.Series(
            pd.Categorical(
                ["hold"],
                categories=rba_classifier.ACTION_ORDER,
                ordered=True,
            )
        ),
    )

    rba_classifier.predict_single_quarter(
        train,
        test,
        r_star=1.0,
        threshold_simulation_frame=threshold_simulation_frame,
    )

    assert captured["train_frame_index"].tolist() == [pd.Period("2020Q3", freq="Q")]
    assert captured["sarima_series"].index.tolist() == list(raw_sarima_series.index)
    assert len(captured["sarima_series"]) == 3


def test_prediction_confidence_table_reports_predicted_action_probability():
    predictions = pd.DataFrame(
        {
            "target_quarter": ["2020Q1", "2020Q1", "2020Q1", "2020Q1"],
            "model": [
                "threshold",
                "ordered_logit",
                "ordered_probit",
                "taylor_rule_estimated",
            ],
            "predicted_action": ["hold", "cut", "hold", "hike"],
            "p_cut": [0.4, 0.8, 0.1, 0.2],
            "p_hold": [0.3, 0.1, 0.7, 0.2],
            "p_hike": [0.3, 0.1, 0.2, 0.6],
            "confidence": [0.3, 0.8, 0.7, 0.6],
        }
    )

    table = rba_classifier.prediction_confidence_table(predictions)

    assert table["confidence"].tolist() == pytest.approx([0.3, 0.8, 0.7, 0.6])


def test_frank_hall_probabilities_are_monotone_corrected_before_argmax():
    predictions = rba_classifier._frank_hall_probabilities_to_actions(
        p_y_gt_cut=pd.Series([0.2, 0.8, 0.4]),
        p_y_gt_hold=pd.Series([0.1, 0.3, 0.9]),
        index=pd.RangeIndex(3),
    )

    assert predictions.tolist() == ["cut", "hold", "cut"]
    assert predictions.dtype == rba_classifier.ACTION_DTYPE


def test_policy_metric_computation_uses_macro_f1_and_action_ordered_confusion_matrix():
    predictions = pd.DataFrame(
        {
            "model": ["threshold"] * 6,
            "actual_action": ["cut", "cut", "hold", "hold", "hike", "hike"],
            "predicted_action": ["cut", "hold", "hold", "hold", "hold", "hike"],
        }
    )

    metrics = rba_classifier.compute_policy_metrics(predictions)
    matrix = rba_classifier.confusion_matrix_frame(
        predictions["actual_action"],
        predictions["predicted_action"],
    )

    assert metrics.loc[0, "model"] == "threshold"
    assert metrics.loc[0, "macro_f1"] == pytest.approx(2 / 3)
    assert metrics.loc[0, "accuracy"] == pytest.approx(4 / 6)
    assert matrix.index.tolist() == ["cut", "hold", "hike"]
    assert matrix.columns.tolist() == ["cut", "hold", "hike"]
    assert matrix.loc["cut", "cut"] == 1
    assert matrix.loc["cut", "hold"] == 1
    assert matrix.loc["hold", "hold"] == 2
    assert matrix.loc["hike", "hold"] == 1
    assert matrix.loc["hike", "hike"] == 1
