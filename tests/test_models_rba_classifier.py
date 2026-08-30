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
