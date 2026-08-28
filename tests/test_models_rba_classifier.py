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
    assert audit.loc[audit["initial_train_size"].eq(2), "degenerate_classes"].iloc[0] == "cut"
    assert audit.loc[audit["initial_train_size"].eq(3), "degenerate_classes"].iloc[0] == "none"
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
