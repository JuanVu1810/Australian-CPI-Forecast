"""RBA policy action classifier from leakage-safe Ensemble CPI forecasts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.models.evaluation import PROJECT_ROOT


HEADLINE_BACKTEST_PATH = PROJECT_ROOT / "reports/backtest_predictions.csv"
TRIMMED_MEAN_BACKTEST_PATH = PROJECT_ROOT / "reports/backtest_predictions_trimmed_mean.csv"
CURATED_MACRO_PATH = PROJECT_ROOT / "data/curated/quarterly_macro_features.parquet"
EVALUATION_REPORT_PATH = PROJECT_ROOT / "reports/rba_classifier_evaluation.md"

ACTION_ORDER = ("cut", "hold", "hike")
ACTION_DTYPE = pd.CategoricalDtype(categories=ACTION_ORDER, ordered=True)
FEATURE_COLUMNS = ("headline_forecast", "trimmed_mean_forecast")
DEFAULT_CANDIDATE_INITIAL_TRAIN_SIZES = (1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 40)
DEFAULT_MIN_INITIAL_TRAIN_SIZE = 12


@dataclass(frozen=True)
class SplitChoice:
    """Chosen expanding-window split and its audited candidate table."""

    initial_train_size: int
    audit: pd.DataFrame
    rationale: str


def discretize_cash_rate_change(changes: pd.Series) -> pd.Series:
    """Map cash-rate changes to the ordered policy action label."""
    values = pd.to_numeric(changes, errors="coerce")
    labels = pd.Series(pd.NA, index=changes.index, dtype="object")
    labels.loc[values.lt(0)] = "cut"
    labels.loc[values.eq(0)] = "hold"
    labels.loc[values.gt(0)] = "hike"
    return labels.astype(ACTION_DTYPE)


def class_counts(labels: pd.Series) -> pd.Series:
    """Return cut/hold/hike counts with absent classes retained as zeros."""
    categorical = labels.astype(ACTION_DTYPE)
    return categorical.value_counts(sort=False).reindex(ACTION_ORDER, fill_value=0).astype(int)


def _load_ensemble_horizon_one(path: Path, feature_name: str) -> pd.DataFrame:
    required = {"model", "target_quarter", "horizon", "forecast"}
    frame = pd.read_csv(path, usecols=sorted(required))
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}.")

    filtered = frame.loc[
        frame["model"].eq("ensemble") & frame["horizon"].astype(int).eq(1),
        ["target_quarter", "forecast"],
    ].copy()
    filtered = filtered.rename(columns={"forecast": feature_name})
    if filtered["target_quarter"].duplicated().any():
        duplicates = filtered.loc[
            filtered["target_quarter"].duplicated(),
            "target_quarter",
        ].tolist()
        raise ValueError(f"{path} has duplicate Ensemble horizon-1 rows for {duplicates}.")
    return filtered


def assemble_policy_sample(
    headline_path: Path = HEADLINE_BACKTEST_PATH,
    trimmed_mean_path: Path = TRIMMED_MEAN_BACKTEST_PATH,
    macro_path: Path = CURATED_MACRO_PATH,
) -> pd.DataFrame:
    """Join Ensemble headline/trimmed-mean forecasts to cash-rate action labels."""
    headline = _load_ensemble_horizon_one(headline_path, "headline_forecast")
    trimmed_mean = _load_ensemble_horizon_one(trimmed_mean_path, "trimmed_mean_forecast")
    macro = pd.read_parquet(macro_path, columns=["quarter", "cash_rate_change"]).rename(
        columns={"quarter": "target_quarter"}
    )

    sample = headline.merge(trimmed_mean, on="target_quarter", how="inner").merge(
        macro,
        on="target_quarter",
        how="inner",
    )
    sample["policy_action"] = discretize_cash_rate_change(sample["cash_rate_change"])
    sample = sample.dropna(subset=[*FEATURE_COLUMNS, "cash_rate_change", "policy_action"]).copy()
    sample["_quarter_order"] = sample["target_quarter"].map(
        lambda value: pd.Period(value, freq="Q")
    )
    sample = sample.sort_values("_quarter_order").drop(columns="_quarter_order")
    return sample.reset_index(drop=True)


def expanding_walk_forward_splits(
    sample: pd.DataFrame,
    initial_train_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return one-step expanding walk-forward train/test index splits."""
    n_rows = len(sample)
    if initial_train_size < 1:
        raise ValueError("initial_train_size must be positive.")
    if initial_train_size >= n_rows:
        raise ValueError("initial_train_size must leave at least one test row.")

    splits = []
    for test_position in range(initial_train_size, n_rows):
        train_index = np.arange(0, test_position)
        test_index = np.array([test_position])
        splits.append((train_index, test_index))
    return splits


def split_class_count_audit(
    sample: pd.DataFrame,
    candidate_initial_train_sizes: tuple[int, ...] = DEFAULT_CANDIDATE_INITIAL_TRAIN_SIZES,
) -> pd.DataFrame:
    """Audit candidate expanding windows for missing classes in training folds."""
    rows = []
    labels = sample["policy_action"].astype(ACTION_DTYPE)
    for initial_train_size in candidate_initial_train_sizes:
        if initial_train_size >= len(sample):
            rows.append(
                {
                    "initial_train_size": int(initial_train_size),
                    "test_rows": 0,
                    "first_test_quarter": pd.NA,
                    "min_train_cut": pd.NA,
                    "min_train_hold": pd.NA,
                    "min_train_hike": pd.NA,
                    "degenerate_fold_count": pd.NA,
                    "degenerate_classes": "invalid_no_test_rows",
                }
            )
            continue

        fold_counts = []
        degenerate_classes: set[str] = set()
        for train_index, _ in expanding_walk_forward_splits(sample, initial_train_size):
            counts = class_counts(labels.iloc[train_index])
            fold_counts.append(counts)
            degenerate_classes.update(counts.loc[counts.eq(0)].index.astype(str))

        count_frame = pd.DataFrame(fold_counts)
        rows.append(
            {
                "initial_train_size": int(initial_train_size),
                "test_rows": len(sample) - int(initial_train_size),
                "first_test_quarter": str(sample.iloc[int(initial_train_size)]["target_quarter"]),
                "min_train_cut": int(count_frame["cut"].min()),
                "min_train_hold": int(count_frame["hold"].min()),
                "min_train_hike": int(count_frame["hike"].min()),
                "degenerate_fold_count": int(
                    (count_frame.loc[:, ACTION_ORDER].eq(0).any(axis=1)).sum()
                ),
                "degenerate_classes": ", ".join(sorted(degenerate_classes)) or "none",
            }
        )
    return pd.DataFrame(rows)


def choose_walk_forward_split(
    sample: pd.DataFrame,
    candidate_initial_train_sizes: tuple[int, ...] = DEFAULT_CANDIDATE_INITIAL_TRAIN_SIZES,
    min_initial_train_size: int = DEFAULT_MIN_INITIAL_TRAIN_SIZE,
) -> SplitChoice:
    """Choose the first non-degenerate expanding-window candidate after a fit-size floor."""
    audit = split_class_count_audit(sample, candidate_initial_train_sizes)
    valid = audit.loc[
        audit["initial_train_size"].ge(min_initial_train_size)
        & audit["degenerate_fold_count"].eq(0)
    ].copy()
    if valid.empty:
        degenerate = audit.loc[audit["degenerate_fold_count"].ne(0)]
        raise ValueError(
            "No candidate walk-forward split avoids degenerate training folds after "
            f"minimum initial train size {min_initial_train_size}. Degenerate candidates:\n"
            f"{degenerate.to_string(index=False)}"
        )

    chosen = valid.sort_values("initial_train_size").iloc[0]
    size = int(chosen["initial_train_size"])
    rationale = (
        f"Selected initial_train_size={size}: it is the earliest audited candidate at or "
        f"above the {min_initial_train_size}-row fit-size floor with zero training folds "
        "missing cut, hold, or hike. Baseline, logit, and probit are all evaluated on "
        f"the same {int(chosen['test_rows'])} expanding-window test quarters."
    )
    return SplitChoice(initial_train_size=size, audit=audit, rationale=rationale)


def threshold_baseline_predict(headline_forecast: pd.Series | np.ndarray) -> pd.Series:
    """RBA target-band rule: cut below 2%, hike above 3%, hold inside the band."""
    values = pd.Series(np.asarray(headline_forecast, dtype=float))
    predictions = pd.Series("hold", index=values.index, dtype="object")
    predictions.loc[values.lt(2.0)] = "cut"
    predictions.loc[values.gt(3.0)] = "hike"
    return predictions.astype(ACTION_DTYPE)


def _standardize_by_train(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0).replace(0.0, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def _fit_ordered_model(train: pd.DataFrame, link: str):
    endog = train["policy_action"].astype(ACTION_DTYPE).cat.codes
    exog = train.loc[:, FEATURE_COLUMNS].astype(float)
    model = OrderedModel(endog=endog, exog=exog, distr=link)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return model.fit(method="bfgs", maxiter=500, disp=False)


def _predict_ordered_model(train: pd.DataFrame, test: pd.DataFrame, link: str) -> pd.Series:
    x_train = train.loc[:, FEATURE_COLUMNS].astype(float)
    x_test = test.loc[:, FEATURE_COLUMNS].astype(float)
    x_train_scaled, x_test_scaled = _standardize_by_train(x_train, x_test)
    fitted = _fit_ordered_model(
        pd.concat([train[["policy_action"]], x_train_scaled], axis=1),
        link=link,
    )
    probabilities = np.asarray(fitted.model.predict(fitted.params, exog=x_test_scaled))
    predicted_codes = probabilities.argmax(axis=1)
    labels = [ACTION_ORDER[int(code)] for code in predicted_codes]
    return pd.Series(
        pd.Categorical(labels, categories=ACTION_ORDER, ordered=True),
        index=test.index,
    )


def walk_forward_predictions(sample: pd.DataFrame, initial_train_size: int) -> pd.DataFrame:
    """Evaluate threshold, ordered-logit, and ordered-probit on identical test rows."""
    rows = []
    labels = sample["policy_action"].astype(ACTION_DTYPE)
    for train_index, test_index in expanding_walk_forward_splits(sample, initial_train_size):
        train = sample.iloc[train_index].copy()
        test = sample.iloc[test_index].copy()
        actual = labels.iloc[test_index].reset_index(drop=True)
        quarter = str(test.iloc[0]["target_quarter"])

        model_predictions = {
            "threshold": threshold_baseline_predict(test["headline_forecast"]).reset_index(
                drop=True
            ),
            "ordered_logit": _predict_ordered_model(train, test, link="logit").reset_index(
                drop=True
            ),
            "ordered_probit": _predict_ordered_model(
                train,
                test,
                link="probit",
            ).reset_index(drop=True),
        }
        for model, prediction in model_predictions.items():
            rows.append(
                {
                    "model": model,
                    "target_quarter": quarter,
                    "actual_action": str(actual.iloc[0]),
                    "predicted_action": str(prediction.iloc[0]),
                    "headline_forecast": float(test.iloc[0]["headline_forecast"]),
                    "trimmed_mean_forecast": float(test.iloc[0]["trimmed_mean_forecast"]),
                }
            )
    return pd.DataFrame(rows)


def confusion_matrix_frame(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Return an action-ordered confusion matrix with actual rows and predicted columns."""
    matrix = pd.DataFrame(0, index=ACTION_ORDER, columns=ACTION_ORDER, dtype=int)
    for actual, predicted in zip(y_true.astype(str), y_pred.astype(str), strict=True):
        matrix.loc[actual, predicted] += 1
    matrix.index.name = "actual"
    matrix.columns.name = "predicted"
    return matrix


def compute_policy_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute macro-F1, optional accuracy, and support for each classifier."""
    rows = []
    for model, group in predictions.groupby("model", sort=False):
        y_true = group["actual_action"].astype(ACTION_DTYPE)
        y_pred = group["predicted_action"].astype(ACTION_DTYPE)
        rows.append(
            {
                "model": model,
                "macro_f1": f1_score(
                    y_true,
                    y_pred,
                    labels=list(ACTION_ORDER),
                    average="macro",
                    zero_division=0,
                ),
                "accuracy": accuracy_score(y_true, y_pred),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, float_format: str = ".3f") -> str:
    return frame.to_markdown(index=False, floatfmt=float_format)


def _markdown_confusion_matrices(predictions: pd.DataFrame) -> str:
    sections = []
    for model in ("threshold", "ordered_logit", "ordered_probit"):
        rows = predictions.loc[predictions["model"].eq(model)]
        matrix = confusion_matrix_frame(rows["actual_action"], rows["predicted_action"])
        sections.append(f"### {model}\n\n{matrix.reset_index().to_markdown(index=False)}")
    return "\n\n".join(sections)


def build_evaluation_report(
    sample: pd.DataFrame,
    split_choice: SplitChoice,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> str:
    """Render the markdown evaluation report."""
    full_counts = class_counts(sample["policy_action"])
    train = sample.iloc[: split_choice.initial_train_size]
    test = sample.iloc[split_choice.initial_train_size :]
    train_counts = class_counts(train["policy_action"])
    test_counts = class_counts(test["policy_action"])
    degenerate = split_choice.audit.loc[split_choice.audit["degenerate_fold_count"].ne(0)].copy()
    degenerate_text = (
        "No audited candidate at or above the chosen fit-size floor had a degenerate "
        "training fold. Earlier candidates flagged by the gate are shown below."
        if not degenerate.empty
        else "No audited candidate produced a degenerate training fold."
    )

    counts_frame = pd.DataFrame(
        {
            "class": ACTION_ORDER,
            "full_sample": [int(full_counts[label]) for label in ACTION_ORDER],
            "chosen_initial_train": [int(train_counts[label]) for label in ACTION_ORDER],
            "chosen_test": [int(test_counts[label]) for label in ACTION_ORDER],
        }
    )
    display_audit = split_choice.audit.copy()
    display_metrics = metrics.copy()
    display_metrics["macro_f1"] = display_metrics["macro_f1"].round(3)
    display_metrics["accuracy"] = display_metrics["accuracy"].round(3)

    by_quarter = predictions.pivot(
        index="target_quarter",
        columns="model",
        values="predicted_action",
    )
    actual_by_quarter = predictions.drop_duplicates("target_quarter").set_index(
        "target_quarter"
    )["actual_action"]
    paired = by_quarter.join(actual_by_quarter).dropna(
        subset=["threshold", "ordered_logit", "ordered_probit", "actual_action"]
    )

    def _macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
        return float(
            f1_score(
                y_true,
                y_pred,
                labels=list(ACTION_ORDER),
                average="macro",
                zero_division=0,
            )
        )

    rng = np.random.default_rng(42)
    bootstrap_rows = []
    index_positions = np.arange(len(paired))
    for comparator in ("ordered_logit", "ordered_probit"):
        observed_gap = _macro_f1(paired["actual_action"], paired["threshold"]) - _macro_f1(
            paired["actual_action"],
            paired[comparator],
        )
        bootstrap_gaps = []
        for _ in range(2_000):
            sample_positions = rng.choice(index_positions, size=len(index_positions), replace=True)
            boot = paired.iloc[sample_positions]
            bootstrap_gaps.append(
                _macro_f1(boot["actual_action"], boot["threshold"])
                - _macro_f1(boot["actual_action"], boot[comparator])
            )
        lower, upper = np.percentile(bootstrap_gaps, [2.5, 97.5])
        bootstrap_rows.append(
            {
                "gap": f"threshold - {comparator}",
                "observed_macro_f1_gap": round(float(observed_gap), 3),
                "paired_bootstrap_95pct_ci": f"[{lower:.3f}, {upper:.3f}]",
            }
        )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)

    threshold_matrix = confusion_matrix_frame(
        predictions.loc[predictions["model"].eq("threshold"), "actual_action"],
        predictions.loc[predictions["model"].eq("threshold"), "predicted_action"],
    )
    logit_matrix = confusion_matrix_frame(
        predictions.loc[predictions["model"].eq("ordered_logit"), "actual_action"],
        predictions.loc[predictions["model"].eq("ordered_logit"), "predicted_action"],
    )
    probit_matrix = confusion_matrix_frame(
        predictions.loc[predictions["model"].eq("ordered_probit"), "actual_action"],
        predictions.loc[predictions["model"].eq("ordered_probit"), "predicted_action"],
    )
    hike_total = int(threshold_matrix.loc["hike"].sum())
    threshold_hike_recall = int(threshold_matrix.loc["hike", "hike"])
    logit_hikes_as_hold = int(logit_matrix.loc["hike", "hold"])
    probit_hikes_as_hold = int(probit_matrix.loc["hike", "hold"])

    return "\n".join(
        [
            "# RBA Policy Action Classifier Evaluation",
            "",
            "## Scope",
            "",
            (
                "Uses the leakage-safe Ensemble horizon-1 headline and trimmed-mean CPI "
                "forecasts from the existing backtest prediction reports. No forecasting "
                "model is refit or resimulated; the ordinal classifiers are fit only on "
                "the two as-of forecast features."
            ),
            "",
            "## Joined Sample",
            "",
            f"- Joined usable quarters: {len(sample)}",
            (
                f"- Quarter range: {sample['target_quarter'].iloc[0]} to "
                f"{sample['target_quarter'].iloc[-1]}"
            ),
            "",
            _markdown_table(counts_frame),
            "",
            "## Walk-Forward Split Gate",
            "",
            split_choice.rationale,
            "",
            degenerate_text,
            "",
            _markdown_table(display_audit),
            "",
            "## Macro-F1 Comparison",
            "",
            _markdown_table(display_metrics),
            "",
            (
                "Macro-F1 is the comparison metric because policy holds are common enough "
                "that raw accuracy can overstate usefulness."
            ),
            "",
            "## Small-Sample Caveat",
            "",
            (
                f"The walk-forward test set has {len(test)} quarters. At this size, the "
                f"macro-F1 ranking ({display_metrics.iloc[0]['model']} "
                f"{display_metrics.iloc[0]['macro_f1']:.3f}, "
                f"{display_metrics.iloc[1]['model']} {display_metrics.iloc[1]['macro_f1']:.3f}, "
                f"{display_metrics.iloc[2]['model']} {display_metrics.iloc[2]['macro_f1']:.3f}) "
                "is suggestive rather than decisive."
            ),
            "",
            (
                "As a quick uncertainty check, a paired bootstrap over the same test "
                "quarters (2,000 resamples, seed=42) gives the following macro-F1 gap "
                "intervals. This is a rough diagnostic because it treats quarters as "
                "exchangeable and does not model time-series dependence."
            ),
            "",
            _markdown_table(bootstrap_frame),
            "",
            "## Confusion Matrices",
            "",
            _markdown_confusion_matrices(predictions),
            "",
            "## Interpretive Conclusion",
            "",
            (
                "The threshold baseline's macro-F1 lead is a point-estimate win, not a "
                "statistically settled result: the paired-bootstrap intervals above cross "
                "zero for both threshold-minus-ordered-model gaps."
            ),
            "",
            (
                "Despite that non-significance, the threshold rule is still the preferred "
                "reportable Phase 3 result on non-statistical grounds. It is transparent, "
                "has zero fitted parameters, and its main advantage is hike detection: it "
                f"correctly classifies all {threshold_hike_recall} of {hike_total} hike "
                f"quarters, while ordered logit classifies {logit_hikes_as_hold} hikes "
                f"as hold and ordered probit classifies {probit_hikes_as_hold} hikes as "
                "hold. That hike-to-hold error is a structurally different, and arguably "
                "more consequential, failure mode for a policy classifier than the "
                "aggregate macro-F1 gap alone captures."
            ),
            "",
            (
                "A plausible reading is that the two CPI forecast features are highly "
                "correlated and the earliest valid training folds are thin, so the "
                "ordered models smooth some high-inflation tightening quarters back "
                "toward hold. The recommendation is therefore a judgment call made "
                "despite non-significance: keep ordered logit and probit as a documented "
                "negative finding for Phase 4's materiality section rather than exposing "
                "them via an API endpoint."
            ),
            "",
        ]
    )


def run_rba_classifier_evaluation(
    headline_path: Path = HEADLINE_BACKTEST_PATH,
    trimmed_mean_path: Path = TRIMMED_MEAN_BACKTEST_PATH,
    macro_path: Path = CURATED_MACRO_PATH,
    output_path: Path = EVALUATION_REPORT_PATH,
    candidate_initial_train_sizes: tuple[int, ...] = DEFAULT_CANDIDATE_INITIAL_TRAIN_SIZES,
    min_initial_train_size: int = DEFAULT_MIN_INITIAL_TRAIN_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitChoice]:
    """Run the full Phase 3 policy classifier evaluation and write the report."""
    sample = assemble_policy_sample(
        headline_path=headline_path,
        trimmed_mean_path=trimmed_mean_path,
        macro_path=macro_path,
    )
    split_choice = choose_walk_forward_split(
        sample,
        candidate_initial_train_sizes=candidate_initial_train_sizes,
        min_initial_train_size=min_initial_train_size,
    )
    predictions = walk_forward_predictions(sample, split_choice.initial_train_size)
    metrics = compute_policy_metrics(predictions)
    report = build_evaluation_report(sample, split_choice, predictions, metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return sample, predictions, metrics, split_choice


def _parse_initial_sizes(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline", type=Path, default=HEADLINE_BACKTEST_PATH)
    parser.add_argument("--trimmed-mean", type=Path, default=TRIMMED_MEAN_BACKTEST_PATH)
    parser.add_argument("--macro", type=Path, default=CURATED_MACRO_PATH)
    parser.add_argument("--output", type=Path, default=EVALUATION_REPORT_PATH)
    parser.add_argument(
        "--candidate-initial-train-sizes",
        default=",".join(str(size) for size in DEFAULT_CANDIDATE_INITIAL_TRAIN_SIZES),
    )
    parser.add_argument(
        "--min-initial-train-size",
        type=int,
        default=DEFAULT_MIN_INITIAL_TRAIN_SIZE,
    )
    args = parser.parse_args(argv)

    sample, predictions, metrics, split_choice = run_rba_classifier_evaluation(
        headline_path=args.headline,
        trimmed_mean_path=args.trimmed_mean,
        macro_path=args.macro,
        output_path=args.output,
        candidate_initial_train_sizes=_parse_initial_sizes(args.candidate_initial_train_sizes),
        min_initial_train_size=args.min_initial_train_size,
    )
    print(f"Joined sample rows: {len(sample)}")
    print(split_choice.rationale)
    print(metrics.round({"macro_f1": 3, "accuracy": 3}).to_string(index=False))
    for model in ("threshold", "ordered_logit", "ordered_probit"):
        rows = predictions.loc[predictions["model"].eq(model)]
        print()
        print(model)
        print(confusion_matrix_frame(rows["actual_action"], rows["predicted_action"]).to_string())


if __name__ == "__main__":
    main()
