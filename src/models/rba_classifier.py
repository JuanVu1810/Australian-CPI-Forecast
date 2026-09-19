"""RBA policy action classifier from leakage-safe Ensemble CPI forecasts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import ConvergenceWarning

try:  # Optional at import time so lightweight classifier tests can still run.
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - exercised only in environments without xgboost.
    XGBClassifier = None

from src.models.elastic_net import (
    ELASTIC_NET_FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_elastic_net_feature_frame,
)
from src.models.ensemble import horizon_rmse_weights, simulate_ensemble_paths
from src.models.evaluation import CURATED_DATA_PATH, PROJECT_ROOT, load_target_series
from src.models.interval_coverage import DEFAULT_N_SIMS, DEFAULT_SEED


HEADLINE_BACKTEST_PATH = PROJECT_ROOT / "reports/backtest_predictions.csv"
TRIMMED_MEAN_BACKTEST_PATH = PROJECT_ROOT / "reports/backtest_predictions_trimmed_mean.csv"
CURATED_MACRO_PATH = PROJECT_ROOT / "data/curated/quarterly_macro_features.parquet"
EVALUATION_REPORT_PATH = PROJECT_ROOT / "reports/rba_classifier_evaluation.md"

ACTION_ORDER = ("cut", "hold", "hike")
ACTION_DTYPE = pd.CategoricalDtype(categories=ACTION_ORDER, ordered=True)
MODEL_ORDER = (
    "threshold",
    "taylor_rule",
    "taylor_rule_estimated",
    "ordered_logit",
    "ordered_probit",
    "frank_hall_xgboost",
    "majority_vote_ensemble",
)
MAJORITY_VOTE_ENSEMBLE_MODELS = (
    "threshold",
    "taylor_rule_estimated",
    "ordered_logit",
    "ordered_probit",
)
CONFIDENCE_MODELS = (
    "threshold",
    "ordered_logit",
    "ordered_probit",
    "taylor_rule_estimated",
)
PROBABILITY_COLUMNS = tuple(f"p_{action}" for action in ACTION_ORDER)
THRESHOLD_FEATURE_COLUMNS = ("headline_forecast",)
TAYLOR_FEATURE_COLUMNS = ("headline_forecast", "unemployment_rate_change_lag1")
BASE_CLASSIFIER_FEATURE_COLUMNS = (
    "headline_forecast",
    "trimmed_mean_forecast",
    "unemployment_rate_change_lag1",
)
ORDINAL_FEATURE_COLUMNS = BASE_CLASSIFIER_FEATURE_COLUMNS
TAYLOR_IMPLIED_CHANGE_COLUMN = "taylor_implied_change"
XGBOOST_FEATURE_COLUMNS = (*BASE_CLASSIFIER_FEATURE_COLUMNS, TAYLOR_IMPLIED_CHANGE_COLUMN)
FEATURE_COLUMNS = ORDINAL_FEATURE_COLUMNS
REVERTED_ORDERED_FIVE_FEATURE_COLUMNS = (
    *BASE_CLASSIFIER_FEATURE_COLUMNS,
    "cash_rate_lag1",
    "commodity_growth_lag1",
)
REVERTED_ORDERED_FIVE_FEATURE_RESULTS = {
    "ordered_logit": {
        "tested_macro_f1": 0.620,
        "tested_accuracy": 0.610,
        "threshold_minus_tested_gap": 0.155,
        "paired_bootstrap_95pct_ci": "[0.000, 0.338]",
    },
    "ordered_probit": {
        "tested_macro_f1": 0.653,
        "tested_accuracy": 0.634,
        "threshold_minus_tested_gap": 0.122,
        "paired_bootstrap_95pct_ci": "[-0.024, 0.281]",
    },
}
TAYLOR_INFLATION_TARGET_MIDPOINT = 2.5
TAYLOR_INFLATION_COEFFICIENT = 0.5
TAYLOR_UNEMPLOYMENT_CHANGE_COEFFICIENT = 0.5
TAYLOR_ESTIMATED_HOLD_BAND = 0.125
TAYLOR_ESTIMATED_COEFFICIENT_MAGNITUDE_WARNING = 10.0
XGBOOST_PARAMS = {
    "n_estimators": 50,
    "max_depth": 2,
    "min_child_weight": 3,
    "learning_rate": 0.05,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": 1,
}
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
    macro_columns = [
        "quarter",
        "cash_rate",
        "cash_rate_lag1",
        "cash_rate_change",
        "unemployment_rate_change_lag1",
    ]
    macro = pd.read_parquet(macro_path, columns=macro_columns).rename(
        columns={"quarter": "target_quarter"}
    )

    sample = headline.merge(trimmed_mean, on="target_quarter", how="inner").merge(
        macro,
        on="target_quarter",
        how="inner",
    )
    sample["policy_action"] = discretize_cash_rate_change(sample["cash_rate_change"])
    sample = sample.dropna(
        subset=[
            *FEATURE_COLUMNS,
            "cash_rate",
            "cash_rate_lag1",
            "cash_rate_change",
            "policy_action",
        ]
    ).copy()
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
        degenerate_binary_subproblems: set[str] = set()
        binary_fold_count = 0
        for train_index, _ in expanding_walk_forward_splits(sample, initial_train_size):
            train_codes = labels.iloc[train_index].cat.codes
            counts = class_counts(labels.iloc[train_index])
            fold_counts.append(counts)
            degenerate_classes.update(counts.loc[counts.eq(0)].index.astype(str))
            y_gt_cut = train_codes.gt(0)
            y_gt_hold = train_codes.gt(1)
            binary_degenerate = False
            if not (bool(y_gt_cut.any()) and bool((~y_gt_cut).any())):
                degenerate_binary_subproblems.add("Y>cut")
                binary_degenerate = True
            if not (bool(y_gt_hold.any()) and bool((~y_gt_hold).any())):
                degenerate_binary_subproblems.add("Y>hold")
                binary_degenerate = True
            binary_fold_count += int(binary_degenerate)

        count_frame = pd.DataFrame(fold_counts)
        binary_gt_cut_positive = count_frame["hold"] + count_frame["hike"]
        binary_gt_cut_negative = count_frame["cut"]
        binary_gt_hold_positive = count_frame["hike"]
        binary_gt_hold_negative = count_frame["cut"] + count_frame["hold"]
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
                "min_binary_gt_cut_negative": int(binary_gt_cut_negative.min()),
                "min_binary_gt_cut_positive": int(binary_gt_cut_positive.min()),
                "min_binary_gt_hold_negative": int(binary_gt_hold_negative.min()),
                "min_binary_gt_hold_positive": int(binary_gt_hold_positive.min()),
                "degenerate_binary_fold_count": int(binary_fold_count),
                "degenerate_binary_subproblems": (
                    ", ".join(sorted(degenerate_binary_subproblems)) or "none"
                ),
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
        & audit["degenerate_binary_fold_count"].eq(0)
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
        "missing cut, hold, hike, or either Frank-Hall binary outcome. All seven "
        f"classifier outputs are evaluated on the same {int(chosen['test_rows'])} "
        "expanding-window test quarters."
    )
    return SplitChoice(initial_train_size=size, audit=audit, rationale=rationale)


def threshold_baseline_predict(headline_forecast: pd.Series | np.ndarray) -> pd.Series:
    """RBA target-band rule: cut below 2%, hike above 3%, hold inside the band."""
    values = pd.Series(np.asarray(headline_forecast, dtype=float))
    predictions = pd.Series("hold", index=values.index, dtype="object")
    predictions.loc[values.lt(2.0)] = "cut"
    predictions.loc[values.gt(3.0)] = "hike"
    return predictions.astype(ACTION_DTYPE)


def threshold_input_sensitivity(test: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Apply the threshold rule to the headline and to the trimmed-mean forecast.

    A sensitivity check on the rule's input over the same rows, not a candidate model:
    the reportable threshold baseline is defined on ``headline_forecast``. Returns the
    per-input scores and the number of rows where the two inputs give different calls.
    """
    actual = test["policy_action"].astype(str).to_numpy()
    calls = {}
    rows = []
    for column, label in (
        ("headline_forecast", "headline (reportable)"),
        ("trimmed_mean_forecast", "trimmed mean"),
    ):
        predicted = threshold_baseline_predict(test[column]).astype(str).to_numpy()
        calls[label] = predicted
        rows.append(
            {
                "threshold_input": label,
                "macro_f1": float(
                    f1_score(
                        actual,
                        predicted,
                        labels=list(ACTION_ORDER),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "accuracy": float(accuracy_score(actual, predicted)),
                **{
                    f"predicted_{action}": int((predicted == action).sum())
                    for action in ACTION_ORDER
                },
            }
        )
    disagreements = int((calls["headline (reportable)"] != calls["trimmed mean"]).sum())
    return pd.DataFrame(rows), disagreements


def majority_vote_ensemble_predict(
    voter_predictions: Mapping[str, pd.Series | np.ndarray],
) -> tuple[pd.Series, pd.Series]:
    """Combine the four precomputed classifier predictions with threshold tie-breaks."""
    voter_names = set(voter_predictions)
    expected_names = set(MAJORITY_VOTE_ENSEMBLE_MODELS)
    if voter_names != expected_names:
        missing = sorted(expected_names.difference(voter_names))
        extra = sorted(voter_names.difference(expected_names))
        raise ValueError(
            "majority_vote_ensemble requires exactly these voters: "
            f"{', '.join(MAJORITY_VOTE_ENSEMBLE_MODELS)}. "
            f"Missing: {', '.join(missing) or 'none'}; extra: {', '.join(extra) or 'none'}."
        )

    voter_frame = pd.DataFrame(
        {
            model: pd.Series(voter_predictions[model]).reset_index(drop=True).astype(str)
            for model in MAJORITY_VOTE_ENSEMBLE_MODELS
        }
    )
    invalid_labels = sorted(
        {
            label
            for label in voter_frame.to_numpy().ravel()
            if label not in set(ACTION_ORDER)
        }
    )
    if invalid_labels:
        raise ValueError(
            "majority_vote_ensemble received invalid action labels: "
            f"{', '.join(invalid_labels)}."
        )

    labels = []
    tie_breaks = []
    for _, row in voter_frame.iterrows():
        counts = row.value_counts()
        max_votes = int(counts.max())
        winners = [label for label in ACTION_ORDER if int(counts.get(label, 0)) == max_votes]
        tied = len(winners) > 1
        labels.append(str(row["threshold"]) if tied else winners[0])
        tie_breaks.append(tied)

    return (
        pd.Series(pd.Categorical(labels, categories=ACTION_ORDER, ordered=True)),
        pd.Series(tie_breaks, dtype=bool),
    )


def calibrate_taylor_r_star(sample: pd.DataFrame) -> float:
    """Use the sample's average real policy rate as the Taylor-rule r_star proxy."""
    real_policy_rate = sample["cash_rate"].astype(float) - sample["headline_forecast"].astype(
        float
    )
    return float(real_policy_rate.mean())


def taylor_rule_implied_change(
    headline_forecast: pd.Series | np.ndarray,
    unemployment_rate_change_lag1: pd.Series | np.ndarray,
    current_cash_rate: pd.Series | np.ndarray,
    *,
    r_star: float,
) -> pd.Series:
    """Return the fixed-coefficient Taylor-rule implied cash-rate change."""
    headline = pd.Series(np.asarray(headline_forecast, dtype=float))
    unemployment_change = pd.Series(np.asarray(unemployment_rate_change_lag1, dtype=float))
    current_rate = pd.Series(np.asarray(current_cash_rate, dtype=float))
    implied_rate = (
        float(r_star)
        + headline
        + TAYLOR_INFLATION_COEFFICIENT * (headline - TAYLOR_INFLATION_TARGET_MIDPOINT)
        + TAYLOR_UNEMPLOYMENT_CHANGE_COEFFICIENT * unemployment_change
    )
    return implied_rate - current_rate


def taylor_rule_baseline_predict(
    headline_forecast: pd.Series | np.ndarray,
    unemployment_rate_change_lag1: pd.Series | np.ndarray,
    current_cash_rate: pd.Series | np.ndarray,
    *,
    r_star: float,
) -> pd.Series:
    """Fixed-coefficient Taylor-style rule discretized by implied rate change."""
    implied_change = taylor_rule_implied_change(
        headline_forecast,
        unemployment_rate_change_lag1,
        current_cash_rate,
        r_star=r_star,
    )
    return discretize_cash_rate_change(implied_change)


def discretize_taylor_estimated_change(changes: pd.Series | np.ndarray) -> pd.Series:
    """Map estimated Taylor-rule changes to actions using the fixed +/-0.125pp band."""
    values = pd.Series(np.asarray(changes, dtype=float))
    predictions = pd.Series("hold", index=values.index, dtype="object")
    predictions.loc[values.lt(-TAYLOR_ESTIMATED_HOLD_BAND)] = "cut"
    predictions.loc[values.gt(TAYLOR_ESTIMATED_HOLD_BAND)] = "hike"
    return predictions.astype(ACTION_DTYPE)


def _probability_frame(probabilities: np.ndarray, index: pd.Index) -> pd.DataFrame:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape != (len(index), len(ACTION_ORDER)):
        raise ValueError(
            "Expected one cut/hold/hike probability vector per prediction row; "
            f"got shape {values.shape} for {len(index)} rows."
        )
    frame = pd.DataFrame(values, columns=PROBABILITY_COLUMNS, index=index)
    if not np.allclose(frame.loc[:, PROBABILITY_COLUMNS].sum(axis=1), 1.0):
        raise ValueError("Predicted action probabilities do not sum to 1.")
    return frame


def _threshold_probability_frame(draws: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Convert horizon-1 CPI simulation draws into threshold class probabilities."""
    values = np.asarray(draws, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Threshold confidence requires at least one simulated draw.")
    p_cut = float(np.mean(values < 2.0))
    p_hike = float(np.mean(values > 3.0))
    p_hold = 1.0 - p_cut - p_hike
    return _probability_frame(
        np.asarray([[p_cut, p_hold, p_hike]], dtype=float),
        index=index,
    )


def _load_threshold_simulation_frame(
    curated_path: Path = CURATED_DATA_PATH,
) -> pd.DataFrame:
    target = load_target_series(curated_path, target_column=TARGET_COLUMN)
    exog = load_elastic_net_feature_frame(
        curated_path,
        feature_columns=ELASTIC_NET_FEATURE_COLUMNS,
    )
    frame = pd.concat([target.rename(TARGET_COLUMN), exog], axis=1).dropna()
    if frame.empty:
        raise ValueError("Threshold confidence simulation frame is empty.")
    return frame.sort_index()


def _predict_threshold_confidence(
    test: pd.DataFrame,
    simulation_frame: pd.DataFrame,
    *,
    sarima_series: pd.Series,
    n_sims: int,
    seed: int,
    weights: dict[int, tuple[float, float]],
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    target_quarter = pd.Period(str(test.iloc[0]["target_quarter"]), freq="Q")
    forecast_origin = target_quarter - 1
    train_frame = simulation_frame.loc[:forecast_origin].dropna()
    train_sarima = sarima_series.loc[:forecast_origin].dropna()
    if train_frame.empty:
        raise ValueError(
            "Threshold confidence has no training rows through "
            f"forecast origin {forecast_origin}."
        )
    if train_sarima.empty:
        raise ValueError(
            "Threshold confidence has no SARIMA training rows through "
            f"forecast origin {forecast_origin}."
        )

    paths = simulate_ensemble_paths(
        train_frame,
        steps=1,
        n_sims=n_sims,
        weights=weights,
        seed=seed,
        target_column=TARGET_COLUMN,
        elastic_net_feature_columns=ELASTIC_NET_FEATURE_COLUMNS,
        sarima_series=train_sarima,
    )
    horizon_one_draws = np.asarray(paths, dtype=float)[:, 0]
    probabilities = _threshold_probability_frame(horizon_one_draws, index=test.index)
    simulated_median = float(np.median(horizon_one_draws))
    simulated_action = str(threshold_baseline_predict(pd.Series([simulated_median])).iloc[0])
    diagnostics: dict[str, float | str] = {
        "threshold_forecast_origin": str(forecast_origin),
        "threshold_simulated_median": simulated_median,
        "threshold_simulated_action": simulated_action,
    }
    return probabilities, diagnostics


def _estimated_taylor_probability_frame(
    predicted_change: pd.Series,
    *,
    sigma: float,
) -> pd.DataFrame:
    """Convert estimated Taylor-rule point changes into normal class probabilities."""
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"Estimated Taylor-rule residual SE must be positive; got {sigma}.")

    values = predicted_change.astype(float)
    lower_z = (-TAYLOR_ESTIMATED_HOLD_BAND - values) / sigma
    upper_z = (TAYLOR_ESTIMATED_HOLD_BAND - values) / sigma
    p_cut = norm.cdf(lower_z)
    p_hold = norm.cdf(upper_z) - p_cut
    p_hike = 1.0 - norm.cdf(upper_z)
    return _probability_frame(
        np.column_stack([p_cut, p_hold, p_hike]),
        index=predicted_change.index,
    )


def _estimated_taylor_exog(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "inflation_gap": (
                frame["headline_forecast"].astype(float) - TAYLOR_INFLATION_TARGET_MIDPOINT
            ),
            "unemployment_rate_change_lag1": frame["unemployment_rate_change_lag1"].astype(float),
        },
        index=frame.index,
    )


def _predict_estimated_taylor_rule(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, dict[str, float | str]]:
    y_train = train["cash_rate_change"].astype(float)
    x_train = sm.add_constant(_estimated_taylor_exog(train), has_constant="add")
    x_test = sm.add_constant(_estimated_taylor_exog(test), has_constant="add")
    fitted = sm.OLS(y_train, x_train).fit()
    predicted_change = pd.Series(fitted.predict(x_test), index=test.index)
    residual_se = float(np.sqrt(float(fitted.mse_resid)))
    probabilities = _estimated_taylor_probability_frame(
        predicted_change,
        sigma=residual_se,
    )
    coefficients = {
        "intercept": float(fitted.params["const"]),
        "coef_inflation_gap": float(fitted.params["inflation_gap"]),
        "coef_unemployment_rate_change_lag1": float(
            fitted.params["unemployment_rate_change_lag1"]
        ),
    }
    unstable = [
        name
        for name, value in coefficients.items()
        if name != "intercept"
        and abs(value) > TAYLOR_ESTIMATED_COEFFICIENT_MAGNITUDE_WARNING
    ]
    diagnostics: dict[str, float | str] = {
        **coefficients,
        "predicted_change": float(predicted_change.iloc[0]),
        "residual_se": residual_se,
        "unstable_coefficients": ", ".join(unstable),
    }
    return discretize_taylor_estimated_change(predicted_change), probabilities, diagnostics


def _standardize_by_train(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0).replace(0.0, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def _fit_ordered_model(train: pd.DataFrame, link: str):
    endog = train["policy_action"].astype(ACTION_DTYPE).cat.codes
    exog = train.loc[:, ORDINAL_FEATURE_COLUMNS].astype(float)
    model = OrderedModel(endog=endog, exog=exog, distr=link)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=ConvergenceWarning)
        fitted = model.fit(method="bfgs", maxiter=500, disp=False)
    warning_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    return fitted, warning_messages


def _ordered_model_parameter_diagnostics(
    fitted,
    *,
    link: str,
    target_quarter: str,
    train_rows: int,
) -> dict[str, float | str | int]:
    params = fitted.params
    thresholds = fitted.model.transform_threshold_params(params)[1:-1]
    diagnostics: dict[str, float | str | int] = {
        "model": f"ordered_{link}",
        "target_quarter": target_quarter,
        "train_rows": int(train_rows),
        "ordered_alpha_cut_hold": float(thresholds[0]),
        "ordered_alpha_hold_hike": float(thresholds[1]),
    }
    for feature in ORDINAL_FEATURE_COLUMNS:
        diagnostics[f"ordered_beta_{feature}"] = float(params[feature])
    return diagnostics


def _predict_ordered_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    link: str,
) -> tuple[pd.Series, pd.DataFrame, list[str], dict[str, float | str | int]]:
    x_train = train.loc[:, ORDINAL_FEATURE_COLUMNS].astype(float)
    x_test = test.loc[:, ORDINAL_FEATURE_COLUMNS].astype(float)
    x_train_scaled, x_test_scaled = _standardize_by_train(x_train, x_test)
    fitted, warning_messages = _fit_ordered_model(
        pd.concat([train[["policy_action"]], x_train_scaled], axis=1),
        link=link,
    )
    diagnostics = _ordered_model_parameter_diagnostics(
        fitted,
        link=link,
        target_quarter=str(test.iloc[0]["target_quarter"]),
        train_rows=len(train),
    )
    probabilities = _probability_frame(
        np.asarray(fitted.model.predict(fitted.params, exog=x_test_scaled)),
        index=test.index,
    )
    predicted_codes = probabilities.loc[:, PROBABILITY_COLUMNS].to_numpy().argmax(axis=1)
    labels = [ACTION_ORDER[int(code)] for code in predicted_codes]
    return (
        pd.Series(
            pd.Categorical(labels, categories=ACTION_ORDER, ordered=True),
            index=test.index,
        ),
        probabilities,
        warning_messages,
        diagnostics,
    )


def _require_xgboost() -> type:
    if XGBClassifier is None:
        raise ImportError(
            "Frank-Hall XGBoost requires the optional xgboost dependency. "
            "Install project requirements before running the full RBA classifier evaluation."
        )
    return XGBClassifier


def _frank_hall_probabilities_to_actions(
    p_y_gt_cut: np.ndarray,
    p_y_gt_hold: np.ndarray,
    index: pd.Index,
) -> pd.Series:
    p_gt_cut = np.clip(np.asarray(p_y_gt_cut, dtype=float), 0.0, 1.0)
    p_gt_hold = np.minimum(np.clip(np.asarray(p_y_gt_hold, dtype=float), 0.0, 1.0), p_gt_cut)
    probabilities = np.column_stack(
        [
            1.0 - p_gt_cut,
            p_gt_cut - p_gt_hold,
            p_gt_hold,
        ]
    )
    labels = [ACTION_ORDER[int(code)] for code in probabilities.argmax(axis=1)]
    return pd.Series(
        pd.Categorical(labels, categories=ACTION_ORDER, ordered=True),
        index=index,
    )


def _predict_frank_hall_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    classifier_cls = _require_xgboost()
    train_codes = train["policy_action"].astype(ACTION_DTYPE).cat.codes
    x_train = train.loc[:, XGBOOST_FEATURE_COLUMNS].astype(float)
    x_test = test.loc[:, XGBOOST_FEATURE_COLUMNS].astype(float)

    probabilities = []
    for threshold_code in (0, 1):
        binary_target = train_codes.gt(threshold_code).astype(int)
        if binary_target.nunique() != 2:
            raise ValueError(
                f"Frank-Hall XGBoost fold has only one outcome for Y>{ACTION_ORDER[threshold_code]}."
            )
        model = classifier_cls(**XGBOOST_PARAMS)
        model.fit(x_train, binary_target)
        probabilities.append(model.predict_proba(x_test)[:, 1])
    return _frank_hall_probabilities_to_actions(
        probabilities[0],
        probabilities[1],
        index=test.index,
    )


def predict_single_quarter(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    r_star: float | None = None,
    threshold_simulation_frame: pd.DataFrame | None = None,
    threshold_n_sims: int = DEFAULT_N_SIMS,
    threshold_seed: int = DEFAULT_SEED,
    sarima_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Return all policy-classifier predictions for one target quarter."""
    if len(test) != 1:
        raise ValueError("predict_single_quarter expects exactly one test row.")
    if r_star is None:
        r_star = calibrate_taylor_r_star(train)
    if threshold_simulation_frame is None:
        threshold_simulation_frame = _load_threshold_simulation_frame()
    if sarima_series is None:
        sarima_series = load_target_series(CURATED_DATA_PATH, target_column=TARGET_COLUMN)

    train = train.copy()
    test = test.copy()
    for frame in (train, test):
        frame[TAYLOR_IMPLIED_CHANGE_COLUMN] = taylor_rule_implied_change(
            frame["headline_forecast"],
            frame["unemployment_rate_change_lag1"],
            frame["cash_rate_lag1"],
            r_star=r_star,
        ).to_numpy()

    quarter = str(test.iloc[0]["target_quarter"])
    actual_action = (
        str(test.iloc[0]["policy_action"])
        if "policy_action" in test.columns and pd.notna(test.iloc[0]["policy_action"])
        else None
    )
    threshold_prediction = threshold_baseline_predict(test["headline_forecast"]).reset_index(
        drop=True
    )
    threshold_probabilities, threshold_diagnostics = _predict_threshold_confidence(
        test,
        threshold_simulation_frame,
        sarima_series=sarima_series,
        n_sims=threshold_n_sims,
        seed=threshold_seed,
        weights=horizon_rmse_weights(horizons=(1,)),
    )
    (
        estimated_taylor_prediction,
        estimated_taylor_probabilities,
        estimated_taylor_diagnostics,
    ) = _predict_estimated_taylor_rule(train, test)
    (
        logit_prediction,
        logit_probabilities,
        logit_warnings,
        logit_diagnostics,
    ) = _predict_ordered_model(
        train,
        test,
        link="logit",
    )
    (
        probit_prediction,
        probit_probabilities,
        probit_warnings,
        probit_diagnostics,
    ) = _predict_ordered_model(
        train,
        test,
        link="probit",
    )
    model_predictions = {
        "threshold": threshold_prediction,
        "taylor_rule": taylor_rule_baseline_predict(
            test["headline_forecast"],
            test["unemployment_rate_change_lag1"],
            test["cash_rate_lag1"],
            r_star=r_star,
        ).reset_index(drop=True),
        "taylor_rule_estimated": estimated_taylor_prediction.reset_index(drop=True),
        "ordered_logit": logit_prediction.reset_index(drop=True),
        "ordered_probit": probit_prediction.reset_index(drop=True),
        "frank_hall_xgboost": _predict_frank_hall_xgboost(train, test).reset_index(drop=True),
    }
    ensemble_prediction, ensemble_tie_break = majority_vote_ensemble_predict(
        {
            model: model_predictions[model]
            for model in MAJORITY_VOTE_ENSEMBLE_MODELS
        }
    )
    model_predictions["majority_vote_ensemble"] = ensemble_prediction.reset_index(drop=True)
    warning_by_model = {
        "ordered_logit": "; ".join(logit_warnings),
        "ordered_probit": "; ".join(probit_warnings),
    }
    ordered_diagnostics_by_model = {
        "ordered_logit": logit_diagnostics,
        "ordered_probit": probit_diagnostics,
    }
    probabilities_by_model = {
        "threshold": threshold_probabilities.reset_index(drop=True),
        "taylor_rule_estimated": estimated_taylor_probabilities.reset_index(drop=True),
        "ordered_logit": logit_probabilities.reset_index(drop=True),
        "ordered_probit": probit_probabilities.reset_index(drop=True),
    }

    rows = []
    for model, prediction in model_predictions.items():
        ordered_diagnostics = ordered_diagnostics_by_model.get(model, {})
        probabilities = probabilities_by_model.get(model)
        probability_values = (
            probabilities.iloc[0].to_dict()
            if probabilities is not None
            else {column: np.nan for column in PROBABILITY_COLUMNS}
        )
        predicted_action = str(prediction.iloc[0])
        rows.append(
            {
                "model": model,
                "target_quarter": quarter,
                "actual_action": actual_action,
                "predicted_action": predicted_action,
                **probability_values,
                "confidence": (
                    float(probability_values[f"p_{predicted_action}"])
                    if model in CONFIDENCE_MODELS
                    else np.nan
                ),
                "majority_vote_tie_break": (
                    bool(ensemble_tie_break.iloc[0])
                    if model == "majority_vote_ensemble"
                    else False
                ),
                "headline_forecast": float(test.iloc[0]["headline_forecast"]),
                "trimmed_mean_forecast": float(test.iloc[0]["trimmed_mean_forecast"]),
                "unemployment_rate_change_lag1": float(
                    test.iloc[0]["unemployment_rate_change_lag1"]
                ),
                TAYLOR_IMPLIED_CHANGE_COLUMN: float(
                    test.iloc[0][TAYLOR_IMPLIED_CHANGE_COLUMN]
                ),
                "fit_warnings": warning_by_model.get(model, ""),
                "estimated_taylor_intercept": (
                    estimated_taylor_diagnostics["intercept"]
                    if model == "taylor_rule_estimated"
                    else np.nan
                ),
                "estimated_taylor_coef_inflation_gap": (
                    estimated_taylor_diagnostics["coef_inflation_gap"]
                    if model == "taylor_rule_estimated"
                    else np.nan
                ),
                "estimated_taylor_coef_unemployment_change": (
                    estimated_taylor_diagnostics["coef_unemployment_rate_change_lag1"]
                    if model == "taylor_rule_estimated"
                    else np.nan
                ),
                "estimated_taylor_predicted_change": (
                    estimated_taylor_diagnostics["predicted_change"]
                    if model == "taylor_rule_estimated"
                    else np.nan
                ),
                "estimated_taylor_unstable_coefficients": (
                    estimated_taylor_diagnostics["unstable_coefficients"]
                    if model == "taylor_rule_estimated"
                    else ""
                ),
                "threshold_forecast_origin": (
                    threshold_diagnostics["threshold_forecast_origin"]
                    if model == "threshold"
                    else ""
                ),
                "threshold_simulated_median": (
                    threshold_diagnostics["threshold_simulated_median"]
                    if model == "threshold"
                    else np.nan
                ),
                "threshold_simulated_action": (
                    threshold_diagnostics["threshold_simulated_action"]
                    if model == "threshold"
                    else ""
                ),
                **{
                    f"ordered_beta_{feature}": ordered_diagnostics.get(
                        f"ordered_beta_{feature}",
                        np.nan,
                    )
                    for feature in ORDINAL_FEATURE_COLUMNS
                },
                "ordered_alpha_cut_hold": ordered_diagnostics.get(
                    "ordered_alpha_cut_hold",
                    np.nan,
                ),
                "ordered_alpha_hold_hike": ordered_diagnostics.get(
                    "ordered_alpha_hold_hike",
                    np.nan,
                ),
                "ordered_train_rows": ordered_diagnostics.get("train_rows", np.nan),
            }
        )
    return pd.DataFrame(rows)


def walk_forward_predictions(
    sample: pd.DataFrame,
    initial_train_size: int,
    *,
    r_star: float | None = None,
    threshold_simulation_frame: pd.DataFrame | None = None,
    threshold_n_sims: int = DEFAULT_N_SIMS,
    threshold_seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Evaluate all policy classifiers on identical expanding-window test rows."""
    rows = []
    if r_star is None:
        r_star = calibrate_taylor_r_star(sample)
    if threshold_simulation_frame is None:
        threshold_simulation_frame = _load_threshold_simulation_frame()
    threshold_sarima_series = load_target_series(CURATED_DATA_PATH, target_column=TARGET_COLUMN)
    for fold_position, (train_index, test_index) in enumerate(
        expanding_walk_forward_splits(sample, initial_train_size)
    ):
        fold_rows = predict_single_quarter(
            sample.iloc[train_index],
            sample.iloc[test_index],
            r_star=r_star,
            threshold_simulation_frame=threshold_simulation_frame,
            threshold_n_sims=threshold_n_sims,
            threshold_seed=threshold_seed + fold_position,
            sarima_series=threshold_sarima_series,
        )
        rows.extend(fold_rows.to_dict("records"))
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
    order_lookup = {model: position for position, model in enumerate(MODEL_ORDER)}
    return (
        pd.DataFrame(rows)
        .assign(_model_order=lambda frame: frame["model"].map(order_lookup))
        .sort_values(["macro_f1", "_model_order"], ascending=[False, True])
        .drop(columns="_model_order")
        .reset_index(drop=True)
    )


def _markdown_table(frame: pd.DataFrame, float_format: str = ".3f") -> str:
    return frame.to_markdown(index=False, floatfmt=float_format)


def _markdown_confusion_matrices(predictions: pd.DataFrame) -> str:
    sections = []
    for model in MODEL_ORDER:
        rows = predictions.loc[predictions["model"].eq(model)]
        matrix = confusion_matrix_frame(rows["actual_action"], rows["predicted_action"])
        sections.append(f"### {model}\n\n{matrix.reset_index().to_markdown(index=False)}")
    return "\n\n".join(sections)


def prediction_confidence_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return report-ready confidence rows for models with class probabilities."""
    required_columns = {
        "target_quarter",
        "model",
        "predicted_action",
        *PROBABILITY_COLUMNS,
        "confidence",
    }
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        raise ValueError(
            "Prediction confidence table is missing columns: "
            f"{', '.join(missing_columns)}."
        )

    rows = predictions.loc[
        predictions["model"].isin(CONFIDENCE_MODELS),
        [
            "target_quarter",
            "model",
            "predicted_action",
            *PROBABILITY_COLUMNS,
            "confidence",
        ],
    ].copy()
    if rows.loc[:, [*PROBABILITY_COLUMNS, "confidence"]].isna().any().any():
        raise ValueError("Prediction confidence rows contain missing probabilities.")
    if not np.allclose(rows.loc[:, PROBABILITY_COLUMNS].sum(axis=1), 1.0):
        raise ValueError("Prediction confidence probabilities do not sum to 1.")

    expected_confidence = [
        row[f"p_{row['predicted_action']}"]
        for row in rows.to_dict(orient="records")
    ]
    if not np.allclose(rows["confidence"].to_numpy(dtype=float), expected_confidence):
        raise ValueError("Prediction confidence does not match the predicted action probability.")

    model_order = {model: position for position, model in enumerate(CONFIDENCE_MODELS)}
    return (
        rows.assign(_model_order=lambda frame: frame["model"].map(model_order))
        .sort_values(["target_quarter", "_model_order"])
        .drop(columns="_model_order")
        .rename(
            columns={
                "p_cut": "P(cut)",
                "p_hold": "P(hold)",
                "p_hike": "P(hike)",
            }
        )
        .reset_index(drop=True)
    )


def threshold_confidence_consistency_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return threshold rows whose point-forecast label disagrees with simulation median."""
    required_columns = {
        "target_quarter",
        "predicted_action",
        "headline_forecast",
        "threshold_forecast_origin",
        "threshold_simulated_median",
        "threshold_simulated_action",
        *PROBABILITY_COLUMNS,
    }
    threshold_rows = predictions.loc[predictions["model"].eq("threshold")].copy()
    missing_columns = sorted(required_columns.difference(threshold_rows.columns))
    if missing_columns:
        raise ValueError(
            "Threshold consistency table is missing columns: "
            f"{', '.join(missing_columns)}."
        )

    flagged = threshold_rows.loc[
        threshold_rows["predicted_action"].astype(str).ne(
            threshold_rows["threshold_simulated_action"].astype(str)
        ),
        [
            "target_quarter",
            "threshold_forecast_origin",
            "headline_forecast",
            "predicted_action",
            "threshold_simulated_median",
            "threshold_simulated_action",
            *PROBABILITY_COLUMNS,
        ],
    ].copy()
    if flagged.empty:
        return pd.DataFrame(
            columns=[
                "target_quarter",
                "forecast_origin",
                "point_forecast",
                "point_forecast_action",
                "simulated_median",
                "simulated_median_action",
                "P(cut)",
                "P(hold)",
                "P(hike)",
                "disagreement",
            ]
        )

    flagged["disagreement"] = (
        flagged["predicted_action"].astype(str)
        + " from point forecast vs "
        + flagged["threshold_simulated_action"].astype(str)
        + " from simulated median"
    )
    return (
        flagged.rename(
            columns={
                "threshold_forecast_origin": "forecast_origin",
                "headline_forecast": "point_forecast",
                "predicted_action": "point_forecast_action",
                "threshold_simulated_median": "simulated_median",
                "threshold_simulated_action": "simulated_median_action",
                "p_cut": "P(cut)",
                "p_hold": "P(hold)",
                "p_hike": "P(hike)",
            }
        )
        .reset_index(drop=True)
    )


def _format_model_list(models: list[str]) -> str:
    if not models:
        return ""
    if len(models) == 1:
        return models[0]
    if len(models) == 2:
        return f"{models[0]} and {models[1]}"
    return f"{', '.join(models[:-1])}, and {models[-1]}"


def build_evaluation_report(
    sample: pd.DataFrame,
    split_choice: SplitChoice,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    r_star: float,
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
    threshold_input_frame, threshold_input_disagreements = threshold_input_sensitivity(test)
    display_audit = split_choice.audit.copy()
    display_metrics = metrics.copy()
    display_metrics["macro_f1"] = display_metrics["macro_f1"].round(3)
    display_metrics["accuracy"] = display_metrics["accuracy"].round(3)
    active_metric_lookup = metrics.set_index("model")
    reverted_rows = []
    for model, tested in REVERTED_ORDERED_FIVE_FEATURE_RESULTS.items():
        active_macro_f1 = float(active_metric_lookup.loc[model, "macro_f1"])
        reverted_rows.append(
            {
                "model": model,
                "active_3_feature_macro_f1": round(active_macro_f1, 3),
                "tested_5_feature_macro_f1": tested["tested_macro_f1"],
                "macro_f1_change": round(tested["tested_macro_f1"] - active_macro_f1, 3),
                "tested_5_feature_accuracy": tested["tested_accuracy"],
                "threshold_minus_tested_gap": tested["threshold_minus_tested_gap"],
                "tested_paired_bootstrap_95pct_ci": tested["paired_bootstrap_95pct_ci"],
                "status": "tested_once_and_reverted",
            }
        )
    reverted_ordered_frame = pd.DataFrame(reverted_rows)
    ordered_warning_rows = (
        predictions.loc[
            predictions.get("fit_warnings", pd.Series("", index=predictions.index))
            .fillna("")
            .ne("")
            & predictions["model"].isin(["ordered_logit", "ordered_probit"]),
            ["model", "target_quarter", "fit_warnings"],
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    ordered_warning_text = (
        _markdown_table(ordered_warning_rows)
        if not ordered_warning_rows.empty
        else "No OrderedModel ConvergenceWarning was captured during the walk-forward run."
    )
    ordered_beta_columns = [f"ordered_beta_{feature}" for feature in ORDINAL_FEATURE_COLUMNS]
    ordered_parameter_rows = predictions.loc[
        predictions["model"].isin(["ordered_logit", "ordered_probit"]),
        [
            "model",
            "target_quarter",
            "ordered_train_rows",
            *ordered_beta_columns,
            "ordered_alpha_cut_hold",
            "ordered_alpha_hold_hike",
        ],
    ].copy()
    ordered_parameter_rows = ordered_parameter_rows.rename(
        columns={
            "ordered_train_rows": "train_rows",
            **{
                f"ordered_beta_{feature}": f"beta_{feature}"
                for feature in ORDINAL_FEATURE_COLUMNS
            },
            "ordered_alpha_cut_hold": "alpha_cut_hold",
            "ordered_alpha_hold_hike": "alpha_hold_hike",
        }
    )
    final_ordered_parameters = (
        ordered_parameter_rows.groupby("model", sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    final_ordered_parameters["train_rows"] = final_ordered_parameters["train_rows"].astype(int)
    alpha_comparison_rows = []
    for model, group in ordered_parameter_rows.groupby("model", sort=False):
        ordered_group = group.sort_values("target_quarter")
        first = ordered_group.iloc[0]
        last = ordered_group.iloc[-1]
        alpha_comparison_rows.append(
            {
                "model": model,
                "first_test_quarter": first["target_quarter"],
                "last_test_quarter": last["target_quarter"],
                "first_alpha_cut_hold": first["alpha_cut_hold"],
                "last_alpha_cut_hold": last["alpha_cut_hold"],
                "change_alpha_cut_hold": last["alpha_cut_hold"] - first["alpha_cut_hold"],
                "range_alpha_cut_hold": (
                    ordered_group["alpha_cut_hold"].max() - ordered_group["alpha_cut_hold"].min()
                ),
                "first_alpha_hold_hike": first["alpha_hold_hike"],
                "last_alpha_hold_hike": last["alpha_hold_hike"],
                "change_alpha_hold_hike": last["alpha_hold_hike"] - first["alpha_hold_hike"],
                "range_alpha_hold_hike": (
                    ordered_group["alpha_hold_hike"].max() - ordered_group["alpha_hold_hike"].min()
                ),
            }
        )
    alpha_comparison_frame = pd.DataFrame(alpha_comparison_rows)
    estimated_rows = predictions.loc[
        predictions["model"].eq("taylor_rule_estimated"),
        [
            "target_quarter",
            "estimated_taylor_intercept",
            "estimated_taylor_coef_inflation_gap",
            "estimated_taylor_coef_unemployment_change",
            "estimated_taylor_predicted_change",
            "estimated_taylor_unstable_coefficients",
        ],
    ].copy()
    estimated_display = estimated_rows.rename(
        columns={
            "estimated_taylor_intercept": "intercept",
            "estimated_taylor_coef_inflation_gap": "coef_inflation_gap",
            "estimated_taylor_coef_unemployment_change": "coef_unemployment_change_lag1",
            "estimated_taylor_predicted_change": "predicted_change",
            "estimated_taylor_unstable_coefficients": "unstable_coefficients",
        }
    )
    final_estimated_coefficients = estimated_display.tail(1).copy()
    unstable_estimated_coefficients = estimated_display.loc[
        estimated_display["unstable_coefficients"].fillna("").ne("")
    ].copy()
    unstable_estimated_text = (
        _markdown_table(unstable_estimated_coefficients)
        if not unstable_estimated_coefficients.empty
        else (
            "No estimated Taylor-rule fold had |coef_inflation_gap| or "
            f"|coef_unemployment_change_lag1| above "
            f"{TAYLOR_ESTIMATED_COEFFICIENT_MAGNITUDE_WARNING:.1f}."
        )
    )
    tie_break_count = int(
        predictions.loc[
            predictions["model"].eq("majority_vote_ensemble"),
            "majority_vote_tie_break",
        ]
        .fillna(False)
        .sum()
    )
    confidence_display = prediction_confidence_table(predictions)
    threshold_consistency_flags = threshold_confidence_consistency_table(predictions)
    threshold_consistency_text = (
        _markdown_table(threshold_consistency_flags)
        if not threshold_consistency_flags.empty
        else "No threshold hard-label/simulated-median disagreements were flagged."
    )

    by_quarter = predictions.pivot(
        index="target_quarter",
        columns="model",
        values="predicted_action",
    )
    actual_by_quarter = predictions.drop_duplicates("target_quarter").set_index(
        "target_quarter"
    )["actual_action"]
    paired = by_quarter.join(actual_by_quarter).dropna(
        subset=[*MODEL_ORDER, "actual_action"]
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
    unsettled_comparators = []
    threshold_positive_comparators = []
    candidate_positive_comparators = []
    index_positions = np.arange(len(paired))
    for comparator in [model for model in MODEL_ORDER if model != "threshold"]:
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
        rounded_lower = round(float(lower), 3)
        rounded_upper = round(float(upper), 3)
        if rounded_lower <= 0.0 <= rounded_upper:
            unsettled_comparators.append(comparator)
        elif rounded_lower > 0.0:
            threshold_positive_comparators.append(comparator)
        else:
            candidate_positive_comparators.append(comparator)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)

    threshold_matrix = confusion_matrix_frame(
        predictions.loc[predictions["model"].eq("threshold"), "actual_action"],
        predictions.loc[predictions["model"].eq("threshold"), "predicted_action"],
    )
    estimated_taylor_matrix = confusion_matrix_frame(
        predictions.loc[predictions["model"].eq("taylor_rule_estimated"), "actual_action"],
        predictions.loc[predictions["model"].eq("taylor_rule_estimated"), "predicted_action"],
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
    estimated_taylor_hike_recall = int(estimated_taylor_matrix.loc["hike", "hike"])
    estimated_taylor_hike_misses = hike_total - estimated_taylor_hike_recall
    logit_hikes_as_hold = int(logit_matrix.loc["hike", "hold"])
    probit_hikes_as_hold = int(probit_matrix.loc["hike", "hold"])
    ranking_text = ", ".join(
        f"{row.model} {row.macro_f1:.3f}" for row in display_metrics.itertuples(index=False)
    )
    best_macro_f1 = float(metrics["macro_f1"].max())
    best_models = [
        str(row.model)
        for row in metrics.itertuples(index=False)
        if np.isclose(float(row.macro_f1), best_macro_f1)
    ]
    threshold_wins_or_ties = "threshold" in best_models
    fixed_taylor_macro_f1 = float(active_metric_lookup.loc["taylor_rule", "macro_f1"])
    estimated_taylor_macro_f1 = float(
        active_metric_lookup.loc["taylor_rule_estimated", "macro_f1"]
    )
    majority_macro_f1 = float(active_metric_lookup.loc["majority_vote_ensemble", "macro_f1"])
    estimated_vs_fixed = (
        "The estimated Taylor rule beats the fixed Taylor rule on macro-F1 "
        f"({estimated_taylor_macro_f1:.3f} vs {fixed_taylor_macro_f1:.3f})."
        if estimated_taylor_macro_f1 > fixed_taylor_macro_f1
        else (
            "The estimated Taylor rule does not beat the fixed Taylor rule on macro-F1 "
            f"({estimated_taylor_macro_f1:.3f} vs {fixed_taylor_macro_f1:.3f})."
        )
    )
    threshold_vs_majority = bootstrap_frame.loc[
        bootstrap_frame["gap"].eq("threshold - majority_vote_ensemble")
    ]
    majority_note = (
        f"The majority-vote ensemble records macro-F1 {majority_macro_f1:.3f}; "
        f"threshold minus majority_vote_ensemble is "
        f"{threshold_vs_majority.iloc[0]['observed_macro_f1_gap']:.3f} with paired-bootstrap "
        f"95% CI {threshold_vs_majority.iloc[0]['paired_bootstrap_95pct_ci']}."
    )
    if threshold_wins_or_ties:
        tied_with_threshold = [model for model in best_models if model != "threshold"]
        threshold_position = (
            "the highest macro-F1 point estimate"
            if best_models == ["threshold"]
            else (
                "a tied-highest macro-F1 point estimate with "
                f"{_format_model_list(tied_with_threshold)}"
            )
        )
        conclusion_parts = [
            f"The threshold baseline remains the preferred reportable result: it has {threshold_position}."
        ]
        if unsettled_comparators:
            conclusion_parts.append(
                f"Its lead over {_format_model_list(unsettled_comparators)} remains "
                "statistically unsettled because those paired-bootstrap intervals touch or "
                "cross zero."
            )
        if threshold_positive_comparators:
            conclusion_parts.append(
                f"Its gaps over {_format_model_list(threshold_positive_comparators)} are "
                "strictly positive across the paired-bootstrap 95% intervals."
            )
        if candidate_positive_comparators:
            conclusion_parts.append(
                f"The bootstrap intervals favor {_format_model_list(candidate_positive_comparators)} "
                "over threshold despite the aggregate ranking, so that tension should be treated "
                "as a materiality caveat."
            )
        conclusion_parts.append(majority_note)
        conclusion_parts.append(estimated_vs_fixed)
        conclusion = " ".join(conclusion_parts)
    else:
        best_model = str(display_metrics.iloc[0]["model"])
        best_gap = bootstrap_frame.loc[bootstrap_frame["gap"].eq(f"threshold - {best_model}")]
        bootstrap_note = (
            f" The paired-bootstrap threshold-minus-{best_model} interval is "
            f"{best_gap.iloc[0]['paired_bootstrap_95pct_ci']}."
            if not best_gap.empty
            else ""
        )
        conclusion = (
            f"The preferred reportable result changes on point estimates: "
            f"{display_metrics.iloc[0]['model']} has the highest macro-F1. The "
            "paired-bootstrap interval against the threshold baseline should be treated "
            f"as the materiality check before making a stronger claim.{bootstrap_note}"
        )

    return "\n".join(
        [
            "# RBA Policy Action Classifier Evaluation",
            "",
            "## Scope",
            "",
            (
                "Uses the leakage-safe Ensemble horizon-1 headline and trimmed-mean CPI "
                "forecasts from the existing backtest prediction reports, plus the "
                "curated lag-safe unemployment-rate change feature."
            ),
            "",
            (
                "Feature sets are fixed before evaluation: the threshold baseline uses "
                "headline_forecast only; the Taylor-rule baseline uses headline_forecast "
                "and unemployment_rate_change_lag1 only; the estimated Taylor-rule model "
                "fits cash_rate_change on headline inflation_gap and "
                "unemployment_rate_change_lag1; ordered logit and ordered probit use "
                "headline_forecast, trimmed_mean_forecast, and "
                "unemployment_rate_change_lag1; Frank-Hall XGBoost uses those three plus "
                "taylor_implied_change."
            ),
            "",
            (
                "The majority_vote_ensemble is not a fitted model. It takes the same "
                "quarter's already-computed threshold, taylor_rule_estimated, ordered_logit, "
                "and ordered_probit predictions and applies a plain majority vote. The "
                "fixed Taylor rule and Frank-Hall XGBoost are deliberately excluded because "
                "the existing report already established them as clearly weaker. Any 2-2 "
                "split is resolved by the threshold prediction before inspecting tie frequency "
                "or performance."
            ),
            "",
            (
                "The Taylor-rule baseline is a deliberate simplification of a textbook "
                "Taylor rule: unemployment_rate_change_lag1 is used as an Okun's-law-style "
                "change term instead of an unemployment gap, because the project does not "
                "estimate NAIRU or potential output. The rule uses fixed 0.5/0.5 Taylor "
                "coefficients, the RBA 2.5% target-band midpoint, and the joined sample's "
                f"mean real policy rate, cash_rate - headline_forecast, as r_star "
                f"({r_star:.3f}); cash_rate_lag1 is used only as the status-quo rate for "
                "discretizing the implied change. The negative r_star reflects this "
                "sample's average real-rate proxy rather than a nominal cash-rate mean."
            ),
            "",
            (
                "The estimated Taylor-rule model is refit separately on each expanding "
                "training fold by OLS with an intercept: cash_rate_change is regressed on "
                "inflation_gap = headline_forecast - 2.5 and "
                "unemployment_rate_change_lag1. Its predicted continuous change is "
                f"discretized with the fixed +/-{TAYLOR_ESTIMATED_HOLD_BAND:.3f} percentage "
                "point hold band."
            ),
            "",
            (
                "Frank-Hall XGBoost is fit as two binary classifiers, P(Y > cut) and "
                "P(Y > hold). Unlike ordered logit and probit, it additionally receives "
                "the continuous taylor_implied_change feature computed from the same fixed "
                "Taylor-rule calibration. XGBoost-vs-ordered-model comparisons are therefore "
                "not clean like-for-like feature comparisons. Its conservative hyperparameters "
                "are chosen for the small sample: max_depth=2, n_estimators=50, "
                "min_child_weight=3, subsample=0.75, colsample_bytree=0.75, reg_alpha=0.1, "
                "and reg_lambda=2.0. These settings are not searched or tuned."
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
            "## Why The Threshold Rule Uses Headline CPI",
            "",
            (
                "The threshold baseline applies the RBA's published 2-3% target band to the "
                "headline CPI forecast, because that band is a target for CPI inflation. "
                "Trimmed mean is the underlying-inflation measure, and it does enter the "
                "ordered logit, ordered probit and Frank-Hall models as a feature. As a "
                "sensitivity check, the same rule was also applied to the trimmed-mean "
                "forecast over the same test quarters:"
            ),
            "",
            _markdown_table(threshold_input_frame),
            "",
            (
                f"The two inputs give different calls in {threshold_input_disagreements} of "
                f"{len(test)} test quarters; compare each input's predicted-action counts "
                "with the actual test counts above to see where the calls differ. This "
                "check was run after the model design was "
                "fixed and on the same test quarters as every other comparison, so treat it "
                "as a sensitivity check rather than a model-selection step; the gap between "
                "the two inputs has no bootstrap interval."
            ),
            "",
            "## Prediction Confidence",
            "",
            (
                "Confidence is the probability assigned to the predicted action by that "
                "model's own class-probability calculation."
            ),
            "",
            (
                "Threshold probabilities use fresh horizon-1 Ensemble CPI simulation "
                f"draws for each threshold test quarter (n_sims={DEFAULT_N_SIMS}, seed={DEFAULT_SEED} "
                "with one deterministic increment per fold)."
            ),
            "",
            (
                'This "confidence" measures how far a prediction sits from the model\'s '
                "own decision boundary, not a validated track record of being correct; "
                "a model can be confidently wrong."
            ),
            "",
            _markdown_table(confidence_display),
            "",
            "Threshold hard-label/simulated-median consistency:",
            "",
            threshold_consistency_text,
            "",
            "## Majority Vote Ensemble Audit",
            "",
            (
                "The ensemble combines exactly four existing per-fold predictions: "
                f"{', '.join(MAJORITY_VOTE_ENSEMBLE_MODELS)}. It estimates no parameters "
                "and uses no separate train/test split."
            ),
            "",
            (
                f"Tie-break cases using the threshold prediction: {tie_break_count} of "
                f"{len(test)} test quarters."
            ),
            "",
            "## Tested And Reverted Ordered Feature Variant",
            "",
            (
                "A single pre-committed 5-feature ordered-model variant added cash_rate_lag1 "
                "and commodity_growth_lag1 to the active 3-feature ordered set. It was "
                "reverted because it worsened both ordered models on the same walk-forward "
                "test window; the result is retained here as a tested-and-rejected variant, "
                "not erased."
            ),
            "",
            (
                "Rejected feature set: "
                f"{', '.join(REVERTED_ORDERED_FIVE_FEATURE_COLUMNS)}."
            ),
            "",
            _markdown_table(reverted_ordered_frame),
            "",
            "## Estimated Taylor Coefficient Audit",
            "",
            "Final walk-forward fold coefficients:",
            "",
            _markdown_table(final_estimated_coefficients),
            "",
            "Unstable coefficient flags:",
            "",
            unstable_estimated_text,
            "",
            "## Ordered Model Interpretability",
            "",
            (
                "Ordered logit and probit are refit at every walk-forward step, so there "
                "is no single coefficient vector for the whole exercise. The table below "
                "reports the final fold, which uses the most training data. Betas are on "
                "the train-standardized feature scale; alphas are the transformed latent "
                "cutoffs between cut/hold and hold/hike."
            ),
            "",
            _markdown_table(final_ordered_parameters),
            "",
            (
                "The first-vs-last alpha comparison checks whether the thin initial folds "
                "produce visibly different cutoffs from the final fold."
            ),
            "",
            _markdown_table(alpha_comparison_frame),
            "",
            "## OrderedModel Convergence Warnings",
            "",
            ordered_warning_text,
            "",
            "## Small-Sample Caveat",
            "",
            (
                f"The walk-forward test set has {len(test)} quarters. At this size, the "
                f"macro-F1 ranking ({ranking_text}) is suggestive rather than decisive."
            ),
            "",
            (
                "As a quick uncertainty check, a paired bootstrap over the same test "
                "quarters (2,000 resamples, seed=42) gives the following macro-F1 gap "
                "intervals for threshold minus each candidate. This is a rough diagnostic "
                "because it treats quarters as exchangeable and does not model time-series "
                "dependence."
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
            conclusion,
            "",
            (
                "When the threshold rule is preferred, the non-statistical reason remains "
                "its transparency and hike detection: it "
                f"correctly classifies all {threshold_hike_recall} of {hike_total} hike "
                f"quarters, while taylor_rule_estimated correctly classifies "
                f"{estimated_taylor_hike_recall} of {hike_total} and misses "
                f"{estimated_taylor_hike_misses}; ordered logit classifies "
                f"{logit_hikes_as_hold} hikes as hold and ordered probit classifies "
                f"{probit_hikes_as_hold} hikes as hold. That hike-to-hold error is a "
                "structurally different, and arguably more consequential, failure mode "
                "for a policy classifier than the aggregate macro-F1 gap alone captures."
            ),
            "",
            (
                "A plausible reading remains that the CPI forecast features are highly "
                "correlated and the earliest valid training folds are thin. The expanded "
                "exercise should therefore remain a documented classifier comparison for "
                "the materiality discussion rather than an API-exposed policy predictor."
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
    r_star = calibrate_taylor_r_star(sample)
    predictions = walk_forward_predictions(
        sample,
        split_choice.initial_train_size,
        r_star=r_star,
    )
    metrics = compute_policy_metrics(predictions)
    report = build_evaluation_report(sample, split_choice, predictions, metrics, r_star)
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
    for model in MODEL_ORDER:
        rows = predictions.loc[predictions["model"].eq(model)]
        print()
        print(model)
        print(confusion_matrix_frame(rows["actual_action"], rows["predicted_action"]).to_string())


if __name__ == "__main__":
    main()
