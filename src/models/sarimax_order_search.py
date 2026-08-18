"""AIC/BIC-only SARIMAX order search and screened-feature ablation reports.

Candidate SARIMAX specifications are ranked by one in-sample fit per fixed
feature group. Walk-forward RMSE/MAE are reported only after the ARIMA order
and feature group have already been fixed by the EDA-screened design.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.evaluation import (
    CURATED_DATA_PATH,
    DEFAULT_HORIZONS,
    DEFAULT_INITIAL_TRAIN_SIZE,
    PROJECT_ROOT,
    RBA_FORECAST_PATH,
    compute_metric_table,
    infer_min_lag_from_columns,
    load_target_series,
    restrict_to_common_grid,
)
from src.models.sarima import DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER, forecast_sarima
from src.models.sarima_order_search import iter_candidate_orders
from src.models.sarimax import fit_sarimax, forecast_sarimax


SARIMAX_COMPARISON_OUTPUT_PATH = PROJECT_ROOT / "reports/model_comparison_sarimax.csv"
SARIMAX_COEFFICIENT_OUTPUT_PATH = PROJECT_ROOT / "reports/sarimax_coefficients.csv"

CASH_RATE_LEVEL_FEATURES = ("cash_rate_lag2",)
CASH_RATE_CHANGE_FEATURES = ("cash_rate_change_lag1",)
UNEMPLOYMENT_LEVEL_FEATURES = ("unemployment_rate_lag2",)
UNEMPLOYMENT_CHANGE_FEATURES = ("unemployment_rate_change_lag1",)
EXTENDED_FEATURES = ("wpi_growth_lag1", "aud_usd_change_lag1", "brent_growth_lag1")
# Lag suffixes reflect each dummy's real-world announcement lead time (see
# data/metadata/intervention_quarters.csv's lead_quarters column), not a
# blanket horizon. covid_shock_down had zero genuine lead (the childcare/fuel
# shock was unknowable as of the prior origin), so it caps any group that
# contains it at horizon 0 -- i.e. not forecast-viable, only in-sample
# explanatory. See _not_forecast_viable_metrics.
INTERVENTION_FEATURES = ("covid_shock_down_lag0", "covid_shock_rebound_lag1")
NESTED_GROUP_IDS = {"A", "B", "C", "D", "E", "H"}
SHARP_SAMPLE_DROP_RATIO = 0.75
LEVEL_CHANGE_AIC_MATERIALITY_THRESHOLD = 2.0
NONSTATIONARY_LEVEL_FAMILIES = {"cash_rate", "unemployment_rate"}
DEFAULT_COMPARISON_MAX_ARMA_ORDER = 1
# Order/level-vs-change selection uses only data up to this many quarters
# before the end of the sample, so the walk-forward test tail is never used
# to pick the ARIMA order or the level-vs-change feature choice (previously
# both were selected on the full sample, including every backtest origin's
# own future).
ORDER_SELECTION_HOLDOUT_QUARTERS = 20
ORDER_SELECTION_NOTE = (
    f"ARIMA order and level-vs-change feature choice were selected by AIC on "
    f"development-only data (excludes the most recent {ORDER_SELECTION_HOLDOUT_QUARTERS} "
    "quarters), never the full sample, so the walk-forward test tail was not used to pick them."
)
NOT_FORECAST_VIABLE_NOTE = (
    "This group's minimum feature lag is below 1 quarter (zero genuine forecast "
    "lead time), so no walk-forward horizon is leakage-safe; only in-sample "
    "coefficients are reported for this group."
)

COEFFICIENT_CAVEAT = (
    "Model-conditional correlation only; not a causal estimate. Interpret with "
    "the same caution as the EDA Granger screening, especially for cash_rate."
)


@dataclass(frozen=True)
class FeatureGroup:
    group_id: str
    group_name: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class LevelChangeChoice:
    family: str
    winner: str
    winner_features: tuple[str, ...]
    loser: str
    winner_aic: float
    loser_aic: float
    aic_winner: str
    aic_winner_aic: float
    aic_runner_up: str
    aic_runner_up_aic: float
    aic_gap_abs: float
    selection_note: str

    @property
    def aic_gap(self) -> float:
        return self.loser_aic - self.winner_aic


def load_exog_frame(path: Path = CURATED_DATA_PATH) -> pd.DataFrame:
    """Load curated macro features with a quarterly PeriodIndex."""
    df = pd.read_csv(path)
    if "quarter" not in df:
        raise ValueError(f"{path} must contain a 'quarter' column.")
    df.index = pd.PeriodIndex(df.pop("quarter").astype(str), freq="Q")
    return df.sort_index()


def run_sarimax_order_search(
    series: pd.Series,
    exog: pd.DataFrame,
    max_p: int = 2,
    max_q: int = 2,
    max_p_seasonal: int = 2,
    max_q_seasonal: int = 2,
    d_values: Iterable[int] = (0,),
    seasonal_d_values: Iterable[int] = (0,),
    seasonal_period: int = 4,
    trend: str = "n",
    maxiter: int = 100,
) -> pd.DataFrame:
    """Fit candidate SARIMAX orders and return AIC/BIC-ranked results."""
    rows: list[dict[str, object]] = []
    for order, seasonal_order in iter_candidate_orders(
        max_p=max_p,
        max_q=max_q,
        max_p_seasonal=max_p_seasonal,
        max_q_seasonal=max_q_seasonal,
        d_values=d_values,
        seasonal_d_values=seasonal_d_values,
        seasonal_period=seasonal_period,
    ):
        row: dict[str, object] = {
            "order": order,
            "seasonal_order": seasonal_order,
            "trend": trend,
            "converged": False,
            "aic": float("inf"),
            "bic": float("inf"),
            "error": "",
        }
        try:
            fitted = fit_sarimax(
                series=series,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                maxiter=maxiter,
            )
            row.update(
                {
                    "converged": bool(fitted.mle_retvals.get("converged", False)),
                    "aic": float(fitted.aic),
                    "bic": float(fitted.bic),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive CLI path
            row["error"] = str(exc)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["aic", "bic"], ignore_index=True)


def resolve_level_change_features(
    series: pd.Series,
    exog: pd.DataFrame,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
    trend: str = "n",
    maxiter: int = 100,
    aic_materiality_threshold: float = LEVEL_CHANGE_AIC_MATERIALITY_THRESHOLD,
) -> dict[str, LevelChangeChoice]:
    """Select level-vs-change rate families by matched same-order in-sample AIC.

    If a documented-I(1) level series only beats its stationary change transform
    by an immaterial AIC margin, prefer the change transform rather than treating
    a near-tie as evidence for the non-stationary level regressor.
    """
    comparisons = {
        "cash_rate": {
            "level": CASH_RATE_LEVEL_FEATURES,
            "change": CASH_RATE_CHANGE_FEATURES,
        },
        "unemployment_rate": {
            "level": UNEMPLOYMENT_LEVEL_FEATURES,
            "change": UNEMPLOYMENT_CHANGE_FEATURES,
        },
    }
    choices: dict[str, LevelChangeChoice] = {}
    for family, candidates in comparisons.items():
        fitted_aic: dict[str, float] = {}
        for label, features in candidates.items():
            fitted = fit_sarimax(
                series=series,
                exog=exog.loc[:, list(features)],
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                maxiter=maxiter,
            )
            fitted_aic[label] = float(fitted.aic)

        ranked = sorted(fitted_aic, key=fitted_aic.get)
        aic_winner = ranked[0]
        aic_runner_up = ranked[1]
        aic_gap_abs = fitted_aic[aic_runner_up] - fitted_aic[aic_winner]
        winner = aic_winner
        selection_note = (
            f"AIC selected {aic_winner} over {aic_runner_up} "
            f"with delta AIC {aic_gap_abs:.3f}."
        )
        if (
            aic_winner == "level"
            and family in NONSTATIONARY_LEVEL_FAMILIES
            and aic_gap_abs < aic_materiality_threshold
        ):
            winner = "change"
            selection_note = (
                f"AIC was inconclusive: level beat change by delta AIC "
                f"{aic_gap_abs:.3f}, below the {aic_materiality_threshold:.1f} "
                "materiality threshold; selected stationary change transform "
                "because the EDA classifies the level series as I(1)."
            )
        elif aic_winner == "level" and family in NONSTATIONARY_LEVEL_FAMILIES:
            selection_note += " Level series is documented I(1); interpret coefficients cautiously."
        loser = "change" if winner == "level" else "level"
        choices[family] = LevelChangeChoice(
            family=family,
            winner=winner,
            winner_features=tuple(candidates[winner]),
            loser=loser,
            winner_aic=fitted_aic[winner],
            loser_aic=fitted_aic[loser],
            aic_winner=aic_winner,
            aic_winner_aic=fitted_aic[aic_winner],
            aic_runner_up=aic_runner_up,
            aic_runner_up_aic=fitted_aic[aic_runner_up],
            aic_gap_abs=aic_gap_abs,
            selection_note=selection_note,
        )
    return choices


def build_feature_groups(choices: dict[str, LevelChangeChoice]) -> tuple[FeatureGroup, ...]:
    """Build the EDA-screened nested SARIMAX ablation groups."""
    rate_features = (
        *choices["cash_rate"].winner_features,
        *choices["unemployment_rate"].winner_features,
    )
    group_a = rate_features
    group_b = (*group_a, "inflation_expectations_business_lag1")
    group_c = (*group_b, "ppi_growth_lag2", "commodity_growth_lag1")
    group_d = (*group_c, "wti_growth_lag1")
    group_e = (*group_d, *EXTENDED_FEATURES)
    group_f = ("cash_rate_lag4", "unemployment_rate_lag4")
    group_g = (*group_b, "ppi_growth_lag1", "commodity_growth_lag1")
    group_h = (*group_e, "household_spending_growth_lag1")
    group_i = INTERVENTION_FEATURES
    group_j = (*group_d, *INTERVENTION_FEATURES)
    return (
        FeatureGroup("A", "policy rates", group_a),
        FeatureGroup("B", "A + inflation expectations", group_b),
        FeatureGroup("C", "B + producer/commodity prices", group_c),
        FeatureGroup("D", "full core", group_d),
        FeatureGroup("E", "full core + extended", group_e),
        FeatureGroup("F", "long-horizon rates only", group_f),
        FeatureGroup("G", "C variant: fresher PPI lag (ppi_growth_lag1)", group_g),
        FeatureGroup("H", "E + household spending", group_h),
        FeatureGroup("I", "COVID intervention dummies only", group_i),
        FeatureGroup("J", "D + COVID intervention dummies", group_j),
    )


def expected_sign_for_feature(feature: str) -> str:
    """Return a one-character economic-prior sign for a screened regressor."""
    if feature.startswith("cash_rate"):
        return "-"
    if feature.startswith("unemployment_rate"):
        return "-"
    if feature.startswith("inflation_expectations_business"):
        return "+"
    if feature.startswith("ppi_growth"):
        return "+"
    if feature.startswith("commodity_growth"):
        return "+"
    if feature.startswith("wti_growth"):
        return "+"
    if feature.startswith("wpi_growth"):
        return "+"
    if feature.startswith("aud_usd_change"):
        return "-"
    if feature.startswith("brent_growth"):
        return "+"
    if feature.startswith("household_spending_growth"):
        return "+"
    if feature.startswith("covid_shock_down"):
        return "-"
    if feature.startswith("covid_shock_rebound"):
        return "+"
    return "?"


def sign_check(feature: str, coefficient: float) -> tuple[str, str, str]:
    """Classify the observed coefficient sign against the economic prior."""
    expected = expected_sign_for_feature(feature)
    observed = "+" if coefficient > 0 else "-" if coefficient < 0 else "0"
    if expected == "?":
        result = "no prior"
    elif observed == "0":
        result = "near zero"
    elif observed == expected:
        result = "matches prior"
    else:
        result = "opposes prior"
    note = f"Expected {expected}; observed {observed}: {result}."
    return expected, observed, note


def coefficient_table_for_group(
    group: FeatureGroup,
    fitted,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    criterion: str,
    horizon_cap: int,
    selection_notes_by_feature: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Extract coefficient estimates and economic-prior sign checks."""
    rows: list[dict[str, object]] = []
    selection_notes_by_feature = selection_notes_by_feature or {}
    for feature in group.features:
        coefficient = float(fitted.params[feature])
        std_err = float(fitted.bse[feature])
        p_value = float(fitted.pvalues[feature])
        expected, observed, note = sign_check(feature, coefficient)
        rows.append(
            {
                "group_id": group.group_id,
                "group_name": group.group_name,
                "selected_by": criterion,
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": float(fitted.aic),
                "bic": float(fitted.bic),
                "horizon_cap": horizon_cap,
                "feature": feature,
                "coef": coefficient,
                "std_err": std_err,
                "p_value": p_value,
                "expected_sign": expected,
                "observed_sign": observed,
                "sign_check": note,
                "level_change_selection_note": selection_notes_by_feature.get(feature, ""),
                "caveat": COEFFICIENT_CAVEAT,
            }
        )
    return pd.DataFrame(rows)


def _selection_notes_by_feature(
    choices: dict[str, LevelChangeChoice],
) -> dict[str, str]:
    notes: dict[str, str] = {}
    for choice in choices.values():
        for feature in choice.winner_features:
            notes[feature] = f"{choice.family}: {choice.selection_note}"
    notes["cash_rate_lag4"] = (
        "cash_rate: separate lag-4 level-rate feature for long-horizon comparability; "
        "the EDA classifies the level series as I(1)."
    )
    notes["unemployment_rate_lag4"] = (
        "unemployment_rate: separate lag-4 level-rate feature for long-horizon comparability; "
        "the EDA classifies the level series as I(1)."
    )
    return notes


def _level_change_summary(choices: dict[str, LevelChangeChoice]) -> str:
    return " | ".join(
        f"{choice.family}: {choice.selection_note}" for choice in choices.values()
    )


def _group_level_change_summary(
    group: FeatureGroup,
    choices: dict[str, LevelChangeChoice],
) -> str:
    if group.group_id == "F":
        return (
            "Group F intentionally uses lag-4 level-rate features for long-horizon "
            "comparability; it is reported separately from the nested "
            "level-vs-change resolution, and the EDA classifies these level "
            "series as I(1)."
        )
    return _level_change_summary(choices)


def _parse_int_values(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def _horizon_range_label(horizons: Iterable[int]) -> str:
    values = sorted({int(horizon) for horizon in horizons})
    if not values:
        return "none"
    if values == list(range(values[0], values[-1] + 1)):
        return str(values[0]) if values[0] == values[-1] else f"{values[0]}-{values[-1]}"
    return ",".join(str(value) for value in values)


def _select_order(search_results: pd.DataFrame, criterion: str) -> pd.Series:
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be either 'aic' or 'bic'.")
    finite = search_results.loc[search_results[criterion].notna()]
    finite = finite.loc[finite[criterion] < float("inf")]
    if finite.empty:
        raise ValueError("No SARIMAX candidate fit successfully.")
    return finite.sort_values([criterion, "aic", "bic"]).iloc[0]


def _group_predictions(
    group: FeatureGroup,
    series: pd.Series,
    exog: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    initial_train_size: int,
    horizons: tuple[int, ...],
    maxiter: int,
) -> pd.DataFrame:
    from src.models import evaluation as model_evaluation

    return model_evaluation.walk_forward_backtest_with_exog(
        series=series,
        exog=exog.loc[:, list(group.features)],
        forecast_func=lambda train_y, train_x, future_x, steps: forecast_sarimax(
            train_y,
            train_x,
            future_x,
            steps=steps,
            order=order,
            seasonal_order=seasonal_order,
            maxiter=maxiter,
        ),
        initial_train_size=initial_train_size,
        horizons=horizons,
        model_name="sarimax",
    )


def _baseline_predictions(
    series: pd.Series,
    sarimax_predictions: pd.DataFrame,
    rba_path: Path,
    initial_train_size: int,
    horizons: tuple[int, ...],
    maxiter: int,
) -> list[pd.DataFrame]:
    from src.models import evaluation as model_evaluation

    sarima = model_evaluation.walk_forward_backtest(
        series=series,
        forecast_func=lambda train, steps: forecast_sarima(train, steps=steps, maxiter=maxiter),
        initial_train_size=initial_train_size,
        horizons=horizons,
        model_name="sarima",
    )
    naive = model_evaluation.seasonal_naive_backtest(
        series=series,
        initial_train_size=initial_train_size,
        horizons=horizons,
    )
    frames = [sarimax_predictions, sarima, naive]
    if rba_path.exists():
        rba = model_evaluation.align_rba_forecasts_to_grid(
            pd.read_csv(rba_path),
            sarimax_predictions,
            horizons=horizons,
        )
        if not rba.empty:
            frames.append(rba)
    return frames


def _development_split(
    series: pd.Series,
    exog: pd.DataFrame,
    holdout_quarters: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return the leading development-only slice used for order/feature selection.

    Excludes the most recent ``holdout_quarters`` quarters of ``series`` so
    that ARIMA order and level-vs-change choices are never picked using data
    from the tail of the walk-forward evaluation grid.
    """
    if holdout_quarters <= 0 or len(series) <= holdout_quarters:
        return series, exog.reindex(series.index)
    cutoff = series.index[-holdout_quarters]
    development_series = series.loc[series.index < cutoff]
    development_exog = exog.reindex(development_series.index)
    return development_series, development_exog


def _not_forecast_viable_metrics(
    group: FeatureGroup,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    criterion: str,
    selection_aic: float,
    selection_bic: float,
    level_change_note: str,
    horizon_cap: int,
) -> pd.DataFrame:
    """Build a metrics stub for a group whose minimum feature lag is below 1.

    Such a group has no leakage-safe walk-forward horizon (see
    ``NOT_FORECAST_VIABLE_NOTE``), so this returns one ``overall`` row with
    ``n=0`` and ``rmse``/``mae`` left as NaN rather than calling the
    walk-forward harness, which would otherwise raise on an empty horizon set.
    """
    metrics = pd.DataFrame(
        [{"model": "sarimax", "horizon": "overall", "n": 0, "rmse": float("nan"), "mae": float("nan")}]
    )
    metrics.insert(0, "group_id", group.group_id)
    metrics.insert(1, "group_name", group.group_name)
    metrics.insert(2, "horizon_range", "none")
    metrics.insert(3, "horizon_cap", horizon_cap)
    metrics.insert(4, "features", ";".join(group.features))
    metrics.insert(5, "level_change_selection_note", level_change_note)
    metrics.insert(6, "selected_order", str(order))
    metrics.insert(7, "selected_seasonal_order", str(seasonal_order))
    metrics.insert(8, "selected_by", criterion)
    metrics.insert(9, "selection_aic", selection_aic)
    metrics.insert(10, "selection_bic", selection_bic)
    metrics.insert(11, "group_origin_n", 0)
    metrics.insert(12, "group_overall_metric_n", 0)
    metrics.insert(13, "previous_nested_origin_n", None)
    metrics.insert(14, "origin_drop_from_previous_nested_group", None)
    metrics.insert(15, "sample_size_note", NOT_FORECAST_VIABLE_NOTE)
    return metrics


def run_sarimax_comparison(
    curated_path: Path = CURATED_DATA_PATH,
    rba_path: Path = RBA_FORECAST_PATH,
    comparison_output_path: Path = SARIMAX_COMPARISON_OUTPUT_PATH,
    coefficient_output_path: Path = SARIMAX_COEFFICIENT_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    criterion: str = "aic",
    max_p: int = DEFAULT_COMPARISON_MAX_ARMA_ORDER,
    max_q: int = DEFAULT_COMPARISON_MAX_ARMA_ORDER,
    max_p_seasonal: int = DEFAULT_COMPARISON_MAX_ARMA_ORDER,
    max_q_seasonal: int = DEFAULT_COMPARISON_MAX_ARMA_ORDER,
    d_values: Iterable[int] = (0,),
    seasonal_d_values: Iterable[int] = (0,),
    maxiter: int = 100,
    order_selection_holdout_quarters: int = ORDER_SELECTION_HOLDOUT_QUARTERS,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, LevelChangeChoice]]:
    """Run screened SARIMAX ablations and save metric/coefficient reports."""
    from src.models import tracking

    requested_horizons = tuple(int(horizon) for horizon in horizons)
    series = load_target_series(curated_path)
    exog = load_exog_frame(curated_path)
    development_series, development_exog = _development_split(
        series, exog, holdout_quarters=order_selection_holdout_quarters
    )
    if verbose:
        print(
            "Resolving level-vs-change rate families by same-order AIC on "
            f"development-only data (excludes the last {order_selection_holdout_quarters} "
            "quarters)...",
            flush=True,
        )
    choices = resolve_level_change_features(development_series, development_exog, maxiter=maxiter)
    groups = build_feature_groups(choices)
    selection_notes_by_feature = _selection_notes_by_feature(choices)

    metric_tables: list[pd.DataFrame] = []
    coefficient_tables: list[pd.DataFrame] = []
    tracking_payloads: list[dict[str, object]] = []
    previous_nested_origin_n: int | None = None
    for group in groups:
        if verbose:
            print(
                f"Running group {group.group_id} ({group.group_name}) with "
                f"{len(group.features)} features...",
                flush=True,
            )
        group_exog = exog.loc[:, list(group.features)]
        horizon_cap = infer_min_lag_from_columns(group.features)
        development_group_exog = development_exog.loc[:, list(group.features)]
        search_results = run_sarimax_order_search(
            series=development_series,
            exog=development_group_exog,
            max_p=max_p,
            max_q=max_q,
            max_p_seasonal=max_p_seasonal,
            max_q_seasonal=max_q_seasonal,
            d_values=d_values,
            seasonal_d_values=seasonal_d_values,
            trend="n",
            maxiter=maxiter,
        )
        best = _select_order(search_results, criterion)
        order = best["order"]
        seasonal_order = best["seasonal_order"]
        if verbose:
            print(
                f"  selected order={order}, seasonal_order={seasonal_order} by "
                f"development-only {criterion.upper()}={float(best[criterion]):.3f}; "
                f"running capped backtest to horizon {horizon_cap}",
                flush=True,
            )
        fitted = fit_sarimax(
            series=series,
            exog=group_exog,
            order=order,
            seasonal_order=seasonal_order,
            maxiter=maxiter,
        )
        coefficient_tables.append(
            coefficient_table_for_group(
                group=group,
                fitted=fitted,
                order=order,
                seasonal_order=seasonal_order,
                criterion=criterion,
                horizon_cap=horizon_cap,
                selection_notes_by_feature=selection_notes_by_feature,
            )
        )

        if horizon_cap < 1:
            if verbose:
                print(
                    f"  group {group.group_id} has horizon_cap={horizon_cap} (zero "
                    "genuine forecast lead time); skipping walk-forward backtest, "
                    "keeping in-sample coefficients only",
                    flush=True,
                )
            metrics = _not_forecast_viable_metrics(
                group=group,
                order=order,
                seasonal_order=seasonal_order,
                criterion=criterion,
                selection_aic=float(best["aic"]),
                selection_bic=float(best["bic"]),
                level_change_note=_group_level_change_summary(group, choices),
                horizon_cap=horizon_cap,
            )
            actual_horizons: tuple[int, ...] = ()
        else:
            sarimax_predictions = _group_predictions(
                group=group,
                series=series,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                initial_train_size=initial_train_size,
                horizons=requested_horizons,
                maxiter=maxiter,
            )
            actual_horizons = tuple(sorted(sarimax_predictions["horizon"].unique()))
            baseline_frames = _baseline_predictions(
                series=series,
                sarimax_predictions=sarimax_predictions,
                rba_path=rba_path,
                initial_train_size=initial_train_size,
                horizons=actual_horizons,
                maxiter=maxiter,
            )
            common_predictions = restrict_to_common_grid(baseline_frames)
            metrics = compute_metric_table(common_predictions)
            sarimax_common = common_predictions.loc[common_predictions["model"] == "sarimax"]
            origin_n = int(sarimax_common["forecast_origin"].nunique())
            overall_metric_n = int(
                metrics.loc[
                    (metrics["model"] == "sarimax") & (metrics["horizon"] == "overall"),
                    "n",
                ].iloc[0]
            )
            metrics.insert(0, "group_id", group.group_id)
            metrics.insert(1, "group_name", group.group_name)
            metrics.insert(2, "horizon_range", _horizon_range_label(actual_horizons))
            metrics.insert(3, "horizon_cap", horizon_cap)
            metrics.insert(4, "features", ";".join(group.features))
            metrics.insert(5, "level_change_selection_note", _group_level_change_summary(group, choices))
            metrics.insert(6, "selected_order", str(order))
            metrics.insert(7, "selected_seasonal_order", str(seasonal_order))
            metrics.insert(8, "selected_by", criterion)
            metrics.insert(9, "selection_aic", float(best["aic"]))
            metrics.insert(10, "selection_bic", float(best["bic"]))
            drop_from_previous = (
                None if previous_nested_origin_n is None else previous_nested_origin_n - origin_n
            )
            sample_note = ORDER_SELECTION_NOTE
            if group.group_id in NESTED_GROUP_IDS and previous_nested_origin_n is not None:
                if origin_n < previous_nested_origin_n * SHARP_SAMPLE_DROP_RATIO:
                    sample_note = (
                        f"Sharp nested origin sample drop from n={previous_nested_origin_n} "
                        f"to n={origin_n}; compare this group over its own window. {ORDER_SELECTION_NOTE}"
                    )
            elif group.group_id not in NESTED_GROUP_IDS:
                sample_note = (
                    "Independent variant group; not a nested feature add-on on the A-E "
                    f"chain. {ORDER_SELECTION_NOTE}"
                )
            previous_origin_for_output = (
                previous_nested_origin_n if group.group_id in NESTED_GROUP_IDS else None
            )
            origin_drop_for_output = (
                drop_from_previous if group.group_id in NESTED_GROUP_IDS else None
            )
            metrics.insert(11, "group_origin_n", origin_n)
            metrics.insert(12, "group_overall_metric_n", overall_metric_n)
            metrics.insert(13, "previous_nested_origin_n", previous_origin_for_output)
            metrics.insert(14, "origin_drop_from_previous_nested_group", origin_drop_for_output)
            metrics.insert(15, "sample_size_note", sample_note)
            if group.group_id in NESTED_GROUP_IDS:
                previous_nested_origin_n = origin_n

        metric_tables.append(metrics)
        tracking_payloads.append(
            {
                "group": group,
                "metrics": metrics.copy(),
                "fitted": fitted,
                "order": order,
                "seasonal_order": seasonal_order,
                "horizon_cap": horizon_cap,
                "actual_horizons": actual_horizons,
                "selection_aic": float(best["aic"]),
                "selection_bic": float(best["bic"]),
            }
        )
        if verbose:
            print(f"  completed group {group.group_id}", flush=True)

    comparison = pd.concat(metric_tables, ignore_index=True)
    coefficients = pd.concat(coefficient_tables, ignore_index=True)
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    coefficient_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.round({"rmse": 6, "mae": 6, "selection_aic": 6, "selection_bic": 6}).to_csv(
        comparison_output_path,
        index=False,
    )
    coefficients.round({"aic": 6, "bic": 6, "coef": 6, "std_err": 6, "p_value": 6}).to_csv(
        coefficient_output_path,
        index=False,
    )
    artifact_paths = [comparison_output_path, coefficient_output_path]
    with tracking.start_parent_run(
        run_name="sarimax_comparison",
        params={
            "selection_criterion": criterion,
            "initial_train_size": initial_train_size,
            "horizons": requested_horizons,
            "max_p": max_p,
            "max_q": max_q,
            "max_p_seasonal": max_p_seasonal,
            "max_q_seasonal": max_q_seasonal,
            "d_values": tuple(int(value) for value in d_values),
            "seasonal_d_values": tuple(int(value) for value in seasonal_d_values),
        },
        tags={
            "model_family": "sarimax",
            "run_role": "comparison_parent",
        },
    ):
        tracking.log_existing_artifacts(artifact_paths)
        for payload in tracking_payloads:
            group = payload["group"]
            assert isinstance(group, FeatureGroup)
            tracking.log_model_run(
                run_name=f"sarimax_group_{group.group_id}",
                model_name="sarimax",
                metrics=payload["metrics"],
                params={
                    "order": payload["order"],
                    "seasonal_order": payload["seasonal_order"],
                    "features": group.features,
                    "selection_criterion": criterion,
                    "initial_train_size": initial_train_size,
                    "horizons": requested_horizons,
                    "actual_horizons": payload["actual_horizons"],
                    "horizon_cap": payload["horizon_cap"],
                    "selection_aic": payload["selection_aic"],
                    "selection_bic": payload["selection_bic"],
                },
                tags={
                    "model_family": "sarimax",
                    "sarimax_feature_group_id": group.group_id,
                    "sarimax_feature_group_name": group.group_name,
                    "run_role": "comparison_with_full_sample_model",
                },
                artifact_paths=artifact_paths,
                model_logger=lambda fitted=payload["fitted"]: tracking.log_statsmodels_model(
                    fitted
                ),
                nested=True,
            )
    return comparison, coefficients, choices


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--rba-data", type=Path, default=RBA_FORECAST_PATH)
    parser.add_argument("--comparison-output", type=Path, default=SARIMAX_COMPARISON_OUTPUT_PATH)
    parser.add_argument("--coefficients-output", type=Path, default=SARIMAX_COEFFICIENT_OUTPUT_PATH)
    parser.add_argument("--criterion", choices=("aic", "bic"), default="aic")
    parser.add_argument("--d-values", default="0")
    parser.add_argument("--seasonal-d-values", default="0")
    parser.add_argument("--max-p", type=int, default=DEFAULT_COMPARISON_MAX_ARMA_ORDER)
    parser.add_argument("--max-q", type=int, default=DEFAULT_COMPARISON_MAX_ARMA_ORDER)
    parser.add_argument("--max-p-seasonal", type=int, default=DEFAULT_COMPARISON_MAX_ARMA_ORDER)
    parser.add_argument("--max-q-seasonal", type=int, default=DEFAULT_COMPARISON_MAX_ARMA_ORDER)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument(
        "--order-selection-holdout-quarters",
        type=int,
        default=ORDER_SELECTION_HOLDOUT_QUARTERS,
        help="Quarters excluded from the tail of the series before ARIMA order "
        "and level-vs-change selection, so the walk-forward test tail never "
        "informs those choices.",
    )
    args = parser.parse_args(argv)

    comparison, coefficients, choices = run_sarimax_comparison(
        curated_path=args.data,
        rba_path=args.rba_data,
        comparison_output_path=args.comparison_output,
        coefficient_output_path=args.coefficients_output,
        criterion=args.criterion,
        max_p=args.max_p,
        max_q=args.max_q,
        max_p_seasonal=args.max_p_seasonal,
        max_q_seasonal=args.max_q_seasonal,
        d_values=_parse_int_values(args.d_values),
        seasonal_d_values=_parse_int_values(args.seasonal_d_values),
        maxiter=args.maxiter,
        order_selection_holdout_quarters=args.order_selection_holdout_quarters,
        verbose=True,
    )

    print("Level-vs-change choices:")
    for choice in choices.values():
        print(f"- {choice.family}: selected {choice.winner}. {choice.selection_note}")
    print("\nSaved SARIMAX comparison:")
    print(comparison.loc[comparison["horizon"] == "overall"].to_string(index=False))
    notes = (
        comparison.loc[
            (comparison["horizon"] == "overall") & comparison["sample_size_note"].astype(bool),
            ["group_id", "sample_size_note"],
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if not notes.empty:
        print("\nSample-size notes:")
        print(notes.to_string(index=False))
    print("\nCoefficient sign checks:")
    print(coefficients[["group_id", "feature", "coef", "sign_check"]].to_string(index=False))
    print(f"\nCaveat: {COEFFICIENT_CAVEAT}")


if __name__ == "__main__":
    main()
