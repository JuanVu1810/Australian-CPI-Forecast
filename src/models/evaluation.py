"""Walk-forward model evaluation for quarterly CPI year-ended inflation.

This harness evaluates ``cpi_yoy`` rather than ``cpi_index`` or ``cpi_qoq``.
That choice keeps the SARIMA baseline, seasonal-naive benchmark, and RBA
historical forecasts on the same basis: year-ended CPI inflation in percent.

The seasonal-naive benchmark is therefore: for each forecast origin, predict
future ``cpi_yoy`` by copying the latest observed value from the same calendar
quarter one year earlier. For horizons beyond four quarters, the same four
seasonal values are repeated recursively. Because ``cpi_yoy`` is already a
four-quarter change, this benchmark is close to persistence, but keeping it on
``cpi_yoy`` avoids mixing units with the RBA comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURATED_DATA_PATH = PROJECT_ROOT / "data/curated/quarterly_macro_features.csv"
RBA_FORECAST_PATH = (
    PROJECT_ROOT / "dataset/rba/rba_historical_cpi_forecasts_by_horizon_1995_2025.csv"
)
COMPARISON_OUTPUT_PATH = PROJECT_ROOT / "reports/model_comparison_sarima.csv"

TARGET_COLUMN = "cpi_yoy"
QUARTER_COLUMN = "quarter"
DEFAULT_HORIZONS = tuple(range(1, 9))
DEFAULT_INITIAL_TRAIN_SIZE = 32

ForecastFunction = Callable[[pd.Series, int], Sequence[float] | pd.Series | np.ndarray]
ExogForecastFunction = Callable[
    [pd.Series, pd.DataFrame, pd.DataFrame, int],
    Sequence[float] | pd.Series | np.ndarray,
]
DirectMultihorizonForecastFunction = Callable[
    [pd.DataFrame, int],
    Sequence[float] | pd.Series | np.ndarray,
]
IntervalSimulationFunction = Callable[..., np.ndarray]

LAG_COLUMN_PATTERN = re.compile(r"_lag(\d+)$")
INTERVAL_COVERAGE_COLUMNS = [
    "model",
    "forecast_origin",
    "target_quarter",
    "horizon",
    "actual",
    "point_forecast_proxy",
    "interval_lower",
    "interval_upper",
    "hit",
    "interval_width",
    "nonconformity_score",
]


@dataclass(frozen=True)
class TrainWindowScaler:
    """Small train-window standard scaler shared by models that need one.

    Fitting must always happen on exactly the caller's training window (never
    the full series) so that per-origin walk-forward scaling stays leakage
    safe; see the direct-multihorizon Elastic Net fit for the intended usage
    pattern.
    """

    columns: tuple[str, ...]
    mean_: pd.Series
    scale_: pd.Series

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = pd.DataFrame(frame).loc[:, list(self.columns)].astype(float)
        return ((values - self.mean_) / self.scale_).to_numpy(dtype=np.float32)

    def inverse_transform_target(
        self,
        values: np.ndarray | pd.Series | list[float],
        target_column: str,
    ) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float32)
        return raw * float(self.scale_.loc[target_column]) + float(self.mean_.loc[target_column])


def fit_train_window_scaler(frame: pd.DataFrame) -> TrainWindowScaler:
    """Fit a standard scaler on exactly the provided training window."""
    clean = pd.DataFrame(frame).astype(float)
    mean = clean.mean(axis=0)
    scale = clean.std(axis=0, ddof=0).replace(0.0, 1.0)
    return TrainWindowScaler(columns=tuple(clean.columns), mean_=mean, scale_=scale)


def assert_consecutive_quarters(index: pd.Index, context: str) -> None:
    """Raise if ``index`` is not a gap-free run of consecutive quarters."""
    period_index = pd.PeriodIndex(index, freq="Q")
    if period_index.empty:
        return
    expected = pd.period_range(period_index[0], periods=len(period_index), freq="Q")
    if not period_index.equals(expected):
        raise ValueError(f"{context} must contain consecutive quarterly observations.")


def _as_quarter_period(value: object) -> pd.Period:
    if isinstance(value, pd.Period):
        return value.asfreq("Q")
    if isinstance(value, pd.Timestamp):
        return value.to_period("Q")
    text = str(value)
    if "Q" in text:
        return pd.Period(text, freq="Q")
    return pd.Timestamp(text).to_period("Q")


def _normalise_quarter_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ("forecast_origin", "target_quarter"):
        if column in result:
            result[column] = result[column].map(_as_quarter_period)
    if "horizon" in result:
        result["horizon"] = result["horizon"].astype(int)
    return result


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    if not requested_horizons or min(requested_horizons) < 1:
        raise ValueError("horizons must contain positive integers.")
    return requested_horizons


def _prepare_walk_forward_series(
    series: pd.Series,
    initial_train_size: int,
    max_horizon: int,
) -> pd.Series:
    y = pd.Series(series).dropna().astype(float).sort_index()
    if initial_train_size < 1:
        raise ValueError("initial_train_size must be at least 1.")
    if len(y) < initial_train_size + max_horizon:
        raise ValueError("series is too short for the requested initial window and horizons.")
    if y.index.has_duplicates:
        raise ValueError("series index must not contain duplicate quarters.")
    return y


def _iter_expanding_origin_positions(
    series: pd.Series,
    initial_train_size: int,
    max_horizon: int,
):
    for origin_pos in range(initial_train_size - 1, len(series) - max_horizon):
        yield origin_pos


def infer_min_lag_from_columns(columns: Iterable[str]) -> int:
    """Infer the shortest forecast-origin-safe horizon from lagged feature names."""
    lags = []
    for column in columns:
        match = LAG_COLUMN_PATTERN.search(str(column))
        if match:
            lags.append(int(match.group(1)))
    if not lags:
        raise ValueError("exogenous feature columns must include lag suffixes like '_lag1'.")
    return min(lags)


def _coerce_quarter_index(df: pd.DataFrame, quarter_column: str = QUARTER_COLUMN) -> pd.DataFrame:
    """Return a copy with a quarterly PeriodIndex."""
    result = df.copy()
    if isinstance(result.index, pd.PeriodIndex):
        result.index = result.index.asfreq("Q")
        return result.sort_index()
    if quarter_column in result.columns:
        result.index = pd.PeriodIndex(result.pop(quarter_column).astype(str), freq="Q")
        return result.sort_index()
    result.index = pd.Index(result.index).map(_as_quarter_period)
    result.index = pd.PeriodIndex(result.index, freq="Q")
    return result.sort_index()


def load_target_series(
    path: Path = CURATED_DATA_PATH,
    target_column: str = TARGET_COLUMN,
    quarter_column: str = QUARTER_COLUMN,
) -> pd.Series:
    """Load the modelling target as a quarterly ``PeriodIndex`` series."""
    df = pd.read_csv(path, usecols=[quarter_column, target_column])
    if quarter_column not in df or target_column not in df:
        raise ValueError(f"{path} must contain {quarter_column!r} and {target_column!r}.")

    index = pd.PeriodIndex(df[quarter_column].astype(str), freq="Q")
    series = pd.Series(df[target_column].to_numpy(), index=index, name=target_column)
    return series.dropna().astype(float).sort_index()


def seasonal_naive_forecast(
    train_series: pd.Series,
    steps: int = 8,
    seasonal_period: int = 4,
) -> pd.Series:
    """Forecast by repeating the last observed value from the same quarter."""
    train = pd.Series(train_series).dropna().astype(float)
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if len(train) < seasonal_period:
        raise ValueError("seasonal naive requires at least one full seasonal cycle.")

    seasonal_values = train.iloc[-seasonal_period:].to_numpy()
    forecast_values = [
        seasonal_values[(step - 1) % seasonal_period] for step in range(1, steps + 1)
    ]

    if isinstance(train.index, pd.PeriodIndex):
        index = pd.period_range(train.index[-1] + 1, periods=steps, freq=train.index.freq)
    else:
        index = pd.RangeIndex(start=len(train), stop=len(train) + steps)
    return pd.Series(forecast_values, index=index, name="forecast")


def walk_forward_backtest(
    series: pd.Series,
    forecast_func: ForecastFunction,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    model_name: str = "model",
) -> pd.DataFrame:
    """Run an expanding-window rolling-origin backtest over fixed horizons."""
    horizons = _validate_horizons(horizons)
    max_horizon = max(horizons)
    y = _prepare_walk_forward_series(series, initial_train_size, max_horizon)

    rows: list[dict[str, object]] = []
    for origin_pos in _iter_expanding_origin_positions(y, initial_train_size, max_horizon):
        train = y.iloc[: origin_pos + 1]
        raw_forecast = forecast_func(train, max_horizon)
        forecast_values = np.asarray(pd.Series(raw_forecast), dtype=float)
        if len(forecast_values) < max_horizon:
            raise ValueError("forecast_func returned fewer values than the maximum horizon.")

        forecast_origin = y.index[origin_pos]
        for horizon in horizons:
            target_pos = origin_pos + horizon
            target_quarter = y.index[target_pos]
            actual = float(y.iloc[target_pos])
            forecast = float(forecast_values[horizon - 1])
            rows.append(
                {
                    "model": model_name,
                    "forecast_origin": forecast_origin,
                    "target_quarter": target_quarter,
                    "horizon": horizon,
                    "actual": actual,
                    "forecast": forecast,
                    "error": actual - forecast,
                }
            )

    return pd.DataFrame(rows)


def walk_forward_backtest_with_exog(
    series: pd.Series,
    exog: pd.DataFrame,
    forecast_func: ExogForecastFunction,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    model_name: str = "model",
    horizon_cap: int | None = None,
) -> pd.DataFrame:
    """Run an expanding-window backtest with lag-safe exogenous predictors.

    Lagged macro features are only forecast-origin available through horizons
    less than or equal to their lag. For example, ``cash_rate_lag2`` at
    ``origin + 2`` is the cash rate observed at the origin, but the same column
    at ``origin + 3`` would require a future cash-rate value. This harness
    therefore evaluates only horizons up to the minimum lag in the feature set.
    """
    requested_horizons = _validate_horizons(horizons)

    x = _coerce_quarter_index(pd.DataFrame(exog))
    if x.empty:
        raise ValueError("exog must contain at least one column.")
    if x.index.has_duplicates:
        raise ValueError("exog index must not contain duplicate quarters.")

    inferred_cap = infer_min_lag_from_columns(x.columns)
    cap = inferred_cap if horizon_cap is None else min(int(horizon_cap), inferred_cap)
    capped_horizons = tuple(horizon for horizon in requested_horizons if horizon <= cap)
    if not capped_horizons:
        return pd.DataFrame(
            columns=[
                "model",
                "forecast_origin",
                "target_quarter",
                "horizon",
                "actual",
                "forecast",
                "error",
                "horizon_cap",
            ]
        )

    max_horizon = max(capped_horizons)
    y = _prepare_walk_forward_series(series, initial_train_size, max_horizon)

    rows: list[dict[str, object]] = []
    for origin_pos in _iter_expanding_origin_positions(y, initial_train_size, max_horizon):
        train_y_raw = y.iloc[: origin_pos + 1]
        train_exog_raw = x.reindex(train_y_raw.index)
        train_frame = pd.concat(
            [train_y_raw.rename("__target__"), train_exog_raw],
            axis=1,
        ).dropna()
        if len(train_frame) < initial_train_size:
            continue

        target_index = y.index[origin_pos + 1 : origin_pos + max_horizon + 1]
        future_exog = x.reindex(target_index)
        if future_exog.isna().any(axis=None):
            continue

        train_y = train_frame["__target__"]
        train_exog = train_frame.drop(columns="__target__")
        raw_forecast = forecast_func(train_y, train_exog, future_exog, max_horizon)
        forecast_values = np.asarray(pd.Series(raw_forecast), dtype=float)
        if len(forecast_values) < max_horizon:
            raise ValueError("forecast_func returned fewer values than the maximum horizon.")

        forecast_origin = y.index[origin_pos]
        for horizon in capped_horizons:
            target_pos = origin_pos + horizon
            target_quarter = y.index[target_pos]
            actual = float(y.iloc[target_pos])
            forecast = float(forecast_values[horizon - 1])
            rows.append(
                {
                    "model": model_name,
                    "forecast_origin": forecast_origin,
                    "target_quarter": target_quarter,
                    "horizon": horizon,
                    "actual": actual,
                    "forecast": forecast,
                    "error": actual - forecast,
                    "horizon_cap": cap,
                }
            )

    return pd.DataFrame(rows)


def walk_forward_backtest_direct_multihorizon(
    series: pd.Series,
    exog: pd.DataFrame,
    forecast_func: DirectMultihorizonForecastFunction,
    initial_train_size: int = 40,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    model_name: str = "model",
    target_column: str = TARGET_COLUMN,
    max_origins: int | None = None,
    skip_origins: int = 0,
) -> pd.DataFrame:
    """Run an expanding-window backtest for direct multi-horizon sequence models.

    The forecaster receives only the target and feature history available at
    each forecast origin. Unlike ``walk_forward_backtest_with_exog``, this
    harness never constructs or passes a post-origin exogenous frame.

    ``skip_origins`` skips that many valid origins (without calling
    ``forecast_func`` on them) before evaluation starts. Combined with
    ``max_origins``, this lets a caller carve out chronologically disjoint
    origin sets -- e.g. ``skip_origins=0, max_origins=20`` for a screening
    subset and ``skip_origins=20`` for a held-out final-validation subset --
    so hyperparameters are never selected using the same origins they are
    ultimately reported against.
    """
    requested_horizons = _validate_horizons(horizons)
    if initial_train_size < 1:
        raise ValueError("initial_train_size must be at least 1.")
    if max_origins is not None and max_origins < 1:
        raise ValueError("max_origins must be at least 1 when supplied.")
    if skip_origins < 0:
        raise ValueError("skip_origins must be at least 0.")

    x = _coerce_quarter_index(pd.DataFrame(exog))
    if x.empty:
        raise ValueError("exog must contain at least one column.")
    if x.index.has_duplicates:
        raise ValueError("exog index must not contain duplicate quarters.")

    max_horizon = max(requested_horizons)
    y = _prepare_walk_forward_series(series, initial_train_size, max_horizon)

    rows: list[dict[str, object]] = []
    completed_origins = 0
    skipped_origins = 0
    for origin_pos in _iter_expanding_origin_positions(y, initial_train_size, max_horizon):
        train_y_raw = y.iloc[: origin_pos + 1]
        train_exog_raw = x.reindex(train_y_raw.index)
        train_frame = pd.concat(
            [train_y_raw.rename(target_column), train_exog_raw],
            axis=1,
        ).dropna()
        if len(train_frame) < initial_train_size:
            continue
        if skipped_origins < skip_origins:
            skipped_origins += 1
            continue

        raw_forecast = forecast_func(train_frame, max_horizon)
        forecast_values = np.asarray(pd.Series(raw_forecast), dtype=float)
        if len(forecast_values) < max_horizon:
            raise ValueError("forecast_func returned fewer values than the maximum horizon.")
        if not np.isfinite(forecast_values[:max_horizon]).all():
            raise ValueError("forecast_func returned non-finite forecast values.")

        forecast_origin = y.index[origin_pos]
        for horizon in requested_horizons:
            target_pos = origin_pos + horizon
            target_quarter = y.index[target_pos]
            actual = float(y.iloc[target_pos])
            forecast = float(forecast_values[horizon - 1])
            rows.append(
                {
                    "model": model_name,
                    "forecast_origin": forecast_origin,
                    "target_quarter": target_quarter,
                    "horizon": horizon,
                    "actual": actual,
                    "forecast": forecast,
                    "error": actual - forecast,
                }
            )

        completed_origins += 1
        if max_origins is not None and completed_origins >= max_origins:
            break

    return pd.DataFrame(rows)


def walk_forward_interval_coverage_backtest(
    series: pd.Series,
    simulate_func: IntervalSimulationFunction,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    model_name: str = "model",
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
    n_sims: int = 1000,
    seed: int = 42,
    exog: pd.DataFrame | None = None,
    training_data: str = "series",
    target_column: str = TARGET_COLUMN,
    horizon_cap: int | None = None,
    max_origins: int | None = None,
    skip_origins: int = 0,
) -> pd.DataFrame:
    """Backtest forecast interval coverage over expanding walk-forward origins.

    ``training_data`` controls the adapter shape for model-specific simulators:
    ``"series"`` calls ``simulate_func(train_series, steps=..., n_sims=..., seed=...)``;
    ``"frame"`` calls ``simulate_func(train_frame, steps=..., n_sims=..., seed=...)``;
    and ``"series_exog"`` calls
    ``simulate_func(train_series, train_exog, steps=..., n_sims=..., seed=...)``.
    """
    requested_horizons = _validate_horizons(horizons)
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("lower_quantile and upper_quantile must satisfy 0 <= lower < upper <= 1.")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1.")
    if max_origins is not None and max_origins < 1:
        raise ValueError("max_origins must be at least 1 when supplied.")
    if skip_origins < 0:
        raise ValueError("skip_origins must be at least 0.")
    if training_data not in {"series", "frame", "series_exog"}:
        raise ValueError("training_data must be one of: 'series', 'frame', 'series_exog'.")

    cap = max(requested_horizons) if horizon_cap is None else int(horizon_cap)
    capped_horizons = tuple(horizon for horizon in requested_horizons if horizon <= cap)
    if not capped_horizons:
        return pd.DataFrame(columns=INTERVAL_COVERAGE_COLUMNS)

    max_horizon = max(capped_horizons)
    y = _prepare_walk_forward_series(series, initial_train_size, max_horizon)
    x: pd.DataFrame | None = None
    if training_data in {"frame", "series_exog"}:
        if exog is None:
            raise ValueError(f"exog is required when training_data={training_data!r}.")
        x = _coerce_quarter_index(pd.DataFrame(exog))
        if x.empty:
            raise ValueError("exog must contain at least one column.")
        if x.index.has_duplicates:
            raise ValueError("exog index must not contain duplicate quarters.")

    from src.models.ensemble import ensemble_interval_from_paths

    rows: list[dict[str, object]] = []
    completed_origins = 0
    skipped_origins = 0
    for origin_pos in _iter_expanding_origin_positions(y, initial_train_size, max_horizon):
        train_y_raw = y.iloc[: origin_pos + 1]
        train_y = train_y_raw
        train_exog: pd.DataFrame | None = None
        train_frame: pd.DataFrame | None = None
        if x is not None:
            train_exog_raw = x.reindex(train_y_raw.index)
            train_frame = pd.concat(
                [train_y_raw.rename(target_column), train_exog_raw],
                axis=1,
            ).dropna()
            if len(train_frame) < initial_train_size:
                continue
            train_y = train_frame[target_column]
            train_exog = train_frame.drop(columns=target_column)

        if skipped_origins < skip_origins:
            skipped_origins += 1
            continue

        origin_seed = seed + completed_origins
        if training_data == "series":
            paths = simulate_func(train_y, steps=max_horizon, n_sims=n_sims, seed=origin_seed)
        elif training_data == "frame":
            if train_frame is None:
                raise RuntimeError("internal error: train_frame was not built.")
            paths = simulate_func(train_frame, steps=max_horizon, n_sims=n_sims, seed=origin_seed)
        else:
            if train_exog is None:
                raise RuntimeError("internal error: train_exog was not built.")
            paths = simulate_func(
                train_y,
                train_exog,
                steps=max_horizon,
                n_sims=n_sims,
                seed=origin_seed,
            )

        path_values = np.asarray(paths, dtype=float)
        if path_values.ndim != 2:
            raise ValueError("simulate_func must return a 2D array with shape (n_sims, steps).")
        if path_values.shape[0] < n_sims or path_values.shape[1] < max_horizon:
            raise ValueError("simulate_func returned fewer simulations or steps than requested.")
        if not np.isfinite(path_values[:, :max_horizon]).all():
            raise ValueError("simulate_func returned non-finite path values.")

        lower_values, upper_values = ensemble_interval_from_paths(
            path_values[:, :max_horizon],
            lower=lower_quantile,
            upper=upper_quantile,
        )
        point_values = np.median(path_values[:, :max_horizon], axis=0)
        forecast_origin = y.index[origin_pos]
        for horizon in capped_horizons:
            target_pos = origin_pos + horizon
            actual = float(y.iloc[target_pos])
            point_forecast_proxy = float(point_values[horizon - 1])
            interval_lower = float(lower_values[horizon - 1])
            interval_upper = float(upper_values[horizon - 1])
            interval_width = interval_upper - interval_lower
            half_width = interval_width / 2.0
            nonconformity_score = (
                np.inf
                if np.isclose(half_width, 0.0)
                else abs(actual - point_forecast_proxy) / half_width
            )
            rows.append(
                {
                    "model": model_name,
                    "forecast_origin": forecast_origin,
                    "target_quarter": y.index[target_pos],
                    "horizon": horizon,
                    "actual": actual,
                    "point_forecast_proxy": point_forecast_proxy,
                    "interval_lower": interval_lower,
                    "interval_upper": interval_upper,
                    "hit": bool(interval_lower <= actual <= interval_upper),
                    "interval_width": interval_width,
                    "nonconformity_score": float(nonconformity_score),
                }
            )

        completed_origins += 1
        if max_origins is not None and completed_origins >= max_origins:
            break

    return pd.DataFrame(rows, columns=INTERVAL_COVERAGE_COLUMNS)


def seasonal_naive_backtest(
    series: pd.Series,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    seasonal_period: int = 4,
    model_name: str = "seasonal_naive",
) -> pd.DataFrame:
    """Run the shared walk-forward harness for the seasonal-naive benchmark."""
    return walk_forward_backtest(
        series=series,
        forecast_func=lambda train, steps: seasonal_naive_forecast(train, steps, seasonal_period),
        initial_train_size=initial_train_size,
        horizons=horizons,
        model_name=model_name,
    )


def compute_baseline_predictions(
    series: pd.Series,
    rba_path: Path,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    include_rba: bool = True,
) -> pd.DataFrame:
    """Compute shared seasonal-naive and RBA walk-forward baseline predictions."""
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    seasonal_naive = seasonal_naive_backtest(
        series=series,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
    )
    frames = [seasonal_naive]
    if include_rba and rba_path.exists():
        rba = align_rba_forecasts_to_grid(
            pd.read_csv(rba_path),
            seasonal_naive,
            horizons=requested_horizons,
        )
        if not rba.empty:
            frames.append(rba)
    return pd.concat(frames, ignore_index=True, sort=False)


def align_rba_forecasts_to_grid(
    rba_forecasts: pd.DataFrame,
    forecast_grid: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    model_name: str = "rba",
) -> pd.DataFrame:
    """Align RBA historical forecasts to a model forecast-origin/horizon grid."""
    required = {"forecast_date", "horizon_quarters", "rba_forecast_cpi_yoy"}
    missing = required.difference(rba_forecasts.columns)
    if missing:
        raise ValueError(f"RBA forecasts missing required columns: {sorted(missing)}")

    grid = _normalise_quarter_columns(
        forecast_grid[["forecast_origin", "target_quarter", "horizon", "actual"]]
    )
    rba = rba_forecasts.copy()
    rba["forecast_origin"] = pd.to_datetime(rba["forecast_date"], errors="coerce").dt.to_period("Q")
    rba["horizon"] = rba["horizon_quarters"].astype(int)
    rba = rba.loc[
        rba["forecast_origin"].notna()
        & rba["horizon"].isin({int(h) for h in horizons})
        & rba["rba_forecast_cpi_yoy"].notna()
    ].copy()
    rba = rba.rename(
        columns={
            "rba_forecast_cpi_yoy": "forecast",
            "rba_actual_cpi_yoy": "rba_actual",
            "rba_forecast_error_cpi_yoy": "rba_reported_error",
        }
    )
    rba = rba.drop_duplicates(subset=["forecast_origin", "horizon"], keep="last")

    aligned = grid.merge(
        rba[["forecast_origin", "horizon", "forecast", "rba_actual", "rba_reported_error"]],
        on=["forecast_origin", "horizon"],
        how="inner",
    )
    aligned.insert(0, "model", model_name)
    aligned["error"] = aligned["actual"] - aligned["forecast"]
    return aligned[
        [
            "model",
            "forecast_origin",
            "target_quarter",
            "horizon",
            "actual",
            "forecast",
            "error",
            "rba_actual",
            "rba_reported_error",
        ]
    ]


def restrict_to_common_grid(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Keep only forecast-origin/horizon rows available for every model."""
    combined = _normalise_quarter_columns(pd.concat(frames, ignore_index=True, sort=False))
    complete = combined.dropna(subset=["actual", "forecast", "error"]).copy()
    model_count = complete["model"].nunique()
    key_columns = ["forecast_origin", "target_quarter", "horizon"]
    complete_keys = complete.groupby(key_columns)["model"].nunique().loc[
        lambda counts: counts == model_count
    ].index
    return complete.set_index(key_columns).loc[complete_keys].reset_index()


def compute_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute RMSE and MAE overall and by forecast horizon."""

    def summarise(group: pd.DataFrame, horizon: str | int) -> dict[str, object]:
        errors = group["error"].astype(float)
        return {
            "model": group["model"].iloc[0],
            "horizon": horizon,
            "n": int(errors.notna().sum()),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mae": float(np.mean(np.abs(errors))),
        }

    rows: list[dict[str, object]] = []
    for model, model_rows in predictions.dropna(subset=["error"]).groupby("model", sort=True):
        rows.append(summarise(model_rows, "overall"))
        for horizon, horizon_rows in model_rows.groupby("horizon", sort=True):
            rows.append(summarise(horizon_rows, int(horizon)))

    metrics = pd.DataFrame(rows)
    metrics["_horizon_order"] = metrics["horizon"].map(
        lambda horizon: 0 if horizon == "overall" else int(horizon)
    )
    return metrics.sort_values(["model", "_horizon_order"]).drop(columns="_horizon_order")


def compute_interval_coverage_table(
    predictions: pd.DataFrame,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compute empirical interval coverage overall and by forecast horizon."""
    from scipy.stats import binomtest

    required_columns = {"model", "horizon", "hit", "interval_width"}
    missing = required_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"interval predictions missing required columns: {sorted(missing)}")
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("lower_quantile and upper_quantile must satisfy 0 <= lower < upper <= 1.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must satisfy 0 < confidence_level < 1.")

    nominal_coverage = float(upper_quantile - lower_quantile)
    alpha = 1.0 - confidence_level

    def summarise(group: pd.DataFrame, horizon: str | int) -> dict[str, object]:
        hits = group["hit"].astype(bool)
        widths = group["interval_width"].astype(float)
        hit_count = int(hits.sum())
        n = int(hits.notna().sum())
        test = binomtest(hit_count, n=n, p=nominal_coverage)
        ci = test.proportion_ci(confidence_level=confidence_level, method="exact")
        return {
            "model": group["model"].iloc[0],
            "horizon": horizon,
            "n": n,
            "nominal_coverage": nominal_coverage,
            "empirical_coverage": float(hits.mean()),
            "coverage_ci_lower": float(ci.low),
            "coverage_ci_upper": float(ci.high),
            "binom_p_value": float(test.pvalue),
            "significantly_miscalibrated": bool(test.pvalue < alpha),
            "mean_interval_width": float(widths.mean()),
        }

    rows: list[dict[str, object]] = []
    clean = predictions.dropna(subset=["hit", "interval_width"]).copy()
    for model, model_rows in clean.groupby("model", sort=True):
        rows.append(summarise(model_rows, "overall"))
        for horizon, horizon_rows in model_rows.groupby("horizon", sort=True):
            rows.append(summarise(horizon_rows, int(horizon)))

    coverage = pd.DataFrame(rows)
    if coverage.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "horizon",
                "n",
                "nominal_coverage",
                "empirical_coverage",
                "coverage_ci_lower",
                "coverage_ci_upper",
                "binom_p_value",
                "significantly_miscalibrated",
                "mean_interval_width",
            ]
        )
    coverage["_horizon_order"] = coverage["horizon"].map(
        lambda horizon: 0 if horizon == "overall" else int(horizon)
    )
    return coverage.sort_values(["model", "_horizon_order"]).drop(columns="_horizon_order")


def compute_conformal_scale_factors(
    predictions: pd.DataFrame,
    target_coverage: float,
) -> pd.DataFrame:
    """Estimate multiplicative interval half-width scale factors by model+horizon."""
    required_columns = {"model", "horizon", "nonconformity_score"}
    missing = required_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"interval predictions missing required columns: {sorted(missing)}")
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must satisfy 0 < target_coverage < 1.")

    rows: list[dict[str, object]] = []
    clean = predictions.dropna(subset=["nonconformity_score"]).copy()
    for (model, horizon), group in clean.groupby(["model", "horizon"], sort=True):
        scores = group["nonconformity_score"].astype(float).to_numpy()
        rows.append(
            {
                "model": str(model),
                "horizon": int(horizon),
                "n": int(len(scores)),
                "target_coverage": float(target_coverage),
                "scale_factor": float(np.quantile(scores, target_coverage)),
            }
        )

    factors = pd.DataFrame(
        rows,
        columns=["model", "horizon", "n", "target_coverage", "scale_factor"],
    )
    if factors.empty:
        return factors
    return factors.sort_values(["model", "horizon"]).reset_index(drop=True)


def run_sarima_comparison(
    curated_path: Path = CURATED_DATA_PATH,
    rba_path: Path = RBA_FORECAST_PATH,
    output_path: Path = COMPARISON_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    target_column: str = TARGET_COLUMN,
    order: tuple[int, int, int] | None = None,
    seasonal_order: tuple[int, int, int, int] | None = None,
    include_rba: bool = True,
    run_name: str = "sarima_comparison",
    model_family_tag: str = "sarima",
    selection_criterion: str = "fixed_cpi_yoy_default",
) -> pd.DataFrame:
    """Run SARIMA, seasonal naive, and RBA comparison and save the metric table."""
    from src.models import tracking
    from src.models.sarima import (
        DEFAULT_ORDER,
        DEFAULT_SEASONAL_ORDER,
        fit_sarima,
        forecast_sarima,
    )

    horizons = tuple(horizons)
    order = DEFAULT_ORDER if order is None else order
    seasonal_order = DEFAULT_SEASONAL_ORDER if seasonal_order is None else seasonal_order
    target = load_target_series(curated_path, target_column=target_column)
    sarima = walk_forward_backtest(
        target,
        lambda train, steps: forecast_sarima(
            train,
            steps=steps,
            order=order,
            seasonal_order=seasonal_order,
        ),
        initial_train_size=initial_train_size,
        horizons=horizons,
        model_name="sarima",
    )
    baselines = compute_baseline_predictions(
        series=target,
        rba_path=rba_path,
        initial_train_size=initial_train_size,
        horizons=horizons,
        include_rba=include_rba,
    )
    common_predictions = restrict_to_common_grid([sarima, baselines])
    metrics = compute_metric_table(common_predictions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.round({"rmse": 6, "mae": 6}).to_csv(output_path, index=False)
    final_fit = fit_sarima(
        target,
        order=order,
        seasonal_order=seasonal_order,
    )
    tracking.log_model_run(
        run_name=run_name,
        model_name="sarima",
        metrics=metrics,
        params={
            "order": order,
            "seasonal_order": seasonal_order,
            "features": (),
            "target_column": target_column,
            "selection_criterion": selection_criterion,
            "initial_train_size": initial_train_size,
            "horizons": horizons,
        },
        tags={
            "model_family": model_family_tag,
            "target_column": target_column,
            "run_role": "comparison_with_full_sample_model",
        },
        artifact_paths=[output_path],
        model_logger=lambda: tracking.log_statsmodels_model(final_fit),
    )
    return metrics


if __name__ == "__main__":
    table = run_sarima_comparison()
    print(table.round({"rmse": 3, "mae": 3}).to_string(index=False))
