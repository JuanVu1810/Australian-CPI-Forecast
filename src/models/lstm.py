"""Compact direct multi-horizon LSTM baseline for CPI year-ended inflation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from src.models.evaluation import (
    CURATED_DATA_PATH,
    DEFAULT_HORIZONS,
    DEFAULT_INITIAL_TRAIN_SIZE as STATISTICAL_INITIAL_TRAIN_SIZE,
    PROJECT_ROOT,
    RBA_FORECAST_PATH,
    align_rba_forecasts_to_grid,
    compute_metric_table,
    load_target_series,
    restrict_to_common_grid,
    seasonal_naive_backtest,
    walk_forward_backtest,
    walk_forward_backtest_direct_multihorizon,
    walk_forward_backtest_with_exog,
)
from src.models.sarima import forecast_sarima
from src.models.sarimax import forecast_sarimax


TARGET_COLUMN = "cpi_yoy"
LSTM_FEATURE_COLUMNS = (
    "cash_rate_change_lag1",
    "unemployment_rate_change_lag1",
    "inflation_expectations_business_lag1",
    "ppi_growth_lag2",
    "commodity_growth_lag1",
    "wti_growth_lag1",
)
LSTM_INPUT_COLUMNS = (TARGET_COLUMN, *LSTM_FEATURE_COLUMNS)

LOOKBACK_QUARTERS = 8
FORECAST_HORIZON = 8
LSTM_UNITS = 16
DROPOUT = 0.2
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 8
VALIDATION_FRACTION = 0.2
DEFAULT_INITIAL_TRAIN_SIZE = 40
DEFAULT_SEED = 42

LSTM_COMPARISON_OUTPUT_PATH = PROJECT_ROOT / "reports/model_comparison_lstm.csv"
LSTM_PERMUTATION_OUTPUT_PATH = PROJECT_ROOT / "reports/lstm_permutation_importance.csv"
SARIMAX_COMPARISON_REPORT_PATH = PROJECT_ROOT / "reports/model_comparison_sarimax.csv"

LSTM_MODEL_NOTE = (
    "Fixed compact LSTM: lookback=8 quarters, one LSTM layer with 16 units, "
    "dropout=0.2, Adam, direct 8-quarter output, chronological validation tail, "
    "early stopping patience=10, max_epochs=100."
)
SARIMAX_D_FEATURES = ";".join(LSTM_FEATURE_COLUMNS)
SARIMAX_D_NOTE = (
    "SARIMAX(D) remains lag-safety capped at horizon 1 because Group D includes "
    "lag-1 regressors; LSTM horizons 1-8 use only the past origin window."
)
PERMUTATION_IMPORTANCE_CAVEAT = (
    "Small-sample chronological holdout; negative RMSE increases mean that this "
    "single permutation run improved RMSE and should be read as no robust "
    "positive importance, not as evidence of a beneficially harmful feature."
)


@dataclass(frozen=True)
class TrainWindowScaler:
    """Small train-window standard scaler to avoid an sklearn dependency."""

    columns: tuple[str, ...]
    mean_: pd.Series
    scale_: pd.Series

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = pd.DataFrame(frame).loc[:, list(self.columns)].astype(float)
        return ((values - self.mean_) / self.scale_).to_numpy(dtype=np.float32)

    def inverse_transform_target(
        self,
        values: np.ndarray | pd.Series | list[float],
        target_column: str = TARGET_COLUMN,
    ) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float32)
        return raw * float(self.scale_.loc[target_column]) + float(self.mean_.loc[target_column])


@dataclass
class LSTMDirectFit:
    model: Any
    scaler: TrainWindowScaler
    feature_columns: tuple[str, ...]
    target_column: str
    lookback: int
    horizon: int
    history: Any | None = None


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "TensorFlow is required for the LSTM baseline. Install project "
            "requirements, including tensorflow, before running this model."
        ) from exc
    return tf


def set_lstm_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set numpy and TensorFlow seeds for reproducible LSTM fits."""
    np.random.seed(seed)
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)


def build_model(
    n_features: int,
    lookback: int = LOOKBACK_QUARTERS,
    horizon: int = FORECAST_HORIZON,
    units: int = LSTM_UNITS,
    dropout: float = DROPOUT,
    seed: int = DEFAULT_SEED,
):
    """Build the compact LSTM architecture (units/dropout default to the fixed baseline)."""
    set_lstm_seeds(seed)
    tf = _tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, n_features)),
            tf.keras.layers.LSTM(units, dropout=dropout),
            tf.keras.layers.Dense(horizon),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model


def load_lstm_feature_frame(path: Path = CURATED_DATA_PATH) -> pd.DataFrame:
    """Load the fixed SARIMAX Group D feature set with a quarterly index."""
    required_columns = ("quarter", *LSTM_FEATURE_COLUMNS)
    df = pd.read_csv(path, usecols=list(required_columns))
    df.index = pd.PeriodIndex(df.pop("quarter").astype(str), freq="Q")
    return df.sort_index().astype(float)


def _assert_consecutive_quarters(index: pd.Index, context: str) -> None:
    period_index = pd.PeriodIndex(index, freq="Q")
    if period_index.empty:
        return
    expected = pd.period_range(period_index[0], periods=len(period_index), freq="Q")
    if not period_index.equals(expected):
        raise ValueError(f"{context} must contain consecutive quarterly observations.")


def clean_lstm_frame(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: tuple[str, ...] = LSTM_FEATURE_COLUMNS,
) -> pd.DataFrame:
    columns = (target_column, *feature_columns)
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"LSTM frame missing required columns: {sorted(missing)}")
    clean = pd.DataFrame(frame).loc[:, list(columns)].dropna().astype(float).sort_index()
    if clean.index.has_duplicates:
        raise ValueError("LSTM frame index must not contain duplicate quarters.")
    _assert_consecutive_quarters(clean.index, "LSTM frame")
    return clean


def fit_train_window_scaler(frame: pd.DataFrame) -> TrainWindowScaler:
    """Fit a standard scaler on exactly the provided training window."""
    clean = pd.DataFrame(frame).astype(float)
    mean = clean.mean(axis=0)
    scale = clean.std(axis=0, ddof=0).replace(0.0, 1.0)
    return TrainWindowScaler(columns=tuple(clean.columns), mean_=mean, scale_=scale)


def make_direct_multihorizon_sequences(
    frame: pd.DataFrame,
    scaler: TrainWindowScaler,
    lookback: int = LOOKBACK_QUARTERS,
    horizon: int = FORECAST_HORIZON,
    target_column: str = TARGET_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    """Create scaled past-window inputs and direct future-horizon targets."""
    if lookback < 1:
        raise ValueError("lookback must be at least 1.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    clean = pd.DataFrame(frame).loc[:, list(scaler.columns)].dropna().astype(float)
    _assert_consecutive_quarters(clean.index, "LSTM sequence frame")
    if len(clean) < lookback + horizon:
        raise ValueError("training frame is too short for lookback plus forecast horizon.")

    scaled = scaler.transform(clean)
    target_pos = list(scaler.columns).index(target_column)
    x_windows: list[np.ndarray] = []
    y_vectors: list[np.ndarray] = []
    for end_pos in range(lookback - 1, len(clean) - horizon):
        start_pos = end_pos - lookback + 1
        x_windows.append(scaled[start_pos : end_pos + 1, :])
        y_vectors.append(scaled[end_pos + 1 : end_pos + horizon + 1, target_pos])

    if not x_windows:
        raise ValueError("no supervised LSTM windows could be built.")
    return np.asarray(x_windows, dtype=np.float32), np.asarray(y_vectors, dtype=np.float32)


def fit_lstm_direct(
    train_frame: pd.DataFrame,
    lookback: int = LOOKBACK_QUARTERS,
    horizon: int = FORECAST_HORIZON,
    units: int = LSTM_UNITS,
    dropout: float = DROPOUT,
    validation_fraction: float = VALIDATION_FRACTION,
    epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
) -> LSTMDirectFit:
    """Fit one direct-output LSTM on a chronological train window."""
    clean = clean_lstm_frame(train_frame)
    scaler = fit_train_window_scaler(clean)
    x_all, y_all = make_direct_multihorizon_sequences(
        clean,
        scaler=scaler,
        lookback=lookback,
        horizon=horizon,
    )
    if len(x_all) < 2:
        raise ValueError("LSTM requires at least two supervised windows for validation.")

    val_size = max(1, int(np.ceil(len(x_all) * validation_fraction)))
    if val_size >= len(x_all):
        val_size = 1
    train_end = len(x_all) - val_size
    x_train, y_train = x_all[:train_end], y_all[:train_end]
    x_val, y_val = x_all[train_end:], y_all[train_end:]

    model = build_model(
        n_features=x_all.shape[-1],
        lookback=lookback,
        horizon=horizon,
        units=units,
        dropout=dropout,
        seed=seed,
    )
    tf = _tensorflow()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=verbose,
        callbacks=[early_stopping],
    )
    return LSTMDirectFit(
        model=model,
        scaler=scaler,
        feature_columns=tuple(clean.columns),
        target_column=TARGET_COLUMN,
        lookback=lookback,
        horizon=horizon,
        history=history,
    )


def forecast_from_fit(fitted: LSTMDirectFit, train_frame: pd.DataFrame) -> np.ndarray:
    clean = pd.DataFrame(train_frame).loc[:, list(fitted.feature_columns)].dropna().astype(float)
    _assert_consecutive_quarters(clean.index, "LSTM forecast frame")
    if len(clean) < fitted.lookback:
        raise ValueError("forecast window is shorter than the fitted lookback.")
    latest_window = fitted.scaler.transform(clean.iloc[-fitted.lookback :])
    scaled_forecast = fitted.model.predict(
        latest_window.reshape(1, fitted.lookback, len(fitted.feature_columns)),
        verbose=0,
    )[0]
    forecast = fitted.scaler.inverse_transform_target(
        scaled_forecast,
        target_column=fitted.target_column,
    )
    return np.asarray(forecast[: fitted.horizon], dtype=float)


def forecast_lstm_direct(
    train_frame: pd.DataFrame,
    steps: int = FORECAST_HORIZON,
    lookback: int = LOOKBACK_QUARTERS,
    units: int = LSTM_UNITS,
    dropout: float = DROPOUT,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
) -> np.ndarray:
    """Fit the LSTM and return a direct 8-quarter forecast vector."""
    if steps > FORECAST_HORIZON:
        raise ValueError("the LSTM baseline emits at most 8 horizons.")
    fitted = fit_lstm_direct(
        train_frame,
        lookback=lookback,
        units=units,
        dropout=dropout,
        seed=seed,
        verbose=verbose,
    )
    return forecast_from_fit(fitted, train_frame)[:steps]


def _make_interpretation_holdout_sequences(
    clean_frame: pd.DataFrame,
    scaler: TrainWindowScaler,
    holdout_size: int,
    lookback: int = LOOKBACK_QUARTERS,
    horizon: int = FORECAST_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    tail_start = len(clean_frame) - holdout_size
    if tail_start <= lookback:
        raise ValueError("holdout tail leaves too little training history.")

    scaled = scaler.transform(clean_frame)
    target_pos = list(scaler.columns).index(TARGET_COLUMN)
    x_windows: list[np.ndarray] = []
    y_vectors: list[np.ndarray] = []
    for end_pos in range(lookback - 1, len(clean_frame) - horizon):
        output_start = end_pos + 1
        output_end = end_pos + horizon
        if output_start >= tail_start and output_end < len(clean_frame):
            x_windows.append(scaled[end_pos - lookback + 1 : end_pos + 1, :])
            y_vectors.append(clean_frame.iloc[output_start : output_end + 1][TARGET_COLUMN].to_numpy())

    if not x_windows:
        raise ValueError("no chronological holdout windows could be built.")
    return np.asarray(x_windows, dtype=np.float32), np.asarray(y_vectors, dtype=float)


def _predict_original_scale(fitted: LSTMDirectFit, x_values: np.ndarray) -> np.ndarray:
    scaled = fitted.model.predict(x_values, verbose=0)
    return fitted.scaler.inverse_transform_target(scaled, target_column=fitted.target_column)


def permutation_importance(
    full_frame: pd.DataFrame,
    holdout_size: int = 20,
    n_repeats: int = 10,
    seed: int = DEFAULT_SEED,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Permutation importance from one separate full-sample interpretation model."""
    clean = clean_lstm_frame(full_frame)
    if len(clean) <= holdout_size + LOOKBACK_QUARTERS + FORECAST_HORIZON:
        raise ValueError("sample is too short for the requested interpretation holdout.")

    train_frame = clean.iloc[:-holdout_size]
    fitted = fit_lstm_direct(train_frame, seed=seed)
    x_holdout, y_holdout = _make_interpretation_holdout_sequences(
        clean_frame=clean,
        scaler=fitted.scaler,
        holdout_size=holdout_size,
    )
    baseline_pred = _predict_original_scale(fitted, x_holdout)
    baseline_rmse = float(np.sqrt(np.mean(np.square(y_holdout - baseline_pred))))

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for channel_idx, channel in enumerate(fitted.feature_columns):
        permuted_rmse_values = []
        for _ in range(n_repeats):
            permuted = x_holdout.copy()
            flat_channel = permuted[:, :, channel_idx].reshape(-1)
            permuted[:, :, channel_idx] = rng.permutation(flat_channel).reshape(
                permuted[:, :, channel_idx].shape
            )
            prediction = _predict_original_scale(fitted, permuted)
            permuted_rmse_values.append(
                float(np.sqrt(np.mean(np.square(y_holdout - prediction))))
            )
        permuted_rmse = float(np.mean(permuted_rmse_values))
        rows.append(
            {
                "feature": channel,
                "baseline_rmse": baseline_rmse,
                "permuted_rmse": permuted_rmse,
                "rmse_increase": permuted_rmse - baseline_rmse,
                "n_windows": int(len(x_holdout)),
                "n_values": int(y_holdout.size),
                "n_repeats": int(n_repeats),
                "interpretation_note": PERMUTATION_IMPORTANCE_CAVEAT,
            }
        )

    importance = pd.DataFrame(rows).sort_values(
        ["rmse_increase", "permuted_rmse"],
        ascending=[False, False],
        ignore_index=True,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        importance.round(
            {"baseline_rmse": 6, "permuted_rmse": 6, "rmse_increase": 6}
        ).to_csv(output_path, index=False)
    return importance


def _comparison_metadata(
    metrics: pd.DataFrame,
    origin_n: int,
    note: str,
    group_id: str = "LSTM",
    group_name: str = "compact direct multihorizon",
    horizon_range: str = "1-8",
    horizon_cap: int = FORECAST_HORIZON,
    features: str = SARIMAX_D_FEATURES,
    level_change_selection_note: str = "Reuses fixed SARIMAX Group D features.",
    selected_order: str = "fixed_lstm",
    selected_seasonal_order: str = "",
    selected_by: str = "fixed_architecture",
    selection_aic: float = np.nan,
    selection_bic: float = np.nan,
) -> pd.DataFrame:
    result = metrics.copy()
    result.insert(0, "group_id", group_id)
    result.insert(1, "group_name", group_name)
    result.insert(2, "horizon_range", horizon_range)
    result.insert(3, "horizon_cap", horizon_cap)
    result.insert(4, "features", features)
    result.insert(5, "level_change_selection_note", level_change_selection_note)
    result.insert(6, "selected_order", selected_order)
    result.insert(7, "selected_seasonal_order", selected_seasonal_order)
    result.insert(8, "selected_by", selected_by)
    result.insert(9, "selection_aic", selection_aic)
    result.insert(10, "selection_bic", selection_bic)
    result.insert(11, "group_origin_n", origin_n)
    result.insert(12, "group_overall_metric_n", int(metrics.loc[metrics["horizon"].eq("overall"), "n"].max()))
    result.insert(13, "previous_nested_origin_n", np.nan)
    result.insert(14, "origin_drop_from_previous_nested_group", np.nan)
    result.insert(15, "sample_size_note", note)
    return result


def _sarimax_d_predictions(
    series: pd.Series,
    exog: pd.DataFrame,
    initial_train_size: int,
    maxiter: int,
) -> pd.DataFrame:
    # Fixed from the existing SARIMAX(D) report; this does not tune on LSTM backtest errors.
    order = (0, 0, 1)
    seasonal_order = (1, 0, 1, 4)
    return walk_forward_backtest_with_exog(
        series=series,
        exog=exog.loc[:, list(LSTM_FEATURE_COLUMNS)],
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
        horizons=DEFAULT_HORIZONS,
        model_name="sarimax",
    )


def _load_existing_sarimax_d_metrics(
    path: Path = SARIMAX_COMPARISON_REPORT_PATH,
) -> pd.DataFrame:
    """Load canonical SARIMAX(D) metrics from the existing SARIMAX report."""
    if not path.exists():
        return pd.DataFrame()
    report = pd.read_csv(path)
    required_columns = {
        "group_id",
        "model",
        "horizon",
        "n",
        "rmse",
        "mae",
    }
    if not required_columns.issubset(report.columns):
        return pd.DataFrame()
    rows = report.loc[report["group_id"].eq("D") & report["model"].eq("sarimax")].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["sample_size_note"] = (
        "Copied from reports/model_comparison_sarimax.csv Group D; not "
        "intersected with the LSTM 8-horizon grid."
    )
    return rows


def run_lstm_comparison(
    curated_path: Path = CURATED_DATA_PATH,
    rba_path: Path = RBA_FORECAST_PATH,
    comparison_output_path: Path = LSTM_COMPARISON_OUTPUT_PATH,
    permutation_output_path: Path = LSTM_PERMUTATION_OUTPUT_PATH,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    seed: int = DEFAULT_SEED,
    max_origins: int | None = None,
    sarimax_maxiter: int = 100,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run the fixed LSTM baseline, comparison metrics, and permutation report."""
    from src.models import tracking

    started = time.perf_counter()
    requested_horizons = tuple(int(horizon) for horizon in horizons)
    series = load_target_series(curated_path)
    exog = load_lstm_feature_frame(curated_path)

    if verbose:
        print("Running fixed direct-output LSTM walk-forward backtest...", flush=True)
    lstm_predictions = walk_forward_backtest_direct_multihorizon(
        series=series,
        exog=exog,
        forecast_func=lambda train_frame, steps: forecast_lstm_direct(
            train_frame,
            steps=steps,
            seed=seed,
        ),
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        model_name="lstm",
        target_column=TARGET_COLUMN,
        max_origins=max_origins,
    )
    if lstm_predictions.empty:
        raise ValueError("LSTM backtest produced no forecast origins.")

    sarima = walk_forward_backtest(
        series=series,
        forecast_func=lambda train, steps: forecast_sarima(train, steps=steps),
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
        model_name="sarima",
    )
    naive = seasonal_naive_backtest(
        series=series,
        initial_train_size=initial_train_size,
        horizons=requested_horizons,
    )
    frames = [lstm_predictions, sarima, naive]
    if rba_path.exists():
        rba = align_rba_forecasts_to_grid(
            pd.read_csv(rba_path),
            lstm_predictions,
            horizons=requested_horizons,
        )
        if not rba.empty:
            frames.append(rba)

    full_horizon_common = restrict_to_common_grid(frames)
    full_metrics = compute_metric_table(full_horizon_common)
    origin_n = int(
        full_horizon_common.loc[
            full_horizon_common["model"].eq("lstm"),
            "forecast_origin",
        ].nunique()
    )
    comparison = _comparison_metadata(
        full_metrics,
        origin_n=origin_n,
        note=SARIMAX_D_NOTE,
    )

    if max_origins is None:
        if verbose:
            print("Running SARIMAX(D) lag-safe horizon-1 benchmark...", flush=True)
        sarimax_comparison = _load_existing_sarimax_d_metrics()
        if sarimax_comparison.empty:
            sarimax_d = _sarimax_d_predictions(
                series=series,
                exog=exog,
                initial_train_size=STATISTICAL_INITIAL_TRAIN_SIZE,
                maxiter=sarimax_maxiter,
            )
            sarimax_frames = [
                sarimax_d,
                walk_forward_backtest(
                    series=series,
                    forecast_func=lambda train, steps: forecast_sarima(train, steps=steps),
                    initial_train_size=STATISTICAL_INITIAL_TRAIN_SIZE,
                    horizons=tuple(sorted(sarimax_d["horizon"].unique())),
                    model_name="sarima",
                ),
                seasonal_naive_backtest(
                    series=series,
                    initial_train_size=STATISTICAL_INITIAL_TRAIN_SIZE,
                    horizons=tuple(sorted(sarimax_d["horizon"].unique())),
                ),
            ]
            if rba_path.exists():
                rba_for_sarimax = align_rba_forecasts_to_grid(
                    pd.read_csv(rba_path),
                    sarimax_d,
                    horizons=tuple(sorted(sarimax_d["horizon"].unique())),
                )
                if not rba_for_sarimax.empty:
                    sarimax_frames.append(rba_for_sarimax)
            sarimax_common = restrict_to_common_grid(sarimax_frames)
            sarimax_metrics = compute_metric_table(
                sarimax_common.loc[sarimax_common["model"].eq("sarimax")]
            )
            sarimax_comparison = _comparison_metadata(
                sarimax_metrics,
                origin_n=int(sarimax_common["forecast_origin"].nunique()),
                note=(
                    "SARIMAX(D) metrics use its lag-safe horizon-1 grid; "
                    "not intersected with the LSTM 8-horizon grid."
                ),
                group_id="D",
                group_name="full core",
                horizon_range="1",
                horizon_cap=1,
                selected_order="(0, 0, 1)",
                selected_seasonal_order="(1, 0, 1, 4)",
                selected_by="aic",
            )
        comparison = pd.concat([comparison, sarimax_comparison], ignore_index=True, sort=False)

    full_frame = pd.concat([series.rename(TARGET_COLUMN), exog], axis=1)
    if verbose:
        print("Training separate interpretation model for permutation importance...", flush=True)
    importance = permutation_importance(
        full_frame=full_frame,
        seed=seed,
        output_path=permutation_output_path,
    )

    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.round({"rmse": 6, "mae": 6, "selection_aic": 6, "selection_bic": 6}).to_csv(
        comparison_output_path,
        index=False,
    )
    if verbose:
        print("Training final full-sample LSTM for MLflow model logging...", flush=True)
    final_fit = fit_lstm_direct(
        full_frame,
        lookback=LOOKBACK_QUARTERS,
        horizon=FORECAST_HORIZON,
        epochs=MAX_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        batch_size=BATCH_SIZE,
        seed=seed,
    )
    permutation_metrics = {
        f"permutation_importance_{row['feature']}": float(row["rmse_increase"])
        for _, row in importance.iterrows()
    }
    tracking.log_model_run(
        run_name="lstm_comparison",
        model_name="lstm",
        metrics=comparison,
        params={
            "order": "fixed_lstm",
            "seasonal_order": "",
            "features": LSTM_FEATURE_COLUMNS,
            "selection_criterion": "fixed_architecture",
            "initial_train_size": initial_train_size,
            "horizons": requested_horizons,
            "lookback": LOOKBACK_QUARTERS,
            "units": LSTM_UNITS,
            "dropout": DROPOUT,
            "epochs": MAX_EPOCHS,
            "patience": EARLY_STOPPING_PATIENCE,
            "batch_size": BATCH_SIZE,
            "seed": seed,
        },
        tags={
            "model_family": "lstm",
            "reused_feature_group_id": "D",
            "run_role": "comparison_with_full_sample_model",
        },
        artifact_paths=[comparison_output_path, permutation_output_path],
        extra_metrics=permutation_metrics,
        model_logger=lambda: tracking.log_lstm_keras_model(final_fit, full_frame),
    )
    runtime_seconds = time.perf_counter() - started
    return comparison, importance, runtime_seconds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--rba-data", type=Path, default=RBA_FORECAST_PATH)
    parser.add_argument("--comparison-output", type=Path, default=LSTM_COMPARISON_OUTPUT_PATH)
    parser.add_argument("--permutation-output", type=Path, default=LSTM_PERMUTATION_OUTPUT_PATH)
    parser.add_argument("--initial-train-size", type=int, default=DEFAULT_INITIAL_TRAIN_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--sarimax-maxiter", type=int, default=100)
    args = parser.parse_args(argv)

    comparison, importance, runtime_seconds = run_lstm_comparison(
        curated_path=args.data,
        rba_path=args.rba_data,
        comparison_output_path=args.comparison_output,
        permutation_output_path=args.permutation_output,
        initial_train_size=args.initial_train_size,
        seed=args.seed,
        max_origins=args.max_origins,
        sarimax_maxiter=args.sarimax_maxiter,
        verbose=True,
    )
    print(LSTM_MODEL_NOTE)
    print(f"Runtime seconds: {runtime_seconds:.1f}")
    print("\nComparison:")
    print(comparison.to_string(index=False))
    print("\nPermutation importance:")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
