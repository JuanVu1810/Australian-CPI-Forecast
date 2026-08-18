"""Shared MLflow tracking helpers for model comparison orchestrators."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TRACKING_URI = "mlruns"
DEFAULT_EXPERIMENT_NAME = "CPI Forecast"
FULL_SAMPLE_MODEL_TAG = "full_sample_production_fit_not_backtest_validated"


def configure_mlflow() -> Any:
    """Configure local MLflow tracking from env vars, with credential-free defaults."""
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME
    if "://" not in tracking_uri:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Period):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _stringify_param(value: object) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    return json.dumps(value, default=_json_default, sort_keys=True)


def _clean_params(params: Mapping[str, object]) -> dict[str, str | int | float | bool]:
    return {key: _stringify_param(value) for key, value in params.items()}


def _clean_tags(tags: Mapping[str, object]) -> dict[str, str]:
    return {key: str(value) for key, value in tags.items()}


def metric_values_from_table(
    metrics: pd.DataFrame,
    model_name: str,
) -> dict[str, float]:
    """Extract overall and horizon RMSE/MAE metrics for one model only."""
    required = {"model", "horizon", "rmse", "mae"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"metrics table missing required columns: {sorted(missing)}")

    model_rows = pd.DataFrame(metrics).loc[metrics["model"].eq(model_name)]
    if model_rows.empty:
        raise ValueError(f"metrics table has no rows for model {model_name!r}.")

    logged: dict[str, float] = {}
    for _, row in model_rows.iterrows():
        horizon = row["horizon"]
        if str(horizon) == "overall":
            suffix = "overall"
        else:
            suffix = f"h{int(horizon)}"
        logged[f"rmse_{suffix}"] = float(row["rmse"])
        logged[f"mae_{suffix}"] = float(row["mae"])
    return logged


def log_existing_artifacts(paths: Iterable[Path]) -> None:
    """Log existing report artifacts under the MLflow ``reports`` artifact path."""
    mlflow = configure_mlflow()
    for path in paths:
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"MLflow artifact does not exist: {artifact_path}")
        mlflow.log_artifact(str(artifact_path), artifact_path="reports")


def log_statsmodels_model(model: Any, artifact_path: str = "model") -> None:
    """Log a full-sample statsmodels fit via the MLflow statsmodels flavor."""
    mlflow = configure_mlflow()
    import mlflow.statsmodels

    mlflow.set_tag("model_artifact_role", FULL_SAMPLE_MODEL_TAG)
    model_info = mlflow.statsmodels.log_model(model, artifact_path=artifact_path)
    _tag_logged_model(model_info)


def log_lstm_keras_model(
    fitted: Any,
    train_frame: pd.DataFrame,
    artifact_path: str = "model",
) -> None:
    """Log a full-sample Keras LSTM and its preprocessing metadata."""
    mlflow = configure_mlflow()
    import mlflow.keras
    from mlflow.models import infer_signature

    columns = list(fitted.feature_columns)
    mean = {column: float(fitted.scaler.mean_.loc[column]) for column in columns}
    scale = {column: float(fitted.scaler.scale_.loc[column]) for column in columns}
    mlflow.log_dict(
        {
            "feature_columns": columns,
            "target_column": fitted.target_column,
            "lookback": int(fitted.lookback),
            "horizon": int(fitted.horizon),
            "scaler_mean": mean,
            "scaler_scale": scale,
        },
        "model_preprocessing/lstm_scaler.json",
    )

    signature = None
    clean = pd.DataFrame(train_frame).loc[:, columns].dropna().astype(float)
    if len(clean) >= int(fitted.lookback):
        latest_window = fitted.scaler.transform(clean.iloc[-int(fitted.lookback) :])
        latest_window = latest_window.reshape(1, int(fitted.lookback), len(columns))
        prediction = fitted.model.predict(latest_window, verbose=0)
        signature = infer_signature(latest_window, prediction)

    mlflow.set_tag("model_artifact_role", FULL_SAMPLE_MODEL_TAG)
    model_info = mlflow.keras.log_model(
        fitted.model,
        artifact_path=artifact_path,
        signature=signature,
    )
    _tag_logged_model(model_info)


def log_elastic_net_model(
    fitted: Any,
    train_frame: pd.DataFrame,
    artifact_path: str = "model",
) -> None:
    """Log a full-sample Elastic Net (one scaler+model pipeline per horizon).

    ``fitted.models[horizon]`` is a ``scaler -> ElasticNet`` sklearn
    ``Pipeline`` selected by per-fold cross-validated ``GridSearchCV``
    (``src/models/elastic_net.py``), so alpha/l1_ratio/intercept come from
    its ``"model"`` step, and the logged signature uses raw (unscaled)
    inputs -- the pipeline scales internally, unlike the plain
    ``TrainWindowScaler``-based LSTM model this mirrors.
    """
    mlflow = configure_mlflow()
    import mlflow.sklearn
    from mlflow.models import infer_signature

    columns = list(fitted.feature_columns)
    mean = {column: float(fitted.scaler.mean_.loc[column]) for column in columns}
    scale = {column: float(fitted.scaler.scale_.loc[column]) for column in columns}
    per_horizon = {
        str(horizon): {
            "alpha": float(pipeline.named_steps["model"].alpha),
            "l1_ratio": float(pipeline.named_steps["model"].l1_ratio),
            "intercept": float(pipeline.named_steps["model"].intercept_),
        }
        for horizon, pipeline in fitted.models.items()
    }
    mlflow.log_dict(
        {
            "feature_columns": columns,
            "target_column": fitted.target_column,
            "horizons": list(fitted.horizons),
            "scaler_mean": mean,
            "scaler_scale": scale,
            "per_horizon_selection": per_horizon,
        },
        "model_preprocessing/elastic_net_scaler.json",
    )

    signature = None
    clean = pd.DataFrame(train_frame).loc[:, columns].dropna().astype(float)
    if not clean.empty:
        latest = clean.iloc[[-1]]
        prediction = fitted.predict_next(train_frame).reshape(1, -1)
        signature = infer_signature(latest, prediction)

    mlflow.set_tag("model_artifact_role", FULL_SAMPLE_MODEL_TAG)
    # ElasticNetDirectFit is a project-defined dataclass wrapping one
    # scaler+ElasticNet Pipeline per horizon, not a single scikit-learn
    # estimator, so the default "skops" serializer (which only supports
    # plain sklearn classes) cannot pickle it. cloudpickle can serialize
    # arbitrary Python objects;
    # the tradeoff is that loading the logged model later re-executes
    # arbitrary code, same as loading any pickle -- acceptable here since
    # these artifacts are produced and consumed by this project's own code.
    model_info = mlflow.sklearn.log_model(
        fitted,
        artifact_path=artifact_path,
        signature=signature,
        serialization_format="cloudpickle",
    )
    _tag_logged_model(model_info)


def _tag_logged_model(model_info: Any) -> None:
    """Persist MLflow's authoritative model URI/id on the active run."""
    mlflow = configure_mlflow()
    model_uri = getattr(model_info, "model_uri", None)
    model_id = getattr(model_info, "model_id", None)
    if model_uri:
        mlflow.set_tag("logged_model_uri", str(model_uri))
    if model_id:
        mlflow.set_tag("logged_model_id", str(model_id))


def log_model_run(
    run_name: str,
    model_name: str,
    metrics: pd.DataFrame,
    params: Mapping[str, object],
    tags: Mapping[str, object],
    artifact_paths: Iterable[Path],
    extra_metrics: Mapping[str, float] | None = None,
    model_logger: Callable[[], None] | None = None,
    nested: bool = False,
) -> str:
    """Create one MLflow run with params, own-row metrics, artifacts, and model."""
    mlflow = configure_mlflow()
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tags(_clean_tags(tags))
        mlflow.log_params(_clean_params(params))
        metric_values = metric_values_from_table(metrics, model_name=model_name)
        if extra_metrics:
            metric_values.update({key: float(value) for key, value in extra_metrics.items()})
        mlflow.log_metrics(metric_values)
        log_existing_artifacts(artifact_paths)
        if model_logger is not None:
            model_logger()
        return run.info.run_id


def start_parent_run(
    run_name: str,
    params: Mapping[str, object],
    tags: Mapping[str, object],
):
    """Start a configured MLflow parent run for nested child comparison runs."""
    mlflow = configure_mlflow()
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tags(_clean_tags(tags))
    mlflow.log_params(_clean_params(params))
    return run
