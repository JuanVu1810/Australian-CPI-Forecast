"""FastAPI service for serving the MLflow champion CPI forecast model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.models import tracking
from src.models.evaluation import TARGET_COLUMN, load_target_series
from src.models.lstm import (
    LSTMDirectFit,
    TrainWindowScaler,
    forecast_from_fit,
    load_lstm_feature_frame,
)
from src.models.registry import CHAMPION_ALIAS, REGISTERED_MODEL_NAME


CURATED_DATA_PATH = Path("data/curated/quarterly_macro_features.csv")
MAX_FORECAST_HORIZON = 8

app = FastAPI(
    title="Australian CPI Forecast API",
    version="0.2.0",
    description="Portfolio API for serving the MLflow Model Registry champion.",
)


class ForecastRequest(BaseModel):
    horizon: int = Field(default=8, ge=1, le=MAX_FORECAST_HORIZON)


class ForecastResponse(BaseModel):
    model_name: str
    model_version: str
    model_family: str
    run_id: str
    horizon: int
    forecast: list[float]
    quarters: list[str]
    forecast_origin: str


@dataclass(frozen=True)
class ModelForecast:
    values: list[float]
    forecast_origin: str
    quarters: list[str]


def load_curated_data() -> pd.DataFrame:
    if not CURATED_DATA_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Curated dataset not found at {CURATED_DATA_PATH}",
        )
    df = pd.read_csv(CURATED_DATA_PATH)
    if "quarter" not in df.columns or TARGET_COLUMN not in df.columns:
        raise HTTPException(
            status_code=503,
            detail=f"Curated dataset must contain quarter and {TARGET_COLUMN} columns.",
        )
    return df.sort_values("quarter").reset_index(drop=True)


def next_quarters(last_quarter: str, horizon: int) -> list[str]:
    start = pd.Period(last_quarter, freq="Q") + 1
    return [str(start + offset) for offset in range(horizon)]


def _mlflow_client() -> MlflowClient:
    tracking.configure_mlflow()
    return MlflowClient()


def _champion_model_version(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except Exception as exc:  # pragma: no cover - MLflow exception classes vary by version
        raise HTTPException(
            status_code=503,
            detail=(
                f"No MLflow champion alias found for {REGISTERED_MODEL_NAME!r}. "
                "Run `python -m src.models.registry` after model orchestrators log runs."
            ),
        ) from exc


def _model_family(client: MlflowClient, version) -> str:
    tags = dict(getattr(version, "tags", {}) or {})
    if tags.get("model_family"):
        return str(tags["model_family"])
    if getattr(version, "run_id", None):
        run = client.get_run(version.run_id)
        if run.data.tags.get("model_family"):
            return str(run.data.tags["model_family"])
    raise HTTPException(
        status_code=503,
        detail=f"Champion version {version.version} is missing a model_family tag.",
    )


def _metric_payload(client: MlflowClient, version) -> dict[str, object]:
    if not getattr(version, "run_id", None):
        raise HTTPException(
            status_code=503,
            detail=f"Champion version {version.version} is missing its source run.",
        )
    run = client.get_run(version.run_id)
    metrics = {
        key: float(value)
        for key, value in run.data.metrics.items()
        if key == "rmse_overall"
        or key == "mae_overall"
        or key.startswith("rmse_h")
        or key.startswith("mae_h")
    }
    return {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": str(version.version),
        "model_family": _model_family(client, version),
        "run_id": version.run_id,
        "metrics": metrics,
    }


def _current_lstm_frame() -> pd.DataFrame:
    series = load_target_series(CURATED_DATA_PATH)
    exog = load_lstm_feature_frame(CURATED_DATA_PATH)
    return pd.concat([series.rename(TARGET_COLUMN), exog], axis=1)


def _load_lstm_fit(client: MlflowClient, version, model_uri: str) -> LSTMDirectFit:
    import mlflow.keras

    model = mlflow.keras.load_model(model_uri)
    scaler_path = client.download_artifacts(
        version.run_id,
        "model_preprocessing/lstm_scaler.json",
    )
    scaler_data = json.loads(Path(scaler_path).read_text(encoding="utf-8"))
    columns = tuple(scaler_data["feature_columns"])
    mean = pd.Series(scaler_data["scaler_mean"], dtype=float).loc[list(columns)]
    scale = pd.Series(scaler_data["scaler_scale"], dtype=float).loc[list(columns)]
    return LSTMDirectFit(
        model=model,
        scaler=TrainWindowScaler(columns=columns, mean_=mean, scale_=scale),
        feature_columns=columns,
        target_column=str(scaler_data.get("target_column", TARGET_COLUMN)),
        lookback=int(scaler_data["lookback"]),
        horizon=int(scaler_data["horizon"]),
    )


def _sarima_forecast_metadata(values) -> tuple[str, list[str]]:
    index = getattr(values, "index", None)
    if index is None or len(index) == 0:
        raise HTTPException(
            status_code=503,
            detail="SARIMA champion forecast did not include a date index.",
        )
    try:
        periods = pd.PeriodIndex(index, freq="Q")
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="SARIMA champion forecast index cannot be interpreted as quarters.",
        ) from exc
    forecast_origin = str(periods[0] - 1)
    quarters = [str(period) for period in periods]
    return forecast_origin, quarters


def _forecast_values(
    client: MlflowClient,
    version,
    family: str,
    horizon: int,
) -> ModelForecast:
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
    if family == "sarima":
        import mlflow.statsmodels

        model = mlflow.statsmodels.load_model(model_uri)
        values = model.forecast(steps=horizon)
        forecast_origin, quarters = _sarima_forecast_metadata(values)
        return ModelForecast(
            values=np.asarray(values, dtype=float).tolist(),
            forecast_origin=forecast_origin,
            quarters=quarters,
        )
    if family == "lstm":
        fitted = _load_lstm_fit(client, version, model_uri)
        if horizon > fitted.horizon:
            raise HTTPException(
                status_code=400,
                detail=f"LSTM champion only supports up to {fitted.horizon} horizons.",
            )
        current_frame = _current_lstm_frame()
        values = forecast_from_fit(fitted, current_frame)[:horizon]
        clean_frame = (
            current_frame.loc[:, list(fitted.feature_columns)].dropna().astype(float).sort_index()
        )
        forecast_origin = str(clean_frame.index[-1])
        return ModelForecast(
            values=np.asarray(values, dtype=float).tolist(),
            forecast_origin=forecast_origin,
            quarters=next_quarters(forecast_origin, horizon),
        )
    raise HTTPException(
        status_code=503,
        detail=f"Champion model family {family!r} is not servable by this API.",
    )


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "curated_dataset_available": CURATED_DATA_PATH.exists(),
        "curated_dataset": str(CURATED_DATA_PATH),
        "registered_model_name": REGISTERED_MODEL_NAME,
        "champion_alias": CHAMPION_ALIAS,
    }


@app.get("/features")
def features() -> dict[str, object]:
    df = load_curated_data()
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "start_quarter": df["quarter"].iloc[0],
        "end_quarter": df["quarter"].iloc[-1],
    }


@app.get("/models")
def models() -> dict[str, object]:
    client = _mlflow_client()
    registered = []
    for model in client.search_registered_models():
        versions = []
        model_versions = client.search_model_versions(f"name = '{model.name}'")
        for version in sorted(model_versions, key=lambda item: int(item.version)):
            versions.append(
                {
                    "version": str(version.version),
                    "run_id": version.run_id,
                    "status": version.status,
                    "aliases": list(getattr(version, "aliases", []) or []),
                    "tags": dict(getattr(version, "tags", {}) or {}),
                }
            )
        registered.append(
            {
                "name": model.name,
                "aliases": dict(getattr(model, "aliases", {}) or {}),
                "versions": versions,
            }
        )
    return {"registered_models": registered}


@app.get("/metrics")
def metrics() -> dict[str, object]:
    client = _mlflow_client()
    version = _champion_model_version(client)
    return _metric_payload(client, version)


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    load_curated_data()
    client = _mlflow_client()
    version = _champion_model_version(client)
    family = _model_family(client, version)
    forecast_result = _forecast_values(client, version, family, request.horizon)
    return ForecastResponse(
        model_name=REGISTERED_MODEL_NAME,
        model_version=str(version.version),
        model_family=family,
        run_id=version.run_id,
        horizon=request.horizon,
        forecast=[round(float(value), 4) for value in forecast_result.values],
        quarters=forecast_result.quarters,
        forecast_origin=forecast_result.forecast_origin,
    )
