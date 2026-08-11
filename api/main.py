"""Minimal FastAPI service for CPI forecasting portfolio deployment.

This service is intentionally lightweight until the SARIMAX model is trained.
It exposes health, feature metadata, simple metric summaries, and a seasonal
naive forecast based on the curated quarterly CPI dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


CURATED_DATA_PATH = Path("data/curated/quarterly_macro_features.csv")
SEASONAL_PERIOD = 4

app = FastAPI(
    title="Australian CPI Forecast API",
    version="0.1.0",
    description="Portfolio API for serving CPI forecast data and baseline forecasts.",
)


class ForecastRequest(BaseModel):
    model: Literal["seasonal_naive"] = "seasonal_naive"
    horizon: int = Field(default=8, ge=1, le=8)


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    forecast: list[float]
    quarters: list[str]


def load_curated_data() -> pd.DataFrame:
    if not CURATED_DATA_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Curated dataset not found at {CURATED_DATA_PATH}",
        )
    df = pd.read_csv(CURATED_DATA_PATH)
    if "quarter" not in df.columns or "cpi_index" not in df.columns:
        raise HTTPException(
            status_code=503,
            detail="Curated dataset must contain quarter and cpi_index columns.",
        )
    return df.sort_values("quarter").reset_index(drop=True)


def next_quarters(last_quarter: str, horizon: int) -> list[str]:
    start = pd.Period(last_quarter, freq="Q") + 1
    return [str(start + offset) for offset in range(horizon)]


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "curated_dataset_available": CURATED_DATA_PATH.exists(),
        "curated_dataset": str(CURATED_DATA_PATH),
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


@app.get("/metrics")
def metrics() -> dict[str, object]:
    df = load_curated_data()
    target = df["cpi_yoy"].dropna()
    return {
        "target": "cpi_yoy",
        "observations": int(target.shape[0]),
        "mean": round(float(target.mean()), 4),
        "std": round(float(target.std()), 4),
        "latest": round(float(target.iloc[-1]), 4),
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    df = load_curated_data()
    recent = df["cpi_index"].dropna().tail(SEASONAL_PERIOD).tolist()
    if len(recent) < SEASONAL_PERIOD:
        raise HTTPException(
            status_code=503,
            detail="At least four quarterly CPI observations are required.",
        )

    values = [recent[i % SEASONAL_PERIOD] for i in range(request.horizon)]
    return ForecastResponse(
        model=request.model,
        horizon=request.horizon,
        forecast=[round(float(value), 4) for value in values],
        quarters=next_quarters(df["quarter"].iloc[-1], request.horizon),
    )
