"""FastAPI service serving CPI forecasts for every trained model family.

Each family's most recent finished MLflow run is loaded directly (see
``src/models/registry.py``) -- there is no promoted "champion" model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.models import elastic_net, ensemble, sarima, sarimax, tracking
from src.models.elastic_net import (
    TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
    TRIMMED_MEAN_TARGET_COLUMN,
    load_elastic_net_feature_frame,
)
from src.models.evaluation import TARGET_COLUMN, load_target_series
from src.models.registry import (
    TRIMMED_MEAN_MODEL_FAMILY_TAGS,
    _client_and_experiment,
    _latest_model_run_id,
    _logged_model_uri,
)


CURATED_DATA_PATH = Path("data/curated/quarterly_macro_features.csv")
INTERVAL_COVERAGE_REPORT_PATH = Path("reports/model_interval_coverage.csv")
INTERVAL_CALIBRATION_FACTORS_PATH = Path("reports/model_interval_calibration_factors.csv")
INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH = Path(
    "reports/model_interval_calibration_validation.csv"
)
TRIMMED_MEAN_INTERVAL_COVERAGE_REPORT_PATH = Path(
    "reports/model_interval_coverage_trimmed_mean.csv"
)
TRIMMED_MEAN_INTERVAL_CALIBRATION_FACTORS_PATH = Path(
    "reports/model_interval_calibration_factors_trimmed_mean.csv"
)
TRIMMED_MEAN_INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH = Path(
    "reports/model_interval_calibration_validation_trimmed_mean.csv"
)
MAX_FORECAST_HORIZON = 8
VALIDATED_INTERVAL_LOWER = 0.1
VALIDATED_INTERVAL_UPPER = 0.9
TRIMMED_MEAN_FAMILY_RUN_TAGS = {
    **TRIMMED_MEAN_MODEL_FAMILY_TAGS,
    "sarimax_trimmed_mean_primary": "sarimax_trimmed_mean_primary",
    "ensemble": "trimmed_mean_ensemble",
}

app = FastAPI(
    title="Australian CPI Forecast API",
    version="0.2.0",
    description="Portfolio API serving CPI forecasts for every trained model family.",
)


class AllForecastsRequest(BaseModel):
    horizon: int = Field(default=8, ge=1, le=MAX_FORECAST_HORIZON)
    n_sims: int = Field(default=1000, ge=100, le=5000)
    interval_lower: float = Field(default=VALIDATED_INTERVAL_LOWER, ge=0.0, lt=0.5)
    interval_upper: float = Field(default=VALIDATED_INTERVAL_UPPER, gt=0.5, le=1.0)


class FamilyForecast(BaseModel):
    model_family: str
    run_id: str
    horizon: int
    horizon_cap: int | None
    forecast: list[float]
    interval_lower: list[float]
    interval_upper: list[float]
    empirical_coverage: list[float | None]
    coverage_n: list[int | None]
    significantly_miscalibrated: list[bool | None]
    quarters: list[str]
    forecast_origin: str


class UnavailableFamily(BaseModel):
    model_family: str
    reason: str


class AllForecastsResponse(BaseModel):
    requested_horizon: int
    models: list[FamilyForecast]
    unavailable: list[UnavailableFamily]


@dataclass(frozen=True)
class FamilyForecastData:
    forecast: list[float]
    draws: np.ndarray
    quarters: list[str]
    forecast_origin: str
    horizon_served: int
    horizon_cap: int | None


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


def _current_elastic_net_frame(
    target_column: str = TARGET_COLUMN,
    feature_columns: tuple[str, ...] = elastic_net.ELASTIC_NET_FEATURE_COLUMNS,
) -> pd.DataFrame:
    series = load_target_series(CURATED_DATA_PATH, target_column=target_column)
    exog = load_elastic_net_feature_frame(
        CURATED_DATA_PATH,
        feature_columns=feature_columns,
    )
    return pd.concat([series.rename(target_column), exog], axis=1)


def _load_elastic_net_fit(model_uri: str):
    """Load the logged ElasticNetDirectFit via the sklearn flavor.

    This project's serving code always uses MLflow's flavor-specific loaders
    (``mlflow.statsmodels`` and here ``mlflow.sklearn``)
    rather than the generic pyfunc flavor, so it gets back the real
    ``ElasticNetDirectFit`` object -- including its ``predict_next`` method --
    the same way the SARIMA branch gets back its real fitted object.
    """
    import mlflow.sklearn

    return mlflow.sklearn.load_model(model_uri)


def _load_curated_frame_for_group_d() -> pd.DataFrame:
    df = pd.read_csv(CURATED_DATA_PATH)
    if "quarter" not in df.columns:
        raise RuntimeError(
            f"Curated dataset at {CURATED_DATA_PATH} must contain quarter column."
        )
    df.index = pd.PeriodIndex(df.pop("quarter").astype(str), freq="Q")
    return df.sort_index()


def _sarima_forecast_metadata(values) -> tuple[str, list[str]]:
    index = getattr(values, "index", None)
    if index is None or len(index) == 0:
        raise HTTPException(
            status_code=503,
            detail="SARIMA forecast did not include a date index.",
        )
    try:
        periods = pd.PeriodIndex(index, freq="Q")
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="SARIMA forecast index cannot be interpreted as quarters.",
        ) from exc
    forecast_origin = str(periods[0] - 1)
    quarters = [str(period) for period in periods]
    return forecast_origin, quarters


def _interval_from_paths(
    draws: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[list[float], list[float]]:
    lower_values, upper_values = np.percentile(
        np.asarray(draws, dtype=float),
        [lower * 100, upper * 100],
        axis=0,
    )
    return lower_values.astype(float).tolist(), upper_values.astype(float).tolist()


def _parse_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


@lru_cache(maxsize=1)
def _load_interval_coverage_report(
    interval_coverage_report_path: Path | None = None,
    interval_calibration_validation_report_path: Path | None = None,
) -> dict[tuple[str, int], dict[str, object]]:
    """Load non-circular interval diagnostics for the default served bounds."""
    interval_coverage_report_path = (
        INTERVAL_COVERAGE_REPORT_PATH
        if interval_coverage_report_path is None
        else interval_coverage_report_path
    )
    interval_calibration_validation_report_path = (
        INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH
        if interval_calibration_validation_report_path is None
        else interval_calibration_validation_report_path
    )
    validation = _load_interval_calibration_validation_report(
        interval_calibration_validation_report_path
    )
    if validation:
        return validation
    return _load_full_sample_interval_coverage_report(interval_coverage_report_path)

def _load_interval_calibration_validation_report(
    path: Path | None = None,
) -> dict[tuple[str, int], dict[str, object]]:
    """Load held-out calibrated coverage diagnostics used for API disclosure."""
    path = INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH if path is None else path
    if not path.exists():
        return {}
    try:
        report = pd.read_csv(path)
    except (OSError, ValueError):
        return {}

    required = {
        "model",
        "horizon",
        "validation_n",
        "target_coverage",
        "calibrated_empirical_coverage",
        "calibrated_significantly_miscalibrated",
    }
    if not required.issubset(report.columns):
        return {}

    report = report.loc[~report["horizon"].astype(str).eq("overall")].copy()
    report["horizon"] = pd.to_numeric(report["horizon"], errors="coerce")
    report["validation_n"] = pd.to_numeric(report["validation_n"], errors="coerce")
    report["target_coverage"] = pd.to_numeric(report["target_coverage"], errors="coerce")
    report["calibrated_empirical_coverage"] = pd.to_numeric(
        report["calibrated_empirical_coverage"],
        errors="coerce",
    )
    report = report.dropna(
        subset=[
            "model",
            "horizon",
            "validation_n",
            "target_coverage",
            "calibrated_empirical_coverage",
            "calibrated_significantly_miscalibrated",
        ]
    )

    coverage: dict[tuple[str, int], dict[str, object]] = {}
    for _, row in report.iterrows():
        key = (str(row["model"]), int(row["horizon"]))
        coverage[key] = {
            "nominal_coverage": float(row["target_coverage"]),
            "empirical_coverage": float(row["calibrated_empirical_coverage"]),
            "coverage_n": int(row["validation_n"]),
            "significantly_miscalibrated": _parse_bool(
                row["calibrated_significantly_miscalibrated"]
            ),
        }
    return coverage


def _load_full_sample_interval_coverage_report(
    path: Path | None = None,
) -> dict[tuple[str, int], dict[str, object]]:
    """Fallback loader for legacy/raw coverage reports when validation is absent."""
    path = INTERVAL_COVERAGE_REPORT_PATH if path is None else path
    if not path.exists():
        return {}
    try:
        report = pd.read_csv(path)
    except (OSError, ValueError):
        return {}

    required = {
        "model",
        "horizon",
        "n",
        "nominal_coverage",
        "empirical_coverage",
        "significantly_miscalibrated",
    }
    if not required.issubset(report.columns):
        return {}

    report = report.loc[~report["horizon"].astype(str).eq("overall")].copy()
    report["horizon"] = pd.to_numeric(report["horizon"], errors="coerce")
    report = report.dropna(
        subset=[
            "model",
            "horizon",
            "n",
            "nominal_coverage",
            "empirical_coverage",
            "significantly_miscalibrated",
        ]
    )

    coverage: dict[tuple[str, int], dict[str, object]] = {}
    for _, row in report.iterrows():
        key = (str(row["model"]), int(row["horizon"]))
        coverage[key] = {
            "nominal_coverage": float(row["nominal_coverage"]),
            "empirical_coverage": float(row["empirical_coverage"]),
            "coverage_n": int(row["n"]),
            "significantly_miscalibrated": _parse_bool(
                row["significantly_miscalibrated"]
            ),
        }
    return coverage


def _validated_interval_bounds(lower: float, upper: float) -> bool:
    return lower == VALIDATED_INTERVAL_LOWER and upper == VALIDATED_INTERVAL_UPPER


def _coverage_fields_for_family(
    family: str,
    horizon_served: int,
    lower: float,
    upper: float,
    interval_coverage_report_path: Path | None = None,
    interval_calibration_validation_report_path: Path | None = None,
) -> tuple[list[float | None], list[int | None], list[bool | None]]:
    if not _validated_interval_bounds(lower, upper):
        return (
            [None] * horizon_served,
            [None] * horizon_served,
            [None] * horizon_served,
        )

    coverage = _load_interval_coverage_report(
        interval_coverage_report_path,
        interval_calibration_validation_report_path,
    )
    requested_nominal = upper - lower
    empirical: list[float | None] = []
    sample_sizes: list[int | None] = []
    significant: list[bool | None] = []
    for horizon in range(1, horizon_served + 1):
        row = coverage.get((family, horizon))
        if row is not None and not np.isclose(
            float(row["nominal_coverage"]),
            requested_nominal,
        ):
            row = None
        empirical.append(
            None if row is None else float(row["empirical_coverage"])
        )
        sample_sizes.append(None if row is None else int(row["coverage_n"]))
        significant.append(
            None if row is None else bool(row["significantly_miscalibrated"])
        )
    return empirical, sample_sizes, significant


@lru_cache(maxsize=1)
def _load_interval_calibration_factors(
    path: Path | None = None,
) -> dict[tuple[str, int], dict[str, float]]:
    """Load static interval scale factors for the default validated interval."""
    path = INTERVAL_CALIBRATION_FACTORS_PATH if path is None else path
    if not path.exists():
        return {}
    try:
        factors = pd.read_csv(path)
    except (OSError, ValueError):
        return {}

    required = {"model", "horizon", "scale_factor"}
    if not required.issubset(factors.columns):
        return {}

    factors = factors.copy()
    factors["horizon"] = pd.to_numeric(factors["horizon"], errors="coerce")
    factors["scale_factor"] = pd.to_numeric(factors["scale_factor"], errors="coerce")
    if "target_coverage" in factors:
        factors["target_coverage"] = pd.to_numeric(factors["target_coverage"], errors="coerce")
    factors = factors.dropna(subset=["model", "horizon", "scale_factor"])

    result: dict[tuple[str, int], dict[str, float]] = {}
    for _, row in factors.iterrows():
        scale_factor = float(row["scale_factor"])
        if not np.isfinite(scale_factor) or scale_factor < 0:
            continue
        value = {"scale_factor": scale_factor}
        if "target_coverage" in factors and pd.notna(row.get("target_coverage")):
            value["target_coverage"] = float(row["target_coverage"])
        result[(str(row["model"]), int(row["horizon"]))] = value
    return result


def _calibrated_interval_bounds_for_family(
    family: str,
    forecast: list[float],
    interval_lower: list[float],
    interval_upper: list[float],
    lower: float,
    upper: float,
    interval_calibration_factors_path: Path | None = None,
) -> tuple[list[float], list[float]]:
    if not _validated_interval_bounds(lower, upper):
        return interval_lower, interval_upper

    factors = _load_interval_calibration_factors(interval_calibration_factors_path)
    requested_nominal = upper - lower
    calibrated_lower: list[float] = []
    calibrated_upper: list[float] = []
    for position, (point, raw_lower, raw_upper) in enumerate(
        zip(forecast, interval_lower, interval_upper),
        start=1,
    ):
        row = factors.get((family, position))
        if row is None or (
            "target_coverage" in row
            and not np.isclose(float(row["target_coverage"]), requested_nominal)
        ):
            calibrated_lower.append(float(raw_lower))
            calibrated_upper.append(float(raw_upper))
            continue

        raw_half_width = (float(raw_upper) - float(raw_lower)) / 2.0
        calibrated_half_width = float(row["scale_factor"]) * raw_half_width
        calibrated_lower.append(float(point) - calibrated_half_width)
        calibrated_upper.append(float(point) + calibrated_half_width)
    return calibrated_lower, calibrated_upper


def _sarima_family_forecast(
    model_uri: str,
    requested_horizon: int,
    n_sims: int,
    seed: int,
) -> FamilyForecastData:
    import mlflow.statsmodels

    fitted = mlflow.statsmodels.load_model(model_uri)
    point = fitted.forecast(steps=requested_horizon)
    draws = sarima.simulate_paths_from_fit(
        fitted,
        steps=requested_horizon,
        n_sims=n_sims,
        seed=seed,
    )
    forecast_origin, quarters = _sarima_forecast_metadata(point)
    return FamilyForecastData(
        forecast=np.asarray(point, dtype=float).tolist(),
        draws=draws,
        quarters=quarters,
        forecast_origin=forecast_origin,
        horizon_served=requested_horizon,
        horizon_cap=None,
    )


def _elastic_net_family_forecast(
    model_uri: str,
    requested_horizon: int,
    n_sims: int,
    seed: int,
    target_column: str = TARGET_COLUMN,
    feature_columns: tuple[str, ...] = elastic_net.ELASTIC_NET_FEATURE_COLUMNS,
) -> FamilyForecastData:
    fitted = _load_elastic_net_fit(model_uri)
    current_frame = _current_elastic_net_frame(
        target_column=target_column,
        feature_columns=feature_columns,
    )
    horizon_to_value = dict(zip(fitted.horizons, fitted.predict_next(current_frame)))
    missing = [
        horizon
        for horizon in range(1, requested_horizon + 1)
        if horizon not in horizon_to_value
    ]
    if missing:
        raise RuntimeError(
            f"Elastic Net model does not have fitted horizons {missing}."
        )
    forecast = [
        float(horizon_to_value[horizon])
        for horizon in range(1, requested_horizon + 1)
    ]
    draws = elastic_net.simulate_paths_from_fit(
        fitted,
        current_frame,
        steps=requested_horizon,
        n_sims=n_sims,
        seed=seed,
    )
    clean_frame = (
        current_frame.loc[:, list(fitted.feature_columns)].dropna().astype(float).sort_index()
    )
    forecast_origin = str(clean_frame.index[-1])
    return FamilyForecastData(
        forecast=forecast,
        draws=draws,
        quarters=next_quarters(forecast_origin, requested_horizon),
        forecast_origin=forecast_origin,
        horizon_served=requested_horizon,
        horizon_cap=None,
    )


def _sarimax_group_d_family_forecast(
    model_uri: str,
    requested_horizon: int,
    n_sims: int,
    seed: int,
) -> FamilyForecastData:
    import mlflow.statsmodels

    curated_frame = _load_curated_frame_for_group_d()
    fitted = mlflow.statsmodels.load_model(model_uri)
    try:
        future = sarimax.group_d_future_exog(curated_frame)
    except KeyError as exc:
        raise RuntimeError(
            "SARIMAX Group D future exog could not be built; curated dataset "
            f"is missing required column {exc.args[0]!r}."
        ) from exc
    point = fitted.get_forecast(steps=1, exog=future).predicted_mean
    draws = sarimax.simulate_paths_from_fit(
        fitted,
        future,
        steps=1,
        n_sims=n_sims,
        seed=seed,
    )
    last_quarter = pd.Period(curated_frame.sort_index().index[-1], freq="Q")
    return FamilyForecastData(
        forecast=np.asarray(point, dtype=float).tolist(),
        draws=draws,
        quarters=next_quarters(str(last_quarter), 1),
        forecast_origin=str(last_quarter),
        horizon_served=1,
        horizon_cap=1,
    )


def _sarimax_trimmed_mean_family_forecast(
    model_uri: str,
    requested_horizon: int,
    n_sims: int,
    seed: int,
) -> FamilyForecastData:
    del requested_horizon
    import mlflow.statsmodels

    curated_frame = _load_curated_frame_for_group_d()
    fitted = mlflow.statsmodels.load_model(model_uri)
    try:
        future = sarimax.trimmed_mean_primary_future_exog(curated_frame)
    except KeyError as exc:
        raise RuntimeError(
            "SARIMAX trimmed-mean primary future exog could not be built; "
            f"curated dataset is missing required column {exc.args[0]!r}."
        ) from exc
    point = fitted.get_forecast(steps=1, exog=future).predicted_mean
    draws = sarimax.simulate_paths_from_fit(
        fitted,
        future,
        steps=1,
        n_sims=n_sims,
        seed=seed,
    )
    last_quarter = pd.Period(curated_frame.sort_index().index[-1], freq="Q")
    return FamilyForecastData(
        forecast=np.asarray(point, dtype=float).tolist(),
        draws=draws,
        quarters=next_quarters(str(last_quarter), 1),
        forecast_origin=str(last_quarter),
        horizon_served=1,
        horizon_cap=1,
    )


def _ensemble_family_forecast(
    model_uri: str,
    requested_horizon: int,
    n_sims: int,
    seed: int,
) -> FamilyForecastData:
    del model_uri
    _, client, experiment = _client_and_experiment()
    sarima_uri = _logged_model_uri(
        client,
        _latest_model_run_id(client, experiment.experiment_id, "sarima"),
    )
    elastic_net_uri = _logged_model_uri(
        client,
        _latest_model_run_id(client, experiment.experiment_id, "elastic_net"),
    )
    sarima_result = _sarima_family_forecast(
        sarima_uri,
        requested_horizon,
        n_sims,
        seed,
    )
    elastic_net_result = _elastic_net_family_forecast(
        elastic_net_uri,
        requested_horizon,
        n_sims,
        seed + 1,
    )
    if sarima_result.horizon_served != requested_horizon:
        raise RuntimeError("SARIMA component did not serve the requested ensemble horizon.")
    if elastic_net_result.horizon_served != requested_horizon:
        raise RuntimeError(
            "Elastic Net component did not serve the requested ensemble horizon."
        )
    if sarima_result.quarters != elastic_net_result.quarters:
        raise RuntimeError("Ensemble components returned different forecast quarters.")
    if sarima_result.forecast_origin != elastic_net_result.forecast_origin:
        raise RuntimeError("Ensemble components returned different forecast origins.")

    horizons = tuple(range(1, requested_horizon + 1))
    weights = ensemble.horizon_rmse_weights(horizons=horizons)
    forecast = ensemble.combine_point_forecasts(
        sarima_result.forecast,
        elastic_net_result.forecast,
        weights=weights,
        horizons=horizons,
    )
    draws = ensemble.combine_paths(
        sarima_result.draws,
        elastic_net_result.draws,
        weights=weights,
        horizons=horizons,
    )
    return FamilyForecastData(
        forecast=forecast.astype(float).tolist(),
        draws=draws,
        quarters=sarima_result.quarters,
        forecast_origin=sarima_result.forecast_origin,
        horizon_served=requested_horizon,
        horizon_cap=None,
    )


def _ensemble_trimmed_mean_family_forecast(
    model_uri: str,
    requested_horizon: int,
    n_sims: int,
    seed: int,
) -> FamilyForecastData:
    del model_uri
    _, client, experiment = _client_and_experiment()
    sarima_uri = _logged_model_uri(
        client,
        _latest_model_run_id(
            client,
            experiment.experiment_id,
            "sarima",
            model_family_tag=TRIMMED_MEAN_MODEL_FAMILY_TAGS["sarima"],
        ),
    )
    elastic_net_uri = _logged_model_uri(
        client,
        _latest_model_run_id(
            client,
            experiment.experiment_id,
            "elastic_net",
            model_family_tag=TRIMMED_MEAN_MODEL_FAMILY_TAGS["elastic_net"],
        ),
    )
    sarima_result = _sarima_family_forecast(
        sarima_uri,
        requested_horizon,
        n_sims,
        seed,
    )
    elastic_net_result = _elastic_net_family_forecast(
        elastic_net_uri,
        requested_horizon,
        n_sims,
        seed + 1,
        target_column=TRIMMED_MEAN_TARGET_COLUMN,
        feature_columns=TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
    )
    if sarima_result.horizon_served != requested_horizon:
        raise RuntimeError(
            "SARIMA component did not serve the requested trimmed-mean ensemble horizon."
        )
    if elastic_net_result.horizon_served != requested_horizon:
        raise RuntimeError(
            "Elastic Net component did not serve the requested trimmed-mean ensemble horizon."
        )
    if sarima_result.quarters != elastic_net_result.quarters:
        raise RuntimeError("Trimmed-mean ensemble components returned different quarters.")
    if sarima_result.forecast_origin != elastic_net_result.forecast_origin:
        raise RuntimeError("Trimmed-mean ensemble components returned different origins.")

    horizons = tuple(range(1, requested_horizon + 1))
    weights = {horizon: ensemble.DEFAULT_WEIGHTS for horizon in horizons}
    forecast = ensemble.combine_point_forecasts(
        sarima_result.forecast,
        elastic_net_result.forecast,
        weights=weights,
        horizons=horizons,
    )
    draws = ensemble.combine_paths(
        sarima_result.draws,
        elastic_net_result.draws,
        weights=weights,
        horizons=horizons,
    )
    return FamilyForecastData(
        forecast=forecast.astype(float).tolist(),
        draws=draws,
        quarters=sarima_result.quarters,
        forecast_origin=sarima_result.forecast_origin,
        horizon_served=requested_horizon,
        horizon_cap=None,
    )


FAMILY_HANDLERS: dict[str, Callable[[str, int, int, int], FamilyForecastData]] = {
    "sarima": _sarima_family_forecast,
    "elastic_net": _elastic_net_family_forecast,
    "sarimax_group_d": _sarimax_group_d_family_forecast,
    "ensemble": _ensemble_family_forecast,
}

TRIMMED_MEAN_FAMILY_HANDLERS: dict[str, Callable[[str, int, int, int], FamilyForecastData]] = {
    "sarima": _sarima_family_forecast,
    "elastic_net": partial(
        _elastic_net_family_forecast,
        target_column=TRIMMED_MEAN_TARGET_COLUMN,
        feature_columns=TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
    ),
    "sarimax_trimmed_mean_primary": _sarimax_trimmed_mean_family_forecast,
    "ensemble": _ensemble_trimmed_mean_family_forecast,
}


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


def _forecast_all_with_handlers(
    request: AllForecastsRequest,
    family_handlers: dict[str, Callable[[str, int, int, int], FamilyForecastData]],
    model_family_tags: dict[str, str] | None = None,
    interval_coverage_report_path: Path | None = None,
    interval_calibration_factors_path: Path | None = None,
    interval_calibration_validation_report_path: Path | None = None,
) -> AllForecastsResponse:
    load_curated_data()
    client = _mlflow_client()
    try:
        _, _, experiment = _client_and_experiment()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    models: list[FamilyForecast] = []
    unavailable: list[UnavailableFamily] = []
    for family, handler in family_handlers.items():
        try:
            model_family_tag = None if model_family_tags is None else model_family_tags.get(family)
            run_id = _latest_model_run_id(
                client,
                experiment.experiment_id,
                family,
                model_family_tag=model_family_tag,
            )
            model_uri = _logged_model_uri(client, run_id)
            result = handler(model_uri, request.horizon, request.n_sims, seed=42)
        except (RuntimeError, MlflowException, OSError) as exc:
            unavailable.append(UnavailableFamily(model_family=family, reason=str(exc)))
            continue

        interval_lower, interval_upper = _interval_from_paths(
            result.draws,
            request.interval_lower,
            request.interval_upper,
        )
        interval_lower, interval_upper = _calibrated_interval_bounds_for_family(
            family=family,
            forecast=[float(value) for value in result.forecast],
            interval_lower=interval_lower,
            interval_upper=interval_upper,
            lower=request.interval_lower,
            upper=request.interval_upper,
            interval_calibration_factors_path=interval_calibration_factors_path,
        )
        empirical_coverage, coverage_n, significantly_miscalibrated = (
            _coverage_fields_for_family(
                family,
                result.horizon_served,
                request.interval_lower,
                request.interval_upper,
                interval_coverage_report_path=interval_coverage_report_path,
                interval_calibration_validation_report_path=(
                    interval_calibration_validation_report_path
                ),
            )
        )
        models.append(
            FamilyForecast(
                model_family=family,
                run_id=run_id,
                horizon=result.horizon_served,
                horizon_cap=result.horizon_cap,
                forecast=[float(value) for value in result.forecast],
                interval_lower=interval_lower,
                interval_upper=interval_upper,
                empirical_coverage=empirical_coverage,
                coverage_n=coverage_n,
                significantly_miscalibrated=significantly_miscalibrated,
                quarters=result.quarters,
                forecast_origin=result.forecast_origin,
            )
        )

    return AllForecastsResponse(
        requested_horizon=request.horizon,
        models=models,
        unavailable=unavailable,
    )


@app.post("/forecast/all", response_model=AllForecastsResponse)
def forecast_all(request: AllForecastsRequest) -> AllForecastsResponse:
    return _forecast_all_with_handlers(request, FAMILY_HANDLERS)


@app.post("/forecast/trimmed-mean/all", response_model=AllForecastsResponse)
def forecast_trimmed_mean_all(request: AllForecastsRequest) -> AllForecastsResponse:
    return _forecast_all_with_handlers(
        request,
        TRIMMED_MEAN_FAMILY_HANDLERS,
        model_family_tags=TRIMMED_MEAN_FAMILY_RUN_TAGS,
        interval_coverage_report_path=TRIMMED_MEAN_INTERVAL_COVERAGE_REPORT_PATH,
        interval_calibration_factors_path=TRIMMED_MEAN_INTERVAL_CALIBRATION_FACTORS_PATH,
        interval_calibration_validation_report_path=(
            TRIMMED_MEAN_INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH
        ),
    )
