"""FastAPI service serving CPI forecasts for every trained model family.

Each model is a joblib file in ``models/`` written by its training script.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.models import (
    credit_stress,
    elastic_net,
    ensemble,
    rba_classifier,
    sarima,
    scenario,
    svar,
)
from src.models.elastic_net import (
    TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
    TRIMMED_MEAN_TARGET_COLUMN,
    load_elastic_net_feature_frame,
)
from src.models.evaluation import (
    ELASTIC_NET_MODEL_PATH,
    SARIMA_MODEL_PATH,
    TARGET_COLUMN,
    TRIMMED_MEAN_ELASTIC_NET_MODEL_PATH,
    TRIMMED_MEAN_SARIMA_MODEL_PATH,
    load_target_series,
)


CURATED_DATA_PATH = Path("data/curated/quarterly_macro_features.csv")
INTERVAL_CALIBRATION_FACTORS_PATH = Path("reports/model_interval_calibration_factors.csv")
INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH = Path(
    "reports/model_interval_calibration_validation.csv"
)
TRIMMED_MEAN_INTERVAL_CALIBRATION_FACTORS_PATH = Path(
    "reports/model_interval_calibration_factors_trimmed_mean.csv"
)
TRIMMED_MEAN_INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH = Path(
    "reports/model_interval_calibration_validation_trimmed_mean.csv"
)
MAX_FORECAST_HORIZON = 8
# The scenario endpoint refits an SVAR and simulates per request (~40s at 1000 draws), so it gets a lower cap.
MAX_SCENARIO_SIMS = 2000
MAX_ABS_SHOCK_VALUE = 1000.0
VALIDATED_INTERVAL_LOWER = 0.1
VALIDATED_INTERVAL_UPPER = 0.9
HEADLINE_MODEL_PATHS = {"sarima": SARIMA_MODEL_PATH, "elastic_net": ELASTIC_NET_MODEL_PATH}
TRIMMED_MEAN_MODEL_PATHS = {
    "sarima": TRIMMED_MEAN_SARIMA_MODEL_PATH,
    "elastic_net": TRIMMED_MEAN_ELASTIC_NET_MODEL_PATH,
}
RBA_ACTION_CAVEAT = (
    "RBA action classifications are a documented comparison exercise, not a "
    "validated policy predictor. The threshold baseline is the reportable result, "
    "but no classifier is statistically shown to beat it because paired-bootstrap "
    "confidence intervals cross zero in reports/rba_classifier_evaluation.md."
)
UNCALIBRATED_SCENARIO_NOTE = (
    "The interval is uncalibrated: it uses percentiles other than the validated 10th and 90th."
)
# Four quarters ahead, the same default as svar.forecast_cumulative_unemployment_change_quantiles.
CREDIT_STRESS_DEFAULT_HORIZON = 4

app = FastAPI(
    title="Australian CPI Forecast API",
    version="0.2.0",
    description="Portfolio API serving CPI forecasts for every trained model family.",
)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return the usual 422 without echoing the input, which cannot be sent back when it is NaN."""
    errors = [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


class AllForecastsRequest(BaseModel):
    horizon: int = Field(default=8, ge=1, le=MAX_FORECAST_HORIZON)
    n_sims: int = Field(default=1000, ge=100, le=5000)
    interval_lower: float = Field(default=VALIDATED_INTERVAL_LOWER, ge=0.0, lt=0.5)
    interval_upper: float = Field(default=VALIDATED_INTERVAL_UPPER, gt=0.5, le=1.0)


class FamilyForecast(BaseModel):
    model_family: str
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


class RbaActionModelPrediction(BaseModel):
    model: str
    predicted_action: str
    confidence: float | None
    p_cut: float | None
    p_hold: float | None
    p_hike: float | None
    reportable: bool
    majority_vote_tie_break: bool


class RbaActionResponse(BaseModel):
    target_quarter: str
    forecast_origin: str
    reportable_model: str
    reportable_action: str
    headline_forecast: float
    trimmed_mean_forecast: float
    models: list[RbaActionModelPrediction]
    caveat: str


class CreditRiskScenario(BaseModel):
    name: str
    probability_weight: float
    delta_unemployment_cumulative: float


class CreditStressSegmentResult(BaseModel):
    segment: str
    pd_base: float
    ur_sensitivity: float
    lgd: float
    ead_aud_m: float
    discount_rate: float
    pd_stressed_by_scenario: dict[str, float]
    ecl_aud_m_by_scenario: dict[str, float]
    ecl_aud_m_12m_probability_weighted: float


class CreditRiskStressTestResponse(BaseModel):
    forecast_origin: str
    target_quarter: str
    horizon: int
    scenarios: list[CreditRiskScenario]
    segments: list[CreditStressSegmentResult]
    caveat: str


class ScenarioForecastRequest(BaseModel):
    target: str = Field(default="headline")
    shock_variable: str
    shock_value: float = Field(allow_inf_nan=False, ge=-MAX_ABS_SHOCK_VALUE, le=MAX_ABS_SHOCK_VALUE)
    horizons: list[int] = Field(
        default_factory=lambda: list(range(1, MAX_FORECAST_HORIZON + 1)),
        max_length=MAX_FORECAST_HORIZON,
    )
    n_sims: int = Field(default=1000, ge=100, le=MAX_SCENARIO_SIMS)
    interval_lower: float = Field(default=VALIDATED_INTERVAL_LOWER, ge=0.0, lt=0.5)
    interval_upper: float = Field(default=VALIDATED_INTERVAL_UPPER, gt=0.5, le=1.0)


class ScenarioForecastResponse(BaseModel):
    target: str
    shock_variable: str
    shock_value: float
    shock_size: float
    horizons: list[int]
    forecast: list[float]
    interval_lower: list[float]
    interval_upper: list[float]
    quarters: list[str]
    forecast_origin: str
    caveat: str


@dataclass(frozen=True)
class FamilyForecastData:
    """One model family's point forecast and simulated draws, before intervals are added."""

    forecast: list[float]
    draws: np.ndarray
    quarters: list[str]
    forecast_origin: str
    horizon_served: int
    horizon_cap: int | None


def load_curated_data() -> pd.DataFrame:
    """Read the curated CSV sorted by quarter; 503 if it is missing or lacks the target."""
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
    """The ``horizon`` quarter labels after ``last_quarter``."""
    start = pd.Period(last_quarter, freq="Q") + 1
    return [str(start + offset) for offset in range(horizon)]


def _optional_float(value: object) -> float | None:
    """NaN becomes None (JSON null), anything else a float."""
    if pd.isna(value):
        return None
    return float(value)


def _current_elastic_net_frame(
    target_column: str = TARGET_COLUMN,
    feature_columns: tuple[str, ...] = elastic_net.ELASTIC_NET_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Target plus Elastic Net features from the full curated data, indexed by quarter."""
    series = load_target_series(CURATED_DATA_PATH, target_column=target_column)
    exog = load_elastic_net_feature_frame(
        CURATED_DATA_PATH,
        feature_columns=feature_columns,
    )
    return pd.concat([series.rename(target_column), exog], axis=1)


def _load_model(path: Path):
    """Load a fitted model saved by its training script."""
    if not path.exists():
        raise RuntimeError(f"Saved model {path.name} not found. Train the models first (see README).")
    try:
        return joblib.load(path)
    except Exception as exc:  # a pickle from another statsmodels/scikit-learn version fails in many ways
        raise RuntimeError(
            f"Saved model {path.name} could not be loaded ({type(exc).__name__}). "
            "Retrain it with the pinned package versions (see README)."
        ) from exc


def _sarima_forecast_metadata(values) -> tuple[str, list[str]]:
    """Forecast origin and quarter labels read from the SARIMA forecast's date index."""
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
    """Lower and upper percentiles of the simulated draws at each horizon."""
    lower_values, upper_values = np.percentile(
        np.asarray(draws, dtype=float),
        [lower * 100, upper * 100],
        axis=0,
    )
    return lower_values.astype(float).tolist(), upper_values.astype(float).tolist()


def _public_message(exc: Exception) -> str:
    """Show this project's own error messages; hide file errors, which can name server paths."""
    if isinstance(exc, (RuntimeError, ValueError)):
        return str(exc)
    return f"{type(exc).__name__} while reading a file on the server."


def _parse_bool(value: object) -> bool:
    """Read a CSV true/false cell that may come back as text."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


@lru_cache(maxsize=2)
def _load_interval_coverage_report(
    path: Path | None = None,
) -> dict[tuple[str, int], dict[str, object]]:
    """Load the held-out calibrated coverage report that the API discloses.

    Only the held-out validation report is used. The in-sample coverage report would
    look similar but is circular, so a missing report gives null coverage fields
    instead of quietly switching to it.
    """
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

    coverage = {}
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


def _validated_interval_bounds(lower: float, upper: float) -> bool:
    """True only for the 10th/90th percentile interval that the calibration and coverage reports cover."""
    return lower == VALIDATED_INTERVAL_LOWER and upper == VALIDATED_INTERVAL_UPPER


def _coverage_fields_for_family(
    family: str,
    horizon_served: int,
    lower: float,
    upper: float,
    interval_calibration_validation_report_path: Path | None = None,
) -> tuple[list[float | None], list[int | None], list[bool | None]]:
    """Held-out coverage, sample size and miscalibration flag per horizon; None where there is no matching report row."""
    if not _validated_interval_bounds(lower, upper):
        return (
            [None] * horizon_served,
            [None] * horizon_served,
            [None] * horizon_served,
        )

    coverage = _load_interval_coverage_report(interval_calibration_validation_report_path)
    requested_nominal = upper - lower
    empirical = []
    sample_sizes = []
    significant = []
    for horizon in range(1, horizon_served + 1):
        row = coverage.get((family, horizon))
        if row is None or not np.isclose(float(row["nominal_coverage"]), requested_nominal):
            empirical.append(None)
            sample_sizes.append(None)
            significant.append(None)
        else:
            empirical.append(float(row["empirical_coverage"]))
            sample_sizes.append(int(row["coverage_n"]))
            significant.append(bool(row["significantly_miscalibrated"]))
    return empirical, sample_sizes, significant


@lru_cache(maxsize=2)
def _load_interval_calibration_factors(
    path: Path | None = None,
) -> dict[tuple[str, int], dict[str, float]]:
    """Load static interval scale factors for the default validated interval."""
    path = INTERVAL_CALIBRATION_FACTORS_PATH if path is None else path
    # Without these factors the API would silently serve the narrower, uncalibrated intervals.
    required = {"model", "horizon", "scale_factor"}
    try:
        factors = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503, detail=f"Interval calibration factors {path.name} cannot be read."
        ) from exc
    if not required.issubset(factors.columns):
        raise HTTPException(
            status_code=503, detail=f"Interval calibration factors {path.name} are malformed."
        )

    factors = factors.copy()
    factors["horizon"] = pd.to_numeric(factors["horizon"], errors="coerce")
    factors["scale_factor"] = pd.to_numeric(factors["scale_factor"], errors="coerce")
    if "target_coverage" in factors:
        factors["target_coverage"] = pd.to_numeric(factors["target_coverage"], errors="coerce")
    factors = factors.dropna(subset=["model", "horizon", "scale_factor"])

    result = {}
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
    horizons: Sequence[int] | None = None,
) -> tuple[list[float], list[float]]:
    """Widen the interval around the point forecast; ``horizons`` names each entry's horizon (default 1, 2, ...)."""
    if not _validated_interval_bounds(lower, upper):
        return interval_lower, interval_upper

    factors = _load_interval_calibration_factors(interval_calibration_factors_path)
    requested_nominal = upper - lower
    calibrated_lower = []
    calibrated_upper = []
    horizons = range(1, len(forecast) + 1) if horizons is None else horizons
    for horizon, point, raw_lower, raw_upper in zip(horizons, forecast, interval_lower, interval_upper):
        row = factors.get((family, horizon))
        if row is None or (
            "target_coverage" in row
            and not np.isclose(float(row["target_coverage"]), requested_nominal)
        ):
            calibrated_lower.append(float(raw_lower))
            calibrated_upper.append(float(raw_upper))
            continue

        scale_factor = float(row["scale_factor"])
        calibrated_lower.append(float(point) - scale_factor * (float(point) - float(raw_lower)))
        calibrated_upper.append(float(point) + scale_factor * (float(raw_upper) - float(point)))
    return calibrated_lower, calibrated_upper


def _sarima_family_forecast(
    model_paths: dict[str, Path],
    requested_horizon: int,
    n_sims: int,
    seed: int,
) -> FamilyForecastData:
    """Forecast and simulate from the saved SARIMA model."""
    fitted = _load_model(model_paths["sarima"])
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
    model_paths: dict[str, Path],
    requested_horizon: int,
    n_sims: int,
    seed: int,
    target_column: str = TARGET_COLUMN,
    feature_columns: tuple[str, ...] = elastic_net.ELASTIC_NET_FEATURE_COLUMNS,
    forecast_origin: str | None = None,
) -> FamilyForecastData:
    """Forecast and simulate from the saved Elastic Net model, optionally from an earlier ``forecast_origin``."""
    fitted = _load_model(model_paths["elastic_net"])
    current_frame = _current_elastic_net_frame(
        target_column=target_column,
        feature_columns=feature_columns,
    )
    if forecast_origin is not None:
        origin_period = pd.Period(forecast_origin, freq="Q")
        current_frame = current_frame.loc[current_frame.index <= origin_period]
    horizon_to_value = dict(zip(fitted.horizons, fitted.predict_next(current_frame)))
    missing = [horizon for horizon in range(1, requested_horizon + 1) if horizon not in horizon_to_value]
    if missing:
        raise RuntimeError(f"Elastic Net model does not have fitted horizons {missing}.")
    forecast = [float(horizon_to_value[horizon]) for horizon in range(1, requested_horizon + 1)]
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


def _ensemble_family_forecast(
    model_paths: dict[str, Path],
    requested_horizon: int,
    n_sims: int,
    seed: int,
    target_column: str = TARGET_COLUMN,
    feature_columns: tuple[str, ...] = elastic_net.ELASTIC_NET_FEATURE_COLUMNS,
) -> FamilyForecastData:
    """Blend the SARIMA and Elastic Net forecasts with inverse-RMSE weights."""
    sarima_result = _sarima_family_forecast(model_paths, requested_horizon, n_sims, seed)
    elastic_net_result = _elastic_net_family_forecast(
        model_paths,
        requested_horizon,
        n_sims,
        seed + 1,
        target_column=target_column,
        feature_columns=feature_columns,
        forecast_origin=sarima_result.forecast_origin,
    )
    if sarima_result.quarters != elastic_net_result.quarters:
        raise RuntimeError("Ensemble components returned different forecast quarters.")
    if sarima_result.forecast_origin != elastic_net_result.forecast_origin:
        raise RuntimeError("Ensemble components returned different forecast origins.")

    horizons = tuple(range(1, requested_horizon + 1))
    weights = ensemble.horizon_rmse_weights(
        horizons=horizons,
        path=ensemble.dynamic_weights_source_path(target_column),
    )
    forecast = ensemble.combine_point_forecasts(
        sarima_result.forecast,
        elastic_net_result.forecast,
        weights=weights,
        horizons=horizons,
    )
    draws = ensemble.recenter_paths_to_median(
        ensemble.combine_paths(
            sarima_result.draws,
            elastic_net_result.draws,
            weights=weights,
            horizons=horizons,
        ),
        forecast,
    )
    return FamilyForecastData(
        forecast=forecast.astype(float).tolist(),
        draws=draws,
        quarters=sarima_result.quarters,
        forecast_origin=sarima_result.forecast_origin,
        horizon_served=requested_horizon,
        horizon_cap=None,
    )


def _build_live_rba_action_row(n_sims: int, seed: int) -> pd.DataFrame:
    """The classifier's input row for the next quarter: both ensemble forecasts plus the origin's macro values."""
    curated = load_curated_data()
    required_columns = {
        "quarter",
        "cash_rate",
        "cash_rate_lag1",
        "unemployment_rate_change_lag1",
    }
    missing_columns = sorted(required_columns.difference(curated.columns))
    if missing_columns:
        raise HTTPException(
            status_code=503,
            detail=(
                "Curated dataset is missing RBA action input columns: "
                f"{', '.join(missing_columns)}."
            ),
        )

    headline = FAMILY_HANDLERS["ensemble"](
        HEADLINE_MODEL_PATHS,
        requested_horizon=1,
        n_sims=n_sims,
        seed=seed,
    )
    trimmed_mean = TRIMMED_MEAN_FAMILY_HANDLERS["ensemble"](
        TRIMMED_MEAN_MODEL_PATHS,
        requested_horizon=1,
        n_sims=n_sims,
        seed=seed + 2,
    )
    if headline.quarters[0] != trimmed_mean.quarters[0]:
        raise HTTPException(
            status_code=503,
            detail=(
                "Live RBA action forecasts do not align with each other: "
                f"headline={headline.quarters[0]}, "
                f"trimmed_mean={trimmed_mean.quarters[0]}."
            ),
        )

    target_quarter = headline.quarters[0]
    forecast_origin = str(pd.Period(target_quarter, freq="Q") - 1)
    origin_rows = curated.loc[curated["quarter"].astype(str).eq(forecast_origin)]
    if origin_rows.empty:
        raise HTTPException(
            status_code=503,
            detail=(
                "Curated dataset is missing the RBA action forecast-origin row "
                f"{forecast_origin}."
            ),
        )

    latest = origin_rows.iloc[0]
    null_columns = [column for column in sorted(required_columns) if pd.isna(latest[column])]
    if null_columns:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Curated macro row for {forecast_origin} is missing RBA action inputs: "
                f"{', '.join(null_columns)}."
            ),
        )

    return pd.DataFrame(
        [
            {
                "target_quarter": target_quarter,
                "headline_forecast": float(headline.forecast[0]),
                "trimmed_mean_forecast": float(trimmed_mean.forecast[0]),
                "cash_rate": float(latest["cash_rate"]),
                "cash_rate_lag1": float(latest["cash_rate_lag1"]),
                "unemployment_rate_change_lag1": float(
                    latest["unemployment_rate_change_lag1"]
                ),
            }
        ]
    )


def _rba_action_model_predictions(predictions: pd.DataFrame) -> list[RbaActionModelPrediction]:
    """One response entry per classifier, in MODEL_ORDER."""
    rows = []
    for model in rba_classifier.MODEL_ORDER:
        model_rows = predictions.loc[predictions["model"].eq(model)]
        if model_rows.empty:
            continue
        row = model_rows.iloc[0]
        rows.append(
            RbaActionModelPrediction(
                model=model,
                predicted_action=str(row["predicted_action"]),
                confidence=_optional_float(row["confidence"]),
                p_cut=_optional_float(row["p_cut"]),
                p_hold=_optional_float(row["p_hold"]),
                p_hike=_optional_float(row["p_hike"]),
                reportable=model == "threshold",
                majority_vote_tie_break=bool(row["majority_vote_tie_break"]),
            )
        )
    return rows


def _trimmed_mean_elastic_net_family_forecast(
    model_paths: dict[str, Path],
    requested_horizon: int,
    n_sims: int,
    seed: int,
    forecast_origin: str | None = None,
) -> FamilyForecastData:
    """Elastic Net forecast for trimmed-mean CPI, using its own target and features."""
    return _elastic_net_family_forecast(
        model_paths,
        requested_horizon,
        n_sims,
        seed,
        target_column=TRIMMED_MEAN_TARGET_COLUMN,
        feature_columns=TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
        forecast_origin=forecast_origin,
    )


def _trimmed_mean_ensemble_family_forecast(
    model_paths: dict[str, Path],
    requested_horizon: int,
    n_sims: int,
    seed: int,
) -> FamilyForecastData:
    """Ensemble forecast for trimmed-mean CPI, using its own target and features."""
    return _ensemble_family_forecast(
        model_paths,
        requested_horizon,
        n_sims,
        seed,
        target_column=TRIMMED_MEAN_TARGET_COLUMN,
        feature_columns=TRIMMED_MEAN_ELASTIC_NET_PRIMARY_WTI_FEATURE_COLUMNS,
    )


# Which function produces each model family's forecast, per target.
FAMILY_HANDLERS = {
    "sarima": _sarima_family_forecast,
    "elastic_net": _elastic_net_family_forecast,
    "ensemble": _ensemble_family_forecast,
}

TRIMMED_MEAN_FAMILY_HANDLERS = {
    "sarima": _sarima_family_forecast,
    "elastic_net": _trimmed_mean_elastic_net_family_forecast,
    "ensemble": _trimmed_mean_ensemble_family_forecast,
}


@app.get("/health")
def health(response: Response) -> dict[str, object]:
    """Report ok only when every file the forecast endpoints need is present (503 otherwise)."""
    required_files = [
        CURATED_DATA_PATH,
        INTERVAL_CALIBRATION_FACTORS_PATH,
        TRIMMED_MEAN_INTERVAL_CALIBRATION_FACTORS_PATH,
        *HEADLINE_MODEL_PATHS.values(),
        *TRIMMED_MEAN_MODEL_PATHS.values(),
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        response.status_code = 503
    return {
        "status": "degraded" if missing else "ok",
        "curated_dataset_available": CURATED_DATA_PATH.exists(),
        "curated_dataset": str(CURATED_DATA_PATH),
        "missing_files": missing,
    }


@app.get("/features")
def features() -> dict[str, object]:
    """Row count, columns and quarter range of the curated dataset."""
    df = load_curated_data()
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "start_quarter": df["quarter"].iloc[0],
        "end_quarter": df["quarter"].iloc[-1],
    }


def _forecast_all_with_handlers(
    request: AllForecastsRequest,
    family_handlers: dict,
    model_paths: dict[str, Path],
    interval_calibration_factors_path: Path | None = None,
    interval_calibration_validation_report_path: Path | None = None,
) -> AllForecastsResponse:
    """Run every family's handler, then add calibrated intervals and coverage; failed families go to ``unavailable``."""
    load_curated_data()

    models = []
    unavailable = []
    # Elastic Net forecasts from the same origin as SARIMA so the families line up.
    anchor_forecast_origin = None
    for family, handler in family_handlers.items():
        try:
            if family == "elastic_net" and anchor_forecast_origin is not None:
                result = handler(
                    model_paths,
                    request.horizon,
                    request.n_sims,
                    seed=42,
                    forecast_origin=anchor_forecast_origin,
                )
            else:
                result = handler(model_paths, request.horizon, request.n_sims, seed=42)
        except (RuntimeError, OSError) as exc:
            unavailable.append(UnavailableFamily(model_family=family, reason=_public_message(exc)))
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
        empirical_coverage, coverage_n, significantly_miscalibrated = _coverage_fields_for_family(
            family,
            result.horizon_served,
            request.interval_lower,
            request.interval_upper,
            interval_calibration_validation_report_path=interval_calibration_validation_report_path,
        )
        models.append(
            FamilyForecast(
                model_family=family,
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
        if family == "sarima":
            anchor_forecast_origin = result.forecast_origin

    return AllForecastsResponse(
        requested_horizon=request.horizon,
        models=models,
        unavailable=unavailable,
    )


@app.post("/forecast/all", response_model=AllForecastsResponse)
def forecast_all(request: AllForecastsRequest) -> AllForecastsResponse:
    """Headline CPI forecasts from every model family."""
    return _forecast_all_with_handlers(request, FAMILY_HANDLERS, HEADLINE_MODEL_PATHS)


@app.post("/forecast/trimmed-mean/all", response_model=AllForecastsResponse)
def forecast_trimmed_mean_all(request: AllForecastsRequest) -> AllForecastsResponse:
    """Trimmed-mean CPI forecasts from every model family."""
    return _forecast_all_with_handlers(
        request,
        TRIMMED_MEAN_FAMILY_HANDLERS,
        TRIMMED_MEAN_MODEL_PATHS,
        interval_calibration_factors_path=TRIMMED_MEAN_INTERVAL_CALIBRATION_FACTORS_PATH,
        interval_calibration_validation_report_path=(
            TRIMMED_MEAN_INTERVAL_CALIBRATION_VALIDATION_REPORT_PATH
        ),
    )


@lru_cache(maxsize=1)
def _rba_action_response() -> RbaActionResponse:
    """Fit and score the classifiers once. The seeds and data are fixed, so the answer only changes on redeploy."""
    n_sims = rba_classifier.DEFAULT_N_SIMS
    seed = rba_classifier.DEFAULT_SEED
    try:
        train = rba_classifier.assemble_policy_sample()
        live_row = _build_live_rba_action_row(n_sims=n_sims, seed=seed)
        threshold_simulation_frame = rba_classifier._load_threshold_simulation_frame(
            curated_path=CURATED_DATA_PATH
        )
        predictions = rba_classifier.predict_single_quarter(
            train,
            live_row,
            threshold_simulation_frame=threshold_simulation_frame,
            threshold_n_sims=n_sims,
            threshold_seed=seed,
            sarima_series=load_target_series(CURATED_DATA_PATH, target_column=TARGET_COLUMN),
        )
    except (RuntimeError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=_public_message(exc)) from exc

    threshold = predictions.loc[predictions["model"].eq("threshold")].iloc[0]
    live = live_row.iloc[0]
    return RbaActionResponse(
        target_quarter=str(live["target_quarter"]),
        forecast_origin=str(pd.Period(str(live["target_quarter"]), freq="Q") - 1),
        reportable_model="threshold",
        reportable_action=str(threshold["predicted_action"]),
        headline_forecast=float(live["headline_forecast"]),
        trimmed_mean_forecast=float(live["trimmed_mean_forecast"]),
        models=_rba_action_model_predictions(predictions),
        caveat=RBA_ACTION_CAVEAT,
    )


@app.get("/rba-action", response_model=RbaActionResponse)
def rba_action() -> RbaActionResponse:
    """Next quarter's cut/hold/hike call from every classifier; the threshold rule is the reportable one."""
    return _rba_action_response()


@app.get("/credit-risk/stress-test", response_model=CreditRiskStressTestResponse)
def credit_risk_stress_test(
    horizon: Annotated[int, Query(ge=1, le=MAX_FORECAST_HORIZON)] = CREDIT_STRESS_DEFAULT_HORIZON,
) -> CreditRiskStressTestResponse:
    """Illustrative 12-month expected credit loss under downside, base and upside unemployment scenarios."""
    scenario_names = list(credit_stress.SCENARIO_QUANTILES)
    quantiles = [credit_stress.SCENARIO_QUANTILES[name] for name in scenario_names]
    try:
        deltas, forecast_origin = svar.forecast_cumulative_unemployment_change_quantiles(
            horizon=horizon, quantiles=quantiles
        )
        target_quarter = next_quarters(forecast_origin, horizon)[-1]
        scenario_deltas = dict(zip(scenario_names, deltas))
        stress_frame = credit_stress.run_credit_stress_test(scenario_deltas=scenario_deltas)
    except (RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail=_public_message(exc)) from exc

    scenarios = [
        CreditRiskScenario(
            name=name,
            probability_weight=float(credit_stress.SCENARIO_PROBABILITY_WEIGHTS[name]),
            delta_unemployment_cumulative=float(scenario_deltas[name]),
        )
        for name in scenario_names
    ]
    segments = []
    for segment, segment_frame in stress_frame.groupby("segment", sort=False):
        pd_stressed_by_scenario = dict(zip(segment_frame["scenario"], segment_frame["pd_stressed"].astype(float)))
        ecl_by_scenario = dict(zip(segment_frame["scenario"], segment_frame["ecl_aud_m"].astype(float)))
        ecl_weighted = float((segment_frame["probability_weight"] * segment_frame["ecl_aud_m"]).sum())
        first_row = segment_frame.iloc[0]
        segments.append(
            CreditStressSegmentResult(
                segment=str(segment),
                pd_base=float(first_row["pd_base"]),
                ur_sensitivity=float(first_row["ur_sensitivity"]),
                lgd=float(first_row["lgd"]),
                ead_aud_m=float(first_row["ead_aud_m"]),
                discount_rate=float(first_row["discount_rate"]),
                pd_stressed_by_scenario=pd_stressed_by_scenario,
                ecl_aud_m_by_scenario=ecl_by_scenario,
                ecl_aud_m_12m_probability_weighted=ecl_weighted,
            )
        )

    return CreditRiskStressTestResponse(
        forecast_origin=forecast_origin,
        target_quarter=target_quarter,
        horizon=int(horizon),
        scenarios=scenarios,
        segments=segments,
        caveat=credit_stress.CREDIT_STRESS_CAVEAT,
    )


@lru_cache(maxsize=32)
def _scenario_response(
    target: str,
    shock_variable: str,
    shock_value: float,
    horizons: tuple[int, ...],
    n_sims: int,
    interval_lower: float,
    interval_upper: float,
) -> ScenarioForecastResponse:
    """Run one scenario. Seeds and data are fixed, so repeating a request gives the same answer."""
    try:
        result = scenario.run_scenario(
            target=target,
            shock_variable=shock_variable,
            shock_value=shock_value,
            horizons=horizons,
            n_sims=n_sims,
            lower_quantile=interval_lower,
            upper_quantile=interval_upper,
            curated_path=CURATED_DATA_PATH,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=503, detail=_public_message(exc)) from exc

    # Same calibration as /forecast/all, so the scenario band is comparable with the baseline band.
    calibrated = _validated_interval_bounds(interval_lower, interval_upper)
    interval_lower_values, interval_upper_values = _calibrated_interval_bounds_for_family(
        family="ensemble",
        forecast=result.forecast,
        interval_lower=result.interval_lower,
        interval_upper=result.interval_upper,
        lower=interval_lower,
        upper=interval_upper,
        interval_calibration_factors_path=(
            TRIMMED_MEAN_INTERVAL_CALIBRATION_FACTORS_PATH if target == "trimmed_mean" else None
        ),
        horizons=result.horizons,
    )
    caveat = result.caveat if calibrated else f"{result.caveat} {UNCALIBRATED_SCENARIO_NOTE}"
    return ScenarioForecastResponse(
        target=result.target,
        shock_variable=result.shock_variable,
        shock_value=result.shock_value,
        shock_size=result.shock_size,
        horizons=list(result.horizons),
        forecast=result.forecast,
        interval_lower=interval_lower_values,
        interval_upper=interval_upper_values,
        quarters=result.quarters,
        forecast_origin=result.forecast_origin,
        caveat=caveat,
    )


@app.post("/forecast/scenario", response_model=ScenarioForecastResponse)
def forecast_scenario(request: ScenarioForecastRequest) -> ScenarioForecastResponse:
    """CPI forecast after a user-chosen macro shock (SVAR impulse responses added to ensemble draws)."""
    return _scenario_response(
        request.target,
        request.shock_variable,
        request.shock_value,
        tuple(request.horizons),
        request.n_sims,
        request.interval_lower,
        request.interval_upper,
    )
