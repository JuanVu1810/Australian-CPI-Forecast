"""MLflow Model Registry promotion helpers for the CPI forecast champion."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Literal

import pandas as pd

from src.models import tracking


REGISTERED_MODEL_NAME = "cpi_forecast_champion"
TRIMMED_MEAN_REGISTERED_MODEL_NAME = "trimmed_mean_forecast_champion"
CHAMPION_ALIAS = "champion"
ELIGIBLE_FAMILIES = ("sarima", "elastic_net")
TRIMMED_MEAN_ELIGIBLE_FAMILIES = ("sarima", "elastic_net")
REPORT_PATHS = {
    "sarima": Path("reports/model_comparison_sarima.csv"),
    "elastic_net": Path("reports/model_comparison_elastic_net.csv"),
}
SHARED_GRID_REPORT_PATH = Path("reports/model_comparison_all.csv")
TRIMMED_MEAN_SHARED_GRID_REPORT_PATH = Path("reports/model_comparison_trimmed_mean_all.csv")
TRIMMED_MEAN_REPORT_PATHS = {
    "sarima": Path("reports/model_comparison_sarima_trimmed_mean.csv"),
    "elastic_net": Path("reports/model_comparison_elastic_net_trimmed_mean.csv"),
}
TRIMMED_MEAN_MODEL_FAMILY_TAGS = {
    "sarima": "trimmed_mean_sarima",
    "elastic_net": "trimmed_mean_elastic_net",
}
MetricSource = Literal["mlflow", "report", "shared_grid_report"]


@dataclass(frozen=True)
class Candidate:
    family: Literal["sarima", "elastic_net"]
    rmse_overall: float
    metric_source: MetricSource
    run_id: str | None


@dataclass(frozen=True)
class PromotionResult:
    registered_model_name: str
    alias: str
    family: Literal["sarima", "elastic_net"]
    version: str
    run_id: str
    rmse_overall: float
    metric_source: MetricSource


def _client_and_experiment():
    mlflow = tracking.configure_mlflow()
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or tracking.DEFAULT_EXPERIMENT_NAME
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {experiment_name!r} is not configured.")
    return mlflow, client, experiment


def _latest_run_candidate(
    client,
    experiment_id: str,
    family: str,
    model_family_tag: str | None = None,
) -> Candidate | None:
    model_family_tag = family if model_family_tag is None else model_family_tag
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.model_family = '{model_family_tag}'",
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        if "rmse_overall" not in run.data.metrics:
            continue
        return Candidate(
            family=family,  # type: ignore[arg-type]
            rmse_overall=float(run.data.metrics["rmse_overall"]),
            metric_source="mlflow",
            run_id=run.info.run_id,
        )
    return None


def _report_candidate(
    family: str,
    report_paths: dict[str, Path] | None = None,
) -> Candidate | None:
    report_paths = REPORT_PATHS if report_paths is None else report_paths
    path = report_paths[family]
    if not path.exists():
        return None
    table = pd.read_csv(path)
    rows = table.loc[table["model"].eq(family) & table["horizon"].astype(str).eq("overall")]
    if rows.empty:
        return None
    return Candidate(
        family=family,  # type: ignore[arg-type]
        rmse_overall=float(rows.iloc[0]["rmse"]),
        metric_source="report",
        run_id=None,
    )


def _shared_grid_rmse(family: str, path: Path = SHARED_GRID_REPORT_PATH) -> float:
    if not path.exists():
        raise RuntimeError(
            f"Shared-grid comparison report is missing at {path}; cannot rank champion "
            "candidates without comparing them on identical forecast origins."
        )

    table = pd.read_csv(path)
    required_columns = {"model", "horizon", "rmse"}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(
            f"Shared-grid comparison report {path} is missing required columns: {missing}."
        )

    rows = table.loc[table["model"].eq(family) & table["horizon"].astype(str).eq("overall")]
    if rows.empty:
        raise RuntimeError(
            f"Shared-grid RMSE for family {family!r} was not found in {path}; cannot "
            "fall back to mismatched-grid metrics for champion promotion."
        )

    rmse = float(rows.iloc[0]["rmse"])
    if math.isnan(rmse):
        raise RuntimeError(f"Shared-grid RMSE for family {family!r} in {path} is NaN.")
    return rmse


def _rank_candidate_on_shared_grid(
    candidate: Candidate,
    path: Path = SHARED_GRID_REPORT_PATH,
) -> Candidate:
    return Candidate(
        family=candidate.family,
        rmse_overall=_shared_grid_rmse(candidate.family, path),
        metric_source="shared_grid_report",
        run_id=candidate.run_id,
    )


def _rank_candidates_on_shared_grid(
    candidates: list[Candidate],
    path: Path = SHARED_GRID_REPORT_PATH,
) -> Candidate:
    shared_grid_candidates = [
        _rank_candidate_on_shared_grid(candidate, path) for candidate in candidates
    ]
    return min(shared_grid_candidates, key=lambda candidate: candidate.rmse_overall)


def _ensure_shared_grid_report_fresh(
    client,
    candidates: list[Candidate],
    path: Path = SHARED_GRID_REPORT_PATH,
) -> None:
    if not path.exists():
        raise RuntimeError(
            f"Shared-grid comparison report is missing at {path}; cannot check whether "
            "champion metrics match the candidate model runs."
        )

    report_mtime_ms = path.stat().st_mtime * 1000
    for candidate in candidates:
        if candidate.family == "elastic_net":
            continue
        if candidate.run_id is None:
            continue
        run = client.get_run(candidate.run_id)
        if run.info.start_time is not None and report_mtime_ms < run.info.start_time:
            raise RuntimeError(
                f"Shared-grid comparison report {path} predates the latest "
                f"{candidate.family!r} MLflow candidate run {candidate.run_id}; rerun "
                "python -m src.models.model_comparison before promoting a champion."
            )


def _candidate_for_family(
    client,
    experiment_id: str,
    family: str,
    report_paths: dict[str, Path] | None = None,
    model_family_tags: dict[str, str] | None = None,
) -> Candidate | None:
    model_family_tag = None if model_family_tags is None else model_family_tags.get(family)
    return _latest_run_candidate(
        client,
        experiment_id,
        family,
        model_family_tag=model_family_tag,
    ) or _report_candidate(family, report_paths=report_paths)


def _latest_model_run_id(
    client,
    experiment_id: str,
    family: str,
    model_family_tag: str | None = None,
) -> str:
    model_family_tag = family if model_family_tag is None else model_family_tag
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.model_family = '{model_family_tag}'",
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    for run in runs:
        if run.info.status == "FINISHED":
            return run.info.run_id
    raise RuntimeError(
        f"No finished MLflow run found for {family!r}; run that orchestrator before promotion."
    )


def _logged_model_uri(client, run_id: str) -> str:
    tags = client.get_run(run_id).data.tags
    if tags.get("logged_model_uri"):
        return tags["logged_model_uri"]
    if tags.get("logged_model_id"):
        return f"models:/{tags['logged_model_id']}"
    return f"runs:/{run_id}/model"


def _promote_champion_for_config(
    registered_model_name: str,
    eligible_families: tuple[str, ...],
    report_paths: dict[str, Path],
    shared_grid_report_path: Path,
    model_family_tags: dict[str, str] | None = None,
) -> PromotionResult:
    """Promote the best eligible full-horizon family to an MLflow alias."""
    mlflow, client, experiment = _client_and_experiment()
    candidates = [
        candidate
        for family in eligible_families
        if (
            candidate := _candidate_for_family(
                client,
                experiment.experiment_id,
                family,
                report_paths=report_paths,
                model_family_tags=model_family_tags,
            )
        )
        is not None
    ]
    if not candidates:
        raise RuntimeError("No SARIMA or Elastic Net RMSE candidates found in MLflow or reports.")

    _ensure_shared_grid_report_fresh(client, candidates, path=shared_grid_report_path)
    winner = _rank_candidates_on_shared_grid(candidates, path=shared_grid_report_path)
    model_family_tag = None if model_family_tags is None else model_family_tags.get(winner.family)
    run_id = winner.run_id or _latest_model_run_id(
        client,
        experiment.experiment_id,
        winner.family,
        model_family_tag=model_family_tag,
    )
    model_uri = _logged_model_uri(client, run_id)
    version = mlflow.register_model(
        model_uri,
        registered_model_name,
        tags={
            "model_family": winner.family,
            "source_run_id": run_id,
            "rmse_overall": winner.rmse_overall,
            "metric_source": winner.metric_source,
            "promotion_note": "Full-sample model artifact selected by shared-grid RMSE.",
        },
    )
    client.set_registered_model_alias(
        registered_model_name,
        CHAMPION_ALIAS,
        version.version,
    )
    return PromotionResult(
        registered_model_name=registered_model_name,
        alias=CHAMPION_ALIAS,
        family=winner.family,  # type: ignore[arg-type]
        version=str(version.version),
        run_id=run_id,
        rmse_overall=winner.rmse_overall,
        metric_source=winner.metric_source,
    )


def promote_champion() -> PromotionResult:
    """Promote the best eligible full-horizon family to MLflow ``@champion``.

    The shared-grid report comes from ``python -m src.models.model_comparison``; rerun
    that comparison before promotion whenever SARIMA or Elastic Net specs change.
    """
    return _promote_champion_for_config(
        registered_model_name=REGISTERED_MODEL_NAME,
        eligible_families=ELIGIBLE_FAMILIES,
        report_paths=REPORT_PATHS,
        shared_grid_report_path=SHARED_GRID_REPORT_PATH,
    )


def promote_trimmed_mean_champion() -> PromotionResult:
    """Promote the best trimmed-mean eligible family to its separate MLflow model."""
    return _promote_champion_for_config(
        registered_model_name=TRIMMED_MEAN_REGISTERED_MODEL_NAME,
        eligible_families=TRIMMED_MEAN_ELIGIBLE_FAMILIES,
        report_paths=TRIMMED_MEAN_REPORT_PATHS,
        shared_grid_report_path=TRIMMED_MEAN_SHARED_GRID_REPORT_PATH,
        model_family_tags=TRIMMED_MEAN_MODEL_FAMILY_TAGS,
    )


def main() -> None:
    result = promote_champion()
    print(
        "Promoted "
        f"{result.family} run {result.run_id} "
        f"(RMSE {result.rmse_overall:.6f}, {result.metric_source}) "
        f"to {result.registered_model_name}@{result.alias} version {result.version}."
    )


if __name__ == "__main__":
    main()
