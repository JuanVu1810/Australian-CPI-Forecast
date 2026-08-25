"""MLflow run-lookup helpers shared by the FastAPI serving layer.

``/forecast/all`` and ``/forecast/trimmed-mean/all`` serve every model family
by loading each family's most recent finished MLflow run and its logged model
artifact directly (see ``src/models/tracking.py``) -- there is no promoted
"champion" model or MLflow Model Registry alias in this project.
"""

from __future__ import annotations

import os

from src.models import tracking


TRIMMED_MEAN_MODEL_FAMILY_TAGS = {
    "sarima": "trimmed_mean_sarima",
    "elastic_net": "trimmed_mean_elastic_net",
}


def _client_and_experiment():
    mlflow = tracking.configure_mlflow()
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or tracking.DEFAULT_EXPERIMENT_NAME
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment {experiment_name!r} is not configured.")
    return mlflow, client, experiment


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
        f"No finished MLflow run found for {family!r}; run that orchestrator before serving."
    )


def _logged_model_uri(client, run_id: str) -> str:
    tags = client.get_run(run_id).data.tags
    if tags.get("logged_model_uri"):
        return tags["logged_model_uri"]
    if tags.get("logged_model_id"):
        return f"models:/{tags['logged_model_id']}"
    return f"runs:/{run_id}/model"
