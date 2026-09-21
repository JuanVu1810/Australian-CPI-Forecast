"""Report implementation status for target portfolio platforms."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd


REPORT_PATH = Path("reports/platform_implementation_status.csv")


def package_available(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except ModuleNotFoundError:
        return False


def exists(path: str) -> bool:
    return Path(path).exists()


def main() -> int:
    rows = [
        {
            "platform": "ETL pipeline",
            "status": "implemented",
            "evidence": "src/build_curated_dataset.py",
            "next_step": "Use output in EDA and modelling.",
        },
        {
            "platform": "Custom validation",
            "status": "implemented",
            "evidence": "src/validation.py; reports/data_quality_report.csv",
            "next_step": "Keep core custom checks as the fallback validation layer.",
        },
        {
            "platform": "Pandera validation",
            "status": "implemented" if package_available("pandera") else "dependency_missing",
            "evidence": "src/platform_validation.py",
            "next_step": (
                "Pandera schema checks now run during ETL."
                if package_available("pandera")
                else "Install requirements so Pandera schema checks run during ETL."
            ),
        },
        {
            "platform": "pytest",
            "status": "implemented" if exists("tests") else "missing",
            "evidence": "tests/",
            "next_step": "Install requirements and run python -m pytest tests.",
        },
        {
            "platform": "DuckDB/SQL analytics",
            "status": (
                "implemented"
                if package_available("duckdb")
                and exists("data/analytics/cpi_forecast.duckdb")
                else "dependency_missing"
                if not package_available("duckdb")
                else "load_pending"
            ),
            "evidence": "src/platform_loads.py; sql/queries/",
            "next_step": (
                "DuckDB database is created by the ETL run."
                if package_available("duckdb")
                and exists("data/analytics/cpi_forecast.duckdb")
                else "Run python -m src.build_curated_dataset after installing DuckDB."
            ),
        },
        {
            "platform": "Supabase PostgreSQL",
            "status": "configuration_required",
            "evidence": (
                "src/platform_loads.py; src/models/forecast_snapshot.py; "
                ".env.example DATABASE_URL; sql/schema_app_metadata.sql"
            ),
            "next_step": (
                "Run schema SQL, then python -m src.build_curated_dataset --load-postgres "
                "and python -m src.models.forecast_snapshot snapshot."
            ),
        },
        {
            "platform": "MLflow",
            "status": (
                "implemented"
                if exists("src/models/tracking.py")
                and exists("src/models/registry.py")
                else "dependency_declared"
                if package_available("mlflow")
                else "dependency_missing"
            ),
            "evidence": "src/models/tracking.py; src/models/registry.py; mlruns/ (local)",
            "next_step": "Train the models first so mlruns/ exists; FastAPI then serves the latest runs to the Streamlit live forecast tool.",
        },
        {
            "platform": "FastAPI",
            "status": "implemented" if exists("api/main.py") else "missing",
            "evidence": "api/main.py",
            "next_step": "Run uvicorn api.main:app --reload after installing requirements.",
        },
        {
            "platform": "Streamlit",
            "status": "deployed" if exists("app/streamlit_app.py") else "missing",
            "evidence": "app/streamlit_app.py; app/requirements.txt",
            "next_step": (
                "Hosted at https://cpi-forecast-demo.streamlit.app/ (verified 2026-09-21); "
                "run locally with streamlit run app/streamlit_app.py."
            ),
        },
        {
            "platform": "Docker",
            "status": "scaffolded" if exists("Dockerfile") else "missing",
            "evidence": "Dockerfile",
            "next_step": "Build image after API dependencies are installed.",
        },
        {
            "platform": "Google Cloud Run",
            "status": "deployed",
            "evidence": "Dockerfile; api/main.py; live revision cpi-forecast-api-00016-vig (verified 2026-09-21)",
            "next_step": (
                "Redeploy is manual after serving changes: docker build, push to "
                "Artifact Registry, then gcloud run deploy cpi-forecast-api."
            ),
        },
        {
            "platform": "GitHub Actions CI",
            "status": "implemented" if exists(".github/workflows/tests.yml") else "missing",
            "evidence": ".github/workflows/tests.yml",
            "next_step": "Push to GitHub and confirm workflow passes.",
        },
        {
            "platform": "GitHub Actions scheduled ETL",
            "status": "implemented" if exists(".github/workflows/scheduled_etl.yml") else "missing",
            "evidence": ".github/workflows/scheduled_etl.yml",
            "next_step": "Push to GitHub and enable scheduled workflow.",
        },
    ]

    if os.getenv("DATABASE_URL"):
        for row in rows:
            if row["platform"] == "Supabase PostgreSQL":
                row["status"] = "configured"

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(REPORT_PATH, index=False)
    print(report.to_string(index=False))
    print(f"\nSaved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
