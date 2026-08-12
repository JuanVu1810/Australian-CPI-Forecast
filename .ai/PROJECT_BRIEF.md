# Project Brief

This is the compact shared memory for Codex and Claude Code. Read this before
opening the full README or architecture document.

## Snapshot

This repository is an Australian CPI forecasting portfolio project. The
original university assignment built a univariate quarterly SARIMA forecast.
The portfolio version expands it into an end-to-end forecasting platform with
data ingestion, ETL, validation, SQL analytics, model comparison, experiment
tracking, API serving, dashboarding, tests, and deployment scaffolding.

Core question:

Can Australian CPI forecasts improve by combining historical CPI with external
macroeconomic indicators, and how do SARIMA, SARIMAX, and LSTM compare against
seasonal naive and the RBA's own published inflation forecasts?

## Models And Benchmarks

Primary models:

- SARIMA: univariate CPI history.
- SARIMAX: CPI plus selected macro predictors.
- LSTM: compact multivariate sequence challenger.

Benchmarks:

- Seasonal naive.
- RBA published inflation forecast.

## Architecture Principles

- Keep the forecasting problem central.
- Use external indicators only when economically justified.
- Avoid look-ahead bias in feature engineering, EDA, validation, and modelling.
- Separate raw data, processed data, curated modelling data, and analytics.
- Prefer local, reproducible infrastructure unless a cloud service has a clear
  job.
- Track experiments so model comparisons are reproducible.
- Make Streamlit useful for exploration and model comparison, not just
  decorative.
- Be explicit about what is implemented, scaffolded, planned, optional, or not
  deployed.

## Platform Decisions

- DuckDB is the built SQL/analytics flagship for the small curated Parquet
  dataset.
- BigQuery is documented target architecture only unless explicitly requested.
- Supabase/PostgreSQL is scoped to run, metrics, and forecast-output metadata,
  not a duplicate copy of the curated dataset.
- MLflow + FastAPI + Docker + Render is the ML engineering flagship path.
- Streamlit is the exploratory dashboard and should call FastAPI for forecasts
  instead of fitting models in the UI.
- Kubernetes, Terraform, Spark, Kafka, and Airflow are intentionally excluded
  unless the scope changes enough to justify them.

## Repo Map

- `data_retrieval.py`: downloads ABS, RBA, and market data into `dataset/`.
- `src/build_curated_dataset.py`: ETL entry point.
- `src/transform.py`: raw-to-processed transformations.
- `src/features.py`: quarterly feature engineering.
- `src/validation.py`: reusable custom validation helpers.
- `src/platform_validation.py`: curated/Pandera-style validation and reporting.
- `src/platform_loads.py`: DuckDB plus optional BigQuery/Postgres load hooks.
- `src/platform_status.py`: implementation status reporting.
- `api/main.py`: baseline FastAPI service.
- `app/streamlit_app.py`: initial Streamlit dashboard.
- `sql/queries/`: DuckDB analytics examples.
- `sql/schema_app_metadata.sql`: app metadata schema.
- `tests/`: pytest coverage for features, transforms, and validation.
- `notebooks/cpi_forecast_V1.ipynb`: original SARIMA assignment notebook.
- `notebooks/EDA.ipynb`: exploratory notebook.
- `data/curated/quarterly_macro_features.*`: curated modelling table.
- `reports/`: generated quality/status reports.

## Commands

Use focused commands first:

```bash
python -m pytest tests
python -m src.build_curated_dataset
python -m src.platform_status
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

Data retrieval can require network access and should not be rerun unless fresh
source data is explicitly needed:

```bash
python data_retrieval.py 1995 2025 --output-dir dataset
```

Optional cloud hooks are stretch paths and may require credentials:

```bash
python -m src.build_curated_dataset --load-bigquery
python -m src.build_curated_dataset --load-postgres
```

## Full Context Escalation

- Open `README.md` for user-facing story, setup commands, project status, and
  portfolio wording.
- Open `PROJECT_ARCHITECTURE.md` for architecture rationale, roadmap, trade-off
  explanations, and recruiter-facing narrative.
- Open notebooks only when the task is specifically about notebook cells,
  figures, or original modelling results.
