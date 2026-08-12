# Status And Claim Rules

This project is meant to be portfolio-credible. Preserve clear status language
and do not inflate planned work into completed work.

## Status Labels

Use these labels consistently:

- implemented: working code exists in the repository.
- scaffolded: structure/config exists, but full behavior or external setup is
  not complete.
- planned: described as future work or roadmap.
- optional/stretch: supported only when credentials/time/scope allow.
- target architecture: documented design choice, not necessarily built.
- deployed: only use when a live deployment is verified.
- not deployed: use for cloud services that still require account setup or
  deployment configuration.

## Current Claims To Preserve

- DuckDB is implemented as the local SQL/analytics layer.
- BigQuery is documented target architecture, not deployed by default.
- Supabase/PostgreSQL is for run, metrics, and forecast-output metadata, not a
  second copy of the curated dataset.
- FastAPI exists as a baseline service in `api/main.py`; final selected-model
  serving and live Render deployment must be verified before saying deployed.
- Streamlit exists as an initial dashboard in `app/streamlit_app.py`; it should
  call FastAPI for forecasts once serving is wired.
- MLflow is the intended experiment tracking flagship; do not claim real run
  history unless the repository contains or produces it.
- Seasonal naive is a baseline, but the RBA published forecast is the stronger
  credibility benchmark.
- LSTM is a compact challenger, not an assumed winner, because quarterly data
  has few observations.

## Architecture Rationale

- The dataset is small, roughly quarterly macro data, so a managed warehouse is
  not necessary for the current repository.
- DuckDB is the practical built SQL choice because it is local,
  credential-free, and can query the curated Parquet data directly.
- BigQuery becomes useful when the project needs larger volume, concurrency, or
  shared warehouse access.
- Postgres/Supabase is valuable only with a distinct metadata role; duplicating
  analytical data there weakens the architecture story.
- MLflow + FastAPI + Docker + Render is favored because it creates a clickable,
  testable model-serving path.
- Kubernetes, Terraform, Spark, Kafka, and Airflow are intentionally excluded
  unless the problem scale changes.

## Documentation Tone

Use clear, practical, interview-defensible language. Prefer:

- "implemented locally with DuckDB"
- "documented target architecture"
- "optional credential-dependent hook"
- "planned deployment"
- "scaffolded baseline service"

Avoid:

- claiming cloud deployment without evidence
- describing planned tools as already built
- adding new platforms without a specific role
- implying more data scale or production load than the project has

## Collaboration Rules For Codex And Claude Code

- One agent edits a file at a time.
- Best loop: Codex implements and runs tests; Claude reviews diff and wording;
  Codex applies final fixes.
- Before editing after the other agent, inspect relevant diffs or current file
  contents.
- Give each agent a narrow task and explicit file scope whenever possible.

## Prompt Template

```text
Use the project instructions first.

Task:
[one specific task]

Scope:
Only inspect these files unless necessary:
- file A
- file B

Avoid:
- notebooks
- dataset/
- data/
- reports/
- PROJECT_ARCHITECTURE.md unless architecture rationale is needed

After changes:
Run the narrowest relevant test.
Summarize files changed and any remaining risk.
```
