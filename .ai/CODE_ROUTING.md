# Code Routing

Use this file to avoid reading the whole project. Start with the listed files,
then expand only if imports, callers, tests, or failures show the contract is
wider.

## Task Routes

- API or endpoint task: start with `api/main.py`. Check `requirements.txt` only
  for dependency issues. Add API tests if/when they exist.
- Streamlit/dashboard task: start with `app/streamlit_app.py`. Read
  `api/main.py` only if endpoint shapes or forecast contracts are involved.
- ETL orchestration task: start with `src/build_curated_dataset.py`, then read
  the specific helper module it calls.
- Raw/processed transformation task: start with `src/transform.py` and
  `tests/test_transform.py`.
- Feature engineering task: start with `src/features.py` and
  `tests/test_features.py`.
- Validation or data-quality task: start with `src/validation.py`,
  `src/platform_validation.py`, and `tests/test_validation.py`.
- DuckDB, BigQuery, or Postgres load task: start with `src/platform_loads.py`;
  add `sql/queries/` or `sql/schema_app_metadata.sql` only when query/schema
  behavior matters.
- Platform status task: start with `src/platform_status.py` and use
  `reports/platform_implementation_status.csv` only for generated output
  examples.
- Data retrieval task: start with `data_retrieval.py`. Avoid rerunning it unless
  fresh network data is explicitly needed.
- Modelling/SARIMA notebook task: start with `notebooks/cpi_forecast_V1.ipynb`
  only if notebook-specific results or cells are required; otherwise work in
  source modules.
- EDA notebook task: start with `notebooks/EDA.ipynb` only for notebook content;
  otherwise use `data/curated/quarterly_macro_features.*` schema/columns.
- Documentation/story task: start with `.ai/PROJECT_BRIEF.md` and
  `.ai/STATUS_RULES.md`, then targeted README sections. Open
  `PROJECT_ARCHITECTURE.md` only for rationale or roadmap detail.
- CI or dependency task: start with `.github/workflows/`, `requirements.txt`,
  `requirements-data.txt`, and the smallest relevant source/test file.
- Docker/deployment task: start with `Dockerfile`, `api/main.py`, and
  `requirements.txt`.

## Module Contracts

- `data_retrieval.py` owns external data download and
  `dataset/download_manifest.json`. Do not mix ETL feature logic into it.
- `src/build_curated_dataset.py` owns pipeline orchestration and file outputs.
  Keep transformation, feature, validation, and load details in helper modules.
- `src/transform.py` owns source-specific cleanup and frequency alignment into
  processed series.
- `src/features.py` owns quarterly modelling features, lag/change calculations,
  and leakage-aware feature construction.
- `src/validation.py` owns reusable custom checks.
- `src/platform_validation.py` owns curated-table validation and data-quality
  reporting.
- `src/platform_loads.py` owns platform writes/loads. DuckDB should work
  locally; BigQuery/Postgres paths are optional and credential-dependent.
- `src/platform_status.py` owns human-readable implementation status reporting.
- `api/main.py` owns web API schemas and endpoint behavior. Keep heavy
  modelling or ETL work out of request handlers.
- `app/streamlit_app.py` owns UI layout and dashboard interaction. Prefer API
  calls or prepared outputs over fitting models in Streamlit.
- `tests/` owns regression coverage. Add or update the narrow test file that
  matches the changed module.

## Usually Skip

Avoid opening these unless the user specifically asks about them or a targeted
command shows they are relevant:

- `dataset/`: downloaded raw source data.
- `data/processed/`: generated processed CSVs.
- `data/curated/*.csv` and `data/curated/*.parquet`: inspect with column/schema
  commands instead of reading full files.
- `data/analytics/*.duckdb`: binary/generated analytics database.
- `reports/*.csv`: generated reports; use `head` or targeted rows if needed.
- `notebooks/*.ipynb`: high-token JSON; inspect only when notebook content
  matters.
- `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`, `.readabs_cache/`.
- `.git/` internals.

## Data Inspection Shortcuts

Prefer cheap metadata checks over loading full data files:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/curated/quarterly_macro_features.csv", nrows=5)
print(df.shape)
print(df.dtypes)
print(df.head())
PY
```

For CSV files, prefer `head`, `wc -l`, and targeted `rg`. For Parquet schema,
use Python/pandas or DuckDB only when needed.

## Test Selection

- Transform changes: `python -m pytest tests/test_transform.py`
- Feature changes: `python -m pytest tests/test_features.py`
- Validation changes: `python -m pytest tests/test_validation.py`
- Shared pipeline changes: run the narrow relevant tests first, then
  `python -m pytest tests`.
