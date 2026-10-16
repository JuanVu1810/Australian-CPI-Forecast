# CPI Forecast

[![content: AI-generated](book/australian_cpi_forecasting/_static/badges/ai-generated.svg)](https://juanvu1810.github.io/Australian-CPI-Forecast/intro.html#ai-acknowledgement) [![content: human-reviewed & edited](book/australian_cpi_forecasting/_static/badges/human-reviewed-edited.svg)](https://juanvu1810.github.io/Australian-CPI-Forecast/intro.html#ai-acknowledgement)

Australian year-ended CPI inflation forecasting, taken from a university
time-series assignment to an end-to-end project: reproducible data retrieval
and ETL, SARIMA / Elastic Net / Ensemble models walk-forward validated
against seasonal naive and the RBA's own forecasts, a structural VAR with a
scenario engine, an RBA policy-action classifier, an illustrative credit-stress
test, and a FastAPI + Streamlit app with a Cloud Run deployment.

**Live links** (nothing to install):

| | |
|---|---|
| Interactive demo | https://cpi-forecast-demo.streamlit.app/ |
| Jupyter Book (results and methodology) | https://juanvu1810.github.io/Australian-CPI-Forecast/ |
| Forecast API docs | https://cpi-forecast-juan-vqvgjnkdra-as.a.run.app/docs |
| Source code | https://github.com/JuanVu1810/Australian-CPI-Forecast |

Start the book with *Key Findings*. The demo lets you run a live forecast, watch the
Ensemble's simulated paths, try your own shock in the scenario engine and see the RBA
call. This README only covers status and how to run the project.

## Status

| Component | Status |
|---|---|
| Data retrieval, ETL, validation (custom + Pandera), Parquet, DuckDB | implemented |
| SARIMA, Elastic Net, Ensemble for headline and trimmed-mean CPI, walk-forward validated | implemented |
| SVAR and scenario engine | implemented locally; illustrative, since both VAR systems fail residual whiteness and normality diagnostics |
| RBA policy-action classifier | implemented; served by the Cloud Run API since 2026-09-21 |
| Credit-risk stress test and illustrative 12-month ECL | implemented; illustrative only; served by the Cloud Run API since 2026-09-21 |
| Saved models | the fitted SARIMA and Elastic Net models (headline and trimmed mean) are `joblib` files in `models/`, committed to the repository (about 5 MB); the API loads them. No experiment tracking, no model registry |
| FastAPI | implemented, 7 endpoints; live on Cloud Run |
| Docker and Google Cloud Run | deployed 2026-10-16 as the Cloud Run service `cpi-forecast-juan` (image in Artifact Registry repo `cpi-forecast`), checked the same day through the Streamlit demo. Serves the forecast, trimmed-mean, scenario, RBA and credit endpoints; redeploy is manual |
| Streamlit | deployed on Streamlit Community Cloud, verified live 2026-09-21: <https://cpi-forecast-demo.streamlit.app/>. A four-tab interactive demo that goes with the book (live forecast, Ensemble path reveal, scenario engine, RBA call) |
| Jupyter Book | deployed to GitHub Pages by GitHub Actions |
| Reproducibility | pinned versions (`constraints.txt`), saved models, and tests on the saved reports, SQL queries and models; see [Reproduce the results](#reproduce-the-results) |
| GitHub Actions (tests, book deployment) | implemented; no Cloud Run deploy step |
| Supabase PostgreSQL | optional example, see [the appendix](#optional-supabase-postgresql-example): one table and one saved query. Not read by the API or the Streamlit app |

## Choose how to run it

| I want to | Use | Needs |
|---|---|---|
| Read the results | the book, or the live API docs | nothing |
| Run everything: ETL, tests, training, API, Streamlit, book | [Python](#run-with-python) | Python 3.11; about 45 minutes of model training |
| Check that I get the same results as the checked-in files | [Reproduce the results](#reproduce-the-results) | Python 3.11; about 10 minutes with the saved models, over an hour if you retrain |
| Serve just the API in a container | [Docker](#run-with-docker-api-only) | Docker only: the models are in the repository |

```mermaid
flowchart LR
    X["data_retrieval.py<br/>(optional, needs network)"] -.-> R["dataset/ + data/<br/>(committed)"]
    R --> E["ETL<br/>src.build_curated_dataset"]
    E --> C["data/curated/<br/>+ DuckDB"]
    C --> T["Train: 5 commands<br/>(about 45 min)"]
    T --> M["models/<br/>joblib files"]
    M --> A["FastAPI :8000"]
    A --> S["Streamlit :8501"]
    M --> D["Docker image<br/>(API only)"]
```

## Run with Python

### Prerequisites

- **Python 3.11** (what CI and the Dockerfile use). Older versions such as 3.8 cannot import `src/models`.
- **git**, and network access for `pip install` (and for the optional data download).
- **Windows:** use WSL2. The commands below are bash.

### 1. Get the code and install

```bash
git clone https://github.com/JuanVu1810/Australian-CPI-Forecast.git
cd CPI-Forecast
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt        # about 2 minutes
```

`requirements.txt` is unpinned. To get the versions the checked-in reports and book were produced
with, add the constraints file:

```bash
python -m pip install -r requirements.txt -c constraints.txt
```

[constraints.txt](constraints.txt) pins the direct dependencies, plus numpy and scipy, to the environment
that produced the results. The tests workflow and the Docker image install with it too.
Other packages still float; the book's
[Reproducibility](https://juanvu1810.github.io/Australian-CPI-Forecast/06_deployment.html#reproducibility-what-you-can-rely-on)
section says how much that matters.

### 2. Data (optional: it is already in the repository)

The repository ships the downloaded source data (`dataset/`) and the curated
modelling dataset (`data/curated/`). To rebuild the curated dataset, which also
rebuilds the DuckDB database (about a second):

```bash
python -m src.build_curated_dataset
```

It writes `data/processed/`, `data/curated/quarterly_macro_features.csv` and
`.parquet` (128 quarters, 54 columns), `data/analytics/cpi_forecast.duckdb`, and
`reports/data_quality_report.csv`.

To query the DuckDB database, run the ETL command above first. The database file is
gitignored, so it does not exist on a fresh clone. Then, from the repository root:

```bash
python - <<'EOF'
import duckdb, pathlib
con = duckdb.connect("data/analytics/cpi_forecast.duckdb", read_only=True)
print(con.sql(pathlib.Path("sql/queries/annual_inflation_summary.sql").read_text()).df().head())
EOF
```

`sql/queries/` holds three saved queries, and `tests/test_duckdb_queries.py` runs each one against the
database the ETL builds, so a renamed column fails the tests:

| Query | What it shows |
|---|---|
| `annual_inflation_summary.sql` | yearly averages of CPI, unemployment, cash rate, wage and producer price growth |
| `surge_vs_before_and_after.sql` | the same series averaged before, during and after the 2020-23 surge; it feeds a chart in book chapter 2 |
| `yearly_change_in_inflation.sql` | how far year-ended CPI and the cash rate moved over the previous four quarters (a window function) |

DuckDB is a local SQL layer over the curated table. The models, API and Streamlit app read the curated CSV
directly and do not depend on it.

To re-download the source data from ABS, RBA, APRA and Yahoo Finance (network
needed, under a minute), then rebuild:

```bash
python data_retrieval.py 1995 "$(date +%Y)" --output-dir dataset
python -m src.build_curated_dataset
```

Downloads are named `<series>_<start>_<end>.csv` after the year range you pass.
The ETL uses the newest file for each series (largest end year), so a refresh
with a later end year is picked up without editing any paths.

The models and the EDA are pinned at the 2025Q4 forecast origin
(`svar.FORECAST_ORIGIN_PIN`), so newly downloaded quarters do not change their
results. The curated table keeps the 2026 quarters only so the pinned forecasts
can be benchmarked against them. The EDA (`notebooks/EDA.ipynb`, the book's
appendix and `python -m src.eda_export`, which writes
`reports/eda_*.csv`) never sees them.

### 3. Run the tests

```bash
python -m pytest tests                            # 207 tests, about 3 minutes
```

### 4. Train the models

The API loads four saved models from `models/` (`sarima`, `elastic_net`, `trimmed_mean_sarima` and
`trimmed_mean_elastic_net`, each a `.joblib` file). They are already in the repository, trained on the data up
to the 2025Q4 forecast origin, so **you can skip this step**. To retrain, run these five commands once, in
this order:

| Command | Saves | Time* |
|---|---|---|
| `python -m src.models.evaluation` | `models/sarima.joblib` | 25 s |
| `python -m src.models.elastic_net` | `models/elastic_net.joblib` | 9 min |
| `python -m src.models.ensemble` | nothing (the Ensemble blends the two models above) | 9 min |
| `python -m src.models.model_comparison --target trimmed_mean` | `models/trimmed_mean_sarima.joblib` and `models/trimmed_mean_elastic_net.joblib` | 17 min |
| `python -m src.models.ensemble --target trimmed_mean` | nothing | 10 min |

\*Measured on one WSL2 machine; expect variation.

Training also rewrites the CSVs in `reports/`. With the pinned versions they match the committed files;
see [Reproduce the results](#reproduce-the-results) for how to check. Only the bytes
of the Parquet file from step 2 can differ, depending on your `pyarrow` version. Install with
`constraints.txt` before retraining or loading the models, because a model saved with one version of
scikit-learn or statsmodels may not load with another.

### 5. Start the API

```bash
uvicorn api.main:app --port 8000
```

Interactive docs are at <http://localhost:8000/docs>. Check it:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/forecast/all \
  -H "Content-Type: application/json" -d '{"horizon": 4}'
```

`/forecast/all` returns `models` (one entry each for `sarima`, `elastic_net` and
`ensemble`, with `forecast`, `interval_lower`, `interval_upper`, `quarters` and
`forecast_origin`) and `unavailable` (families whose model file is missing). A scenario
request shocks one SVAR variable to a level:

```bash
curl -X POST http://localhost:8000/forecast/scenario \
  -H "Content-Type: application/json" \
  -d '{"target": "headline", "shock_variable": "unemployment_rate", "shock_value": 6.0, "horizons": [1, 2, 3, 4]}'
```

`target` is `headline` or `trimmed_mean`; `shock_variable` is one of
`unemployment_rate`, `cash_rate`, `commodity_growth` or
`inflation_expectations_business`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness and curated-dataset check |
| GET | `/features` | row count, columns and quarter range of the curated dataset |
| POST | `/forecast/all` | headline forecasts from every family; body: `horizon` (1 to 8, default 8), `n_sims`, `interval_lower`, `interval_upper` |
| POST | `/forecast/trimmed-mean/all` | the same for trimmed-mean CPI |
| POST | `/forecast/scenario` | SVAR shock scenario |
| GET | `/rba-action` | RBA cut/hold/hike readings from seven classifiers |
| GET | `/credit-risk/stress-test` | illustrative credit stress and 12-month ECL |

The Cloud Run deployment serves `/health` and the forecast, trimmed-mean,
scenario, RBA and credit endpoints (verified live 2026-09-21). On its single CPU,
`/forecast/scenario` takes about 40 to 46 seconds and `/rba-action` about 30 to
45. The simulation-based outputs of those two can differ slightly from a local run;
the book's
[Reproducibility](https://juanvu1810.github.io/Australian-CPI-Forecast/06_deployment.html#reproducibility-what-you-can-rely-on)
section measures by how much.

### 6. Start Streamlit

A hosted copy is live at <https://cpi-forecast-demo.streamlit.app/>; run it locally to change it or to use
your own API.

In a second terminal, with the same virtual environment active:

```bash
streamlit run app/streamlit_app.py               # http://localhost:8501
```

The app is a four-tab demo to go with the book: Live forecast, 4.3 Ensemble,
4.5 Scenario Engine and 4.6 RBA Policy Classifier. All but the Ensemble tab call
the API, so start the API first (a scenario takes about 40 seconds). The "API base URL" box defaults to
`http://localhost:8000`; set `API_BASE_URL` to change the default.

### 7. Build the book

Use a separate environment, as CI does:

```bash
python3.11 -m venv .venv-book
source .venv-book/bin/activate
python -m pip install -r book/requirements.txt
jupyter-book build book/australian_cpi_forecasting
```

Open `book/australian_cpi_forecasting/_build/html/index.html`. The build takes
seconds: chapter outputs are pre-executed (`execute_notebooks: off`), so it
never trains a model. To re-run the analysis yourself, use `notebooks/EDA.ipynb`. The
book's appendix notebooks share one kernel session, so run them in order or not at all.

### Configuration

Everything has a working default; set these only to change it.

| Variable | Default | Effect |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | API the Streamlit app calls |
| `CPI_DEMO_HOSTED` | unset | set to `1` on a public deployment: locks the API address to `API_BASE_URL` and swaps the local "start uvicorn" messages for visitor-friendly ones |
| `PORT` | `8000` | port the Docker image listens on |

`.env.example` also lists the optional Supabase setting (`DATABASE_URL`). A normal run does not use it.

## Reproduce the results

This is the recipe for getting the same numbers as the checked-in files, and for checking that you did.
It was tested on Linux (WSL2 Ubuntu, Python 3.11).

The book's
[Reproducibility](https://juanvu1810.github.io/Australian-CPI-Forecast/06_deployment.html#reproducibility-what-you-can-rely-on)
section explains what holds the results in place and measures where a rerun can still differ. This section is
the recipe.

### 1. Set up

```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints.txt
```

Do not run `data_retrieval.py`. ABS and RBA revise past figures, so a fresh download can give slightly
different numbers.

### 2. Rebuild the data and run the tests

```bash
python -m src.build_curated_dataset      # about 1 second
python -m pytest tests                   # 207 tests, about 3 minutes
git status --short data reports          # expect nothing, or only the .parquet file (see below)
```

Pytest prints your own pass count and runtime on its final line. For this checkout,
expect `207 passed`; a different result means your checkout or environment needs checking.

The curated CSV and `reports/data_quality_report.csv` come out byte-identical. Only the bytes of the Parquet
file can differ, depending on your `pyarrow` version; its contents match.

### 3. Get the models

Pick one.

**A. Use the saved models (nothing to do).** `models/` is in the repository. These are the models behind the
checked-in forecasts.

**B. Retrain (about 45 minutes).** Run the five commands in [step 4](#4-train-the-models) above. They overwrite
`models/`. The files can differ in bytes between machines; compare the forecasts, not the files.

### 4. Check that you got the same results

**After A,** start the API (`uvicorn api.main:app --port 8000`) and request a forecast:

```bash
curl -s -X POST http://localhost:8000/forecast/all -H "Content-Type: application/json" -d '{"horizon": 4}'
```

The Ensemble's four headline forecasts should read 3.154, 3.128, 2.938 and 3.037. Every model, both targets and
all eight horizons match `reports/tableau/forecast.csv` to within 1e-15.

**After B,** regenerate the other reports and let git do the comparing:

```bash
python -m src.models.model_comparison --no-rba                   # about 7 minutes
python -m src.models.rba_classifier                              # about 9 minutes
python -m src.models.simulation_fan                              # about 1.5 minutes
python -m src.eda_export                                         # about 1 second
python -m src.models.svar_unemployment_export                    # about 1 second
python -m src.models.interval_coverage                           # about 22 minutes
python -m src.models.interval_coverage --target trimmed_mean     # about 23 minutes
python -m src.models.interval_calibration                        # about 24 minutes
python -m src.models.interval_calibration --target trimmed_mean  # about 24 minutes
git diff --stat -- reports                                       # expect no output
```

No output means every regenerated file is byte-identical to the checked-in one. If you see a difference,
check the book's
[Reproducibility](https://juanvu1810.github.io/Australian-CPI-Forecast/06_deployment.html#reproducibility-what-you-can-rely-on)
section before assuming something is wrong.

**Optional: the Tableau exports.** With the API running (`uvicorn api.main:app --port 8000` in one terminal),
`python -m src.models.tableau_export` in another rewrites `reports/tableau/` in about 20 seconds. Everything
matches except `dataset_overview.csv`, whose `curated_dataset_last_modified` column is the date the curated
data was last written, so it changes whenever the ETL runs.

### Repeat the drift measurements

The numbers behind the book's
[Reproducibility](https://juanvu1810.github.io/Australian-CPI-Forecast/06_deployment.html#reproducibility-what-you-can-rely-on)
chart come from `python -m src.reproducibility_check`, whose sub-commands compare a rerun with a baseline and save
the result to `reports/reproducibility_check.csv`. Each scenario is rerun in a scratch copy of the repo, so the
checked-in model and analysis reports are never touched. Copy-paste commands, prerequisites, safety notes, helper
scripts and saved artefacts for the recorded book/retraining/CPU/API/classifier/data-vintage audit are in
[reports/reproducibility_runs/](reports/reproducibility_runs/README.md).

## Run with Docker (API only)

Use Docker to run or test the packaged API with its own Python runtime and dependencies,
for example before deploying that image to Cloud Run. It starts the same FastAPI app as
[the local Python command](#5-start-the-api) and serves the same saved models; it does
not produce a second set of results. If you only want to check the results or run the
API locally with Python, skip Docker. Running the container locally does not deploy
anything to Cloud Run.

This image serves FastAPI only. It does not run the ETL, training or tests, and does
not include Streamlit, notebooks or the book. The models are copied in from `models/`,
which is in the repository, so a fresh clone builds as it is.

```bash
docker build -t cpi-forecast-api .
docker run --rm -d --name cpi-forecast-api -p 127.0.0.1:8000:8000 cpi-forecast-api
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/forecast/all \
  -H "Content-Type: application/json" -d '{"horizon": 4}'
docker stop cpi-forecast-api
```

The `Dockerfile` copies `reports/` whole, so a new report the API reads needs no extra step.

### Deploy to Google Cloud Run

The live service is `cpi-forecast-juan` in `asia-southeast1`. The steps below push the exact image
you tested to Artifact Registry and deploy it. Commit first, so the image tag names the code in it.

```bash
PROJECT=<project-id>; REGION=asia-southeast1; SERVICE=cpi-forecast-juan
REPO=$REGION-docker.pkg.dev/$PROJECT/cpi-forecast
TAG=$(git rev-parse --short HEAD)

docker build -t $REPO/cpi-forecast-api:$TAG .          # then run and check it locally, as above
gcloud auth login
gcloud config set project $PROJECT
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin $REGION-docker.pkg.dev
docker push $REPO/cpi-forecast-api:$TAG
```

The first deploy creates the service:

```bash
gcloud run deploy $SERVICE --image $REPO/cpi-forecast-api:$TAG --region $REGION \
  --cpu 1 --memory 2Gi --concurrency 4 --timeout 300 \
  --min-instances 0 --max-instances 3 --allow-unauthenticated
```

Later deploys can hold the new version back, test it, then switch:

```bash
gcloud run deploy $SERVICE --image $REPO/cpi-forecast-api:$TAG --region $REGION \
  --no-traffic --tag candidate                                   # test the candidate URL it prints
gcloud run services update-traffic $SERVICE --region $REGION --to-latest
gcloud run services update-traffic $SERVICE --region $REGION --to-revisions <older-revision>=100   # roll back
```

`--max-instances 3` caps how many copies Cloud Run starts, which limits the bill if someone hammers
the public endpoints. `--concurrency 4` stops several 40-second simulations sharing one CPU. Cloud Run
sets `PORT` and the image listens on it. The image installs with `constraints.txt`, so check a new
revision's forecasts against the old one before you switch traffic.

## Host the Streamlit demo

The demo can run on Streamlit Community Cloud from this GitHub repo, at a public
`*.streamlit.app` link. It only needs the four packages in `app/requirements.txt`
(the root `requirements.txt` is much heavier). It is live at
<https://cpi-forecast-demo.streamlit.app/> (verified 2026-09-21), set up with these steps.

1. Push the repository to GitHub.
2. At share.streamlit.io choose **Create app**, pick this repository, branch `main`,
   main file `app/streamlit_app.py`, and Python 3.11 under **Advanced settings**.
   Confirm in the build log that it installed `app/requirements.txt` and not the
   root one.
3. Under **Advanced settings > Secrets**, add:

   ```toml
   API_BASE_URL = "https://cpi-forecast-juan-vqvgjnkdra-as.a.run.app"
   CPI_DEMO_HOSTED = "1"
   ```

`CPI_DEMO_HOSTED = "1"` hides the API address box, so visitors cannot make the
server call an arbitrary URL. Expect the first call after an idle spell to take
about 15 seconds while Cloud Run wakes, and the scenario and RBA calls about 30 to
60 seconds. The demo remembers scenario and RBA answers for an hour (they only change
when the API is redeployed), so the same request repeated by anyone comes back
instantly, and Cloud Run is not charged for it. The 4.3 Ensemble tab reads committed
files and needs no API.

## GitHub Actions

Two workflows in `.github/workflows/` run on GitHub. Each one starts a fresh Ubuntu machine,
downloads the code and installs Python 3.11. None of them deploys the API to Cloud Run, which
stays a manual step (see [Deploy to Google Cloud Run](#deploy-to-google-cloud-run)).

| Workflow | When it runs | Purpose |
|---|---|---|
| `tests.yml` | every push and pull request | checks that the code still works |
| `deploy_book.yml` (`deploy-book`) | pushes to `main` that change the book, and on demand | publishes the Jupyter Book to GitHub Pages |

### tests.yml

1. Install the packages at the pinned versions (`requirements.txt` with `constraints.txt`).
2. Rebuild the curated dataset from the committed data. Some tests, such as the DuckDB query
   tests, need the files this creates.
3. Run the whole test suite with `pytest`.

The result is a green tick or red cross next to the commit. It changes nothing.

### deploy_book.yml

Two jobs run one after the other:

1. Build: install the book's own requirements, turn the notebooks into web pages with
   `jupyter-book build`, and package the result. It uses the saved notebook outputs and does not
   re-run any model code.
2. Deploy: publish that package to GitHub Pages. A new deployment waits for one already running
   instead of cutting it off.

The live book updates a minute or two after a book change is pushed.

To start `deploy-book` by hand, open the **Actions** tab, choose the workflow
and click **Run workflow**.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `/forecast/all` returns HTTP 200 with `unavailable` entries reading "Saved model ... not found" | A file in `models/` is missing. Run `git checkout models` or retrain with [step 4](#4-train-the-models). |
| `/forecast/all` fails to load a model with an unpickling or version error | The packages differ from the ones that saved the model. Install with `-c constraints.txt`. |
| `/forecast/all` returns HTTP 503 about "interval calibration factors" | `reports/model_interval_calibration_factors*.csv` is missing or malformed. Restore it from git. |
| `docker build` fails at `COPY constraints.txt` | The file is missing from the folder. It lives in the repository root. |
| `git diff reports/` shows a change after a re-run | `tableau/dataset_overview.csv` (a date column) is expected to differ. For anything else see where a rerun can drift in the book's [Reproducibility](https://juanvu1810.github.io/Australian-CPI-Forecast/06_deployment.html#reproducibility-what-you-can-rely-on) section. |
| Import errors, or syntax errors from `src/models` | Wrong Python version. Use 3.11. |
| Streamlit shows an API-unavailable banner in a tab | Start the API, or fix the "API base URL" box. |
| `Address already in use` | Pass a different port, for example `uvicorn api.main:app --port 8001`. |

## Repository layout

```text
.
+-- api/                 FastAPI service (api/main.py)
+-- app/                 Streamlit demo (streamlit_app.py)
+-- book/                Jupyter Book source (book/australian_cpi_forecasting)
+-- data/                processed series, curated dataset, metadata; analytics/ holds the DuckDB file built by the ETL
+-- dataset/             raw downloads from ABS, RBA, APRA and Yahoo Finance
+-- models/              the four saved models the API serves (joblib files)
+-- notebooks/           the original assignment notebook and the EDA source
+-- reports/             model comparison and interval CSVs, plus the decision notes below
+-- scripts/             helpers for the reproducibility audit
+-- sql/                 three tested DuckDB queries (queries/) and the optional Postgres example (postgres/)
+-- src/                 ETL, validation and features; src/models/ holds every model
+-- tests/               pytest suite, including checks on the saved reports, SQL queries and models
+-- .github/workflows/   tests and book deployment
+-- data_retrieval.py, Dockerfile, constraints.txt, requirements.txt, requirements-data.txt, requirements-notebooks.txt
```

`notebooks/` is not needed to run the project. To run the notebooks, install
`python -m pip install -r requirements-notebooks.txt` and start Jupyter from
`notebooks/`. `EDA.ipynb` runs end to end. `cpi_forecast_V1.ipynb` runs through
section 6; section 7 needs `CPI_test.csv`, the course's held-out test set, which
is not in the repository. The executed EDA is in the book's appendix.

## Scope and caveats

The forecasts are a methodology demonstration, not decision-ready. Both SVAR
systems fail multivariate residual whiteness and normality diagnostics, so the
scenario outputs are illustrative rather than causal. The credit-stress ECL is
a simplified, Stage-1-only, 12-month calculation. It is not comparable to any
bank's real provision and must not be used for credit, regulatory or accounting
decisions. The RBA classifier's fitted models are not statistically shown to
beat its threshold baseline. The numbers reproduce on the machine that produced them;
the book's
[Reproducibility](https://juanvu1810.github.io/Australian-CPI-Forecast/06_deployment.html#reproducibility-what-you-can-rely-on)
section lists where they might not.

"Headline CPI" in this repository means the ABS seasonally adjusted, year-ended All
groups CPI (series A3604508K), taken from the ABS analytical series on the pre-October
2025 basis. It is not the original-terms figure the ABS quotes in its releases. For
2026Q1 the repository holds 4.0% against the ABS's 4.6%, and the trimmed mean is 3.5%
against 3.3%. The models are trained and scored on the series in the repository, so
every accuracy check here, including the Supabase snapshot, uses it too.

## AI acknowledgement

A large part of this project was written with AI. It sits outside my own field (macroeconomics,
time-series econometrics and credit risk), so I used Codex and Claude Code together, as a review
loop over shared project notes (an `.ai/` folder I keep locally; it is not in this repository), to help me learn it and to write the code, formulas and
explanations. I reviewed the results to check how valid they are, but I can't guarantee they are
comparable to expert or published work, because of the time I had.

Every page of the book, and this README, opens with two tags that are read together: AI wrote
the first draft, then I went through it.

![content: AI-generated](book/australian_cpi_forecasting/_static/badges/ai-generated.svg)
The first draft was generated by AI.

![content: human-reviewed & edited](book/australian_cpi_forecasting/_static/badges/human-reviewed-edited.svg)
Then I read the outputs, re-ran the code, and edited the wording and any inconsistencies.

The markdown notes in [reports/](reports/README.md) carry the AI-generated tag only: AI drafted
them and I haven't marked them as human-reviewed. The CSV files beside them are results the
project's code computes from the data, so they aren't AI-written text and aren't tagged.

The full explanation is in the
[book's introduction](https://juanvu1810.github.io/Australian-CPI-Forecast/intro.html#ai-acknowledgement).

## Optional: Supabase PostgreSQL example

A small example of saving forecasts in a hosted PostgreSQL database and querying them with SQL. It
is not part of the forecasting story: no model, the API, Streamlit or the ETL reads this table.

- `sql/postgres/forecast_snapshots.sql` creates one table, `forecast_snapshots` (target, model, forecast
  origin, target quarter, point forecast, actual). Its key stops a rerun from saving a row twice.
- `sql/postgres/forecast_errors.sql` compares each saved forecast with the observed CPI.
- `python -m src.models.forecast_snapshot` saves the one-quarter-ahead forecast of each model for both
  targets and prints the comparison. Set `DATABASE_URL` first (use the Session pooler connection string
  from Supabase). Without it the command saves nothing and says so.

The quarter it saves (2026Q1) was already published when it was run, so this is a retrospective wiring
check, not a live forecast record. Result of the query on those six rows (the actual is the repository's
series, see the note on headline CPI above):

| target | model | quarter | forecast | actual | error |
|---|---|---|---|---|---|
| headline | elastic_net | 2026Q1 | 2.743 | 4.0 | 1.257 |
| headline | ensemble | 2026Q1 | 3.154 | 4.0 | 0.846 |
| headline | sarima | 2026Q1 | 3.384 | 4.0 | 0.616 |
| trimmed_mean | elastic_net | 2026Q1 | 2.860 | 3.5 | 0.640 |
| trimmed_mean | ensemble | 2026Q1 | 3.298 | 3.5 | 0.202 |
| trimmed_mean | sarima | 2026Q1 | 3.508 | 3.5 | -0.008 |

## Further reading

The book carries the methodology and results. The decision notes behind
specific modelling choices live in `reports/`. To check that you get the same numbers, see
[Reproduce the results](#reproduce-the-results).

- [Interval calibration remediation and the SARIMAX removal](reports/model_interval_calibration_remediation_decisions.md)
- [Trimmed-mean refit decisions](reports/model_refit_phase1_decisions.md)
- [SVAR gate decisions](reports/svar_gate_decisions.md)
- [SVAR five-variable evidence note](reports/svar_five_variable_evidence_note.md)
- [RBA classifier evaluation](reports/rba_classifier_evaluation.md) (generated by `python -m src.models.rba_classifier`)
