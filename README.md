# CPI Forecast

This repository contains a university time-series forecasting project for Australian Consumer Price Index (CPI), now being extended into an end-to-end data science portfolio project. The original notebook, `notebooks/cpi_forecast_V1.ipynb`, builds and evaluates a quarterly CPI forecasting model using historical CPI observations from 1995 Q1 to 2022 Q4.

## Project Objective

The assignment goal was to forecast Australian CPI for the next 8 quarters and evaluate how well a time-series model can capture CPI trend and seasonality. CPI is an important inflation indicator, so the project frames the forecast as useful for economic planning, policy analysis, budgeting, and business decision-making.

The portfolio extension compares **three primary forecasting approaches** against **two benchmarks**:

1. **SARIMA** - the existing univariate statistical model using CPI history
2. **SARIMAX** - a multivariate statistical model using selected macroeconomic predictors
3. **LSTM** - a TensorFlow/Keras deep-learning model using multivariate historical sequences

Benchmarks:

- **Seasonal naive** - a simple statistical reference forecast
- **RBA published inflation forecast** - the Reserve Bank of Australia's own
  projection for the same period, used to check whether the project's models
  are competitive with a real professional forecaster, not just better than a
  naive baseline

Neither benchmark is counted as one of the three primary models.

## Architecture Decisions

This project deliberately does **not** try to wire up every platform a data
role might touch. Each tool below has a specific, defensible job; anything
without a clear job is documented as a target rather than built. Full
reasoning is in `PROJECT_ARCHITECTURE.md` Section 5 -- summary:

| Decision | Reasoning |
|---|---|
| DuckDB is the built SQL/analytics layer; BigQuery is a documented target, not deployed | The curated dataset is ~125 quarterly rows read from a local Parquet file. A managed cloud warehouse adds real value at higher data volume or concurrency, neither of which applies yet. |
| Supabase PostgreSQL holds MLflow backend and application run metadata, not a second copy of the curated dataset | Two databases holding the same data demonstrates nothing new. Postgres gets a distinct job: storing experiment/run metadata that can be queried from the dashboard, while artifacts stay in MLflow artifact storage. |
| MLflow + FastAPI + Docker + Render is the flagship deployment path | This is the combination that produces something clickable once deployed -- a tracked, served model -- rather than partially-configured infrastructure. |
| Kubernetes, Terraform, Spark, Kafka, Airflow are excluded | None of them solve a problem this project actually has: one small model, a handful of requests, no elastic-scaling requirement. |
| A second benchmark (RBA's own published forecast) was added alongside seasonal naive | Beating seasonal naive is a low bar for an inflation model. Comparing against a real institutional forecaster is the bar that actually matters, and the project reports honestly if it isn't cleared. |

The project is organised around three "flagship" deliverables -- one per data
role -- so each pillar is demonstrated deeply rather than every pillar being
demonstrated shallowly:

| Pillar | Flagship portfolio deliverable | Supporting evidence |
|---|---|---|
| Data Science | SARIMA vs SARIMAX vs LSTM, walk-forward validated against seasonal naive **and** the RBA forecast, with SARIMAX coefficient and LSTM SHAP interpretability | EDA notebook, feature engineering |
| Data Engineering | Ingestion → validation → curated Parquet → DuckDB, fully local and credential-free | BigQuery documented as target, not deployed |
| ML Engineering | MLflow runs, model registry with `@champion`, FastAPI selected-model serving, Docker, and planned Render deployment | Postgres as the optional MLflow backend and run metadata store |

## What `cpi_forecast_V1.ipynb` Does

The notebook follows a complete forecasting workflow:

1. Loads `CPI_train.csv`, which contains quarterly CPI observations.
2. Cleans the data by checking data types, missing values, and outliers.
3. Converts the `Quarter` column into a quarterly time-series index.
4. Performs exploratory data analysis with time-series plots, boxplots, and seasonal decomposition.
5. Tests stationarity using ACF, PACF, and Augmented Dickey-Fuller tests.
6. Applies first-order differencing to remove trend.
7. Applies seasonal differencing with lag 4 to handle quarterly seasonality.
8. Uses `pmdarima.auto_arima` to search for a suitable SARIMA model.
9. Compares the SARIMA model against a seasonal random walk benchmark using rolling-window validation.
10. Fits the final model and evaluates out-of-sample forecast accuracy.
11. Produces forecast outputs and confidence intervals.

The selected model in the notebook is:

```text
SARIMA(0, 1, 1)(0, 1, 1)[4]
```

This model was chosen because the CPI series has a clear upward trend and a repeating quarterly seasonal pattern. The notebook shows that first differencing removes the trend, while seasonal differencing at lag 4 handles the yearly seasonal cycle.

## Main Findings

The notebook found that SARIMA slightly improved on a simple seasonal random walk benchmark during rolling validation. On the 8-quarter test period, the final SARIMA model achieved a test MSE of about `46.25` and an RMSE of about `6.8` CPI points.

Residual diagnostics in the notebook suggest that the final model residuals are reasonably well behaved: there is no strong remaining autocorrelation, the residuals are approximately normal, and no obvious trend remains in the residual series.

**Note on benchmarking:** these results are currently only compared against
the seasonal naive benchmark. The next modelling phase adds the RBA's own
published inflation forecast as a second, more demanding benchmark -- see
"Planned Three-Model Comparison" below.

## Current Portfolio Implementation

The project now has three implemented layers:

1. **Univariate forecasting baseline:** `notebooks/cpi_forecast_V1.ipynb`
2. **Reproducible data retrieval:** `data_retrieval.py`
3. **ETL and validation platform:** `src/build_curated_dataset.py`

The ETL/validation layer uses the target local platforms first:

- custom validation checks for raw, processed, and curated datasets
- Pandera schema validation for the curated modelling table
- Parquet output for modelling and analytics
- DuckDB local analytical database load (the primary SQL layer -- see Architecture Decisions)
- pytest tests for transformation, validation, and feature logic
- optional BigQuery and PostgreSQL/Supabase load hooks, treated as stretch goals

Generated ETL outputs:

```text
data/processed/
data/curated/quarterly_macro_features.csv
data/curated/quarterly_macro_features.parquet
data/analytics/cpi_forecast.duckdb
reports/data_quality_report.csv
reports/platform_implementation_status.csv
```

## Current Update: `data_retrieval.py`

The project has been updated with a separate data collection script, `data_retrieval.py`. This script is not the original modelling notebook; it is a reproducible data pipeline for downloading additional Australian macroeconomic and market indicators that could support future versions of the CPI forecast.

The script can download data from:

- ABS through `readabs`
- RBA tables through `readabs`
- Market data through `yfinance`

The current script retrieves and saves:

- CPI index
- unemployment rate
- wage price index
- producer price index
- household spending
- RBA cash rate
- AUD/USD exchange rate
- inflation expectations
- commodity price indexes
- WTI crude oil futures
- Brent crude oil futures

**Planned addition:** RBA published inflation forecast series, retrieved
through the same pipeline so the second benchmark (see below) is built
reproducibly rather than pasted in manually.

Downloaded data is saved under `dataset/`, separated into `abs/`, `rba/`, and `market/` folders. A `download_manifest.json` file is also created to record the download time, package versions, selected year range, output files, row counts, and column names.

This update makes the project easier to extend from a univariate SARIMA model into a future multivariate forecasting project, where CPI could be modelled together with labour market, interest rate, exchange rate, commodity, and oil price indicators.

## Why Add More Variables?

The first version of the project uses only historical CPI values. This is useful for capturing trend and seasonality, but it limits the model because CPI is affected by broader economic conditions. During unusual periods, such as the post-COVID inflation surge, a univariate SARIMA model may underperform because it cannot observe external shocks or policy changes.

The additional variables are included because they represent possible drivers of inflation:

- **Unemployment rate:** captures labour market tightness. Lower unemployment can increase wage pressure and demand.
- **Wage Price Index:** measures wage growth, which can affect business costs and household spending.
- **Producer Price Index:** captures upstream price pressure before it reaches consumers.
- **Household spending:** measures demand-side pressure in the economy.
- **RBA cash rate:** represents monetary policy, which can influence inflation with a delay.
- **AUD/USD exchange rate:** affects import prices and imported inflation.
- **Inflation expectations:** captures forward-looking views about future inflation.
- **Commodity prices and oil prices:** capture energy, fuel, transport, and global supply-cost pressure.

Adding these variables should help the next version of the project move beyond "CPI depends only on past CPI" toward a more realistic economic forecasting model.

## Exploratory Data Analysis

`notebooks/EDA.ipynb` implements a leakage-aware EDA pass over `data/curated/quarterly_macro_features.csv` / `.parquet`, covering the CPI target and all external indicators. This was needed because the ETL aligns the original monthly, quarterly, and daily datasets into one quarterly modelling table, and the resulting predictors needed a stationarity, lead-lag, and forecast-origin-availability screen before they could be trusted in SARIMAX.

The EDA covers the following steps. Steps marked **(implemented)** are in `notebooks/EDA.ipynb`; steps marked **(planned)** are not yet built:

1. **Validate data integrity and time alignment (implemented)**

   Check that each dataset has a valid date column, no duplicate timestamps, a monotonically increasing time index, and the expected frequency. CPI is quarterly, while unemployment, cash rate, commodity prices, exchange rates, and household spending are monthly, and oil prices are daily. These series need to be resampled to a common quarterly frequency before modelling.

2. **Audit missing values and usable history (implemented)**

   Summarise the start date, end date, row count, missing values, and frequency of each variable. Some indicators do not cover the full CPI history: for example, household spending starts later than CPI, WTI and Brent oil prices start later than 1995, and some inflation expectation series contain many missing values. This audit will help decide whether to build one long-history model or several shorter-sample models.

3. **Avoid look-ahead bias**

   Missing values and frequency conversion must be handled without using future information. Back-filling should be avoided because it can leak future values into earlier quarters. Forward-filling or interpolation should only be used when it is economically reasonable and clearly documented.

4. **Inspect the CPI target (implemented)**

   Plot the CPI index, quarterly CPI growth, and year-ended CPI growth. The EDA should check trend, seasonality, volatility, outliers, and structural breaks, especially around the Global Financial Crisis, COVID period, post-COVID inflation surge, and rapid RBA cash rate increases.

5. **Inspect each external variable (implemented)**

   Plot each predictor over time and review its scale, distribution, outliers, and economic interpretation. Index variables such as wages, producer prices, commodity prices, and CPI may need differencing or percentage-change transformations. Rate variables such as unemployment, cash rate, and inflation expectations may be useful in levels or changes.

6. **Test stationarity and choose transformations (implemented)**

   Apply Augmented Dickey-Fuller (ADF) and KPSS tests to CPI and candidate predictors. These tests should guide whether each series is modelled in levels, first differences, seasonal differences, percentage changes, or log changes.

7. **Explore lead-lag relationships (implemented)**

   Use cross-correlation analysis to test whether external variables lead CPI inflation. Candidate lags should include 1-quarter, 2-quarter, and 4-quarter lags. This is especially important for variables such as cash rate, wage growth, producer prices, exchange rates, commodity prices, and oil prices, which may affect inflation with a delay.

8. **Test predictive usefulness (implemented)**

   Use Granger causality tests to check whether lagged external variables add information beyond CPI's own past values. These tests should be treated as screening tools rather than final proof, but they can help justify which predictors should enter a SARIMAX model.

9. **Check relationship stability (planned)**

   Use rolling correlations to see whether relationships between CPI and candidate predictors are stable through time or only strong during unusual periods. Variables whose relationships reverse or disappear may be less reliable for forecasting.

10. **Check multicollinearity (implemented)**

    Build a predictor correlation matrix and calculate variance inflation factors (VIFs) for candidate features. This is needed because commodity prices, oil prices, producer prices, and exchange rates may carry overlapping information.

11. **Audit feature availability (implemented)**

    For each candidate predictor, document whether the value would actually be known at the forecast origin. Many macroeconomic indicators are published with a delay, and future values of external variables are unknown for an 8-quarter forecast unless they are separately forecast. The final SARIMAX setup should therefore distinguish between lagged historical features that are available at forecast time and future exogenous paths that would need their own assumptions or forecasts.

12. **Review historical RBA forecast accuracy (planned)**

    Before comparing model results to the RBA benchmark, summarise how accurate the RBA's own published forecasts have historically been at each horizon. This sets a realistic accuracy target and avoids over-interpreting a small RMSE difference against a moving reference point.

`notebooks/EDA.ipynb` finishes with a variable coverage table, transformation decisions, candidate lag choices, multicollinearity diagnostics, and a justified screening-stage shortlist of external predictors for SARIMAX. A note on historical RBA forecast accuracy is still planned and not yet part of the notebook.

## Planned Interactive Interface

The project uses **Streamlit** as the interactive user interface, and **FastAPI** as the model-serving backend behind it -- both are part of the plan, not an either/or choice. Streamlit is the right fit for exploration, EDA, and model comparison because this is primarily a data science and forecasting project; FastAPI is the right fit for serving the selected model as a reusable endpoint once it's chosen. Streamlit calls FastAPI rather than fitting models directly in the UI, which keeps the dashboard simple and keeps model logic in one place.

A future Streamlit dashboard could include:

1. **Overview**

   Summarise the project goal, CPI forecasting objective, model choices, and key findings.

2. **Data Explorer**

   Display the available CPI, ABS, RBA, and market datasets, including date ranges, frequencies, row counts, missing values, and column descriptions.

3. **EDA Dashboard**

   Show CPI trends, quarterly and year-ended inflation, seasonal patterns, external indicator plots, correlation heatmaps, lag-correlation results, and stationarity test summaries.

4. **Forecasting Interface**

   Allow users to select a model type, forecast horizon, training window, and candidate external variables. The interface should display forecasts with confidence intervals and make it easy to compare SARIMA, SARIMAX, and LSTM outputs against both the seasonal naive and RBA published forecast benchmarks.

5. **Model Evaluation**

   Present RMSE, MSE, MAE, benchmark comparisons against both baselines, rolling-window validation results, residual diagnostics, and interpretability outputs (SARIMAX coefficients, LSTM SHAP values).

```text
POST /forecast
GET /model-metrics
GET /available-features
```

A possible structure once FastAPI and Streamlit are both wired up:

```text
.
+-- app/
|   +-- streamlit_app.py        # Streamlit dashboard, calls the FastAPI backend
+-- api/
|   +-- main.py                 # FastAPI model-serving endpoints
+-- src/
|   +-- data_processing.py      # Data loading, cleaning, merging, resampling
|   +-- eda.py                  # EDA summaries and plotting helpers
|   +-- modelling.py            # SARIMA, SARIMAX, LSTM, and benchmark models
|   +-- forecasting.py          # Forecast generation and evaluation helpers
+-- dataset/
+-- README.md
+-- requirements.txt
```

## Advanced Forecasting Ideas From Recent Literature

Recent multivariate time-series forecasting research highlights three ideas that are relevant to a future version of this CPI project: multiscale temporal modelling, external data augmentation, and careful evaluation of how external variables improve forecasts.

Peng et al. (2025) propose MSP-EDA, a multivariate forecasting framework that combines multiscale patch representations with external data enhancement. Their model uses Fourier-based analysis to capture dominant global periodic patterns, wavelet-based analysis to capture local time-frequency variation, and attention mechanisms to learn temporal dependencies, cross-variable relationships, and the influence of external data.

The full MSP-EDA deep learning architecture is probably too complex for the current CPI dataset because the project has a relatively small number of quarterly observations. However, several ideas from the framework can still strengthen this project:

1. **Add multiscale CPI analysis**

   Analyse CPI at several time scales instead of only modelling the quarterly index level. Useful views include quarter-to-quarter inflation, year-ended inflation, rolling 2-year averages, rolling 4-year averages, and seasonal quarterly patterns.

2. **Add frequency-domain diagnostics**

   Use Fourier or periodogram analysis to check dominant CPI cycles and confirm whether quarterly seasonality is strong. Wavelet analysis can be treated as an optional advanced EDA extension for detecting local changes in inflation behaviour during periods such as COVID or the post-COVID inflation surge.

3. **Create an external-data quality score**

   Before modelling, score each external variable based on coverage, missingness, frequency alignment, publication delay, and economic relevance. This makes the choice of SARIMAX predictors more transparent.

4. **Evaluate forecasts by horizon**

   Since the project forecasts 8 quarters ahead, model accuracy should be reported separately for each forecast horizon, not only as one overall RMSE. This can show whether external variables help short-term CPI forecasts, longer-term CPI forecasts, or both.

5. **Run ablation studies by variable group**

   Compare the baseline SARIMA model against several SARIMAX variants to test which groups of external variables improve forecast accuracy. Candidate groups include labour market variables, price-pressure variables, monetary and exchange-rate variables, commodity and oil variables, and all selected variables combined.

6. **Analyse cross-variable relationships**

   Translate the paper's attention-based variable-relationship idea into interpretable diagnostics suitable for this project, such as lag correlations, Granger causality tests, predictor correlation matrices, VIF scores, and model coefficient interpretation.

These additions would make the next version more research-informed while keeping the modelling approach realistic for the available data size.

## Planned Three-Model Comparison

The portfolio version uses two benchmarks -- seasonal naive and the RBA's own
published inflation forecast -- and compares three primary models that
represent increasing modelling complexity.

| Model | Type | Main inputs | Purpose |
|---|---|---|---|
| SARIMA | univariate statistical | CPI history | measure how far CPI trend, autocorrelation, and seasonality can forecast CPI |
| SARIMAX | multivariate statistical | CPI + selected macro indicators | test whether external economic information improves the forecast |
| LSTM | multivariate deep learning | sequences of CPI + selected macro indicators | test whether nonlinear temporal relationships add predictive value |

**Why two benchmarks:** seasonal naive tests whether a model beats a simple
statistical floor. The RBA published forecast tests something more useful --
whether the model is competitive with a real institution that has access to
policy intentions, business liaison data, and analyst judgment the statistical
models don't see. A model that only clears the naive bar is a weaker result
than one that also holds up against the RBA's own track record.

### SARIMA

The existing SARIMA model remains the statistical baseline. It uses only
historical CPI and therefore provides a clean reference for measuring the value
of external predictors.

### SARIMAX

SARIMAX will use economically justified lagged predictors selected through the
EDA and feature-availability audit. Candidate features include unemployment,
WPI growth, PPI growth, cash rate, commodity prices, oil-price changes, and
inflation expectations.

Where possible, the final LSTM should use the same core predictor set so that
the SARIMAX-versus-LSTM comparison reflects modelling differences rather than
different information sets.

**Interpretability:** fitted SARIMAX coefficients will be reported alongside
accuracy metrics, with a check that each coefficient's sign and approximate
magnitude match basic macroeconomic expectations (e.g. cash rate increases
should eventually be associated with lower inflation, with a lag).

### LSTM

A compact Long Short-Term Memory network will be implemented using
**TensorFlow/Keras**.

The LSTM will receive multivariate historical sequences, for example the
previous 8 quarters of selected features, and forecast future CPI. The initial
architecture should remain deliberately small, such as one LSTM layer with
approximately 8-16 hidden units, regularisation/dropout, and a dense output
layer.

The LSTM is included as a **deep-learning challenger**, not because it is
expected to outperform SARIMAX automatically.

The quarterly sample is small: a history from approximately 1995 onward
provides only around 120-125 quarterly observations before sequence creation
and train/test splitting. This creates a meaningful risk of overfitting.

To make the comparison credible:

- all models should use the same chronological evaluation periods
- the LSTM scaler must be fitted on training data only
- validation/test observations must never influence preprocessing
- the LSTM architecture should remain small
- early stopping and regularisation should be used
- results should be reported by forecast horizon where possible
- worse LSTM performance should be treated as an informative result rather
  than a failed experiment

**Interpretability:** permutation importance or SHAP values will be computed
for the trained LSTM and compared to the SARIMAX coefficients. Similar
rankings across both models strengthen confidence in which predictors
actually matter; disagreement is itself worth discussing.

A useful research framing is:

> How does forecasting performance change as the project moves from univariate
> statistical modelling (SARIMA), to multivariate statistical modelling
> (SARIMAX), to nonlinear sequence modelling (LSTM) -- and does any of that
> added complexity close the gap to the RBA's own forecast accuracy?

## Next Development Plan

The next stage of the project extends `notebooks/cpi_forecast_V1.ipynb` into a
consistent three-model forecasting comparison, then builds outward into the
two other flagship pillars (see Architecture Decisions). The ETL and
validation platform is already implemented, so the modelling work can use the
curated quarterly macroeconomic dataset directly.

**Data Science flagship (build first):**

1. **EDA on the curated modelling dataset (implemented)**

   `notebooks/EDA.ipynb` audits coverage, missingness, CPI behaviour,
   predictor relationships, lag correlations, stationarity, multicollinearity,
   and forecast-origin feature availability. A historical RBA forecast-accuracy
   review is still planned and not yet in the notebook.

2. **Choose a common long-sample predictor set**

   Start with variables that have enough history for a fair multivariate
   comparison, such as unemployment, cash rate, WPI growth, PPI growth,
   commodity growth, WTI growth, and business inflation expectations. Treat
   shorter-history variables as optional sensitivity experiments.

3. **Refactor the existing SARIMA model**

   Preserve the original SARIMA specification and evaluation logic as the
   univariate statistical baseline, then move the reusable forecasting logic
   into the modelling pipeline.

4. **Build SARIMAX**

   Fit SARIMAX using selected lagged external variables, run feature-group or
   ablation experiments, and report coefficient interpretation alongside
   accuracy metrics.

5. **Build a compact TensorFlow/Keras LSTM**

   Convert the curated dataset into chronological multivariate sequences,
   scale features using training data only, fit a deliberately small LSTM
   with regularisation and early stopping, and compute SHAP or permutation
   importance.

6. **Keep both benchmarks in every comparison**

   Seasonal naive and the RBA published forecast should both appear in every
   result table, so all three primary models are shown against a statistical
   floor and a real-world bar.

7. **Use the same walk-forward validation framework**

   SARIMA, SARIMAX, and LSTM should be evaluated over comparable forecast
   origins and horizons. Report RMSE, MAE, and other justified metrics overall
   and by horizon, against both benchmarks.

8. **Interpret the result rather than assuming the most complex model wins**

   SARIMAX may outperform LSTM because the quarterly sample is small, and none
   of the models may beat the RBA. If that occurs, the result should be
   discussed as evidence that model complexity must be matched to data
   availability, and that institutional forecasters have information
   advantages a purely statistical model can't replicate.

**ML Engineering flagship (build second):**

9. **Track all experiments in MLflow**

   Record model parameters, selected features, forecast horizons, metrics,
   and model-specific settings for every run from step 3-5 above, not as a
   later add-on. For LSTM, also record lookback length, hidden units,
   dropout, epochs, and early-stopping information.

10. **Serve the selected model through FastAPI, and deploy it**

    Wire up `/health`, `/models`, `/metrics`, and `/forecast`, containerise
    with Docker, and deploy to Render so there is a live, clickable endpoint.

**Data Engineering flagship (build third):**

11. **Finalise DuckDB analytics and automate the pipeline**

    Confirm the DuckDB SQL examples run against the finished curated dataset,
    add the Postgres run/metrics metadata store, and wire up GitHub Actions
    for CI and scheduled ETL.

**Supporting polish (build last):**

12. **Build out the Streamlit dashboard**

    Add the EDA, forecast, and model-comparison pages, calling the forecast
    API endpoint rather than fitting models in the UI.

The goal is to turn the original univariate assignment into a defensible
comparison of **SARIMA vs SARIMAX vs LSTM**, against both a statistical
baseline and a real institutional forecaster, supported by leakage-aware
preprocessing, chronological validation, and one deep, working example of
each of the three skill pillars in Architecture Decisions.

## Repository Structure

```text
.
+-- notebooks/
|   +-- cpi_forecast_V1.ipynb       # Main university forecasting notebook
|   +-- CPI_train.csv               # Original CPI training data
|   +-- CPI_forecast.csv            # Forecast output from the notebook
+-- dataset/                        # Downloaded ABS, RBA, and market datasets
+-- data/
|   +-- processed/                  # Quarterly individual series
|   +-- curated/                    # Final modelling dataset
+-- reports/                        # Data quality and platform status reports
+-- src/                            # ETL, validation, feature, and status code
+-- tests/                          # Validation/transform/feature tests
+-- api/                            # FastAPI model-serving service
+-- app/                            # Streamlit dashboard
+-- sql/                            # SQL queries and metadata schema
+-- .github/workflows/              # CI and scheduled ETL workflows
+-- data_retrieval.py
+-- PROJECT_ARCHITECTURE.md
+-- requirements-data.txt
+-- requirements.txt
+-- README.md
```

## Running the Data Retrieval Script

Install the data retrieval dependencies:

```bash
python -m pip install -r requirements-data.txt
```

Run the script for an inclusive year range:

```bash
python data_retrieval.py 1995 2025
```

Or choose a custom output folder:

```bash
python data_retrieval.py 1995 2025 --output-dir dataset
```

## Running the ETL and Validation Pipeline

After the source datasets exist under `dataset/`, build the quarterly modelling
dataset with:

```bash
python -m src.build_curated_dataset
```

The ETL pipeline:

1. validates the raw ABS, RBA, and market CSV files
2. converts monthly, quarterly, and daily series to quarterly frequency
3. creates CPI inflation, growth-rate, and lagged predictor features
4. merges all indicators into one modelling table
5. validates the curated output with custom checks
6. attempts Pandera schema validation when Pandera is installed
7. attempts a DuckDB analytical load when DuckDB is installed
8. writes a data-quality report

Outputs:

```text
data/processed/
data/curated/quarterly_macro_features.csv
data/curated/quarterly_macro_features.parquet
data/analytics/cpi_forecast.duckdb
reports/data_quality_report.csv
```

The Parquet file requires `pyarrow`. Pandera and DuckDB are also optional local
platform dependencies. If one is not installed, the pipeline records a warning
and continues with the CSV output and custom validation.

Optional, stretch-goal cloud ETL loads (documented target architecture, not
required for the MVP -- see Architecture Decisions above):

```bash
python -m src.build_curated_dataset --load-bigquery
python -m src.build_curated_dataset --load-postgres
```

`--load-postgres` writes run/metrics metadata, not a duplicate of the curated
dataset. These require the relevant environment variables in `.env.example`
to be configured first.

Run the validation and transformation tests with:

```bash
python -m pytest tests
```

## Target Platform Implementation Status

The repository includes implementation hooks for the target portfolio
platforms while keeping cloud credentials out of source control. The
"Flagship" column indicates which pieces are the three deep, headline
deliverables described in Architecture Decisions above -- these should be
fully working before anything else gets added.

| Platform | Current status | Flagship? | Evidence |
|---|---|---|---|
| ETL | implemented | | `src/build_curated_dataset.py` |
| Data validation | implemented | | `src/validation.py`, `src/platform_validation.py`, `reports/data_quality_report.csv` |
| Pandera | implemented in ETL | | `src/platform_validation.py` |
| Parquet | implemented in ETL | | `data/curated/quarterly_macro_features.parquet` |
| DuckDB/SQL analytics | implemented in ETL | **DE flagship** | `data/analytics/cpi_forecast.duckdb`, `src/platform_loads.py`, `sql/queries/` |
| BigQuery | documented target only, optional load hook, not deployed | | `src/platform_loads.py`, `.env.example`, `sql/queries/`, `PROJECT_ARCHITECTURE.md` Section 5.1 |
| Supabase PostgreSQL | schema scaffolded, scoped to MLflow backend and run metadata | supporting for MLE flagship | `sql/schema_app_metadata.sql`, `.env.example`, `PROJECT_ARCHITECTURE.md` Section 5.2 |
| SARIMA / SARIMAX / LSTM comparison + RBA benchmark | planned modelling implementation | **DS flagship** | future `src/models/`, `src/interpretability/` |
| MLflow | dependency/config scaffolded, real runs planned | **MLE flagship** | `requirements.txt`, `.env.example` |
| FastAPI | baseline service implemented, deployment planned | **MLE flagship** | `api/main.py` |
| Docker / Render | container scaffolded, live deployment planned | **MLE flagship** | `Dockerfile` |
| Streamlit | initial dashboard implemented | supporting | `app/streamlit_app.py` |
| GitHub Actions | CI and scheduled ETL scaffolded | supporting | `.github/workflows/` |

Generate a platform status report with:

```bash
python -m src.platform_status
```

Run the baseline API locally after installing `requirements.txt`:

```bash
uvicorn api.main:app --reload
```

Run the initial dashboard locally with:

```bash
streamlit run app/streamlit_app.py
```

Cloud services such as Supabase and Render still require account setup,
credentials, and deployment configuration. BigQuery is intentionally not on
that path yet (see Architecture Decisions). The repository contains the
code/configuration entry points, but it should not claim services are
deployed until public URLs and credentials are configured and verified.

## Notes

The notebook is the main submitted university project. The newer `data_retrieval.py` script is an update that improves reproducibility and prepares the repository for future model extensions using external economic indicators. The next modelling step is expected to be a new notebook or script that merges these datasets and tests whether external variables, nonlinear sequence modelling, and comparison against a real professional forecaster improve on the original SARIMA result.

## References

Peng, S., Sun, W., Chen, P., Xu, H., Ma, D., Chen, M., Wang, Y., & Li, H. (2025). MSP-EDA: Multivariate time series forecasting based on multiscale patches and external data augmentation. *Electronics, 14*(13), 2618. https://doi.org/10.3390/electronics14132618
