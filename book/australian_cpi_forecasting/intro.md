# Australian CPI Forecasting

## Why this exists

Most university coursework ends the same way: a notebook cleans some data, fits a
model, prints a forecast, gets a mark, and is never opened again. Mine was no
exception, [the original assignment](https://github.com/JuanVu1810/CPI-Forecast/blob/main/notebooks/cpi_forecast_V1.ipynb)
was a single notebook that cleaned ABS CPI data (even this data was given, not being extracted) and fit a quarterly SARIMA
model. It did exactly what it was asked to do, and then it was done.

This Jupyter book is that notebook’s afterlife. The original question is still being kept (which simply just uses the historical CPI data and builds a model to forecast the next 8 quarters), while everything downstream of it is new: reproducible data retrieval and ETL, a DuckDB analytics layer, SARIMA/Elastic Net/Ensemble competing on genuine walk-forward evidence, a structural VAR for attributing why CPI moved when it did, a scenario engine, an RBA policy-action classifier, an illustrative credit-stress test, and a FastAPI + Streamlit app to serve all of it (with a live Cloud Run deployment behind it too).

## The question, stated with appropriate gravity

Officially, and for the record:

**Can external macroeconomic features, including unemployment, cash rate,
producer prices, commodity/oil prices, and business inflation expectations
improve Australian CPI forecasts with comparison to a seasonal-naive baseline and
the RBA's own published forecasts, for both headline (`cpi_yoy`) and
trimmed-mean (`trimmed_mean_cpi_yoy`) inflation?**

It's the sentence every chapter after this one exists to answer.

## How this book is organised

This book follows the CRISP-DM structure used throughout the project:
Business Understanding, Data Understanding, Data Preparation, Modeling,
Evaluation, Deployment, and Conclusion. Every equation is taken from the
code that actually runs, and every one is followed by a plain-English,
step-by-step explanation — no background in time series or classification
is assumed (hopefully).

```{admonition} Static book vs. the live app
:class: note
This book is a static, pinned snapshot — good for reading end to end. A handful
of sections (the Ensemble's Monte-Carlo reveal, the Scenario Engine, and the RBA
Policy Classifier's live refresh) are genuinely interactive and can't be replayed
on a static page; those sections show a fixed example plus a callout box telling
you how to reach the live version instead.

- **Live, interactive app:** run `streamlit run app/streamlit_app.py` from the
  project root.
- **Live forecast/scenario API:** <https://cpi-forecast-api-887232555982.asia-southeast1.run.app/docs>
  (verified live 2026-08-29; serves `/forecast/all`, `/forecast/trimmed-mean/all`,
  and `/forecast/scenario` from baked local MLflow runs — a redeploy step is not
  wired into CI, so it may lag the repository).
- **Source code:** <https://github.com/JuanVu1810/CPI-Forecast>
```
