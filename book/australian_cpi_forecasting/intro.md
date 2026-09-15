# Australian CPI Forecasting

**Core question.** Can external macroeconomic indicators — unemployment, the cash
rate, producer prices, commodity/oil prices, and business inflation expectations —
improve Australian CPI forecasts relative to a seasonal-naive baseline and the RBA's
own published forecasts, for both headline (`cpi_yoy`) and trimmed-mean
(`trimmed_mean_cpi_yoy`) inflation?

This book follows the CRISP-DM structure used throughout the project: Business
Understanding, Data Understanding, Data Preparation, Modeling, Evaluation,
Deployment, and Conclusion. Every equation is taken from the code that actually
runs, and every one is followed by a plain-English, step-by-step explanation —
no background in time series or classification is assumed.

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
