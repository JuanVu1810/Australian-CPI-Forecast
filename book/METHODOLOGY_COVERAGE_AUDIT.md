# Methodology Coverage Audit — Jupyter Book

Audit date: 2026-09-17. Every methodology implemented in the codebase (`src/`, `api/`,
`data_retrieval.py`, per README's architecture section and `.ai/CODE_ROUTING.md`'s
ownership contracts), cross-checked against what `book/australian_cpi_forecasting/`
currently covers. Findings only — the fix in "Suggested next steps" is not yet applied.

## Clear gaps — implemented in code, not mentioned anywhere in the book

| Methodology | Where it lives | Book coverage |
|---|---|---|
| **Data retrieval** (ABS/RBA/yfinance pull, download manifest) | `data_retrieval.py` | Never mentioned — "2. Data Understanding" just says the table is "built from ABS CPI..." with no retrieval process |
| **ETL / frequency alignment** (monthly/daily → quarterly, source cleanup, join logic) | `src/build_curated_dataset.py`, `src/transform.py` | Never mentioned — "3. Data Preparation" only covers *feature* engineering (lags, dummies), not how raw sources actually become the curated table |
| **Data validation** (custom checks + Pandera schema) | `src/validation.py`, `src/platform_validation.py` | Never mentioned |
| **DuckDB analytics load** | `src/platform_loads.py` | Never mentioned, despite being a named pillar in the project brief |
| **Tableau dashboard export** | `src/models/tableau_export.py` | Never mentioned anywhere — this is a whole separate deliverable (6 tabs, per `.ai/TABLEAU_DASHBOARD_GUIDE.md`) with zero presence in the book |
| **Drift monitoring** (the z-score>2 flagging rule used by both credit-stress and SVAR shock exports) | `src/models/drift_monitor.py` | Never mentioned |

## Partial gaps — mentioned, but the method itself isn't explained

| Methodology | Where it lives | What's missing |
|---|---|---|
| **Interval calibration** (why served intervals are narrower/wider than raw simulation) | `src/models/interval_calibration.py` | 4.3 Ensemble *shows* the raw-vs-calibrated width gap but never explains the calibration method itself |
| **Walk-forward model comparison** (common-grid restriction, RBA alignment) | `src/models/evaluation.py`, `model_comparison.py` | 5. Evaluation shows results, not the walk-forward procedure that produced them |
| **MLflow tracking** | `src/models/registry.py`, `tracking.py` | 6. Deployment only says "no champion/registry promotion" — never explains what MLflow actually tracks or why |
| **SVAR shock-location & historical decomposition** | `src/models/svar.py` (`structural_shocks()`, `historical_decomposition()`) | Project notes say this was *planned* to move into "Streamlit Methodology" (i.e., this book) instead of becoming a Tableau tab — it never actually landed here. 4.4 SVAR only covers the IRF panel |

## Correctly covered, no action needed

SARIMA, Elastic Net, Ensemble, Scenario Engine, RBA Classifier, Credit Stress, FastAPI
serving, Streamlit, Docker/Cloud Run, and the EDA appendix.

## Lower priority (flagged, not recommended for this pass)

Testing (pytest/AppTest) and CI/CD (GitHub Actions, scheduled ETL) — real but more
"engineering scaffolding" than data-science methodology; left out unless requested.

## Suggested next steps (proposed, not yet applied)

- New **"3.0 ETL & Data Pipeline"** subsection at the front of Data Preparation:
  retrieval → validation → transform → DuckDB.
- Short **"Interval calibration"** addition to 4.3 Ensemble.
- **"How runs are tracked"** addition to 6. Deployment, covering MLflow.
- New **"4.4b SVAR shock attribution"** subsection for the missing
  structural-shocks/historical-decomposition content.
- A Tableau mention in Deployment or Conclusion.
