# Model Refit Phase 1 Decisions

Date: 2026-08-25

Scope: decision record only. No production model was fitted or registered, and no
SARIMA/SARIMAX/Elastic Net/API/MLflow wiring was changed.

## Inputs Checked

- Curated target data: `data/curated/quarterly_macro_features.csv`.
- Order-selection sample: development-only series, excluding the most recent 20
  quarters. Both targets used 1995Q1-2020Q4, `n=104`.
- Stability screen: each candidate was refit on every expanding training window
  from 32 quarters through the full development sample (`73` windows), then
  forecast for 8 quarters. Candidates with non-finite forecasts or explosive
  short-window forecasts were rejected.
- EDA stationarity/seasonality reference:
  - `cpi_yoy`: `I(0)`, level ADF p=0.0299, KPSS p=0.1000.
  - `trimmed_mean_cpi_yoy`: `I(1)`, level ADF p=0.1896; first-difference ADF
    p=0.0000, KPSS p=0.1000.
  - STL seasonal strength is `0.000` for both `cpi_yoy` and
    `trimmed_mean_cpi_yoy`, so no seasonal differencing is warranted by the EDA.

## SARIMA Order Decisions

### Trimmed Mean CPI YoY

Chosen order for Phase 2 refit:

```text
order=(1, 1, 1)
seasonal_order=(0, 0, 1, 4)
trend="n"
```

Reasoning:

- Search grid included `d_values=(0, 1)` and `seasonal_d_values=(0, 1)`.
- The raw development-only AIC winner was also the chosen order:
  `(1,1,1)x(0,0,1,4)`, AIC `-35.905`, BIC `-25.606`.
- This matches the EDA classification of trimmed-mean YoY as `I(1)` and the
  zero STL seasonal strength (`D=0`).
- The best `D=1` candidate was materially worse by AIC
  (`(0,1,2)x(0,1,2,4)`, AIC `-18.937`), so seasonal differencing was rejected.
- Stability check: all 73 expanding-window refits completed with finite,
  bounded forecasts; max absolute 8-quarter forecast was `4.699`, with no
  non-converged windows.
- Nearby AIC alternatives were checked. The rank-3 alternative
  `(2,1,0)x(0,0,2,4)` produced a short-window max absolute forecast of
  `8483.225`, so it illustrates why the raw search needs a stability screen,
  but it did not affect the selected raw winner.

### Headline CPI YoY, SA Sourced

Chosen order for Phase 2 refit:

```text
order=(1, 0, 2)
seasonal_order=(1, 0, 2, 4)
trend="n"
```

Reasoning:

- Headline `cpi_yoy` remains `I(0)` in the EDA, and STL seasonal strength is
  `0.000`, so the primary selection class is `d=0, D=0`.
- An expanded diagnostic grid with `d_values=(0, 1)` and
  `seasonal_d_values=(0, 1)` had raw AIC winner
  `(2,1,2)x(0,0,2,4)`, AIC `169.979`, but it was rejected because it
  differences an EDA-stationary target and produced an explosive short-window
  forecast (`max_abs_forecast=197.898`) with 3 non-converged windows.
- Within the stationarity-consistent `d=0, D=0` class, the raw AIC winner was
  the existing default `(2,0,2)x(0,0,2,4)`, AIC `171.845`, but this is no
  longer stable on the SA-sourced series. It produced an early-window
  `max_abs_forecast=47686.634` and had 7 non-converged expanding-window fits.
- `(2,0,2)x(1,0,2,4)` was bounded, but the full-development AIC fit did not
  converge and 10 expanding-window fits did not converge.
- The selected replacement, `(1,0,2)x(1,0,2,4)`, is the best AIC-ranked
  stationarity-consistent candidate that passed the bounded-forecast screen and
  converged on the full development sample. Stability check: 0 failures, 0
  non-finite forecasts, 0 explosive windows, max absolute 8-quarter forecast
  `20.687`, and 1 non-converged expanding-window fit.

This means the headline order should change for the post-ETL refit; do not
carry forward `(2,0,2)x(0,0,2,4)` unchanged.

## Trimmed-Mean Exogenous Feature Decision

Primary Phase 2/3 candidate list:

```text
commodity_growth_lag1
wti_growth_lag1
```

Alternative oil proxy:

```text
brent_growth_lag1
```

Use `brent_growth_lag1` as a shorter-sample alternative to
`wti_growth_lag1`, not as a default companion oil regressor in the primary
specification.

Nested/diagnostic candidates, stationarity-safe but weak in Section 10b:

```text
inflation_expectations_business_lag1
ppi_growth_lag1
unemployment_rate_change_lag1
cash_rate_change_lag1
```

Stationarity exclusions or engineering flags:

- `wpi_growth_lag1`: exclude for now. The EDA stationarity table classifies
  base `wpi_growth` as `I(1)` despite being a growth-rate transform
  (level ADF p=0.3861, KPSS p=0.0210). A differenced WPI-growth feature would
  need explicit engineering in `src/features.py` before use. It was also weak
  for trimmed mean (`candidate_granger_p_value=0.8517`) and high-VIF
  (`17.028`).
- `cash_rate_lag1`, `unemployment_rate_lag4`, and other rate-level lags are
  not in the primary trimmed-mean feature list. Their base rate levels are
  `I(1)` and the EDA already prefers change transforms for SARIMAX screening.
- Price/index levels without engineered lag features, including
  `commodity_price_index`, `wti_price`, `producer_price_index`, and
  `aud_usd`, are excluded from this phase. Several are `I(1)` and would require
  feature engineering before use.

Stationarity checks for the retained or reviewed engineered lag candidates:

| Feature | Base integration order | Section 10b status | Phase 1 decision |
| --- | --- | --- | --- |
| `commodity_growth_lag1` | `I(0)` | strong candidate | Retain primary |
| `wti_growth_lag1` | `I(0)` | strong candidate | Retain primary |
| `brent_growth_lag1` | `I(0)` | optional/shorter-sample candidate | Retain as alternate oil proxy |
| `inflation_expectations_business_lag1` | `I(0)` | candidate | Retain only for nested/diagnostic spec |
| `ppi_growth_lag1` | `I(0)` | optional/shorter-sample candidate | Retain only for nested/diagnostic spec |
| `unemployment_rate_change_lag1` | `I(0)` | optional/shorter-sample candidate | Retain only for nested/diagnostic spec |
| `cash_rate_change_lag1` | `I(0)` | optional | Retain only for nested/diagnostic spec |
| `wpi_growth_lag1` | `I(1)` | optional | Exclude unless differenced feature is engineered |

## Non-Linear Elastic Net Term Decisions

Scope: Elastic Net feature-screening only. These are candidate engineered input
columns for a still-linear-in-parameters regularized regression. No production
feature engineering, model fitting, MLflow logging, or API wiring was changed.

Guardrail: keep the nonlinear expansion deliberately small because Elastic Net's
shortest walk-forward training window is 40 quarters. Full polynomial expansion
over the retained feature pools is out of scope.

## Headline CPI YoY Elastic Net Terms

Headline baseline matrix for this check:

```text
cpi_yoy_lag1
cpi_yoy_lag4
cash_rate_change_lag1
unemployment_rate_change_lag1
inflation_expectations_business_lag1
ppi_growth_lag2
commodity_growth_lag1
wti_growth_lag1
```

Candidate nonlinear terms tried:

| Candidate term | Rationale | Integration order | Term VIF vs baseline | Decision |
| --- | --- | --- | ---: | --- |
| `inflation_expectations_business_lag1_sq` | Nonlinear expectations pass-through or de-anchoring effect | `I(0)` | 13.400 | Drop: stationary, but too collinear with the linear expectations term; max matrix VIF rose to 18.108 |
| `ppi_growth_lag2_sq` | Nonlinear producer-price pass-through when upstream inflation is large | `I(0)` | 3.066 | Keep |
| `wti_growth_lag1_sq` | Nonlinear oil-price shock effect | `I(0)` | 1.308 | Keep |
| `cash_rate_change_lag1_x_unemployment_rate_change_lag1` | Policy-stance/slack interaction | `I(0)` | 1.843 | Keep |

Final headline nonlinear candidate list:

```text
ppi_growth_lag2_sq
wti_growth_lag1_sq
cash_rate_change_lag1_x_unemployment_rate_change_lag1
```

Final keep-only matrix check: 100 complete rows from 2001Q1-2025Q4. The largest
VIF in the final matrix was `4.311` (`ppi_growth_lag2`); kept nonlinear-term
VIFs were `4.178`, `1.862`, and `1.955`, respectively. This is acceptable for
Phase 2 testing.

No AUD/oil imported-inflation interaction was added for headline because
`aud_usd_change_lag1` is not in the retained headline Elastic Net macro block;
adding it here would expand the linear candidate pool rather than screen
nonlinear transforms of retained inputs.

## Trimmed-Mean CPI YoY Elastic Net Terms

Trimmed-mean primary baseline matrix for this check:

```text
commodity_growth_lag1
wti_growth_lag1
inflation_expectations_business_lag1
ppi_growth_lag1
unemployment_rate_change_lag1
cash_rate_change_lag1
```

Candidate nonlinear terms tried:

| Candidate term | Rationale | Integration order | Term VIF vs baseline | Decision |
| --- | --- | --- | ---: | --- |
| `commodity_growth_lag1_sq` | Nonlinear broad commodity-price pass-through | `I(0)` | 1.350 | Keep |
| `wti_growth_lag1_sq` | Nonlinear oil-price shock effect in the primary WTI oil spec | `I(0)` | 1.285 | Keep |
| `brent_growth_lag1_sq` | Nonlinear oil-price shock effect for the shorter-sample Brent alternate spec | `I(0)` | 1.412 | Keep only with the Brent alternate oil spec |
| `cash_rate_change_lag1_x_unemployment_rate_change_lag1` | Policy-stance/slack interaction for nested diagnostic specs | `I(0)` | 1.746 | Keep |

Final trimmed-mean primary nonlinear candidate list:

```text
commodity_growth_lag1_sq
wti_growth_lag1_sq
cash_rate_change_lag1_x_unemployment_rate_change_lag1
```

Final trimmed-mean Brent-alternate nonlinear candidate:

```text
brent_growth_lag1_sq
```

Final primary WTI matrix check: 100 complete rows from 2001Q1-2025Q4. The
largest VIF in the final matrix was `1.973` (`ppi_growth_lag1`); kept
nonlinear-term VIFs were `1.441`, `1.375`, and `1.877`, respectively.

Final Brent-alternate matrix check: 72 complete rows from 2008Q1-2025Q4. The
largest VIF in the alternate matrix was `2.021` (`ppi_growth_lag1`), and
`brent_growth_lag1_sq` had VIF `1.412`. This is numerically usable, but the
shorter sample should remain explicit in Phase 2 comparisons.

No commodity/oil interaction was added for trimmed mean in this phase. The
primary retained commodity/oil terms are already two closely related supply
shock channels, and the small 40-quarter minimum training window argues for
testing their squared terms before adding another cross-product.
