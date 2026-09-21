# Model Refit Phase 1 Decisions

Date: 2026-08-25

Scope: decision record only. No production model was fitted or registered, and
no SARIMA/SARIMAX/Elastic Net/API/MLflow wiring was changed.

## Inputs Checked

- Curated target data: `data/curated/quarterly_macro_features.csv`.
- Order-selection sample: development-only, excluding the most recent 20
  quarters. Both targets used 1995Q1-2020Q4, `n=104`.
- Stability screen: each candidate was refit on every expanding training
  window from 32 quarters through the full development sample (73 windows),
  then forecast 8 quarters ahead. Candidates with non-finite or explosive
  short-window forecasts were rejected.
- EDA reference: `cpi_yoy` is `I(0)` (level ADF p=0.0299, KPSS p=0.10);
  `trimmed_mean_cpi_yoy` is `I(1)` (level ADF p=0.1896; first-difference ADF
  p=0.0000, KPSS p=0.10). STL seasonal strength is `0.000` for both, so no
  seasonal differencing is warranted.

## SARIMA Order Decisions

### Trimmed Mean CPI YoY

Chosen order for Phase 2 refit:

```text
order=(1, 1, 1)
seasonal_order=(0, 0, 1, 4)
trend="n"
```

The raw development-only AIC winner (search grid `d in (0,1)`,
`seasonal_d in (0,1)`) was also the chosen order: AIC `-35.905`, BIC
`-25.606`, matching the EDA's `I(1)`, zero-seasonal-strength classification.
The best `D=1` candidate was materially worse (`(0,1,2)x(0,1,2,4)`, AIC
`-18.937`), so seasonal differencing was rejected. Stability check: all 73
expanding-window refits completed with finite, bounded forecasts (max
absolute 8-quarter forecast `4.699`). The rank-3 alternative
`(2,1,0)x(0,0,2,4)` produced a short-window max absolute forecast of
`8483.225`, illustrating why the raw AIC search needs a stability screen,
though it did not change the selected winner.

### Headline CPI YoY, SA Sourced

Chosen order for Phase 2 refit:

```text
order=(1, 0, 2)
seasonal_order=(1, 0, 2, 4)
trend="n"
```

Headline stays `I(0)` with zero seasonal strength, so `d=0, D=0` is the
primary class. An expanded grid's raw AIC winner, `(2,1,2)x(0,0,2,4)`
(AIC `169.979`), was rejected: it differences an EDA-stationary target and
produced an explosive short-window forecast (`max_abs_forecast=197.898`,
3 non-converged windows). Within `d=0, D=0`, the existing default
`(2,0,2)x(0,0,2,4)` (AIC `171.845`) is no longer stable on the SA-sourced
series (`max_abs_forecast=47686.634`, 7 non-converged fits), and
`(2,0,2)x(1,0,2,4)` failed to converge on the full development sample. The
selected replacement, `(1,0,2)x(1,0,2,4)`, is the best AIC-ranked
stationarity-consistent candidate that passed the bounded-forecast screen: 0
failures, 0 non-finite or explosive forecasts, max absolute 8-quarter
forecast `20.687`, 1 non-converged expanding-window fit.

**This means the headline order changes for the post-ETL refit; do not carry
forward `(2,0,2)x(0,0,2,4)` unchanged.**

## Trimmed-Mean Exogenous Feature Decision

Primary Phase 2/3 candidates: `commodity_growth_lag1`, `wti_growth_lag1`.
Alternative oil proxy: `brent_growth_lag1`, used as a shorter-sample
alternative to WTI, not a default companion regressor.

Nested/diagnostic candidates, stationarity-safe but weak in Section 10b:
`inflation_expectations_business_lag1`, `ppi_growth_lag1`,
`unemployment_rate_change_lag1`, `cash_rate_change_lag1`.

Excluded: `wpi_growth_lag1` (base `wpi_growth` is `I(1)` despite being a
growth-rate transform, level ADF p=0.3861, KPSS p=0.0210; would need explicit
differencing engineering; also weak, Granger p=0.8517, and high-VIF, 17.028).
Rate-level lags (`cash_rate_lag1`, `unemployment_rate_lag4`, etc.) are
excluded since their base levels are `I(1)` and the EDA prefers change
transforms. Price/index levels without engineered lags
(`commodity_price_index`, `wti_price`, `producer_price_index`, `aud_usd`) are
excluded this phase, several are `I(1)`.

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

Scope: feature-screening only, for a still-linear-in-parameters regularized
regression; no production fitting, logging or API wiring changed. Guardrail:
keep the nonlinear expansion small, Elastic Net's shortest walk-forward
training window is 40 quarters, so full polynomial expansion is out of scope.

### Headline CPI YoY

Baseline matrix: `cpi_yoy_lag1`, `cpi_yoy_lag4`, `cash_rate_change_lag1`,
`unemployment_rate_change_lag1`, `inflation_expectations_business_lag1`,
`ppi_growth_lag2`, `commodity_growth_lag1`, `wti_growth_lag1`.

| Candidate term | Rationale | Integration order | Term VIF vs baseline | Decision |
| --- | --- | --- | ---: | --- |
| `inflation_expectations_business_lag1_sq` | Nonlinear expectations pass-through or de-anchoring effect | `I(0)` | 13.400 | Drop: too collinear with the linear expectations term; max matrix VIF rose to 18.108 |
| `ppi_growth_lag2_sq` | Nonlinear producer-price pass-through when upstream inflation is large | `I(0)` | 3.066 | Keep |
| `wti_growth_lag1_sq` | Nonlinear oil-price shock effect | `I(0)` | 1.308 | Keep |
| `cash_rate_change_lag1_x_unemployment_rate_change_lag1` | Policy-stance/slack interaction | `I(0)` | 1.843 | Keep |

Final headline nonlinear list: `ppi_growth_lag2_sq`, `wti_growth_lag1_sq`,
`cash_rate_change_lag1_x_unemployment_rate_change_lag1`. Final keep-only
matrix check (100 complete rows, 2001Q1-2025Q4): largest VIF `4.311`
(`ppi_growth_lag2`); kept nonlinear-term VIFs `4.178`, `1.862`, `1.955`,
acceptable for Phase 2 testing.

No AUD/oil imported-inflation interaction was added for headline:
`aud_usd_change_lag1` is not in the retained headline macro block, so adding
it here would expand the linear candidate pool rather than screen nonlinear
transforms of retained inputs.

### Trimmed-Mean CPI YoY

Baseline matrix: `commodity_growth_lag1`, `wti_growth_lag1`,
`inflation_expectations_business_lag1`, `ppi_growth_lag1`,
`unemployment_rate_change_lag1`, `cash_rate_change_lag1`.

| Candidate term | Rationale | Integration order | Term VIF vs baseline | Decision |
| --- | --- | --- | ---: | --- |
| `commodity_growth_lag1_sq` | Nonlinear broad commodity-price pass-through | `I(0)` | 1.350 | Keep |
| `wti_growth_lag1_sq` | Nonlinear oil-price shock effect in the primary WTI spec | `I(0)` | 1.285 | Keep |
| `brent_growth_lag1_sq` | Nonlinear oil-price shock effect for the shorter-sample Brent spec | `I(0)` | 1.412 | Keep only with the Brent alternate oil spec |
| `cash_rate_change_lag1_x_unemployment_rate_change_lag1` | Policy-stance/slack interaction for nested diagnostic specs | `I(0)` | 1.746 | Keep |

Final primary nonlinear list: `commodity_growth_lag1_sq`,
`wti_growth_lag1_sq`, `cash_rate_change_lag1_x_unemployment_rate_change_lag1`.
Final Brent-alternate nonlinear candidate: `brent_growth_lag1_sq`.

Primary WTI matrix check (100 rows, 2001Q1-2025Q4): largest VIF `1.973`
(`ppi_growth_lag1`); kept nonlinear-term VIFs `1.441`, `1.375`, `1.877`.
Brent-alternate matrix check (72 rows, 2008Q1-2025Q4): largest VIF `2.021`
(`ppi_growth_lag1`), `brent_growth_lag1_sq` VIF `1.412`, numerically usable
but the shorter sample should stay explicit in Phase 2 comparisons.

No commodity/oil interaction was added for trimmed mean this phase: the
retained commodity/oil terms are already two closely related supply-shock
channels, and the small 40-quarter minimum training window argues for
testing their squared terms before adding another cross-product.
