# SVAR Phase 1a Gate Decisions

[![content: AI-generated](../book/australian_cpi_forecasting/_static/badges/ai-generated.svg)](https://juanvu1810.github.io/Australian-CPI-Forecast/intro.html#ai-acknowledgement)

Date: 2026-08-28

Scope: gate/decision pass only — no IRFs, bootstrap intervals, forecasts,
simulations, wrappers, diagnostics, or backtests.

Data source: `data/curated/quarterly_macro_features.parquet`. Both systems use
level columns only, not pre-built `_lag`/`_change` features. `commodity_growth`
is missing in 1995Q1, so complete-case estimation uses 123 quarterly
observations (1995Q2-2025Q4).

## Gate Settings

- Variables per system: 5.
- Lag-order cap rule: candidate VAR lags `p` allowed only where
  `(123 - p) / (5 * p) >= 10`, capping `p <= 2` (`p=2` gives 12.1; `p=3` gives
  8.0, excluded before BIC selection).
- Johansen: `det_order=0`, `k_ar_diff = selected lag - 1`, sequential trace
  test against 5% critical values, corroborated by per-series ADF pretests at
  5%.
- Stability margin: `stable_with_margin` requires maximum companion-eigenvalue
  modulus `<= 0.98`.

## Systems A and B

System A (headline) and System B (trimmed mean) share 4 of 5 variables
(`unemployment_rate`, `cash_rate`, `commodity_growth`,
`inflation_expectations_business`) and an identical selection process; only
the target series (`cpi_yoy` vs `trimmed_mean_cpi_yoy`) and the numbers differ.

### Lag order

Selected `p=2` for both — the DoF-capped maximum, so Phase 1b should treat it
as sample-constrained, not an unqualified optimum over a wider search.

| Lag | System A BIC | System B BIC |
| --- | ---: | ---: |
| 1 | -2.685288 | -4.473348 |
| 2 | -2.901030 | -4.993345 |

Selected-lag DoF ratio (both): `(123-2)/(5*2) = 12.1`.

### Cointegration

Johansen trace test at `p=2` (`k_ar_diff=1`) finds full rank (`rank=5`) for
both systems:

| Null rank | A trace stat | B trace stat | 5% crit |
| ---: | ---: | ---: | ---: |
| 0 | 164.305876 | 169.132139 | 69.818900 |
| 1 | 78.942821 | 89.622066 | 47.854500 |
| 2 | 41.124598 | 40.960815 | 29.796100 |
| 3 | 16.231460 | 19.518958 | 15.494300 |
| 4 | 6.084672 | 6.894409 | 3.841500 |

ADF unit-root pretest corroboration (`unemployment_rate`, `cash_rate`,
`commodity_growth`, `inflation_expectations_business` are identical series in
both systems, so one p-value each):

| Variable | ADF p-value | Reject unit root at 5%? |
| --- | ---: | --- |
| `cpi_yoy` (System A target) | 0.026540 | yes |
| `trimmed_mean_cpi_yoy` (System B target) | 0.163291 | no |
| `unemployment_rate` | 0.206528 | no |
| `cash_rate` | 0.175548 | no |
| `commodity_growth` | 0.000000 | yes |
| `inflation_expectations_business` | 0.004426 | yes |

Gate rule: `rank==0` or `rank==n_vars` maps to `levels_var_svar`;
`0<rank<n_vars` maps to `vecm` (full rank means the system is already
stationary in levels). Both systems hit `rank=5`, so the Phase 1b decision
for both is **`levels_var_svar`**.

**ADF/Johansen caveat.** ADF corroboration doesn't cleanly support the
full-rank reading: 2 of 5 variables (System A) or 3 of 5 (System B, including
its own target) fail to reject the unit-root null at 5%. Johansen rank
testing assumes a common integration order across the system, so `rank=5` is
also consistent with the trace test over-rejecting in a short
(123-observation, `det_order=0`) sample. Estimating in levels (Sims-style,
under unit-root uncertainty rather than imposing possibly wrong cointegrating
restrictions) is a **documented judgment call, not a resolved stationarity
finding** — carry it forward into any later 5-variable trade-off or
Cholesky-ordering note.

### Stability

| | System A | System B |
| --- | ---: | ---: |
| Max companion-eigenvalue modulus | 0.934367 | 0.931750 |
| Stable with 0.02 margin | yes | yes |
| Min inverse-root modulus | 1.070243 | 1.073250 |

Both fitted VARs are dynamically stable at `p=2`; this corroborates but does
not drive the lag choice — the DoF cap remains binding.

### Cholesky ordering for Phase 1b

Same recursive ordering for both systems, target series in its own slot:

1. `commodity_growth`
2. `unemployment_rate`
3. Target CPI series (`cpi_yoy` / `trimmed_mean_cpi_yoy`)
4. `inflation_expectations_business`
5. `cash_rate`

Rationale: commodity growth is the most externally driven same-quarter shock;
unemployment and CPI are slower-moving domestic state variables; business
inflation expectations can update within the quarter to commodity,
labour-market and CPI information; the cash rate is ordered last so policy
observes the macro block contemporaneously while its own shocks affect the
block with a lag.

## Fixed Escalation-Rule Parameters For Phase 1b

Fixed inputs for Phase 1b's non-overlapping-bands escalation check, reused
unchanged (no IRFs or bootstrap bands are computed here):

- Bootstrap band width: 80%, quantiles `0.1`/`0.9` (`DEFAULT_LOWER_QUANTILE`/
  `DEFAULT_UPPER_QUANTILE`).
- Reported horizons: `1-8` (`DEFAULT_HORIZONS`).
