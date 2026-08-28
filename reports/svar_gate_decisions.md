# SVAR Phase 1a Gate Decisions

Date: 2026-08-28

Scope: gate/decision pass only. No IRFs, bootstrap intervals, forecasts,
simulations, wrappers, diagnostics, or backtests are built here.

Data source: `data/curated/quarterly_macro_features.parquet`.

Both systems use level columns only, not pre-built `_lag` or `_change`
features. `commodity_growth` is missing in 1995Q1, so complete-case estimation
uses 123 quarterly observations from 1995Q2 to 2025Q4.

## Gate Settings

- Variables per system: 5.
- Lag-order cap rule: choose candidate positive VAR lags `p` only where
  `(complete observations - p) / (5 * p) >= 10`.
- Resulting DoF cap: `p <= 2`.
- Cap arithmetic: at `p=2`, `(123 - 2) / (5 * 2) = 12.1`; at `p=3`,
  `(123 - 3) / (5 * 3) = 8.0`, so `p=3` is excluded before BIC selection.
- Johansen deterministic term: `det_order=0`.
- Johansen lag mapping: `k_ar_diff = selected VAR lag order - 1`.
- Johansen rank decision: sequential trace test against 5% critical values.
- Unit-root corroboration: per-series augmented Dickey-Fuller pretests at 5%.
- Stability margin: `stable_with_margin` requires maximum companion-eigenvalue
  modulus `<= 0.98`, matching the existing `stability_margin=0.02` convention.

## System A: Headline CPI Block

Variables:

1. `cpi_yoy`
2. `unemployment_rate`
3. `cash_rate`
4. `commodity_growth`
5. `inflation_expectations_business`

### Lag Order

Selected VAR lag order: `p=2`.

DoF cap: `p=2`.

Cap-edge flag: yes. BIC selected the maximum lag allowed by the DoF rule, so
Phase 1b should treat `p=2` as constrained by sample size rather than as an
unqualified optimum over a wider lag search.

BIC values:

| Lag | BIC |
| --- | ---: |
| 1 | -2.685288 |
| 2 | -2.901030 |

Selected-lag DoF ratio: `(123 - 2) / (5 * 2) = 12.1`.

### Cointegration

Johansen trace-test rank at `p=2` (`k_ar_diff=1`): `rank=5` with `n_vars=5`.

| Null rank | Trace statistic | 5% critical value |
| ---: | ---: | ---: |
| 0 | 164.305876 | 69.818900 |
| 1 | 78.942821 | 47.854500 |
| 2 | 41.124598 | 29.796100 |
| 3 | 16.231460 | 15.494300 |
| 4 | 6.084672 | 3.841500 |

ADF unit-root pretest corroboration:

| Variable | ADF statistic | p-value | Reject unit root at 5% |
| --- | ---: | ---: | --- |
| `cpi_yoy` | -3.100179 | 0.026540 | yes |
| `unemployment_rate` | -2.199221 | 0.206528 | no |
| `cash_rate` | -2.288976 | 0.175548 | no |
| `commodity_growth` | -6.250387 | 0.000000 | yes |
| `inflation_expectations_business` | -3.678500 | 0.004426 | yes |

Phase 1b decision: use `levels_var_svar`. The corrected gate rule is explicit:
`rank == 0` maps to `levels_var_svar`, `0 < rank < n_vars` maps to `vecm`,
and `rank == n_vars` maps to `levels_var_svar` because full rank indicates the
system is already stationary in levels rather than partially cointegrated.

#### ADF/Johansen Caveat

The ADF corroboration does not cleanly support the Johansen full-rank reading.
For System A, `unemployment_rate` (`p=0.206528`) and `cash_rate` (`p=0.175548`)
fail to reject the unit-root null at 5%. Because Johansen rank testing assumes a
common order of integration across the system, this mixed univariate evidence
means the `rank=5` reading is not unambiguous; it is also consistent with the
trace test over-rejecting in a short 123-observation, `det_order=0`
specification.

The pragmatic Phase 1b judgment remains `levels_var_svar`: following the
Sims-style rationale, estimating the system in levels is a defensible and
commonly used approach for dynamic simulation and IRF work under unit-root
uncertainty because it avoids imposing possibly wrong cointegrating
restrictions. This is a documented judgment call, not a resolved stationarity
finding. Phase 1b's planned evidence-based note on the 5-variable trade-offs
and Cholesky ordering rationale must carry this caveat forward; it should not
exist only in this gate report.

### Stability

Fitted VAR stability at selected lag `p=2`: stable.

Maximum companion-eigenvalue modulus: `0.934367`.

Stable with 0.02 margin: yes (`0.934367 <= 0.98`).

Minimum inverse-root modulus from `VARResults.roots`: `1.070243`.

This corroborates that the selected finite VAR is dynamically stable, but it is
not the primary lag-order signal; the DoF cap remains binding.

### Cholesky Ordering For Phase 1b

Fixed recursive ordering:

1. `commodity_growth`
2. `unemployment_rate`
3. `cpi_yoy`
4. `inflation_expectations_business`
5. `cash_rate`

Rationale: commodity growth is treated as the most externally driven
same-quarter shock; unemployment and headline CPI are slower-moving domestic
state variables; business inflation expectations can update within the quarter
to commodity, labour-market, and CPI information; the cash rate is ordered last
so the policy reaction can contemporaneously observe the macro block while
policy shocks affect the block with a lag.

## System B: Trimmed-Mean CPI Block

Variables:

1. `trimmed_mean_cpi_yoy`
2. `unemployment_rate`
3. `cash_rate`
4. `commodity_growth`
5. `inflation_expectations_business`

### Lag Order

Selected VAR lag order: `p=2`.

DoF cap: `p=2`.

Cap-edge flag: yes. BIC selected the maximum lag allowed by the DoF rule, so
Phase 1b should treat `p=2` as constrained by sample size rather than as an
unqualified optimum over a wider lag search.

BIC values:

| Lag | BIC |
| --- | ---: |
| 1 | -4.473348 |
| 2 | -4.993345 |

Selected-lag DoF ratio: `(123 - 2) / (5 * 2) = 12.1`.

### Cointegration

Johansen trace-test rank at `p=2` (`k_ar_diff=1`): `rank=5` with `n_vars=5`.

| Null rank | Trace statistic | 5% critical value |
| ---: | ---: | ---: |
| 0 | 169.132139 | 69.818900 |
| 1 | 89.622066 | 47.854500 |
| 2 | 40.960815 | 29.796100 |
| 3 | 19.518958 | 15.494300 |
| 4 | 6.894409 | 3.841500 |

ADF unit-root pretest corroboration:

| Variable | ADF statistic | p-value | Reject unit root at 5% |
| --- | ---: | ---: | --- |
| `trimmed_mean_cpi_yoy` | -2.327376 | 0.163291 | no |
| `unemployment_rate` | -2.199221 | 0.206528 | no |
| `cash_rate` | -2.288976 | 0.175548 | no |
| `commodity_growth` | -6.250387 | 0.000000 | yes |
| `inflation_expectations_business` | -3.678500 | 0.004426 | yes |

Phase 1b decision: use `levels_var_svar`. The corrected gate rule is explicit:
`rank == 0` maps to `levels_var_svar`, `0 < rank < n_vars` maps to `vecm`,
and `rank == n_vars` maps to `levels_var_svar` because full rank indicates the
system is already stationary in levels rather than partially cointegrated.

#### ADF/Johansen Caveat

The ADF corroboration does not cleanly support the Johansen full-rank reading.
For System B, `trimmed_mean_cpi_yoy` (`p=0.163291`), `unemployment_rate`
(`p=0.206528`), and `cash_rate` (`p=0.175548`) fail to reject the unit-root
null at 5%. Because Johansen rank testing assumes a common order of integration
across the system, this mixed univariate evidence means the `rank=5` reading is
not unambiguous; it is also consistent with the trace test over-rejecting in a
short 123-observation, `det_order=0` specification.

The pragmatic Phase 1b judgment remains `levels_var_svar`: following the
Sims-style rationale, estimating the system in levels is a defensible and
commonly used approach for dynamic simulation and IRF work under unit-root
uncertainty because it avoids imposing possibly wrong cointegrating
restrictions. This is a documented judgment call, not a resolved stationarity
finding. Phase 1b's planned evidence-based note on the 5-variable trade-offs
and Cholesky ordering rationale must carry this caveat forward; it should not
exist only in this gate report.

### Stability

Fitted VAR stability at selected lag `p=2`: stable.

Maximum companion-eigenvalue modulus: `0.931750`.

Stable with 0.02 margin: yes (`0.931750 <= 0.98`).

Minimum inverse-root modulus from `VARResults.roots`: `1.073250`.

This corroborates that the selected finite VAR is dynamically stable, but it is
not the primary lag-order signal; the DoF cap remains binding.

### Cholesky Ordering For Phase 1b

Fixed recursive ordering:

1. `commodity_growth`
2. `unemployment_rate`
3. `trimmed_mean_cpi_yoy`
4. `inflation_expectations_business`
5. `cash_rate`

Rationale: commodity growth is treated as the most externally driven
same-quarter shock; unemployment and trimmed-mean CPI are slower-moving domestic
state variables; business inflation expectations can update within the quarter
to commodity, labour-market, and underlying-inflation information; the cash
rate is ordered last so the policy reaction can contemporaneously observe the
macro block while policy shocks affect the block with a lag.

## Fixed Escalation-Rule Parameters For Phase 1b

These are fixed inputs for Phase 1b's non-overlapping-bands escalation check
and should be reused unchanged:

- Bootstrap band width: 80%.
- Lower/upper quantiles: `0.1` / `0.9`, matching the existing
  `DEFAULT_LOWER_QUANTILE` / `DEFAULT_UPPER_QUANTILE` convention.
- Reported horizons: `1-8`, matching `DEFAULT_HORIZONS`.

Phase 1b should consume these parameters as given; this Phase 1a report does
not compute the IRFs or bootstrap bands.
