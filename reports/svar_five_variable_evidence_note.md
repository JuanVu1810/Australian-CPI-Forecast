# SVAR Five-Variable Evidence Note

Date: 2026-08-28

Scope: Phase 1b evidence note for the two five-variable SVAR systems gated in
`reports/svar_gate_decisions.md`.

## Five-Variable Trade-Offs

Both systems use the five level variables settled in Phase 1a:

- System A: `cpi_yoy`, `unemployment_rate`, `cash_rate`,
  `commodity_growth`, `inflation_expectations_business`.
- System B: `trimmed_mean_cpi_yoy`, `unemployment_rate`, `cash_rate`,
  `commodity_growth`, `inflation_expectations_business`.

Business inflation expectations are the strongest empirically supported shared
macro variable in simple contemporaneous screening: raw correlation is `0.697`
with `cpi_yoy` and `0.595` with `trimmed_mean_cpi_yoy` on the curated
quarterly dataset.

Unemployment and the cash rate are retained primarily on economic-theory
grounds rather than raw correlation alone. Their raw correlations are weaker:
for headline CPI, `unemployment_rate = -0.385` and `cash_rate = 0.236`; for
trimmed-mean CPI, `unemployment_rate = -0.487` and `cash_rate = 0.274`.
They remain central state and policy variables for impulse-response analysis.

Commodity growth is retained over the stronger raw PPI alternative because it
supports the scenario feature and preserves sample coverage. `ppi_growth` has
stronger raw correlation with both CPI targets (`0.522` with headline CPI and
`0.384` with trimmed-mean CPI), but replacing `commodity_growth` with
`ppi_growth` would reduce complete-case system size from 123 observations to
109 observations. The selected commodity-growth systems keep 1995Q2-2025Q4
coverage.

## Cholesky Ordering Rationale

System A fixed recursive ordering:

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

System B fixed recursive ordering:

1. `commodity_growth`
2. `unemployment_rate`
3. `trimmed_mean_cpi_yoy`
4. `inflation_expectations_business`
5. `cash_rate`

Rationale: commodity growth is treated as the most externally driven
same-quarter shock; unemployment and trimmed-mean CPI are slower-moving
domestic state variables; business inflation expectations can update within
the quarter to commodity, labour-market, and underlying-inflation information;
the cash rate is ordered last so the policy reaction can contemporaneously
observe the macro block while policy shocks affect the block with a lag.

## ADF/Johansen Caveat

### System A

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

### System B

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

## Phase 1b Treatment Result

System B is the primary system and receives full treatment: recursive
Cholesky IRFs with 80% block-residual-bootstrap bands, VAR diagnostics, and
the diagnostic walk-forward point-forecast backtest.

System A was initially confirmatory, but the fixed escalation check found
non-overlapping 80% block-bootstrap IRF bands against System B at four
shared-shock horizons: `cash_rate` horizons 1-3 and
`inflation_expectations_business` horizon 2. System A therefore remains
escalated to the same full diagnostic and backtest treatment as System B. The
previous i.i.d. bootstrap run had triggered five horizons; after the block
bootstrap replacement, `inflation_expectations_business` horizon 1 no longer
triggers because the 80% bands overlap.

Backtest RMSE is diagnostic only. SVAR is not used here as a forecast-accuracy
competitor to SARIMA, Elastic Net, or the Ensemble.

## Persisted Phase 1b Diagnostics

These are the full-sample VAR(2)-in-levels diagnostics produced for the two
systems after System A escalated. The tests reject at 5% where `reject_5pct` is
`yes`.

### Multivariate Tests

| System | Test | Statistic | p-value | Reject 5% |
| --- | --- | ---: | ---: | --- |
| System A | Portmanteau whiteness | 283.464294 | 0.000094 | yes |
| System A | Multivariate normality | 3074.523062 | 0.000000 | yes |
| System B | Portmanteau whiteness | 317.096153 | 0.000000253 | yes |
| System B | Multivariate normality | 2774.021940 | 0.000000 | yes |

### Per-Equation Residual Tests

| System | Variable | Test | Statistic | p-value | Reject 5% |
| --- | --- | --- | ---: | ---: | --- |
| System A | `commodity_growth` | Jarque-Bera normality | 8.283162 | 0.015898 | yes |
| System A | `commodity_growth` | ARCH-LM heteroskedasticity | 3.076198 | 0.545156 | no |
| System A | `unemployment_rate` | Jarque-Bera normality | 2866.801056 | 0.000000 | yes |
| System A | `unemployment_rate` | ARCH-LM heteroskedasticity | 1.539473 | 0.819627 | no |
| System A | `cpi_yoy` | Jarque-Bera normality | 24.798171 | 0.000004 | yes |
| System A | `cpi_yoy` | ARCH-LM heteroskedasticity | 16.444539 | 0.002477 | yes |
| System A | `inflation_expectations_business` | Jarque-Bera normality | 20.104870 | 0.000043 | yes |
| System A | `inflation_expectations_business` | ARCH-LM heteroskedasticity | 9.025540 | 0.060464 | no |
| System A | `cash_rate` | Jarque-Bera normality | 976.642773 | 0.000000 | yes |
| System A | `cash_rate` | ARCH-LM heteroskedasticity | 0.805165 | 0.937754 | no |
| System B | `commodity_growth` | Jarque-Bera normality | 13.495174 | 0.001174 | yes |
| System B | `commodity_growth` | ARCH-LM heteroskedasticity | 1.546422 | 0.818388 | no |
| System B | `unemployment_rate` | Jarque-Bera normality | 2693.496309 | 0.000000 | yes |
| System B | `unemployment_rate` | ARCH-LM heteroskedasticity | 1.139165 | 0.888010 | no |
| System B | `trimmed_mean_cpi_yoy` | Jarque-Bera normality | 0.083072 | 0.959315 | no |
| System B | `trimmed_mean_cpi_yoy` | ARCH-LM heteroskedasticity | 14.341001 | 0.006283 | yes |
| System B | `inflation_expectations_business` | Jarque-Bera normality | 16.775923 | 0.000228 | yes |
| System B | `inflation_expectations_business` | ARCH-LM heteroskedasticity | 9.810471 | 0.043744 | yes |
| System B | `cash_rate` | Jarque-Bera normality | 1176.956953 | 0.000000 | yes |
| System B | `cash_rate` | ARCH-LM heteroskedasticity | 0.592155 | 0.963931 | no |

## Diagnostic Backtest RMSE

The SVAR walk-forward backtest uses
`DEFAULT_INITIAL_TRAIN_SIZE=32` and horizons 1-8. These RMSE values are
diagnostic only. The SVAR origin grid has not been verified against the
SARIMA, Elastic Net, or Ensemble comparison grids, so the figures below should
not be read as directly comparable to those models even informally.

| System | Target | Overall RMSE | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| System A | `cpi_yoy` | 1.791583 | 0.843101 | 1.252857 | 1.642059 | 1.966088 | 2.064319 | 2.065749 | 2.033528 | 2.042510 |
| System B | `trimmed_mean_cpi_yoy` | 1.206149 | 0.305645 | 0.577790 | 0.848872 | 1.116049 | 1.325052 | 1.483851 | 1.591184 | 1.659975 |

## COVID-Era Diagnostic Sensitivity

The project intervention metadata identifies two COVID-era intervention
quarters: `2020Q2` (`covid_shock_down`) and `2020Q3`
(`covid_shock_rebound`). Sensitivity checks were run two ways: excluding those
two rows, and fitting the same VAR(2)-in-levels with those two intervention
dummies as exogenous terms.

COVID treatment weakens some diagnostics but does not resolve the core
failures. Both systems still reject multivariate residual whiteness and
multivariate normality after either excluding or dummying the COVID quarters.
Normality improves materially under dummying because the unemployment residual
Jarque-Bera failures resolve, but other residual series continue to reject
normality. ARCH-LM failures partly improve: System B's
`inflation_expectations_business` ARCH-LM rejection resolves under both
treatments, and System B's target ARCH-LM rejection resolves when the COVID
quarters are excluded, but System A's target ARCH-LM rejection persists.

| System | COVID treatment | Whiteness statistic | Whiteness p-value | Normality statistic | Normality p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| System A | Full sample | 283.464294 | 0.000094 | 3074.523062 | 0.000000 |
| System A | Exclude `2020Q2`/`2020Q3` | 273.119255 | 0.000452 | 2283.146293 | 0.000000 |
| System A | Dummy `2020Q2`/`2020Q3` | 267.196082 | 0.001049 | 616.723640 | 0.000000 |
| System B | Full sample | 317.096153 | 0.000000253 | 2774.021940 | 0.000000 |
| System B | Exclude `2020Q2`/`2020Q3` | 299.177357 | 0.000007 | 2147.311178 | 0.000000 |
| System B | Dummy `2020Q2`/`2020Q3` | 298.007220 | 0.000008 | 578.874441 | 0.000000 |

Selected equation-level changes:

| System | Variable | Test | Full-sample p-value | Exclude-COVID p-value | Dummy-COVID p-value | Read |
| --- | --- | --- | ---: | ---: | ---: | --- |
| System A | `unemployment_rate` | Jarque-Bera | 0.000000 | 0.000000 | 0.115153 | Resolves only with dummies |
| System A | `cpi_yoy` | ARCH-LM | 0.002477 | 0.015802 | 0.031329 | Persists, weaker |
| System B | `unemployment_rate` | Jarque-Bera | 0.000000 | 0.000000 | 0.110974 | Resolves only with dummies |
| System B | `trimmed_mean_cpi_yoy` | ARCH-LM | 0.006283 | 0.227851 | 0.015966 | Resolves when excluded; persists with dummies |
| System B | `inflation_expectations_business` | ARCH-LM | 0.043744 | 0.079466 | 0.152525 | Resolves under both treatments |

## Bootstrap Validity Probe

The shipped IRF bootstrap in `src/models/svar.py` is now a contiguous
residual block bootstrap using block length `DEFAULT_ARCH_LAGS = 4`. Blocks
are drawn as non-wrapping contiguous runs from the fitted residual series,
concatenated, and then trimmed only at the final boundary to reconstruct the
required residual-draw length. The simulate-and-refit logic is otherwise the
same as the original Phase 1b bootstrap.

This replaced the earlier i.i.d. residual bootstrap because the persistent
whiteness, normality, and some ARCH-LM failures mean single-row residual draws
are too optimistic for dependence-sensitive IRF uncertainty. The historical
scratch prototype compared the old i.i.d. bootstrap with contiguous residual
block bootstraps using block lengths 4 and 8, keeping the same 80% quantiles
and horizons.

At the requested shocks and horizons, block-bootstrap bands were generally
wider for `inflation_expectations_business` shocks and mixed for `cash_rate`
shocks. Across the six checked rows per system, mean block-to-i.i.d. width
ratios were `1.137` for block length 4 and `1.149` for block length 8 in
System A, and `1.078` for block length 4 and `1.177` for block length 8 in
System B.

| System | Response | Shock | Horizon | IID width | Block-4 width | Block-4 / IID | Block-8 width | Block-8 / IID |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| System A | `cpi_yoy` | `cash_rate` | 1 | 0.140772 | 0.127305 | 0.904 | 0.126158 | 0.896 |
| System A | `cpi_yoy` | `cash_rate` | 2 | 0.176521 | 0.171665 | 0.972 | 0.180391 | 1.022 |
| System A | `cpi_yoy` | `cash_rate` | 3 | 0.200113 | 0.209950 | 1.049 | 0.240383 | 1.201 |
| System A | `cpi_yoy` | `inflation_expectations_business` | 1 | 0.149923 | 0.217089 | 1.448 | 0.193791 | 1.293 |
| System A | `cpi_yoy` | `inflation_expectations_business` | 2 | 0.185438 | 0.229735 | 1.239 | 0.231597 | 1.249 |
| System A | `cpi_yoy` | `inflation_expectations_business` | 3 | 0.211901 | 0.256034 | 1.208 | 0.261848 | 1.236 |
| System B | `trimmed_mean_cpi_yoy` | `cash_rate` | 1 | 0.046010 | 0.047878 | 1.041 | 0.045854 | 0.997 |
| System B | `trimmed_mean_cpi_yoy` | `cash_rate` | 2 | 0.082849 | 0.081774 | 0.987 | 0.088915 | 1.073 |
| System B | `trimmed_mean_cpi_yoy` | `cash_rate` | 3 | 0.108209 | 0.106784 | 0.987 | 0.126240 | 1.167 |
| System B | `trimmed_mean_cpi_yoy` | `inflation_expectations_business` | 1 | 0.047921 | 0.057729 | 1.205 | 0.063274 | 1.320 |
| System B | `trimmed_mean_cpi_yoy` | `inflation_expectations_business` | 2 | 0.075770 | 0.086086 | 1.136 | 0.093378 | 1.232 |
| System B | `trimmed_mean_cpi_yoy` | `inflation_expectations_business` | 3 | 0.102668 | 0.114165 | 1.112 | 0.130621 | 1.272 |

This is evidence that the i.i.d. residual bootstrap may understate uncertainty
for some IRFs, especially the inflation-expectations shock responses. The
evidence was not uniformly one-way for every cash-rate row, but it was strong
enough to ship the block bootstrap before Phase 2 consumes the IRF bands.

After shipping the block bootstrap with block length 4 and rerunning
`bootstrap_cholesky_irf_bands` for both systems with 1000 replications, the
same shock/horizon rows have the following widths. Differences from the
scratch block-4 prototype are expected stochastic variation from rerunning the
bootstrap; the shipped values are close to the prototype on average.

| System | Response | Shock | Horizon | Old IID width | Historical block-4 probe width | Shipped block-4 width | Shipped / IID | Shipped / probe |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| System A | `cpi_yoy` | `cash_rate` | 1 | 0.140772 | 0.127305 | 0.129771 | 0.922 | 1.019 |
| System A | `cpi_yoy` | `cash_rate` | 2 | 0.176521 | 0.171665 | 0.175809 | 0.996 | 1.024 |
| System A | `cpi_yoy` | `cash_rate` | 3 | 0.200113 | 0.209950 | 0.224765 | 1.123 | 1.071 |
| System A | `cpi_yoy` | `inflation_expectations_business` | 1 | 0.149923 | 0.217089 | 0.231212 | 1.542 | 1.065 |
| System A | `cpi_yoy` | `inflation_expectations_business` | 2 | 0.185438 | 0.229735 | 0.240182 | 1.295 | 1.045 |
| System A | `cpi_yoy` | `inflation_expectations_business` | 3 | 0.211901 | 0.256034 | 0.277061 | 1.308 | 1.082 |
| System B | `trimmed_mean_cpi_yoy` | `cash_rate` | 1 | 0.046010 | 0.047878 | 0.044321 | 0.963 | 0.926 |
| System B | `trimmed_mean_cpi_yoy` | `cash_rate` | 2 | 0.082849 | 0.081774 | 0.076596 | 0.925 | 0.937 |
| System B | `trimmed_mean_cpi_yoy` | `cash_rate` | 3 | 0.108209 | 0.106784 | 0.099904 | 0.923 | 0.936 |
| System B | `trimmed_mean_cpi_yoy` | `inflation_expectations_business` | 1 | 0.047921 | 0.057729 | 0.060777 | 1.268 | 1.053 |
| System B | `trimmed_mean_cpi_yoy` | `inflation_expectations_business` | 2 | 0.075770 | 0.086086 | 0.085438 | 1.128 | 0.992 |
| System B | `trimmed_mean_cpi_yoy` | `inflation_expectations_business` | 3 | 0.102668 | 0.114165 | 0.112828 | 1.099 | 0.988 |

Across these rows, the shipped block-4 bands average `1.198x` the old i.i.d.
widths for System A and `1.051x` for System B. Relative to the historical
block-4 probe, shipped widths average `1.051x` for System A and `0.972x` for
System B.
