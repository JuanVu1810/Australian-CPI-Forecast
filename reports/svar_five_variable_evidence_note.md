# SVAR Five-Variable Evidence Note

[![content: AI-generated](../book/australian_cpi_forecasting/_static/badges/ai-generated.svg)](https://juanvu1810.github.io/CPI-Forecast/intro.html#ai-acknowledgement)

Date: 2026-08-28

Scope: Phase 1b evidence note for the two five-variable SVAR systems gated in
`reports/svar_gate_decisions.md` (variables, lag order, cointegration rank,
Cholesky ordering, and the ADF/Johansen caveat are defined there, not
repeated here).

## Five-Variable Trade-Offs

Business inflation expectations is the strongest empirically supported
shared macro variable in simple contemporaneous screening: raw correlation
`0.697` with `cpi_yoy`, `0.595` with `trimmed_mean_cpi_yoy`.

Unemployment and the cash rate are retained on economic-theory grounds, not
raw correlation — weaker for both targets (`unemployment_rate` -0.385/-0.487,
`cash_rate` 0.236/0.274 for headline/trimmed mean), but they remain central
state and policy variables for impulse-response analysis.

Commodity growth is retained over the stronger raw PPI alternative
(`ppi_growth` correlation 0.522 headline / 0.384 trimmed mean) because it
supports the scenario feature and preserves sample coverage: swapping to
`ppi_growth` would shrink complete-case coverage from 123 to 109 observations
(loses 1995Q2-1997Q4).

## Phase 1b Treatment Result

System B (trimmed mean) is the primary system and gets full treatment:
recursive Cholesky IRFs with 80% block-residual-bootstrap bands, VAR
diagnostics, and a diagnostic walk-forward backtest.

System A (headline) was initially confirmatory, but the fixed escalation
check found non-overlapping 80% block-bootstrap IRF bands against System B at
four shared-shock horizons (`cash_rate` horizons 1-3,
`inflation_expectations_business` horizon 2), so it was escalated to full
treatment too. The earlier i.i.d. bootstrap had triggered five horizons;
after the block-bootstrap replacement, `inflation_expectations_business`
horizon 1 no longer triggers (its 80% bands now overlap).

Backtest RMSE below is diagnostic only — SVAR isn't used as a
forecast-accuracy competitor to SARIMA, Elastic Net, or the Ensemble.

## Persisted Phase 1b Diagnostics

Full-sample VAR(2)-in-levels diagnostics for both systems. `reject_5pct=yes`
means the test rejects at 5%.

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

Walk-forward, `DEFAULT_INITIAL_TRAIN_SIZE=32`, horizons 1-8. The SVAR origin
grid hasn't been verified against the SARIMA/Elastic Net/Ensemble comparison
grids, so these aren't directly comparable to those models even informally.

| System | Target | Overall RMSE | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| System A | `cpi_yoy` | 1.791583 | 0.843101 | 1.252857 | 1.642059 | 1.966088 | 2.064319 | 2.065749 | 2.033528 | 2.042510 |
| System B | `trimmed_mean_cpi_yoy` | 1.206149 | 0.305645 | 0.577790 | 0.848872 | 1.116049 | 1.325052 | 1.483851 | 1.591184 | 1.659975 |

## COVID-Era Diagnostic Sensitivity

Two intervention quarters (`2020Q2` shock-down, `2020Q3` rebound) were
checked two ways: excluding both rows, and fitting the same VAR(2)-in-levels
with them as exogenous dummies.

COVID treatment weakens some diagnostics but doesn't resolve the core
failures: both systems still reject multivariate whiteness and normality
under either treatment. Normality improves materially under dummying (the
unemployment Jarque-Bera failure resolves), but other residual series still
reject. ARCH-LM failures partly improve: System B's
`inflation_expectations_business` rejection resolves under both treatments,
and its own target ARCH-LM rejection resolves when COVID quarters are
excluded, but System A's target ARCH-LM rejection persists regardless.

| System | COVID treatment | Whiteness stat | Whiteness p | Normality stat | Normality p |
| --- | --- | ---: | ---: | ---: | ---: |
| System A | Full sample | 283.464294 | 0.000094 | 3074.523062 | 0.000000 |
| System A | Exclude `2020Q2`/`2020Q3` | 273.119255 | 0.000452 | 2283.146293 | 0.000000 |
| System A | Dummy `2020Q2`/`2020Q3` | 267.196082 | 0.001049 | 616.723640 | 0.000000 |
| System B | Full sample | 317.096153 | 0.000000253 | 2774.021940 | 0.000000 |
| System B | Exclude `2020Q2`/`2020Q3` | 299.177357 | 0.000007 | 2147.311178 | 0.000000 |
| System B | Dummy `2020Q2`/`2020Q3` | 298.007220 | 0.000008 | 578.874441 | 0.000000 |

Selected equation-level changes:

| System | Variable | Test | Full-sample p | Exclude-COVID p | Dummy-COVID p | Read |
| --- | --- | --- | ---: | ---: | ---: | --- |
| System A | `unemployment_rate` | Jarque-Bera | 0.000000 | 0.000000 | 0.115153 | Resolves only with dummies |
| System A | `cpi_yoy` | ARCH-LM | 0.002477 | 0.015802 | 0.031329 | Persists, weaker |
| System B | `unemployment_rate` | Jarque-Bera | 0.000000 | 0.000000 | 0.110974 | Resolves only with dummies |
| System B | `trimmed_mean_cpi_yoy` | ARCH-LM | 0.006283 | 0.227851 | 0.015966 | Resolves when excluded; persists with dummies |
| System B | `inflation_expectations_business` | ARCH-LM | 0.043744 | 0.079466 | 0.152525 | Resolves under both treatments |

## Bootstrap Validity Probe

The shipped IRF bootstrap (`src/models/svar.py`) is a contiguous residual
block bootstrap, block length `DEFAULT_ARCH_LAGS = 4`, non-wrapping
contiguous runs drawn from the fitted residuals and trimmed only at the final
boundary. This replaced the earlier i.i.d. residual bootstrap because the
persistent whiteness/normality/ARCH-LM failures mean single-row residual
draws are too optimistic for dependence-sensitive IRF uncertainty.

A scratch prototype compared i.i.d. against block lengths 4 and 8 at the same
80% quantiles/horizons: bands were generally wider for
`inflation_expectations_business` shocks and mixed for `cash_rate` shocks.
Across six checked rows per system, mean block-to-i.i.d. width ratios were
`1.137` (block 4) and `1.149` (block 8) for System A, `1.078` and `1.177` for
System B — enough evidence to ship block length 4 before Phase 2 consumed the
IRF bands.

After shipping and rerunning `bootstrap_cholesky_irf_bands` (1000
replications), the same rows compare as follows (differences from the
prototype are expected stochastic variation):

| System | Response | Shock | Horizon | Old IID width | Prototype block-4 width | Shipped block-4 width | Shipped / IID | Shipped / prototype |
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

Across these rows, shipped block-4 bands average `1.198x` the old i.i.d.
widths for System A and `1.051x` for System B; relative to the scratch
prototype, `1.051x` for System A and `0.972x` for System B.
