# Interval Calibration / Model Assumption Remediation Decisions

[![content: AI-generated](../book/australian_cpi_forecasting/_static/badges/ai-generated.svg)](https://juanvu1810.github.io/Australian-CPI-Forecast/intro.html#ai-acknowledgement)

Date: 2026-08-26 (latest fix: 2026-09-19)

Decision record: why conformal-calibrated intervals still under-cover, and what
was tried. One fix shipped from a full classical-assumption audit of every
shipped model (SARIMA, SARIMAX Group D, Elastic Net); three well-motivated
remediation attempts were rejected because each measurably hurt real
walk-forward RMSE and/or coverage. SARIMAX was later removed from the project
entirely (2026-08-26); the findings below are the evidence base for that
decision.

## Assumption Audit Summary

One-shot diagnostics (Ljung-Box, Jarque-Bera, breakvar heteroskedasticity,
AR/MA root moduli, VIF) on each shipped model's full-sample specification.

| Model | Autocorrelation | Normality | Homoskedasticity | Stationary/invertible | Multicollinearity |
| --- | --- | --- | --- | --- | --- |
| SARIMA headline `(1,0,2)x(1,0,2,4)` | Pass (all lag p>=0.30) | **Fails** (JB p~7e-80, kurtosis 11.7) | Pass (breakvar p=0.887) | **Fails** (AR root modulus 0.997; MA roots ~1.00) | n/a |
| SARIMA trimmed-mean `(1,1,1)x(0,0,1,4)` | Pass (all lag p>=0.50) | Pass (JB p=0.554) | **Fails** (breakvar p=0.005) | Pass | n/a |
| SARIMAX Group D `(1,0,1)x(0,0,1,4)` (removed 2026-08-26) | Pass (all lag p>=0.067) | Pass (JB p=0.807) | Pass (breakvar p=0.487) | **Fails** (AR root modulus 0.986) | Pass (max VIF 1.80) |
| Elastic Net headline (11 features) | n/a | n/a | n/a | Features all stationary, ADF p<0.0001 | Pass (max VIF 4.31) |
| Elastic Net trimmed-mean primary-WTI (7 features) | n/a | n/a | n/a | Features all stationary | Pass (max VIF 2.20) |

`reports/eda_vif.csv`/`reports/eda_stationarity.csv` test a broader
pre-selection candidate screen that no shipped model actually uses; VIF
recomputed on the real shipped feature sets above is clean everywhere.

## Shipped Fix: Elastic Net Out-Of-Sample Residual Bootstrap

`simulate_paths_from_fit` bootstrapped from **in-sample** residuals (the final
pipeline predicting on rows it was fit on), understating forecast-error
variance. Replaced with a manual `TimeSeriesSplit`-fold walk
(`_time_series_oos_residuals`) collecting genuinely out-of-sample residuals —
`cross_val_predict` doesn't work here since it requires every row to appear in
some test fold, which `TimeSeriesSplit`'s initial training-only block never does.

Raw walk-forward coverage, 80% nominal, before vs after:

| Horizon | In-sample (before) | Out-of-sample (after) |
| --- | ---: | ---: |
| Overall | 52.1% | **62.3%** |
| 1 | 58.5% | **71.7%** |
| 2 | 50.9% | **66.0%** |
| 3 | 49.1% | **58.5%** |
| 4 | 50.9% | **62.3%** |
| 5 | 50.9% | **58.5%** |
| 6 | 52.8% | **56.6%** |
| 7 | 52.8% | **60.4%** |
| 8 | 50.9% | **64.2%** |

Improves every horizon by 9-13 points, but still well under the 80% target:
this closes the in-sample-understatement bug, not the full under-coverage.
Fresh calibrated coverage: `sarima` 70.8%, `ensemble` 65.8%, `elastic_net` 64.4%.

## Rejected Attempt 1: SARIMA/SARIMAX Residual-Bootstrap Simulation

Motivation: headline SARIMA's badly non-normal residuals mean Gaussian
`simulate()` innovations understate tail risk. Tried resampling the fit's own
standardized innovations as `state_shocks`. A first comparison looked like an
improvement but used a stale pre-refit baseline; recomputed on the current
dataset:

| Horizon | Gaussian (current data) | Residual bootstrap (current data) |
| --- | ---: | ---: |
| Overall | 75.3% | 70.6% (worse) |
| 1 | 83.1% | 81.8% |
| 5 | 72.7% | 64.9% (worse) |
| 6 | 76.6% | 66.2% (worse) |
| 7 | 76.6% | 68.8% (worse) |
| 8 | 76.6% | 68.8% (worse) |

SARIMAX Group D (near-Gaussian residuals, JB p=0.807) was unaffected either
way (66.7% both). **Mechanism:** a fat-tailed empirical distribution
concentrates more mass near its center than a Gaussian of the same variance,
so resampling it can *narrow* the 10th-90th percentile interval even though
the true tails are fatter — exactly the range this project's intervals target.
Reverted to Gaussian innovations.

## Rejected Attempt 2: SARIMAX Group D Order Re-Selection

Motivation: Group D's AR(1) root (modulus 0.986) sits inside the unit circle.
Feature pruning didn't fix it (instability came from the `(1,0,1)` order
itself, not the exogenous block); a full grid search found a stable,
near-AIC-tied alternative:

| Order | Stable | Full-sample AIC | Walk-forward RMSE (h1, n=60) |
| --- | --- | ---: | ---: |
| `(1,0,1)x(0,0,1,4)` (shipped at the time) | No (AR root 0.986) | 214.41 | **0.753** |
| `(2,0,0)x(1,0,0,4)` (candidate) | Yes (AR root 1.370) | 214.47 | 0.822 (~9% worse) |

The stable candidate looked strictly better on every diagnostic but forecast
measurably worse out-of-sample. Not adopted; combined with Attempt 1 and
Group D's persistently unreliable coefficients (3 of 6 not significant at
5%), this fed the decision to remove SARIMAX entirely.

## Rejected Attempt 3: Yeo-Johnson Target Transform for SARIMA

Motivation: a variance-stabilizing transform of `cpi_yoy` might reduce
residual kurtosis/skew if the fat tails are a scale-dependent artifact. Used
Yeo-Johnson, not Box-Cox, since `cpi_yoy` has negative quarters. Refit the
transform per walk-forward window, fit the same SARIMA order on the
transformed series, and inverse-transformed simulated draws back (avoiding
Jensen's-inequality bias via Monte-Carlo-averaging, not analytical correction).

| | Vanilla (original scale) | Yeo-Johnson (transformed scale) |
| --- | ---: | ---: |
| Jarque-Bera p | ~7e-80 | ~0.0 (still rejects) |
| Kurtosis | 11.74 | **13.51** (worse) |
| Skew | 0.51 | **-0.87** (worse, sign-flipped) |
| Heteroskedasticity p | 0.887 | 0.074 (worse, borderline) |

| | Vanilla | Yeo-Johnson |
| --- | ---: | ---: |
| RMSE (overall) | 1.514 | 1.657 (~9% worse, and worse at every horizon) |
| Coverage (overall, 80% nominal) | 75.3% | 73.7% (worse) |

A transform chosen specifically to maximize `cpi_yoy`'s marginal normality
made the *model's residuals* less normal — evidence the fat tails come from
genuine shock quarters (COVID, 2021-22 energy/inflation surge), not a smooth,
removable scale effect. Not adopted.

## Why All Three Attempts Failed The Same Way

Classical assumptions (normality, comfortably-stationary roots) exist for
inference validity, not forecast accuracy — satisfying them doesn't guarantee
improving either:

- Near-unit-root AR behaviour likely reflects genuine inflation persistence,
  not a numerical artifact; forcing stationary roots implicitly asserts
  faster mean-reversion than the data shows, a real bias.
- Fat-tailed residuals in a ~124-quarter sample are dominated by a handful of
  genuine structural-shock quarters (COVID, 2021-22 energy). No resampling or
  transform removes those events; both just redistribute how the model reacts
  to everything else.
- AIC/Jarque-Bera/Ljung-Box are in-sample diagnostics; walk-forward RMSE and
  coverage are out-of-sample. With ~100-125 observations these can diverge,
  and the best in-sample fit can be overfit to that sample's idiosyncrasies.

## Outcome: SARIMAX Fully Removed (2026-08-26)

Two independent, validated remediation attempts both made real out-of-sample
performance worse, and Group D's coefficients were persistently unreliable (3
of 6 not significant at 5%). SARIMAX (`src/models/sarimax.py`,
`sarimax_order_search.py`, their tests, `reports/model_comparison_sarimax.csv`)
was removed entirely rather than patched further. SARIMA, Elastic Net and
their Ensemble are now the project's forecasting families; SVAR was added
separately for scenario/impulse-response work, since its identification
restrictions give a more defensible causal reading than an unregularized
single-equation SARIMAX at this sample size.

## Remaining Documented Limitations (Not Remediated)

- Headline SARIMA: non-normal, near-unit-root residuals. Two validated
  remediation attempts (residual bootstrap, Yeo-Johnson) both made real
  performance worse; not remediated further.
- Trimmed-mean SARIMA: heteroskedastic residuals (breakvar p=0.005). Not
  attempted: GARCH-type variance modelling is disproportionate for a
  ~124-quarter sample and unlikely to survive walk-forward validation.
- Elastic Net/ensemble: coverage improved by the shipped fix but remains well
  under 80% nominal at every horizon — a genuine, unresolved underdispersion
  in the base point-forecast model, not a residual-sampling bug.

## Adopted Fix: Identity-Preserving Interval Calibration (2026-08-27)

`apply_interval_calibration` rebuilt calibrated intervals symmetrically
around the simulated-path median (`median +/- raw_half_width * scale_factor`),
but raw intervals are 10th/90th simulated-path percentiles, not necessarily
symmetric. For skewed paths, `scale_factor=1.0` wasn't an identity operation,
making calibrated coverage worse than raw for several model/horizon
combinations (all 8 trimmed-mean Elastic Net horizons; 6 of 8 headline SARIMA).

Fixed to scale each side of the interval separately around the same point
forecast (`lower_dist`/`upper_dist` scaled independently), and floored
conformal scale factors at 1.0 — a conservative no-shrink policy since every
shipped raw interval family under-covers 80% nominal. Regression tests now
assert an asymmetric interval is unchanged at `scale_factor=1.0`.

Held-out overall coverage, raw -> calibrated: headline Elastic Net 33.6% ->
35.2%, Ensemble 31.2% -> 38.3%, SARIMA 58.3% -> 58.3%; trimmed mean Elastic
Net 46.1% -> 46.1%, Ensemble 35.9% -> 39.8%, SARIMA 57.8% -> 63.0%. Still well
below 80%, a bug fix and conservative guardrail, not a full resolution.

Known simplification carried forward: `nonconformity_score` was still a
symmetric half-width measure, not a proper asymmetric conformalized quantile
score. A larger/rolling window and a genuinely asymmetric score were left as
future work, implemented next.

## Adopted Fix: Side-Specific Score + Ex-Ante Rolling Calibration (2026-09-19)

Implements that future work: an asymmetric conformal score, pooling across
horizons, and a rolling window, for both targets.

**What was wrong.**

1. *Score/application mismatch.* The score used a symmetric half-width even
   though calibration scaled each side separately, so a skewed raw interval
   could call a miss "covered" (trimmed-mean Elastic Net's larger side is a
   median 2.56x the smaller). Quiet pre-2020 calibrated Elastic Net coverage
   was ~49% against an 80% target.
2. *Non-exchangeable calibration split.* The old 70/30 split fitted factors
   on quiet 2010-2019 origins and judged them on 2020-2023 origins, where
   forecast errors were 2.4-3.7x their pre-2020 size.
3. *Serving mismatch.* `api/main.py` rebuilt a symmetric interval while the
   coverage report scaled each side separately.
4. *Reporting mismatch.* The trimmed-mean report was uncalibrated while
   headline's was calibrated.

**What shipped:** a side-specific score (error / the interval's distance to
the point on the side the actual landed); `compute_rolling_conformal_scale_factors`
using only already-observed scores, pooled across horizons over the last 12
target quarters, floored at 1.0; the same logic for the live "current" factor
the API serves; and the API now scales each side around the point to match
what the coverage report evaluates.

**Window selection** (offline, averaged over 3 families x 2 targets, pooled
across horizons):

| Window (quarters) | Dev (<2020) mean abs. coverage error | Post-2020 coverage | Post-2020 width vs raw |
| --- | --- | --- | --- |
| 4 | 0.068 | 66.1% | 3.32x |
| 8 | 0.072 | 61.5% | 2.92x |
| **12 (shipped)** | 0.058 | 58.7% | 2.58x |
| 16 | 0.050 | 57.3% | 2.29x |
| 20 | 0.051 | 54.4% | 2.04x |
| 24 | 0.051 | 53.0% | 1.88x |
| expanding | 0.057 | 50.9% | 1.48x |

No window dominates: development error is flat, 16-24 are marginally lowest,
and post-2020 coverage falls as the window lengthens. 12 quarters was kept as
a middle choice — as good as longer windows on post-2020 coverage, avoids the
small-sample noise of 4-8, and serves a smaller factor today than 16-24
(which still contain the 2022 error peak).

**Held-out result** (80% target, last 16 origins, 2020-2023, calibrated only
from already-observed errors; calibrated coverage / mean width):

| Target | Family | Old static split | New rolling |
| --- | --- | --- | --- |
| Trimmed mean | Elastic Net | 49.2% / 2.15 | 71.9% / 4.61 |
| Trimmed mean | Ensemble | 39.1% / 1.69 | 56.2% / 3.78 |
| Trimmed mean | SARIMA | 63.0% / 1.88 | 69.3% / 3.41 |
| Headline | Elastic Net | 48.4% / 2.71 | 59.4% / 5.76 |
| Headline | Ensemble | 46.9% / 2.54 | 55.5% / 6.18 |
| Headline | SARIMA | 58.3% / 2.77 | 62.0% / 4.29 |

All-origin coverage (80% nominal): trimmed mean SARIMA 77.8%, Elastic Net
67.9%, ensemble 66.3%; headline SARIMA 78.1%, Elastic Net 68.4%, ensemble
66.5%.

**Limitations, stated plainly.**

- Coverage is still below 80% after 2020 for every family. No interval built
  from past errors can anticipate the 2021-23 jump; the rolling method only
  follows it with a lag, paying for it in width (held-out widths roughly
  double, e.g. headline ensemble 2.54 -> 6.18).
- The factor is pooled across horizons, so short-horizon intervals are
  widened by long-horizon errors. Per-horizon calibration was compared: it
  fixes trimmed mean's normal-period h1-2 over-coverage but under-covers long
  horizons and headline, so the simpler pooled design was kept.
  `DEFAULT_CALIBRATION_WINDOW_QUARTERS` in `evaluation.py` is the single
  knob; changing it needs one rerun of `interval_coverage` and
  `interval_calibration` per target.
- The window was chosen on quiet-period data with no volatility shock, so the
  choice among windows is weakly identified; the post-2020 numbers are
  out-of-sample for it but are one episode, not a distribution.
