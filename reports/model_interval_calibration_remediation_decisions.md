# Interval Calibration / Model Assumption Remediation Decisions

Date: 2026-08-26

Scope: decision record only, in the same spirit as
`model_refit_phase1_decisions.md`. Follows up on the interval-calibration
investigation (why conformal-calibrated intervals still under-cover on
`reports/model_interval_calibration_validation.csv`) and a full
classical-assumption audit of every shipped model (SARIMA, SARIMAX Group D,
Elastic Net). One fix shipped; three separately-motivated remediation
attempts were tried, measured with leakage-free walk-forward backtesting on
the current (post-SA-basis-refit) curated dataset, and rejected because each
one measurably hurt real out-of-sample RMSE and/or interval coverage despite
being individually well-motivated. SARIMAX itself was later fully removed
from the project (2026-08-26) after this audit; the findings below are the
evidence base for that decision.

## Assumption Audit Summary

One-shot diagnostics (Ljung-Box, Jarque-Bera, Breakvar heteroskedasticity,
AR/MA root moduli, VIF) fit on each shipped model's full-sample specification.

| Model | Autocorrelation | Normality | Homoskedasticity | Stationary/invertible | Multicollinearity |
| --- | --- | --- | --- | --- | --- |
| SARIMA headline `(1,0,2)x(1,0,2,4)` | Pass (all lag p>=0.30) | **Fails** (JB p~7e-80, kurtosis 11.7) | Pass (breakvar p=0.887) | **Fails** (AR root modulus 0.997; MA roots ~1.00) | n/a |
| SARIMA trimmed-mean `(1,1,1)x(0,0,1,4)` | Pass (all lag p>=0.50) | Pass (JB p=0.554) | **Fails** (breakvar p=0.005) | Pass | n/a |
| SARIMAX Group D `(1,0,1)x(0,0,1,4)` (removed 2026-08-26) | Pass (all lag p>=0.067) | Pass (JB p=0.807) | Pass (breakvar p=0.487) | **Fails** (AR root modulus 0.986) | Pass (max VIF 1.80) |
| Elastic Net headline (11 features) | n/a (not a residual-diagnostic model) | n/a | n/a | Features all stationary, ADF p<0.0001 | Pass (max VIF 4.31) |
| Elastic Net trimmed-mean primary-WTI (7 features) | n/a | n/a | n/a | Features all stationary | Pass (max VIF 2.20) |

`reports/eda_vif.csv`/`reports/eda_stationarity.csv` (2026-08-15) test a
broader pre-selection candidate screen -- raw rate levels and both
`wti_growth_lag1`/`brent_growth_lag1` together -- that no shipped model
actually uses; VIF recomputed on the real shipped feature sets above is clean
everywhere.

## Shipped Fix: Elastic Net Out-Of-Sample Residual Bootstrap

`src/models/elastic_net.py`'s `simulate_paths_from_fit` previously bootstrapped
from **in-sample** residuals (the final full-training-window pipeline
predicting on the same rows it was fit on) -- already flagged in the prior
code comment as understating true forecast-error variance. Replaced with a
manual `TimeSeriesSplit`-fold walk (`_time_series_oos_residuals`) using the
same hyperparameters `GridSearchCV` already selected, collecting genuinely
out-of-sample residuals. (`sklearn.model_selection.cross_val_predict` cannot
be used directly with `TimeSeriesSplit` here -- it requires every row to
appear in some test fold, which `TimeSeriesSplit`'s initial training-only
block never does.)

Raw (uncalibrated) walk-forward coverage, 80% nominal, identical origins,
current curated dataset, before vs after:

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

Improves every horizon by 9-13 points. Still well under the 80% target --
this closes the in-sample-understatement bug, it does not fully resolve
Elastic Net/ensemble's interval under-coverage. On the fresh 2026-08-26
regeneration of the full calibrated `reports/model_interval_coverage.csv`,
overall calibrated coverage is `sarima` 70.8%, `ensemble` 65.8%,
`elastic_net` 64.4% -- all still below 80% nominal.

## Rejected Attempt 1: SARIMA/SARIMAX Residual-Bootstrap Simulation

Motivation: headline SARIMA's badly non-normal residuals mean
`fitted.simulate()`'s Gaussian innovations understate tail risk. Implemented
a residual bootstrap -- resample the fit's own standardized innovations
(re-centered to zero mean, rescaled to unit variance) as `state_shocks` via
`fitted.simulate(..., state_shocks=..., pretransformed_state_shocks=False)`,
looped per-repetition since statsmodels does not support custom shocks with
`repetitions>1` in one vectorized call (benchmarked at ~0.7ms/call, so the
loop itself is not the problem).

First comparison (headline SARIMA h1: 69.2% raw -> 81.8% bootstrapped) looked
like an improvement, but that used a stale baseline: the committed
`reports/model_interval_coverage.csv` predated the 2026-08-25 SA-basis
refit. Recomputing the Gaussian baseline on the **current** dataset and
comparing properly:

| Horizon | Gaussian (current data) | Residual bootstrap (current data) |
| --- | ---: | ---: |
| Overall | 75.3% | 70.6% (worse) |
| 1 | 83.1% | 81.8% |
| 2 | 76.6% | 75.3% |
| 5 | 72.7% | 64.9% (worse) |
| 6 | 76.6% | 66.2% (worse) |
| 7 | 76.6% | 68.8% (worse) |
| 8 | 76.6% | 68.8% (worse) |

SARIMAX Group D (residuals already near-Gaussian, JB p=0.807) was
unaffected either way once compared to its own current-data Gaussian
baseline (66.7% both ways) -- confirmed via a Jarque-Bera-gated version
(bootstrap only when the fit's own JB test rejects normality) that produced
identical results to the always-bootstrap version, because Group D's
per-origin fits consistently fail to reject normality.

**Mechanism for the regression:** a leptokurtic (fat-tailed) empirical
distribution concentrates more probability mass near its center than a
Gaussian with the same variance. Resampling it can *narrow* the 10th-90th
percentile interval even though the true far tails are fatter -- exactly the
quantile range this project's intervals target. Reverted; `sarima.py` (and
`sarimax.py`, before its later full removal) went back to Gaussian
innovations, with the finding recorded in code comments.

## Rejected Attempt 2: SARIMAX Group D Order Re-Selection for Stability

Motivation: Group D's AR(1) root (modulus 0.986) sits inside the unit
circle -- not enforced against, since `enforce_stationarity=False`. An
AR/MA root-stability screen was added to `src/models/evaluation.py`
(`root_stability_summary`) and wired into `sarima_order_search.py` (the
SARIMA one; a SARIMAX equivalent existed at the time but was removed with
SARIMAX itself).

Feature pruning (dropping the 2 insignificant exogenous regressors) did not
fix the root instability at all (AR root modulus stayed ~0.986-0.987
regardless of feature set) -- the instability comes from the `(1,0,1)`
order itself, not the exogenous block. A full order/seasonal-order grid
search over the same 6-feature exog found a genuinely stable, near-AIC-tied
alternative:

| Order | Stable | Full-sample AIC | Walk-forward RMSE (h1, n=60) |
| --- | --- | ---: | ---: |
| `(1,0,1)x(0,0,1,4)` (shipped at the time) | No (AR root 0.986) | 214.41 | **0.753** |
| `(2,0,0)x(1,0,0,4)` (candidate) | Yes (AR root 1.370) | 214.47 | 0.822 (~9% worse) |

The stable candidate looked strictly better on every full-sample diagnostic
(cleaner Ljung-Box/JB/heteroskedasticity, more significant coefficients) but
forecast measurably worse out-of-sample. Not adopted. This finding, combined
with attempt 1's regression and Group D's persistent 3-of-6
not-significant-at-5% exogenous coefficients, is part of the evidence base
for removing SARIMAX from the project entirely on 2026-08-26 rather than
continuing to patch it.

## Rejected Attempt 3: Yeo-Johnson Target Transform for SARIMA

Motivation: a variance-stabilizing transform of `cpi_yoy` before fitting
SARIMA might reduce residual kurtosis/skew if the fat tails are partly a
scale-dependent (multiplicative-shock) artifact. Used Yeo-Johnson rather
than Box-Cox because `cpi_yoy` has 4 negative quarters (min -0.3 in
1997Q3/Q4 and 2020Q2); Box-Cox requires strictly positive values. Refit the
transform on each walk-forward training window only (no leakage), fit the
same `(1,0,2)x(1,0,2,4)` SARIMA on the transformed series, simulated paths on
the transformed scale, inverse-transformed each simulated draw back to the
original scale (avoids Jensen's-inequality point-forecast bias by
Monte-Carlo-averaging post-transform draws rather than analytically
correcting a transformed-scale mean).

Full-sample transformed-scale residual diagnostics (lambda=0.470, MLE-fit for
marginal normality of `cpi_yoy` itself):

| | Vanilla (original scale) | Yeo-Johnson (transformed scale) |
| --- | ---: | ---: |
| Jarque-Bera p | ~7e-80 | ~0.0 (still rejects) |
| Kurtosis | 11.74 | **13.51** (worse) |
| Skew | 0.51 | **-0.87** (worse, sign-flipped) |
| Heteroskedasticity p | 0.887 | 0.074 (worse, borderline) |

The transform, chosen specifically to maximize `cpi_yoy`'s marginal
normality, made the *model's residuals* less normal, not more -- evidence
that the fat tails come from a handful of genuine shock quarters (COVID,
2021-22 energy/inflation surge), not a smooth scale-dependent effect a
pointwise transform can remove. Real walk-forward comparison confirms this
translates to worse forecasts, not just a worse diagnostic:

| | Vanilla | Yeo-Johnson |
| --- | ---: | ---: |
| RMSE (overall) | 1.514 | 1.657 (~9% worse) |
| RMSE (every horizon 1-8) | -- | worse at every horizon |
| Coverage (overall, 80% nominal) | 75.3% | 73.7% (worse) |

Not adopted; no transform-related code was added to `sarima.py`.

## Why Three Independent Remediation Attempts All Failed The Same Way

Classical assumptions (normality, comfortably-stationary roots) exist for
inference validity -- trustworthy standard errors, p-values, intervals --
not for point-forecast accuracy, and satisfying them is not guaranteed to
improve either. On this dataset specifically:

- Near-unit-root AR behavior likely reflects genuine inflation persistence
  (a shock's effect decaying slowly is a real, documented macro phenomenon,
  not a numerical artifact). Forcing the fit toward comfortably-stationary
  roots implicitly asserts faster mean-reversion than the data shows --
  a real bias, not just "more stability."
- Fat-tailed residuals in a ~124-quarter sample are dominated by a small
  number of genuine structural-shock quarters (COVID, 2021-22 energy
  prices). No amount of resampling or pointwise transformation removes
  those events; both approaches just redistribute how the model reacts to
  everything else while still missing (or over-reacting to) the same
  handful of quarters.
- AIC/Jarque-Bera/Ljung-Box are in-sample fit diagnostics; walk-forward RMSE
  and coverage are genuinely out-of-sample. With ~100-125 observations
  these can diverge, and a specification that fits the historical sample
  best can be overfit to that sample's specific idiosyncrasies.

## Outcome: SARIMAX Fully Removed (2026-08-26)

Given two independent, validated remediation attempts for SARIMAX Group D
(residual-bootstrap simulation, order re-selection) both made real
out-of-sample performance worse, and the model's own coefficients were
persistently unreliable (3 of 6 not significant at 5%), SARIMAX
(`src/models/sarimax.py`, `sarimax_order_search.py`, their tests, and
`reports/model_comparison_sarimax.csv`) was removed from the project
entirely rather than continuing to patch it. The project's forecasting
families are now SARIMA, Elastic Net, and their Ensemble combiner. A
structural VAR (SVAR) is planned as a separate scenario/impulse-response
addition, specifically because its identification restrictions give a more
defensible causal reading for exogenous-variable sensitivity analysis than
an unregularized single-equation fit like SARIMAX ever did on this sample
size.

## Remaining Documented Limitations (Not Remediated)

- Headline SARIMA: non-normal residuals and near-unit-root/near-noninvertible
  fit. Two independent, validated remediation attempts (residual bootstrap,
  Yeo-Johnson transform) both made real forecast/coverage performance worse;
  not remediated further.
- Trimmed-mean SARIMA: heteroskedastic residuals (breakvar p=0.005). Not
  attempted -- a proportionate fix (e.g. GARCH-type variance modelling) is
  disproportionate complexity for a ~124-quarter sample, and the pattern
  above gives no reason to expect it would survive real walk-forward
  validation either.
- Elastic Net / ensemble: interval coverage improved by the shipped fix
  above but remains well under 80% nominal at every horizon. The residual
  bootstrap now correctly reflects out-of-sample uncertainty; the
  remaining gap is a genuine, currently-unresolved underdispersion in the
  base point-forecast model, not a bug in how residuals are sampled.

## Adopted Fix: Identity-Preserving Interval Calibration (2026-08-27)

After regenerating the trimmed-mean interval reports, the held-out validation
slice exposed a separate calibration bug that was not specific to the
trimmed-mean target and predated the Elastic Net residual-bootstrap fix:
`apply_interval_calibration` rebuilt calibrated intervals symmetrically around
the simulated-path median (`point_forecast_proxy`) using
`median +/- raw_half_width * scale_factor`. Raw intervals, however, are the
10th/90th percentiles of simulated paths, not necessarily symmetric around the
median. For skewed path distributions, `scale_factor=1.0` was therefore not an
identity operation and could move bounds even when calibration should have left
the raw quantile interval unchanged. This made held-out calibrated coverage
worse than raw for several model/horizon combinations, including
trimmed-mean Elastic Net at all 8 horizons in the stale report and headline
SARIMA at 6 of 8 horizons in the post-SARIMAX-removal report.

The adopted fix scales each side of the raw interval separately around the same
point forecast:

- `lower_dist = point_forecast_proxy - interval_lower`
- `upper_dist = interval_upper - point_forecast_proxy`
- `calibrated_lower = point_forecast_proxy - lower_dist * scale_factor`
- `calibrated_upper = point_forecast_proxy + upper_dist * scale_factor`

Regression coverage now asserts that an asymmetric interval is unchanged when
`scale_factor=1.0`. In addition, conformal scale factors are floored at `1.0`.
This no-shrink rule is a conservative calibration policy for this project
because all shipped raw interval families under-cover 80% nominal coverage;
shrinking intervals fights that established out-of-sample evidence.

All affected headline and trimmed-mean interval reports were regenerated after
the fix. On the held-out validation reports, no model/horizon row now has
calibrated coverage below raw coverage. Headline held-out overall coverage is:
Elastic Net 33.6% raw -> 35.2% calibrated, Ensemble 31.2% -> 38.3%, SARIMA
58.3% -> 58.3%. Trimmed-mean held-out overall coverage is: Elastic Net 46.1%
-> 46.1%, Ensemble 35.9% -> 39.8%, SARIMA 57.8% -> 63.0%. These remain well
below the 80% nominal target, so this is a bug fix and conservative guardrail,
not a full interval-calibration resolution.

Known simplification not resolved here: `nonconformity_score` is still
`abs(actual - point_forecast_proxy) / half_width`, a symmetric half-width
measure. It is not a proper asymmetric conformalized quantile regression
score. Pooling calibration across nearby horizons, using a larger/rolling
calibration window, or switching to a genuinely asymmetric conformal score are
future work, not implemented.


## Adopted Fix: Side-Specific Score + Ex-Ante Rolling Calibration (2026-09-19)

This implements the "future work" named at the end of the previous section: an
asymmetric conformal score, pooling across horizons, and a rolling window. It
applies to headline and trimmed-mean alike.

**What was wrong.**

1. *Score/application mismatch.* Calibration scaled each side of the interval
   around the point forecast, but `nonconformity_score` used the symmetric
   half-width. For a skewed raw interval the score calls a miss "covered": for
   trimmed-mean Elastic Net the raw interval's larger side is a median 2.56x the
   smaller (headline 1.42x, ensemble 1.13-1.33x, SARIMA 1.05x), and about half
   of its misses had a score <= 1. In the quiet pre-2020 period calibrated
   Elastic Net coverage was ~49% against an 80% target.
2. *Non-exchangeable calibration split.* The 70/30 split fitted factors on the
   quiet 2010-2019 origins and judged them on the 2020-2023 origins, where
   forecast errors were 2.4-2.6x (headline) and 2.9-3.7x (trimmed mean) their
   pre-2020 size. Trimmed mean was calmer before 2020, so its raw intervals
   (ensemble width 1.2 vs headline 2.3) absorbed the shock worse.
3. *Serving mismatch.* `api/main.py` rebuilt a symmetric `point +/- factor *
   half_width` interval while the coverage report scaled each side separately,
   so the API did not serve what the report evaluated.
4. *Reporting mismatch.* The trimmed-mean coverage report was written with
   `apply_calibration=False` (raw) while headline's was calibrated.

**What shipped.**

- Score = error / the interval's distance from the point on the side the
  actual landed (`evaluation.walk_forward_interval_coverage_backtest`).
- `compute_rolling_conformal_scale_factors`: each origin's factor uses only
  scores whose target quarter was already observed at that origin (no
  look-ahead), pooled across horizons over the last 12 target quarters,
  finite-sample-corrected 80th percentile, floored at 1.0, none below 8 scores.
- `compute_current_conformal_scale_factors`: the same window evaluated at the
  latest observed quarter; written to the factors CSV the API already reads
  (schema unchanged).
- `interval_coverage.py` derives its calibration from its own backtest (no
  dependency on a previously written factors file, so no two-pass ordering);
  `interval_calibration.py` runs one backtest instead of a calibration run plus
  a validation run and reports an ex-ante held-out comparison on the last 30%
  of origins. The stale-coverage-report origin-count guard was removed with the
  dependency it protected. The trimmed-mean coverage report is now calibrated.
- API scales each side around the point.

**Selection protocol.** Window and pooling were compared on saved raw backtests
(offline, no project reports written), with one shared window for both targets.
Averages over 3 families x 2 targets, pooled across horizons:

| Window (quarters) | Dev (<2020) mean abs. coverage error | Post-2020 coverage | Post-2020 width vs raw | Factor served as of latest quarter (mean / max) |
| --- | --- | --- | --- | --- |
| 4 | 0.068 | 66.1% | 3.32x | 1.00 / 1.00 |
| 8 | 0.072 | 61.5% | 2.92x | 1.34 / 1.81 |
| **12 (shipped)** | 0.058 | 58.7% | 2.58x | 2.61 / 3.61 |
| 16 | 0.050 | 57.3% | 2.29x | 3.94 / 5.51 |
| 20 | 0.051 | 54.4% | 2.04x | 3.33 / 4.28 |
| 24 | 0.051 | 53.0% | 1.88x | 3.10 / 4.21 |
| expanding | 0.057 | 50.9% | 1.48x | 1.64 / 2.00 |

The evidence does not identify a single best window. Development-only error is
flat (0.050-0.072); 16-24 quarters are marginally lowest, and 4-8 are noisier
because they pool few scores. Post-2020 coverage falls steadily as the window
lengthens, at the price of narrower intervals. 12 quarters was kept as a
middle choice: it is at least as good as the longer windows on post-2020
coverage, avoids the small-sample noise of 4-8, and serves a smaller factor
today than 16-24 (which still contain the 2022 error peak). An earlier version
of this note claimed 12 had the lowest development error; that held only
against the few alternatives compared first and is corrected here.

**Held-out result** (80% target, last 16 origins, 2020-2023, every origin
calibrated only from errors already observed; calibrated coverage / mean width):

| Target | Family | Old static split | New rolling |
| --- | --- | --- | --- |
| Trimmed mean | Elastic Net | 49.2% / 2.15 | 71.9% / 4.61 |
| Trimmed mean | Ensemble | 39.1% / 1.69 | 56.2% / 3.78 |
| Trimmed mean | SARIMA | 63.0% / 1.88 | 69.3% / 3.41 |
| Headline | Elastic Net | 48.4% / 2.71 | 59.4% / 5.76 |
| Headline | Ensemble | 46.9% / 2.54 | 55.5% / 6.18 |
| Headline | SARIMA | 58.3% / 2.77 | 62.0% / 4.29 |

All-origin coverage reports (80% nominal): trimmed mean SARIMA 77.8%, Elastic
Net 67.9%, ensemble 66.3% (previously raw 67.9/47.9/45.0); headline SARIMA
78.1%, Elastic Net 68.4%, ensemble 66.5% (previously 75.3/67.7/66.3).

**Limitations, stated plainly.**

- Coverage is still below 80% after 2020 for every family. No interval built
  from past errors can anticipate the 2021-23 error jump; the rolling method
  only follows it with a lag, and pays for it in width.
- Widths roughly double in the held-out period (headline ensemble 2.54 -> 6.18).
  Served factors as of the latest quarter are large because the last 12
  quarters still contain the 2023-25 disinflation errors: trimmed mean
  Elastic Net/ensemble/SARIMA 2.58/3.02/2.67, headline 2.13/3.62/1.64.
- The factor is pooled across horizons, so short-horizon intervals are widened
  by long-horizon errors (headline ensemble h1 80% interval is ~4.6 wide).
  Per-horizon rolling calibration was compared: it fixes trimmed mean's
  normal-period h1-2 over-coverage (81-84% vs 88%) but under-covers long
  horizons and headline; no variant dominated, so the simpler pooled design
  was kept. A longer window narrows the post-2020 backtest intervals (20
  quarters: 2.04x raw width vs 2.58x) at lower coverage (54% vs 59%), but does
  not narrow what would be served today, because it still contains the 2022
  error peak (see the table above). `DEFAULT_CALIBRATION_WINDOW_QUARTERS` in `evaluation.py` is the single
  knob; changing it needs one rerun of `interval_coverage` and
  `interval_calibration` per target.
- The window was chosen on quiet-period data that contains no volatility shock,
  so the choice among windows is weakly identified; the post-2020 numbers above
  are out-of-sample for it but they are one episode, not a distribution.
