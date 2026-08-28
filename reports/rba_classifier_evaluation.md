# RBA Policy Action Classifier Evaluation

## Scope

Uses the leakage-safe Ensemble horizon-1 headline and trimmed-mean CPI forecasts from the existing backtest prediction reports. No forecasting model is refit or resimulated; the ordinal classifiers are fit only on the two as-of forecast features.

## Joined Sample

- Joined usable quarters: 53
- Quarter range: 2011Q1 to 2024Q1

| class   |   full_sample |   chosen_initial_train |   chosen_test |
|:--------|--------------:|-----------------------:|--------------:|
| cut     |            22 |                      9 |            13 |
| hold    |            22 |                      2 |            20 |
| hike    |             9 |                      1 |             8 |

## Walk-Forward Split Gate

Selected initial_train_size=12: it is the earliest audited candidate at or above the 12-row fit-size floor with zero training folds missing cut, hold, or hike. Baseline, logit, and probit are all evaluated on the same 41 expanding-window test quarters.

No audited candidate at or above the chosen fit-size floor had a degenerate training fold. Earlier candidates flagged by the gate are shown below.

|   initial_train_size |   test_rows | first_test_quarter   |   min_train_cut |   min_train_hold |   min_train_hike |   degenerate_fold_count | degenerate_classes   |
|---------------------:|------------:|:---------------------|----------------:|-----------------:|-----------------:|------------------------:|:---------------------|
|                    1 |          52 | 2011Q2               |               0 |                0 |                1 |                       3 | cut, hold            |
|                    2 |          51 | 2011Q3               |               0 |                1 |                1 |                       2 | cut                  |
|                    3 |          50 | 2011Q4               |               0 |                2 |                1 |                       1 | cut                  |
|                    4 |          49 | 2012Q1               |               1 |                2 |                1 |                       0 | none                 |
|                    8 |          45 | 2013Q1               |               5 |                2 |                1 |                       0 | none                 |
|                   12 |          41 | 2014Q1               |               9 |                2 |                1 |                       0 | none                 |
|                   16 |          37 | 2015Q1               |               9 |                6 |                1 |                       0 | none                 |
|                   20 |          33 | 2016Q1               |              12 |                7 |                1 |                       0 | none                 |
|                   24 |          29 | 2017Q1               |              15 |                8 |                1 |                       0 | none                 |
|                   32 |          21 | 2019Q1               |              15 |               16 |                1 |                       0 | none                 |
|                   40 |          13 | 2021Q1               |              21 |               18 |                1 |                       0 | none                 |

## Macro-F1 Comparison

| model          |   macro_f1 |   accuracy |   n |
|:---------------|-----------:|-----------:|----:|
| threshold      |      0.775 |      0.756 |  41 |
| ordered_logit  |      0.677 |      0.634 |  41 |
| ordered_probit |      0.643 |      0.610 |  41 |

Macro-F1 is the comparison metric because policy holds are common enough that raw accuracy can overstate usefulness.

## Small-Sample Caveat

The walk-forward test set has 41 quarters. At this size, the macro-F1 ranking (threshold 0.775, ordered_logit 0.677, ordered_probit 0.643) is suggestive rather than decisive.

As a quick uncertainty check, a paired bootstrap over the same test quarters (2,000 resamples, seed=42) gives the following macro-F1 gap intervals. This is a rough diagnostic because it treats quarters as exchangeable and does not model time-series dependence.

| gap                        |   observed_macro_f1_gap | paired_bootstrap_95pct_ci   |
|:---------------------------|------------------------:|:----------------------------|
| threshold - ordered_logit  |                   0.098 | [-0.060, 0.255]             |
| threshold - ordered_probit |                   0.132 | [-0.035, 0.315]             |

## Confusion Matrices

### threshold

| actual   |   cut |   hold |   hike |
|:---------|------:|-------:|-------:|
| cut      |    10 |      3 |      0 |
| hold     |     5 |     13 |      2 |
| hike     |     0 |      0 |      8 |

### ordered_logit

| actual   |   cut |   hold |   hike |
|:---------|------:|-------:|-------:|
| cut      |     9 |      4 |      0 |
| hold     |     9 |     11 |      0 |
| hike     |     0 |      2 |      6 |

### ordered_probit

| actual   |   cut |   hold |   hike |
|:---------|------:|-------:|-------:|
| cut      |     9 |      4 |      0 |
| hold     |     9 |     11 |      0 |
| hike     |     0 |      3 |      5 |

## Interpretive Conclusion

The threshold baseline's macro-F1 lead is a point-estimate win, not a statistically settled result: the paired-bootstrap intervals above cross zero for both threshold-minus-ordered-model gaps.

Despite that non-significance, the threshold rule is still the preferred reportable Phase 3 result on non-statistical grounds. It is transparent, has zero fitted parameters, and its main advantage is hike detection: it correctly classifies all 8 of 8 hike quarters, while ordered logit classifies 2 hikes as hold and ordered probit classifies 3 hikes as hold. That hike-to-hold error is a structurally different, and arguably more consequential, failure mode for a policy classifier than the aggregate macro-F1 gap alone captures.

A plausible reading is that the two CPI forecast features are highly correlated and the earliest valid training folds are thin, so the ordered models smooth some high-inflation tightening quarters back toward hold. The recommendation is therefore a judgment call made despite non-significance: keep ordered logit and probit as a documented negative finding for Phase 4's materiality section rather than exposing them via an API endpoint.
