# RBA Policy Action Classifier Evaluation

## Scope

Uses the leakage-safe Ensemble horizon-1 headline and trimmed-mean CPI forecasts from the existing backtest prediction reports, plus the curated lag-safe unemployment-rate change feature.

Feature sets are fixed before evaluation: the threshold baseline uses headline_forecast only; the Taylor-rule baseline uses headline_forecast and unemployment_rate_change_lag1 only; the estimated Taylor-rule model fits cash_rate_change on headline inflation_gap and unemployment_rate_change_lag1; ordered logit and ordered probit use headline_forecast, trimmed_mean_forecast, and unemployment_rate_change_lag1; Frank-Hall XGBoost uses those three plus taylor_implied_change.

The majority_vote_ensemble is not a fitted model. It takes the same quarter's already-computed threshold, taylor_rule_estimated, ordered_logit, and ordered_probit predictions and applies a plain majority vote. The fixed Taylor rule and Frank-Hall XGBoost are deliberately excluded because the existing report already established them as clearly weaker. Any 2-2 split is resolved by the threshold prediction before inspecting tie frequency or performance.

The Taylor-rule baseline is a deliberate simplification of a textbook Taylor rule: unemployment_rate_change_lag1 is used as an Okun's-law-style change term instead of an unemployment gap, because the project does not estimate NAIRU or potential output. The rule uses fixed 0.5/0.5 Taylor coefficients, the RBA 2.5% target-band midpoint, and the joined sample's mean real policy rate, cash_rate - headline_forecast, as r_star (-0.560); cash_rate_lag1 is used only as the status-quo rate for discretizing the implied change. The negative r_star reflects this sample's average real-rate proxy rather than a nominal cash-rate mean.

The estimated Taylor-rule model is refit separately on each expanding training fold by OLS with an intercept: cash_rate_change is regressed on inflation_gap = headline_forecast - 2.5 and unemployment_rate_change_lag1. Its predicted continuous change is discretized with the fixed +/-0.125 percentage point hold band.

Frank-Hall XGBoost is fit as two binary classifiers, P(Y > cut) and P(Y > hold). Unlike ordered logit and probit, it additionally receives the continuous taylor_implied_change feature computed from the same fixed Taylor-rule calibration. XGBoost-vs-ordered-model comparisons are therefore not clean like-for-like feature comparisons. Its conservative hyperparameters are chosen for the small sample: max_depth=2, n_estimators=50, min_child_weight=3, subsample=0.75, colsample_bytree=0.75, reg_alpha=0.1, and reg_lambda=2.0. These settings are not searched or tuned.

## Joined Sample

- Joined usable quarters: 53
- Quarter range: 2011Q1 to 2024Q1

| class   |   full_sample |   chosen_initial_train |   chosen_test |
|:--------|--------------:|-----------------------:|--------------:|
| cut     |            22 |                      9 |            13 |
| hold    |            22 |                      2 |            20 |
| hike    |             9 |                      1 |             8 |

## Walk-Forward Split Gate

Selected initial_train_size=12: it is the earliest audited candidate at or above the 12-row fit-size floor with zero training folds missing cut, hold, hike, or either Frank-Hall binary outcome. All seven classifier outputs are evaluated on the same 41 expanding-window test quarters.

No audited candidate at or above the chosen fit-size floor had a degenerate training fold. Earlier candidates flagged by the gate are shown below.

|   initial_train_size |   test_rows | first_test_quarter   |   min_train_cut |   min_train_hold |   min_train_hike |   degenerate_fold_count | degenerate_classes   |   min_binary_gt_cut_negative |   min_binary_gt_cut_positive |   min_binary_gt_hold_negative |   min_binary_gt_hold_positive |   degenerate_binary_fold_count | degenerate_binary_subproblems   |
|---------------------:|------------:|:---------------------|----------------:|-----------------:|-----------------:|------------------------:|:---------------------|-----------------------------:|-----------------------------:|------------------------------:|------------------------------:|-------------------------------:|:--------------------------------|
|                    1 |          52 | 2011Q2               |               0 |                0 |                1 |                       3 | cut, hold            |                            0 |                            1 |                             0 |                             1 |                              3 | Y>cut, Y>hold                   |
|                    2 |          51 | 2011Q3               |               0 |                1 |                1 |                       2 | cut                  |                            0 |                            2 |                             1 |                             1 |                              2 | Y>cut                           |
|                    3 |          50 | 2011Q4               |               0 |                2 |                1 |                       1 | cut                  |                            0 |                            3 |                             2 |                             1 |                              1 | Y>cut                           |
|                    4 |          49 | 2012Q1               |               1 |                2 |                1 |                       0 | none                 |                            1 |                            3 |                             3 |                             1 |                              0 | none                            |
|                    8 |          45 | 2013Q1               |               5 |                2 |                1 |                       0 | none                 |                            5 |                            3 |                             7 |                             1 |                              0 | none                            |
|                   12 |          41 | 2014Q1               |               9 |                2 |                1 |                       0 | none                 |                            9 |                            3 |                            11 |                             1 |                              0 | none                            |
|                   16 |          37 | 2015Q1               |               9 |                6 |                1 |                       0 | none                 |                            9 |                            7 |                            15 |                             1 |                              0 | none                            |
|                   20 |          33 | 2016Q1               |              12 |                7 |                1 |                       0 | none                 |                           12 |                            8 |                            19 |                             1 |                              0 | none                            |
|                   24 |          29 | 2017Q1               |              15 |                8 |                1 |                       0 | none                 |                           15 |                            9 |                            23 |                             1 |                              0 | none                            |
|                   32 |          21 | 2019Q1               |              15 |               16 |                1 |                       0 | none                 |                           15 |                           17 |                            31 |                             1 |                              0 | none                            |
|                   40 |          13 | 2021Q1               |              21 |               18 |                1 |                       0 | none                 |                           21 |                           19 |                            39 |                             1 |                              0 | none                            |

## Macro-F1 Comparison

| model                  |   macro_f1 |   accuracy |   n |
|:-----------------------|-----------:|-----------:|----:|
| threshold              |      0.775 |      0.756 |  41 |
| majority_vote_ensemble |      0.769 |      0.732 |  41 |
| taylor_rule_estimated  |      0.769 |      0.732 |  41 |
| ordered_logit          |      0.715 |      0.683 |  41 |
| ordered_probit         |      0.696 |      0.659 |  41 |
| taylor_rule            |      0.365 |      0.415 |  41 |
| frank_hall_xgboost     |      0.274 |      0.366 |  41 |

Macro-F1 is the comparison metric because policy holds are common enough that raw accuracy can overstate usefulness.

## Why The Threshold Rule Uses Headline CPI

The threshold baseline applies the RBA's published 2-3% target band to the headline CPI forecast, because that band is a target for CPI inflation. Trimmed mean is the underlying-inflation measure, and it does enter the ordered logit, ordered probit and Frank-Hall models as a feature. As a sensitivity check, the same rule was also applied to the trimmed-mean forecast over the same test quarters:

| threshold_input       |   macro_f1 |   accuracy |   predicted_cut |   predicted_hold |   predicted_hike |
|:----------------------|-----------:|-----------:|----------------:|-----------------:|-----------------:|
| headline (reportable) |      0.775 |      0.756 |              15 |               16 |               10 |
| trimmed mean          |      0.696 |      0.634 |              22 |               11 |                8 |

The two inputs give different calls in 15 of 41 test quarters; compare each input's predicted-action counts with the actual test counts above to see where the calls differ. This check was run after the model design was fixed and on the same test quarters as every other comparison, so treat it as a sensitivity check rather than a model-selection step; the gap between the two inputs has no bootstrap interval.

## Prediction Confidence

Confidence is the probability assigned to the predicted action by that model's own class-probability calculation.

Threshold probabilities use fresh horizon-1 Ensemble CPI simulation draws for each threshold test quarter (n_sims=1000, seed=42 with one deterministic increment per fold).

This "confidence" measures how far a prediction sits from the model's own decision boundary, not a validated track record of being correct; a model can be confidently wrong.

| target_quarter   | model                 | predicted_action   |   P(cut) |   P(hold) |   P(hike) |   confidence |
|:-----------------|:----------------------|:-------------------|---------:|----------:|----------:|-------------:|
| 2014Q1           | threshold             | hold               |    0.027 |     0.632 |     0.341 |        0.632 |
| 2014Q1           | ordered_logit         | cut                |    0.998 |     0.002 |     0.000 |        0.998 |
| 2014Q1           | ordered_probit        | cut                |    1.000 |     0.000 |     0.000 |        1.000 |
| 2014Q1           | taylor_rule_estimated | cut                |    0.688 |     0.308 |     0.004 |        0.688 |
| 2014Q2           | threshold             | hike               |    0.005 |     0.468 |     0.527 |        0.527 |
| 2014Q2           | ordered_logit         | cut                |    0.589 |     0.362 |     0.049 |        0.589 |
| 2014Q2           | ordered_probit        | cut                |    0.607 |     0.347 |     0.046 |        0.607 |
| 2014Q2           | taylor_rule_estimated | hold               |    0.405 |     0.559 |     0.036 |        0.559 |
| 2014Q3           | threshold             | hold               |    0.020 |     0.619 |     0.361 |        0.619 |
| 2014Q3           | ordered_logit         | hold               |    0.464 |     0.489 |     0.047 |        0.489 |
| 2014Q3           | ordered_probit        | cut                |    0.541 |     0.421 |     0.038 |        0.541 |
| 2014Q3           | taylor_rule_estimated | hold               |    0.450 |     0.525 |     0.025 |        0.525 |
| 2014Q4           | threshold             | hold               |    0.127 |     0.732 |     0.141 |        0.732 |
| 2014Q4           | ordered_logit         | cut                |    0.983 |     0.016 |     0.000 |        0.983 |
| 2014Q4           | ordered_probit        | cut                |    0.985 |     0.015 |     0.000 |        0.985 |
| 2014Q4           | taylor_rule_estimated | cut                |    0.896 |     0.104 |     0.000 |        0.896 |
| 2015Q1           | threshold             | cut                |    0.623 |     0.367 |     0.010 |        0.623 |
| 2015Q1           | ordered_logit         | cut                |    0.896 |     0.099 |     0.005 |        0.896 |
| 2015Q1           | ordered_probit        | cut                |    0.895 |     0.104 |     0.001 |        0.895 |
| 2015Q1           | taylor_rule_estimated | cut                |    0.903 |     0.096 |     0.001 |        0.903 |
| 2015Q2           | threshold             | cut                |    0.701 |     0.298 |     0.001 |        0.701 |
| 2015Q2           | ordered_logit         | cut                |    0.857 |     0.136 |     0.006 |        0.857 |
| 2015Q2           | ordered_probit        | cut                |    0.847 |     0.151 |     0.002 |        0.847 |
| 2015Q2           | taylor_rule_estimated | cut                |    0.867 |     0.132 |     0.001 |        0.867 |
| 2015Q3           | threshold             | hold               |    0.314 |     0.632 |     0.054 |        0.632 |
| 2015Q3           | ordered_logit         | cut                |    0.565 |     0.406 |     0.029 |        0.565 |
| 2015Q3           | ordered_probit        | cut                |    0.572 |     0.403 |     0.025 |        0.572 |
| 2015Q3           | taylor_rule_estimated | cut                |    0.624 |     0.366 |     0.011 |        0.624 |
| 2015Q4           | threshold             | hold               |    0.458 |     0.529 |     0.013 |        0.529 |
| 2015Q4           | ordered_logit         | cut                |    0.898 |     0.096 |     0.005 |        0.898 |
| 2015Q4           | ordered_probit        | cut                |    0.874 |     0.124 |     0.002 |        0.874 |
| 2015Q4           | taylor_rule_estimated | cut                |    0.843 |     0.156 |     0.001 |        0.843 |
| 2016Q1           | threshold             | cut                |    0.549 |     0.440 |     0.011 |        0.549 |
| 2016Q1           | ordered_logit         | cut                |    0.699 |     0.281 |     0.019 |        0.699 |
| 2016Q1           | ordered_probit        | cut                |    0.652 |     0.332 |     0.016 |        0.652 |
| 2016Q1           | taylor_rule_estimated | cut                |    0.653 |     0.336 |     0.011 |        0.653 |
| 2016Q2           | threshold             | cut                |    0.873 |     0.127 |     0.000 |        0.873 |
| 2016Q2           | ordered_logit         | cut                |    0.662 |     0.322 |     0.016 |        0.662 |
| 2016Q2           | ordered_probit        | cut                |    0.638 |     0.352 |     0.010 |        0.638 |
| 2016Q2           | taylor_rule_estimated | cut                |    0.818 |     0.179 |     0.003 |        0.818 |
| 2016Q3           | threshold             | cut                |    0.776 |     0.223 |     0.001 |        0.776 |
| 2016Q3           | ordered_logit         | cut                |    0.676 |     0.309 |     0.015 |        0.676 |
| 2016Q3           | ordered_probit        | cut                |    0.654 |     0.336 |     0.010 |        0.654 |
| 2016Q3           | taylor_rule_estimated | cut                |    0.692 |     0.301 |     0.007 |        0.692 |
| 2016Q4           | threshold             | cut                |    0.530 |     0.463 |     0.007 |        0.530 |
| 2016Q4           | ordered_logit         | cut                |    0.550 |     0.424 |     0.027 |        0.550 |
| 2016Q4           | ordered_probit        | cut                |    0.497 |     0.473 |     0.030 |        0.497 |
| 2016Q4           | taylor_rule_estimated | cut                |    0.662 |     0.330 |     0.008 |        0.662 |
| 2017Q1           | threshold             | cut                |    0.533 |     0.463 |     0.004 |        0.533 |
| 2017Q1           | ordered_logit         | cut                |    0.773 |     0.216 |     0.011 |        0.773 |
| 2017Q1           | ordered_probit        | cut                |    0.717 |     0.274 |     0.009 |        0.717 |
| 2017Q1           | taylor_rule_estimated | cut                |    0.722 |     0.273 |     0.005 |        0.722 |
| 2017Q2           | threshold             | hold               |    0.329 |     0.619 |     0.052 |        0.619 |
| 2017Q2           | ordered_logit         | hold               |    0.337 |     0.604 |     0.059 |        0.604 |
| 2017Q2           | ordered_probit        | hold               |    0.309 |     0.611 |     0.079 |        0.611 |
| 2017Q2           | taylor_rule_estimated | cut                |    0.591 |     0.394 |     0.015 |        0.591 |
| 2017Q3           | threshold             | hold               |    0.346 |     0.619 |     0.035 |        0.619 |
| 2017Q3           | ordered_logit         | hold               |    0.284 |     0.656 |     0.060 |        0.656 |
| 2017Q3           | ordered_probit        | hold               |    0.289 |     0.642 |     0.069 |        0.642 |
| 2017Q3           | taylor_rule_estimated | hold               |    0.415 |     0.541 |     0.044 |        0.541 |
| 2017Q4           | threshold             | cut                |    0.514 |     0.478 |     0.008 |        0.514 |
| 2017Q4           | ordered_logit         | hold               |    0.351 |     0.613 |     0.036 |        0.613 |
| 2017Q4           | ordered_probit        | hold               |    0.349 |     0.614 |     0.037 |        0.614 |
| 2017Q4           | taylor_rule_estimated | cut                |    0.540 |     0.440 |     0.020 |        0.540 |
| 2018Q1           | threshold             | hold               |    0.418 |     0.562 |     0.020 |        0.562 |
| 2018Q1           | ordered_logit         | hold               |    0.275 |     0.683 |     0.042 |        0.683 |
| 2018Q1           | ordered_probit        | hold               |    0.274 |     0.680 |     0.046 |        0.680 |
| 2018Q1           | taylor_rule_estimated | cut                |    0.503 |     0.471 |     0.026 |        0.503 |
| 2018Q2           | threshold             | hold               |    0.196 |     0.715 |     0.089 |        0.715 |
| 2018Q2           | ordered_logit         | hold               |    0.170 |     0.766 |     0.064 |        0.766 |
| 2018Q2           | ordered_probit        | hold               |    0.164 |     0.757 |     0.079 |        0.757 |
| 2018Q2           | taylor_rule_estimated | hold               |    0.483 |     0.488 |     0.029 |        0.488 |
| 2018Q3           | threshold             | hold               |    0.167 |     0.729 |     0.104 |        0.729 |
| 2018Q3           | ordered_logit         | hold               |    0.069 |     0.794 |     0.137 |        0.794 |
| 2018Q3           | ordered_probit        | hold               |    0.057 |     0.761 |     0.182 |        0.761 |
| 2018Q3           | taylor_rule_estimated | hold               |    0.341 |     0.597 |     0.061 |        0.597 |
| 2018Q4           | threshold             | hold               |    0.335 |     0.631 |     0.034 |        0.631 |
| 2018Q4           | ordered_logit         | hold               |    0.080 |     0.820 |     0.100 |        0.820 |
| 2018Q4           | ordered_probit        | hold               |    0.073 |     0.798 |     0.129 |        0.798 |
| 2018Q4           | taylor_rule_estimated | hold               |    0.266 |     0.646 |     0.088 |        0.646 |
| 2019Q1           | threshold             | cut                |    0.617 |     0.379 |     0.004 |        0.617 |
| 2019Q1           | ordered_logit         | hold               |    0.444 |     0.546 |     0.010 |        0.546 |
| 2019Q1           | ordered_probit        | hold               |    0.449 |     0.545 |     0.005 |        0.545 |
| 2019Q1           | taylor_rule_estimated | hold               |    0.435 |     0.532 |     0.033 |        0.532 |
| 2019Q2           | threshold             | cut                |    0.839 |     0.161 |     0.000 |        0.839 |
| 2019Q2           | ordered_logit         | cut                |    0.660 |     0.336 |     0.004 |        0.660 |
| 2019Q2           | ordered_probit        | cut                |    0.651 |     0.348 |     0.001 |        0.651 |
| 2019Q2           | taylor_rule_estimated | cut                |    0.629 |     0.362 |     0.009 |        0.629 |
| 2019Q3           | threshold             | cut                |    0.642 |     0.355 |     0.003 |        0.642 |
| 2019Q3           | ordered_logit         | hold               |    0.425 |     0.566 |     0.009 |        0.566 |
| 2019Q3           | ordered_probit        | hold               |    0.412 |     0.583 |     0.006 |        0.583 |
| 2019Q3           | taylor_rule_estimated | cut                |    0.700 |     0.295 |     0.005 |        0.700 |
| 2019Q4           | threshold             | cut                |    0.536 |     0.449 |     0.015 |        0.536 |
| 2019Q4           | ordered_logit         | hold               |    0.252 |     0.729 |     0.020 |        0.729 |
| 2019Q4           | ordered_probit        | hold               |    0.249 |     0.731 |     0.020 |        0.731 |
| 2019Q4           | taylor_rule_estimated | cut                |    0.570 |     0.414 |     0.016 |        0.570 |
| 2020Q1           | threshold             | hold               |    0.389 |     0.585 |     0.026 |        0.585 |
| 2020Q1           | ordered_logit         | hold               |    0.185 |     0.781 |     0.035 |        0.781 |
| 2020Q1           | ordered_probit        | hold               |    0.188 |     0.771 |     0.041 |        0.771 |
| 2020Q1           | taylor_rule_estimated | hold               |    0.398 |     0.560 |     0.042 |        0.560 |
| 2020Q2           | threshold             | hold               |    0.351 |     0.635 |     0.014 |        0.635 |
| 2020Q2           | ordered_logit         | hold               |    0.423 |     0.562 |     0.015 |        0.562 |
| 2020Q2           | ordered_probit        | hold               |    0.423 |     0.566 |     0.012 |        0.566 |
| 2020Q2           | taylor_rule_estimated | cut                |    0.517 |     0.463 |     0.020 |        0.517 |
| 2020Q3           | threshold             | cut                |    1.000 |     0.000 |     0.000 |        1.000 |
| 2020Q3           | ordered_logit         | cut                |    1.000 |     0.000 |     0.000 |        1.000 |
| 2020Q3           | ordered_probit        | cut                |    1.000 |     0.000 |     0.000 |        1.000 |
| 2020Q3           | taylor_rule_estimated | cut                |    1.000 |     0.000 |     0.000 |        1.000 |
| 2020Q4           | threshold             | cut                |    1.000 |     0.000 |     0.000 |        1.000 |
| 2020Q4           | ordered_logit         | cut                |    0.798 |     0.197 |     0.005 |        0.798 |
| 2020Q4           | ordered_probit        | cut                |    0.806 |     0.192 |     0.002 |        0.806 |
| 2020Q4           | taylor_rule_estimated | cut                |    0.688 |     0.301 |     0.011 |        0.688 |
| 2021Q1           | threshold             | cut                |    0.937 |     0.063 |     0.000 |        0.937 |
| 2021Q1           | ordered_logit         | cut                |    0.763 |     0.230 |     0.007 |        0.763 |
| 2021Q1           | ordered_probit        | cut                |    0.763 |     0.235 |     0.002 |        0.763 |
| 2021Q1           | taylor_rule_estimated | cut                |    0.593 |     0.386 |     0.021 |        0.593 |
| 2021Q2           | threshold             | hold               |    0.050 |     0.722 |     0.228 |        0.722 |
| 2021Q2           | ordered_logit         | hold               |    0.280 |     0.668 |     0.052 |        0.668 |
| 2021Q2           | ordered_probit        | hold               |    0.265 |     0.665 |     0.070 |        0.665 |
| 2021Q2           | taylor_rule_estimated | hold               |    0.466 |     0.492 |     0.042 |        0.492 |
| 2021Q3           | threshold             | hold               |    0.125 |     0.730 |     0.145 |        0.730 |
| 2021Q3           | ordered_logit         | hold               |    0.348 |     0.618 |     0.034 |        0.618 |
| 2021Q3           | ordered_probit        | hold               |    0.347 |     0.615 |     0.038 |        0.615 |
| 2021Q3           | taylor_rule_estimated | hold               |    0.417 |     0.530 |     0.053 |        0.530 |
| 2021Q4           | threshold             | hold               |    0.020 |     0.525 |     0.455 |        0.525 |
| 2021Q4           | ordered_logit         | hold               |    0.267 |     0.689 |     0.045 |        0.689 |
| 2021Q4           | ordered_probit        | hold               |    0.269 |     0.678 |     0.053 |        0.678 |
| 2021Q4           | taylor_rule_estimated | hold               |    0.359 |     0.571 |     0.070 |        0.571 |
| 2022Q1           | threshold             | hike               |    0.001 |     0.197 |     0.802 |        0.802 |
| 2022Q1           | ordered_logit         | hold               |    0.239 |     0.714 |     0.046 |        0.714 |
| 2022Q1           | ordered_probit        | hold               |    0.240 |     0.703 |     0.057 |        0.703 |
| 2022Q1           | taylor_rule_estimated | hold               |    0.345 |     0.583 |     0.073 |        0.583 |
| 2022Q2           | threshold             | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2022Q2           | ordered_logit         | hold               |    0.366 |     0.611 |     0.023 |        0.611 |
| 2022Q2           | ordered_probit        | hold               |    0.375 |     0.604 |     0.021 |        0.604 |
| 2022Q2           | taylor_rule_estimated | hold               |    0.197 |     0.649 |     0.154 |        0.649 |
| 2022Q3           | threshold             | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2022Q3           | ordered_logit         | hike               |    0.021 |     0.463 |     0.516 |        0.516 |
| 2022Q3           | ordered_probit        | hike               |    0.014 |     0.468 |     0.518 |        0.518 |
| 2022Q3           | taylor_rule_estimated | hike               |    0.031 |     0.440 |     0.529 |        0.529 |
| 2022Q4           | threshold             | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2022Q4           | ordered_logit         | hike               |    0.008 |     0.238 |     0.755 |        0.755 |
| 2022Q4           | ordered_probit        | hike               |    0.004 |     0.289 |     0.707 |        0.707 |
| 2022Q4           | taylor_rule_estimated | hike               |    0.000 |     0.001 |     0.999 |        0.999 |
| 2023Q1           | threshold             | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2023Q1           | ordered_logit         | hike               |    0.011 |     0.321 |     0.668 |        0.668 |
| 2023Q1           | ordered_probit        | hike               |    0.006 |     0.341 |     0.653 |        0.653 |
| 2023Q1           | taylor_rule_estimated | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2023Q2           | threshold             | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2023Q2           | ordered_logit         | hike               |    0.008 |     0.250 |     0.742 |        0.742 |
| 2023Q2           | ordered_probit        | hike               |    0.003 |     0.248 |     0.750 |        0.750 |
| 2023Q2           | taylor_rule_estimated | hike               |    0.000 |     0.006 |     0.994 |        0.994 |
| 2023Q3           | threshold             | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2023Q3           | ordered_logit         | hike               |    0.010 |     0.306 |     0.683 |        0.683 |
| 2023Q3           | ordered_probit        | hike               |    0.005 |     0.309 |     0.687 |        0.687 |
| 2023Q3           | taylor_rule_estimated | hike               |    0.002 |     0.041 |     0.957 |        0.957 |
| 2023Q4           | threshold             | hike               |    0.000 |     0.001 |     0.999 |        0.999 |
| 2023Q4           | ordered_logit         | hike               |    0.018 |     0.423 |     0.559 |        0.559 |
| 2023Q4           | ordered_probit        | hike               |    0.011 |     0.413 |     0.576 |        0.576 |
| 2023Q4           | taylor_rule_estimated | hike               |    0.012 |     0.133 |     0.855 |        0.855 |
| 2024Q1           | threshold             | hike               |    0.003 |     0.150 |     0.847 |        0.847 |
| 2024Q1           | ordered_logit         | hold               |    0.104 |     0.720 |     0.176 |        0.720 |
| 2024Q1           | ordered_probit        | hold               |    0.101 |     0.689 |     0.210 |        0.689 |
| 2024Q1           | taylor_rule_estimated | hike               |    0.097 |     0.367 |     0.536 |        0.536 |

Threshold hard-label/simulated-median consistency:

No threshold hard-label/simulated-median disagreements were flagged.

## Majority Vote Ensemble Audit

The ensemble combines exactly four existing per-fold predictions: threshold, taylor_rule_estimated, ordered_logit, ordered_probit. It estimates no parameters and uses no separate train/test split.

Tie-break cases using the threshold prediction: 4 of 41 test quarters.

## Tested And Reverted Ordered Feature Variant

A single pre-committed 5-feature ordered-model variant added cash_rate_lag1 and commodity_growth_lag1 to the active 3-feature ordered set. It was reverted because it worsened both ordered models on the same walk-forward test window; the result is retained here as a tested-and-rejected variant, not erased.

Rejected feature set: headline_forecast, trimmed_mean_forecast, unemployment_rate_change_lag1, cash_rate_lag1, commodity_growth_lag1.

| model          |   active_3_feature_macro_f1 |   tested_5_feature_macro_f1 |   macro_f1_change |   tested_5_feature_accuracy |   threshold_minus_tested_gap | tested_paired_bootstrap_95pct_ci   | status                   |
|:---------------|----------------------------:|----------------------------:|------------------:|----------------------------:|-----------------------------:|:-----------------------------------|:-------------------------|
| ordered_logit  |                       0.715 |                       0.620 |            -0.095 |                       0.610 |                        0.155 | [0.000, 0.338]                     | tested_once_and_reverted |
| ordered_probit |                       0.696 |                       0.653 |            -0.043 |                       0.634 |                        0.122 | [-0.024, 0.281]                    | tested_once_and_reverted |

## Estimated Taylor Coefficient Audit

Final walk-forward fold coefficients:

| target_quarter   |   intercept |   coef_inflation_gap |   coef_unemployment_change_lag1 |   predicted_change | unstable_coefficients   |
|:-----------------|------------:|---------------------:|--------------------------------:|-------------------:|:------------------------|
| 2024Q1           |      -0.039 |                0.178 |                           0.013 |              0.144 |                         |

Unstable coefficient flags:

No estimated Taylor-rule fold had |coef_inflation_gap| or |coef_unemployment_change_lag1| above 10.0.

## Ordered Model Interpretability

Ordered logit and probit are refit at every walk-forward step, so there is no single coefficient vector for the whole exercise. The table below reports the final fold, which uses the most training data. Betas are on the train-standardized feature scale; alphas are the transformed latent cutoffs between cut/hold and hold/hike.

| model          | target_quarter   |   train_rows |   beta_headline_forecast |   beta_trimmed_mean_forecast |   beta_unemployment_rate_change_lag1 |   alpha_cut_hold |   alpha_hold_hike |
|:---------------|:-----------------|-------------:|-------------------------:|-----------------------------:|-------------------------------------:|-----------------:|------------------:|
| ordered_logit  | 2024Q1           |           52 |                    3.362 |                       -1.042 |                                0.260 |           -0.841 |             2.854 |
| ordered_probit | 2024Q1           |           52 |                    1.936 |                       -0.610 |                                0.215 |           -0.474 |             1.606 |

The first-vs-last alpha comparison checks whether the thin initial folds produce visibly different cutoffs from the final fold.

| model          | first_test_quarter   | last_test_quarter   |   first_alpha_cut_hold |   last_alpha_cut_hold |   change_alpha_cut_hold |   range_alpha_cut_hold |   first_alpha_hold_hike |   last_alpha_hold_hike |   change_alpha_hold_hike |   range_alpha_hold_hike |
|:---------------|:---------------------|:--------------------|-----------------------:|----------------------:|------------------------:|-----------------------:|------------------------:|-----------------------:|-------------------------:|------------------------:|
| ordered_logit  | 2014Q1               | 2024Q1              |                  1.962 |                -0.841 |                  -2.803 |                  2.803 |                   5.520 |                  2.854 |                   -2.666 |                   2.666 |
| ordered_probit | 2014Q1               | 2024Q1              |                  1.205 |                -0.474 |                  -1.679 |                  1.679 |                   3.161 |                  1.606 |                   -1.555 |                   1.555 |

## OrderedModel Convergence Warnings

No OrderedModel ConvergenceWarning was captured during the walk-forward run.

## Small-Sample Caveat

The walk-forward test set has 41 quarters. At this size, the macro-F1 ranking (threshold 0.775, majority_vote_ensemble 0.769, taylor_rule_estimated 0.769, ordered_logit 0.715, ordered_probit 0.696, taylor_rule 0.365, frank_hall_xgboost 0.274) is suggestive rather than decisive.

As a quick uncertainty check, a paired bootstrap over the same test quarters (2,000 resamples, seed=42) gives the following macro-F1 gap intervals for threshold minus each candidate. This is a rough diagnostic because it treats quarters as exchangeable and does not model time-series dependence.

| gap                                |   observed_macro_f1_gap | paired_bootstrap_95pct_ci   |
|:-----------------------------------|------------------------:|:----------------------------|
| threshold - taylor_rule            |                   0.410 | [0.283, 0.538]              |
| threshold - taylor_rule_estimated  |                   0.007 | [-0.152, 0.168]             |
| threshold - ordered_logit          |                   0.060 | [-0.091, 0.223]             |
| threshold - ordered_probit         |                   0.079 | [-0.079, 0.251]             |
| threshold - frank_hall_xgboost     |                   0.502 | [0.342, 0.640]              |
| threshold - majority_vote_ensemble |                   0.006 | [-0.129, 0.131]             |

## Confusion Matrices

### threshold

| actual   |   cut |   hold |   hike |
|:---------|------:|-------:|-------:|
| cut      |    10 |      3 |      0 |
| hold     |     5 |     13 |      2 |
| hike     |     0 |      0 |      8 |

### taylor_rule

| actual   |   cut |   hold |   hike |
|:---------|------:|-------:|-------:|
| cut      |    10 |      0 |      3 |
| hold     |    12 |      0 |      8 |
| hike     |     1 |      0 |      7 |

### taylor_rule_estimated

| actual   |   cut |   hold |   hike |
|:---------|------:|-------:|-------:|
| cut      |    12 |      1 |      0 |
| hold     |     9 |     11 |      0 |
| hike     |     0 |      1 |      7 |

### ordered_logit

| actual   |   cut |   hold |   hike |
|:---------|------:|-------:|-------:|
| cut      |     9 |      4 |      0 |
| hold     |     7 |     13 |      0 |
| hike     |     0 |      2 |      6 |

### ordered_probit

| actual   |   cut |   hold |   hike |
|:---------|------:|-------:|-------:|
| cut      |     9 |      4 |      0 |
| hold     |     8 |     12 |      0 |
| hike     |     0 |      2 |      6 |

### frank_hall_xgboost

| actual   |   cut |   hold |   hike |
|:---------|------:|-------:|-------:|
| cut      |     8 |      5 |      0 |
| hold     |    13 |      7 |      0 |
| hike     |     0 |      8 |      0 |

### majority_vote_ensemble

| actual   |   cut |   hold |   hike |
|:---------|------:|-------:|-------:|
| cut      |    11 |      2 |      0 |
| hold     |     8 |     12 |      0 |
| hike     |     0 |      1 |      7 |

## Interpretive Conclusion

The threshold baseline remains the preferred reportable result: it has the highest macro-F1 point estimate. Its lead over taylor_rule_estimated, ordered_logit, ordered_probit, and majority_vote_ensemble remains statistically unsettled because those paired-bootstrap intervals touch or cross zero. Its gaps over taylor_rule and frank_hall_xgboost are strictly positive across the paired-bootstrap 95% intervals. The majority-vote ensemble records macro-F1 0.769; threshold minus majority_vote_ensemble is 0.006 with paired-bootstrap 95% CI [-0.129, 0.131]. The estimated Taylor rule beats the fixed Taylor rule on macro-F1 (0.769 vs 0.365).

When the threshold rule is preferred, the non-statistical reason remains its transparency and hike detection: it correctly classifies all 8 of 8 hike quarters, while taylor_rule_estimated correctly classifies 7 of 8 and misses 1; ordered logit classifies 2 hikes as hold and ordered probit classifies 2 hikes as hold. That hike-to-hold error is a structurally different, and arguably more consequential, failure mode for a policy classifier than the aggregate macro-F1 gap alone captures.

A plausible reading remains that the CPI forecast features are highly correlated and the earliest valid training folds are thin. The expanded exercise should therefore remain a documented classifier comparison for the materiality discussion rather than an API-exposed policy predictor.
