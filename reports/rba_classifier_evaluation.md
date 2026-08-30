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

## Prediction Confidence

Confidence is the probability assigned to the predicted action by that model's own class-probability calculation.

Threshold probabilities use fresh horizon-1 Ensemble CPI simulation draws for each threshold test quarter (n_sims=1000, seed=42 with one deterministic increment per fold).

This "confidence" measures how far a prediction sits from the model's own decision boundary, not a validated track record of being correct; a model can be confidently wrong.

| target_quarter   | model                 | predicted_action   |   P(cut) |   P(hold) |   P(hike) |   confidence |
|:-----------------|:----------------------|:-------------------|---------:|----------:|----------:|-------------:|
| 2014Q1           | threshold             | hold               |    0.027 |     0.627 |     0.346 |        0.627 |
| 2014Q1           | ordered_logit         | cut                |    0.990 |     0.009 |     0.001 |        0.990 |
| 2014Q1           | ordered_probit        | cut                |    0.998 |     0.002 |     0.000 |        0.998 |
| 2014Q1           | taylor_rule_estimated | cut                |    0.688 |     0.308 |     0.004 |        0.688 |
| 2014Q2           | threshold             | hike               |    0.005 |     0.444 |     0.551 |        0.551 |
| 2014Q2           | ordered_logit         | cut                |    0.555 |     0.388 |     0.057 |        0.555 |
| 2014Q2           | ordered_probit        | cut                |    0.584 |     0.360 |     0.057 |        0.584 |
| 2014Q2           | taylor_rule_estimated | hold               |    0.405 |     0.559 |     0.036 |        0.559 |
| 2014Q3           | threshold             | hold               |    0.025 |     0.636 |     0.339 |        0.636 |
| 2014Q3           | ordered_logit         | hold               |    0.395 |     0.542 |     0.063 |        0.542 |
| 2014Q3           | ordered_probit        | cut                |    0.471 |     0.468 |     0.061 |        0.471 |
| 2014Q3           | taylor_rule_estimated | hold               |    0.450 |     0.525 |     0.025 |        0.525 |
| 2014Q4           | threshold             | hold               |    0.126 |     0.731 |     0.143 |        0.731 |
| 2014Q4           | ordered_logit         | cut                |    0.985 |     0.014 |     0.000 |        0.985 |
| 2014Q4           | ordered_probit        | cut                |    0.989 |     0.011 |     0.000 |        0.989 |
| 2014Q4           | taylor_rule_estimated | cut                |    0.896 |     0.104 |     0.000 |        0.896 |
| 2015Q1           | threshold             | cut                |    0.627 |     0.363 |     0.010 |        0.627 |
| 2015Q1           | ordered_logit         | cut                |    0.902 |     0.093 |     0.005 |        0.902 |
| 2015Q1           | ordered_probit        | cut                |    0.887 |     0.111 |     0.002 |        0.887 |
| 2015Q1           | taylor_rule_estimated | cut                |    0.903 |     0.096 |     0.001 |        0.903 |
| 2015Q2           | threshold             | cut                |    0.684 |     0.315 |     0.001 |        0.684 |
| 2015Q2           | ordered_logit         | cut                |    0.873 |     0.121 |     0.006 |        0.873 |
| 2015Q2           | ordered_probit        | cut                |    0.842 |     0.155 |     0.003 |        0.842 |
| 2015Q2           | taylor_rule_estimated | cut                |    0.867 |     0.132 |     0.001 |        0.867 |
| 2015Q3           | threshold             | hold               |    0.284 |     0.659 |     0.057 |        0.659 |
| 2015Q3           | ordered_logit         | cut                |    0.571 |     0.398 |     0.031 |        0.571 |
| 2015Q3           | ordered_probit        | cut                |    0.545 |     0.419 |     0.035 |        0.545 |
| 2015Q3           | taylor_rule_estimated | cut                |    0.624 |     0.366 |     0.011 |        0.624 |
| 2015Q4           | threshold             | hold               |    0.487 |     0.503 |     0.010 |        0.503 |
| 2015Q4           | ordered_logit         | cut                |    0.930 |     0.067 |     0.004 |        0.930 |
| 2015Q4           | ordered_probit        | cut                |    0.914 |     0.085 |     0.002 |        0.914 |
| 2015Q4           | taylor_rule_estimated | cut                |    0.843 |     0.156 |     0.001 |        0.843 |
| 2016Q1           | threshold             | cut                |    0.588 |     0.404 |     0.008 |        0.588 |
| 2016Q1           | ordered_logit         | cut                |    0.793 |     0.194 |     0.013 |        0.793 |
| 2016Q1           | ordered_probit        | cut                |    0.731 |     0.256 |     0.012 |        0.731 |
| 2016Q1           | taylor_rule_estimated | cut                |    0.653 |     0.336 |     0.011 |        0.653 |
| 2016Q2           | threshold             | cut                |    0.892 |     0.108 |     0.000 |        0.892 |
| 2016Q2           | ordered_logit         | cut                |    0.796 |     0.195 |     0.010 |        0.796 |
| 2016Q2           | ordered_probit        | cut                |    0.755 |     0.239 |     0.006 |        0.755 |
| 2016Q2           | taylor_rule_estimated | cut                |    0.818 |     0.179 |     0.003 |        0.818 |
| 2016Q3           | threshold             | cut                |    0.776 |     0.223 |     0.001 |        0.776 |
| 2016Q3           | ordered_logit         | cut                |    0.735 |     0.251 |     0.013 |        0.735 |
| 2016Q3           | ordered_probit        | cut                |    0.690 |     0.299 |     0.011 |        0.690 |
| 2016Q3           | taylor_rule_estimated | cut                |    0.692 |     0.301 |     0.007 |        0.692 |
| 2016Q4           | threshold             | cut                |    0.535 |     0.458 |     0.007 |        0.535 |
| 2016Q4           | ordered_logit         | cut                |    0.740 |     0.247 |     0.013 |        0.740 |
| 2016Q4           | ordered_probit        | cut                |    0.696 |     0.293 |     0.011 |        0.696 |
| 2016Q4           | taylor_rule_estimated | cut                |    0.662 |     0.330 |     0.008 |        0.662 |
| 2017Q1           | threshold             | cut                |    0.487 |     0.506 |     0.007 |        0.487 |
| 2017Q1           | ordered_logit         | cut                |    0.853 |     0.140 |     0.007 |        0.853 |
| 2017Q1           | ordered_probit        | cut                |    0.810 |     0.186 |     0.004 |        0.810 |
| 2017Q1           | taylor_rule_estimated | cut                |    0.722 |     0.273 |     0.005 |        0.722 |
| 2017Q2           | threshold             | hold               |    0.342 |     0.609 |     0.049 |        0.609 |
| 2017Q2           | ordered_logit         | hold               |    0.381 |     0.565 |     0.054 |        0.565 |
| 2017Q2           | ordered_probit        | hold               |    0.354 |     0.572 |     0.074 |        0.572 |
| 2017Q2           | taylor_rule_estimated | cut                |    0.591 |     0.394 |     0.015 |        0.591 |
| 2017Q3           | threshold             | hold               |    0.338 |     0.627 |     0.035 |        0.627 |
| 2017Q3           | ordered_logit         | hold               |    0.328 |     0.617 |     0.054 |        0.617 |
| 2017Q3           | ordered_probit        | hold               |    0.335 |     0.600 |     0.064 |        0.600 |
| 2017Q3           | taylor_rule_estimated | hold               |    0.415 |     0.541 |     0.044 |        0.541 |
| 2017Q4           | threshold             | cut                |    0.437 |     0.551 |     0.012 |        0.437 |
| 2017Q4           | ordered_logit         | hold               |    0.378 |     0.587 |     0.035 |        0.587 |
| 2017Q4           | ordered_probit        | hold               |    0.381 |     0.580 |     0.039 |        0.580 |
| 2017Q4           | taylor_rule_estimated | cut                |    0.540 |     0.440 |     0.020 |        0.540 |
| 2018Q1           | threshold             | hold               |    0.380 |     0.593 |     0.027 |        0.593 |
| 2018Q1           | ordered_logit         | hold               |    0.316 |     0.645 |     0.039 |        0.645 |
| 2018Q1           | ordered_probit        | hold               |    0.322 |     0.634 |     0.044 |        0.634 |
| 2018Q1           | taylor_rule_estimated | cut                |    0.503 |     0.471 |     0.026 |        0.503 |
| 2018Q2           | threshold             | hold               |    0.188 |     0.715 |     0.097 |        0.715 |
| 2018Q2           | ordered_logit         | hold               |    0.223 |     0.725 |     0.052 |        0.725 |
| 2018Q2           | ordered_probit        | hold               |    0.226 |     0.709 |     0.065 |        0.709 |
| 2018Q2           | taylor_rule_estimated | hold               |    0.483 |     0.488 |     0.029 |        0.488 |
| 2018Q3           | threshold             | hold               |    0.152 |     0.729 |     0.119 |        0.729 |
| 2018Q3           | ordered_logit         | hold               |    0.103 |     0.792 |     0.104 |        0.792 |
| 2018Q3           | ordered_probit        | hold               |    0.099 |     0.758 |     0.143 |        0.758 |
| 2018Q3           | taylor_rule_estimated | hold               |    0.341 |     0.597 |     0.061 |        0.597 |
| 2018Q4           | threshold             | hold               |    0.297 |     0.657 |     0.046 |        0.657 |
| 2018Q4           | ordered_logit         | hold               |    0.103 |     0.808 |     0.088 |        0.808 |
| 2018Q4           | ordered_probit        | hold               |    0.101 |     0.779 |     0.120 |        0.779 |
| 2018Q4           | taylor_rule_estimated | hold               |    0.266 |     0.646 |     0.088 |        0.646 |
| 2019Q1           | threshold             | cut                |    0.584 |     0.412 |     0.004 |        0.584 |
| 2019Q1           | ordered_logit         | hold               |    0.429 |     0.559 |     0.012 |        0.559 |
| 2019Q1           | ordered_probit        | hold               |    0.433 |     0.558 |     0.009 |        0.558 |
| 2019Q1           | taylor_rule_estimated | hold               |    0.435 |     0.532 |     0.033 |        0.532 |
| 2019Q2           | threshold             | cut                |    0.783 |     0.217 |     0.000 |        0.783 |
| 2019Q2           | ordered_logit         | cut                |    0.646 |     0.349 |     0.005 |        0.646 |
| 2019Q2           | ordered_probit        | cut                |    0.636 |     0.363 |     0.002 |        0.636 |
| 2019Q2           | taylor_rule_estimated | cut                |    0.629 |     0.362 |     0.009 |        0.629 |
| 2019Q3           | threshold             | cut                |    0.597 |     0.400 |     0.003 |        0.597 |
| 2019Q3           | ordered_logit         | hold               |    0.452 |     0.538 |     0.010 |        0.538 |
| 2019Q3           | ordered_probit        | hold               |    0.446 |     0.547 |     0.007 |        0.547 |
| 2019Q3           | taylor_rule_estimated | cut                |    0.700 |     0.295 |     0.005 |        0.700 |
| 2019Q4           | threshold             | cut                |    0.495 |     0.486 |     0.019 |        0.495 |
| 2019Q4           | ordered_logit         | hold               |    0.299 |     0.682 |     0.018 |        0.682 |
| 2019Q4           | ordered_probit        | hold               |    0.305 |     0.677 |     0.019 |        0.677 |
| 2019Q4           | taylor_rule_estimated | cut                |    0.570 |     0.414 |     0.016 |        0.570 |
| 2020Q1           | threshold             | hold               |    0.336 |     0.632 |     0.032 |        0.632 |
| 2020Q1           | ordered_logit         | hold               |    0.190 |     0.773 |     0.037 |        0.773 |
| 2020Q1           | ordered_probit        | hold               |    0.199 |     0.753 |     0.047 |        0.753 |
| 2020Q1           | taylor_rule_estimated | hold               |    0.398 |     0.560 |     0.042 |        0.560 |
| 2020Q2           | threshold             | hold               |    0.327 |     0.655 |     0.018 |        0.655 |
| 2020Q2           | ordered_logit         | hold               |    0.394 |     0.588 |     0.018 |        0.588 |
| 2020Q2           | ordered_probit        | hold               |    0.400 |     0.584 |     0.017 |        0.584 |
| 2020Q2           | taylor_rule_estimated | cut                |    0.517 |     0.463 |     0.020 |        0.517 |
| 2020Q3           | threshold             | cut                |    1.000 |     0.000 |     0.000 |        1.000 |
| 2020Q3           | ordered_logit         | cut                |    1.000 |     0.000 |     0.000 |        1.000 |
| 2020Q3           | ordered_probit        | cut                |    1.000 |     0.000 |     0.000 |        1.000 |
| 2020Q3           | taylor_rule_estimated | cut                |    1.000 |     0.000 |     0.000 |        1.000 |
| 2020Q4           | threshold             | cut                |    1.000 |     0.000 |     0.000 |        1.000 |
| 2020Q4           | ordered_logit         | cut                |    0.761 |     0.232 |     0.007 |        0.761 |
| 2020Q4           | ordered_probit        | cut                |    0.771 |     0.227 |     0.003 |        0.771 |
| 2020Q4           | taylor_rule_estimated | cut                |    0.688 |     0.301 |     0.011 |        0.688 |
| 2021Q1           | threshold             | cut                |    0.931 |     0.069 |     0.000 |        0.931 |
| 2021Q1           | ordered_logit         | cut                |    0.778 |     0.216 |     0.006 |        0.778 |
| 2021Q1           | ordered_probit        | cut                |    0.779 |     0.218 |     0.002 |        0.779 |
| 2021Q1           | taylor_rule_estimated | cut                |    0.593 |     0.386 |     0.021 |        0.593 |
| 2021Q2           | threshold             | hold               |    0.042 |     0.707 |     0.251 |        0.707 |
| 2021Q2           | ordered_logit         | hold               |    0.317 |     0.639 |     0.045 |        0.639 |
| 2021Q2           | ordered_probit        | hold               |    0.313 |     0.630 |     0.057 |        0.630 |
| 2021Q2           | taylor_rule_estimated | hold               |    0.466 |     0.492 |     0.042 |        0.492 |
| 2021Q3           | threshold             | hold               |    0.116 |     0.728 |     0.156 |        0.728 |
| 2021Q3           | ordered_logit         | hold               |    0.331 |     0.632 |     0.037 |        0.632 |
| 2021Q3           | ordered_probit        | hold               |    0.337 |     0.620 |     0.044 |        0.620 |
| 2021Q3           | taylor_rule_estimated | hold               |    0.417 |     0.530 |     0.053 |        0.530 |
| 2021Q4           | threshold             | hold               |    0.016 |     0.518 |     0.466 |        0.518 |
| 2021Q4           | ordered_logit         | hold               |    0.249 |     0.701 |     0.049 |        0.701 |
| 2021Q4           | ordered_probit        | hold               |    0.254 |     0.683 |     0.063 |        0.683 |
| 2021Q4           | taylor_rule_estimated | hold               |    0.359 |     0.571 |     0.070 |        0.571 |
| 2022Q1           | threshold             | hike               |    0.001 |     0.180 |     0.819 |        0.819 |
| 2022Q1           | ordered_logit         | hold               |    0.188 |     0.750 |     0.063 |        0.750 |
| 2022Q1           | ordered_probit        | hold               |    0.185 |     0.727 |     0.088 |        0.727 |
| 2022Q1           | taylor_rule_estimated | hold               |    0.345 |     0.583 |     0.073 |        0.583 |
| 2022Q2           | threshold             | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2022Q2           | ordered_logit         | hold               |    0.235 |     0.722 |     0.043 |        0.722 |
| 2022Q2           | ordered_probit        | hold               |    0.231 |     0.711 |     0.058 |        0.711 |
| 2022Q2           | taylor_rule_estimated | hold               |    0.197 |     0.649 |     0.154 |        0.649 |
| 2022Q3           | threshold             | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2022Q3           | ordered_logit         | hike               |    0.021 |     0.464 |     0.515 |        0.515 |
| 2022Q3           | ordered_probit        | hike               |    0.013 |     0.458 |     0.529 |        0.529 |
| 2022Q3           | taylor_rule_estimated | hike               |    0.031 |     0.440 |     0.529 |        0.529 |
| 2022Q4           | threshold             | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2022Q4           | ordered_logit         | hike               |    0.011 |     0.310 |     0.680 |        0.680 |
| 2022Q4           | ordered_probit        | hike               |    0.006 |     0.353 |     0.641 |        0.641 |
| 2022Q4           | taylor_rule_estimated | hike               |    0.000 |     0.001 |     0.999 |        0.999 |
| 2023Q1           | threshold             | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2023Q1           | ordered_logit         | hike               |    0.016 |     0.407 |     0.577 |        0.577 |
| 2023Q1           | ordered_probit        | hike               |    0.010 |     0.418 |     0.572 |        0.572 |
| 2023Q1           | taylor_rule_estimated | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2023Q2           | threshold             | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2023Q2           | ordered_logit         | hike               |    0.012 |     0.335 |     0.654 |        0.654 |
| 2023Q2           | ordered_probit        | hike               |    0.005 |     0.328 |     0.667 |        0.667 |
| 2023Q2           | taylor_rule_estimated | hike               |    0.000 |     0.006 |     0.994 |        0.994 |
| 2023Q3           | threshold             | hike               |    0.000 |     0.000 |     1.000 |        1.000 |
| 2023Q3           | ordered_logit         | hike               |    0.013 |     0.354 |     0.633 |        0.633 |
| 2023Q3           | ordered_probit        | hike               |    0.006 |     0.352 |     0.641 |        0.641 |
| 2023Q3           | taylor_rule_estimated | hike               |    0.002 |     0.041 |     0.957 |        0.957 |
| 2023Q4           | threshold             | hike               |    0.000 |     0.001 |     0.999 |        0.999 |
| 2023Q4           | ordered_logit         | hike               |    0.018 |     0.426 |     0.556 |        0.556 |
| 2023Q4           | ordered_probit        | hike               |    0.011 |     0.414 |     0.575 |        0.575 |
| 2023Q4           | taylor_rule_estimated | hike               |    0.012 |     0.133 |     0.855 |        0.855 |
| 2024Q1           | threshold             | hike               |    0.002 |     0.114 |     0.884 |        0.884 |
| 2024Q1           | ordered_logit         | hold               |    0.095 |     0.714 |     0.191 |        0.714 |
| 2024Q1           | ordered_probit        | hold               |    0.092 |     0.681 |     0.228 |        0.681 |
| 2024Q1           | taylor_rule_estimated | hike               |    0.097 |     0.367 |     0.536 |        0.536 |

Threshold hard-label/simulated-median consistency:

| target_quarter   | forecast_origin   |   point_forecast | point_forecast_action   |   simulated_median | simulated_median_action   |   P(cut) |   P(hold) |   P(hike) | disagreement                                          |
|:-----------------|:------------------|-----------------:|:------------------------|-------------------:|:--------------------------|---------:|----------:|----------:|:------------------------------------------------------|
| 2017Q1           | 2016Q4            |            1.964 | cut                     |              2.019 | hold                      |    0.487 |     0.506 |     0.007 | cut from point forecast vs hold from simulated median |
| 2017Q4           | 2017Q3            |            1.990 | cut                     |              2.064 | hold                      |    0.437 |     0.551 |     0.012 | cut from point forecast vs hold from simulated median |
| 2019Q4           | 2019Q3            |            1.963 | cut                     |              2.004 | hold                      |    0.495 |     0.486 |     0.019 | cut from point forecast vs hold from simulated median |

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
| ordered_logit  | 2024Q1           |           52 |                    3.200 |                       -0.879 |                                0.255 |           -0.848 |             2.848 |
| ordered_probit | 2024Q1           |           52 |                    1.842 |                       -0.514 |                                0.210 |           -0.478 |             1.598 |

The first-vs-last alpha comparison checks whether the thin initial folds produce visibly different cutoffs from the final fold.

| model          | first_test_quarter   | last_test_quarter   |   first_alpha_cut_hold |   last_alpha_cut_hold |   change_alpha_cut_hold |   range_alpha_cut_hold |   first_alpha_hold_hike |   last_alpha_hold_hike |   change_alpha_hold_hike |   range_alpha_hold_hike |
|:---------------|:---------------------|:--------------------|-----------------------:|----------------------:|------------------------:|-----------------------:|------------------------:|-----------------------:|-------------------------:|------------------------:|
| ordered_logit  | 2014Q1               | 2024Q1              |                  1.958 |                -0.848 |                  -2.806 |                  2.806 |                   4.612 |                  2.848 |                   -1.765 |                   1.860 |
| ordered_probit | 2014Q1               | 2024Q1              |                  1.207 |                -0.478 |                  -1.686 |                  1.686 |                   2.684 |                  1.598 |                   -1.085 |                   1.085 |

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
