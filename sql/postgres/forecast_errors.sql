-- Compare each saved forecast with the observed CPI, where the quarter has been published.
SELECT
    target,
    model,
    target_quarter,
    point_forecast,
    actual,
    actual - point_forecast AS error,
    ABS(actual - point_forecast) AS abs_error
FROM forecast_snapshots
WHERE actual IS NOT NULL
ORDER BY target, target_quarter, model;
