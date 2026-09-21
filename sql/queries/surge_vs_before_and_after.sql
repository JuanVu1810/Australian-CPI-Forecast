-- Average inflation, cash rate, unemployment and business inflation expectations before, during and after the
-- 2020-23 surge (the one surge window the book uses). Stops at 2025Q4, the forecast-origin pin
-- (svar.FORECAST_ORIGIN_PIN); later quarters are held out as benchmark only.

SELECT
    CASE
        WHEN quarter <= '2019Q4' THEN 'before'
        WHEN quarter <= '2023Q4' THEN 'surge'
        ELSE 'after'
    END AS period,
    MIN(quarter) AS first_quarter,
    MAX(quarter) AS last_quarter,
    COUNT(*) AS quarters,
    AVG(cpi_yoy) AS avg_cpi_yoy,
    AVG(cash_rate) AS avg_cash_rate,
    AVG(unemployment_rate) AS avg_unemployment_rate,
    AVG(inflation_expectations_business) AS avg_business_inflation_expectations
FROM quarterly_macro_features
WHERE quarter <= '2025Q4'
GROUP BY period
ORDER BY first_quarter;
