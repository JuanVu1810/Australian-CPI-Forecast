-- Each quarter's year-ended CPI and cash rate, and how far each moved over the previous four quarters (a window
-- function over the quarter order). Stops at 2025Q4, the forecast-origin pin (svar.FORECAST_ORIGIN_PIN).

SELECT
    quarter,
    cpi_yoy,
    cpi_yoy - LAG(cpi_yoy, 4) OVER (ORDER BY quarter) AS cpi_yoy_change_1y,
    cash_rate,
    cash_rate - LAG(cash_rate, 4) OVER (ORDER BY quarter) AS cash_rate_change_1y
FROM quarterly_macro_features
WHERE quarter <= '2025Q4'
ORDER BY quarter;
