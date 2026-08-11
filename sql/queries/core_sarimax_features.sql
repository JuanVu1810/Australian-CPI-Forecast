-- Feature extract for the first long-sample SARIMAX experiment.

SELECT
    quarter,
    cpi_yoy,
    cpi_yoy_lag1,
    cpi_yoy_lag4,
    unemployment_rate_lag1,
    cash_rate_lag1,
    cash_rate_lag2,
    wpi_growth_lag1,
    ppi_growth_lag1,
    commodity_growth_lag1,
    inflation_expectations_business_lag1
FROM quarterly_macro_features
WHERE
    cpi_yoy IS NOT NULL
    AND cpi_yoy_lag1 IS NOT NULL
    AND cpi_yoy_lag4 IS NOT NULL
    AND unemployment_rate_lag1 IS NOT NULL
    AND cash_rate_lag1 IS NOT NULL
    AND cash_rate_lag2 IS NOT NULL
    AND wpi_growth_lag1 IS NOT NULL
    AND ppi_growth_lag1 IS NOT NULL
    AND commodity_growth_lag1 IS NOT NULL
    AND inflation_expectations_business_lag1 IS NOT NULL
ORDER BY quarter;
