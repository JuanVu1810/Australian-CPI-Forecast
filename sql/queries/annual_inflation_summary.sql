-- DuckDB analytical query for annual CPI and macro summaries.

SELECT
    CAST(SUBSTR(quarter, 1, 4) AS INTEGER) AS year,
    AVG(cpi_yoy) AS avg_cpi_yoy,
    AVG(unemployment_rate) AS avg_unemployment_rate,
    AVG(cash_rate) AS avg_cash_rate,
    AVG(wpi_growth) AS avg_wpi_growth,
    AVG(ppi_growth) AS avg_ppi_growth
FROM quarterly_macro_features
GROUP BY year
ORDER BY year;
