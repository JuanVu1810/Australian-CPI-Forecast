-- One table for the optional Supabase/PostgreSQL example (see README appendix).
-- The key stops a rerun from saving the same forecast twice.
CREATE TABLE IF NOT EXISTS forecast_snapshots (
    target TEXT NOT NULL,
    model TEXT NOT NULL,
    forecast_origin TEXT NOT NULL,
    target_quarter TEXT NOT NULL,
    point_forecast DOUBLE PRECISION NOT NULL,
    actual DOUBLE PRECISION,
    PRIMARY KEY (target, model, target_quarter)
);
