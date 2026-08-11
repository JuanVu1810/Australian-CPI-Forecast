-- Application metadata tables for a future Supabase PostgreSQL deployment.

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    source_start_year INTEGER,
    source_end_year INTEGER,
    rows_curated INTEGER,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS data_quality_results (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT REFERENCES pipeline_runs(run_id),
    dataset TEXT NOT NULL,
    status TEXT NOT NULL,
    rows INTEGER,
    columns INTEGER,
    start_date TEXT,
    end_date TEXT,
    missing_values INTEGER,
    duplicate_dates INTEGER,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS model_metrics (
    run_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    run_date TIMESTAMPTZ NOT NULL,
    forecast_horizon INTEGER NOT NULL,
    feature_set TEXT,
    rmse DOUBLE PRECISION,
    mae DOUBLE PRECISION,
    mse DOUBLE PRECISION,
    selected_model BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS forecast_results (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT REFERENCES model_metrics(run_id),
    quarter TEXT NOT NULL,
    forecast DOUBLE PRECISION NOT NULL,
    lower_ci DOUBLE PRECISION,
    upper_ci DOUBLE PRECISION
);
