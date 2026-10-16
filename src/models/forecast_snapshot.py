"""Optional Supabase/PostgreSQL example: save each model's next-quarter forecast and compare it with the observed CPI.

Run from the project root:

    python -m src.models.forecast_snapshot

This is a retrospective wiring check. The quarter it saves (2026Q1) was already published,
so the table shows that the SQL works, not a live forecast record. Nothing else in the
project (ETL, models, API, Streamlit) reads this table.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from api import main as api_main
from src.models.elastic_net import TRIMMED_MEAN_TARGET_COLUMN
from src.models.evaluation import CURATED_DATA_PATH, PROJECT_ROOT, TARGET_COLUMN, load_target_series


SCHEMA_PATH = PROJECT_ROOT / "sql/postgres/forecast_snapshots.sql"
ERRORS_QUERY_PATH = PROJECT_ROOT / "sql/postgres/forecast_errors.sql"

# A rerun keeps the saved forecast and only fills in an actual that was missing before.
# This syntax works in PostgreSQL and in SQLite, which the tests use.
INSERT_SQL = """
INSERT INTO forecast_snapshots
    (target, model, forecast_origin, target_quarter, point_forecast, actual)
VALUES (:target, :model, :forecast_origin, :target_quarter, :point_forecast, :actual)
ON CONFLICT (target, model, target_quarter)
DO UPDATE SET actual = COALESCE(forecast_snapshots.actual, excluded.actual)
"""


def collect_snapshot_rows() -> pd.DataFrame:
    """Serve the one-quarter-ahead forecast of every model and target, with the observed CPI if published."""
    request = api_main.AllForecastsRequest(horizon=1)
    responses = [
        ("headline", TARGET_COLUMN, api_main.forecast_all(request)),
        ("trimmed_mean", TRIMMED_MEAN_TARGET_COLUMN, api_main.forecast_trimmed_mean_all(request)),
    ]
    rows = []
    for target, target_column, response in responses:
        observed = load_target_series(CURATED_DATA_PATH, target_column=target_column)
        for model in response.models:
            target_quarter = model.quarters[0]
            actual = observed.get(pd.Period(target_quarter, freq="Q"))
            rows.append(
                {
                    "target": target,
                    "model": model.model_family,
                    "forecast_origin": model.forecast_origin,
                    "target_quarter": target_quarter,
                    "point_forecast": model.forecast[0],
                    "actual": None if actual is None else float(actual),
                }
            )
    return pd.DataFrame(rows)


def save_snapshots(engine, rows: pd.DataFrame) -> None:
    """Create the table if needed and save the rows; saving the same row twice changes nothing."""
    with engine.begin() as conn:
        conn.execute(text(SCHEMA_PATH.read_text()))
        conn.execute(text(INSERT_SQL), rows.to_dict("records"))


def read_forecast_errors(engine) -> pd.DataFrame:
    """Run the saved comparison query."""
    return pd.read_sql(text(ERRORS_QUERY_PATH.read_text()), engine)


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is not set, so nothing was saved. See the optional README appendix.")
        return

    engine = create_engine(database_url)
    rows = collect_snapshot_rows()
    save_snapshots(engine, rows)
    print(f"Saved {len(rows)} forecast snapshot(s). Forecast errors:")
    print(read_forecast_errors(engine).to_string(index=False))


if __name__ == "__main__":
    main()
