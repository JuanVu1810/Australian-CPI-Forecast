import pandas as pd

from src.validation import validate_curated_dataset, validate_time_series


def test_validate_time_series_flags_duplicate_dates():
    source = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01"],
            "value": [1.0, 2.0],
        }
    )

    result = validate_time_series(
        source,
        dataset="example",
        date_col="date",
        value_cols=["value"],
    )

    assert result.status == "FAIL"
    assert result.duplicate_dates == 1
    assert "duplicate dates" in result.notes


def test_validate_curated_dataset_does_not_require_cpi_index():
    source = pd.DataFrame(
        {
            "quarter": ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1"],
            "cpi_qoq": [None, 1.0, 0.99, 0.98, 0.97],
            "cpi_yoy": [None, None, None, None, 4.0],
            "trimmed_mean_cpi_yoy": [None, 2.1, 2.2, 2.3, 2.4],
        }
    )

    records = validate_curated_dataset(source, min_rows=1)
    curated_record = next(
        record for record in records if record.dataset == "curated_quarterly_macro_features"
    )
    trimmed_record = next(
        record for record in records if record.dataset == "curated_column:trimmed_mean_cpi_yoy"
    )

    assert curated_record.status == "PASS"
    assert trimmed_record.status == "PASS"
    assert trimmed_record.missing_values == 1
