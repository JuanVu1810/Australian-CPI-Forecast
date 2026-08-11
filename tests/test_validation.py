import pandas as pd

from src.validation import validate_time_series


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
