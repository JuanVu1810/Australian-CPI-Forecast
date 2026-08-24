import pandas as pd

from src.transform import merge_quarterly_frames, to_quarterly


def test_monthly_series_converts_to_quarterly_mean():
    source = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
            "value": [1.0, 2.0, 3.0, 10.0],
        }
    )

    result = to_quarterly(source, value_col="value", output_col="monthly_mean")

    assert result.to_dict("records") == [
        {"quarter": "2020Q1", "monthly_mean": 2.0},
        {"quarter": "2020Q2", "monthly_mean": 10.0},
    ]


def test_merge_quarterly_frames_preserves_chronological_order():
    left = pd.DataFrame({"quarter": ["2020Q2", "2020Q1"], "a": [2, 1]})
    right = pd.DataFrame({"quarter": ["2020Q1", "2020Q2"], "b": [10, 20]})

    result = merge_quarterly_frames([left, right])

    assert result["quarter"].tolist() == ["2020Q1", "2020Q2"]
    assert result["a"].tolist() == [1, 2]
    assert result["b"].tolist() == [10, 20]


def test_merge_quarterly_frames_preserves_leading_missing_values_from_outer_join():
    long_series = pd.DataFrame(
        {"quarter": ["2020Q1", "2020Q2", "2020Q3"], "long_value": [100.0, 101.0, 102.0]}
    )
    later_series = pd.DataFrame({"quarter": ["2020Q2", "2020Q3"], "later_value": [99.0, 100.0]})

    result = merge_quarterly_frames([long_series, later_series])

    assert result["quarter"].tolist() == ["2020Q1", "2020Q2", "2020Q3"]
    assert pd.isna(result.loc[0, "later_value"])
    assert result["later_value"].iloc[1:].tolist() == [99.0, 100.0]
