import numpy as np
import pandas as pd

from src.models import simulation_fan


def _write_synthetic_curated(path):
    quarters = pd.period_range("2020Q1", periods=12, freq="Q")
    pd.DataFrame(
        {
            "quarter": quarters.astype(str),
            "cpi_yoy": np.linspace(2.0, 3.1, len(quarters)),
        }
    ).to_csv(path, index=False)


def test_run_sarima_fan_writes_ordered_percentile_bands(monkeypatch, tmp_path):
    curated_path = tmp_path / "curated.csv"
    output_path = tmp_path / "simulation_fan_sarima.csv"
    _write_synthetic_curated(curated_path)

    def fake_simulate_sarima_paths(series, steps, n_sims, order, seasonal_order, seed):
        assert str(series.index[-1]) == "2022Q4"
        assert steps == 8
        base = np.arange(1, steps + 1, dtype=float)
        offsets = np.linspace(-2.0, 2.0, n_sims).reshape(n_sims, 1)
        return base + offsets

    monkeypatch.setattr(
        simulation_fan,
        "simulate_sarima_paths",
        fake_simulate_sarima_paths,
    )

    frame = simulation_fan.run_sarima_fan(
        curated_path=curated_path,
        output_path=output_path,
        n_sims=9,
        seed=123,
        verbose=False,
    )
    written = pd.read_csv(output_path)

    expected_columns = set(simulation_fan.FAN_OUTPUT_COLUMNS)
    assert expected_columns.issubset(frame.columns)
    assert len(frame) == 8
    assert written.shape == frame.shape
    assert frame["horizon"].tolist() == list(range(1, 9))
    assert frame["forecast_origin"].unique().tolist() == ["2022Q4"]
    assert frame["target_quarter"].tolist() == [
        "2023Q1",
        "2023Q2",
        "2023Q3",
        "2023Q4",
        "2024Q1",
        "2024Q2",
        "2024Q3",
        "2024Q4",
    ]
    assert (
        frame["p10"].le(frame["p25"])
        & frame["p25"].le(frame["median"])
        & frame["median"].le(frame["p75"])
        & frame["p75"].le(frame["p90"])
    ).all()


def test_simulation_fan_output_paths_are_distinct_and_named():
    expected_names = {
        "simulation_fan_sarima.csv",
        "simulation_fan_sarima_trimmed_mean.csv",
        "simulation_fan_elastic_net.csv",
        "simulation_fan_elastic_net_trimmed_mean.csv",
        "simulation_fan_ensemble.csv",
        "simulation_fan_ensemble_trimmed_mean.csv",
        "simulation_fan_svar_headline.csv",
        "simulation_fan_svar_trimmed_mean.csv",
    }
    output_paths = tuple(simulation_fan.SIMULATION_FAN_OUTPUT_PATHS.values())

    assert len(output_paths) == 8
    assert len(set(output_paths)) == 8
    assert {path.name for path in output_paths} == expected_names
