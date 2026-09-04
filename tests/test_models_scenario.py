import numpy as np
import pandas as pd

from api import main as api_main
from src.models import scenario


def test_compute_shock_size_uses_horizon_one_svar_surprise():
    forecast = pd.DataFrame(
        {
            "unemployment_rate": [4.25, 4.4],
            "cash_rate": [3.6, 3.5],
        }
    )

    shock_size = scenario.compute_shock_size(
        forecast,
        shock_variable="unemployment_rate",
        shock_value=4.75,
    )

    assert shock_size == 0.5


def test_combine_scenario_paths_zero_shock_collapses_exactly_to_tier1():
    tier1_paths = np.array(
        [
            [1.0, 1.2, 1.4],
            [2.0, 2.2, 2.4],
            [3.0, 3.2, 3.4],
        ]
    )
    irf_draws = np.arange(3 * 4 * 2 * 2, dtype=float).reshape(3, 4, 2, 2) + 1.0

    combined, contribution = scenario.combine_scenario_paths(
        tier1_paths=tier1_paths,
        irf_draws=irf_draws,
        shock_size=0.0,
        response_index=0,
        shock_index=1,
        horizons=(1, 2, 3),
    )

    assert np.array_equal(contribution, np.zeros_like(tier1_paths))
    assert np.array_equal(combined, tier1_paths)


def test_combined_interval_width_is_monotonic_in_absolute_shock_size():
    tier1_paths = np.full((5, 2), 2.0)
    irf_draws = np.zeros((5, 3, 2, 2), dtype=float)
    irf_draws[:, 0, 0, 1] = [-2.0, -1.0, 0.0, 1.0, 2.0]
    irf_draws[:, 1, 0, 1] = [-4.0, -2.0, 0.0, 2.0, 4.0]

    widths = []
    for shock_size in (0.0, 0.5, 1.0):
        combined, _ = scenario.combine_scenario_paths(
            tier1_paths=tier1_paths,
            irf_draws=irf_draws,
            shock_size=shock_size,
            response_index=0,
            shock_index=1,
            horizons=(1, 2),
        )
        lower, upper = scenario.scenario_interval_from_paths(combined)
        widths.append(upper - lower)

    assert np.all(widths[1] >= widths[0])
    assert np.all(widths[2] >= widths[1])
    assert np.any(widths[2] > widths[0])


def test_scenario_paths_are_pairwise_and_purely_additive_without_double_counting():
    tier1_paths = np.array(
        [
            [10.0, 20.0],
            [11.0, 21.0],
            [12.0, 22.0],
        ]
    )
    irf_draws = np.zeros((3, 3, 2, 2), dtype=float)
    irf_draws[:, 0, 0, 1] = [0.05, 0.10, 0.15]
    irf_draws[:, 1, 0, 1] = [0.1, 0.2, 0.3]
    irf_draws[:, 2, 0, 1] = [0.4, 0.5, 0.6]

    scenario_paths, contribution = scenario.combine_scenario_paths(
        tier1_paths=tier1_paths,
        irf_draws=irf_draws,
        shock_size=2.0,
        response_index=0,
        shock_index=1,
        horizons=(1, 2),
    )

    expected_contribution = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.4],
            [0.3, 0.6],
        ]
    )
    np.testing.assert_allclose(contribution, expected_contribution)
    np.testing.assert_allclose(scenario_paths - tier1_paths, expected_contribution)
    assert scenario_paths.shape == tier1_paths.shape


def test_run_scenario_keeps_tier1_paths_independent_of_shock_size(monkeypatch, tmp_path):
    quarters = pd.period_range("2020Q1", periods=8, freq="Q")
    curated_path = tmp_path / "curated.csv"
    pd.DataFrame(
        {
            "quarter": quarters.astype(str),
            "cpi_yoy": np.linspace(2.0, 3.0, len(quarters)),
            "unemployment_rate": np.linspace(4.0, 5.0, len(quarters)),
            "cash_rate": np.linspace(1.0, 2.0, len(quarters)),
            "commodity_growth": np.linspace(-1.0, 1.0, len(quarters)),
            "inflation_expectations_business": np.linspace(3.0, 4.0, len(quarters)),
            "cpi_yoy_lag1": np.linspace(1.9, 2.9, len(quarters)),
            "cpi_yoy_lag4": np.linspace(1.6, 2.6, len(quarters)),
            "cash_rate_change_lag1": np.zeros(len(quarters)),
            "unemployment_rate_change_lag1": np.zeros(len(quarters)),
            "inflation_expectations_business_lag1": np.linspace(2.8, 3.8, len(quarters)),
            "ppi_growth_lag2": np.ones(len(quarters)),
            "commodity_growth_lag1": np.linspace(-0.5, 0.5, len(quarters)),
            "wti_growth_lag1": np.ones(len(quarters)),
            "ppi_growth_lag2_sq": np.ones(len(quarters)),
            "wti_growth_lag1_sq": np.ones(len(quarters)),
            "cash_rate_change_lag1_x_unemployment_rate_change_lag1": np.zeros(
                len(quarters)
            ),
        }
    ).to_csv(curated_path, index=False)

    class FakeFit:
        names = (
            "commodity_growth",
            "unemployment_rate",
            "cpi_yoy",
            "inflation_expectations_business",
            "cash_rate",
        )

    baseline = np.array(
        [
            [2.0, 2.1],
            [2.2, 2.3],
            [2.4, 2.5],
        ]
    )
    irf_draws = np.zeros((3, 3, 5, 5), dtype=float)
    irf_draws[:, 1, 2, 1] = [0.2, 0.3, 0.4]
    irf_draws[:, 2, 2, 1] = [0.5, 0.6, 0.7]
    captured_tier1 = []

    monkeypatch.setattr(scenario.svar, "fit_svar", lambda *args, **kwargs: FakeFit())
    monkeypatch.setattr(
        scenario.svar,
        "forecast_from_fit",
        lambda *args, **kwargs: pd.DataFrame({"unemployment_rate": [4.5, 4.6]}),
    )
    monkeypatch.setattr(
        scenario.svar,
        "bootstrap_orth_irf_draws",
        lambda *args, **kwargs: irf_draws.copy(),
    )
    monkeypatch.setattr(
        scenario.ensemble,
        "horizon_rmse_weights",
        lambda horizons, path=None: {horizon: (0.5, 0.5) for horizon in horizons},
    )

    def fake_simulate_ensemble_paths(*args, **kwargs):
        del args, kwargs
        paths = baseline.copy()
        captured_tier1.append(paths.copy())
        return paths

    monkeypatch.setattr(
        scenario.ensemble,
        "simulate_ensemble_paths",
        fake_simulate_ensemble_paths,
    )

    zero = scenario.run_scenario(
        target="headline",
        shock_variable="unemployment_rate",
        shock_value=4.5,
        horizons=(1, 2),
        n_sims=3,
        curated_path=curated_path,
    )
    nonzero = scenario.run_scenario(
        target="headline",
        shock_variable="unemployment_rate",
        shock_value=5.5,
        horizons=(1, 2),
        n_sims=3,
        curated_path=curated_path,
    )

    assert np.array_equal(captured_tier1[0], captured_tier1[1])
    assert np.array_equal(zero.combined_paths, captured_tier1[0])
    assert nonzero.shock_size == 1.0
    np.testing.assert_allclose(nonzero.baseline_paths, captured_tier1[1])
    np.testing.assert_allclose(
        nonzero.combined_paths - nonzero.baseline_paths,
        nonzero.shock_contribution,
    )


def test_forecast_scenario_api_returns_caveat_and_adjusted_interval(monkeypatch):
    fake_result = scenario.ScenarioResult(
        target="headline",
        target_column="cpi_yoy",
        shock_variable="cash_rate",
        shock_value=4.35,
        shock_size=0.25,
        horizons=(1, 2),
        forecast=[3.1, 3.2],
        interval_lower=[2.6, 2.7],
        interval_upper=[3.6, 3.7],
        quarters=["2026Q1", "2026Q2"],
        forecast_origin="2025Q4",
        caveat=scenario.SCENARIO_CAVEAT,
        baseline_paths=np.zeros((3, 2)),
        shock_contribution=np.zeros((3, 2)),
        combined_paths=np.zeros((3, 2)),
    )

    monkeypatch.setattr(api_main.scenario, "run_scenario", lambda **kwargs: fake_result)

    response = api_main.forecast_scenario(
        api_main.ScenarioForecastRequest(
            target="headline",
            shock_variable="cash_rate",
            shock_value=4.35,
            horizons=[1, 2],
            n_sims=100,
        )
    ).model_dump()

    assert response["shock_size"] == 0.25
    assert response["forecast"] == [3.1, 3.2]
    assert response["interval_lower"] == [2.6, 2.7]
    assert response["interval_upper"] == [3.6, 3.7]
    assert "fail multivariate residual whiteness and normality diagnostics" in response["caveat"]
