import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

import src.models.svar as svar
from src.models.svar import (
    DEFAULT_HORIZONS,
    calculate_dof_lag_cap,
    compare_system_a_to_b_irf_bands,
    check_var_stability,
    bootstrap_cholesky_irf_bands,
    johansen_cointegration_check,
    select_bic_lag_order,
    system_a_requires_full_treatment,
)


def test_calculate_dof_lag_cap_uses_lag_adjusted_usable_observations():
    assert calculate_dof_lag_cap(n_obs=123, n_vars=5) == 2

    selected_cap = calculate_dof_lag_cap(n_obs=63, n_vars=5)

    assert selected_cap == 1


def test_select_bic_lag_order_respects_dof_cap_and_flags_cap_edge():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        rng.normal(size=(63, 5)),
        columns=[f"y{i}" for i in range(5)],
    )

    decision = select_bic_lag_order(data)

    assert decision.max_lag_dof_cap == 1
    assert decision.selected_order == 1
    assert decision.hit_dof_cap is True
    assert decision.selected_usable_obs == 62
    assert decision.selected_obs_per_parameter == 62 / 5


def test_check_var_stability_reports_stable_synthetic_var_process():
    rng = np.random.default_rng(7)
    values = np.zeros((160, 5))
    coefficient_matrix = np.array(
        [
            [0.35, 0.05, 0.00, 0.00, 0.00],
            [0.00, 0.30, 0.04, 0.00, 0.00],
            [0.00, 0.00, 0.25, 0.03, 0.00],
            [0.00, 0.00, 0.00, 0.20, 0.02],
            [0.01, 0.00, 0.00, 0.00, 0.15],
        ]
    )
    innovations = rng.normal(scale=0.1, size=values.shape)
    for index in range(1, len(values)):
        values[index] = coefficient_matrix @ values[index - 1] + innovations[index]
    data = pd.DataFrame(values, columns=[f"y{i}" for i in range(5)])

    decision = check_var_stability(data, lag_order=1)

    assert decision.is_stable is True
    assert decision.stable_with_margin is True
    assert decision.max_companion_eigenvalue_modulus < 1.0
    assert min(decision.inverse_root_moduli) > 1.0


def test_johansen_cointegration_check_maps_all_rank_regimes(monkeypatch):
    rng = np.random.default_rng(11)
    data = pd.DataFrame(
        rng.normal(size=(120, 5)),
        columns=[f"y{i}" for i in range(5)],
    )

    def fake_result(trace_statistics):
        critical_values = np.column_stack(
            [
                np.full(5, 5.0),
                np.full(5, 10.0),
                np.full(5, 15.0),
            ]
        )
        return SimpleNamespace(lr1=np.asarray(trace_statistics), cvt=critical_values)

    regimes = [
        ([9.0, 8.0, 7.0, 6.0, 5.0], 0, "levels_var_svar"),
        ([20.0, 19.0, 9.0, 8.0, 7.0], 2, "vecm"),
        ([20.0, 19.0, 18.0, 17.0, 16.0], 5, "levels_var_svar"),
    ]
    for trace_statistics, expected_rank, expected_model in regimes:
        monkeypatch.setattr(
            svar,
            "coint_johansen",
            lambda *args, trace_statistics=trace_statistics, **kwargs: fake_result(
                trace_statistics
            ),
        )

        decision = johansen_cointegration_check(data, var_lag_order=2)

        assert decision.rank == expected_rank
        assert decision.n_vars == 5
        assert decision.phase_1b_model == expected_model
        assert set(decision.unit_root_pretests) == set(data.columns)


def test_johansen_cointegration_check_maps_synthetic_full_rank_to_levels_var_svar():
    rng = np.random.default_rng(2026)
    values = np.zeros((500, 5))
    innovations = rng.normal(size=values.shape)
    coefficients = np.array([0.20, 0.25, 0.30, 0.35, 0.40])
    for index in range(1, len(values)):
        values[index] = coefficients * values[index - 1] + innovations[index]
    data = pd.DataFrame(values, columns=[f"y{i}" for i in range(5)])

    decision = johansen_cointegration_check(data, var_lag_order=1)

    assert decision.rank == decision.n_vars == 5
    assert decision.phase_1b_model == "levels_var_svar"
    assert all(result.reject_unit_root_5pct for result in decision.unit_root_pretests.values())


def test_check_var_stability_margin_can_fail_for_near_unit_root_process():
    rng = np.random.default_rng(1)
    values = np.zeros((1200, 5))
    innovations = rng.normal(scale=0.1, size=values.shape)
    coefficients = np.array([0.985, 0.20, 0.25, 0.30, 0.35])
    for index in range(1, len(values)):
        values[index] = coefficients * values[index - 1] + innovations[index]
    data = pd.DataFrame(values, columns=[f"y{i}" for i in range(5)])

    decision = check_var_stability(data, lag_order=1)

    assert decision.is_stable is True
    assert decision.max_companion_eigenvalue_modulus > 1.0 - decision.stability_margin
    assert decision.stable_with_margin is False


def test_bootstrap_cholesky_irf_bands_returns_long_form_shape():
    rng = np.random.default_rng(123)
    values = np.zeros((90, 3))
    innovations = rng.normal(scale=0.2, size=values.shape)
    coefficients = np.diag([0.25, 0.20, 0.15])
    for index in range(1, len(values)):
        values[index] = coefficients @ values[index - 1] + innovations[index]
    data = pd.DataFrame(values, columns=["commodity_growth", "cpi_yoy", "cash_rate"])
    fitted = svar.fit_svar(data, lag_order=1)

    bands = bootstrap_cholesky_irf_bands(
        fitted,
        horizons=(1, 2, 3),
        n_bootstrap=8,
        seed=9,
    )

    assert bands.shape[0] == 3 * 3 * 3
    assert set(["response", "shock", "horizon", "irf", "lower", "upper"]).issubset(
        bands.columns
    )
    assert set(bands["horizon"]) == {1, 2, 3}
    assert bands["bootstrap_replications"].eq(8).all()
    assert bands["lower_quantile"].eq(0.1).all()
    assert bands["upper_quantile"].eq(0.9).all()
    assert np.isfinite(bands[["irf", "lower", "upper"]].to_numpy()).all()


def test_recursive_cholesky_period_zero_last_ordered_shock_has_no_earlier_effect():
    rng = np.random.default_rng(2028)
    values = np.zeros((140, 3))
    innovations = rng.normal(scale=0.2, size=values.shape)
    coefficients = np.diag([0.30, 0.25, 0.20])
    for index in range(1, len(values)):
        values[index] = coefficients @ values[index - 1] + innovations[index]
    data = pd.DataFrame(values, columns=["commodity_growth", "cpi_yoy", "cash_rate"])
    ordering = ("commodity_growth", "cpi_yoy", "cash_rate")
    fitted = svar.fit_svar(data, lag_order=1, ordering=ordering)

    long_form = svar.recursive_cholesky_irfs(fitted, horizons=(1,))
    period_zero_irfs = np.asarray(fitted.irf(periods=1).orth_irfs, dtype=float)[0]

    assert not long_form.empty
    cash_rate_index = ordering.index("cash_rate")
    earlier_indices = [ordering.index("commodity_growth"), ordering.index("cpi_yoy")]
    assert np.array_equal(
        period_zero_irfs[earlier_indices, cash_rate_index],
        np.zeros(len(earlier_indices)),
    )


def test_simulate_paths_from_fit_slices_presample_and_returns_stochastic_first_step():
    rng = np.random.default_rng(808)
    values = np.zeros((140, 3))
    coefficient_lag1 = np.diag([0.30, 0.25, 0.20])
    coefficient_lag2 = np.diag([0.10, 0.08, 0.06])
    innovations = rng.normal(scale=0.25, size=values.shape)
    for index in range(2, len(values)):
        values[index] = (
            coefficient_lag1 @ values[index - 1]
            + coefficient_lag2 @ values[index - 2]
            + innovations[index]
        )
    data = pd.DataFrame(values, columns=["commodity_growth", "cpi_yoy", "cash_rate"])
    fitted = svar.fit_svar(data, lag_order=2)

    paths = svar.simulate_paths_from_fit(fitted, steps=4, n_sims=12, seed=123)

    assert paths.shape == (12, 4, 3)
    assert (np.var(paths[:, 0, :], axis=0) > 0).all()
    assert not np.allclose(paths[:, 0, :], paths[0, 0, :])


def test_forecast_cumulative_unemployment_change_uses_system_b(monkeypatch):
    frame = pd.DataFrame(
        {
            "trimmed_mean_cpi_yoy": [2.1, 2.2, 2.3],
            "unemployment_rate": [4.1, 4.2, 4.3],
            "cash_rate": [3.8, 3.9, 4.0],
            "commodity_growth": [0.5, 0.6, 0.7],
            "inflation_expectations_business": [3.0, 3.1, 3.2],
        }
    )
    fit_marker = object()

    def fake_load_svar_level_frame(columns):
        assert columns == svar.SYSTEM_B_COLUMNS
        return frame

    def fake_fit_svar(data, lag_order, ordering):
        assert data is frame
        assert lag_order == svar.DEFAULT_SVAR_LAG_ORDER
        assert ordering == svar.SYSTEM_B_CHOLESKY_ORDER
        return fit_marker

    def fake_forecast_from_fit(fitted, steps):
        assert fitted is fit_marker
        assert steps == 3
        return pd.DataFrame({"unemployment_rate": [4.4, 4.6, 4.9]})

    monkeypatch.setattr(svar, "load_svar_level_frame", fake_load_svar_level_frame)
    monkeypatch.setattr(svar, "fit_svar", fake_fit_svar)
    monkeypatch.setattr(svar, "forecast_from_fit", fake_forecast_from_fit)

    change = svar.forecast_cumulative_unemployment_change(horizon=3)

    assert change == pytest.approx(0.6)


def test_forecast_cumulative_unemployment_change_requires_positive_horizon():
    with pytest.raises(ValueError, match="horizon must be at least 1"):
        svar.forecast_cumulative_unemployment_change(horizon=0)


def test_draw_contiguous_residual_blocks_preserves_residual_runs():
    residuals = np.arange(24, dtype=float).reshape(12, 2)
    rng = np.random.default_rng(42)

    drawn = svar._draw_contiguous_residual_blocks(
        residuals=residuals,
        draw_length=10,
        block_length=svar.DEFAULT_ARCH_LAGS,
        rng=rng,
    )

    assert drawn.shape == (10, 2)
    for start in range(0, len(drawn), svar.DEFAULT_ARCH_LAGS):
        chunk = drawn[start : start + svar.DEFAULT_ARCH_LAGS]
        assert any(
            np.array_equal(chunk, residuals[candidate : candidate + len(chunk)])
            for candidate in range(0, len(residuals) - len(chunk) + 1)
        )


def test_compare_system_a_to_b_irf_bands_flags_non_overlap_by_shared_shock_horizon():
    rows_a = []
    rows_b = []
    for shock in svar.SHARED_ESCALATION_SHOCKS:
        for horizon in DEFAULT_HORIZONS:
            rows_a.append(
                {
                    "response": "cpi_yoy",
                    "shock": shock,
                    "horizon": horizon,
                    "lower": -0.2,
                    "upper": 0.2,
                }
            )
            rows_b.append(
                {
                    "response": "trimmed_mean_cpi_yoy",
                    "shock": shock,
                    "horizon": horizon,
                    "lower": -0.1,
                    "upper": 0.3,
                }
            )
    rows_b[0]["lower"] = 0.25
    rows_b[0]["upper"] = 0.40
    system_a_bands = pd.DataFrame(rows_a)
    system_b_bands = pd.DataFrame(rows_b)

    comparison = compare_system_a_to_b_irf_bands(system_a_bands, system_b_bands)

    triggers = comparison.loc[comparison["escalate_system_a"]]
    assert len(comparison) == len(svar.SHARED_ESCALATION_SHOCKS) * len(DEFAULT_HORIZONS)
    assert len(triggers) == 1
    assert triggers.iloc[0]["shock"] == "unemployment_rate"
    assert triggers.iloc[0]["horizon"] == 1
    assert system_a_requires_full_treatment(comparison) is True


def test_compare_system_a_to_b_irf_bands_allows_irfs_only_when_bands_overlap():
    system_a_bands = pd.DataFrame(
        [
            {
                "response": "cpi_yoy",
                "shock": shock,
                "horizon": horizon,
                "lower": -0.2,
                "upper": 0.2,
            }
            for shock in svar.SHARED_ESCALATION_SHOCKS
            for horizon in DEFAULT_HORIZONS
        ]
    )
    system_b_bands = pd.DataFrame(
        [
            {
                "response": "trimmed_mean_cpi_yoy",
                "shock": shock,
                "horizon": horizon,
                "lower": -0.1,
                "upper": 0.3,
            }
            for shock in svar.SHARED_ESCALATION_SHOCKS
            for horizon in DEFAULT_HORIZONS
        ]
    )

    comparison = compare_system_a_to_b_irf_bands(system_a_bands, system_b_bands)

    assert comparison["escalate_system_a"].any() is np.False_
    assert system_a_requires_full_treatment(comparison) is False
