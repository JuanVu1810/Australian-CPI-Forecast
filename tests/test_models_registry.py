import os
from types import SimpleNamespace

import pandas as pd
import pytest

from src.models import registry


def _write_shared_grid_report(path):
    pd.DataFrame(
        [
            {"model": "sarima", "horizon": 1, "rmse": 1.25},
            {"model": "sarima", "horizon": "overall", "rmse": 1.765204},
            {"model": "elastic_net", "horizon": 1, "rmse": 1.10},
            {"model": "elastic_net", "horizon": "overall", "rmse": 1.703591},
            {"model": "seasonal_naive", "horizon": "overall", "rmse": 2.0},
        ]
    ).to_csv(path, index=False)


def test_shared_grid_rmse_reads_family_overall_row(tmp_path):
    report_path = tmp_path / "model_comparison_elastic_net.csv"
    _write_shared_grid_report(report_path)

    rmse = registry._shared_grid_rmse("sarima", report_path)

    assert rmse == pytest.approx(1.765204)


def test_shared_grid_ranking_picks_elastic_net_when_sarima_own_grid_looks_better(tmp_path):
    report_path = tmp_path / "model_comparison_elastic_net.csv"
    _write_shared_grid_report(report_path)
    candidates = [
        registry.Candidate(
            family="sarima",
            rmse_overall=1.576414,
            metric_source="report",
            run_id="sarima-run",
        ),
        registry.Candidate(
            family="elastic_net",
            rmse_overall=1.703591,
            metric_source="mlflow",
            run_id="elastic-net-run",
        ),
    ]

    winner = registry._rank_candidates_on_shared_grid(candidates, report_path)

    assert winner.family == "elastic_net"
    assert winner.rmse_overall == pytest.approx(1.703591)
    assert winner.metric_source == "shared_grid_report"
    assert winner.run_id == "elastic-net-run"


def test_missing_shared_grid_report_raises_clear_error(tmp_path):
    missing_report = tmp_path / "missing_model_comparison_elastic_net.csv"

    with pytest.raises(RuntimeError, match="Shared-grid comparison report is missing"):
        registry._shared_grid_rmse("sarima", missing_report)


def test_shared_grid_report_missing_columns_raises_clear_error(tmp_path):
    report_path = tmp_path / "model_comparison_elastic_net.csv"
    pd.DataFrame([{"model": "sarima", "horizon": "overall"}]).to_csv(
        report_path,
        index=False,
    )

    with pytest.raises(RuntimeError, match="missing required columns: rmse"):
        registry._shared_grid_rmse("sarima", report_path)


def test_shared_grid_nan_rmse_raises_clear_error(tmp_path):
    report_path = tmp_path / "model_comparison_elastic_net.csv"
    pd.DataFrame(
        [{"model": "sarima", "horizon": "overall", "rmse": float("nan")}]
    ).to_csv(report_path, index=False)

    with pytest.raises(RuntimeError, match="Shared-grid RMSE for family 'sarima'.*is NaN"):
        registry._shared_grid_rmse("sarima", report_path)


def test_stale_shared_grid_report_raises_before_promotion_ranking(tmp_path):
    report_path = tmp_path / "model_comparison_elastic_net.csv"
    _write_shared_grid_report(report_path)
    os.utime(report_path, (1000, 1000))
    candidate = registry.Candidate(
        family="sarima",
        rmse_overall=1.576414,
        metric_source="mlflow",
        run_id="new-sarima-run",
    )
    client = SimpleNamespace(
        get_run=lambda run_id: SimpleNamespace(
            info=SimpleNamespace(start_time=1_001_000)
        )
    )

    with pytest.raises(RuntimeError, match="predates.*python -m src.models.model_comparison"):
        registry._ensure_shared_grid_report_fresh(client, [candidate], report_path)


def test_elastic_net_candidate_is_not_checked_against_its_own_report_mtime(tmp_path):
    report_path = tmp_path / "model_comparison_elastic_net.csv"
    _write_shared_grid_report(report_path)
    os.utime(report_path, (1000, 1000))
    candidate = registry.Candidate(
        family="elastic_net",
        rmse_overall=1.703591,
        metric_source="mlflow",
        run_id="fresh-elastic-net-run",
    )

    def fail_if_called(run_id):
        raise AssertionError("elastic_net's own run should not be freshness-checked")

    client = SimpleNamespace(get_run=fail_if_called)

    registry._ensure_shared_grid_report_fresh(client, [candidate], report_path)


def test_trimmed_mean_registry_config_is_separate_from_headline():
    assert registry.REGISTERED_MODEL_NAME == "cpi_forecast_champion"
    assert registry.TRIMMED_MEAN_REGISTERED_MODEL_NAME == "trimmed_mean_forecast_champion"
    assert registry.TRIMMED_MEAN_SHARED_GRID_REPORT_PATH != registry.SHARED_GRID_REPORT_PATH
    assert registry.TRIMMED_MEAN_REPORT_PATHS != registry.REPORT_PATHS
    assert registry.TRIMMED_MEAN_ELIGIBLE_FAMILIES == registry.ELIGIBLE_FAMILIES
    assert registry.TRIMMED_MEAN_MODEL_FAMILY_TAGS["sarima"] == "trimmed_mean_sarima"


def test_trimmed_mean_candidate_lookup_uses_target_specific_model_family_tag(tmp_path):
    report_path = tmp_path / "missing_report.csv"
    seen = {}

    class FakeClient:
        def search_runs(self, experiment_ids, filter_string, order_by, max_results):
            seen["filter"] = filter_string
            return []

    candidate = registry._candidate_for_family(
        FakeClient(),
        "experiment-id",
        "sarima",
        report_paths={"sarima": report_path},
        model_family_tags={"sarima": "trimmed_mean_sarima"},
    )

    assert candidate is None
    assert seen["filter"] == "tags.model_family = 'trimmed_mean_sarima'"
