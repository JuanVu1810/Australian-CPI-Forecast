"""Tests for the reproducibility measurements behind the book's chapter 6 chart."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reproducibility_check import SUMMARY_COLUMNS, SUMMARY_PATH, compare_frames, compare_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_compare_frames_counts_numeric_and_text_changes():
    base = pd.DataFrame({"quarter": ["2026Q1", "2026Q2"], "x": [1.0, 2.0], "y": [None, 3.0]})
    cand = pd.DataFrame({"quarter": ["2026Q1", "2026Q3"], "x": [1.0, 2.5], "y": [None, 3.0]})
    cells, changed, max_diff = compare_frames(base, cand)
    assert cells == 6
    assert changed == 2  # one text cell, one number
    assert max_diff == 0.5


def test_compare_frames_treats_shape_mismatch_as_all_changed():
    base = pd.DataFrame({"x": [1.0, 2.0]})
    cand = pd.DataFrame({"x": [1.0]})
    cells, changed, _ = compare_frames(base, cand)
    assert cells == changed == 2


def test_compare_json_splits_numbers_from_labels_and_ignores_run_ids():
    base = {"run_id": "a", "forecast": [1.0, 2.0], "action": "hold", "models": [{"p_cut": 0.006}]}
    cand = {"run_id": "b", "forecast": [1.0, 2.5], "action": "hike", "models": [{"p_cut": 0.008}]}
    result = compare_json(base, cand)
    n, changed, max_diff = result["numbers"]
    assert (n, changed) == (3, 2)
    assert abs(max_diff - 0.5) < 1e-12
    assert result["labels"] == (1, 1, 0.0)


def test_checked_in_summary_has_the_columns_the_book_reads():
    summary = pd.read_csv(PROJECT_ROOT / SUMMARY_PATH.relative_to(PROJECT_ROOT))
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert summary["scenario_id"].nunique() >= 6
    assert (summary["n_changed"] <= summary["n_compared"]).all()


def test_every_recipe_points_at_files_that_exist():
    """The `how` column names scripts and saved records; they must be in the repo so a row can be repeated."""
    import re

    summary = pd.read_csv(SUMMARY_PATH)
    mentioned = set()
    for how in summary["how"]:
        mentioned.update(m.rstrip(".,;:)") for m in re.findall(r"(?:scripts|reports)/[\w./<>-]+", how))
    concrete = {m for m in mentioned if "<" not in m}
    assert concrete, "no file references found in the how column"
    missing = sorted(m for m in concrete if not (PROJECT_ROOT / m).exists())
    assert not missing, f"recipes mention files that are not in the repo: {missing}"


def test_saved_run_records_are_present():
    runs = PROJECT_ROOT / "reports/reproducibility_runs"
    for name in ("README.md", "machine.txt", "code_version.txt", "environment_pinned.txt", "environment_newest.txt",
                 "retrain_newest_steps.log"):
        assert (runs / name).is_file(), name
    for kernel in ("default", "sandybridge", "nehalem", "prescott"):
        assert (runs / "rba_classifier_runs" / kernel / "predictions.csv").is_file(), kernel


def test_compare_json_groups_numbers_by_path():
    base = {"models": [{"p_cut": 0.5, "p_hold": 0.5}], "headline_forecast": 3.0, "run_id": "a"}
    cand = {"models": [{"p_cut": 0.52, "p_hold": 0.5}], "headline_forecast": 3.0, "run_id": "b"}
    groups = [("probabilities", "probability", r"/p_(cut|hold|hike)$"), ("forecasts", "pp", r"/headline_forecast$")]
    result = compare_json(base, cand, groups)
    assert result["probabilities"][:2] == (2, 1) and abs(result["probabilities"][2] - 0.02) < 1e-12
    assert result["forecasts"] == (1, 0, 0.0)
    assert "labels" not in result


def test_raw_api_responses_are_saved_for_the_api_scenarios():
    api = PROJECT_ROOT / "reports/reproducibility_runs/api_responses"
    for scenario in ("one_thread", "other_cpu", "cloud_run", "cloud_run_repeat"):
        files = sorted((api / scenario).glob("*.json"))
        assert len(files) == 10, (scenario, len(files))  # five requests, a baseline and a candidate each
