"""Contract tests for the checked-in report files that the Jupyter Book reads.

Several book notebooks parse these files by heading text or column name. When a report is regenerated or
reworded and one of those anchors moves, the notebook fails the next time someone runs it (this happened to
book section 4.6 when the RBA report was rewritten). These tests fail first and say which cells to update.

Notebook cells that depend on these files:
  - 04_modeling/46_rba_classifier.ipynb, the confusion-matrix and bootstrap chart cell and the guard cell
    that re-checks the reported threshold macro-F1
  - 04_modeling/43_ensemble.ipynb, the simulation guard cell and the static fan chart cell
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"

RBA_REPORT = REPORTS / "rba_classifier_evaluation.md"


def _first_table_after(lines: list[str], heading_startswith: str) -> pd.DataFrame:
    """Same parsing the 4.6 notebook uses: the first markdown table after the line starting with the given text."""
    starts = [i for i, line in enumerate(lines) if line.startswith(heading_startswith)]
    assert starts, f"{RBA_REPORT.name} has no line starting with {heading_startswith!r}"
    first = next((i for i in range(starts[0], len(lines)) if lines[i].startswith("|")), None)
    assert first is not None, f"no table after {heading_startswith!r}"
    rows = []
    for line in lines[first:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return pd.DataFrame(rows[2:], columns=rows[0])


@pytest.fixture(scope="module")
def rba_lines() -> list[str]:
    return RBA_REPORT.read_text(encoding="utf-8").splitlines()


def test_rba_report_threshold_confusion_matrix_is_parseable(rba_lines):
    table = _first_table_after(rba_lines, "### threshold").set_index("actual")
    classes = ["cut", "hold", "hike"]
    counts = table.loc[classes, classes].astype(int)
    assert counts.to_numpy().sum() > 0


def test_rba_report_bootstrap_gap_table_is_parseable(rba_lines):
    gaps = _first_table_after(rba_lines, "## Small-Sample Caveat")
    assert {"gap", "observed_macro_f1_gap", "paired_bootstrap_95pct_ci"} <= set(gaps.columns)
    assert gaps["gap"].str.startswith("threshold - ").all()
    assert gaps["observed_macro_f1_gap"].astype(float).notna().all()
    intervals = gaps["paired_bootstrap_95pct_ci"].str.extract(r"\[(-?[\d.]+), (-?[\d.]+)\]")
    assert intervals.notna().all().all(), "every interval must look like [lo, hi]"


def test_rba_report_has_threshold_macro_f1_row(rba_lines):
    pattern = re.compile(r"^\| threshold\s+\|\s+([0-9.]+)\s+\|\s+[0-9.]+\s+\|\s+\d+\s+\|$")
    matches = [pattern.match(line) for line in rba_lines]
    assert any(matches), "the macro-F1 table must keep a `| threshold | macro_f1 | accuracy | n |` row"


@pytest.mark.parametrize("name", ["simulation_fan_ensemble.csv", "simulation_fan_ensemble_trimmed_mean.csv"])
def test_simulation_fan_reports_keep_their_columns(name):
    fan = pd.read_csv(REPORTS / name)
    assert {"forecast_origin", "horizon", "p10", "p25", "median", "p75", "p90"} <= set(fan.columns)
    assert fan["horizon"].nunique() == len(fan) == 8
    assert fan["forecast_origin"].nunique() == 1


@pytest.mark.parametrize(
    "name", ["simulation_paths_sample_ensemble.csv", "simulation_paths_sample_ensemble_trimmed_mean.csv"]
)
def test_simulation_path_samples_keep_their_columns(name):
    paths = pd.read_csv(REPORTS / name)
    assert {"forecast_origin", "draw_id", "horizon", "value"} <= set(paths.columns)


@pytest.mark.parametrize(
    "name",
    ["model_interval_calibration_validation.csv", "model_interval_calibration_validation_trimmed_mean.csv"],
)
def test_interval_calibration_reports_have_overall_raw_coverage(name):
    validation = pd.read_csv(REPORTS / name)
    assert {"model", "horizon", "raw_empirical_coverage"} <= set(validation.columns)
    overall = validation[validation["horizon"].astype(str) == "overall"].set_index("model")["raw_empirical_coverage"]
    assert {"ensemble", "sarima", "elastic_net"} <= set(overall.index)
