"""Shared Streamlit renderers for RBA classifier report artifacts."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


@st.cache_data
def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@st.cache_data
def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _markdown_table(markdown: str, heading: str) -> pd.DataFrame:
    lines = markdown.splitlines()
    try:
        heading_index = next(index for index, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return pd.DataFrame()

    table_lines = []
    for line in lines[heading_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
        elif table_lines:
            break
    if len(table_lines) < 2:
        return pd.DataFrame()
    csv_lines = ["|".join(cell.strip() for cell in line.strip("|").split("|")) for line in table_lines]
    csv_lines = [csv_lines[0], *csv_lines[2:]]
    return pd.read_csv(io.StringIO("\n".join(csv_lines)), sep="|")


def _numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _overall_coverage_summary(
    coverage_reports: Sequence[tuple[str, Path]],
    columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for target_label, path in coverage_reports:
        if not path.exists():
            continue
        coverage = _load_csv(path)
        overall = coverage.loc[coverage["horizon"].astype(str).eq("overall")].copy()
        overall["target"] = target_label
        rows.append(overall)
    if not rows:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(rows, ignore_index=True)[list(columns)]


def _coverage_metric_value(coverage_summary: pd.DataFrame, spec: dict[str, Any]) -> str:
    row = coverage_summary.loc[
        coverage_summary["target"].eq(spec["target"])
        & coverage_summary["model"].eq(spec.get("model", "ensemble"))
    ]
    if row.empty:
        return "n/a"
    value = row[spec.get("column", "empirical_coverage")].iloc[0]
    if pd.isna(value):
        return "n/a"
    if spec.get("format") == "percent":
        return f"{float(value):.1%}"
    return str(value)


def render_historical_backtest_panel(
    *,
    rba_classifier_report_path: Path,
    missing_report_label: str,
    coverage_reports: Sequence[tuple[str, Path]],
    coverage_columns: Sequence[str],
    coverage_metrics: Sequence[dict[str, Any]],
    static_metrics: Sequence[tuple[str, str]] = (),
    caption: str,
) -> None:
    if not rba_classifier_report_path.exists():
        st.info(f"Historical classifier report not found: `{missing_report_label}`.")
        return

    report_text = load_markdown(rba_classifier_report_path)
    macro_f1 = _numeric_columns(
        _markdown_table(report_text, "## Macro-F1 Comparison"),
        ["macro_f1", "accuracy", "n"],
    )
    threshold_confusion = _numeric_columns(
        _markdown_table(report_text, "### threshold"),
        ["cut", "hold", "hike"],
    )
    coverage_summary = _overall_coverage_summary(coverage_reports, coverage_columns)

    with st.expander("Historical backtest panel"):
        threshold_macro = macro_f1.loc[macro_f1["model"].eq("threshold")]
        metric_cols = st.columns(2 + len(coverage_metrics) + len(static_metrics))
        metric_cols[0].metric(
            "Threshold macro-F1",
            "n/a" if threshold_macro.empty else f"{threshold_macro['macro_f1'].iloc[0]:.3f}",
        )
        metric_cols[1].metric("Brier-style score", "not reported")
        next_metric = 2
        for spec in coverage_metrics:
            metric_cols[next_metric].metric(spec["label"], _coverage_metric_value(coverage_summary, spec))
            next_metric += 1
        for label, value in static_metrics:
            metric_cols[next_metric].metric(label, value)
            next_metric += 1

        st.caption(caption)
        st.write("Threshold confusion matrix")
        st.dataframe(threshold_confusion, width="stretch", hide_index=True)
        st.write("Macro-F1 comparison")
        st.dataframe(macro_f1, width="stretch", hide_index=True)
        st.write("Overall calibrated simulation interval coverage")
        st.dataframe(coverage_summary.round(4), width="stretch", hide_index=True)
