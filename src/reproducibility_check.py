"""Measure how far a rerun of this project can drift, and save the measurements.

The book's chapter 6 chart "How different can a rerun look?" reads ``reports/reproducibility_check.csv``. Each
sub-command below compares a baseline with a rerun and upserts one or more rows into that file (one row per scenario
and quantity). ``reports/reproducibility_check_files.csv`` keeps the per-file detail behind the file counts.

    compare-files       regenerated files against the checked-in ones (byte for byte, then cell by cell for CSVs)
    compare-api         two running APIs against each other (forecasts, simulations, RBA calls)
    compare-classifier  RBA classifier predictions from one or more reruns against a baseline
    data-vintages       two downloads of the same ABS series, read from git history

How each scenario was produced. Reruns happen in scratch copies of the repo (``scripts/repro_scratch_copy.sh``), so the
checked-in reports are never touched; the ``how`` column of the CSV repeats the recipe, and
``reports/reproducibility_runs/README.md`` gives the exact commands, the saved artefacts and what was not kept.
Hardware is imitated on one machine by forcing OpenBLAS onto another CPU kernel (``OPENBLAS_CORETYPE``) or onto one
thread. Helpers: ``scripts/repro_retrain_chain.sh`` (retrain and regenerate) and ``scripts/repro_rba_predictions.py``
(classifier predictions under a given kernel).

    book_rebuild    jupyter-book build twice into empty folders; compare-files on the HTML pages
    restore         restore mlruns_snapshot/ into an empty copy, serve it, run tableau_export; compare-files
    retrain_new     a fresh venv with unpinned requirements, then the README's training commands and report
                    scripts; compare-files on every file the run rewrote
    one_thread      serve the restored runs with OPENBLAS_NUM_THREADS=1; compare-api against the default
    other_cpu       the same with OPENBLAS_CORETYPE=SANDYBRIDGE
    cloud_run       compare-api, a local server against the live Cloud Run service
    cloud_run_repeat  compare-api, the live Cloud Run service against itself (are its answers repeatable?)
    classifier_cpu  run_rba_classifier_evaluation() under three other CPU kernels; compare-classifier
    classifier_env  compare-classifier against the prediction table of an older committed report
    new_data        data-vintages
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "reports/reproducibility_check.csv"
DETAIL_PATH = PROJECT_ROOT / "reports/reproducibility_check_files.csv"

SUMMARY_COLUMNS = [
    "scenario_id",
    "scenario",
    "environment",
    "quantity",
    "unit",
    "n_compared",
    "n_changed",
    "max_abs_diff",
    "measured_on",
    "how",
]
DETAIL_COLUMNS = ["scenario_id", "file", "byte_identical", "cells_compared", "cells_changed", "max_abs_diff"]

# Keys that are expected to differ between any two runs (new run IDs) and say nothing about the results.
IGNORED_KEYS = {"run_id"}


def _today() -> str:
    return dt.date.today().isoformat()


def _upsert(path: Path, rows: pd.DataFrame, key: list[str], columns: list[str]) -> None:
    """Add rows, replacing everything already saved for the same scenario (a scenario is re-measured as a whole)."""
    rows = rows[columns].copy()
    rows["max_abs_diff"] = rows["max_abs_diff"].astype(float)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["scenario_id"].isin(rows["scenario_id"].unique())]
        merged = pd.concat([old, rows], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="last")
        # keep scenario order: first appearance of each scenario_id
        order = {sid: i for i, sid in enumerate(pd.unique(merged["scenario_id"]))}
        merged = merged.sort_values("scenario_id", key=lambda s: s.map(order), kind="stable")
    else:
        merged = rows
    merged.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")


def _summary_row(args, quantity: str, unit: str, n_compared: int, n_changed: int, max_abs_diff: float | None) -> dict:
    return {
        "scenario_id": args.scenario_id,
        "scenario": args.scenario,
        "environment": args.environment,
        "quantity": quantity,
        "unit": unit,
        "n_compared": int(n_compared),
        "n_changed": int(n_changed),
        "max_abs_diff": max_abs_diff,
        "measured_on": _today(),
        "how": args.how,
    }


# ---------------------------------------------------------------- files


def compare_frames(base: pd.DataFrame, cand: pd.DataFrame) -> tuple[int, int, float]:
    """Cells compared, cells changed and the largest absolute numeric difference between two same-shaped frames."""
    if base.shape != cand.shape or list(base.columns) != list(cand.columns):
        return int(base.size), int(base.size), float("nan")
    changed = 0
    max_diff = 0.0
    for col in base.columns:
        a, b = base[col], cand[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            both_nan = np.isnan(av) & np.isnan(bv)
            diff = np.where(both_nan, 0.0, np.abs(av - bv))
            diff = np.where(np.isnan(diff), np.inf, diff)
            changed += int((diff > 0).sum())
            if len(diff):
                max_diff = max(max_diff, float(diff.max()))
        else:
            changed += int((a.astype(str).to_numpy() != b.astype(str).to_numpy()).sum())
    return int(base.size), changed, max_diff


def _compare_one(base_path: Path, cand_path: Path) -> tuple[bool, int, int, float]:
    identical = base_path.read_bytes() == cand_path.read_bytes()
    if identical:
        cells = 0
        if base_path.suffix == ".csv":
            cells = int(pd.read_csv(base_path).size)
        return True, cells, 0, 0.0
    if base_path.suffix == ".csv":
        cells, changed, max_diff = compare_frames(pd.read_csv(base_path), pd.read_csv(cand_path))
        return False, cells, changed, max_diff
    return False, 0, 0, float("nan")


def cmd_compare_files(args) -> None:
    baseline, candidate = Path(args.baseline), Path(args.candidate)
    if args.newer_than:
        marker = Path(args.newer_than).stat().st_mtime
    else:
        marker = None
    files = []
    for pattern in args.glob:
        for path in sorted(candidate.glob(pattern)):
            if not path.is_file():
                continue
            if marker is not None and path.stat().st_mtime <= marker:
                continue
            rel = path.relative_to(candidate)
            if (baseline / rel).is_file():
                files.append(rel)
    files = sorted(set(files))
    if not files:
        raise SystemExit("no files to compare")

    detail = []
    for rel in files:
        identical, cells, changed, max_diff = _compare_one(baseline / rel, candidate / rel)
        detail.append(
            {
                "scenario_id": args.scenario_id,
                "file": str(rel),
                "byte_identical": identical,
                "cells_compared": cells,
                "cells_changed": changed,
                "max_abs_diff": max_diff,
            }
        )
    detail_df = pd.DataFrame(detail, columns=DETAIL_COLUMNS)

    n_files = len(detail_df)
    n_diff_files = int((~detail_df["byte_identical"]).sum())
    rows = [_summary_row(args, "files regenerated", "files", n_files, n_diff_files, None)]
    csv_rows = detail_df[detail_df["cells_compared"] > 0]
    if len(csv_rows):
        max_diff = csv_rows["max_abs_diff"].max()
        rows.append(
            _summary_row(
                args, "table cells", "cells", csv_rows["cells_compared"].sum(), csv_rows["cells_changed"].sum(), max_diff
            )
        )
    _upsert(SUMMARY_PATH, pd.DataFrame(rows), ["scenario_id", "quantity"], SUMMARY_COLUMNS)
    if DETAIL_PATH.exists():
        old = pd.read_csv(DETAIL_PATH)
        old = old[old["scenario_id"] != args.scenario_id]
        detail_df = pd.concat([old, detail_df], ignore_index=True)
    detail_df.to_csv(DETAIL_PATH, index=False, float_format="%.6g", lineterminator="\n")
    print(detail_df[detail_df["scenario_id"] == args.scenario_id].to_string(index=False))
    print(pd.DataFrame(rows)[["quantity", "n_compared", "n_changed", "max_abs_diff"]].to_string(index=False))


# ---------------------------------------------------------------- API


def _request(url: str, payload: dict | None, timeout: int) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _leaves(obj, path=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in IGNORED_KEYS:
                continue
            yield from _leaves(value, f"{path}/{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _leaves(value, f"{path}[{i}]")
    else:
        yield path, obj


def compare_json(base: dict, cand: dict, groups=None) -> dict[str, tuple[int, int, float]]:
    """Split two API responses into numbers and labels. Returns {kind: (compared, changed, max_abs_diff)}.

    ``groups`` is an optional list of ``(name, unit, regex)``. A number whose path matches a regex is counted under that
    group's name; the remaining numbers are counted under "other numbers". Without groups every number is "numbers".
    """
    b, c = dict(_leaves(base)), dict(_leaves(cand))
    out: dict[str, list] = {}

    def add(kind: str, changed: bool, diff: float = 0.0) -> None:
        entry = out.setdefault(kind, [0, 0, 0.0])
        entry[0] += 1
        entry[1] += int(changed)
        entry[2] = max(entry[2], diff)

    for path in sorted(set(b) | set(c)):
        x, y = b.get(path), c.get(path)
        is_num = isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float))
        if not is_num:
            add("labels", x != y)
            continue
        kind = "numbers" if groups is None else "other numbers"
        for name, _unit, pattern in groups or []:
            if re.search(pattern, path):
                kind = name
                break
        diff = abs(float(x) - float(y))
        add(kind, diff > 0, diff)
    return {k: (v[0], v[1], v[2]) for k, v in out.items()}


_FORECAST_NUMBERS = ("forecast and interval numbers", "pp", r"/(forecast|interval_lower|interval_upper)\[")
API_CALLS = {
    # quantity label: (path, request body, groups of numbers to report separately)
    "point forecasts and intervals": ("/forecast/all", {"horizon": 8}, [_FORECAST_NUMBERS]),
    "trimmed-mean forecasts and intervals": ("/forecast/trimmed-mean/all", {"horizon": 8}, [_FORECAST_NUMBERS]),
    "credit stress test": ("/credit-risk/stress-test", None, [("numbers", "mixed", r".")]),
    "scenario forecast (simulated)": (
        "/forecast/scenario",
        {"target": "headline", "shock_variable": "unemployment_rate", "shock_value": 6.0, "horizons": [1, 2, 3, 4]},
        [_FORECAST_NUMBERS],
    ),
    "RBA action readings (simulated)": (
        "/rba-action",
        None,
        [
            ("probabilities", "probability", r"/(p_cut|p_hold|p_hike|confidence)$"),
            ("forecasts", "pp", r"/(headline_forecast|trimmed_mean_forecast)$"),
        ],
    ),
}


def cmd_compare_api(args) -> None:
    rows = []
    for quantity, (path, payload, groups) in API_CALLS.items():
        if args.only and quantity not in args.only:
            continue
        print(f"{quantity}: calling both APIs ...", flush=True)
        base = _request(args.baseline_url.rstrip("/") + path, payload, args.timeout)
        cand = _request(args.candidate_url.rstrip("/") + path, payload, args.timeout)
        if args.save_dir:
            out = Path(args.save_dir) / args.scenario_id
            out.mkdir(parents=True, exist_ok=True)
            slug = re.sub(r"[^a-z0-9]+", "_", quantity.lower()).strip("_")
            for side, body in (("baseline", base), ("candidate", cand)):
                (out / f"{slug}_{side}.json").write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        res = compare_json(base, cand, groups)
        units = {name: unit for name, unit, _ in groups}
        for name, (n, changed, mx) in res.items():
            if name == "labels":
                rows.append(_summary_row(args, f"{quantity}: labels", "labels", n, changed, None))
            else:
                rows.append(_summary_row(args, f"{quantity}: {name}", units.get(name, "other"), n, changed, mx))
    df = pd.DataFrame(rows)
    _upsert(SUMMARY_PATH, df, ["scenario_id", "quantity"], SUMMARY_COLUMNS)
    print(df[["quantity", "n_compared", "n_changed", "max_abs_diff"]].to_string(index=False))


# ---------------------------------------------------------------- classifier

PREDICTION_ROW = re.compile(r"^\|\s*(\d{4}Q\d)\s*\|\s*(\w+)\s*\|\s*(cut|hold|hike)\s*\|")
CUTOFF_ROW = re.compile(r"^\|\s*ordered_logit\s*\|\s*\d{4}Q\d\s*\|\s*\d{4}Q\d\s*\|\s*(-?[\d.]+)\s*\|\s*[\d.]+\s*\|\s*(-?[\d.]+)")


def _read_git(spec: str) -> str:
    return subprocess.run(["git", "show", spec], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout


def predictions_from_report(text: str) -> pd.DataFrame:
    """The per-quarter prediction table that older versions of the RBA classifier report carried."""
    rows = [m.groups() for m in map(PREDICTION_ROW.match, text.splitlines()) if m]
    return pd.DataFrame(rows, columns=["target_quarter", "model", "predicted_action"])


def first_fold_cutoffs(text: str) -> list[float]:
    """Ordered logit cut-off changes (first fold to last fold) from the report's first-vs-last-fold table."""
    for line in text.splitlines():
        m = CUTOFF_ROW.match(line)
        if m:
            return [float(m.group(1)), float(m.group(2))]
    raise SystemExit("cut-off table not found in the report")


def cmd_compare_classifier(args) -> None:
    """Compare a baseline with one or more reruns; counts add up over the reruns, differences take the largest."""
    key = ["model", "target_quarter"]
    if args.baseline_git:
        base = predictions_from_report(_read_git(args.baseline_git))
    else:
        base = pd.read_csv(args.baseline)
    calls = [0, 0]
    ordered_calls = [0, 0]
    probs: dict[str, list] = {}
    for cand_path in args.candidate:
        merged = base.merge(pd.read_csv(cand_path), on=key, suffixes=("_base", "_cand"), validate="one_to_one")
        if not len(merged):
            raise SystemExit("no quarters in common")
        changed = merged["predicted_action_base"].ne(merged["predicted_action_cand"])
        calls[0] += len(merged)
        calls[1] += int(changed.sum())
        is_ordered = merged["model"].isin(["ordered_logit", "ordered_probit"])
        ordered_calls[0] += int(is_ordered.sum())
        ordered_calls[1] += int((changed & is_ordered).sum())
        prob_cols = [c for c in ("p_cut", "p_hold", "p_hike") if f"{c}_base" in merged and f"{c}_cand" in merged]
        for group, mask in (
            ("threshold model (simulated)", merged["model"].eq("threshold")),
            ("ordered logit and probit", is_ordered),
            ("other models", ~merged["model"].eq("threshold") & ~is_ordered),
        ):
            if not prob_cols:
                break
            sub = merged[mask]
            diffs = pd.concat([(sub[f"{c}_base"] - sub[f"{c}_cand"]).abs() for c in prob_cols]).dropna()
            if len(diffs):
                cur = probs.setdefault(group, [0, 0, 0.0])
                probs[group] = [cur[0] + len(diffs), cur[1] + int((diffs > 0).sum()), max(cur[2], float(diffs.max()))]
        print(cand_path, "changed calls:")
        print(merged.loc[changed, key + ["predicted_action_base", "predicted_action_cand"]].to_string(index=False))
    rows = [
        _summary_row(args, "RBA classifier calls, all models compared", "calls", *calls, None),
        _summary_row(args, "RBA classifier calls, ordered logit and probit", "calls", *ordered_calls, None),
    ]
    for group, values in probs.items():
        rows.append(_summary_row(args, f"RBA classifier probabilities, {group}", "probability", *values))
    if args.report_baseline and args.report_candidate:
        a = first_fold_cutoffs(Path(args.report_baseline).read_text("utf-8"))
        moved = [
            abs(x - y)
            for cand in args.report_candidate
            for x, y in zip(a, first_fold_cutoffs(Path(cand).read_text("utf-8")))
        ]
        rows.append(
            _summary_row(args, "ordered logit cut-offs, first fold vs last", "cut-offs", len(moved), sum(m > 0 for m in moved), max(moved))
        )
    df = pd.DataFrame(rows)
    _upsert(SUMMARY_PATH, df, ["scenario_id", "quantity"], SUMMARY_COLUMNS)
    print(df[["quantity", "n_compared", "n_changed", "max_abs_diff"]].to_string(index=False))


# ---------------------------------------------------------------- data vintages


def _git_csv(rev: str, path: str) -> pd.DataFrame:
    text = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout
    return pd.read_csv(io.StringIO(text))


def cmd_data_vintages(args) -> None:
    old = _git_csv(args.old_rev, args.old_path)
    new = _git_csv(args.new_rev, args.new_path)
    value = [c for c in old.columns if c != "date"][0]
    both = old.merge(new, on="date", suffixes=("_old", "_new"))
    diff = (both[f"{value}_new"] - both[f"{value}_old"]).abs()
    rel = diff / both[f"{value}_old"].abs()
    rows = [
        _summary_row(args, f"{value} monthly values", "values", len(both), int((diff > 0).sum()), float(diff.max())),
        _summary_row(args, f"{value} monthly values, relative", "share", len(both), int((diff > 0).sum()), float(rel.max())),
    ]
    df = pd.DataFrame(rows)
    _upsert(SUMMARY_PATH, df, ["scenario_id", "quantity"], SUMMARY_COLUMNS)
    print(df[["quantity", "n_compared", "n_changed", "max_abs_diff"]].to_string(index=False))


# ---------------------------------------------------------------- cli


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scenario-id", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--environment", required=True)
    p.add_argument("--how", required=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compare-files")
    _common(p)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--glob", nargs="+", default=["reports/*.csv", "reports/*.md", "reports/tableau/*.csv"])
    p.add_argument("--newer-than", help="only compare candidate files modified after this marker file")
    p.set_defaults(func=cmd_compare_files)

    p = sub.add_parser("compare-api")
    _common(p)
    p.add_argument("--baseline-url", required=True)
    p.add_argument("--candidate-url", required=True)
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--only", nargs="*", help="limit to these quantity labels")
    p.add_argument("--save-dir", help="also keep both servers' raw JSON responses under <dir>/<scenario-id>/")
    p.set_defaults(func=cmd_compare_api)

    p = sub.add_parser("compare-classifier")
    _common(p)
    p.add_argument("--baseline", help="predictions CSV from run_rba_classifier_evaluation")
    p.add_argument("--baseline-git", help="REV:PATH of an older report that still carries the prediction table")
    p.add_argument("--candidate", nargs="+", required=True, help="predictions CSV(s) from run_rba_classifier_evaluation")
    p.add_argument("--report-baseline", help="report whose first-vs-last-fold cut-off table is compared")
    p.add_argument("--report-candidate", nargs="+")
    p.set_defaults(func=cmd_compare_classifier)

    p = sub.add_parser("data-vintages")
    _common(p)
    p.add_argument("--old-rev", default="31ea9a5")
    p.add_argument("--old-path", default="dataset/abs/household_spending_1995_2025.csv")
    p.add_argument("--new-rev", default="962221a")
    p.add_argument("--new-path", default="dataset/abs/household_spending_1995_2026.csv")
    p.set_defaults(func=cmd_data_vintages)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
