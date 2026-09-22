# Reproducibility runs: what was measured, how to repeat it, and what was kept

Results live in `reports/reproducibility_check.csv` (one row per scenario/quantity) and
`reports/reproducibility_check_files.csv` (per-file detail). This folder holds what's needed to repeat and audit them.
All runs were made 2026-09-22 (Australian time) on one machine, always in scratch copies of the repo — never the checkout,
since the model scripts write into `reports/` and `data/`.

## What is saved here

| File | What it is |
|---|---|
| `machine.txt` | CPU, cores, RAM, OS, and the OpenBLAS build and default kernel that numpy and scipy loaded |
| `code_version.txt` | git commit, and the paths that differed from it (the reruns used the working tree) |
| `environment_pinned.txt` | `pip list` of the `cpi_forecast` conda env that produced the checked-in results |
| `environment_newest.txt` | `pip list` of a fresh venv installed **without** `constraints.txt`. Reinstalled after the rerun because the original venv was deleted; the main packages match the versions printed during the rerun |
| `retrain_newest_steps.log` | start and end time and exit code of each step of the newest-packages retrain (about 1 hour in total) |
| `api_responses/<scenario>/` | both servers' raw JSON responses (forecasts, trimmed-mean forecasts, credit stress test, scenario forecast, RBA action readings) for `one_thread`, `other_cpu`, `cloud_run` and `cloud_run_repeat`. The differences in the CSV can be recomputed from these files alone |
| `rba_classifier_runs/<kernel>/` | `predictions.csv` (every model, every test quarter) and `report.md` from `scripts/repro_rba_predictions.py`, for the default Haswell kernel and for `SANDYBRIDGE`, `NEHALEM` and `PRESCOTT` |

## What was not kept

- Regenerated files, retrained MLflow runs, and scratch repos (deleted, ~2.8 GB). File-level results survive as counts
  and differences in the two CSVs; repeat a scenario to get the files back.
- Timings of the API and classifier scenarios.
- Per-step retrain logs (only the timing log above survives) and the first install's pip output.
- The original fresh venv (see `environment_newest.txt`) and the older classifier report `classifier_env` compares
  against (in git: `git show 5cbea9a^:reports/rba_classifier_evaluation.md`).

## Repeating each scenario

`S` is any empty folder outside the repo. Run `scripts/repro_scratch_copy.sh $S/repo` first for scenarios needing a copy.
`compare-*` commands run from the real checkout and add/replace rows in the two CSVs; each also needs `--scenario-id`,
`--scenario`, `--environment` and `--how` (use the values in the CSV).

| Scenario id | Steps |
|---|---|
| `book_rebuild` | `jupyter-book build book/australian_cpi_forecasting --path-output $S/b1`, again with `$S/b2`; then `python -m src.reproducibility_check compare-files --baseline $S/b1/_build/html --candidate $S/b2/_build/html --glob "**/*.html"` |
| `restore` | in `$S/repo`: `python -m src.mlruns_snapshot restore`; `uvicorn api.main:app --port 8201` in one terminal; `touch .chain_start` then `python -m src.models.tableau_export --api-base-url http://localhost:8201`; then `compare-files --baseline . --candidate $S/repo --glob "reports/tableau/*.csv" --newer-than $S/repo/.chain_start` |
| `retrain_new` | `python3.11 -m venv $S/venv_new && $S/venv_new/bin/pip install -r requirements.txt` (no constraints); `scripts/repro_scratch_copy.sh $S/repo_new`; `scripts/repro_retrain_chain.sh $S/repo_new $S/venv_new/bin/python 8101` (about 1 hour); then `compare-files --baseline . --candidate $S/repo_new --glob "reports/*.csv" "reports/*.md" "reports/tableau/*.csv" "data/curated/*.csv" --newer-than $S/repo_new/.chain_start` |
| `one_thread`, `other_cpu` | with the restored copy of `restore` still serving on 8201, start a second server from the same copy: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uvicorn api.main:app --port 8203` (`one_thread`) or `OPENBLAS_CORETYPE=SANDYBRIDGE uvicorn api.main:app --port 8202` (`other_cpu`); then `compare-api --baseline-url http://localhost:8201 --candidate-url http://localhost:820x --save-dir reports/reproducibility_runs/api_responses` |
| `cloud_run` | the same, with `--candidate-url` set to the live service in the README. It sends the same read-only requests as the Streamlit demo and takes about 2 minutes |
| `cloud_run_repeat` | `compare-api` with both `--baseline-url` and `--candidate-url` set to the live service, so the same requests are sent twice |
| `classifier_cpu` | for each kernel, `OPENBLAS_CORETYPE=<KERNEL> python scripts/repro_rba_predictions.py $S/rba_<kernel>` (about 10 minutes each; the baseline run has no variable set); then `compare-classifier --baseline <default>/predictions.csv --candidate <the three kernels>/predictions.csv --report-baseline <default>/report.md --report-candidate <the three kernels>/report.md`. The saved copies in `rba_classifier_runs/` can be used directly, without rerunning |
| `classifier_env` | `compare-classifier --baseline-git "5cbea9a^:reports/rba_classifier_evaluation.md" --candidate reports/reproducibility_runs/rba_classifier_runs/default/predictions.csv` |
| `new_data` | `python -m src.reproducibility_check data-vintages` (reads two commits of the ABS household spending file from git) |

## Checks on this record

- `rba_classifier_runs/default/report.md` is byte-identical to the checked-in `reports/rba_classifier_evaluation.md`: the
  default-kernel rerun reproduced it exactly.
- Both classifier scenarios were recomputed from this folder's saved files and matched the first run.
- `scripts/repro_scratch_copy.sh` and `scripts/repro_rba_predictions.py` were run from the repo copy after being written.
  `scripts/repro_retrain_chain.sh` differs from the version that produced `retrain_newest_steps.log` only in log file
  names and a silenced `kill` error; dry-run checked with a Python stand-in, not repeated end to end.

## Claim audit (2026-09-22)

Every number and label on the chapter 6 chart was checked against the data behind it. Findings and fixes:

- Three notes were too tight: forecast differences reached 1.06e-10 (noted "within 1e-10"), Cloud Run scenario forecasts
  differed by up to 0.0306 pp (noted 0.03), data revisions by up to 0.2348% (noted 0.23%). Every "within"/"up to" is now
  rounded up by the chart code, with a guard cell asserting the bounds.
- The "up to 4.7 pp" probability shift was attributed to the classifier's fits generally; it's only the simulation-based
  threshold model (ordered models moved under 3e-9). Classifier rows now report the two separately.
- RBA numbers in the API comparison mixed probabilities and forecast values; now split (forecast values didn't move, so
  differences are all in probabilities).
- Download dates are 2026-08-24 and 2026-09-03 (UTC, from `dataset/download_manifest.json`), not 2026-08-25.
- "Cloud Run gives one vCPU" and "two images with different package versions gave identical outputs" were unrecorded
  claims; removed or attributed to the README. "Cloud Run answers repeat exactly" is now measured (`cloud_run_repeat`).
- Confirmed: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` gives one thread; the committed RBA report's inputs are unchanged
  since it was made, and the classifier's code changed only in diagnostics/report text (`git diff c3e1bb5`) — not an
  input or code change.
- The four API scenarios were re-measured with raw responses kept and reproduced the first numbers exactly.
- Scenario names now say what was done ("Retrain the models on the newest packages" — the four interval reports weren't
  rerun).
- Not measured: how far data revisions move the forecasts, another physical computer, or the cause of the earlier RBA
  classifier differences.
