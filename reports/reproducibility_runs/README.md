# Reproducibility runs: what was measured, how to repeat it, and what was kept

`reports/reproducibility_check.csv` holds the results (one row per scenario and quantity) and
`reports/reproducibility_check_files.csv` the per-file detail. This folder holds what is needed to repeat and audit them.
All runs were made on 2026-09-22 (Australian time) on one machine. Reruns always happen in scratch copies of the repo, never
in the checkout, because the model scripts write into `reports/` and `data/`.

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

- The regenerated files, retrained MLflow runs and scratch repos (deleted; about 2.8 GB). The file-level results survive as
  the counts and differences in the two CSVs. Repeat a scenario to get the files back.
- Timings of the API and classifier scenarios.
- The per-step logs of the retrain (only the timing log above) and the pip output of the first install.
- The original fresh venv (see `environment_newest.txt`) and the older classifier report that `classifier_env` compares
  against (it is in git: `git show 5cbea9a^:reports/rba_classifier_evaluation.md`).

## Repeating each scenario

`S` is any empty folder outside the repo. Run `scripts/repro_scratch_copy.sh $S/repo` first for the scenarios that
need a copy. `compare-*` commands are run from the real checkout and add or replace rows in the two CSVs. Every
`compare-*` command also needs `--scenario-id`, `--scenario`, `--environment` and `--how`; use the values in the CSV.

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

- `rba_classifier_runs/default/report.md` is byte-identical to the checked-in `reports/rba_classifier_evaluation.md`, so
  the default-kernel rerun reproduced the checked-in report exactly.
- The two classifier scenarios were recomputed from the saved files in this folder and gave the same numbers as the first run.
- `scripts/repro_scratch_copy.sh` and `scripts/repro_rba_predictions.py` were run from the repo copy after being written.
  `scripts/repro_retrain_chain.sh` differs from the version that produced `retrain_newest_steps.log` only in log file
  names and a silenced `kill` error; it was dry-run with a stand-in for Python to check its steps, markers and logs, but not
  repeated end to end.

## Claim audit (2026-09-22)

Every number and label on the chapter 6 chart was checked against the data behind it. What that found and fixed:

- Three notes were slightly too tight. Forecast differences reached 1.06e-10 (the note said "within 1e-10"), the Cloud
  Run scenario forecast differed by up to 0.0306 pp (said 0.03) and the data revisions by up to 0.2348% (said 0.23%). Every
  "within" and "up to" is now rounded up by the chart code, and the guard cell asserts the bounds.
- The "up to 4.7 pp" probability shift had been described as if it came from the classifier's fits. It comes only from the
  simulation-based threshold model; the ordered models' probabilities moved by under 3e-9. The classifier rows now report
  the two separately.
- The RBA numbers in the API comparison mix probabilities and forecast values; they are now split (the forecast values did
  not move, so the differences are all in the probabilities).
- The download dates are 2026-08-24 and 2026-09-03 (UTC, from `dataset/download_manifest.json`), not 2026-08-25.
- "Cloud Run gives one vCPU" and "two images with different package versions gave identical outputs" were not recorded
  anywhere, so they were removed or marked as the README's statement. "Cloud Run answers repeat exactly" is now measured
  (`cloud_run_repeat`).
- Checked and confirmed: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` really gives one thread; the earlier committed RBA report's
  inputs (backtest predictions, curated data) are unchanged since it was made and the classifier's prediction code changed
  only in diagnostics and report text (`git diff c3e1bb5`), so the 4 changed calls are not an input or code change.
- The four API scenarios were measured a second time with raw responses kept, and reproduced the first numbers exactly.
- The scenario names now say what was done ("Retrain the models on the newest packages", since the four interval reports
  were not rerun).
- Not measured: how far the data revisions move the forecasts, another physical computer, and what caused the earlier
  RBA classifier differences.
