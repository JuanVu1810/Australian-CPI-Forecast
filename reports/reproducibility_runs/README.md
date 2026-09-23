# Reproducibility runs: what was measured, how to repeat it, and what was kept

[![content: AI-generated](../../book/australian_cpi_forecasting/_static/badges/ai-generated.svg)](https://juanvu1810.github.io/Australian-CPI-Forecast/intro.html#ai-acknowledgement)

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

## Repeating the audit

These are commands for repeating the recorded audit, not the shortest way to run the project. For ordinary setup,
training and serving, use the repository's main README instead.

### Before running the commands

- Run them from a Linux or WSL Bash shell with Python 3.11. The commands were not tested in native Windows PowerShell.
- Use an empty scratch directory outside the repository. Retraining writes to `data/`, `reports/` and `mlruns/`; never
  pass the real checkout to `scripts/repro_retrain_chain.sh`.
- The `compare-*` commands intentionally add or replace rows in the checked-in
  `reports/reproducibility_check.csv` and `reports/reproducibility_check_files.csv`. Run them on a clean branch or in a
  disposable clone, inspect `git diff` afterwards, and commit the changes only when deliberately refreshing the audit.
- Commands that start APIs occupy a terminal until stopped with `Ctrl+C`. Re-export `REPO_ROOT` and `AUDIT_ROOT` in
  every new terminal, and choose different ports if 8101, 8201, 8202 or 8203 are already in use.
- The retraining scenario takes about one hour and its scratch files can use several gigabytes. The four interval
  coverage and calibration jobs are deliberately excluded; running those separately adds about 90 minutes.
- Do not run `data_retrieval.py` as part of these scenarios. Revised upstream data would test a different question.
- CPU-kernel settings imitate different arithmetic paths on one machine. They do not replace a test on another
  physical computer.

Start in the repository root and choose a new empty location. `mkdir` should fail if that location already exists;
choose another path instead of deleting an uncertain directory.

```bash
cd /absolute/path/to/CPI-Forecast
export REPO_ROOT="$(pwd)"
export AUDIT_ROOT=/tmp/cpi-repro-audit
mkdir "$AUDIT_ROOT"

python3.11 -m venv "$AUDIT_ROOT/venv_pinned"
source "$AUDIT_ROOT/venv_pinned/bin/activate"
python -m pip install -r requirements.txt -c constraints.txt
```

If the project dependencies are already installed in a Python 3.11 environment, activate that instead of creating
`venv_pinned`. In every additional terminal, activate the same environment with
`source "$AUDIT_ROOT/venv_pinned/bin/activate"` before running `python`, `uvicorn` or a project script.

The descriptive `--scenario`, `--environment` and `--how` values are stored beside the measurements. They do not
change the comparison itself. The commands below use concise descriptions, so those text fields can differ from the
historical 2026-09-22 rows even when the measured numbers match.

### Rebuild the book twice (`book_rebuild`)

This compares built HTML only; it does not re-execute the notebooks.

```bash
cd "$REPO_ROOT"
python3.11 -m venv "$AUDIT_ROOT/venv_book"
"$AUDIT_ROOT/venv_book/bin/python" -m pip install -r book/requirements.txt
"$AUDIT_ROOT/venv_book/bin/jupyter-book" build \
  book/australian_cpi_forecasting --path-output "$AUDIT_ROOT/book_1"
"$AUDIT_ROOT/venv_book/bin/jupyter-book" build \
  book/australian_cpi_forecasting --path-output "$AUDIT_ROOT/book_2"

python -m src.reproducibility_check compare-files \
  --scenario-id book_rebuild \
  --scenario "Rebuild the book" \
  --environment "same machine; two builds into empty folders; notebooks not re-executed" \
  --how "Built the Jupyter Book twice and compared every generated HTML page byte for byte." \
  --baseline "$AUDIT_ROOT/book_1/_build/html" \
  --candidate "$AUDIT_ROOT/book_2/_build/html" \
  --glob "**/*.html"
```

### Restore the saved models and regenerate Tableau exports (`restore`)

Create the scratch copy first:

```bash
cd "$REPO_ROOT"
scripts/repro_scratch_copy.sh "$AUDIT_ROOT/repo_restore"
cd "$AUDIT_ROOT/repo_restore"
python -m src.mlruns_snapshot restore
```

In terminal 1, serve that copy and leave it running. Wait until `/health` responds before continuing.

```bash
export AUDIT_ROOT=/tmp/cpi-repro-audit
source "$AUDIT_ROOT/venv_pinned/bin/activate"
cd "$AUDIT_ROOT/repo_restore"
python -m uvicorn api.main:app --port 8201
```

In terminal 2, set `REPO_ROOT` to the real checkout, regenerate the exports in the scratch copy, and compare them:

```bash
cd /absolute/path/to/CPI-Forecast
export REPO_ROOT="$(pwd)"
export AUDIT_ROOT=/tmp/cpi-repro-audit
source "$AUDIT_ROOT/venv_pinned/bin/activate"

cd "$AUDIT_ROOT/repo_restore"
touch .chain_start
python -m src.models.tableau_export --api-base-url http://localhost:8201

cd "$REPO_ROOT"
python -m src.reproducibility_check compare-files \
  --scenario-id restore \
  --scenario "Restore the saved model runs" \
  --environment "same machine; pinned packages; models restored from mlruns_snapshot" \
  --how "Restored the saved runs in a scratch copy, served them locally, regenerated Tableau exports and compared them with the checked-in files." \
  --baseline "$REPO_ROOT" \
  --candidate "$AUDIT_ROOT/repo_restore" \
  --glob "reports/tableau/*.csv" \
  --newer-than "$AUDIT_ROOT/repo_restore/.chain_start"
```

Keep the server on port 8201 running if continuing with the API comparisons below.

### Retrain with the newest available packages (`retrain_new`)

This intentionally installs `requirements.txt` without `constraints.txt`; it measures dependency-version drift rather
than reproducing the pinned environment. Package versions available on a later date may differ from those in the
recorded 2026-09-22 run.

```bash
cd "$REPO_ROOT"
python3.11 -m venv "$AUDIT_ROOT/venv_new"
"$AUDIT_ROOT/venv_new/bin/python" -m pip install -r requirements.txt
"$AUDIT_ROOT/venv_new/bin/python" -m pip list --format=freeze > "$AUDIT_ROOT/environment_newest.txt"

scripts/repro_scratch_copy.sh "$AUDIT_ROOT/repo_new"
set -o pipefail
scripts/repro_retrain_chain.sh \
  "$AUDIT_ROOT/repo_new" \
  "$AUDIT_ROOT/venv_new/bin/python" \
  8101 2>&1 | tee "$AUDIT_ROOT/retrain_newest_steps.log"
if [ -e "$AUDIT_ROOT/repo_new/.chain_failed" ]; then
  echo "A retraining step failed; inspect .chain_failed and log_*.txt before comparing."
  exit 1
fi

python -m src.reproducibility_check compare-files \
  --scenario-id retrain_new \
  --scenario "Retrain the models on the newest packages" \
  --environment "same machine; fresh virtual environment installed without constraints.txt" \
  --how "Created a scratch copy, retrained and regenerated the standard outputs with unpinned packages, then compared every file rewritten by the chain." \
  --baseline "$REPO_ROOT" \
  --candidate "$AUDIT_ROOT/repo_new" \
  --glob "reports/*.csv" "reports/*.md" "reports/tableau/*.csv" "data/curated/*.csv" \
  --newer-than "$AUDIT_ROOT/repo_new/.chain_start"
```

The chain writes detailed `log_*.txt` files inside `$AUDIT_ROOT/repo_new`. A `.chain_failed` file means at least one
step failed even if later steps continued; do not interpret that run as a successful comparison.

### Compare one-thread and alternative-CPU API results (`one_thread`, `other_cpu`)

These commands require the restored baseline API on port 8201 from the `restore` scenario.

In terminal 2, start the one-thread candidate and leave it running:

```bash
export AUDIT_ROOT=/tmp/cpi-repro-audit
source "$AUDIT_ROOT/venv_pinned/bin/activate"
cd "$AUDIT_ROOT/repo_restore"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m uvicorn api.main:app --port 8203
```

In another terminal, run the comparison from the real checkout:

```bash
cd /absolute/path/to/CPI-Forecast
export REPO_ROOT="$(pwd)"
export AUDIT_ROOT=/tmp/cpi-repro-audit
source "$AUDIT_ROOT/venv_pinned/bin/activate"

python -m src.reproducibility_check compare-api \
  --scenario-id one_thread \
  --scenario "Serve the same models on one CPU thread" \
  --environment "same machine; OpenBLAS and OpenMP limited to one thread; restored saved runs" \
  --how "Sent the same five requests to the normal restored API and a one-thread API, then compared every returned number and label." \
  --baseline-url http://localhost:8201 \
  --candidate-url http://localhost:8203 \
  --save-dir "$AUDIT_ROOT/api_responses"
```

Stop the port-8203 server, then start the alternative OpenBLAS kernel in that terminal:

```bash
export AUDIT_ROOT=/tmp/cpi-repro-audit
source "$AUDIT_ROOT/venv_pinned/bin/activate"
cd "$AUDIT_ROOT/repo_restore"
OPENBLAS_CORETYPE=SANDYBRIDGE python -m uvicorn api.main:app --port 8202
```

Run its comparison from the real checkout:

```bash
cd /absolute/path/to/CPI-Forecast
export REPO_ROOT="$(pwd)"
export AUDIT_ROOT=/tmp/cpi-repro-audit
source "$AUDIT_ROOT/venv_pinned/bin/activate"

python -m src.reproducibility_check compare-api \
  --scenario-id other_cpu \
  --scenario "Serve the same models on other CPU arithmetic" \
  --environment "same machine; OpenBLAS forced to the Sandy Bridge kernel; restored saved runs" \
  --how "Sent the same five requests to the normal restored API and a Sandy Bridge OpenBLAS API, then compared every returned number and label." \
  --baseline-url http://localhost:8201 \
  --candidate-url http://localhost:8202 \
  --save-dir "$AUDIT_ROOT/api_responses"
```

The raw responses go to the scratch directory. To refresh the checked-in evidence deliberately, use
`--save-dir "$REPO_ROOT/reports/reproducibility_runs/api_responses"` instead.

### Compare the local API with Cloud Run (`cloud_run`, `cloud_run_repeat`)

The first command requires the restored local API on port 8201. Both commands send the same read-only requests used by
the Streamlit demo to the public service. Each run can take several minutes and depends on network access and the live
deployment still being available.

```bash
cd /absolute/path/to/CPI-Forecast
export REPO_ROOT="$(pwd)"
export AUDIT_ROOT=/tmp/cpi-repro-audit
source "$AUDIT_ROOT/venv_pinned/bin/activate"
export CLOUD_API=https://cpi-forecast-api-887232555982.asia-southeast1.run.app

python -m src.reproducibility_check compare-api \
  --scenario-id cloud_run \
  --scenario "Serve from Cloud Run instead of locally" \
  --environment "live Cloud Run service versus the local API on restored saved runs" \
  --how "Sent the same five read-only requests to the restored local API and the live Cloud Run service." \
  --baseline-url http://localhost:8201 \
  --candidate-url "$CLOUD_API" \
  --save-dir "$AUDIT_ROOT/api_responses"

python -m src.reproducibility_check compare-api \
  --scenario-id cloud_run_repeat \
  --scenario "Ask Cloud Run the same question twice" \
  --environment "live Cloud Run service; two identical sets of requests" \
  --how "Sent the same five read-only requests to the live service twice and compared the two responses." \
  --baseline-url "$CLOUD_API" \
  --candidate-url "$CLOUD_API" \
  --save-dir "$AUDIT_ROOT/api_responses"
```

### Rerun the classifier under other CPU kernels (`classifier_cpu`)

Run these with the same installed Python environment. The default run and each alternative kernel take about 10
minutes. `OPENBLAS_CORETYPE` changes the numerical kernel; it does not emulate a complete operating system or machine.

```bash
cd "$REPO_ROOT"
python scripts/repro_rba_predictions.py "$AUDIT_ROOT/rba_default"
OPENBLAS_CORETYPE=SANDYBRIDGE python scripts/repro_rba_predictions.py "$AUDIT_ROOT/rba_sandybridge"
OPENBLAS_CORETYPE=NEHALEM python scripts/repro_rba_predictions.py "$AUDIT_ROOT/rba_nehalem"
OPENBLAS_CORETYPE=PRESCOTT python scripts/repro_rba_predictions.py "$AUDIT_ROOT/rba_prescott"

python -m src.reproducibility_check compare-classifier \
  --scenario-id classifier_cpu \
  --scenario "Rerun the RBA classifier on other CPU arithmetic" \
  --environment "same machine; three alternative OpenBLAS CPU kernels versus the default kernel" \
  --how "Reran every classifier under the default, Sandy Bridge, Nehalem and Prescott OpenBLAS kernels and compared calls, probabilities and ordered-model cut-offs." \
  --baseline "$AUDIT_ROOT/rba_default/predictions.csv" \
  --candidate \
    "$AUDIT_ROOT/rba_sandybridge/predictions.csv" \
    "$AUDIT_ROOT/rba_nehalem/predictions.csv" \
    "$AUDIT_ROOT/rba_prescott/predictions.csv" \
  --report-baseline "$AUDIT_ROOT/rba_default/report.md" \
  --report-candidate \
    "$AUDIT_ROOT/rba_sandybridge/report.md" \
    "$AUDIT_ROOT/rba_nehalem/report.md" \
    "$AUDIT_ROOT/rba_prescott/report.md"
```

### Compare with the older classifier report (`classifier_env`)

This requires the repository's Git history and the default classifier run created above. The older run's environment
was not recorded, so this scenario detects a difference but cannot prove that the environment caused it.

```bash
cd "$REPO_ROOT"
python -m src.reproducibility_check compare-classifier \
  --scenario-id classifier_env \
  --scenario "Rerun the RBA classifier against the earlier committed report" \
  --environment "older committed report with unrecorded environment versus the current default-kernel rerun" \
  --how "Extracted predictions from the older committed report and compared them with the new default-kernel predictions." \
  --baseline-git "5cbea9a^:reports/rba_classifier_evaluation.md" \
  --candidate "$AUDIT_ROOT/rba_default/predictions.csv"
```

### Compare two saved data vintages (`new_data`)

This reads files already present in Git history; it does not download new data. It must run from a clone that contains
the referenced commits.

```bash
cd "$REPO_ROOT"
python -m src.reproducibility_check data-vintages \
  --scenario-id new_data \
  --scenario "Download the data again" \
  --environment "two saved ABS household-spending downloads from Git history" \
  --how "Read the 2026-08-24 and 2026-09-03 household-spending vintages from Git history and compared values by month."
```

### After the run

Stop any remaining API processes with `Ctrl+C`, then inspect the two generated measurement files and the working-tree
changes:

```bash
cd "$REPO_ROOT"
git diff -- reports/reproducibility_check.csv reports/reproducibility_check_files.csv
git status --short
```

The environment, machine and code-version text files in this folder were captured and annotated manually around the
original run. `pip list --format=freeze`, `python --version`, `git rev-parse HEAD`, `git status --short`, `uname -a` and
`lscpu` provide the raw information for a new audit, but they do not recreate those annotated files automatically.

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
