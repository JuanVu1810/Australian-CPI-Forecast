#!/bin/bash
# Retrain every model and regenerate the reports, in a SCRATCH COPY of the repo (see repro_scratch_copy.sh).
# The model scripts write into reports/ and data/, so never point this at your real checkout.
#
# usage: scripts/repro_retrain_chain.sh <scratch repo dir> <python> <api port>
#
# Environment variables such as OPENBLAS_CORETYPE or OPENBLAS_NUM_THREADS are inherited, so the same script imitates
# other hardware. It runs the README's training commands, then model_comparison --no-rba, rba_classifier,
# simulation_fan, eda_export and svar_unemployment_export, then starts the API on <api port> and runs tableau_export.
# The four interval coverage and calibration reports (about 90 minutes) are NOT run.
#
# Every file the run rewrites is newer than <scratch>/.chain_start, which is how
# `python -m src.reproducibility_check compare-files --newer-than` finds them. One log per step is written next to the
# repo as log_NN_<step>.txt; the step timings and exit codes go to stdout.
set -u
cd "$1" || exit 2
PY="$2"
PORT="$3"
export MLFLOW_DISABLE_AGENT_HINT=1
rm -f .chain_done .chain_failed
touch .chain_start
sleep 1

step=0
run() {
  step=$((step + 1))
  local name
  name=$(printf '%02d' "$step")_$(echo "${*:2}" | tr ' /' '__' | cut -c1-50)
  echo "[$(date +%H:%M:%S)] START ${*:2}"
  "$@" > "log_${name}.txt" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] END rc=$rc ${*:2}"
  if [ $rc -ne 0 ]; then echo "${*:2}" > .chain_failed; fi
}

run "$PY" -m src.build_curated_dataset
run "$PY" -m src.platform_status
run "$PY" -m src.models.evaluation
run "$PY" -m src.models.elastic_net
run "$PY" -m src.models.ensemble
run "$PY" -m src.models.model_comparison --target trimmed_mean
run "$PY" -m src.models.ensemble --target trimmed_mean
run "$PY" -m src.models.model_comparison --no-rba
run "$PY" -m src.models.rba_classifier
run "$PY" -m src.models.simulation_fan
run "$PY" -m src.eda_export
run "$PY" -m src.models.svar_unemployment_export

"$PY" -m uvicorn api.main:app --port "$PORT" > log_api.txt 2>&1 &
API=$!
for _ in $(seq 1 60); do curl -s "localhost:$PORT/health" > /dev/null && break; sleep 2; done
run "$PY" -m src.models.tableau_export --api-base-url "http://localhost:$PORT"
kill "$API" 2> /dev/null || true

touch .chain_done
echo "[$(date +%H:%M:%S)] CHAIN DONE"
