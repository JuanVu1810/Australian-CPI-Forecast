#!/bin/bash
# Copy the repo without git history, model runs, the book and notebooks, so reruns cannot touch the real reports.
#
# usage: scripts/repro_scratch_copy.sh <destination dir>
#
# Afterwards, either restore the saved runs (python -m src.mlruns_snapshot restore, inside the copy) or retrain with
# scripts/repro_retrain_chain.sh. `python -m src.reproducibility_check` then compares the copy against this checkout.
set -eu
cd "$(dirname "$0")/.."
rsync -a --exclude .git --exclude mlruns --exclude __pycache__ --exclude book --exclude docs --exclude notebooks \
  --exclude .pytest_cache ./ "$1/"
echo "copied to $1"
