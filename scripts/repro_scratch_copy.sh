#!/bin/bash
# Copy the repo without git history, the book and notebooks, so reruns cannot touch the real reports or models.
#
# usage: scripts/repro_scratch_copy.sh <destination dir>
#
# Afterwards, either use the saved models already in the copy's models/ folder or retrain with
# scripts/repro_retrain_chain.sh. `python -m src.reproducibility_check` then compares the copy against this checkout.
set -eu
cd "$(dirname "$0")/.."
rsync -a --exclude .git --exclude __pycache__ --exclude book --exclude docs --exclude notebooks \
  --exclude .pytest_cache ./ "$1/"
echo "copied to $1"
