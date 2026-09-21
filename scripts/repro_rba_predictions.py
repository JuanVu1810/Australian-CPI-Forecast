"""Run the RBA classifier walk-forward evaluation and keep its predictions table, for reproducibility comparisons.

usage: python scripts/repro_rba_predictions.py <output dir> [repo dir]

Writes <output dir>/predictions.csv (every model, every test quarter) and <output dir>/report.md. The checked-in report
is not touched. Run it under different OPENBLAS_CORETYPE values to imitate other CPUs, then compare the tables with
`python -m src.reproducibility_check compare-classifier`. It takes about 10 minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
out = Path(sys.argv[1])
repo = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from src.models import rba_classifier as r  # noqa: E402

out.mkdir(parents=True, exist_ok=True)
_, predictions, _, _ = r.run_rba_classifier_evaluation(
    headline_path=r.HEADLINE_BACKTEST_PATH,
    trimmed_mean_path=r.TRIMMED_MEAN_BACKTEST_PATH,
    macro_path=r.CURATED_MACRO_PATH,
    output_path=out / "report.md",
    candidate_initial_train_sizes=r.DEFAULT_CANDIDATE_INITIAL_TRAIN_SIZES,
    min_initial_train_size=r.DEFAULT_MIN_INITIAL_TRAIN_SIZE,
)
predictions.to_csv(out / "predictions.csv", index=False)
print(f"wrote {len(predictions)} predictions to {out}")
