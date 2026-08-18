"""AIC/BIC-only SARIMA order search for the cpi_yoy baseline.

This utility is intentionally separate from the walk-forward evaluation
harness. It fits candidate SARIMA specifications and ranks them by in-sample
information criteria only, so reported forecast RMSE/MAE are not used for
model selection.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from src.models.evaluation import CURATED_DATA_PATH, load_target_series
from src.models.sarima import DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER, fit_sarima


def iter_candidate_orders(
    max_p: int = 2,
    max_q: int = 2,
    max_p_seasonal: int = 2,
    max_q_seasonal: int = 2,
    d_values: Iterable[int] = (0,),
    seasonal_d_values: Iterable[int] = (0,),
    seasonal_period: int = 4,
) -> Iterator[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    """Yield SARIMA candidate orders for a compact Box-Jenkins grid."""
    for p in range(max_p + 1):
        for d in d_values:
            for q in range(max_q + 1):
                order = (p, int(d), q)
                for seasonal_p in range(max_p_seasonal + 1):
                    for seasonal_d in seasonal_d_values:
                        for seasonal_q in range(max_q_seasonal + 1):
                            seasonal_order = (
                                seasonal_p,
                                int(seasonal_d),
                                seasonal_q,
                                seasonal_period,
                            )
                            yield order, seasonal_order


def run_order_search(
    series: pd.Series,
    max_p: int = 2,
    max_q: int = 2,
    max_p_seasonal: int = 2,
    max_q_seasonal: int = 2,
    d_values: Iterable[int] = (0,),
    seasonal_d_values: Iterable[int] = (0,),
    seasonal_period: int = 4,
    trend: str = "n",
    maxiter: int = 100,
) -> pd.DataFrame:
    """Fit candidate SARIMA orders and return AIC/BIC-ranked results."""
    rows: list[dict[str, object]] = []
    for order, seasonal_order in iter_candidate_orders(
        max_p=max_p,
        max_q=max_q,
        max_p_seasonal=max_p_seasonal,
        max_q_seasonal=max_q_seasonal,
        d_values=d_values,
        seasonal_d_values=seasonal_d_values,
        seasonal_period=seasonal_period,
    ):
        row: dict[str, object] = {
            "order": order,
            "seasonal_order": seasonal_order,
            "trend": trend,
            "converged": False,
            "aic": float("inf"),
            "bic": float("inf"),
            "error": "",
        }
        try:
            fitted = fit_sarima(
                series=series,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                maxiter=maxiter,
            )
            row.update(
                {
                    "converged": bool(fitted.mle_retvals.get("converged", False)),
                    "aic": float(fitted.aic),
                    "bic": float(fitted.bic),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive CLI path
            row["error"] = str(exc)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["aic", "bic"], ignore_index=True)


def _parse_int_values(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CURATED_DATA_PATH)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--d-values", default="0")
    parser.add_argument("--seasonal-d-values", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--holdout-quarters",
        type=int,
        default=0,
        help="Exclude this many quarters from the tail of the series before "
        "searching, so DEFAULT_ORDER can be re-derived on development-only "
        "data rather than the full sample (mirrors sarimax_order_search.py's "
        "ORDER_SELECTION_HOLDOUT_QUARTERS).",
    )
    args = parser.parse_args(argv)

    series = load_target_series(args.data)
    if args.holdout_quarters > 0:
        series = series.iloc[: -args.holdout_quarters]
    results = run_order_search(
        series=series,
        d_values=_parse_int_values(args.d_values),
        seasonal_d_values=_parse_int_values(args.seasonal_d_values),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)

    print(results.head(args.top).to_string(index=False))
    print(f"\nConfigured default: order={DEFAULT_ORDER}, seasonal_order={DEFAULT_SEASONAL_ORDER}")


if __name__ == "__main__":
    main()
