"""Compare the earlier eleven-term Elastic Net feature sets with the five-term screen-informed sets, origin by origin.

usage: python scripts/compare_elastic_net_feature_sets.py [output csv]

Runs four walk-forward backtests (headline and trimmed mean, old and new feature set; about 7 minutes each) and scores
both sets on the forecast origins they share, because the smaller set also lets the backtest start earlier. The old sets
are written out here because the code no longer carries them. Writes reports/elastic_net_feature_set_comparison.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import elastic_net as en, svar  # noqa: E402
from src.models.evaluation import CURATED_DATA_PATH, load_target_series, walk_forward_backtest_direct_multihorizon  # noqa: E402

OLD_MACRO = (
    "cash_rate_change_lag1", "unemployment_rate_change_lag1", "inflation_expectations_business_lag1", "ppi_growth_lag2",
    "commodity_growth_lag1", "wti_growth_lag1", "ppi_growth_lag2_sq", "wti_growth_lag1_sq",
    "cash_rate_change_lag1_x_unemployment_rate_change_lag1",
)
OLD_TRIMMED_MACRO = (
    "cash_rate_change_lag1", "unemployment_rate_change_lag1", "inflation_expectations_business_lag1", "ppi_growth_lag1",
    "commodity_growth_lag1", "wti_growth_lag1", "commodity_growth_lag1_sq", "wti_growth_lag1_sq",
    "cash_rate_change_lag1_x_unemployment_rate_change_lag1",
)
SETS = {
    "headline": (en.TARGET_COLUMN, ("cpi_yoy_lag1", "cpi_yoy_lag4", *OLD_MACRO), en.ELASTIC_NET_FEATURE_COLUMNS),
    "trimmed_mean": (en.TRIMMED_MEAN_TARGET_COLUMN, ("trimmed_mean_cpi_yoy_lag1", "trimmed_mean_cpi_yoy_lag4", *OLD_TRIMMED_MACRO),
                     en.TRIMMED_MEAN_ELASTIC_NET_FEATURE_COLUMNS),
}
KEY = ["forecast_origin", "horizon"]


def backtest(target_column: str, features: tuple[str, ...]) -> pd.DataFrame:
    series = load_target_series(CURATED_DATA_PATH, target_column=target_column, max_quarter=svar.FORECAST_ORIGIN_PIN)
    exog = en.load_elastic_net_feature_frame(CURATED_DATA_PATH, feature_columns=features, max_quarter=svar.FORECAST_ORIGIN_PIN)
    return walk_forward_backtest_direct_multihorizon(
        series=series,
        exog=exog,
        forecast_func=lambda frame, steps: en.forecast_elastic_net_direct(
            frame, steps=steps, seed=en.DEFAULT_SEED, feature_columns=features, target_column=target_column
        ),
        initial_train_size=en.DEFAULT_INITIAL_TRAIN_SIZE,
        horizons=en.DEFAULT_HORIZONS,
        model_name="elastic_net",
        target_column=target_column,
    )


def compare(target: str, old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """RMSE of both sets on the origins they share, by horizon and overall."""
    both = old.merge(new, on=KEY, suffixes=("_old", "_new"))
    rows = []
    for horizon, group in [*both.groupby("horizon"), ("overall", both)]:
        old_rmse = float(np.sqrt(np.mean(group["error_old"] ** 2)))
        new_rmse = float(np.sqrt(np.mean(group["error_new"] ** 2)))
        rows.append({
            "target": target,
            "horizon": horizon,
            "shared_origins": int(group["forecast_origin"].nunique()),
            "n": len(group),
            "rmse_eleven_term_set": round(old_rmse, 6),
            "rmse_five_term_set": round(new_rmse, 6),
            "change_pct": round((new_rmse / old_rmse - 1) * 100, 3),
            "origins_new_set_only": int(new["forecast_origin"].nunique() - group["forecast_origin"].nunique()),
            "how": "same walk-forward code, seed and data; scored only on origins both sets can forecast",
        })
    return pd.DataFrame(rows)


def main(output: Path) -> None:
    frames = []
    for target, (column, old_features, new_features) in SETS.items():
        frames.append(compare(target, backtest(column, old_features), backtest(column, new_features)))
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "reports/elastic_net_feature_set_comparison.csv")
