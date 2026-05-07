"""
evaluation/evaluator.py
========================
Multi-granularity evaluation of Energy-TSN predictions.

Supports: minutely, hourly, daily, weekly, monthly granularities.
Produces:
  - Per-granularity metric tables (CSV + console)
  - Actual vs Predicted time-series plots
  - Error distribution plots
  - Scatter plots
  - Residual plots
"""

import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import compute_all, format_metrics_table


class MultiGranularityEvaluator:
    """
    Evaluate predictions at multiple time granularities.

    Parameters
    ----------
    timestamps   : pd.DatetimeIndex
    y_true       : (N, 2) — [consumption, solar]
    y_pred       : (N, 2) — [consumption, solar]
    result_dir   : str
    """

    GRANULARITIES = {
        "minutely": None,           # raw resolution (assumed hourly here → kept as-is)
        "hourly":   "h",
        "daily":    "D",
        "weekly":   "W",
        "monthly":  "ME",
    }

    TARGET_NAMES = ["Consumption (kWh)", "Solar Generation (kWh)"]

    def __init__(self, timestamps: pd.DatetimeIndex,
                 y_true: np.ndarray, y_pred: np.ndarray,
                 result_dir: str) -> None:
        self.ts        = timestamps
        self.y_true    = y_true
        self.y_pred    = y_pred
        self.result_dir= result_dir
        os.makedirs(result_dir, exist_ok=True)

        # Build base DataFrame
        self.df = pd.DataFrame({
            "timestamp":         timestamps,
            "consumption_true":  y_true[:, 0],
            "consumption_pred":  y_pred[:, 0],
            "solar_true":        y_true[:, 1],
            "solar_pred":        y_pred[:, 1],
        }).set_index("timestamp")

    # ── Resample helpers ──────────────────────────────────────────────────

    def _resample(self, freq: Optional[str]) -> pd.DataFrame:
        if freq is None:
            return self.df.copy()
        return self.df.resample(freq).sum()

    # ── Evaluate one granularity ──────────────────────────────────────────

    def evaluate_granularity(self, granularity: str) -> List[Dict]:
        freq  = self.GRANULARITIES.get(granularity)
        data  = self._resample(freq)

        results = []
        for i, name in enumerate(self.TARGET_NAMES):
            col_true = "consumption_true" if i == 0 else "solar_true"
            col_pred = "consumption_pred" if i == 0 else "solar_pred"
            yt = data[col_true].values
            yp = data[col_pred].values
            m  = compute_all(yt, yp, target_name=name)
            m["Granularity"] = granularity
            m["N_samples"]   = len(yt)
            results.append(m)
        return results

    # ── Run all granularities ─────────────────────────────────────────────

    def run_all(self, verbose: bool = True) -> Dict[str, List[Dict]]:
        all_results = {}
        all_rows    = []

        for gran in self.GRANULARITIES:
            res = self.evaluate_granularity(gran)
            all_results[gran] = res

            if verbose:
                print(f"\n{'─'*60}")
                print(f"  Granularity: {gran.upper()}")
                print(format_metrics_table(res, title=gran.upper()))

            for r in res:
                all_rows.append(r)

        # Save full CSV
        out_csv = os.path.join(self.result_dir, "metrics_all_granularities.csv")
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        if verbose:
            print(f"\n  Full metrics saved → {out_csv}")

        # Save per-granularity CSVs
        for gran, res in all_results.items():
            csv_path = os.path.join(self.result_dir, f"metrics_{gran}.csv")
            pd.DataFrame(res).to_csv(csv_path, index=False)

        return all_results

    # ── Resampled DataFrames for plotting ─────────────────────────────────

    def get_resampled(self, granularity: str) -> pd.DataFrame:
        freq = self.GRANULARITIES.get(granularity)
        return self._resample(freq)
