"""
evaluation/metrics.py
======================
All evaluation metrics for energy prediction.

Metrics:
  MSE   — Mean Squared Error
  RMSE  — Root Mean Squared Error
  MAE   — Mean Absolute Error
  MAPE  — Mean Absolute Percentage Error
  R²    — Coefficient of Determination
  SMAPE — Symmetric MAPE
  CVRMSE— Coefficient of Variation RMSE
"""

import numpy as np
from typing import Dict


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray,
         eps: float = 1e-6) -> float:
    """Mean Absolute Percentage Error (%)."""
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask])
                                 / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray,
          eps: float = 1e-6) -> float:
    """Symmetric MAPE (%)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-10:
        return 1.0
    return float(1 - ss_res / ss_tot)


def cvrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Variation RMSE (%)."""
    mean_true = y_true.mean()
    if abs(mean_true) < 1e-6:
        return 0.0
    return float(rmse(y_true, y_pred) / mean_true * 100)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray,
                target_name: str = "") -> Dict[str, float]:
    """
    Compute the full metric suite for one target.

    Returns
    -------
    dict with keys: MSE, RMSE, MAE, MAPE, SMAPE, R2, CVRMSE
    """
    return {
        "Target":  target_name,
        "MSE":     mse(y_true, y_pred),
        "RMSE":    rmse(y_true, y_pred),
        "MAE":     mae(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred),
        "SMAPE(%)":smape(y_true, y_pred),
        "R²":      r2_score(y_true, y_pred),
        "CVRMSE(%)":cvrmse(y_true, y_pred),
    }


def format_metrics_table(metrics_list: list, title: str = "") -> str:
    """Pretty-print a list of metric dicts as a formatted table."""
    if not metrics_list:
        return ""
    cols = list(metrics_list[0].keys())
    widths = {c: max(len(c), max(len(f"{r.get(c,''):.4f}" if isinstance(r.get(c), float) else str(r.get(c,'')))
                                  for r in metrics_list))
              for c in cols}
    sep  = "─" * (sum(widths.values()) + 3 * len(widths) + 1)
    hdr  = " │ ".join(f"{c:>{widths[c]}}" for c in cols)
    rows = []
    for r in metrics_list:
        row = " │ ".join(
            f"{v:.4f}".rjust(widths[c]) if isinstance(v, float) else str(v).rjust(widths[c])
            for c, v in r.items()
        )
        rows.append(row)
    lines = [sep, f"  {title}", sep, hdr, sep] + rows + [sep]
    return "\n".join(lines)
