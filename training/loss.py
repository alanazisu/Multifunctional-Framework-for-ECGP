"""
training/loss.py
================
Loss functions for energy prediction training.
"""
import numpy as np


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> tuple:
    diff = y_pred - y_true
    loss = 0.5 * (diff ** 2).mean()
    grad = diff / y_true.shape[0]
    return loss, grad


def mae_loss(y_pred: np.ndarray, y_true: np.ndarray) -> tuple:
    diff = y_pred - y_true
    loss = np.abs(diff).mean()
    grad = np.sign(diff) / y_true.shape[0]
    return loss, grad


def huber_loss(y_pred: np.ndarray, y_true: np.ndarray,
               delta: float = 1.0) -> tuple:
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    quadratic = np.minimum(abs_diff, delta)
    linear    = abs_diff - quadratic
    loss      = (0.5 * quadratic ** 2 + delta * linear).mean()
    grad      = np.where(abs_diff <= delta, diff, delta * np.sign(diff))
    grad     /= y_true.shape[0]
    return loss, grad


LOSSES = {"mse": mse_loss, "mae": mae_loss, "huber": huber_loss}
