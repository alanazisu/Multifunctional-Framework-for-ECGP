"""
models/fusion.py
================
Fusion layer that combines embeddings from both streams and produces
final predictions for consumption and solar generation.

The fusion applies a learned soft-weighting:
    z_fused = α * z_spatial + (1-α) * z_temporal    (α ∈ [0,1], learned)
then passes z_fused through a calibration MLP to produce predictions.
"""

import numpy as np
from typing import Tuple


def _relu(x): return np.maximum(0.0, x)


class FusionLayer:
    """
    Fuse spatial and temporal embeddings → [consumption, solar] predictions.

    Parameters
    ----------
    spatial_dim  : int  — dimension of spatial embedding
    temporal_dim : int  — dimension of temporal embedding
    hidden_dims  : list — widths of intermediate fusion MLP layers
    n_outputs    : int  — number of prediction targets (2 = consumption + solar)
    dropout      : float
    seed         : int
    """

    def __init__(self, spatial_dim: int, temporal_dim: int,
                 hidden_dims: list, n_outputs: int = 2,
                 dropout: float = 0.10, weight_decay: float = 1e-4,
                 seed: int = 42) -> None:
        self.rng         = np.random.default_rng(seed)
        self.spatial_dim = spatial_dim
        self.temporal_dim= temporal_dim
        self.dropout     = dropout
        self.wd          = weight_decay

        # Learnable fusion weights (soft alpha per stream)
        self.alpha = np.array([0.5])     # α for spatial stream

        # Input dim after concatenation
        concat_dim = spatial_dim + temporal_dim

        # Fusion MLP
        dims = [concat_dim] + hidden_dims + [n_outputs]
        self.Ws = []
        self.bs = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.Ws.append(self.rng.standard_normal((dims[i], dims[i+1])) * scale)
            self.bs.append(np.zeros(dims[i+1]))
        self._cache: dict = {}

    def forward(self, z_s: np.ndarray, z_t: np.ndarray,
                training: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        z_s : (batch, spatial_dim)
        z_t : (batch, temporal_dim)

        Returns
        -------
        preds : (batch, 2)  [consumption_kwh, solar_kwh]
        """
        alpha = float(np.clip(self.alpha[0], 0.05, 0.95))

        # Weighted fusion before concat
        # (keeps gradients flowing to both streams)
        z_s_w = alpha * z_s
        z_t_w = (1 - alpha) * z_t
        z_cat = np.concatenate([z_s_w, z_t_w], axis=1)  # (B, S+T)

        out = z_cat.astype(np.float64)
        activations = [out]
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            out = out @ W + b
            if i < len(self.Ws) - 1:   # no activation on final layer
                out = _relu(out)
                if training:
                    mask = (self.rng.uniform(0,1,out.shape) > self.dropout).astype(np.float64)
                    out  = out * mask / (1 - self.dropout + 1e-8)
            activations.append(out)

        # Final activation: softplus to ensure non-negative predictions
        preds = np.log1p(np.exp(out))   # softplus ≈ ReLU but smooth

        self._cache = {"z_s": z_s, "z_t": z_t, "z_cat": z_cat,
                       "activations": activations, "alpha": alpha}
        return preds   # (B, 2)

    def get_params(self) -> dict:
        return {"Ws": self.Ws, "bs": self.bs, "alpha": self.alpha}

    def set_params(self, p: dict) -> None:
        self.Ws    = p["Ws"]
        self.bs    = p["bs"]
        self.alpha = p["alpha"]

    def count_parameters(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.Ws, self.bs)) + 1

    def __repr__(self) -> str:
        return (f"FusionLayer(spatial={self.spatial_dim}, "
                f"temporal={self.temporal_dim}, alpha={float(self.alpha):.3f})")
