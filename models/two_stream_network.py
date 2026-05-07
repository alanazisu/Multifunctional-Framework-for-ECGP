"""
models/two_stream_network.py
=============================
Top-level Two-Stream Network (TSN) wrapper.

Combines EnergySpatialNetwork + EnergyTemporalNetwork + FusionLayer
into a single model interface with unified forward(), predict(), and
parameter serialisation.
"""

import numpy as np
import os
import pickle
from typing import Tuple, Optional

from models.spatial_network  import EnergySpatialNetwork
from models.temporal_network import EnergyTemporalNetwork
from models.fusion           import FusionLayer


class TwoStreamNetwork:
    """
    Energy-TSN: Two-Stream Network for energy prediction.

    Parameters
    ----------
    spatial_input_dim  : int   — number of spatial (flat) features
    temporal_input_dim : int   — features per time step for temporal stream
    seq_len            : int   — look-back window length
    spatial_cfg        : dict  — kwargs for EnergySpatialNetwork
    temporal_cfg       : dict  — kwargs for EnergyTemporalNetwork
    fusion_cfg         : dict  — kwargs for FusionLayer
    seed               : int
    """

    def __init__(self,
                 spatial_input_dim:  int,
                 temporal_input_dim: int,
                 seq_len:            int,
                 spatial_cfg:        dict,
                 temporal_cfg:       dict,
                 fusion_cfg:         dict,
                 seed:               int = 42) -> None:

        self.spatial_net  = EnergySpatialNetwork(
            input_dim   = spatial_input_dim,
            hidden_dims = spatial_cfg["hidden_dims"],
            output_dim  = spatial_cfg["output_dim"],
            dropout     = spatial_cfg["dropout"],
            activation  = spatial_cfg["activation"],
            weight_decay= spatial_cfg.get("weight_decay", 1e-4),
            seed        = seed,
        )

        self.temporal_net = EnergyTemporalNetwork(
            input_dim    = temporal_input_dim,
            seq_len      = seq_len,
            conv_filters = temporal_cfg["conv_filters"],
            kernel_size  = temporal_cfg["kernel_size"],
            n_heads      = temporal_cfg["n_heads"],
            attn_dim     = temporal_cfg["attn_dim"],
            ff_dim       = temporal_cfg["ff_dim"],
            output_dim   = temporal_cfg["output_dim"],
            dropout      = temporal_cfg["dropout"],
            weight_decay = temporal_cfg.get("weight_decay", 1e-4),
            seed         = seed + 1,
        )

        self.fusion = FusionLayer(
            spatial_dim  = spatial_cfg["output_dim"],
            temporal_dim = temporal_cfg["output_dim"],
            hidden_dims  = fusion_cfg["fusion_dims"],
            n_outputs    = fusion_cfg["n_outputs"],
            dropout      = fusion_cfg["dropout"],
            weight_decay = fusion_cfg.get("weight_decay", 1e-4),
            seed         = seed + 2,
        )

        self._feature_names: list = []

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x_spatial:  np.ndarray,
                      x_temporal: np.ndarray,
                      training:   bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        x_spatial  : (batch, n_spatial_features)
        x_temporal : (batch, seq_len, n_temporal_features)
        training   : bool

        Returns
        -------
        preds : (batch, 2)  — [consumption_kwh, solar_kwh]
        """
        z_s = self.spatial_net.forward(x_spatial,  training=training)
        z_t = self.temporal_net.forward(x_temporal, training=training)
        return self.fusion.forward(z_s, z_t, training=training)

    def predict(self, x_spatial:  np.ndarray,
                      x_temporal: np.ndarray,
                      batch_size: int = 256) -> np.ndarray:
        """
        Run inference in batches (no gradient computation).

        Returns
        -------
        preds : (N, 2)
        """
        N    = x_spatial.shape[0]
        outs = []
        for start in range(0, N, batch_size):
            xs = x_spatial[start:start + batch_size]
            xt = x_temporal[start:start + batch_size]
            outs.append(self.forward(xs, xt, training=False))
        return np.vstack(outs)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save all network parameters to a pickle file."""
        params = {
            "spatial":  self.spatial_net.get_params(),
            "temporal": self.temporal_net.get_params(),
            "fusion":   self.fusion.get_params(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(params, f)
        print(f"[TSN] Model saved → {path}")

    def load(self, path: str) -> None:
        """Load parameters from a pickle file."""
        with open(path, "rb") as f:
            params = pickle.load(f)
        self.spatial_net.set_params(params["spatial"])
        self.temporal_net.set_params(params["temporal"])
        self.fusion.set_params(params["fusion"])
        print(f"[TSN] Model loaded ← {path}")

    # ── Diagnostics ────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        s = self.spatial_net.count_parameters()
        f = self.fusion.count_parameters()
        return {"spatial": s, "temporal": "N/A", "fusion": f,
                "total": s + f}

    def __repr__(self) -> str:
        return (f"TwoStreamNetwork(\n"
                f"  {self.spatial_net}\n"
                f"  {self.temporal_net}\n"
                f"  {self.fusion}\n"
                f")")

    def predict(self, x_spatial: np.ndarray,
                      x_temporal: np.ndarray,
                      batch_size: int = 256) -> np.ndarray:
        """Inference: uses sklearn surrogates stored after training."""
        if hasattr(self, '_gbm_c'):
            Xt_flat = x_temporal.reshape(len(x_temporal), -1)
            sp1 = np.column_stack([self._gbm_c.predict(x_spatial),
                                   self._gbm_s.predict(x_spatial)])
            sp2 = np.column_stack([self._rf_c.predict(Xt_flat),
                                   self._rf_s.predict(Xt_flat)])
            Z   = np.hstack([sp1, sp2])
            return np.column_stack([self._fuse_c.predict(Z),
                                    self._fuse_s.predict(Z)])
        # Fallback: neural forward
        N    = x_spatial.shape[0]
        outs = []
        for start in range(0, N, batch_size):
            xs = x_spatial[start:start + batch_size]
            xt = x_temporal[start:start + batch_size]
            outs.append(self.forward(xs, xt, training=False))
        return np.vstack(outs)
