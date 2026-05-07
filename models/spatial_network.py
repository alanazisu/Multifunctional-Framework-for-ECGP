"""
models/spatial_network.py
==========================
Stream 1: Energy Spatial Network (ESN)

Captures cross-feature relationships between meteorological variables,
calendar encodings, and lagged energy values using a deep MLP with:
  - Batch normalisation after each linear layer
  - ReLU activation
  - Residual (skip) connections where dimensions match
  - Dropout regularisation
  - L2 weight regularisation support

Architecture:
    Input → [BN → Linear → BN → Activation → Dropout] × L → Output embedding
                          ↑_____________skip____________↑  (residual)
"""

import numpy as np
from typing import List, Tuple, Optional


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def _elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def _dropout(x: np.ndarray, rate: float, training: bool,
             rng: np.random.Generator) -> np.ndarray:
    if not training or rate == 0.0:
        return x
    mask = (rng.uniform(0, 1, x.shape) > rate).astype(np.float64)
    return x * mask / (1.0 - rate)


class BatchNorm1D:
    """Running-stats batch normalisation for 1D feature vectors."""

    def __init__(self, n_features: int, eps: float = 1e-5,
                 momentum: float = 0.1) -> None:
        self.n_features = n_features
        self.eps        = eps
        self.momentum   = momentum
        # Learnable parameters
        self.gamma = np.ones(n_features)
        self.beta  = np.zeros(n_features)
        # Running statistics
        self.running_mean = np.zeros(n_features)
        self.running_var  = np.ones(n_features)
        # Cache for backward pass
        self._cache: dict = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mean = x.mean(axis=0)
            var  = x.var(axis=0)
            x_hat = (x - mean) / np.sqrt(var + self.eps)
            self.running_mean = ((1 - self.momentum) * self.running_mean
                                 + self.momentum * mean)
            self.running_var  = ((1 - self.momentum) * self.running_var
                                 + self.momentum * var)
            self._cache = {"x": x, "mean": mean, "var": var, "x_hat": x_hat}
        else:
            x_hat = ((x - self.running_mean)
                     / np.sqrt(self.running_var + self.eps))
        return self.gamma * x_hat + self.beta

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, mean, var, x_hat = (self._cache[k] for k in ["x","mean","var","x_hat"])
        N = x.shape[0]
        dgamma = (dout * x_hat).sum(axis=0)
        dbeta  = dout.sum(axis=0)
        dx_hat = dout * self.gamma
        dvar   = (-0.5 * dx_hat * (x - mean)
                  * (var + self.eps)**(-1.5)).sum(axis=0)
        dmean  = ((-dx_hat / np.sqrt(var + self.eps)).sum(axis=0)
                  + dvar * (-2 * (x - mean)).mean(axis=0))
        dx     = (dx_hat / np.sqrt(var + self.eps)
                  + dvar * 2 * (x - mean) / N
                  + dmean / N)
        return dx, dgamma, dbeta

    def get_params(self) -> dict:
        return {"gamma": self.gamma, "beta": self.beta,
                "running_mean": self.running_mean, "running_var": self.running_var}

    def set_params(self, params: dict) -> None:
        self.gamma        = params["gamma"]
        self.beta         = params["beta"]
        self.running_mean = params["running_mean"]
        self.running_var  = params["running_var"]


class LinearLayer:
    """Fully-connected layer with He initialisation."""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator,
                 weight_decay: float = 1e-4) -> None:
        scale         = np.sqrt(2.0 / in_dim)          # He init for ReLU
        self.W        = rng.standard_normal((in_dim, out_dim)) * scale
        self.b        = np.zeros(out_dim)
        self.weight_decay = weight_decay
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = {"x": x}
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x  = self._cache["x"]
        self.dW = x.T @ dout + self.weight_decay * self.W
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def get_params(self) -> dict:
        return {"W": self.W, "b": self.b}

    def set_params(self, params: dict) -> None:
        self.W = params["W"]
        self.b = params["b"]


class SpatialBlock:
    """One residual block: BN → Linear → BN → Act → Dropout (+ optional skip)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float,
                 activation: str, rng: np.random.Generator,
                 weight_decay: float = 1e-4) -> None:
        self.in_dim    = in_dim
        self.out_dim   = out_dim
        self.dropout   = dropout
        self.activation_name = activation

        self.bn1    = BatchNorm1D(in_dim)
        self.linear = LinearLayer(in_dim, out_dim, rng, weight_decay)
        self.bn2    = BatchNorm1D(out_dim)

        # Projection for residual if dims differ
        self.proj = (LinearLayer(in_dim, out_dim, rng, weight_decay)
                     if in_dim != out_dim else None)
        self._act_fn = {"relu": _relu, "leaky_relu": _leaky_relu, "elu": _elu}[activation]
        self._cache: dict = {}

    def forward(self, x: np.ndarray, training: bool = True,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
        residual = x if self.proj is None else self.proj.forward(x)
        out  = self.bn1.forward(x, training)
        out  = self.linear.forward(out)
        out  = self.bn2.forward(out, training)
        out  = self._act_fn(out)
        out  = _dropout(out, self.dropout, training, rng or np.random.default_rng())
        out  = out + residual
        self._cache = {"x": x, "residual": residual, "pre_skip": out - residual}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        d_pre_skip = dout.copy()
        d_residual = dout.copy()
        # Through BN2 → Linear → BN1
        d = self.bn2.backward(d_pre_skip)[0]
        d = self.linear.backward(d)
        d = self.bn1.backward(d)[0]
        # Projection grad
        if self.proj is not None:
            d_residual = self.proj.backward(d_residual)
        return d + d_residual

    def get_params(self) -> dict:
        p = {"bn1": self.bn1.get_params(),
             "linear": self.linear.get_params(),
             "bn2": self.bn2.get_params()}
        if self.proj: p["proj"] = self.proj.get_params()
        return p

    def set_params(self, params: dict) -> None:
        self.bn1.set_params(params["bn1"])
        self.linear.set_params(params["linear"])
        self.bn2.set_params(params["bn2"])
        if self.proj and "proj" in params:
            self.proj.set_params(params["proj"])


class EnergySpatialNetwork:
    """
    Stream 1 — Energy Spatial Network.

    Processes a flat feature vector [meteo + calendar + lags] through
    a stack of residual MLP blocks to produce a fixed-size spatial embedding.

    Parameters
    ----------
    input_dim   : int    — number of input features
    hidden_dims : list   — widths of each hidden block
    output_dim  : int    — spatial embedding size
    dropout     : float  — dropout probability
    activation  : str    — activation function name
    weight_decay: float  — L2 regularisation coefficient
    seed        : int    — random seed
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.20, activation: str = "relu",
                 weight_decay: float = 1e-4, seed: int = 42) -> None:
        self.rng        = np.random.default_rng(seed)
        self.input_dim  = input_dim
        self.output_dim = output_dim

        # Build residual blocks
        dims    = [input_dim] + hidden_dims + [output_dim]
        self.blocks = [
            SpatialBlock(dims[i], dims[i+1], dropout, activation,
                         self.rng, weight_decay)
            for i in range(len(dims) - 1)
        ]
        # Final BN on output embedding
        self.final_bn = BatchNorm1D(output_dim)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        x        : (batch, input_dim)
        training : bool

        Returns
        -------
        embedding: (batch, output_dim)
        """
        out = x.astype(np.float64)
        for block in self.blocks:
            out = block.forward(out, training, self.rng)
        out = self.final_bn.forward(out, training)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        d = self.final_bn.backward(dout)[0]
        for block in reversed(self.blocks):
            d = block.backward(d)
        return d

    def get_all_params(self) -> List[dict]:
        """Collect all trainable parameters for optimiser."""
        params = []
        for block in self.blocks:
            params.append({"W": block.linear.W, "b": block.linear.b,
                           "tag": "spatial_linear"})
            params.append({"gamma": block.bn2.gamma, "beta": block.bn2.beta,
                           "tag": "spatial_bn"})
            if block.proj:
                params.append({"W": block.proj.W, "b": block.proj.b,
                               "tag": "spatial_proj"})
        params.append({"gamma": self.final_bn.gamma, "beta": self.final_bn.beta,
                       "tag": "spatial_final_bn"})
        return params

    def get_params(self) -> dict:
        return {"blocks": [b.get_params() for b in self.blocks],
                "final_bn": self.final_bn.get_params()}

    def set_params(self, params: dict) -> None:
        for b, p in zip(self.blocks, params["blocks"]):
            b.set_params(p)
        self.final_bn.set_params(params["final_bn"])

    def count_parameters(self) -> int:
        total = 0
        for block in self.blocks:
            total += block.linear.W.size + block.linear.b.size
            if block.proj:
                total += block.proj.W.size + block.proj.b.size
        return total

    def __repr__(self) -> str:
        return (f"EnergySpatialNetwork("
                f"input={self.input_dim}, "
                f"hidden={[b.out_dim for b in self.blocks[:-1]]}, "
                f"output={self.output_dim}, "
                f"params={self.count_parameters():,})")
