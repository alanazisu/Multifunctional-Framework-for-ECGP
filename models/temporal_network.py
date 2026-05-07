"""
models/temporal_network.py
===========================
Stream 2: Energy Temporal Network (ETN)

Captures sequential dynamics of the energy time-series using:
  1. 1D Convolutional front-end  — extracts local patterns (trends, spikes)
  2. Multi-Head Self-Attention   — models long-range temporal dependencies
  3. Position-wise Feed-Forward  — projects to temporal embedding

Architecture (Transformer-style Temporal Encoder):
    Sequence (T, D_in)
        │
        ▼
    Conv1D blocks × L_conv   ← local pattern extraction
        │
        ▼
    Multi-Head Self-Attention ← long-range dependencies
        │
        ▼
    Layer Norm + Skip
        │
        ▼
    Feed-Forward Network
        │
        ▼
    Layer Norm + Skip
        │
        ▼
    Global Average Pool       ← collapse time dimension
        │
        ▼
    Temporal Embedding (output_dim,)
"""

import numpy as np
from typing import List, Tuple, Optional


# ── Utilities ──────────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _dropout(x: np.ndarray, rate: float, training: bool,
             rng: np.random.Generator) -> np.ndarray:
    if not training or rate == 0.0:
        return x
    mask = (rng.uniform(0, 1, x.shape) > rate).astype(np.float64)
    return x * mask / (1.0 - rate)


class LayerNorm:
    """Layer normalisation (normalises over feature dimension)."""

    def __init__(self, n_features: int, eps: float = 1e-6) -> None:
        self.eps    = eps
        self.gamma  = np.ones(n_features)
        self.beta   = np.zeros(n_features)
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean  = x.mean(axis=-1, keepdims=True)
        var   = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        self._cache = {"x": x, "mean": mean, "var": var, "x_hat": x_hat}
        return self.gamma * x_hat + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, mean, var, x_hat = (self._cache[k] for k in ["x","mean","var","x_hat"])
        N    = x.shape[-1]
        dgamma = (dout * x_hat).sum(axis=tuple(range(dout.ndim - 1)))
        dbeta  = dout.sum(axis=tuple(range(dout.ndim - 1)))
        self.dgamma = dgamma
        self.dbeta  = dbeta
        dx_hat = dout * self.gamma
        dvar   = (-0.5 * (dx_hat * (x - mean)).sum(axis=-1, keepdims=True)
                  * (var + self.eps)**(-1.5))
        dmean  = (-(dx_hat / np.sqrt(var + self.eps)).sum(axis=-1, keepdims=True)
                  + dvar * (-2 * (x - mean)).mean(axis=-1, keepdims=True))
        return (dx_hat / np.sqrt(var + self.eps)
                + dvar * 2 * (x - mean) / N
                + dmean / N)

    def get_params(self) -> dict:
        return {"gamma": self.gamma, "beta": self.beta}

    def set_params(self, p: dict) -> None:
        self.gamma = p["gamma"]
        self.beta  = p["beta"]


class Conv1DLayer:
    """
    1D convolution over time axis.
    Input shape:  (batch, time, channels_in)
    Output shape: (batch, time, channels_out)  [same padding]
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 rng: np.random.Generator, weight_decay: float = 1e-4) -> None:
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.k      = kernel
        self.pad    = kernel // 2
        self.wd     = weight_decay
        scale       = np.sqrt(2.0 / (in_ch * kernel))
        self.W      = rng.standard_normal((kernel, in_ch, out_ch)) * scale
        self.b      = np.zeros(out_ch)
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (B, T, C_in) → (B, T, C_out)"""
        B, T, _ = x.shape
        x_pad   = np.pad(x, ((0,0),(self.pad, self.pad),(0,0)), mode="reflect")
        out     = np.zeros((B, T, self.out_ch))
        for k_i in range(self.k):
            out += x_pad[:, k_i:k_i+T, :] @ self.W[k_i]   # (B,T,C_in)@(C_in,C_out)
        out += self.b
        self._cache = {"x": x, "x_pad": x_pad}
        return _relu(out)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, x_pad = self._cache["x"], self._cache["x_pad"]
        B, T, _  = x.shape
        self.dW  = np.zeros_like(self.W)
        self.db  = dout.sum(axis=(0, 1))
        dout_pad = np.pad(dout, ((0,0),(self.pad, self.pad),(0,0)))
        dx_pad   = np.zeros_like(x_pad)
        for k_i in range(self.k):
            self.dW[k_i] += np.einsum("btc,bto->co", x_pad[:,k_i:k_i+T,:], dout)
            self.dW[k_i] += self.wd * self.W[k_i]
            dx_pad[:, k_i:k_i+T, :] += dout @ self.W[k_i].T
        dx = dx_pad[:, self.pad:self.pad+T, :]
        return dx

    def get_params(self) -> dict:
        return {"W": self.W, "b": self.b}

    def set_params(self, p: dict) -> None:
        self.W = p["W"]
        self.b = p["b"]


class MultiHeadSelfAttention:
    """
    Scaled dot-product multi-head self-attention.
    Input:  (B, T, D)
    Output: (B, T, D)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 rng: np.random.Generator, weight_decay: float = 1e-4) -> None:
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads
        self.dropout  = dropout
        self.wd       = weight_decay
        scale = np.sqrt(2.0 / d_model)
        # QKV projection and output projection
        self.Wq = rng.standard_normal((d_model, d_model)) * scale
        self.Wk = rng.standard_normal((d_model, d_model)) * scale
        self.Wv = rng.standard_normal((d_model, d_model)) * scale
        self.Wo = rng.standard_normal((d_model, d_model)) * scale
        self.bq = np.zeros(d_model)
        self.bk = np.zeros(d_model)
        self.bv = np.zeros(d_model)
        self.bo = np.zeros(d_model)
        self._cache: dict = {}

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, T, D) → (B, H, T, d_k)"""
        B, T, D = x.shape
        x = x.reshape(B, T, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(B, H, T, d_k) → (B, T, D)"""
        B, H, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dk)

    def forward(self, x: np.ndarray, training: bool = True,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
        B, T, _ = x.shape
        Q = x @ self.Wq + self.bq       # (B, T, D)
        K = x @ self.Wk + self.bk
        V = x @ self.Wv + self.bv

        Q = self._split_heads(Q)         # (B, H, T, d_k)
        K = self._split_heads(K)
        V = self._split_heads(V)

        scores = Q @ K.transpose(0,1,3,2) / np.sqrt(self.d_k)   # (B,H,T,T)
        # Causal mask (optional — remove for bi-directional)
        # mask = np.triu(np.ones((T,T)), k=1) * -1e9
        # scores += mask
        attn   = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn  /= attn.sum(axis=-1, keepdims=True) + 1e-9

        if training and rng is not None:
            drop_mask = (rng.uniform(0,1, attn.shape) > self.dropout).astype(np.float64)
            attn = attn * drop_mask / (1.0 - self.dropout + 1e-8)

        ctx  = attn @ V                                          # (B,H,T,d_k)
        ctx  = self._merge_heads(ctx)                            # (B,T,D)
        out  = ctx @ self.Wo + self.bo

        self._cache = {"x": x, "Q": Q, "K": K, "V": V,
                       "attn": attn, "ctx": ctx}
        return out

    def get_params(self) -> dict:
        return {k: getattr(self, k) for k in
                ["Wq","Wk","Wv","Wo","bq","bk","bv","bo"]}

    def set_params(self, p: dict) -> None:
        for k in ["Wq","Wk","Wv","Wo","bq","bk","bv","bo"]:
            setattr(self, k, p[k])


class FeedForwardBlock:
    """Position-wise FFN: Linear → ReLU → Linear."""

    def __init__(self, d_model: int, d_ff: int,
                 rng: np.random.Generator, weight_decay: float = 1e-4) -> None:
        scale = np.sqrt(2.0 / d_model)
        self.W1 = rng.standard_normal((d_model, d_ff))   * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.standard_normal((d_ff, d_model))   * np.sqrt(2.0/d_ff)
        self.b2 = np.zeros(d_model)
        self.wd = weight_decay
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = _relu(x @ self.W1 + self.b1)
        o = h @ self.W2 + self.b2
        self._cache = {"x": x, "h": h}
        return o

    def backward(self, dout: np.ndarray) -> np.ndarray:
        h = self._cache["h"]; x = self._cache["x"]
        self.dW2 = h.reshape(-1, h.shape[-1]).T @ dout.reshape(-1, dout.shape[-1]) + self.wd * self.W2
        self.db2 = dout.sum(axis=(0,1))
        dh   = dout @ self.W2.T
        dh   = dh * (h > 0).astype(np.float64)
        self.dW1 = x.reshape(-1, x.shape[-1]).T @ dh.reshape(-1, dh.shape[-1]) + self.wd * self.W1
        self.db1 = dh.sum(axis=(0,1))
        return dh @ self.W1.T

    def get_params(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_params(self, p: dict) -> None:
        for k in ["W1","b1","W2","b2"]: setattr(self, k, p[k])


class EnergyTemporalNetwork:
    """
    Stream 2 — Energy Temporal Network.

    Processes a sequence of length T with D features per step through:
      (a) 1D Conv blocks for local pattern extraction
      (b) Multi-head self-attention for global temporal dependencies
      (c) FFN + Layer norm + residual
      (d) Global average pooling → fixed-size temporal embedding

    Parameters
    ----------
    input_dim    : int   — features per time step (after conv input projection)
    seq_len      : int   — look-back sequence length T
    conv_filters : list  — number of filters for each Conv1D layer
    kernel_size  : int   — 1D conv kernel width
    n_heads      : int   — self-attention heads
    attn_dim     : int   — attention / model dimension
    ff_dim       : int   — feed-forward hidden dimension
    output_dim   : int   — final temporal embedding size
    dropout      : float — dropout rate
    seed         : int   — random seed
    """

    def __init__(self, input_dim: int, seq_len: int,
                 conv_filters: List[int], kernel_size: int,
                 n_heads: int, attn_dim: int, ff_dim: int,
                 output_dim: int, dropout: float = 0.15,
                 weight_decay: float = 1e-4, seed: int = 42) -> None:
        self.rng       = np.random.default_rng(seed)
        self.seq_len   = seq_len
        self.attn_dim  = attn_dim
        self.output_dim= output_dim
        self.dropout   = dropout

        # ── Conv blocks ────────────────────────────────────────────────────
        conv_dims = [input_dim] + conv_filters
        self.conv_layers = [
            Conv1DLayer(conv_dims[i], conv_dims[i+1], kernel_size,
                        self.rng, weight_decay)
            for i in range(len(conv_dims) - 1)
        ]
        self.conv_bn = [LayerNorm(conv_dims[i+1]) for i in range(len(conv_dims)-1)]

        # ── Input projection to attn_dim ───────────────────────────────────
        last_conv_ch = conv_filters[-1]
        self.input_proj = None
        if last_conv_ch != attn_dim:
            scale = np.sqrt(2.0 / last_conv_ch)
            self.proj_W = self.rng.standard_normal((last_conv_ch, attn_dim)) * scale
            self.proj_b = np.zeros(attn_dim)

        # ── Attention block ────────────────────────────────────────────────
        self.attn = MultiHeadSelfAttention(attn_dim, n_heads, dropout,
                                           self.rng, weight_decay)
        self.ln1  = LayerNorm(attn_dim)

        # ── FFN block ──────────────────────────────────────────────────────
        self.ffn  = FeedForwardBlock(attn_dim, ff_dim, self.rng, weight_decay)
        self.ln2  = LayerNorm(attn_dim)

        # ── Output projection ──────────────────────────────────────────────
        scale = np.sqrt(2.0 / attn_dim)
        self.out_W = self.rng.standard_normal((attn_dim, output_dim)) * scale
        self.out_b = np.zeros(output_dim)
        self.out_bn = LayerNorm(output_dim)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)

        Returns
        -------
        embedding : (batch, output_dim)
        """
        out = x.astype(np.float64)

        # Conv blocks
        for conv, bn in zip(self.conv_layers, self.conv_bn):
            out = bn.forward(conv.forward(out))   # (B, T, filters)

        # Project to attn_dim
        if hasattr(self, "proj_W"):
            out = out @ self.proj_W + self.proj_b
            out = _relu(out)

        # Positional encoding (learnable-free sinusoidal)
        B, T, D = out.shape
        pos  = np.arange(T)[:, None]
        dim  = np.arange(0, D, 2)
        pe   = np.zeros((T, D))
        pe[:, 0::2] = np.sin(pos / 10000 ** (dim / D))
        pe[:, 1::2] = np.cos(pos / 10000 ** (dim[:D//2] / D))
        out  = out + pe[None, :, :]

        # Multi-head attention + residual
        attn_out = self.attn.forward(out, training, self.rng)
        attn_out = _dropout(attn_out, self.dropout, training, self.rng)
        out      = self.ln1.forward(out + attn_out)

        # FFN + residual
        ffn_out = self.ffn.forward(out)
        ffn_out = _dropout(ffn_out, self.dropout, training, self.rng)
        out     = self.ln2.forward(out + ffn_out)

        # Global average pool over time → (B, D)
        out = out.mean(axis=1)

        # Output projection
        out = out @ self.out_W + self.out_b
        out = self.out_bn.forward(out)
        return out

    def get_params(self) -> dict:
        return {
            "convs":   [c.get_params() for c in self.conv_layers],
            "conv_bns":[b.get_params() for b in self.conv_bn],
            "proj":    {"W": self.proj_W, "b": self.proj_b} if hasattr(self,"proj_W") else None,
            "attn":    self.attn.get_params(),
            "ln1":     self.ln1.get_params(),
            "ffn":     self.ffn.get_params(),
            "ln2":     self.ln2.get_params(),
            "out":     {"W": self.out_W, "b": self.out_b},
            "out_bn":  self.out_bn.get_params(),
        }

    def set_params(self, p: dict) -> None:
        for c, cp in zip(self.conv_layers, p["convs"]): c.set_params(cp)
        for b, bp in zip(self.conv_bn, p["conv_bns"]):  b.set_params(bp)
        if p.get("proj"):
            self.proj_W = p["proj"]["W"]; self.proj_b = p["proj"]["b"]
        self.attn.set_params(p["attn"])
        self.ln1.set_params(p["ln1"])
        self.ffn.set_params(p["ffn"])
        self.ln2.set_params(p["ln2"])
        self.out_W = p["out"]["W"]; self.out_b = p["out"]["b"]
        self.out_bn.set_params(p["out_bn"])

    def __repr__(self) -> str:
        return (f"EnergyTemporalNetwork("
                f"seq={self.seq_len}, "
                f"attn_dim={self.attn_dim}, "
                f"output={self.output_dim})")
