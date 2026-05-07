"""
config.py — Central configuration for Energy-TSN
All hyperparameters, paths, and experiment settings in one place.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

for d in [DATA_DIR, MODEL_DIR, RESULT_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

CONSUMPTION_CSV = os.path.join(DATA_DIR, "household_consumption.csv")
SOLAR_CSV       = os.path.join(DATA_DIR, "pv_solar_generation.csv")

# ── Dataset ────────────────────────────────────────────────────────────────
DATASET = dict(
    start_date    = "2023-01-01",
    end_date      = "2023-12-31",
    freq          = "1h",          # hourly resolution for full year
    train_ratio   = 0.70,
    val_ratio     = 0.15,
    test_ratio    = 0.15,
    random_seed   = 42,
    # Household consumption parameters (kWh)
    base_load     = 0.35,          # base load kWh/h
    peak_morning  = 0.80,
    peak_evening  = 1.20,
    noise_std     = 0.08,
    # PV solar parameters (kWh)
    peak_solar    = 3.50,          # peak panel output kWh/h
    panel_area_m2 = 20.0,
    panel_eff     = 0.185,
)

# ── Features ───────────────────────────────────────────────────────────────
FEATURES = dict(
    # Temporal features (used by both streams)
    temporal_cols = [
        "hour_sin", "hour_cos",
        "day_sin",  "day_cos",
        "month_sin","month_cos",
        "is_weekend", "is_holiday",
    ],
    # Meteorological features
    meteo_cols = [
        "temperature", "humidity",
        "solar_irradiance", "cloud_cover",
        "wind_speed",
    ],
    # Lagged consumption features
    consumption_lag_cols = [
        "consumption_lag_1h",  "consumption_lag_2h",
        "consumption_lag_24h", "consumption_lag_168h",
        "consumption_roll_mean_24h", "consumption_roll_std_24h",
        "consumption_roll_mean_7d",
    ],
    # Lagged solar features
    solar_lag_cols = [
        "solar_lag_1h",  "solar_lag_2h",
        "solar_lag_24h", "solar_lag_168h",
        "solar_roll_mean_24h", "solar_roll_std_24h",
    ],
    # Sequence look-back window (hours) for temporal stream
    lookback = 24,
    # Targets
    target_consumption = "consumption_kwh",
    target_solar       = "solar_kwh",
)

# ── Stream 1: Energy Spatial Network ──────────────────────────────────────
SPATIAL_NET = dict(
    hidden_dims    = [256, 512, 256, 128],  # MLP layer widths
    dropout        = 0.20,
    batch_norm     = True,
    activation     = "relu",                # relu | leaky_relu | elu
    skip_connect   = True,                  # residual connections
    output_dim     = 64,                    # spatial embedding size
)

# ── Stream 2: Energy Temporal Network ─────────────────────────────────────
TEMPORAL_NET = dict(
    conv_filters   = [32, 64, 64],          # 1D conv layers
    kernel_size    = 3,
    n_heads        = 4,                     # self-attention heads
    attn_dim       = 64,                    # attention dimension
    ff_dim         = 128,                   # feed-forward dimension
    dropout        = 0.15,
    output_dim     = 64,                    # temporal embedding size
)

# ── Fusion Layer ───────────────────────────────────────────────────────────
FUSION = dict(
    fusion_dims    = [128, 64],             # post-concat MLP
    dropout        = 0.10,
    n_outputs      = 2,                     # consumption + solar
)

# ── Training ───────────────────────────────────────────────────────────────
TRAINING = dict(
    epochs         = 80,
    batch_size     = 64,
    learning_rate  = 3e-3,
    lr_decay       = 0.95,
    lr_decay_every = 10,                    # epochs
    weight_decay   = 1e-4,
    patience       = 15,                    # early stopping
    loss           = "huber",              # mse | mae | huber
    huber_delta    = 1.0,
    gradient_clip  = 5.0,
    random_seed    = 42,
)

# ── Evaluation granularities ───────────────────────────────────────────────
GRANULARITIES = ["minutely", "hourly", "daily", "weekly", "monthly"]

# ── Plot style ─────────────────────────────────────────────────────────────
PLOT = dict(
    style       = "seaborn-v0_8-whitegrid",
    dpi         = 160,
    fig_ext     = "png",
    consumption_color = "#E74C3C",
    solar_color       = "#F39C12",
    pred_color        = "#2980B9",
    pred_solar_color  = "#8E44AD",
    alpha             = 0.80,
)
