# Power-Load-Generation-Forecast ⚡

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0+-orange?logo=numpy)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Energy-TSN** — A Two-Stream Network for joint prediction of household energy **consumption** and PV solar **generation**, capturing both spatial cross-feature patterns and temporal time-series dynamics through parallel independent streams fused at inference.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Metrics](#metrics)
- [Figures](#figures)
- [Citation](#citation)
- [License](#license)

---

## Overview

**Energy-TSN** addresses the joint prediction problem of:
1. **Household energy consumption** (kWh/h) — driven by occupant behaviour, temperature, and appliance usage
2. **PV solar generation** (kWh/h) — driven by solar irradiance, cloud cover, and panel efficiency

Unlike single-stream models, Energy-TSN explicitly separates and models:
- **Spatial patterns** — cross-feature correlations between meteorological, calendar, and lagged energy variables
- **Temporal patterns** — sequential dynamics of the energy time-series over a 24-hour look-back window

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Energy-TSN Architecture                       │
└──────────────────────────────────────────────────────────────────┘

 Flat Feature Vector                   Temporal Sequence (24h)
 [temp, humidity, hour_sin,            [t-23, t-22, ..., t-1, t]
  cloud, lags, rolling stats...]       shape: (24, 5)
         │                                     │
         ▼                                     ▼
 ┌─────────────────────┐           ┌─────────────────────────┐
 │  Stream 1: ESN      │           │  Stream 2: ETN          │
 │  Energy Spatial     │           │  Energy Temporal        │
 │  Network            │           │  Network                │
 │                     │           │                         │
 │  GradientBoosting   │           │  RandomForest on        │
 │  Regressor          │           │  flattened sequences    │
 │  (120 estimators)   │           │  (80 estimators)        │
 └──────────┬──────────┘           └───────────┬─────────────┘
            │                                  │
            └──────────────┬───────────────────┘
                           ▼
                ┌─────────────────────┐
                │   Fusion Layer      │
                │   Ridge Regression  │
                │   on [ESN || ETN]   │
                └──────────┬──────────┘
                           ▼
              [consumption_kWh, solar_kWh]
```

### Stream 1 — Energy Spatial Network (ESN)
Gradient Boosting Regressor capturing **cross-feature relationships** between meteorological variables, calendar encodings (cyclically encoded), lagged energy values (1h, 2h, 24h, 168h), and rolling statistics.

### Stream 2 — Energy Temporal Network (ETN)
Random Forest Regressor on **flattened 24-hour look-back sequences** capturing multi-step time-series patterns, intra-day and cross-day temporal dependencies.

### Fusion Layer
Ridge Regression combining both stream outputs with learned weights, producing calibrated predictions for consumption and solar generation simultaneously.

---

## Results

### Hourly Test Set

| Target | MSE | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|---|
| **Consumption (kWh)** | 0.0049 | 0.0702 | 0.0558 | 3.88% | **0.9613** |
| **Solar Generation (kWh)** | 0.0015 | 0.0389 | 0.0181 | 3.50% | **0.9972** |

### Multi-Granularity — Household Consumption

| Granularity | RMSE (kWh) | MAE (kWh) | MAPE (%) | R² |
|---|---|---|---|---|
| Hourly | 0.0702 | 0.0558 | 3.88 | 0.9613 |
| Daily | 0.4471 | 0.3666 | 1.16 | 0.9934 |
| Weekly | 1.3408 | 1.1824 | 0.49 | 0.9949 |
| Monthly | 3.0642 | 3.0424 | 0.32 | **0.9993** |

### Multi-Granularity — PV Solar Generation

| Granularity | RMSE (kWh) | MAE (kWh) | MAPE (%) | R² |
|---|---|---|---|---|
| Hourly | 0.0389 | 0.0181 | 3.50 | 0.9972 |
| Daily | 0.1743 | 0.1301 | 1.13 | 0.9954 |
| Weekly | 0.5238 | 0.4268 | 0.54 | 0.9968 |
| Monthly | 1.3131 | 1.3081 | 0.40 | **0.9925** |

---

## Installation

```bash
git clone https://github.com/alanazisu/Power-Load-Generation-Forecast.git
cd Power-Load-Generation-Forecast
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, NumPy, pandas, scikit-learn, scipy, matplotlib, seaborn

---

## Quick Start

```bash
# Full pipeline: generate data → train → evaluate → plot all figures
python main.py --all

# Individual steps
python main.py --generate-data          # Generate synthetic datasets only
python main.py --train                  # Train both streams + fusion layer
python main.py --evaluate               # Evaluate on test set (all granularities)
python main.py --plot                   # Generate all 14 result figures

# Evaluate a specific granularity
python main.py --evaluate --granularity hourly
python main.py --evaluate --granularity daily
python main.py --evaluate --granularity weekly
python main.py --evaluate --granularity monthly
```

---

## Project Structure

```
Power-Load-Generation-Forecast/
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── config.py                        # All hyperparameters and paths
├── main.py                          # CLI entry point
│
├── data/
│   ├── __init__.py
│   └── generate_datasets.py         # Synthetic dataset generator
│                                    # (household consumption + PV solar)
│
├── models/
│   ├── __init__.py
│   ├── spatial_network.py           # Stream 1: Energy Spatial Network (ESN)
│   ├── temporal_network.py          # Stream 2: Energy Temporal Network (ETN)
│   ├── fusion.py                    # Fusion layer (learned alpha-weighting)
│   └── two_stream_network.py        # Top-level TSN wrapper
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                   # Training loop (GBM + RF + Ridge fusion)
│   └── loss.py                      # Loss functions (MSE, MAE, Huber)
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # MSE, RMSE, MAE, MAPE, SMAPE, R2, CVRMSE
│   └── evaluator.py                 # Multi-granularity evaluation engine
│
├── figures/
│   ├── __init__.py
│   └── plotter.py                   # 14 publication-quality figure generators
│
├── checkpoints/                     # Saved model weights (auto-created, git-ignored)
├── results/                         # CSV metric tables  (auto-created)
└── figures/                         # Output PNG figures (auto-created, git-ignored)
```

---

## Datasets

Both datasets are generated using physically-motivated simulation models:

### Household Consumption
- Model: base load + diurnal Gaussian peaks (morning 07:30, evening 19:30) + weekend boost + HVAC load
- Range: 0.05 – 4.0 kWh/h, mean ≈ 1.21 kWh/h

### PV Solar Generation
- Model: Spencer clear-sky irradiance + cloud cover attenuation + temperature efficiency correction (−0.4%/°C above 25°C)
- Panel spec: 20 m², 18.5% efficiency, peak 3.5 kWh/h

> Datasets are auto-generated by `python main.py --generate-data` and excluded from Git.

### Features Used

| Group | Features | Dim |
|---|---|---|
| Calendar (cyclical) | hour_sin/cos, day_sin/cos, month_sin/cos | 6 |
| Calendar (binary) | is_weekend, is_holiday | 2 |
| Meteorological | temperature, humidity, solar_irradiance, cloud_cover, wind_speed | 5 |
| Consumption lags | lag_1h, lag_2h, lag_24h, lag_168h, roll_mean_24h, roll_std_24h, roll_mean_7d | 7 |
| Solar lags | lag_1h, lag_2h, lag_24h, lag_168h, roll_mean_24h, roll_std_24h | 6 |
| **Total spatial** | | **26** |
| **Temporal (ETN)** | consumption, solar, hour_sin/cos, irradiance × 24 steps | **5 × 24** |

---

## Metrics

| Metric | Formula | Unit |
|---|---|---|
| MSE | mean((ŷ − y)²) | kWh² |
| RMSE | √MSE | kWh |
| MAE | mean(|ŷ − y|) | kWh |
| MAPE | mean(|ŷ − y| / y) × 100 | % |
| SMAPE | mean(2|ŷ−y|/(|ŷ|+|y|)) × 100 | % |
| R² | 1 − SS_res/SS_tot | — |
| CVRMSE | RMSE / mean(y) × 100 | % |

---

## Figures

`python main.py --plot` generates 14 figures in `figures/`:

| # | Filename | Description |
|---|---|---|
| 1 | `01_training_history.png` | Loss curves + MSE per target |
| 2 | `02_actual_vs_pred_minutely.png` | Actual vs Predicted — raw resolution |
| 3 | `03_actual_vs_pred_hourly.png` | Actual vs Predicted — hourly |
| 4 | `04_actual_vs_pred_daily.png` | Actual vs Predicted — daily |
| 5 | `05_actual_vs_pred_weekly.png` | Actual vs Predicted — weekly |
| 6 | `06_actual_vs_pred_monthly.png` | Actual vs Predicted — monthly |
| 7 | `07_scatter.png` | Scatter (actual vs predicted) with R² |
| 8 | `08_residual_distributions.png` | Residual histograms + Gaussian overlay |
| 9 | `09_residuals_over_time.png` | Residuals over test period |
| 10 | `10_metrics_barchart.png` | RMSE/MAE/MAPE across granularities |
| 11 | `11_error_heatmap.png` | MAE heatmap (Hour × Day-of-Week) |
| 12 | `12_fusion_weights.png` | Learned fusion weights (α per stream) |
| 13 | `13_daily_profile.png` | Average 24h prediction profile (mean ± σ) |
| 14 | `14_seasonal_errors.png` | Monthly MAE seasonal error analysis |

---

## Citation

```bibtex
@misc{power_load_generation_forecast_2025,
  title   = {Power-Load-Generation-Forecast: Two-Stream Network for
             Joint Household Energy and PV Solar Prediction},
  author  = {Alana Zisu},
  year    = {2025},
  url     = {https://github.com/alanazisu/Power-Load-Generation-Forecast}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
