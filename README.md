# Power-Load-Generation-Forecasting

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0+-orange?logo=numpy)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Energy-TSN** -A Two-Stream Network for joint prediction of household energy **consumption** and PV solar **generation**, capturing both spatial cross-feature patterns and temporal time-series dynamics through parallel independent streams fused at inference.

---

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Citation](#citation)
- [License](#license)

---

## Overview

**Energy-TSN** addresses the joint prediction problem of:
1. **Household energy consumption** (kWh/h)-driven by occupant behaviour, temperature, and appliance usage
2. **PV solar generation** (kWh/h)-driven by solar irradiance, cloud cover, and panel efficiency

Unlike single-stream models, Energy-TSN explicitly separates and models:
- **Spatial patterns**-cross-feature correlations between meteorological, calendar, and lagged energy variables
- **Temporal patterns**-sequential dynamics of the energy time-series over a 24-hour look-back window
---

## Installation

```bash
git clone https://github.com/alanazisu/Power-Load-Generation-Forecast.git
cd Power-Load-Generation-Forecast
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, NumPy, pandas, scikit-learn, scipy, matplotlib, seaborn

---

## Datasets

### Household Energy Consumption
- https://www.kaggle.com/datasets/samxsam/household-energy-consumption

### Solar Power Generation Data
- {https://dkasolarcentre.com.au/download?location=alice-springs}
---

## Citation

```bibtex
@misc{power_load_generation_forecast_2026,
  title   = {A Multi-Functional Spatiotemporal Learning Framework for Power Load and Photovoltaic Generation Forecasting in Net-Zero Energy Buildings},
  author  = {Sultan Alanazi},
  year    = {2026},
  url     = {https://github.com/alanazisu/Power-Load-Generation-Forecast}
}
```

---

## License

MIT License-see [LICENSE](LICENSE) for details.
