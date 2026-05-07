"""
data/generate_datasets.py
=========================
Generates realistic synthetic datasets for:
  1. Household energy consumption (kWh/h)
  2. PV solar generation (kWh/h)

Both datasets share the same hourly timestamp index across one full year.
The simulation uses physically-motivated models:
  - Consumption: base load + morning/evening peaks + day-of-week + seasonal
  - Solar: clear-sky irradiance model + cloud cover attenuation + temperature correction

Run standalone:
    python data/generate_datasets.py
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET, CONSUMPTION_CSV, SOLAR_CSV


# ── Helpers ────────────────────────────────────────────────────────────────

def _solar_angle(hour: float, day_of_year: int, latitude_deg: float = 33.45) -> float:
    """Compute solar elevation angle (degrees) — simplified Spencer model."""
    B   = 2 * np.pi * (day_of_year - 1) / 365
    Et  = (0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B)
           - 0.014615 * np.cos(2*B) - 0.04089 * np.sin(2*B))
    delta = (0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
             - 0.006758 * np.cos(2*B) + 0.000907 * np.sin(2*B))
    lat_rad  = np.radians(latitude_deg)
    hour_ang = np.radians(15 * (hour - 12 + Et * 12 / np.pi))
    sin_elev = (np.sin(lat_rad) * np.sin(delta)
                + np.cos(lat_rad) * np.cos(delta) * np.cos(hour_ang))
    return np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))


def _is_holiday(dt: pd.Timestamp) -> bool:
    """Simplified UK-style public holidays."""
    holidays = {
        (1,  1), (4, 7), (4, 10), (5, 6), (5, 27),
        (8, 26), (12, 25), (12, 26),
    }
    return (dt.month, dt.day) in holidays


# ── Main generator ─────────────────────────────────────────────────────────

def generate_datasets(cfg: dict = DATASET,
                      consumption_path: str = CONSUMPTION_CSV,
                      solar_path: str       = SOLAR_CSV,
                      verbose: bool         = True) -> pd.DataFrame:
    """
    Generate and save both datasets.

    Returns
    -------
    pd.DataFrame  Combined DataFrame with all features + targets.
    """
    rng = np.random.default_rng(cfg["random_seed"])

    # ── Timestamp index ────────────────────────────────────────────────────
    idx = pd.date_range(cfg["start_date"], cfg["end_date"], freq=cfg["freq"])
    n   = len(idx)
    if verbose:
        print(f"[DataGen] Generating {n:,} hourly samples "
              f"({cfg['start_date']} → {cfg['end_date']})")

    df = pd.DataFrame(index=idx)
    df.index.name = "timestamp"

    # ── Calendar features ─────────────────────────────────────────────────
    df["hour"]        = df.index.hour.astype(float)
    df["day_of_week"] = df.index.dayofweek.astype(float)   # 0=Mon
    df["day_of_year"] = df.index.dayofyear.astype(float)
    df["month"]       = df.index.month.astype(float)
    df["week"]        = df.index.isocalendar().week.astype(float)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(float)
    df["is_holiday"]  = [float(_is_holiday(t)) for t in df.index]

    # Cyclical encodings
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["day_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # ── Meteorological simulation ──────────────────────────────────────────
    # Temperature: seasonal + diurnal + noise
    seasonal_T = 12 * np.sin(2 * np.pi * (df["day_of_year"] - 80) / 365)
    diurnal_T  = 4  * np.sin(2 * np.pi * (df["hour"] - 6) / 24)
    df["temperature"]  = 12 + seasonal_T + diurnal_T + rng.normal(0, 1.5, n)

    # Humidity: anti-correlated with temperature
    df["humidity"]     = np.clip(
        70 - 0.8 * df["temperature"] + rng.normal(0, 8, n), 20, 100)

    # Cloud cover: autoregressive with seasonal bias
    cloud = np.zeros(n)
    cloud[0] = rng.uniform(0, 1)
    for i in range(1, n):
        cloud[i] = np.clip(
            0.92 * cloud[i-1]
            + (0.4 + 0.2 * np.sin(2*np.pi*(df["day_of_year"].iloc[i]-270)/365))
            * rng.normal(0, 0.15), 0, 1)
    df["cloud_cover"] = cloud

    # Solar irradiance (W/m²)
    elev_angles = np.array([
        _solar_angle(df["hour"].iloc[i], df["day_of_year"].iloc[i])
        for i in range(n)
    ])
    clear_sky_irr = np.where(
        elev_angles > 0,
        1000 * np.sin(np.radians(np.clip(elev_angles, 0, 90))),
        0.0
    )
    df["solar_irradiance"] = clear_sky_irr * (1 - 0.75 * df["cloud_cover"]) * \
                              np.clip(1 + rng.normal(0, 0.05, n), 0.8, 1.2)

    # Wind speed: Weibull-distributed, seasonal
    df["wind_speed"] = (rng.weibull(2, n)
                        * (4 + 2 * np.sin(2*np.pi*(df["day_of_year"]-300)/365)))

    # ── Household energy consumption (kWh/h) ──────────────────────────────
    base  = cfg["base_load"]

    # Diurnal profile
    morn  = cfg["peak_morning"] * np.exp(-0.5*((df["hour"] - 7.5) / 1.2)**2)
    eve   = cfg["peak_evening"] * np.exp(-0.5*((df["hour"] - 19.5)/ 2.0)**2)
    night = 0.15 * np.exp(-0.5*((df["hour"] - 0) / 2.0)**2)

    # Weekend / holiday boost
    wkend_boost = 0.15 * df["is_weekend"] + 0.10 * df["is_holiday"]

    # Seasonal: heating in winter, cooling in summer
    seasonal_C  = (0.25 * np.exp(-0.5*((df["temperature"] - 5)  / 5)**2)   # heating
                 + 0.30 * np.exp(-0.5*((df["temperature"] - 30) / 4)**2))  # cooling

    # Temperature sensitivity: HVAC load
    hvac = np.clip(0.04 * np.abs(df["temperature"] - 18), 0, 0.6)

    consumption_raw = (base + morn + eve + night + wkend_boost
                       + seasonal_C + hvac
                       + rng.normal(0, cfg["noise_std"], n))
    consumption_raw = np.clip(np.array(consumption_raw), 0.05, 4.0)

    # Add autocorrelation (sluggish appliance behaviour)
    consumption = np.zeros(n)
    consumption[0] = float(consumption_raw[0])
    for i in range(1, n):
        consumption[i] = 0.30 * consumption[i-1] + 0.70 * float(consumption_raw[i])
    # placeholder.iloc[0]
    for i in range(1, n):
        consumption[i] = 0.30 * consumption[i-1] + 0.70 * consumption_raw[i]

    df["consumption_kwh"] = consumption

    # ── PV Solar generation (kWh/h) ───────────────────────────────────────
    panel_area = cfg["panel_area_m2"]
    eff        = cfg["panel_eff"]
    # Temperature correction: efficiency drops ~0.4%/°C above 25°C
    temp_corr  = 1 - 0.004 * np.maximum(df["temperature"] - 25, 0)
    solar_raw  = (df["solar_irradiance"] / 1000  # kW/m²
                  * panel_area * eff * temp_corr
                  * np.clip(1 + rng.normal(0, 0.04, n), 0.85, 1.15))
    df["solar_kwh"] = np.clip(np.array(solar_raw), 0, cfg["peak_solar"])

    # ── Lag & rolling features ────────────────────────────────────────────
    for h in [1, 2, 24, 168]:
        df[f"consumption_lag_{h}h"] = df["consumption_kwh"].shift(h)
        df[f"solar_lag_{h}h"]       = df["solar_kwh"].shift(h)

    df["consumption_roll_mean_24h"] = (df["consumption_kwh"]
                                       .rolling(24, min_periods=1).mean())
    df["consumption_roll_std_24h"]  = (df["consumption_kwh"]
                                       .rolling(24, min_periods=1).std().fillna(0))
    df["consumption_roll_mean_7d"]  = (df["consumption_kwh"]
                                       .rolling(168, min_periods=1).mean())
    df["solar_roll_mean_24h"]       = (df["solar_kwh"]
                                       .rolling(24, min_periods=1).mean())
    df["solar_roll_std_24h"]        = (df["solar_kwh"]
                                       .rolling(24, min_periods=1).std().fillna(0))

    # Drop NaN rows from lag creation
    df.dropna(inplace=True)

    # ── Save separate CSVs ────────────────────────────────────────────────
    consumption_cols = (["consumption_kwh", "temperature", "humidity",
                          "cloud_cover", "wind_speed", "is_weekend", "is_holiday"]
                        + [c for c in df.columns if "consumption_lag" in c or
                           "consumption_roll" in c]
                        + ["hour_sin","hour_cos","day_sin","day_cos",
                           "month_sin","month_cos"])
    solar_cols = (["solar_kwh", "solar_irradiance", "temperature", "cloud_cover",
                   "wind_speed", "is_weekend"]
                  + [c for c in df.columns if "solar_lag" in c or "solar_roll" in c]
                  + ["hour_sin","hour_cos","day_sin","day_cos",
                     "month_sin","month_cos"])

    df[consumption_cols].to_csv(consumption_path)
    df[solar_cols].to_csv(solar_path)

    if verbose:
        print(f"[DataGen] Consumption saved → {consumption_path}")
        print(f"[DataGen] Solar saved       → {solar_path}")
        print(f"[DataGen] Dataset shape: {df.shape}")
        print(f"[DataGen] Consumption: mean={df['consumption_kwh'].mean():.3f} kWh, "
              f"max={df['consumption_kwh'].max():.3f} kWh")
        print(f"[DataGen] Solar:       mean={df['solar_kwh'].mean():.3f} kWh, "
              f"max={df['solar_kwh'].max():.3f} kWh")

    return df


if __name__ == "__main__":
    generate_datasets(verbose=True)
