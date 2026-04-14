"""
Feature engineering: time features, lag features, rolling stats.
This is the most important step — good features = accurate model.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based, lag, and statistical features.
    
    These mimic what energy analysts hand-craft in industry:
    - Hour of day captures the daily demand curve
    - Lag features give the model recent consumption context
    - Rolling stats capture trend and volatility
    """
    df = df.copy()

    # ── Time features ──────────────────────────────────────────
    df['hour']       = df.index.hour
    df['day_of_week'] = df.index.dayofweek       # 0=Mon, 6=Sun
    df['month']      = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['quarter']    = df.index.quarter

    # Cyclical encoding of hour (preserves 23→0 continuity)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # ── Lag features ───────────────────────────────────────────
    df['lag_1']   = df['energy_kwh'].shift(1)    # 10 min ago
    df['lag_6']   = df['energy_kwh'].shift(6)    # 1 hour ago
    df['lag_144'] = df['energy_kwh'].shift(144)  # 1 day ago (10-min data)
    df['lag_1008']= df['energy_kwh'].shift(1008) # 1 week ago

    # ── Rolling statistics ─────────────────────────────────────
    df['rolling_mean_6h']  = df['energy_kwh'].shift(1).rolling(36).mean()
    df['rolling_std_6h']   = df['energy_kwh'].shift(1).rolling(36).std()
    df['rolling_mean_24h'] = df['energy_kwh'].shift(1).rolling(144).mean()
    df['rolling_max_24h']  = df['energy_kwh'].shift(1).rolling(144).max()

    # ── Interaction feature ────────────────────────────────────
    if 'T_out' in df.columns:
        df['temp_x_hour'] = df['T_out'] * df['hour_sin']

    # ── Drop rows with NaN from lag creation ───────────────────
    df.dropna(inplace=True)

    return df