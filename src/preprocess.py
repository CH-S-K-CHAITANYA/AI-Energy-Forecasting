"""
Data loading, cleaning, and normalization for Energy Forecasting
(Works with household power consumption dataset)
"""

import pandas as pd
import numpy as np


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess energy dataset.

    Expected columns:
    Date, Time, Global_active_power, Global_reactive_power,
    Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3
    """

    # ── 1. Load data ──────────────────────────────────────────
    df = pd.read_csv(filepath)

    # Clean column names (remove hidden spaces)
    df.columns = df.columns.str.strip()

    # ── 2. Combine Date + Time → datetime ─────────────────────
    df['datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d-%m-%Y %H:%M:%S'
    )

    df = df.sort_values('datetime').reset_index(drop=True)
    df.set_index('datetime', inplace=True)

    # ── 3. Drop original Date & Time ──────────────────────────
    df.drop(['Date', 'Time'], axis=1, inplace=True)

    # ── 4. Handle missing values ('?' → NaN) ──────────────────
    df.replace('?', np.nan, inplace=True)

    # ── 5. Convert all columns to numeric safely ──────────────
    df = df.apply(pd.to_numeric, errors='coerce')

    # ── 6. Rename target column ───────────────────────────────
    df.rename(columns={'Global_active_power': 'energy_kwh'}, inplace=True)

    # ── 7. Handle missing values properly ─────────────────────
    df = df.ffill(limit=3).dropna()

    # ── 8. Remove outliers (robust filtering) ─────────────────
    Q1 = df['energy_kwh'].quantile(0.05)
    Q3 = df['energy_kwh'].quantile(0.95)
    df = df[(df['energy_kwh'] >= Q1) & (df['energy_kwh'] <= Q3)]

    # ── 9. Feature engineering (important for ML) ─────────────
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    print(f"✅ Data cleaned successfully. Rows: {len(df):,}")

    return df