"""
Data loading, cleaning, and normalisation.
"""

import pandas as pd
import numpy as np


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load UCI Appliances Energy dataset and clean it.
    
    Dataset: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
    Columns used: date, Appliances (target), T1-T9, RH_1-RH_9, T_out, RH_out
    """
    df = pd.read_csv(filepath)

    # ── 1. Parse datetime ──────────────────────────────────────
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df.set_index('date', inplace=True)

    # ── 2. Rename target column ────────────────────────────────
    df.rename(columns={'Appliances': 'energy_kwh'}, inplace=True)

    # ── 3. Drop lights column (mostly zeros, low signal) ───────
    if 'lights' in df.columns:
        df.drop(columns=['lights'], inplace=True)

    # ── 4. Handle missing values ───────────────────────────────
    # Forward fill gaps up to 3 hours, then drop remaining NaN
    df = df.ffill(limit=3).dropna()

    # ── 5. Remove outliers using IQR on target ─────────────────
    Q1 = df['energy_kwh'].quantile(0.05)
    Q3 = df['energy_kwh'].quantile(0.95)
    df = df[(df['energy_kwh'] >= Q1) & (df['energy_kwh'] <= Q3)]

    print(f"      Outliers removed. Remaining rows: {len(df):,}")
    return df