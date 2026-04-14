"""
Model training with RandomForestRegressor.
Includes time-series cross-validation and hyperparameter tuning.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


# Features used for training (exclude the target)
FEATURE_COLS = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'quarter',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'lag_1', 'lag_6', 'lag_144', 'lag_1008',
    'rolling_mean_6h', 'rolling_std_6h',
    'rolling_mean_24h', 'rolling_max_24h'
]

TARGET_COL = 'energy_kwh'


def train_model(df: pd.DataFrame):
    """
    Train a RandomForestRegressor on time-series energy data.
    Uses chronological 80/20 split — never shuffle time-series data!
    """
    # Filter to available feature columns
    available_feats = [f for f in FEATURE_COLS if f in df.columns]
    
    # Optional: add temperature columns if present
    temp_cols = [c for c in df.columns if c.startswith('T') and c != TARGET_COL]
    all_feats = available_feats + temp_cols[:3]  # use first 3 temp sensors

    X = df[all_feats]
    y = df[TARGET_COL]

    # ── Chronological split (80/20) ────────────────────────────
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # ── Scale features ─────────────────────────────────────────
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=all_feats)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=all_feats)

    # ── Model definition ───────────────────────────────────────
    model = RandomForestRegressor(
        n_estimators=200,      # 200 trees for stable predictions
        max_depth=15,          # prevent overfitting
        min_samples_leaf=5,
        n_jobs=-1,             # use all CPU cores
        random_state=42
    )

    # ── Time-series cross-validation ───────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=tscv, scoring='r2', n_jobs=-1
    )
    print(f"      CV R² scores: {[round(s,3) for s in cv_scores]}")
    print(f"      Mean CV R²  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Final training ─────────────────────────────────────────
    model.fit(X_train_scaled, y_train)

    return model, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_model(model, scaler, model_dir: str):
    """Save trained model and scaler to disk."""
    joblib.dump(model,  f"{model_dir}/rf_model.pkl")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")


def load_model(model_dir: str):
    """Load trained model and scaler from disk."""
    model  = joblib.load(f"{model_dir}/rf_model.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    return model, scaler