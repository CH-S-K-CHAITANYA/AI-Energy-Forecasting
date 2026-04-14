"""
AI-Powered Energy Consumption Forecasting System
Entry point — runs full pipeline end to end.
Run: python main.py
"""

import os
from src.preprocess import load_and_clean_data
from src.features import engineer_features
from src.model import train_model, save_model
from src.evaluate import evaluate_model
from src.forecast import generate_forecast
from src.visualize import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_forecast,
    plot_residuals
)

def main():
    print("=" * 60)
    print("  AI Energy Consumption Forecasting System")
    print("=" * 60)

    # ── Phase 1: Load & clean ──────────────────────────────────
    print("\n[1/6] Loading and cleaning data...")
    df = load_and_clean_data("data/raw/energydata_complete.csv")
    print(f"      Loaded {len(df):,} rows × {df.shape[1]} columns")

    # ── Phase 2: Feature engineering ──────────────────────────
    print("[2/6] Engineering features...")
    df = engineer_features(df)
    df.to_csv("data/processed/energy_features.csv", index=False)
    print(f"      Features: {list(df.columns)}")

    # ── Phase 3: Train model ───────────────────────────────────
    print("[3/6] Training Random Forest model...")
    model, X_train, X_test, y_train, y_test, scaler = train_model(df)
    save_model(model, scaler, "models/")
    print("      Model saved to models/rf_model.pkl")

    # ── Phase 4: Evaluate ──────────────────────────────────────
    print("[4/6] Evaluating model performance...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    print(f"      RMSE  : {metrics['rmse']:.2f} kWh")
    print(f"      MAE   : {metrics['mae']:.2f} kWh")
    print(f"      R²    : {metrics['r2']:.4f}")
    print(f"      MAPE  : {metrics['mape']:.2f}%")

    # ── Phase 5: Forecast ──────────────────────────────────────
    print("[5/6] Generating 7-day forecast...")
    forecast_df = generate_forecast(model, df, scaler, days=7)
    forecast_df.to_csv("outputs/predictions.csv", index=False)
    print("      Forecast saved to outputs/predictions.csv")

    # ── Phase 6: Visualize ─────────────────────────────────────
    print("[6/6] Generating visualizations...")
    plot_actual_vs_predicted(y_test, y_pred)
    plot_feature_importance(model, X_train.columns.tolist())
    plot_forecast(forecast_df)
    plot_residuals(y_test, y_pred)
    print("      Plots saved to outputs/")

    print("\n" + "=" * 60)
    print("  Pipeline complete! Open outputs/ to view results.")
    print("  Run:  streamlit run app.py   for the dashboard")
    print("=" * 60)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    main()