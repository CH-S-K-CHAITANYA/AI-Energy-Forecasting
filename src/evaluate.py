"""
Model evaluation — RMSE, MAE, MAPE, R² computation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X_test, y_test) -> tuple:
    """
    Generate predictions and compute all evaluation metrics.
    Returns dict of metrics and array of predictions.
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

    metrics = {
        'rmse': rmse,
        'mae':  mae,
        'r2':   r2,
        'mape': mape
    }

    # Print summary table
    print("\n  ┌─────────────────────────────┐")
    print("  │   Model Evaluation Summary  │")
    print("  ├─────────────────────────────┤")
    print(f"  │  RMSE   : {rmse:>10.2f} kWh   │")
    print(f"  │  MAE    : {mae:>10.2f} kWh   │")
    print(f"  │  R²     : {r2:>14.4f}   │")
    print(f"  │  MAPE   : {mape:>10.2f} %     │")
    print("  └─────────────────────────────┘\n")

    return metrics, y_pred