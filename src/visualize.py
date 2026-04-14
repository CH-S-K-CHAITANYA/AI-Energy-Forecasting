"""
All visualization functions — saves publication-quality charts to outputs/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Style config ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'text.color':       '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'grid.color':       '#21262d',
    'axes.labelcolor':  '#c9d1d9',
    'font.family':      'monospace',
})
ACCENT  = '#1f6feb'
GREEN   = '#3fb950'
ORANGE  = '#d29922'
RED     = '#f85149'


def plot_actual_vs_predicted(y_test, y_pred, save_path='outputs/actual_vs_predicted.png'):
    """Line chart comparing actual vs model predictions."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Actual vs Predicted Energy Consumption', color='#e6edf3', fontsize=14, y=0.98)

    y_test_arr = np.array(y_test)
    # Main chart
    ax = axes[0]
    x = range(len(y_test_arr))
    ax.plot(x, y_test_arr, color=GREEN,  linewidth=1.2, label='Actual', alpha=0.9)
    ax.plot(x, y_pred,     color=ACCENT, linewidth=1.2, label='Predicted', linestyle='--', alpha=0.85)
    ax.fill_between(x, y_pred*0.95, y_pred*1.05, color=ACCENT, alpha=0.08, label='±5% band')
    ax.set_ylabel('Energy (kWh)', fontsize=11)
    ax.legend(fontsize=10, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"R² = {r2_score_calc(y_test_arr, y_pred):.4f}  |  RMSE = {rmse_calc(y_test_arr, y_pred):.2f} kWh",
                 color='#8b949e', fontsize=10)

    # Residuals
    residuals = y_test_arr - y_pred
    axes[1].axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
    axes[1].bar(x, residuals, color=[RED if r < 0 else GREEN for r in residuals], width=1.0, alpha=0.6)
    axes[1].set_ylabel('Residual', fontsize=10)
    axes[1].set_xlabel('Time Steps', fontsize=10)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {save_path}")


def plot_feature_importance(model, feature_names, save_path='outputs/feature_importance.png'):
    """Horizontal bar chart of top 15 feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle('Feature Importance — Random Forest', color='#e6edf3', fontsize=13)

    colors = [ACCENT if i == indices[-1] else '#1f6feb88' for i in indices]
    bars = ax.barh(range(len(indices)), importances[indices], color=colors, edgecolor='none', height=0.65)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.grid(True, axis='x', alpha=0.3)

    for bar, imp in zip(bars, importances[indices]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', fontsize=9, color='#8b949e')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {save_path}")


def plot_forecast(forecast_df, save_path='outputs/forecast_7day.png'):
    """7-day forecast with confidence interval band."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle('7-Day Energy Consumption Forecast', color='#e6edf3', fontsize=13)

    x = range(len(forecast_df))
    dates = [str(d) for d in forecast_df['date']]

    ax.plot(x, forecast_df['forecast_kwh'], color=ACCENT, linewidth=2, marker='o',
            markersize=6, label='Forecast')
    ax.fill_between(x, forecast_df['lower_bound'], forecast_df['upper_bound'],
                    alpha=0.15, color=ACCENT, label='Confidence interval (±8%)')

    # Annotate peak
    peak_idx = forecast_df['forecast_kwh'].idxmax()
    ax.annotate(f"Peak\n{forecast_df['forecast_kwh'].max():.0f} kWh",
                xy=(list(forecast_df.index).index(peak_idx), forecast_df['forecast_kwh'].max()),
                fontsize=9, color=ORANGE,
                xytext=(0, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Energy (kWh)', fontsize=11)
    ax.legend(fontsize=10, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {save_path}")


def plot_residuals(y_test, y_pred, save_path='outputs/residuals.png'):
    """Residual distribution histogram."""
    residuals = np.array(y_test) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=50, color=ACCENT, alpha=0.7, edgecolor='none')
    ax.axvline(0, color=RED, linewidth=1.5, linestyle='--')
    ax.set_xlabel('Residual (Actual − Predicted)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Residual Distribution', color='#e6edf3', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Helper functions ──────────────────────────────────────────
def rmse_calc(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score_calc(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)