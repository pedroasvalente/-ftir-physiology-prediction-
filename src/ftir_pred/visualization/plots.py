import matplotlib.pyplot as plt
import numpy as np


def plot_predicted_vs_actual(y_true, y_pred, title: str = "", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3, s=40)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "r--", lw=1, label="Identity")
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_wavenumber_importance(wavenumbers, importances, title: str = "", top_n: int = 30, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(wavenumbers, importances, alpha=0.7)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Normalised importance")
    ax.invert_xaxis()
    if title:
        ax.set_title(title)
    return ax


def plot_bland_altman(mean_vals, diff, bias, loa_upper, loa_lower, title: str = "", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(mean_vals, diff, alpha=0.6, edgecolors="k", linewidths=0.3, s=40)
    ax.axhline(bias, color="blue", lw=1.5, label=f"Bias = {bias:.3f}")
    ax.axhline(loa_upper, color="red", lw=1, linestyle="--", label=f"+1.96 SD = {loa_upper:.3f}")
    ax.axhline(loa_lower, color="red", lw=1, linestyle="--", label=f"−1.96 SD = {loa_lower:.3f}")
    ax.set_xlabel("Mean of measured and predicted")
    ax.set_ylabel("Difference (measured − predicted)")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7)
    return ax
