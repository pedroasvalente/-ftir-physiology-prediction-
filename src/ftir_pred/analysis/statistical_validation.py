import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Bland-Altman agreement analysis.

    Returns mean difference (bias), limits of agreement (±1.96 SD),
    and arrays for the plot.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    diff = y_true - y_pred
    mean_vals = (y_true + y_pred) / 2
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    return {
        "mean": mean_vals.tolist(),
        "diff": diff.tolist(),
        "bias": bias,
        "loa_upper": loa_upper,
        "loa_lower": loa_lower,
        "sd": sd,
    }


def pearson_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r, p = pearsonr(y_true, y_pred)
    return {"pearson_r": float(r), "pearson_p": float(p)}


def validation_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise Pearson r, R², RMSE and 95% CI per target × sample_type × model.
    Filters to best model per (target, sample_type) by R².
    """
    non_baseline = results_df[~results_df["is_baseline"]]
    idx = non_baseline.groupby(["target", "sample_type"])["r2"].idxmax()
    best = non_baseline.loc[idx].copy()
    return best[
        ["target", "target_group", "sample_type", "model", "search",
         "r2", "r2_ci95_low", "r2_ci95_high",
         "rmse", "rmse_ci95_low", "rmse_ci95_high",
         "mae", "mape", "pearson_r", "pearson_p", "n_test"]
    ].reset_index(drop=True)
