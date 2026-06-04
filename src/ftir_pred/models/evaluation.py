import math
import warnings

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 500,
    confidence: float = 0.95,
    random_state: int = 52,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    r2_boot, rmse_boot, mae_boot = [], [], []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        r2_boot.append(r2_score(yt, yp))
        rmse_boot.append(math.sqrt(mean_squared_error(yt, yp)))
        mae_boot.append(mean_absolute_error(yt, yp))

    alpha = (1 - confidence) / 2

    def ci(values):
        return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))

    return {"r2": ci(r2_boot), "rmse": ci(rmse_boot), "mae": ci(mae_boot)}


def evaluate(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 500,
) -> dict:
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    mape = _mape(y_test, y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pearson_r, pearson_p = float(pearsonr(y_test, y_pred).statistic), float(pearsonr(y_test, y_pred).pvalue)
        except Exception:
            pearson_r, pearson_p = float("nan"), float("nan")

    ci = _bootstrap_ci(y_test, y_pred, n_resamples=n_bootstrap)

    return {
        "r2": r2,
        "r2_ci95_low": ci["r2"][0],
        "r2_ci95_high": ci["r2"][1],
        "rmse": rmse,
        "rmse_ci95_low": ci["rmse"][0],
        "rmse_ci95_high": ci["rmse"][1],
        "mae": mae,
        "mae_ci95_low": ci["mae"][0],
        "mae_ci95_high": ci["mae"][1],
        "mape": mape,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "n_test": len(y_test),
    }
