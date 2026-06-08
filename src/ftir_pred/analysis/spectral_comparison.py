"""
Spectral group comparison: high-fitness vs low-fitness participants.

For each wavenumber, tests whether absorbance differs between VO2max groups using:
  - Mann-Whitney U (non-parametric, no normality assumption)
  - AUC (receiver operating characteristic, quantifies group separation)

Benjamini-Hochberg FDR correction applied across all wavenumbers.

Primary methodological reference:
  Movasaghi, Z., Rehman, S., & Rehman, I.U. (2008). Applied Spectroscopy Reviews,
  43(2), 134–179. https://doi.org/10.1080/05704920701829043
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from ftir_pred.config import RESULTS_DIR, TRAINING_DATA_PATH
from ftir_pred.data.loader import filter_samples, get_ftir_columns, load_csv


def _bh_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    q = pvalues * n / ranked
    # Enforce monotonicity from right
    q_corrected = np.minimum.accumulate(q[order][::-1])[::-1]
    q_final = np.empty(n)
    q_final[order] = q_corrected
    return np.clip(q_final, 0, 1)


def _auc_for_wavenumber(
    x_high: np.ndarray,
    x_low: np.ndarray,
) -> float:
    """
    Compute the AUC treating wavenumber absorbance as a binary classifier score.

    Returns AUC in [0, 1]; values > 0.5 mean high group has higher absorbance.
    """
    labels = np.concatenate([np.ones(len(x_high)), np.zeros(len(x_low))])
    scores = np.concatenate([x_high, x_low])
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return 0.5
    return float(roc_auc_score(labels, scores))


def run_spectral_comparison(
    sample_type: str,
    group_column: str = "vo2max_classes_simplified",
    high_label: float = 3.0,
    low_label: float = 1.0,
    timepoints: list[int] | None = None,
    alpha: float = 0.05,
    data_path: str | None = None,
) -> pd.DataFrame:
    """
    Perform wavenumber-level Mann-Whitney U test + AUC for one sample matrix.

    Parameters
    ----------
    sample_type   : one of CAPILAR, PLASMA, SALIVA, SERUM, URINE
    group_column  : column that contains the fitness group label
    high_label    : label value for the high-fitness group
    low_label     : label value for the low-fitness group
    timepoints    : restrict to specific timepoints (None = all)
    alpha         : significance threshold for adjusted p-values (FDR)
    data_path     : override default training data path

    Returns
    -------
    DataFrame with one row per wavenumber:
      wavenumber, u_stat, p_value, q_value, auc, n_high, n_low, significant
    """
    path = data_path or str(TRAINING_DATA_PATH)
    df = load_csv(path)

    sub = df[df["sample_type"] == sample_type].copy()
    if timepoints is not None:
        sub = sub[sub["timepoint"].isin(timepoints)]

    if group_column not in sub.columns:
        raise ValueError(f"Column '{group_column}' not found in dataset.")

    ftir_cols = get_ftir_columns(df)
    wavenumbers = np.array([float(c) for c in ftir_cols])

    mask_high = sub[group_column] == high_label
    mask_low  = sub[group_column] == low_label

    sub_high = sub[mask_high][ftir_cols].values.astype(float)
    sub_low  = sub[mask_low][ftir_cols].values.astype(float)

    n_high = len(sub_high)
    n_low  = len(sub_low)

    if n_high < 3 or n_low < 3:
        raise ValueError(
            f"{sample_type}: too few samples — high={n_high}, low={n_low}. "
            "Need ≥ 3 per group."
        )

    u_stats  = np.empty(len(wavenumbers))
    p_values = np.empty(len(wavenumbers))
    aucs     = np.empty(len(wavenumbers))

    for i in range(len(wavenumbers)):
        col_high = sub_high[:, i]
        col_low  = sub_low[:, i]
        col_high = col_high[np.isfinite(col_high)]
        col_low  = col_low[np.isfinite(col_low)]
        if len(col_high) < 2 or len(col_low) < 2:
            u_stats[i], p_values[i], aucs[i] = np.nan, 1.0, 0.5
            continue
        res = mannwhitneyu(col_high, col_low, alternative="two-sided")
        u_stats[i]  = res.statistic
        p_values[i] = res.pvalue
        aucs[i]     = _auc_for_wavenumber(col_high, col_low)

    q_values = _bh_correction(p_values)

    return pd.DataFrame({
        "wavenumber": wavenumbers,
        "u_stat":     np.round(u_stats, 2),
        "p_value":    np.round(p_values, 6),
        "q_value":    np.round(q_values, 6),
        "auc":        np.round(aucs, 4),
        "n_high":     n_high,
        "n_low":      n_low,
        "significant": q_values < alpha,
    })


def run_all_matrices(
    sample_types: list[str] | None = None,
    group_column: str = "vo2max_classes_simplified",
    high_label: float = 3.0,
    low_label: float = 1.0,
    timepoints: list[int] | None = None,
    alpha: float = 0.05,
    out_dir: Path | None = None,
    data_path: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run spectral comparison for all (or selected) sample matrices.
    Saves per-matrix CSV and a combined JSON to out_dir.

    Returns dict mapping sample_type → results DataFrame.
    """
    from ftir_pred.data.config import SAMPLE_TYPES

    matrices = sample_types or SAMPLE_TYPES
    out_dir  = out_dir or (RESULTS_DIR / "spectral_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined: dict[str, list] = {}
    results:  dict[str, pd.DataFrame] = {}

    for st in matrices:
        try:
            res = run_spectral_comparison(
                st,
                group_column=group_column,
                high_label=high_label,
                low_label=low_label,
                timepoints=timepoints,
                alpha=alpha,
                data_path=data_path,
            )
        except ValueError as exc:
            print(f"  Skipping {st}: {exc}")
            continue

        res["sample_type"] = st
        results[st] = res
        res.to_csv(out_dir / f"comparison_{st}.csv", index=False)

        n_sig = res["significant"].sum()
        print(
            f"  {st}: n_high={res['n_high'].iloc[0]}, "
            f"n_low={res['n_low'].iloc[0]}, "
            f"significant wavenumbers={n_sig}"
        )

        combined[st] = {
            "wavenumbers": res["wavenumber"].tolist(),
            "u_stat":      res["u_stat"].tolist(),
            "p_value":     res["p_value"].tolist(),
            "q_value":     res["q_value"].tolist(),
            "auc":         res["auc"].tolist(),
            "n_high":      int(res["n_high"].iloc[0]),
            "n_low":       int(res["n_low"].iloc[0]),
            "group_column": group_column,
            "high_label":   high_label,
            "low_label":    low_label,
        }

    out_path = out_dir / "spectral_comparison.json"
    with open(out_path, "w") as fh:
        json.dump(combined, fh)
    print(f"\nSaved → {out_path}")

    return results
