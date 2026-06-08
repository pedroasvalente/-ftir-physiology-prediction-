"""
PLS-R direct regression — standard chemometric baseline.

Reference: Wold et al. (2001) Chemometrics and Intelligent Laboratory Systems, 58(2), 109-130.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ftir_pred.config import RESULTS_DIR, TRAINING_DATA_PATH
from ftir_pred.data.config import REGRESSION_TARGETS, SAMPLE_TYPES, WATER_REGION, get_target_group
from ftir_pred.data.loader import filter_samples, get_ftir_columns, load_csv
from ftir_pred.data.splits import split_by_person


def _vip_scores(pls: PLSRegression) -> np.ndarray:
    T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p = W.shape[0]
    s = np.diag(T.T @ T @ Q.T @ Q)
    total_s = s.sum()
    if total_s == 0:
        return np.ones(p)
    col_norms = np.linalg.norm(W, axis=0, keepdims=True)
    col_norms = np.where(col_norms == 0, 1.0, col_norms)
    return np.sqrt(p * ((W / col_norms) ** 2 @ s) / total_s)


def run_plsr_cv(X, y, groups, max_components=15, n_splits=5) -> pd.DataFrame:
    cv = GroupKFold(n_splits=n_splits)
    rows = []
    for n_comp in range(1, max_components + 1):
        r2_folds, rmse_folds = [], []
        for tr, te in cv.split(X, y, groups=groups):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr])
            X_te = sc.transform(X[te])
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_tr, y[tr])
            y_pred = pls.predict(X_te).ravel()
            r2_folds.append(r2_score(y[te], y_pred))
            rmse_folds.append(float(np.sqrt(mean_squared_error(y[te], y_pred))))
        rows.append({
            "n_components": n_comp,
            "r2_cv":    round(float(np.mean(r2_folds)), 4),
            "r2_std":   round(float(np.std(r2_folds)),  4),
            "rmse_cv":  round(float(np.mean(rmse_folds)), 4),
            "rmse_std": round(float(np.std(rmse_folds)),  4),
        })
    return pd.DataFrame(rows)


def run_plsr_final(X_train, X_test, y_train, y_test, n_components) -> dict:
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_tr, y_train)
    y_pred = pls.predict(X_te).ravel()
    return {
        "r2":           round(float(r2_score(y_test, y_pred)), 4),
        "rmse":         round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "y_test":       y_test.tolist(),
        "y_pred":       y_pred.tolist(),
        "vip_scores":   _vip_scores(pls).tolist(),
        "n_components": n_components,
    }


def run_all_plsr(
    targets=None, sample_types=None, max_components=15,
    n_splits=5, timepoints=None, out_dir=None, data_path=None,
) -> dict:
    df = load_csv(data_path or str(TRAINING_DATA_PATH))
    wn_all = np.array([float(c) for c in get_ftir_columns(df).tolist()])
    water_mask = (wn_all < WATER_REGION[0]) | (wn_all > WATER_REGION[1])
    valid_wn = wn_all[water_mask]

    targets      = targets      or REGRESSION_TARGETS
    sample_types = sample_types or SAMPLE_TYPES
    out_dir      = out_dir or (RESULTS_DIR / "plsr_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    summary_rows: list[dict] = []

    for target in targets:
        all_results[target] = {}
        for st in sample_types:
            result = filter_samples(df, st, target, timepoints)
            if result is None:
                continue

            X, y, groups = result
            X_arr = X.values.astype(float)
            y_arr = y.values.astype(float)

            X_train, X_test, y_train, y_test, groups_train = split_by_person(
                X_arr, y_arr, groups.values, test_size=0.2
            )
            if len(y_test) < n_splits:
                continue

            cv_df = run_plsr_cv(X_train, y_train, groups_train,
                                max_components=max_components, n_splits=n_splits)
            best_nc = int(cv_df.loc[cv_df["r2_cv"].idxmax(), "n_components"])
            final = run_plsr_final(X_train, X_test, y_train, y_test, n_components=best_nc)

            vip_full = np.array(final["vip_scores"])
            vip_valid = vip_full[water_mask] if len(vip_full) == len(wn_all) else vip_full

            all_results[target][st] = {
                "r2":           final["r2"],
                "rmse":         final["rmse"],
                "n_components": best_nc,
                "n_train":      len(y_train),
                "n_test":       len(y_test),
                "target_group": get_target_group(target),
                "cv_curve":     cv_df.to_dict("records"),
                "y_test":       final["y_test"],
                "y_pred":       final["y_pred"],
                "wavenumbers":  valid_wn.tolist(),
                "vip_scores":   vip_valid.tolist(),
            }
            summary_rows.append({
                "target":       target,
                "target_group": get_target_group(target),
                "sample_type":  st,
                "model":        "PLS-R",
                "n_components": best_nc,
                "r2":           final["r2"],
                "rmse":         final["rmse"],
                "n_train":      len(y_train),
                "n_test":       len(y_test),
            })
            print(f"  {target}/{st}: R²={final['r2']:.3f}, n_comp={best_nc}")

    (out_dir / "plsr_results.json").write_text(json.dumps(all_results))
    pd.DataFrame(summary_rows).to_csv(out_dir / "plsr_summary.csv", index=False)
    print(f"\nSaved → {out_dir}")
    return all_results
