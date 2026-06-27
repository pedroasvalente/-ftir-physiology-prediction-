"""
Spectral region comparison: high vs low VO2max.
VIP-guided regions → trapezoidal AUC per sample → Mann-Whitney U.

Reference: Movasaghi et al. (2008) Applied Spectroscopy Reviews 43(2), 134-179.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import mannwhitneyu

from ftir_pred.analysis.wavenumber_reference import assign_band
from ftir_pred.config import RESULTS_DIR, TRAINING_DATA_PATH
from ftir_pred.data.loader import get_ftir_columns, load_csv


_SIG_LABELS = {1e-4: "****", 1e-3: "***", 1e-2: "**", 5e-2: "*"}


def _sig(p: float) -> str:
    for threshold, label in _SIG_LABELS.items():
        if p < threshold:
            return label
    return "ns"


def find_vip_regions(
    wavenumbers: np.ndarray,
    vip: np.ndarray,
    top_pct: float = 20.0,
    min_consecutive: int = 5,
    max_gap_cm: float = 20.0,
) -> list[dict]:
    water_mask = (wavenumbers < 1850) | (wavenumbers > 2500)
    wn = wavenumbers[water_mask]
    vi = vip[water_mask]

    order = np.argsort(wn)
    wn, vi = wn[order], vi[order]

    threshold = np.percentile(vi, 100 - top_pct)
    above = vi >= threshold

    # Treat large spectral gaps (e.g. the excluded water region) as boundaries
    gaps = np.diff(wn) > max_gap_cm
    boundary = np.concatenate([[False], gaps])

    regions, i = [], 0
    while i < len(above):
        if above[i] and not boundary[i]:
            j = i
            while j < len(above) and above[j] and not boundary[j]:
                j += 1
            if (j - i) >= min_consecutive:
                lo, hi = float(wn[i]), float(wn[j - 1])
                band, bio = assign_band(lo, hi)
                regions.append({
                    "lo": lo, "hi": hi,
                    "label": f"{lo:.0f}–{hi:.0f} cm⁻¹",
                    "max_vip": float(vi[i:j].max()),
                    "mean_vip": float(vi[i:j].mean()),
                    "band_assignment": band,
                    "biochemical_origin": bio,
                })
            i = j
        else:
            i += 1
    return regions


def _region_aucs(X: np.ndarray, wavenumbers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (wavenumbers >= lo) & (wavenumbers <= hi)
    if mask.sum() < 2:
        return np.zeros(X.shape[0])
    order = np.argsort(wavenumbers[mask])
    wn_s = wavenumbers[mask][order]
    X_s = X[:, mask][:, order]
    return np.array([np.abs(trapezoid(X_s[k], wn_s)) for k in range(X.shape[0])])


def _bh_correct(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    p_arr = np.array(p_values)
    p_adj = np.empty(n)
    cummin = np.inf
    for k in range(n - 1, -1, -1):
        i = order[k]
        cummin = min(cummin, p_arr[i] * n / (k + 1))
        p_adj[i] = min(cummin, 1.0)
    return p_adj.tolist()


def compare_groups(
    X: np.ndarray,
    wavenumbers: np.ndarray,
    vip: np.ndarray,
    group_labels: np.ndarray,
    high_label,
    low_label,
    person_codes: np.ndarray | None = None,
    top_pct: float = 20.0,
    min_consecutive: int = 5,
) -> tuple[list[dict], list[dict]]:
    regions = find_vip_regions(wavenumbers, vip, top_pct=top_pct, min_consecutive=min_consecutive)
    results = []

    for reg in regions:
        lo, hi = reg["lo"], reg["hi"]
        aucs_all = _region_aucs(X, wavenumbers, lo, hi)

        if person_codes is not None:
            # Aggregate to person level: mean AUC per person — avoids pseudo-replication
            # from repeated spectra of the same individual.
            import pandas as pd
            tmp = pd.DataFrame({
                "person": person_codes,
                "group":  group_labels,
                "auc":    aucs_all,
            })
            agg = tmp.groupby(["person", "group"])["auc"].mean().reset_index()
            high_aucs = agg.loc[agg["group"] == high_label, "auc"].values
            low_aucs  = agg.loc[agg["group"] == low_label,  "auc"].values
        else:
            high_aucs = aucs_all[group_labels == high_label]
            low_aucs  = aucs_all[group_labels == low_label]

        if len(high_aucs) < 3 or len(low_aucs) < 3:
            continue

        _, p = mannwhitneyu(high_aucs, low_aucs, alternative="two-sided")
        results.append({
            **reg,
            "n_high":      len(high_aucs),
            "n_low":       len(low_aucs),
            "median_high": float(np.median(high_aucs)),
            "median_low":  float(np.median(low_aucs)),
            "p_value":     float(p),
            "p_value_adj": None,  # filled below after BH correction
            "sig":         "",    # filled below
            "aucs_high":   high_aucs.tolist(),
            "aucs_low":    low_aucs.tolist(),
        })

    # Benjamini-Hochberg FDR correction across all regions within this matrix
    if results:
        p_raw = [r["p_value"] for r in results]
        p_adj = _bh_correct(p_raw)
        for r, pa in zip(results, p_adj):
            r["p_value_adj"] = float(pa)
            r["sig"] = _sig(pa)

    sig_only = [r for r in results if r["sig"] != "ns"]
    return results, sig_only


def _load_best_vip(sample_type: str, wavenumbers: np.ndarray, vip_file=None) -> np.ndarray | None:
    """Load VIP scores for sample_type: ML run first, fall back to PLS-R."""
    ml_path = vip_file or (RESULTS_DIR / "study_regression_v1" / "wavenumber_importance.json")
    if ml_path.exists():
        ml_data = json.loads(ml_path.read_text())
        for k, entry in ml_data.items():
            if sample_type in k:
                imp = np.array(entry["importances"])
                if imp.max() > 0:
                    return imp

    plsr_path = RESULTS_DIR / "plsr_results" / "plsr_results.json"
    if plsr_path.exists():
        plsr_data = json.loads(plsr_path.read_text())
        best_r2, best_vip = -np.inf, None
        for target_results in plsr_data.values():
            if sample_type in target_results:
                entry = target_results[sample_type]
                if entry.get("r2", -np.inf) > best_r2:
                    best_r2 = entry["r2"]
                    best_vip = np.array(entry["vip_scores"])
        if best_vip is not None:
            return best_vip

    return None


def run_spectral_comparison(
    sample_type: str,
    vip_file: Path | None = None,
    group_column: str = "vo2max_classes_simplified",
    high_label: float = 3.0,
    low_label: float = 1.0,
    timepoints: list[int] | None = None,
    top_pct: float = 20.0,
    min_consecutive: int = 5,
    data_path: str | None = None,
) -> dict:
    df = load_csv(data_path or str(TRAINING_DATA_PATH))
    ftir_cols = get_ftir_columns(df)
    wavenumbers = np.array([float(c) for c in ftir_cols])

    sub = df[df["sample_type"] == sample_type].copy()
    if timepoints:
        sub = sub[sub["timepoint"].isin(timepoints)]

    valid_rows = sub[group_column].notna() & sub[ftir_cols].notna().all(axis=1)
    sub = sub[valid_rows]
    X = sub[ftir_cols].values.astype(float)
    group_arr   = sub[group_column].values
    person_arr  = sub["person_code"].values if "person_code" in sub.columns else None

    vip_imp = _load_best_vip(sample_type, wavenumbers, vip_file)
    if vip_imp is None:
        raise ValueError(f"No valid VIP scores found for {sample_type}.")

    from ftir_pred.data.config import WATER_REGION
    water_mask = (wavenumbers < WATER_REGION[0]) | (wavenumbers > WATER_REGION[1])
    if len(vip_imp) == wavenumbers[water_mask].shape[0]:
        wavenumbers = wavenumbers[water_mask]
        X = X[:, water_mask]
    elif len(vip_imp) != len(wavenumbers):
        raise ValueError(f"VIP length {len(vip_imp)} does not match wavenumbers {len(wavenumbers)}")

    all_regions, sig_regions = compare_groups(
        X, wavenumbers, vip_imp, group_arr,
        high_label=high_label, low_label=low_label,
        person_codes=person_arr,
        top_pct=top_pct, min_consecutive=min_consecutive,
    )

    import pandas as pd
    if person_arr is not None:
        tmp = pd.DataFrame({"person": person_arr, "group": group_arr})
        n_high_persons = int(tmp[tmp["group"] == high_label]["person"].nunique())
        n_low_persons  = int(tmp[tmp["group"] == low_label]["person"].nunique())
    else:
        n_high_persons = int((group_arr == high_label).sum())
        n_low_persons  = int((group_arr == low_label).sum())

    return {
        "sample_type":    sample_type,
        "n_high":         n_high_persons,
        "n_low":          n_low_persons,
        "n_high_spectra": int((group_arr == high_label).sum()),
        "n_low_spectra":  int((group_arr == low_label).sum()),
        "all_regions":    all_regions,
        "sig_regions":    sig_regions,
        "group_column":   group_column,
        "high_label":     high_label,
        "low_label":      low_label,
        "wavenumbers":    wavenumbers.tolist(),
    }


def run_all_matrices(
    sample_types: list[str] | None = None,
    out_dir: Path | None = None,
    **kwargs,
) -> dict[str, dict]:
    from ftir_pred.data.config import SAMPLE_TYPES

    matrices = sample_types or SAMPLE_TYPES
    out_dir  = out_dir or (RESULTS_DIR / "spectral_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = {}
    for st in matrices:
        try:
            res = run_spectral_comparison(st, **kwargs)
        except ValueError as e:
            print(f"  Skipping {st}: {e}")
            continue

        n_sig = len(res["sig_regions"])
        print(f"  {st}: n_high={res['n_high']}, n_low={res['n_low']}, significant regions={n_sig}")
        combined[st] = {k: v for k, v in res.items() if k not in ("X_high", "X_low")}

    out_path = out_dir / "spectral_comparison.json"
    out_path.write_text(json.dumps(combined))
    print(f"\nSaved → {out_path}")
    return combined
