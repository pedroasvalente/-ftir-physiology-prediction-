import numpy as np
import pandas as pd

from ftir_pred.analysis.wavenumber_reference import annotate_wavenumbers, assign_band

SPECTRAL_REGIONS = {
    "Lipids (C–H stretch)":        (2800, 3050),
    "Ester / Urea C=O":            (1700, 1800),
    "Amide I (proteins)":          (1600, 1700),
    "Amide II (proteins)":         (1480, 1600),
    "Lipid bending / metabolites": (1350, 1480),
    "Amide III / phosphates":      (1200, 1350),
    "Carbohydrates / phosphates":  (950,  1200),
}


def top_wavenumbers(wavenumbers: np.ndarray, importances: np.ndarray, top_n: int = 20) -> pd.DataFrame:
    idx = np.argsort(importances)[::-1][:top_n]
    ann = annotate_wavenumbers(wavenumbers[idx])
    ann.insert(0, "rank", np.arange(1, top_n + 1))
    ann.insert(2, "importance", importances[idx])
    return ann.reset_index(drop=True)


def region_summary(wavenumbers: np.ndarray, importances: np.ndarray) -> pd.DataFrame:
    rows = []
    for region, (lo, hi) in SPECTRAL_REGIONS.items():
        mask = (wavenumbers >= lo) & (wavenumbers <= hi)
        total = float(importances[mask].sum()) if mask.any() else 0.0
        rows.append({"region": region, "range": f"{lo}–{hi} cm⁻¹", "importance": total})
    df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    total = df["importance"].sum()
    df["importance_pct"] = (df["importance"] / total * 100).round(1) if total > 0 else 0.0
    return df
