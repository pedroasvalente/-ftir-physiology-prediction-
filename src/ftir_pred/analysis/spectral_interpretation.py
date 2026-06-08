import numpy as np
import pandas as pd

from ftir_pred.analysis.wavenumber_reference import (
    annotate_wavenumbers,
    get_reference_df,
    top_annotated,
)

# Broad spectral regions for summary statistics
SPECTRAL_REGIONS = {
    "Lipids (C–H stretch)":        (2800, 3050),
    "Ester / Urea C=O":            (1700, 1750),
    "Amide I (proteins)":          (1600, 1700),
    "Amide II (proteins)":         (1480, 1600),
    "Lipid bending / metabolites": (1350, 1480),
    "Amide III / phosphates":      (1200, 1350),
    "Carbohydrates / phosphates":  (950,  1200),
}


def top_wavenumbers(
    wavenumbers: np.ndarray,
    importances: np.ndarray,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the top-N wavenumbers by importance with biochemical annotations."""
    return top_annotated(wavenumbers, importances, top_n=top_n)


def region_summary(wavenumbers: np.ndarray, importances: np.ndarray) -> pd.DataFrame:
    """Aggregate importance by broad spectral region."""
    rows = []
    for region, (lo, hi) in SPECTRAL_REGIONS.items():
        mask = (wavenumbers >= lo) & (wavenumbers <= hi)
        total_imp = float(importances[mask].sum()) if mask.any() else 0.0
        rows.append({"region": region, "range": f"{lo}–{hi} cm⁻¹", "importance": total_imp})
    df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    total = df["importance"].sum()
    df["importance_pct"] = (df["importance"] / total * 100).round(1) if total > 0 else 0.0
    return df
