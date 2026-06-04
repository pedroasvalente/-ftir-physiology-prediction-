import numpy as np
import pandas as pd

# FTIR spectral region assignments (approximate, cm⁻¹)
SPECTRAL_REGIONS = {
    "Lipids (C-H stretch)": (2800, 3050),
    "Amide I (C=O stretch)": (1600, 1700),
    "Amide II (N-H bend)": (1480, 1600),
    "Carbohydrates / Amide III": (1200, 1480),
    "Phosphates": (950, 1200),
}


def top_wavenumbers(
    wavenumbers: np.ndarray,
    importances: np.ndarray,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the top-N wavenumbers by importance."""
    idx = np.argsort(importances)[::-1][:top_n]
    return pd.DataFrame({
        "wavenumber": wavenumbers[idx],
        "importance": importances[idx],
        "rank": np.arange(1, top_n + 1),
    })


def region_summary(wavenumbers: np.ndarray, importances: np.ndarray) -> pd.DataFrame:
    """Aggregate importance by spectral region."""
    rows = []
    for region, (lo, hi) in SPECTRAL_REGIONS.items():
        mask = (wavenumbers >= lo) & (wavenumbers <= hi)
        total_imp = float(importances[mask].sum()) if mask.any() else 0.0
        rows.append({"region": region, "range": f"{lo}–{hi} cm⁻¹", "importance": total_imp})
    df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance_pct"] = df["importance"] / df["importance"].sum() * 100
    return df
