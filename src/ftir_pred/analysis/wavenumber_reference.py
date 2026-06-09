"""
FTIR-ATR wavenumber band assignments for biological fluids.

Source: Movasaghi, Z., Rehman, S., & Rehman, I.U. (2008).
  Fourier transform infrared (FTIR) spectroscopy of biological tissues.
  Applied Spectroscopy Reviews, 43(2), 134–179.
  https://doi.org/10.1080/05704920701829043
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BAND_ASSIGNMENTS: list[tuple[float, float, str, str]] = [
    (929,  1000, "C–O–C ring deformation",      "Polysaccharides / glycoproteins"),
    (1000, 1080, "C–O stretch / phosphodiester", "Carbohydrates, nucleic acids"),
    (1080, 1200, "C–O / P=O symmetric stretch",  "Carbohydrates, phospholipids"),
    (1200, 1300, "P=O asymmetric stretch",        "Phospholipids"),
    (1300, 1400, "C–N / CH₂ wag (Amide III)",    "Proteins (Amide III)"),
    (1400, 1480, "CH₂/CH₃ bending",              "Lipids, fatty acids"),
    (1480, 1600, "N–H bend + C–N (Amide II)",    "Proteins (Amide II)"),
    (1600, 1700, "C=O stretch (Amide I)",         "Proteins (Amide I)"),
    (1700, 1800, "C=O stretch (esters / acids)",  "Lipids, fatty acids"),
    (2800, 2870, "CH₂ symmetric stretch",         "Lipids, fatty acids"),
    (2870, 2960, "CH₃ asymmetric stretch",        "Lipids, proteins"),
]


def assign_band(lo: float, hi: float) -> tuple[str, str]:
    best, best_ov = ("—", "—"), 0.0
    for blo, bhi, band, bio in BAND_ASSIGNMENTS:
        ov = min(hi, bhi) - max(lo, blo)
        if ov > best_ov:
            best_ov, best = ov, (band, bio)
    return best


def annotate_wavenumbers(wavenumbers: np.ndarray) -> pd.DataFrame:
    rows = []
    for wn in wavenumbers:
        band, bio = assign_band(wn - 0.5, wn + 0.5)
        rows.append({"wavenumber": wn, "band_assignment": band, "biochemical_origin": bio})
    return pd.DataFrame(rows)


def get_reference_df() -> pd.DataFrame:
    return pd.DataFrame(
        BAND_ASSIGNMENTS,
        columns=["lo", "hi", "band_assignment", "biochemical_origin"],
    )
