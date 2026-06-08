"""
Peer-reviewed wavenumber assignment reference for biological FTIR-ATR spectra.

Primary sources:
  Movasaghi, Z., Rehman, S., & Rehman, I.U. (2008). Fourier transform infrared (FTIR)
    spectroscopy of biological tissues. Applied Spectroscopy Reviews, 43(2), 134–179.
    https://doi.org/10.1080/05704920701829043

  Socrates, G. (2001). Infrared and Raman Characteristic Group Frequencies: Tables and
    Charts (3rd ed.). Wiley. ISBN 0-471-85298-8.

  Mantsch, H.H., & Chapman, D. (Eds.) (1996). Infrared Spectroscopy of Biomolecules.
    Wiley-Liss. ISBN 0-471-01518-X.

Each entry:
  "lo"       : lower wavenumber (cm⁻¹)
  "hi"       : upper wavenumber (cm⁻¹)
  "peak"     : nominal peak wavenumber (cm⁻¹)
  "assignment"   : concise biochemical assignment
  "molecule_class" : broad molecular category
  "reference"    : primary citation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Wavenumber assignment table
# Each dict defines one spectral band.
# ---------------------------------------------------------------------------

WAVENUMBER_BANDS: list[dict] = [
    # ── Lipid / fatty acid C–H stretches ──────────────────────────────────
    {
        "lo": 2950, "hi": 2965, "peak": 2956,
        "assignment": "CH₃ asymmetric stretch (lipids, fatty acids)",
        "molecule_class": "Lipids",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 2910, "hi": 2935, "peak": 2922,
        "assignment": "CH₂ asymmetric stretch (acyl chains, lipids)",
        "molecule_class": "Lipids",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 2845, "hi": 2865, "peak": 2853,
        "assignment": "CH₂ symmetric stretch (acyl chains, lipids)",
        "molecule_class": "Lipids",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 2870, "hi": 2882, "peak": 2874,
        "assignment": "CH₃ symmetric stretch (lipids, protein side chains)",
        "molecule_class": "Lipids/Proteins",
        "reference": "Socrates 2001",
    },
    # ── Carbonyl / ester ──────────────────────────────────────────────────
    {
        "lo": 1735, "hi": 1750, "peak": 1740,
        "assignment": "C=O ester stretch (triglycerides, phospholipids)",
        "molecule_class": "Lipids",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1700, "hi": 1720, "peak": 1710,
        "assignment": "C=O stretch (urea, nucleic acids, fatty acids)",
        "molecule_class": "Nucleic acids / Urea",
        "reference": "Movasaghi et al. 2008",
    },
    # ── Protein amide bands ───────────────────────────────────────────────
    {
        "lo": 1648, "hi": 1660, "peak": 1653,
        "assignment": "Amide I – α-helix (C=O stretch)",
        "molecule_class": "Proteins",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1625, "hi": 1648, "peak": 1637,
        "assignment": "Amide I – β-sheet / random coil (C=O stretch)",
        "molecule_class": "Proteins",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1610, "hi": 1625, "peak": 1617,
        "assignment": "Amide I – aggregated strands / β-turns",
        "molecule_class": "Proteins",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1540, "hi": 1570, "peak": 1548,
        "assignment": "Amide II (N–H bend + C–N stretch)",
        "molecule_class": "Proteins",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1220, "hi": 1260, "peak": 1238,
        "assignment": "Amide III (C–N stretch, N–H bend) / PO₂⁻ asymm",
        "molecule_class": "Proteins / Nucleic acids",
        "reference": "Movasaghi et al. 2008",
    },
    # ── Lipid bending ─────────────────────────────────────────────────────
    {
        "lo": 1450, "hi": 1475, "peak": 1461,
        "assignment": "CH₂ scissoring (lipids); CH₃ asymm bend",
        "molecule_class": "Lipids",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1365, "hi": 1385, "peak": 1377,
        "assignment": "CH₃ symmetric bend / δ(CH₃) lipids",
        "molecule_class": "Lipids/Proteins",
        "reference": "Socrates 2001",
    },
    # ── Creatinine / metabolites ──────────────────────────────────────────
    {
        "lo": 1410, "hi": 1430, "peak": 1420,
        "assignment": "CH₂ bend / creatinine (urine marker)",
        "molecule_class": "Metabolites",
        "reference": "Movasaghi et al. 2008",
    },
    # ── Nucleic acids ─────────────────────────────────────────────────────
    {
        "lo": 1215, "hi": 1245, "peak": 1224,
        "assignment": "Asymm PO₂⁻ stretch (DNA/RNA backbone, phospholipids)",
        "molecule_class": "Nucleic acids / Phospholipids",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1075, "hi": 1095, "peak": 1085,
        "assignment": "Symm PO₂⁻ stretch (phospholipids, DNA backbone)",
        "molecule_class": "Nucleic acids / Phospholipids",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 960, "hi": 975, "peak": 968,
        "assignment": "Phosphodiester / deoxyribose C–O stretch (DNA)",
        "molecule_class": "Nucleic acids",
        "reference": "Movasaghi et al. 2008",
    },
    # ── Carbohydrates / glucose ───────────────────────────────────────────
    {
        "lo": 1155, "hi": 1175, "peak": 1163,
        "assignment": "Asymm C–O–C stretch (glycogen, polysaccharides)",
        "molecule_class": "Carbohydrates",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1035, "hi": 1050, "peak": 1043,
        "assignment": "C–O stretch of glucose / glycogen (C–OH)",
        "molecule_class": "Carbohydrates",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1020, "hi": 1035, "peak": 1025,
        "assignment": "Glycogen / glucose (C–O–C ring stretch)",
        "molecule_class": "Carbohydrates",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1115, "hi": 1145, "peak": 1130,
        "assignment": "C–O–C ring stretch (carbohydrates, glycogen)",
        "molecule_class": "Carbohydrates",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1140, "hi": 1160, "peak": 1150,
        "assignment": "C–O stretch (polysaccharides, carbohydrates)",
        "molecule_class": "Carbohydrates",
        "reference": "Mantsch & Chapman 1996",
    },
    {
        "lo": 1260, "hi": 1285, "peak": 1270,
        "assignment": "Amide III / C–O–C stretch of glycerophospholipids",
        "molecule_class": "Proteins / Phospholipids",
        "reference": "Movasaghi et al. 2008",
    },
    # ── Amino acids / proteins ────────────────────────────────────────────
    {
        "lo": 995, "hi": 1010, "peak": 1004,
        "assignment": "Phenylalanine ring breathing mode",
        "molecule_class": "Proteins",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1295, "hi": 1320, "peak": 1305,
        "assignment": "Amide III / CH₂ twist (proteins)",
        "molecule_class": "Proteins",
        "reference": "Movasaghi et al. 2008",
    },
    {
        "lo": 1330, "hi": 1360, "peak": 1340,
        "assignment": "CH bending of proline / hydroxyproline; C–N stretch",
        "molecule_class": "Proteins",
        "reference": "Movasaghi et al. 2008",
    },
    # ── Cholesterol ───────────────────────────────────────────────────────
    {
        "lo": 1050, "hi": 1065, "peak": 1058,
        "assignment": "C–O stretch of cholesterol",
        "molecule_class": "Cholesterol",
        "reference": "Movasaghi et al. 2008",
    },
    # ── Phospholipid head group ───────────────────────────────────────────
    {
        "lo": 960, "hi": 980, "peak": 970,
        "assignment": "C–N⁺(CH₃)₃ stretch of choline (phosphatidylcholine)",
        "molecule_class": "Phospholipids",
        "reference": "Movasaghi et al. 2008",
    },
    # ── Thiol / sulfhydryl ────────────────────────────────────────────────
    {
        "lo": 2530, "hi": 2600, "peak": 2550,
        "assignment": "S–H stretch (free thiols: albumin Cys34, glutathione)",
        "molecule_class": "Proteins / Antioxidants",
        "reference": "Barth 2007 / Socrates 2001",
    },
    # ── Urea (high concentration in urine) ────────────────────────────────
    {
        "lo": 1590, "hi": 1610, "peak": 1600,
        "assignment": "C=O / N–H of urea (dominant in urine)",
        "molecule_class": "Metabolites",
        "reference": "Movasaghi et al. 2008",
    },
]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_reference_df() -> pd.DataFrame:
    """Return the full reference table as a DataFrame."""
    return pd.DataFrame(WAVENUMBER_BANDS)


def annotate_wavenumbers(
    wavenumbers: np.ndarray,
    tolerance: float = 8.0,
) -> pd.DataFrame:
    """
    For each input wavenumber return the closest band annotation within ±tolerance cm⁻¹.

    Parameters
    ----------
    wavenumbers : array of cm⁻¹ values to annotate
    tolerance   : max distance from a band's `peak` to accept an assignment (cm⁻¹)

    Returns
    -------
    DataFrame with columns: wavenumber, assignment, molecule_class, reference, distance_cm
    """
    ref = get_reference_df()
    rows = []
    for wn in wavenumbers:
        ref["_d"] = np.abs(ref["peak"] - wn)
        best = ref.loc[ref["_d"].idxmin()]
        if best["_d"] <= tolerance:
            rows.append({
                "wavenumber": wn,
                "assignment": best["assignment"],
                "molecule_class": best["molecule_class"],
                "reference": best["reference"],
                "distance_cm": round(float(best["_d"]), 1),
            })
        else:
            rows.append({
                "wavenumber": wn,
                "assignment": None,
                "molecule_class": None,
                "reference": None,
                "distance_cm": None,
            })
    ref.drop(columns=["_d"], inplace=True)
    return pd.DataFrame(rows)


def annotate_by_range(
    wavenumbers: np.ndarray,
) -> pd.DataFrame:
    """
    For each input wavenumber return all bands whose [lo, hi] range contains it.
    A wavenumber may match multiple bands; the result contains one row per match.

    Parameters
    ----------
    wavenumbers : array of cm⁻¹ values

    Returns
    -------
    DataFrame with columns: wavenumber, assignment, molecule_class, reference
    """
    ref = get_reference_df()
    rows = []
    for wn in wavenumbers:
        matches = ref[(ref["lo"] <= wn) & (ref["hi"] >= wn)]
        if len(matches) == 0:
            rows.append({
                "wavenumber": wn,
                "assignment": None,
                "molecule_class": None,
                "reference": None,
            })
        else:
            for _, row in matches.iterrows():
                rows.append({
                    "wavenumber": wn,
                    "assignment": row["assignment"],
                    "molecule_class": row["molecule_class"],
                    "reference": row["reference"],
                })
    return pd.DataFrame(rows)


def top_annotated(
    wavenumbers: np.ndarray,
    importances: np.ndarray,
    top_n: int = 20,
    tolerance: float = 8.0,
) -> pd.DataFrame:
    """
    Return the top-N wavenumbers by importance, annotated with their biochemical assignment.

    Parameters
    ----------
    wavenumbers  : wavenumber array
    importances  : importance score per wavenumber (e.g. VIP)
    top_n        : how many to return
    tolerance    : ±cm⁻¹ window for annotate_wavenumbers

    Returns
    -------
    DataFrame: rank, wavenumber, importance, assignment, molecule_class, reference
    """
    idx = np.argsort(importances)[::-1][:top_n]
    top_wn = wavenumbers[idx]
    top_imp = importances[idx]
    annotations = annotate_wavenumbers(top_wn, tolerance=tolerance)
    annotations.insert(0, "rank", np.arange(1, top_n + 1))
    annotations.insert(2, "importance", top_imp)
    return annotations.reset_index(drop=True)
