import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import RESULTS_DIR, render_data_sidebar
from ftir_pred.analysis.spectral_interpretation import region_summary, top_wavenumbers
from ftir_pred.analysis.wavenumber_reference import WAVENUMBER_BANDS

st.set_page_config(page_title="Spectral Interpretation", layout="wide")
st.title("Spectral Interpretation — VIP Scores")

render_data_sidebar()

imp_files = sorted(RESULTS_DIR.rglob("wavenumber_importance.json"))
if not imp_files:
    st.info("No wavenumber_importance.json found. Run a training experiment first.")
    st.stop()

imp_data: dict = {}
for f in imp_files:
    try:
        imp_data.update(json.loads(f.read_text()))
    except Exception:
        pass

if not imp_data:
    st.info("No importance data available.")
    st.stop()

keys = sorted(imp_data.keys())
selected = st.selectbox("Select model run", keys)

entry = imp_data[selected]
wavenumbers = np.array(entry["wavenumbers"])
importances = np.array(entry["importances"])
target = entry.get("target", "")
sample_type = entry.get("sample_type", "")
r2 = entry.get("r2", None)

st.subheader(f"VIP scores — {target} ({sample_type})")
if r2 is not None:
    st.caption(f"Model R² = {r2:.3f}  |  VIP > 1 → relevant variable (Wold et al. 2001)")

# VIP spectrum plot
fig = go.Figure()

# Annotation bands from reference
REGION_COLORS = {
    "Lipids":         "rgba(255,165,0,0.12)",
    "Proteins":       "rgba(65,105,225,0.12)",
    "Carbohydrates":  "rgba(60,179,113,0.12)",
    "Nucleic acids / Phospholipids": "rgba(220,20,60,0.12)",
    "Nucleic acids":  "rgba(220,20,60,0.10)",
    "Phospholipids":  "rgba(186,85,211,0.10)",
    "Cholesterol":    "rgba(128,128,0,0.12)",
    "Metabolites":    "rgba(0,128,128,0.12)",
    "Lipids/Proteins": "rgba(255,140,0,0.10)",
    "Lipid bending / metabolites": "rgba(255,140,0,0.10)",
}
DEFAULT_BAND_COLOR = "rgba(180,180,180,0.10)"

shown_classes: set = set()
for band in WAVENUMBER_BANDS:
    mol_class = band["molecule_class"]
    color = REGION_COLORS.get(mol_class, DEFAULT_BAND_COLOR)
    show_legend = mol_class not in shown_classes
    shown_classes.add(mol_class)
    fig.add_vrect(
        x0=band["lo"], x1=band["hi"],
        fillcolor=color, line_width=0,
        annotation_text=band["assignment"].split("(")[0].strip() if show_legend else "",
        annotation_position="top left",
        annotation_font_size=9,
        annotation_font_color="#555",
    )

# VIP threshold line
fig.add_hline(y=1.0, line_dash="dot", line_color="red", opacity=0.6,
              annotation_text="VIP = 1", annotation_position="right",
              annotation_font_size=10)

fig.add_trace(go.Scatter(
    x=wavenumbers.tolist(),
    y=importances.tolist(),
    fill="tozeroy",
    mode="lines",
    line=dict(width=1.5, color="#2166ac"),
    name="VIP score",
    hovertemplate="Wavenumber: %{x:.1f} cm⁻¹<br>VIP: %{y:.3f}<extra></extra>",
))

fig.update_layout(
    xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed",
               showgrid=True, gridcolor="#e5e5e5"),
    yaxis=dict(title="VIP score", showgrid=True, gridcolor="#e5e5e5"),
    plot_bgcolor="white", paper_bgcolor="white",
    height=400, hovermode="x",
    margin=dict(l=60, r=30, t=40, b=60),
)
st.plotly_chart(fig, use_container_width=True)

# Tables
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 20 wavenumbers")
    top = top_wavenumbers(wavenumbers, importances, top_n=20)
    fmt_cols = {"wavenumber": "{:.1f}", "importance": "{:.4f}"}
    if "distance_cm" in top.columns:
        fmt_cols["distance_cm"] = "{:.1f}"
    st.dataframe(
        top.style.format(fmt_cols, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Assignment from Movasaghi et al. (2008) Applied Spectroscopy Reviews "
        "and Socrates (2001) Infrared and Raman Characteristic Group Frequencies."
    )

with col2:
    st.subheader("Importance by spectral region")
    regions = region_summary(wavenumbers, importances)
    st.dataframe(
        regions.style.format({"importance": "{:.4f}", "importance_pct": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )
