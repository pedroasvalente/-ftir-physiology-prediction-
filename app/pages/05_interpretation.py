import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import RESULTS_DIR, render_data_sidebar
from ftir_pred.analysis.spectral_interpretation import region_summary, top_wavenumbers

st.set_page_config(page_title="Spectral Interpretation", layout="wide")
st.title("Spectral Interpretation")

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

st.subheader(f"Wavenumber importance — {target} ({sample_type})")
if r2 is not None:
    st.caption(f"Model R² = {r2:.3f}")

# Spectrum plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=wavenumbers.tolist(), y=importances.tolist(),
    fill="tozeroy", mode="lines", line=dict(width=1.5),
    name="Importance",
))

REGIONS = {
    "Lipids (C-H)": (2800, 3050),
    "Amide I": (1600, 1700),
    "Amide II": (1480, 1600),
    "Carbohydrates": (1200, 1480),
    "Phosphates": (950, 1200),
}
colors = ["rgba(255,100,100,0.15)", "rgba(100,255,100,0.15)", "rgba(100,100,255,0.15)",
          "rgba(255,200,100,0.15)", "rgba(200,100,255,0.15)"]
for (name, (lo, hi)), color in zip(REGIONS.items(), colors):
    fig.add_vrect(x0=lo, x1=hi, fillcolor=color, line_width=0,
                  annotation_text=name, annotation_position="top left")

fig.update_layout(
    xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Normalised importance",
    xaxis=dict(autorange="reversed"), height=350,
)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 20 wavenumbers")
    top = top_wavenumbers(wavenumbers, importances, top_n=20)
    st.dataframe(top.style.format({"wavenumber": "{:.1f}", "importance": "{:.4f}"}),
                 use_container_width=True, hide_index=True)

with col2:
    st.subheader("Importance by spectral region")
    regions = region_summary(wavenumbers, importances)
    st.dataframe(regions.style.format({"importance": "{:.4f}", "importance_pct": "{:.1f}%"}),
                 use_container_width=True, hide_index=True)
