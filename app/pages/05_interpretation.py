import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import RESULTS_DIR, render_appearance_sidebar, render_data_sidebar
from ftir_pred.analysis.spectral_interpretation import region_summary, top_wavenumbers
from ftir_pred.analysis.wavenumber_reference import BAND_ASSIGNMENTS

st.set_page_config(page_title="Spectral Interpretation", layout="wide")
st.title("Spectral Interpretation — PLS-R VIP")
st.caption("VIP scores from PLS-R for each target × matrix combination.")

matrix_colors, _ = render_appearance_sidebar(show_matrices=True)
render_data_sidebar()

plsr_path = RESULTS_DIR / "plsr_results" / "plsr_results.json"
plsr_data: dict = {}
if plsr_path.exists():
    try:
        plsr_data = json.loads(plsr_path.read_text())
    except Exception:
        pass

if not plsr_data:
    st.info("No PLS-R data found. Run `python scripts/run_plsr.py`.")
    st.stop()

all_targets = sorted(plsr_data.keys())
_fc1, _fc2 = st.columns(2)
sel_target = _fc1.selectbox("Target", all_targets)

all_matrices = sorted((plsr_data.get(sel_target) or {}).keys())
sel_matrix = _fc2.selectbox("Matrix", all_matrices if all_matrices else ["—"])

plsr_entry = (plsr_data.get(sel_target) or {}).get(sel_matrix)

if plsr_entry is None:
    st.info("No PLS-R data for this combination.")
    st.stop()

if "vip_scores" not in plsr_entry or "wavenumbers" not in plsr_entry:
    st.info("VIP scores not available for this PLS-R result.")
    st.stop()

color = matrix_colors.get(sel_matrix, "#2166ac")

r2_p = plsr_entry.get("r2")
n_c  = plsr_entry.get("n_components", "?")
rmse = plsr_entry.get("rmse")
c1, c2, c3 = st.columns(3)
c1.metric("R²",              f"{r2_p:.3f}" if r2_p is not None else "—")
c2.metric("PLS components",  str(n_c))
c3.metric("RMSE",            f"{rmse:.3f}" if rmse is not None else "—")

st.divider()

wn  = np.array(plsr_entry["wavenumbers"])
vip = np.array(plsr_entry["vip_scores"])

BAND_COLORS = [
    "rgba(255,165,0,0.10)", "rgba(65,105,225,0.10)", "rgba(60,179,113,0.10)",
    "rgba(220,20,60,0.08)", "rgba(186,85,211,0.08)", "rgba(128,128,0,0.10)",
    "rgba(0,128,128,0.10)", "rgba(255,140,0,0.08)", "rgba(180,180,180,0.08)",
    "rgba(65,105,225,0.08)", "rgba(60,179,113,0.08)",
]

fig = go.Figure()
for i, (lo, hi, band, _) in enumerate(BAND_ASSIGNMENTS):
    fig.add_vrect(
        x0=lo, x1=hi,
        fillcolor=BAND_COLORS[i % len(BAND_COLORS)],
        line_width=0,
        annotation_text=band.split("(")[0].strip(),
        annotation_position="top left",
        annotation_font_size=7,
        annotation_font_color="#555",
    )
fig.add_hline(y=1.0, line_dash="dot", line_color="red", opacity=0.6,
              annotation_text="VIP = 1", annotation_position="right",
              annotation_font_size=10, annotation_font_color="#333")
fig.add_trace(go.Scatter(
    x=wn.tolist(), y=vip.tolist(),
    fill="tozeroy", mode="lines",
    line=dict(width=1.5, color=color),
    hovertemplate="%{x:.1f} cm⁻¹ — VIP: %{y:.3f}<extra></extra>",
))
fig.update_layout(
    xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed",
               showgrid=True, gridcolor="#e5e5e5"),
    yaxis=dict(title="VIP score", showgrid=True, gridcolor="#e5e5e5"),
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(color="#222"),
    height=380, hovermode="x",
    margin=dict(l=60, r=30, t=30, b=50),
)
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Top 20 wavenumbers**")
    st.dataframe(
        top_wavenumbers(wn, vip, top_n=20)
        .style.format({"wavenumber": "{:.1f}", "importance": "{:.4f}"}, na_rep="—"),
        use_container_width=True, hide_index=True,
    )
with c2:
    st.markdown("**Importance by region**")
    st.dataframe(
        region_summary(wn, vip)
        .style.format({"importance": "{:.4f}", "importance_pct": "{:.1f}"}),
        use_container_width=True, hide_index=True,
    )

st.caption("Movasaghi et al. (2008) Applied Spectroscopy Reviews, 43(2), 134–179.")
