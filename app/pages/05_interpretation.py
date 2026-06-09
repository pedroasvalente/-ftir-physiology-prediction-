import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import RESULTS_DIR, render_appearance_sidebar, render_data_sidebar
from ftir_pred.analysis.spectral_interpretation import region_summary, top_wavenumbers
from ftir_pred.analysis.wavenumber_reference import BAND_ASSIGNMENTS

st.set_page_config(page_title="Spectral Interpretation", layout="wide")
st.title("Spectral Interpretation — VIP Scores")

matrix_colors, _ = render_appearance_sidebar(show_matrices=True)
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
    st.stop()

# ── Build a structured index for cascading filters ────────────────────────────
rows = []
for k, v in imp_data.items():
    rows.append({
        "key":         k,
        "target":      v.get("target", k),
        "sample_type": v.get("sample_type", ""),
        "model":       v.get("model", ""),
        "r2":          v.get("r2"),
    })
idx_df = pd.DataFrame(rows)

with st.sidebar:
    st.header("Filters")
    targets_avail = sorted(idx_df["target"].dropna().unique())
    sel_target = st.selectbox("Target", targets_avail)

    matrices_avail = sorted(
        idx_df.loc[idx_df["target"] == sel_target, "sample_type"].dropna().unique()
    )
    sel_matrix = st.selectbox("Matrix", matrices_avail)

    models_avail = sorted(
        idx_df.loc[
            (idx_df["target"] == sel_target) & (idx_df["sample_type"] == sel_matrix),
            "model",
        ].dropna().unique()
    )
    sel_model = st.selectbox("Model", models_avail)

match = idx_df.loc[
    (idx_df["target"] == sel_target)
    & (idx_df["sample_type"] == sel_matrix)
    & (idx_df["model"] == sel_model),
    "key",
]

if match.empty:
    st.warning("No VIP data for this combination.")
    st.stop()

entry       = imp_data[match.iloc[0]]
wavenumbers = np.array(entry["wavenumbers"])
importances = np.array(entry["importances"])
r2          = entry.get("r2")
color       = matrix_colors.get(sel_matrix, "#2166ac")

c1, c2, c3 = st.columns(3)
c1.metric("Target", sel_target)
c2.metric("Matrix", sel_matrix)
c3.metric("Model R²", f"{r2:.3f}" if r2 is not None else "—")

if r2 is not None:
    st.caption(f"VIP > 1 → spectral variable relevant for prediction (Wold et al. 2001)  ·  R² = {r2:.3f}")

# ── VIP spectrum ──────────────────────────────────────────────────────────────
BAND_COLORS = [
    "rgba(255,165,0,0.12)", "rgba(65,105,225,0.12)", "rgba(60,179,113,0.12)",
    "rgba(220,20,60,0.10)", "rgba(186,85,211,0.10)", "rgba(128,128,0,0.12)",
    "rgba(0,128,128,0.12)", "rgba(255,140,0,0.10)", "rgba(180,180,180,0.10)",
    "rgba(255,165,0,0.10)", "rgba(65,105,225,0.10)",
]

fig = go.Figure()
for i, (lo, hi, band, _) in enumerate(BAND_ASSIGNMENTS):
    fig.add_vrect(
        x0=lo, x1=hi,
        fillcolor=BAND_COLORS[i % len(BAND_COLORS)],
        line_width=0,
        annotation_text=band.split("(")[0].strip(),
        annotation_position="top left",
        annotation_font_size=8,
        annotation_font_color="#666",
    )
fig.add_hline(y=1.0, line_dash="dot", line_color="red", opacity=0.6,
              annotation_text="VIP = 1", annotation_position="right",
              annotation_font_size=10)
fig.add_trace(go.Scatter(
    x=wavenumbers.tolist(), y=importances.tolist(),
    fill="tozeroy", mode="lines",
    line=dict(width=1.5, color=color), name="VIP score",
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

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 20 wavenumbers")
    top = top_wavenumbers(wavenumbers, importances, top_n=20)
    st.dataframe(
        top.style.format({"wavenumber": "{:.1f}", "importance": "{:.4f}"}, na_rep="—"),
        use_container_width=True, hide_index=True,
    )
    st.caption("Movasaghi et al. (2008) Applied Spectroscopy Reviews 43(2), 134–179.")

with col2:
    st.subheader("Importance by spectral region")
    regions = region_summary(wavenumbers, importances)
    st.dataframe(
        regions.style.format({"importance": "{:.4f}", "importance_pct": "{:.1f}"}),
        use_container_width=True, hide_index=True,
    )

# ── Cross-model comparison within the same target + matrix ───────────────────
same_combo = idx_df[
    (idx_df["target"] == sel_target) & (idx_df["sample_type"] == sel_matrix)
].sort_values("r2", ascending=False)

if len(same_combo) > 1:
    st.divider()
    st.subheader(f"All models — {sel_target} · {sel_matrix}")
    comp_rows = []
    for _, row in same_combo.iterrows():
        e = imp_data[row["key"]]
        wn  = np.array(e["wavenumbers"])
        vip = np.array(e["importances"])
        above1 = int((vip > 1.0).sum())
        top_wn = float(wn[np.argmax(vip)])
        comp_rows.append({
            "Model":         row["model"],
            "R²":            row["r2"],
            "VIP > 1 (n wn)": above1,
            "Peak VIP wn (cm⁻¹)": top_wn,
        })
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(
        comp_df.style.format({"R²": "{:.3f}", "Peak VIP wn (cm⁻¹)": "{:.1f}"})
        .background_gradient(subset=["R²"], cmap="RdYlGn", vmin=0, vmax=0.5),
        use_container_width=True, hide_index=True,
    )
