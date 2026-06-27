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
st.title("Spectral Interpretation — ML vs PLS-R")
st.caption("VIP scores from both methods for the same target × matrix — do they agree on which spectral regions matter?")

matrix_colors, _ = render_appearance_sidebar(show_matrices=True)
render_data_sidebar()

# ── Load ML VIP data ──────────────────────────────────────────────────────────
imp_files = sorted(RESULTS_DIR.rglob("wavenumber_importance.json"))
imp_data: dict = {}
for f in imp_files:
    try:
        imp_data.update(json.loads(f.read_text()))
    except Exception:
        pass

# ── Load PLS-R VIP data ───────────────────────────────────────────────────────
plsr_path = RESULTS_DIR / "plsr_results" / "plsr_results.json"
plsr_data: dict = {}
if plsr_path.exists():
    try:
        plsr_data = json.loads(plsr_path.read_text())
    except Exception:
        pass

if not imp_data and not plsr_data:
    st.info("No VIP data found. Run a training experiment and `python scripts/run_plsr.py`.")
    st.stop()

# ── Build index for ML cascading filters ──────────────────────────────────────
import pandas as pd
ml_rows = []
for k, v in imp_data.items():
    ml_rows.append({
        "key": k,
        "target":      v.get("target", k),
        "sample_type": v.get("sample_type", ""),
        "model":       v.get("model", ""),
        "r2":          v.get("r2"),
    })
ml_idx = pd.DataFrame(ml_rows) if ml_rows else pd.DataFrame(
    columns=["key", "target", "sample_type", "model", "r2"]
)

# ── Shared filters ────────────────────────────────────────────────────────────
all_targets = sorted(
    set(ml_idx["target"].dropna().unique()) | set(plsr_data.keys())
)
_fc1, _fc2, _fc3 = st.columns(3)
sel_target = _fc1.selectbox("Target", all_targets)

all_matrices = sorted(
    set(ml_idx.loc[ml_idx["target"] == sel_target, "sample_type"].dropna().unique())
    | set((plsr_data.get(sel_target) or {}).keys())
)
sel_matrix = _fc2.selectbox("Matrix", all_matrices if all_matrices else ["—"])

models_avail = sorted(
    ml_idx.loc[
        (ml_idx["target"] == sel_target) & (ml_idx["sample_type"] == sel_matrix),
        "model",
    ].dropna().unique()
)
sel_model = _fc3.selectbox("ML model", models_avail if models_avail else ["—"])

st.divider()

# ── Resolve entries ───────────────────────────────────────────────────────────
ml_match = ml_idx.loc[
    (ml_idx["target"] == sel_target)
    & (ml_idx["sample_type"] == sel_matrix)
    & (ml_idx["model"] == sel_model),
    "key",
]
ml_entry = imp_data.get(ml_match.iloc[0]) if not ml_match.empty else None

plsr_entry = (plsr_data.get(sel_target) or {}).get(sel_matrix)

color = matrix_colors.get(sel_matrix, "#2166ac")

BAND_COLORS = [
    "rgba(255,165,0,0.10)", "rgba(65,105,225,0.10)", "rgba(60,179,113,0.10)",
    "rgba(220,20,60,0.08)", "rgba(186,85,211,0.08)", "rgba(128,128,0,0.10)",
    "rgba(0,128,128,0.10)", "rgba(255,140,0,0.08)", "rgba(180,180,180,0.08)",
    "rgba(65,105,225,0.08)", "rgba(60,179,113,0.08)",
]


def _vip_figure(wn, vip, color, height=340):
    fig = go.Figure()
    for i, (lo, hi, band, _) in enumerate(BAND_ASSIGNMENTS):
        fig.add_vrect(
            x0=lo, x1=hi,
            fillcolor=BAND_COLORS[i % len(BAND_COLORS)],
            line_width=0,
            annotation_text=band.split("(")[0].strip(),
            annotation_position="top left",
            annotation_font_size=7,
            annotation_font_color="#888",
        )
    fig.add_hline(y=1.0, line_dash="dot", line_color="red", opacity=0.6,
                  annotation_text="VIP = 1", annotation_position="right",
                  annotation_font_size=10)
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
        height=height, hovermode="x",
        margin=dict(l=60, r=30, t=30, b=50),
    )
    return fig


# ── Side-by-side columns ──────────────────────────────────────────────────────
col_ml, col_plsr = st.columns(2)

with col_ml:
    st.subheader("ML pipeline — VIP scores")
    if ml_entry is None:
        st.info(
            "No ML VIP data for this combination. "
            "VIP is only saved when the best model R² ≥ 0.3 — "
            "below that threshold, feature importances are not interpretable."
        )
    else:
        wn  = np.array(ml_entry["wavenumbers"])
        vip = np.array(ml_entry["importances"])
        r2  = ml_entry.get("r2")
        st.metric("R²", f"{r2:.3f}" if r2 is not None else "—")
        st.plotly_chart(_vip_figure(wn, vip, color), use_container_width=True)

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

with col_plsr:
    st.subheader("PLS-R — VIP scores")
    if plsr_entry is None:
        st.info("No PLS-R VIP data for this combination. Run `python scripts/run_plsr.py`.")
    elif "vip_scores" not in plsr_entry or "wavenumbers" not in plsr_entry:
        st.info("VIP scores not available for this PLS-R result.")
    else:
        wn_p  = np.array(plsr_entry["wavenumbers"])
        vip_p = np.array(plsr_entry["vip_scores"])
        r2_p  = plsr_entry.get("r2")
        n_c   = plsr_entry.get("n_components", "?")
        st.metric("R²", f"{r2_p:.3f}" if r2_p is not None else "—",
                  f"{n_c} components")
        st.plotly_chart(_vip_figure(wn_p, vip_p, color), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 20 wavenumbers**")
            st.dataframe(
                top_wavenumbers(wn_p, vip_p, top_n=20)
                .style.format({"wavenumber": "{:.1f}", "importance": "{:.4f}"}, na_rep="—"),
                use_container_width=True, hide_index=True,
            )
        with c2:
            st.markdown("**Importance by region**")
            st.dataframe(
                region_summary(wn_p, vip_p)
                .style.format({"importance": "{:.4f}", "importance_pct": "{:.1f}"}),
                use_container_width=True, hide_index=True,
            )

st.caption("Movasaghi et al. (2008) Applied Spectroscopy Reviews, 43(2), 134–179.")
