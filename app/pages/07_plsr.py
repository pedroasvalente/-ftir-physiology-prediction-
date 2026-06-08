"""
PLS-R direct regression results.

Shows cross-validated R² vs number of components, test-set performance,
and VIP scores with biochemical annotations.

Reference: Wold et al. (2001) Chemometrics and Intelligent Laboratory Systems, 58(2), 109–130.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import RESULTS_DIR, render_appearance_sidebar
from ftir_pred.analysis.wavenumber_reference import top_annotated

st.set_page_config(page_title="PLS-R Direct Regression", layout="wide")
st.title("PLS-R — Direct Chemometric Regression")

matrix_colors, _ = render_appearance_sidebar(show_matrices=True)

PLSR_DIR = RESULTS_DIR / "plsr_results"
json_path = PLSR_DIR / "plsr_results.json"
csv_path  = PLSR_DIR / "plsr_summary.csv"

if not json_path.exists():
    st.info(
        "No PLS-R results found.\n\n"
        "Run:\n```\npython scripts/run_plsr.py\n```"
    )
    st.stop()

with open(json_path) as f:
    plsr_data = json.load(f)

summary_df = pd.read_csv(csv_path) if csv_path.exists() else None

# ── Summary table ─────────────────────────────────────────────────────────
if summary_df is not None:
    st.subheader("Summary — best PLS-R R² per target × matrix")

    r2_clip = st.slider(
        "Clip R² below", min_value=-5.0, max_value=0.0, value=-1.0, step=0.5,
        help="Hide catastrophic failures from the summary table",
    )
    disp_df = summary_df[summary_df["r2"] >= r2_clip].copy()
    n_clipped = len(summary_df) - len(disp_df)
    if n_clipped:
        st.caption(f"{n_clipped} rows hidden (R² < {r2_clip})")

    disp_df = disp_df.sort_values("r2", ascending=False).reset_index(drop=True)
    st.dataframe(
        disp_df.style.format({"r2": "{:.3f}", "rmse": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ── Per-combo explorer ─────────────────────────────────────────────────────
st.subheader("Per-target detail")

targets   = sorted(plsr_data.keys())
col1, col2 = st.columns(2)
with col1:
    target = st.selectbox("Target", targets)
with col2:
    matrices_for_target = sorted(plsr_data.get(target, {}).keys())
    if not matrices_for_target:
        st.info("No PLS-R results for this target.")
        st.stop()
    matrix = st.selectbox("Matrix", matrices_for_target)

entry = plsr_data[target][matrix]
r2    = entry["r2"]
rmse  = entry["rmse"]
n_comp = entry["n_components"]
n_train = entry.get("n_train", "?")
n_test  = entry.get("n_test",  "?")

col1, col2, col3, col4 = st.columns(4)
col1.metric("R² (test set)", f"{r2:.3f}")
col2.metric("RMSE (test set)", f"{rmse:.3f}")
col3.metric("Optimal components", str(n_comp))
col4.metric("n_train / n_test", f"{n_train} / {n_test}")

# ── CV curve ──────────────────────────────────────────────────────────────
cv_curve = entry.get("cv_curve")
if cv_curve:
    cv_df = pd.DataFrame(cv_curve)
    fig_cv = go.Figure()
    fig_cv.add_trace(go.Scatter(
        x=cv_df["n_components"], y=cv_df["r2_cv"],
        mode="lines+markers",
        error_y=dict(type="data", array=cv_df["r2_std"].tolist(), visible=True),
        line=dict(color=matrix_colors.get(matrix, "#2166ac"), width=2),
        name="CV R²",
    ))
    fig_cv.add_vline(
        x=n_comp, line_dash="dash", line_color="red", opacity=0.7,
        annotation_text=f"Optimal: {n_comp}",
        annotation_position="top right",
    )
    fig_cv.update_layout(
        xaxis_title="Number of PLS components",
        yaxis_title="Cross-validated R²",
        plot_bgcolor="white", paper_bgcolor="white",
        height=280,
        margin=dict(l=60, r=30, t=30, b=50),
        xaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
        yaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
    )
    st.plotly_chart(fig_cv, use_container_width=True)

# ── Observed vs predicted ──────────────────────────────────────────────────
if "y_test" in entry and "y_pred" in entry:
    y_test = np.array(entry["y_test"])
    y_pred = np.array(entry["y_pred"])
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode="markers",
        marker=dict(color=matrix_colors.get(matrix, "#2166ac"), size=7, opacity=0.7),
        name="Predictions",
        hovertemplate="Observed: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
    ))
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    fig_pv.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines", line=dict(color="red", dash="dash", width=1.5),
        name="Identity",
    ))
    fig_pv.update_layout(
        xaxis_title=f"Observed {target}",
        yaxis_title="Predicted",
        plot_bgcolor="white", paper_bgcolor="white",
        height=320,
        margin=dict(l=60, r=30, t=30, b=50),
        xaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
        yaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
    )
    st.plotly_chart(fig_pv, use_container_width=True)

# ── VIP scores ────────────────────────────────────────────────────────────
if "vip_scores" in entry and "wavenumbers" in entry:
    wn  = np.array(entry["wavenumbers"])
    vip = np.array(entry["vip_scores"])

    st.subheader("VIP scores")
    fig_vip = go.Figure()
    fig_vip.add_hline(
        y=1.0, line_dash="dot", line_color="red", opacity=0.6,
        annotation_text="VIP = 1", annotation_position="right",
    )
    fig_vip.add_trace(go.Scatter(
        x=wn.tolist(), y=vip.tolist(),
        fill="tozeroy", mode="lines",
        line=dict(width=1.5, color=matrix_colors.get(matrix, "#2166ac")),
        hovertemplate="wn: %{x:.1f} cm⁻¹<br>VIP: %{y:.3f}<extra></extra>",
    ))
    fig_vip.update_layout(
        xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed",
                   showgrid=True, gridcolor="#e5e5e5"),
        yaxis=dict(title="VIP score", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=300, margin=dict(l=60, r=30, t=30, b=50),
    )
    st.plotly_chart(fig_vip, use_container_width=True)

    st.subheader("Top VIP wavenumbers with biochemical assignment")
    top_df = top_annotated(wn, vip, top_n=20)
    fmt = {"wavenumber": "{:.1f}", "importance": "{:.4f}"}
    if "distance_cm" in top_df.columns:
        fmt["distance_cm"] = "{:.1f}"
    st.dataframe(
        top_df.style.format(fmt, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Movasaghi et al. (2008) Applied Spectroscopy Reviews, 43(2), 134–179; "
        "Socrates (2001) Infrared and Raman Characteristic Group Frequencies (Wiley)."
    )
