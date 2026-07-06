import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import render_appearance_sidebar, render_data_sidebar
from ftir_pred.data.config import SAMPLE_TYPES

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")

matrix_colors, model_colors = render_appearance_sidebar(show_matrices=True, show_models=True)
df = render_data_sidebar()
if df is None or df.empty:
    st.stop()

METRIC_COLS = [c for c in ["r2", "rmse", "mae", "pearson_r"] if c in df.columns and df[c].notna().any()]

_fc1, _fc2, _fc3 = st.columns([1, 1, 2])
matrix = _fc1.selectbox("Matrix", SAMPLE_TYPES)
if "timepoints" in df.columns:
    tp_opts = sorted(df["timepoints"].fillna("all").unique())
    timepoints = _fc2.selectbox("Timepoints", tp_opts)
else:
    timepoints = None

with st.sidebar:
    r2_clip = st.slider("Clip R² below", min_value=-5.0, max_value=0.0, value=-1.0, step=0.5)

sub = df[df["sample_type"] == matrix].copy()
if timepoints is not None and "timepoints" in sub.columns:
    sub = sub[sub["timepoints"].fillna("all") == timepoints]
sub = sub[sub["r2"] >= r2_clip] if "r2" in sub.columns else sub

if sub.empty:
    st.warning("No results for this matrix / timepoint combination.")
    st.stop()

ml_only = sub[~sub["is_baseline"].astype(bool)] if "is_baseline" in sub.columns else sub

best_per_model = (
    ml_only.sort_values("r2", ascending=False)
    .groupby("model").first().reset_index()
)

# ── Win rate — which model ranks 1st most often? ──────────────────────────────
st.subheader(f"Model win rate — {matrix}")
st.caption("% of (target × timepoint) combinations where each model achieves the highest R²")

if not ml_only.empty and "r2" in ml_only.columns:
    _win_df = ml_only.copy()
    if "timepoints" in _win_df.columns:
        _win_df["timepoints"] = _win_df["timepoints"].fillna("all")
    best_per_combo = (
        _win_df.sort_values("r2", ascending=False)
        .groupby(["target"] + (["timepoints"] if "timepoints" in _win_df.columns else []))
        .first()
        .reset_index()
    )
    total_combos = len(best_per_combo)
    wins = (
        best_per_combo.groupby("model").size()
        .reset_index(name="wins")
    )
    wins["win_pct"] = wins["wins"] / total_combos * 100
    wins = wins.sort_values("win_pct", ascending=False)

    fig_wins = px.bar(
        wins, x="model", y="win_pct",
        color="model", color_discrete_map=model_colors,
        text_auto=".0f",
        labels={"win_pct": "Best model (%)", "model": "Model"},
    )
    fig_wins.update_traces(texttemplate="%{y:.0f}%", textposition="outside")
    fig_wins.update_layout(
        height=340, showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="% target combos won", range=[0, 100],
                   showgrid=True, gridcolor="#e5e5e5"),
        margin=dict(t=20, b=50, l=60, r=20),
    )
    st.plotly_chart(fig_wins, use_container_width=True)

# ── R² vs RMSE scatter ────────────────────────────────────────────────────────
st.subheader("R² vs RMSE — all runs")
if "rmse" in ml_only.columns:
    fig_sc = px.scatter(
        ml_only, x="rmse", y="r2",
        color="model", symbol="model",
        color_discrete_map=model_colors,
        hover_data=[c for c in ["target", "model", "timepoints"] if c in ml_only.columns],
        labels={"rmse": "RMSE", "r2": "R²"},
    )
    fig_sc.add_hline(y=0.3, line_dash="dot", line_color="gray", opacity=0.6,
                     annotation_text="R²=0.30")
    fig_sc.update_traces(marker_size=7, opacity=0.8)
    fig_sc.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                         xaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
                         yaxis=dict(showgrid=True, gridcolor="#e5e5e5"))
    st.plotly_chart(fig_sc, use_container_width=True)

# ── ML vs baseline ────────────────────────────────────────────────────────────
st.subheader("ML vs baseline — R² distribution")
if "is_baseline" in sub.columns:
    fig_base = px.box(
        sub, x="model", y="r2",
        color="is_baseline",
        color_discrete_map={True: "#aaaaaa", False: "#2166ac"},
        points="all",
        labels={"r2": "R²", "model": "Model", "is_baseline": "Baseline"},
    )
    fig_base.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.4,
                       annotation_text="R²=0", annotation_position="bottom right")
    fig_base.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                           yaxis=dict(showgrid=True, gridcolor="#e5e5e5"))
    st.plotly_chart(fig_base, use_container_width=True)

# ── Cross-matrix summary ──────────────────────────────────────────────────────
st.subheader("Cross-matrix summary — best R² per matrix (all timepoints)")
rows = []
for mat in SAMPLE_TYPES:
    sub_mat = df[df["sample_type"] == mat].copy()
    ml_mat = sub_mat[~sub_mat["is_baseline"].astype(bool)] if "is_baseline" in sub_mat.columns else sub_mat
    if ml_mat.empty:
        continue
    rows.append(ml_mat.loc[ml_mat["r2"].idxmax()])

if rows:
    summary = pd.DataFrame(rows)
    show = [c for c in ["sample_type", "target", "model", "r2", "rmse", "pearson_r", "n_test"]
            if c in summary.columns]
    fmt = {c: "{:.3f}" for c in ["r2", "rmse", "pearson_r"] if c in summary.columns}
    st.dataframe(
        summary[show].reset_index(drop=True).style.format(fmt)
        .background_gradient(subset=["r2"], cmap="RdYlGn", vmin=0, vmax=0.5),
        use_container_width=True, hide_index=True,
    )
