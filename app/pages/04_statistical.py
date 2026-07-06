import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import RESULTS_DIR, render_appearance_sidebar, render_data_sidebar
from ftir_pred.analysis.statistical_validation import bland_altman

st.set_page_config(page_title="Statistical Validation", layout="wide")
st.title("Statistical Validation")

matrix_colors, _ = render_appearance_sidebar(show_matrices=True)
df = render_data_sidebar()
if df is None or df.empty:
    st.stop()

pred_file = RESULTS_DIR / "study_regression_v2" / "predictions_data.json"
if not pred_file.exists():
    st.info("No predictions_data.json found. Run a training experiment first.")
    st.stop()

pred_data: dict = {}
try:
    pred_data = json.loads(pred_file.read_text())
except Exception:
    pass

if not pred_data:
    st.stop()

# ── Build index for cascading filters ─────────────────────────────────────────
idx_rows = []
for k, v in pred_data.items():
    idx_rows.append({
        "key":         k,
        "target":      v.get("target", ""),
        "sample_type": v.get("sample_type", ""),
        "model":       v.get("model", ""),
    })
idx_df = pd.DataFrame(idx_rows)

all_targets = sorted(idx_df["target"].dropna().unique())
_fc1, _fc2, _fc3 = st.columns(3)
sel_target = _fc1.selectbox("Target", all_targets)

matrices_for_target = sorted(
    idx_df.loc[idx_df["target"] == sel_target, "sample_type"].dropna().unique()
)
sel_matrix = _fc2.selectbox("Matrix", matrices_for_target)

models_for_combo = sorted(
    idx_df.loc[
        (idx_df["target"] == sel_target) & (idx_df["sample_type"] == sel_matrix),
        "model",
    ].dropna().unique()
)
sel_model = _fc3.selectbox("Model", models_for_combo)

# ── Resolve entry ──────────────────────────────────────────────────────────────
match = idx_df.loc[
    (idx_df["target"] == sel_target)
    & (idx_df["sample_type"] == sel_matrix)
    & (idx_df["model"] == sel_model),
    "key",
]

if match.empty:
    st.warning(f"No predictions for {sel_target} / {sel_matrix} / {sel_model}.")
    st.stop()

entry  = pred_data[match.iloc[0]]
y_test = np.array(entry["y_test"])
y_pred = np.array(entry["y_pred"])
color  = matrix_colors.get(sel_matrix, "#2166ac")

# ── Metrics row ────────────────────────────────────────────────────────────────
meta_row = df[
    (df["target"] == sel_target)
    & (df["sample_type"] == sel_matrix)
    & (df["model"] == sel_model)
]
if not meta_row.empty:
    r = meta_row.sort_values("r2", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{r.get('r2', float('nan')):.3f}",
              f"[{r.get('r2_ci95_low', float('nan')):.3f}, {r.get('r2_ci95_high', float('nan')):.3f}] 95% CI")
    c2.metric("RMSE", f"{r.get('rmse', float('nan')):.3f}")
    c3.metric("Pearson r", f"{r.get('pearson_r', float('nan')):.3f}")
    c4.metric("p-value", f"{r.get('pearson_p', float('nan')):.4f}")
    st.caption(f"n_test = {r.get('n_test', '?')}  ·  {sel_target} — {sel_matrix} — {sel_model}")

st.divider()

col1, col2 = st.columns(2)

# ── Observed vs predicted ──────────────────────────────────────────────────────
with col1:
    st.subheader("Observed vs Predicted")
    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))
    pad = (hi - lo) * 0.05

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
        mode="lines", line=dict(color="gray", dash="dash", width=1.5),
        name="Identity", hoverinfo="skip",
    ))
    fig1.add_trace(go.Scatter(
        x=y_test.tolist(), y=y_pred.tolist(),
        mode="markers",
        marker=dict(color=color, size=7, opacity=0.75,
                    line=dict(color="white", width=0.5)),
        name="Predictions",
        hovertemplate="Observed: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
    ))
    fig1.update_layout(
        xaxis=dict(title=f"Observed {sel_target}", showgrid=True, gridcolor="#e5e5e5"),
        yaxis=dict(title="Predicted", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=400, margin=dict(l=60, r=20, t=30, b=60),
    )
    st.plotly_chart(fig1, use_container_width=True)

# ── Bland-Altman ───────────────────────────────────────────────────────────────
with col2:
    st.subheader("Bland-Altman")
    ba = bland_altman(y_test, y_pred)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ba["mean"], y=ba["diff"],
        mode="markers",
        marker=dict(color=color, size=7, opacity=0.75,
                    line=dict(color="white", width=0.5)),
        hovertemplate="Mean: %{x:.2f}<br>Diff: %{y:.2f}<extra></extra>",
    ))
    for y_val, clr, dash, label in [
        (ba["loa_upper"], "red",  "dash", f"+1.96 SD = {ba['loa_upper']:.3f}"),
        (ba["bias"],      "blue", "solid", f"Bias = {ba['bias']:.3f}"),
        (ba["loa_lower"], "red",  "dash", f"−1.96 SD = {ba['loa_lower']:.3f}"),
    ]:
        fig2.add_hline(
            y=y_val, line_color=clr, line_dash=dash, opacity=0.7,
            annotation_text=label, annotation_position="right",
            annotation_font_size=11,
        )
    fig2.update_layout(
        xaxis=dict(title="Mean of observed and predicted", showgrid=True, gridcolor="#e5e5e5"),
        yaxis=dict(title="Difference (observed − predicted)", showgrid=True, gridcolor="#e5e5e5"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=400, margin=dict(l=70, r=20, t=30, b=60),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Multi-target summary — same matrix + model ─────────────────────────────────
st.divider()
st.subheader(f"All targets — {sel_matrix} · {sel_model}")

metric_cols = [c for c in ["r2", "r2_ci95_low", "r2_ci95_high", "rmse", "pearson_r", "pearson_p", "n_test"]
               if c in df.columns]
multi = (
    df[(df["sample_type"] == sel_matrix) & (df["model"] == sel_model)]
    .copy()
    .sort_values("r2", ascending=False)
)
if not multi.empty:
    show_cols = [c for c in ["target", "target_group"] + metric_cols if c in multi.columns]
    fmt = {c: "{:.3f}" for c in ["r2", "rmse", "pearson_r", "r2_ci95_low", "r2_ci95_high"]
           if c in multi.columns}
    fmt["pearson_p"] = "{:.4f}"
    st.dataframe(
        multi[show_cols].reset_index(drop=True).style
        .format(fmt, na_rep="—")
        .background_gradient(subset=["r2"] if "r2" in show_cols else [],
                             cmap="RdYlGn", vmin=0, vmax=0.5),
        use_container_width=True, hide_index=True,
    )
    st.caption(f"{len(multi)} targets for {sel_matrix} / {sel_model}")
