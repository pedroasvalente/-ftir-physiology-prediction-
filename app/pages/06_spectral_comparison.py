"""
Spectral comparison: high-fitness vs low-fitness groups.

Shows per-wavenumber Mann-Whitney U test results and AUC.
Bands with FDR q < 0.05 are highlighted. Biochemical annotations from
Movasaghi et al. (2008) and Socrates (2001).
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
from ftir_pred.analysis.wavenumber_reference import annotate_wavenumbers

st.set_page_config(page_title="Spectral Group Comparison", layout="wide")
st.title("Spectral Comparison — High vs Low VO2max")

matrix_colors, _ = render_appearance_sidebar(show_matrices=True)

COMPARISON_DIR = RESULTS_DIR / "spectral_comparison"
json_path = COMPARISON_DIR / "spectral_comparison.json"

if not json_path.exists():
    st.info(
        "No comparison results found.\n\n"
        "Run:\n```\npython scripts/run_spectral_comparison.py\n```"
    )
    st.stop()

with open(json_path) as f:
    data = json.load(f)

available_matrices = sorted(data.keys())
if not available_matrices:
    st.info("No matrices in comparison results.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    matrix = st.selectbox("Matrix", available_matrices)
    alpha = st.slider(
        "FDR threshold (q-value)", min_value=0.001, max_value=0.10,
        value=0.05, step=0.005, format="%.3f",
    )
    show_auc = st.checkbox("Show AUC panel", value=True)
    top_n = st.slider("Annotate top-N peaks", min_value=5, max_value=30, value=10, step=5)

entry = data[matrix]
wavenumbers = np.array(entry["wavenumbers"])
p_values    = np.array(entry["p_value"])
q_values    = np.array(entry["q_value"])
aucs        = np.array(entry["auc"])
u_stats     = np.array(entry["u_stat"])
n_high = entry.get("n_high", "?")
n_low  = entry.get("n_low",  "?")

n_sig = (q_values < alpha).sum()
color = matrix_colors.get(matrix, "#2166ac")

st.markdown(
    f"**Matrix: {matrix}** — n_high = {n_high} &nbsp;|&nbsp; n_low = {n_low} &nbsp;|&nbsp; "
    f"significant wavenumbers (FDR q < {alpha}): **{n_sig}**"
)
st.caption(
    "High VO2max = class 3, Low VO2max = class 1 (vo2max_classes_simplified).  "
    "FDR correction: Benjamini-Hochberg.  "
    "AUC > 0.5 → high group has higher absorbance."
)

# ── Main plot ─────────────────────────────────────────────────────────────
n_rows = 2 if show_auc else 1
fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        ["-log₁₀(q-value)  [Mann-Whitney FDR]", "AUC (wavenumber classifier)"]
        if show_auc else
        ["-log₁₀(q-value)  [Mann-Whitney FDR]"]
    ),
)

neg_log_q = -np.log10(np.clip(q_values, 1e-300, 1.0))
threshold_line = -np.log10(alpha)

# Significant vs not
sig_mask = q_values < alpha
col_sig   = color
col_nosig = "#cccccc"

fig.add_trace(
    go.Scatter(
        x=wavenumbers[~sig_mask], y=neg_log_q[~sig_mask],
        mode="markers",
        marker=dict(size=3, color=col_nosig, opacity=0.5),
        name="q ≥ threshold",
        hovertemplate="wn: %{x:.1f} cm⁻¹<br>-log₁₀(q): %{y:.2f}<extra></extra>",
    ), row=1, col=1,
)
fig.add_trace(
    go.Scatter(
        x=wavenumbers[sig_mask], y=neg_log_q[sig_mask],
        mode="markers",
        marker=dict(size=4, color=col_sig, opacity=0.85),
        name="q < threshold",
        hovertemplate="wn: %{x:.1f} cm⁻¹<br>-log₁₀(q): %{y:.2f}<extra></extra>",
    ), row=1, col=1,
)
fig.add_hline(
    y=threshold_line, line_dash="dash", line_color="red", opacity=0.7,
    annotation_text=f"q = {alpha}", annotation_position="right",
    row=1, col=1,
)

if show_auc:
    fig.add_trace(
        go.Scatter(
            x=wavenumbers, y=aucs,
            mode="lines",
            line=dict(width=1.2, color=color),
            name="AUC",
            hovertemplate="wn: %{x:.1f} cm⁻¹<br>AUC: %{y:.3f}<extra></extra>",
        ), row=2, col=1,
    )
    fig.add_hline(
        y=0.5, line_dash="dot", line_color="grey", opacity=0.6,
        annotation_text="AUC = 0.5 (chance)", annotation_position="right",
        row=2, col=1,
    )

fig.update_xaxes(autorange="reversed", title_text="Wavenumber (cm⁻¹)",
                 showgrid=True, gridcolor="#e5e5e5")
fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
fig.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    height=500 if show_auc else 320,
    hovermode="x unified",
    margin=dict(l=60, r=30, t=50, b=60),
)
st.plotly_chart(fig, use_container_width=True)

# ── Top significant peaks with biochemical annotations ────────────────────
st.subheader(f"Top {top_n} significant wavenumbers (sorted by -log₁₀ q-value)")

if n_sig == 0:
    st.info("No significant wavenumbers at this threshold.")
else:
    sig_df = pd.DataFrame({
        "wavenumber": wavenumbers[sig_mask],
        "neg_log_q":  np.round(neg_log_q[sig_mask], 3),
        "q_value":    np.round(q_values[sig_mask], 5),
        "auc":        np.round(aucs[sig_mask], 4),
        "u_stat":     np.round(u_stats[sig_mask], 1),
    }).sort_values("neg_log_q", ascending=False).head(top_n)

    ann = annotate_wavenumbers(sig_df["wavenumber"].values, tolerance=10.0)
    sig_df = sig_df.reset_index(drop=True)
    sig_df["assignment"]    = ann["assignment"].values
    sig_df["molecule_class"] = ann["molecule_class"].values
    sig_df["reference"]     = ann["reference"].values

    st.dataframe(
        sig_df.style.format({
            "wavenumber": "{:.1f}",
            "neg_log_q":  "{:.3f}",
            "q_value":    "{:.2e}",
            "auc":        "{:.4f}",
            "u_stat":     "{:.0f}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Assignments: Movasaghi et al. (2008) Applied Spectroscopy Reviews, 43(2), 134–179; "
        "Socrates (2001) Infrared and Raman Characteristic Group Frequencies (Wiley, 3rd ed.)."
    )

# ── Download ──────────────────────────────────────────────────────────────
full_df = pd.DataFrame({
    "wavenumber": wavenumbers,
    "u_stat":  np.round(u_stats, 2),
    "p_value": np.round(p_values, 6),
    "q_value": np.round(q_values, 6),
    "auc":     np.round(aucs, 4),
    "significant": (q_values < alpha).astype(int),
})
st.download_button(
    label="Download full comparison table (CSV)",
    data=full_df.to_csv(index=False).encode(),
    file_name=f"spectral_comparison_{matrix}.csv",
    mime="text/csv",
)
