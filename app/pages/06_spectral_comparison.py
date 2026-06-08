import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import trapezoid
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import RESULTS_DIR, render_appearance_sidebar
from ftir_pred.config import TRAINING_DATA_PATH
from ftir_pred.data.loader import get_ftir_columns, load_csv

st.set_page_config(page_title="Spectral Comparison", layout="wide")
st.title("Spectral Comparison — High vs Low VO2max")

matrix_colors, _ = render_appearance_sidebar(show_matrices=True)

json_path = RESULTS_DIR / "spectral_comparison" / "spectral_comparison.json"
if not json_path.exists():
    st.info("Run: `python scripts/run_spectral_comparison.py`")
    st.stop()

with open(json_path) as f:
    data = json.load(f)

if not data:
    st.stop()

with st.sidebar:
    st.header("Filters")
    matrix = st.selectbox("Matrix", sorted(data.keys()))

entry = data[matrix]
all_regions = entry.get("all_regions", [])
sig_regions = entry.get("sig_regions", [])
n_high = entry.get("n_high", "?")
n_low  = entry.get("n_low",  "?")
group_col   = entry.get("group_column", "vo2max_classes_simplified")
high_label  = entry.get("high_label", 3.0)
low_label   = entry.get("low_label",  1.0)

st.markdown(
    f"**{matrix}** — n_high = {n_high} | n_low = {n_low} | "
    f"significant regions: **{len(sig_regions)}** / {len(all_regions)}"
)
st.caption(
    "AUC = trapezoidal integral of absorbance over each VIP-guided spectral region. "
    "Mann-Whitney U. Movasaghi et al. (2008) Applied Spectroscopy Reviews 43(2), 134–179."
)

if not all_regions:
    st.info("No VIP regions found.")
    st.stop()

# Summary table
st.subheader("VIP Spectral Regions")
reg_df = pd.DataFrame([{
    "Region (cm⁻¹)":       r["label"],
    "max VIP":              round(r["max_vip"], 3),
    "Band assignment":      r["band_assignment"],
    "Biochemical origin":   r["biochemical_origin"],
    "Median high":          round(r.get("median_high", float("nan")), 5),
    "Median low":           round(r.get("median_low",  float("nan")), 5),
    "p-value":              round(r.get("p_value", 1.0), 4),
    "sig":                  r.get("sig", ""),
} for r in all_regions])

st.dataframe(
    reg_df.style.format({
        "max VIP":     "{:.3f}",
        "Median high": "{:.5f}",
        "Median low":  "{:.5f}",
        "p-value":     "{:.4f}",
    }, na_rep="—").background_gradient(subset=["max VIP"], cmap="Reds"),
    use_container_width=True,
    hide_index=True,
)

if not sig_regions:
    st.info("No significant regions at current VIP threshold.")
    st.stop()

# Load raw spectra
@st.cache_data
def _load_spectra(st_type, grp_col, hi_lbl, lo_lbl):
    df = load_csv(str(TRAINING_DATA_PATH))
    ftir_cols = get_ftir_columns(df)
    wn = np.array([float(c) for c in ftir_cols])
    sub = df[df["sample_type"] == st_type].copy()
    valid = sub[grp_col].notna() & sub[ftir_cols].notna().all(axis=1)
    sub = sub[valid]
    X = sub[ftir_cols].values.astype(float)
    grp = sub[grp_col].values
    return wn, X[grp == hi_lbl], X[grp == lo_lbl]

wn, X_high, X_low = _load_spectra(matrix, group_col, high_label, low_label)
from ftir_pred.data.config import WATER_REGION
water_mask = (wn < WATER_REGION[0]) | (wn > WATER_REGION[1])
wn       = wn[water_mask]
X_high   = X_high[:, water_mask]
X_low    = X_low[:,  water_mask]

color_high = matrix_colors.get(matrix, "#d73027")
color_low  = "#4575b4"


def _rgba(hex_color, alpha=0.20):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


st.subheader("Significant Regions — Mean ± SD spectra and AUC distributions")
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

for row_i in range((len(sig_regions) + 1) // 2):
    cols = st.columns(2)
    for col_j in range(2):
        idx = row_i * 2 + col_j
        if idx >= len(sig_regions):
            break

        reg    = sig_regions[idx]
        lo, hi = reg["lo"], reg["hi"]
        letter = LETTERS[idx]

        zone_mask = (wn >= lo) & (wn <= hi)
        wn_z  = np.sort(wn[zone_mask])
        order = np.argsort(wn[zone_mask])
        sp_high = X_high[:, zone_mask][:, order]
        sp_low  = X_low[:,  zone_mask][:, order]

        aucs_high = np.array([np.abs(trapezoid(sp_high[k], wn_z)) for k in range(len(sp_high))])
        aucs_low  = np.array([np.abs(trapezoid(sp_low[k],  wn_z)) for k in range(len(sp_low))])

        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.45, 0.55],
            horizontal_spacing=0.08,
            subplot_titles=["Mean ± SD spectrum", "AUC by group"],
        )

        for sp_arr, color, name in [
            (sp_high, color_high, "High VO2max"),
            (sp_low,  color_low,  "Low VO2max"),
        ]:
            mn, sd = sp_arr.mean(axis=0), sp_arr.std(axis=0)
            fig.add_trace(go.Scatter(
                x=np.concatenate([wn_z, wn_z[::-1]]),
                y=np.concatenate([mn + sd, (mn - sd)[::-1]]),
                fill="toself", fillcolor=_rgba(color), mode="none",
                showlegend=False, hoverinfo="skip",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=wn_z, y=mn, mode="lines",
                name=name, line=dict(color=color, width=2.5),
                legendgroup=name,
                hovertemplate=f"{name}: %{{y:.5f}}<extra></extra>",
            ), row=1, col=1)

        for aucs, color, name in [
            (aucs_high, color_high, "High"),
            (aucs_low,  color_low,  "Low"),
        ]:
            fig.add_trace(go.Violin(
                x=[name] * len(aucs), y=aucs,
                name=name, legendgroup=name, showlegend=False,
                line_color=color, fillcolor=_rgba(color, 0.50),
                box_visible=True, meanline_visible=True,
                points="all", jitter=0.25, pointpos=0,
                marker=dict(color=color, size=4, opacity=0.7),
            ), row=1, col=2)

        p      = reg.get("p_value", 1.0)
        sig_lbl = reg.get("sig", "")
        all_aucs = np.concatenate([aucs_high, aucs_low])
        y_top = float(np.percentile(all_aucs, 99))
        y_bot = float(np.percentile(all_aucs,  1))
        y_span = max(y_top - y_bot, 1e-12)
        y_br = y_top + y_span * 0.25

        for kw in [
            dict(x0="High", x1="Low",  y0=y_br,               y1=y_br),
            dict(x0="High", x1="High", y0=y_br - y_span * 0.05, y1=y_br),
            dict(x0="Low",  x1="Low",  y0=y_br - y_span * 0.05, y1=y_br),
        ]:
            fig.add_shape(type="line", row=1, col=2,
                          line=dict(color="#222", width=1.2), **kw)
        fig.add_annotation(
            x=0.5, y=y_br + y_span * 0.12,
            xref="x2", yref="y2",
            text=f"<b>{sig_lbl}</b>  p={p:.4f}",
            showarrow=False, font=dict(size=11),
        )

        fig.update_xaxes(autorange="reversed", title="Wavenumber (cm⁻¹)", row=1, col=1)
        fig.update_yaxes(title="Absorbance", row=1, col=1)
        fig.update_yaxes(
            title="AUC",
            range=[y_bot - y_span * 0.05, y_top + y_span * 1.5],
            row=1, col=2,
        )
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>Zone {letter}  ·  {reg['label']}  ·  {reg['band_assignment']}</b><br>"
                    f"<span style='font-size:10px;color:#666'>{reg['biochemical_origin']}</span>"
                ),
                font=dict(size=13), x=0.5, xanchor="center",
            ),
            height=330, margin=dict(t=75, b=45, l=60, r=20),
            paper_bgcolor="white", plot_bgcolor="white",
            violingap=0.25, violingroupgap=0.1,
            legend=dict(
                x=0.01, y=0.99, xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.75)", bordercolor="#ddd",
                borderwidth=1, font=dict(size=10),
            ),
        )
        cols[col_j].plotly_chart(fig, use_container_width=True)

        dl_df = pd.DataFrame({
            "group": (["High"] * len(aucs_high)) + (["Low"] * len(aucs_low)),
            "AUC":   np.concatenate([aucs_high, aucs_low]),
        })
        cols[col_j].download_button(
            label=f"Zone {letter} AUC data (CSV)",
            data=dl_df.to_csv(index=False).encode(),
            file_name=f"zone_{letter}_{matrix}_{lo:.0f}_{hi:.0f}.csv",
            mime="text/csv",
            key=f"dl_{idx}_{matrix}",
        )
