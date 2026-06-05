import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import render_appearance_sidebar

st.set_page_config(page_title="Spectra", layout="wide")
st.title("Mean FTIR Spectra")

matrix_colors, _ = render_appearance_sidebar(show_matrices=True)

from ftir_pred.config import TRAINING_DATA_PATH
from ftir_pred.data.loader import get_ftir_columns, load_csv

@st.cache_data
def _load():
    df = load_csv(str(TRAINING_DATA_PATH))
    ftir_cols = get_ftir_columns(df)
    wavenumbers = np.array([float(c) for c in ftir_cols])
    return df, ftir_cols, wavenumbers

df, ftir_cols, wavenumbers = _load()

with st.sidebar:
    st.header("Filters")
    matrices = sorted(df["sample_type"].dropna().unique())
    sel_matrices = st.multiselect("Matrices", matrices, default=matrices[:2])
    groups = sorted(df["group_fam"].dropna().unique()) if "group_fam" in df.columns else []
    sel_groups = st.multiselect("Sport groups", groups, default=groups)
    show_sd = st.checkbox("Show ±1 SD band", value=True)

if not sel_matrices:
    st.info("Select at least one matrix.")
    st.stop()

def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


fig = go.Figure()

for matrix in sel_matrices:
    sub = df[df["sample_type"] == matrix]
    if sel_groups and "group_fam" in sub.columns:
        sub = sub[sub["group_fam"].isin(sel_groups)]
    if sub.empty:
        continue
    X = sub[ftir_cols].values.astype(float)
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    color = matrix_colors.get(matrix, "#888888")
    fill_color = _hex_to_rgba(color) if color.startswith("#") else color.replace(")", ",0.15)").replace("rgb(", "rgba(")
    fig.add_trace(go.Scatter(
        x=wavenumbers.tolist(), y=mean.tolist(), mode="lines",
        name=matrix, line=dict(color=color, width=1.5),
    ))
    if show_sd:
        fig.add_trace(go.Scatter(
            x=wavenumbers.tolist() + wavenumbers.tolist()[::-1],
            y=(mean + sd).tolist() + (mean - sd).tolist()[::-1],
            fill="toself", fillcolor=fill_color,
            line=dict(width=0), showlegend=False,
        ))

fig.update_layout(
    xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Absorbance (a.u.)",
    xaxis=dict(autorange="reversed"),
    height=450, legend_title="Matrix",
)
st.plotly_chart(fig, use_container_width=True)
