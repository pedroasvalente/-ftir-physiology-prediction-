import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import render_appearance_sidebar

st.set_page_config(page_title="Spectra", layout="wide")
st.title("Spectra Visualisation")

WATER_CO2_MIN = 1850.0
WATER_CO2_MAX = 2500.0

_QUALITATIVE_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]

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
    matrix = st.selectbox("Matrix", matrices)
    has_group = "group_fam" in df.columns
    has_tp = "timepoint" in df.columns
    colour_options = [c for c in ["group_fam", "timepoint"] if c in df.columns]
    colour_by = st.selectbox("Colour by", colour_options) if colour_options else None
    if has_tp:
        all_tps = sorted(df["timepoint"].dropna().unique().tolist())
        timepoints = st.multiselect("Timepoints", all_tps, default=all_tps)
    else:
        timepoints = []
    show_individual = st.checkbox("Overlay individual spectra", value=False)
    show_sd = st.checkbox("Show ±1 SD band", value=True)

data = df[df["sample_type"] == matrix].copy()
if timepoints and has_tp:
    data = data[data["timepoint"].isin(timepoints)]

if data.empty:
    st.warning("No data available for this selection.")
    st.stop()

st.markdown(f"**{len(data)} samples** — {matrix}")

group_col = colour_by if colour_by and colour_by in data.columns else None
groups = sorted(data[group_col].dropna().unique()) if group_col else [matrix]


def _resolve_color(grp_str: str, idx: int = 0) -> str:
    if grp_str in matrix_colors:
        return matrix_colors[grp_str]
    return _QUALITATIVE_PALETTE[idx % len(_QUALITATIVE_PALETTE)]


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


fig = go.Figure()

fig.add_vrect(
    x0=WATER_CO2_MIN, x1=WATER_CO2_MAX,
    fillcolor="grey", opacity=0.15, line_width=0,
    annotation_text="Atmospheric CO₂ / H₂O",
    annotation_position="top left",
    annotation_font_size=11,
    annotation_font_color="grey",
)

mean_records: list[dict] = []

for idx, grp in enumerate(groups):
    grp_str = str(grp)
    color = _resolve_color(grp_str, idx)
    sub = data[data[group_col] == grp] if group_col else data
    spectra = sub[ftir_cols].values.astype(float)
    mean_spectrum = np.nanmean(spectra, axis=0)
    sd = np.nanstd(spectra, axis=0)

    if show_individual:
        for i, row_vals in enumerate(spectra):
            sample_row = sub.iloc[i]
            person = sample_row["person_code"] if "person_code" in sub.columns else ""
            tp = sample_row["timepoint"] if "timepoint" in sub.columns else ""
            fig.add_trace(go.Scatter(
                x=wavenumbers, y=row_vals,
                mode="lines", line=dict(color=color, width=0.7),
                opacity=0.25, showlegend=False,
                hovertemplate=(
                    f"<b>{grp_str}</b><br>"
                    f"Person: {person}<br>Timepoint: {tp}<br>"
                    "Wavenumber: %{x:.1f} cm⁻¹<br>"
                    "Absorbance: %{y:.4f} a.u.<extra></extra>"
                ),
            ))

    if show_sd:
        fill_color = _hex_to_rgba(color) if color.startswith("#") else color
        fig.add_trace(go.Scatter(
            x=wavenumbers.tolist() + wavenumbers.tolist()[::-1],
            y=(mean_spectrum + sd).tolist() + (mean_spectrum - sd).tolist()[::-1],
            fill="toself", fillcolor=fill_color,
            line=dict(width=0), showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=wavenumbers, y=mean_spectrum,
        mode="lines", name=grp_str,
        line=dict(color=color, width=2.5),
        hovertemplate=(
            f"<b>{grp_str} — mean</b><br>"
            "Wavenumber: %{x:.1f} cm⁻¹<br>"
            "Absorbance: %{y:.4f} a.u.<extra></extra>"
        ),
    ))

    mean_records.append({"group": grp_str, **dict(zip(ftir_cols, mean_spectrum))})

legend_title = group_col.replace("_", " ").title() if group_col else "Matrix"
fig.update_layout(
    title=dict(text=f"Mean FTIR Spectra — {matrix}", font_size=16),
    xaxis=dict(
        title="Wavenumber (cm⁻¹)", autorange="reversed",
        showgrid=True, gridcolor="#e5e5e5",
    ),
    yaxis=dict(title="Absorbance (a.u.)", showgrid=True, gridcolor="#e5e5e5"),
    legend=dict(title=legend_title),
    plot_bgcolor="white", paper_bgcolor="white",
    height=480, hovermode="x unified",
    margin=dict(l=60, r=30, t=60, b=60),
)

st.plotly_chart(fig, use_container_width=True)

if mean_records:
    mean_df = pd.DataFrame(mean_records)
    csv_bytes = mean_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download mean spectra (CSV)",
        data=csv_bytes,
        file_name=f"mean_spectra_{matrix}.csv",
        mime="text/csv",
    )
