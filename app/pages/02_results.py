import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import render_appearance_sidebar, render_data_sidebar

st.set_page_config(page_title="Results", layout="wide")
st.title("Model Results")

df = render_data_sidebar()
matrix_colors, model_colors = render_appearance_sidebar(show_matrices=True, show_models=True)

if df is None or df.empty:
    st.stop()

non_baseline = df[~df.get("is_baseline", False)]

with st.sidebar:
    st.header("Filters")
    matrices = sorted(df["sample_type"].dropna().unique()) if "sample_type" in df.columns else []
    sel_matrices = st.multiselect("Matrices", matrices, default=matrices)
    groups = sorted(df["target_group"].dropna().unique()) if "target_group" in df.columns else []
    sel_groups = st.multiselect("Target groups", groups, default=groups)
    metric = st.selectbox("Metric", ["r2", "rmse", "mae", "pearson_r"], index=0)
    show_baseline = st.checkbox("Include baselines", value=False)

mask = df["sample_type"].isin(sel_matrices)
if sel_groups and "target_group" in df.columns:
    mask &= df["target_group"].isin(sel_groups)
if not show_baseline and "is_baseline" in df.columns:
    mask &= ~df["is_baseline"].astype(bool)

filtered = df[mask]

if filtered.empty:
    st.info("No data for the current selection.")
    st.stop()

best = (
    filtered.loc[filtered.groupby(["target", "sample_type"])["r2"].idxmax()]
    .reset_index(drop=True)
)

st.subheader(f"Best model per target × matrix — {metric.upper()}")
pivot = best.pivot(index="target", columns="sample_type", values=metric)
fig = px.imshow(
    pivot,
    color_continuous_scale="RdYlGn" if metric == "r2" else "RdYlGn_r",
    text_auto=".3f",
    aspect="auto",
    labels={"color": metric.upper()},
)
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Full results table"):
    st.dataframe(
        filtered.sort_values("r2", ascending=False),
        use_container_width=True,
    )
