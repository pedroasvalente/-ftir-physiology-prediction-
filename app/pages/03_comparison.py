import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import render_appearance_sidebar, render_data_sidebar

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")

df = render_data_sidebar()
matrix_colors, model_colors = render_appearance_sidebar(show_models=True)

if df is None or df.empty:
    st.stop()

tab_ml_vs_baseline, tab_grid_vs_bayes = st.tabs(["ML vs Baseline", "Grid vs Bayes"])

with tab_ml_vs_baseline:
    st.subheader("ML Models vs Baseline — R²")
    if "is_baseline" not in df.columns:
        st.info("No baseline column in results.")
    else:
        fig = px.box(
            df,
            x="model",
            y="r2",
            color="is_baseline",
            color_discrete_map={True: "#aaaaaa", False: "#2166ac"},
            labels={"r2": "R²", "model": "Model", "is_baseline": "Baseline"},
            points="all",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

with tab_grid_vs_bayes:
    st.subheader("Grid Search vs Bayesian Optimisation — R²")
    ml_only = df[~df["is_baseline"].astype(bool)] if "is_baseline" in df.columns else df
    if "search" not in ml_only.columns:
        st.info("No search column in results.")
    else:
        fig = px.box(
            ml_only,
            x="search",
            y="r2",
            color="model",
            color_discrete_map=model_colors,
            labels={"r2": "R²", "search": "Search type", "model": "Model"},
            points="all",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
